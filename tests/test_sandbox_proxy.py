"""Tests for the vendor-filtering proxy (including IP validation), bwrap sandbox, and secret FD delivery."""

from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xreview.sandbox import (
    SandboxSpec,
    VendorProxy,
    build_bwrap_command,
    bwrap_available,
    host_matches_allowlist,
    is_public_ip,
    resolve_and_connect_public,
    run_in_sandbox,
)


def _send_proxy_request(sock_path: Path, raw_req: bytes) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(str(sock_path))
        s.sendall(raw_req)
        return s.recv(4096)


class SandboxProxyTests(unittest.TestCase):
    def test_host_allowlist_and_public_ip_checks(self) -> None:
        allow = ("api.anthropic.com", "chatgpt.com")
        self.assertTrue(host_matches_allowlist("api.anthropic.com", allow))
        self.assertTrue(host_matches_allowlist("ab.chatgpt.com", allow))
        self.assertFalse(host_matches_allowlist("evilchatgpt.com", allow))
        self.assertFalse(host_matches_allowlist("example.com", allow))

        # IPv4 & IPv6 public vs non-public (astra-17)
        self.assertTrue(is_public_ip("1.1.1.1"))
        self.assertTrue(is_public_ip("2606:4700:4700::1111"))
        for bad_ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1"):
            self.assertFalse(is_public_ip(bad_ip), f"Expected non-public: {bad_ip}")

    def test_vendor_proxy_blocks_non_public_dns_resolution(self) -> None:
        """astra-17: even an allowlisted host resolving to loopback/private IPv4 or IPv6 is blocked."""
        for family, addr in (
            (socket.AF_INET, ("127.0.0.1", 443)),
            (socket.AF_INET6, ("::1", 443, 0, 0)),
        ):
            with mock.patch("socket.getaddrinfo", return_value=[(family, socket.SOCK_STREAM, 6, "", addr)]):
                with self.assertRaises(PermissionError):
                    resolve_and_connect_public("api.openai.com", 443)

    def test_vendor_proxy_blocks_non_allowlisted_and_non_connect_and_non_443(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xr-px-") as td:
            sock = Path(td) / "proxy.sock"
            log = Path(td) / "proxy.log"
            with VendorProxy(sock, allowlist=("api.openai.com",), log_path=log) as proxy:
                r1 = _send_proxy_request(sock, b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
                self.assertIn(b"403 Forbidden", r1)

                r2 = _send_proxy_request(sock, b"GET http://api.openai.com/ HTTP/1.1\r\nHost: api.openai.com\r\n\r\n")
                self.assertIn(b"403 Forbidden", r2)

                r3 = _send_proxy_request(sock, b"CONNECT api.openai.com:8080 HTTP/1.1\r\nHost: api.openai.com:8080\r\n\r\n")
                self.assertIn(b"403 Forbidden", r3)

                self.assertEqual(len(proxy.entries), 3)
                self.assertTrue(all(not allowed for _, _, _, allowed in proxy.entries))
            log_text = log.read_text(encoding="utf-8")
            self.assertIn("BLOCK CONNECT example.com:443", log_text)

    def test_bwrap_command_uses_allowlist_ipc_no_ssl_private_and_no_secret_in_argv(self) -> None:
        spec = SandboxSpec(
            run_id="run-1",
            reviewer_name="astra",
            tree_host_dir=Path("/tmp/tree"),
            inputs_host_dir=Path("/tmp/inputs"),
            state_host_dir=Path("/tmp/state"),
            harness_mounts=((Path("/tmp/bin"), "/harness/bin"),),
            command=("/harness/bin/tool", "--flag"),
            cwd_in_sandbox="/review/run-1/astra/tree",
            secret_env={"CLAUDE_CODE_OAUTH_TOKEN": "SUPER_SECRET_TOKEN_VALUE"},
            vendor_hosts=("api.openai.com",),
        )
        cmd = build_bwrap_command(spec, Path("/tmp/proxy.sock"), Path("/tmp/etc"), use_socat=False, secret_fd=9)
        joined = " ".join(cmd)
        self.assertIn("--unshare-net", cmd)
        self.assertIn("--unshare-pid", cmd)
        self.assertIn("--unshare-ipc", cmd)
        self.assertIn("--die-with-parent", cmd)
        self.assertIn("--clearenv", cmd)
        self.assertNotIn("/etc/ssl/private", joined)
        self.assertNotIn("SUPER_SECRET_TOKEN_VALUE", joined)  # astra-18!
        self.assertNotIn("/var/run/docker.sock", cmd)
        self.assertNotIn("/opt", cmd)

    def test_bwrap_live_isolation_and_secret_fd_delivery(self) -> None:
        if not bwrap_available():
            self.skipTest("bwrap not installed")
        with tempfile.TemporaryDirectory(prefix="xr-sb-") as td:
            root = Path(td)
            tree, inputs, state = root / "tree", root / "inputs", root / "state"
            tree.mkdir(); inputs.mkdir(); state.mkdir()
            (tree / "file.txt").write_text("ok\n", encoding="utf-8")

            check_py = (
                "import os; "
                "assert not os.path.exists('/var/run/docker.sock'); "
                "assert not os.path.exists('/etc/ssl/private'); "
                "assert os.environ.get('MY_SECRET') == 's3cr3t-val'; "
                "assert open('/review/rtest/rev/tree/file.txt').read().strip() == 'ok'; "
                "print('ISOLATED_AND_SECRET_OK')"
            )
            spec = SandboxSpec(
                run_id="rtest",
                reviewer_name="rev",
                tree_host_dir=tree,
                inputs_host_dir=inputs,
                state_host_dir=state,
                harness_mounts=(),
                command=("python3", "-c", check_py),
                cwd_in_sandbox="/review/rtest/rev/tree",
                env={"PATH": "/usr/bin:/bin"},
                secret_env={"MY_SECRET": "s3cr3t-val"},
                vendor_hosts=(),
                timeout_s=15,
                use_sandbox=True,
            )
            res = run_in_sandbox(spec, None, root / "raw.log", root / "proxy.log", capture_stdout=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("ISOLATED_AND_SECRET_OK", res.stdout_lines)

    def test_zero_timeout_disables_timer_and_sets_agy_print_timeout_0s(self) -> None:
        from unittest import mock

        from xreview.adapters.agy import AgyAdapter
        from xreview.adapters.base import TurnRequest
        from xreview.config import AppConfig, ConfigError, ReviewerConfig, load_config
        from xreview.sandbox import SandboxRunResult

        with tempfile.TemporaryDirectory(prefix="xr-to0-") as td:
            root = Path(td)
            cfg_path = root / "config.toml"
            cfg_path.write_text("[policy]\ntimeout_seconds = 0\n", encoding="utf-8")
            cfg = load_config(cfg_path)
            self.assertEqual(cfg.policy.timeout_seconds, 0)

            cfg_default = root / "default.toml"
            cfg_default.write_text("", encoding="utf-8")
            self.assertEqual(load_config(cfg_default).policy.timeout_seconds, 0)

            cfg_bad = root / "bad.toml"
            cfg_bad.write_text("[policy]\ntimeout_seconds = -5\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(cfg_bad)

            # Running with timeout_s=0 must NOT immediately fire a 0s timer
            tree, inputs, state = root / "tree", root / "inputs", root / "state"
            tree.mkdir(); inputs.mkdir(); state.mkdir()
            spec = SandboxSpec(
                run_id="r0",
                reviewer_name="rev",
                tree_host_dir=tree,
                inputs_host_dir=inputs,
                state_host_dir=state,
                harness_mounts=(),
                command=("python3", "-c", "import time; time.sleep(0.05); print('NO_TIMEOUT_OK')"),
                cwd_in_sandbox=str(tree),
                timeout_s=0,
                use_sandbox=False,
            )
            res = run_in_sandbox(spec, None, root / "raw.log", root / "proxy.log", capture_stdout=True)
            self.assertFalse(res.timed_out)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("NO_TIMEOUT_OK", res.stdout_lines)

            # AgyAdapter passes --print-timeout 0s and SandboxSpec.timeout_s=0 (NOT 0+30)
            captured_spec: list[SandboxSpec] = []

            def _fake_sb(sp, *args, **kwargs):
                captured_spec.append(sp)
                return SandboxRunResult(returncode=0, stdout_lines=[], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[])

            req = TurnRequest(
                run_id="r0",
                round_num=1,
                reviewer=ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google"),
                prompt="hi",
                schema_host_path=inputs / "schema.json",
                tree_host_dir=tree,
                inputs_host_dir=inputs,
                state_host_dir=state,
                raw_log_path=root / "agy.jsonl",
                proxy_log_path=root / "agy.proxy.log",
                timeout_s=0,
                allow_no_sandbox=True,
            )
            with (
                mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                mock.patch("xreview.adapters.agy.run_in_sandbox", side_effect=_fake_sb),
            ):
                AgyAdapter().run_turn(req, AppConfig(state_dir=root / "st"))
            self.assertEqual(len(captured_spec), 1)
            self.assertEqual(captured_spec[0].timeout_s, 0)
            idx = captured_spec[0].command.index("--print-timeout")
            self.assertEqual(captured_spec[0].command[idx + 1], "0s")


if __name__ == "__main__":
    unittest.main()
