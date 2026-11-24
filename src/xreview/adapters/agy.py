"""Antigravity CLI (agy) harness adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from xreview.adapters.base import (
    HarnessAdapter,
    TurnRequest,
    TurnResult,
    detect_quota_exhausted,
    parse_json_from_text,
)
from xreview.config import TESTED_HARNESS_VERSIONS, AppConfig, ReviewerConfig
from xreview.sandbox import SandboxSpec, run_in_sandbox


def _find_agy_binary() -> Path:
    exe = shutil.which("agy")
    if not exe:
        raise FileNotFoundError("agy executable not found in PATH")
    return Path(os.path.realpath(exe))


def _write_exclusive_nofollow(path: Path, content: bytes, mode: int = 0o600) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)


def _mkdir_and_open_nofollow(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)


import stat


def _ensure_regular_file_at(dir_fd: int, filename: str, content: bytes, mode: int = 0o600) -> None:
    """Idempotently ensure a regular non-symlink file exists relative to dir_fd (astra-4, astra-68)."""
    try:
        existing_fd = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=dir_fd,
        )
    except FileNotFoundError:
        fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=dir_fd,
        )
        try:
            os.write(fd, content)
        finally:
            os.close(fd)
        return
    try:
        st = os.fstat(existing_fd)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Refusing non-regular file {filename!r} inside reviewer state")
    finally:
        os.close(existing_fd)


class AgyAdapter(HarnessAdapter):
    name = "agy"
    tested_version = TESTED_HARNESS_VERSIONS["agy"]

    def version(self) -> str | None:
        raw = self._run_version_cmd("agy")
        if not raw:
            return None
        m = re.search(r"(\d+\.\d+\.\d+)", raw)
        return m.group(1) if m else raw

    def prepare_state(self, reviewer: ReviewerConfig, state_dir: Path, config: AppConfig) -> None:
        state_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Keep initialization marker OUTSIDE reviewer-writable state_dir (astra-4, astra-68)
        initialized_marker = state_dir.parent / f".{reviewer.name}.agy_init"
        if initialized_marker.exists():
            return

        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(state_dir, 0o700)

        src = config.auth.agy_token_file
        if src is None:
            default_candidate = Path.home() / ".config" / "xreview" / "agy-oauth-token"
            if default_candidate.exists():
                src = default_candidate
        if src is None or not src.exists():
            raise FileNotFoundError(
                "Antigravity file-based OAuth token not found. Create one with:\n"
                "  HOME=<auth_dir> DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent XDG_RUNTIME_DIR=/nonexistent agy acp --login\n"
                "and point auth.agy_token_file to <auth_dir>/.gemini/antigravity-cli/antigravity-oauth-token."
            )
        token_bytes = src.read_bytes()
        use_credits = config.policy.credits != "never"
        cfg_bytes = json.dumps(
            {"userSettings": {"useAiCredits": use_credits, "enableTerminalSandbox": True}},
            indent=2,
        ).encode("utf-8")

        root_fd = os.open(str(state_dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            gem_fd = _mkdir_and_open_nofollow(root_fd, ".gemini")
            try:
                cli_fd = _mkdir_and_open_nofollow(gem_fd, "antigravity-cli")
                try:
                    _ensure_regular_file_at(cli_fd, "antigravity-oauth-token", token_bytes, mode=0o600)
                finally:
                    os.close(cli_fd)

                cfg_fd = _mkdir_and_open_nofollow(gem_fd, "config")
                try:
                    _ensure_regular_file_at(cfg_fd, "config.json", cfg_bytes, mode=0o600)
                finally:
                    os.close(cfg_fd)
            finally:
                os.close(gem_fd)
        finally:
            os.close(root_fd)

        try:
            _write_exclusive_nofollow(initialized_marker, b"1\n", mode=0o600)
        except FileExistsError:
            pass

    def run_turn(self, req: TurnRequest, config: AppConfig) -> TurnResult:
        self.prepare_state(req.reviewer, req.state_host_dir, config)
        use_sandbox = not req.allow_no_sandbox
        agy_bin = _find_agy_binary()

        if use_sandbox:
            exe_path = "/harness/agy"
            tree_path = f"/review/{req.run_id}/{req.reviewer.name}/tree"
            inputs_path = f"/review/{req.run_id}/{req.reviewer.name}/inputs"
            docs_path = f"/review/{req.run_id}/{req.reviewer.name}/docs" if req.docs_host_dir else None
            log_path = f"/home/reviewer/agy_r{req.round_num}.log"
            path_env = "/harness:/usr/bin:/bin"
            harness_mounts = ((agy_bin, "/harness/agy"),)
        else:
            exe_path = str(agy_bin)
            tree_path = str(req.tree_host_dir)
            inputs_path = str(req.inputs_host_dir)
            docs_path = str(req.docs_host_dir) if req.docs_host_dir else None
            log_path = str(req.state_host_dir / f"agy_r{req.round_num}.log")
            path_env = f"{agy_bin.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            harness_mounts = ()

        schema_path = f"{inputs_path}/{req.schema_host_path.name}"
        trusted_dirs = [tree_path, inputs_path] + ([docs_path] if docs_path else [])

        # Pass settings exclusively via --settings JSON (never rewriting files inside reviewer state!)
        use_credits = config.policy.credits != "never"
        settings_json = json.dumps({"useG1Credits": use_credits, "trustedWorkspaces": trusted_dirs})

        # Write full prompt into read-only inputs/ to avoid OS argv E2BIG on large follow-up prompts (astra-40)
        prompt_host = req.inputs_host_dir / req.prompt_filename
        prompt_host.write_text(req.prompt, encoding="utf-8")
        prompt_in_env = f"{inputs_path}/{req.prompt_filename}"
        short_prompt = (
            f"Read `{prompt_in_env}` completely and follow all instructions inside it. "
            f"Return ONLY valid JSON matching `{schema_path}`."
        )

        cmd = [
            exe_path,
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--dangerously-skip-permissions",
            "--model",
            req.reviewer.model,
            "--settings",
            settings_json,
            "--add-dir",
            inputs_path,
            "--json-schema",
            schema_path,
            "--print-timeout",
            f"{req.timeout_s}s",
            "--log-file",
            log_path,
        ]
        if docs_path:
            cmd.extend(["--add-dir", docs_path])
        if req.reviewer.effort:
            cmd.extend(["--effort", req.reviewer.effort])
        if req.session_id:
            cmd.extend(["--conversation", req.session_id])
        cmd.extend(["-p", short_prompt])

        spec = SandboxSpec(
            run_id=req.run_id,
            reviewer_name=req.reviewer.name,
            tree_host_dir=req.tree_host_dir,
            inputs_host_dir=req.inputs_host_dir,
            docs_host_dir=req.docs_host_dir,
            state_host_dir=req.state_host_dir,
            harness_mounts=harness_mounts,
            command=tuple(cmd),
            cwd_in_sandbox=tree_path,
            env={
                "PATH": path_env,
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/nonexistent",
                "XDG_RUNTIME_DIR": "/nonexistent",
            },
            vendor_hosts=config.vendor_hosts_for(req.reviewer.vendor),
            extra_ro_paths=config.profile.extra_ro_paths,
            timeout_s=req.timeout_s + 30,
            use_sandbox=use_sandbox,
        )

        session_id = req.session_id
        structured_output: dict[str, Any] | None = None
        response_text = ""
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cache_read_tokens": 0}
        result_status = ""
        saw_result = False
        stream_errors: list[str] = []

        def _on_line(line: str) -> None:
            nonlocal session_id, structured_output, response_text, result_status, saw_result
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                return
            ev_name = ev.get("event")
            if ev.get("conversation_id") and not session_id:
                session_id = str(ev["conversation_id"])
                if req.on_session_id:
                    req.on_session_id(session_id)
            if ev_name == "step_update":
                su = ev.get("step_update") or {}
                if su.get("conversation_id") and not session_id:
                    session_id = str(su["conversation_id"])
                    if req.on_session_id:
                        req.on_session_id(session_id)
                u = su.get("usage")
                if isinstance(u, dict):
                    for k in usage_totals:
                        if isinstance(u.get(k), (int, float)):
                            usage_totals[k] += int(u[k])
            elif ev_name == "result":
                saw_result = True
                res = ev.get("result") or {}
                if res.get("conversation_id"):
                    session_id = str(res["conversation_id"])
                    if req.on_session_id:
                        req.on_session_id(session_id)
                result_status = str(res.get("status", ""))
                if isinstance(res.get("structured_output"), dict):
                    structured_output = res["structured_output"]
                if isinstance(res.get("response"), str):
                    response_text = res["response"]
                if res.get("error"):
                    stream_errors.append(str(res["error"]))
            elif ev_name == "error":
                stream_errors.append(json.dumps(ev))

        sb_res = run_in_sandbox(
            spec,
            stdin_text=None,
            raw_log_path=req.raw_log_path,
            proxy_log_path=req.proxy_log_path,
            on_stdout_line=_on_line,
        )

        # Require successful process exit AND terminal SUCCESS result status before accepting reply (astra-37)
        succeeded = sb_res.returncode == 0 and not sb_res.timed_out and saw_result and result_status == "SUCCESS"
        reply = (structured_output or parse_json_from_text(response_text)) if succeeded else None

        diag_err = "\n".join(stream_errors + ([sb_res.stderr] if sb_res.stderr else []))
        quota = not succeeded and detect_quota_exhausted(diag_err)
        malformed = bool(succeeded and reply is None)
        resume_failed = bool(
            req.session_id and not succeeded and not quota and not sb_res.timed_out
        )
        error = None
        if sb_res.timed_out:
            error = f"agy timed out after {req.timeout_s}s"
        elif quota:
            error = "agy quota/rate limit reached"
        elif not succeeded:
            error = f"agy failed (rc={sb_res.returncode}, status={result_status!r}): {diag_err[:400].strip()}"
        elif malformed:
            error = f"agy produced malformed non-JSON reply (status={result_status!r})"

        return TurnResult(
            session_id=session_id,
            reply=reply,
            raw_text=response_text,
            usage=usage_totals,
            allowance_used_ratio=None,
            quota_exhausted=quota,
            resume_failed=resume_failed,
            malformed_output=malformed,
            timed_out=sb_res.timed_out,
            error=error,
            duration_s=sb_res.duration_s,
            returncode=sb_res.returncode,
        )
