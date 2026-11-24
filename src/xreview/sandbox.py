"""Allowlist bubblewrap sandbox and vendor-filtering Unix-socket HTTP CONNECT proxy."""

from __future__ import annotations

import ipaddress
import os
import select
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class SandboxUnavailableError(RuntimeError):
    """Raised when bubblewrap is not installed and --no-sandbox was not passed."""


def host_matches_allowlist(host: str, allowlist: tuple[str, ...]) -> bool:
    h = host.strip().lower().rstrip(".")
    if not h:
        return False
    for entry in allowlist:
        e = entry.strip().lower().rstrip(".")
        if h == e or h.endswith("." + e):
            return True
    return False


def is_public_ip(ip_str: str) -> bool:
    """Return True only for globally routable IPv4/IPv6 addresses (no loopback/private/link-local)."""
    try:
        addr = ipaddress.ip_address(ip_str.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return bool(
        addr.is_global
        and not addr.is_private
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_multicast
        and not addr.is_reserved
        and not addr.is_unspecified
    )


def resolve_and_connect_public(host: str, port: int, timeout: float = 30.0) -> socket.socket:
    """Resolve host once, verify every connected candidate IP is public, and connect directly."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos:
        raise OSError(f"No addresses resolved for {host}")

    # Validate ALL resolved addresses; if DNS returns any non-public address, refuse the connection.
    public_infos = []
    for info in infos:
        ip_str = info[4][0]
        if not is_public_ip(ip_str):
            raise PermissionError(f"Blocked non-public resolved address {ip_str} for {host}")
        public_infos.append(info)

    last_err: OSError | None = None
    for family, socktype, proto, _canon, sockaddr in public_infos:
        s = socket.socket(family, socktype, proto)
        s.settimeout(timeout)
        try:
            s.connect(sockaddr)
            return s
        except OSError as exc:
            s.close()
            last_err = exc
    raise last_err or OSError(f"Failed to connect to {host}:{port}")


class _VendorProxyHandler(socketserver.StreamRequestHandler):
    # Unbuffered rfile so post-CONNECT TLS bytes are never over-read before _relay (F-003, F-018)
    rbufsize = 0

    def handle(self) -> None:
        try:
            self.connection.settimeout(30.0)
            raw_line = self.rfile.readline(65536)
            if not raw_line:
                return
            req_line = raw_line.decode("latin1", errors="replace").strip()
            parts = req_line.split()
            if len(parts) < 3:
                self.wfile.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                self.wfile.flush()
                return
            method, target = parts[0].upper(), parts[1]
            while True:
                line = self.rfile.readline(65536)
                if not line or line in (b"\r\n", b"\n"):
                    break

            if method != "CONNECT":
                self.server.record(method, target, 80, False)
                body = b"Blocked by xreview proxy: only HTTPS CONNECT allowed"
                self.wfile.write(
                    f"HTTP/1.1 403 Forbidden\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii")
                    + body
                )
                self.wfile.flush()
                return

            if target.startswith("["):
                end_br = target.find("]")
                host = target[1:end_br] if end_br != -1 else target
                rest = target[end_br + 1 :] if end_br != -1 else ""
                port_s = rest[1:] if rest.startswith(":") else "443"
            else:
                host, _, port_s = target.partition(":")
                port_s = port_s or "443"
            try:
                port = int(port_s)
            except ValueError:
                self.wfile.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                self.wfile.flush()
                return

            allowed = port in self.server.allowed_ports and host_matches_allowlist(
                host, self.server.allowlist
            )
            if not allowed:
                self.server.record(method, host, port, False)
                body = b"Blocked by xreview vendor allowlist"
                self.wfile.write(
                    f"HTTP/1.1 403 Forbidden\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii")
                    + body
                )
                self.wfile.flush()
                return

            try:
                upstream = resolve_and_connect_public(host, port, timeout=30.0)
            except PermissionError as exc:
                self.server.record(method, host, port, False)
                msg = str(exc).encode("utf-8", errors="replace")
                self.wfile.write(
                    f"HTTP/1.1 403 Forbidden\r\nContent-Length: {len(msg)}\r\n\r\n".encode("ascii")
                    + msg
                )
                self.wfile.flush()
                return
            except OSError as exc:
                self.server.record(method, host, port, True)
                msg = f"Upstream connection failed: {exc}".encode("utf-8", errors="replace")
                self.wfile.write(
                    f"HTTP/1.1 502 Bad Gateway\r\nContent-Length: {len(msg)}\r\n\r\n".encode("ascii")
                    + msg
                )
                self.wfile.flush()
                return

            self.server.record(method, host, port, True)
            try:
                self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.wfile.flush()
                self.connection.settimeout(None)
                upstream.settimeout(None)
                self._relay(self.connection, upstream, idle_timeout=self.server.idle_timeout)
            finally:
                upstream.close()
        except OSError:
            pass

    @staticmethod
    def _relay(sock_a: socket.socket, sock_b: socket.socket, idle_timeout: float = 300.0) -> None:
        sockets = [sock_a, sock_b]
        while True:
            readable, _, _ = select.select(sockets, [], [], idle_timeout)
            if not readable:
                break
            for s in readable:
                other = sock_b if s is sock_a else sock_a
                try:
                    data = s.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                try:
                    other.sendall(data)
                except OSError:
                    return


class _UnixVendorProxyServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        sock_path: str,
        allowlist: tuple[str, ...],
        allowed_ports: tuple[int, ...] = (443,),
        log_path: Path | None = None,
        idle_timeout: float = 300.0,
    ) -> None:
        self.allowlist = allowlist
        self.allowed_ports = allowed_ports
        self.log_path = log_path
        self.idle_timeout = idle_timeout
        self.entries: list[tuple[str, str, int, bool]] = []
        self._lock = threading.Lock()
        super().__init__(sock_path, _VendorProxyHandler)

    def record(self, method: str, host: str, port: int, allowed: bool) -> None:
        with self._lock:
            self.entries.append((method, host, port, allowed))
            if self.log_path is not None:
                status = "ALLOW" if allowed else "BLOCK"
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{ts} {status} {method} {host}:{port}\n")


class VendorProxy:
    """Context manager for a host-side Unix-socket HTTPS CONNECT proxy."""

    def __init__(
        self,
        socket_path: Path,
        allowlist: tuple[str, ...],
        allowed_ports: tuple[int, ...] = (443,),
        log_path: Path | None = None,
        idle_timeout: float = 300.0,
    ) -> None:
        self.socket_path = socket_path
        self.allowlist = allowlist
        self.allowed_ports = allowed_ports
        self.log_path = log_path
        self.idle_timeout = idle_timeout
        self._server: _UnixVendorProxyServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def entries(self) -> list[tuple[str, str, int, bool]]:
        return list(self._server.entries) if self._server else []

    def __enter__(self) -> VendorProxy:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self._server = _UnixVendorProxyServer(
            str(self.socket_path),
            allowlist=self.allowlist,
            allowed_ports=self.allowed_ports,
            log_path=self.log_path,
            idle_timeout=self.idle_timeout,
        )
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self.socket_path.unlink(missing_ok=True)


_PY_PROXY_BRIDGE = r"""
import os, select, socket, sys, threading
sock_path = sys.argv[1]
port = int(sys.argv[2])
idle_timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(64)

def pipe(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], idle_timeout)
            if not r:
                break
            for s in r:
                o = b if s is a else a
                d = s.recv(65536)
                if not d:
                    return
                o.sendall(d)
    except OSError:
        pass
    finally:
        for x in (a, b):
            try: x.close()
            except OSError: pass

while True:
    try:
        cli, _ = srv.accept()
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        u.connect(sock_path)
        threading.Thread(target=pipe, args=(cli, u), daemon=True).start()
    except OSError:
        break
"""


@dataclass(frozen=True)
class SandboxSpec:
    run_id: str
    reviewer_name: str
    tree_host_dir: Path
    inputs_host_dir: Path
    state_host_dir: Path
    harness_mounts: tuple[tuple[Path, str], ...]
    command: tuple[str, ...]
    cwd_in_sandbox: str
    docs_host_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    secret_env: dict[str, str] = field(default_factory=dict)  # Never placed in host bwrap argv!
    vendor_hosts: tuple[str, ...] = ()
    extra_ro_paths: tuple[str, ...] = ()
    timeout_s: int = 0
    use_sandbox: bool = True

    @property
    def tree_mount(self) -> str:
        return f"/review/{self.run_id}/{self.reviewer_name}/tree"

    @property
    def inputs_mount(self) -> str:
        return f"/review/{self.run_id}/{self.reviewer_name}/inputs"

    @property
    def docs_mount(self) -> str:
        return f"/review/{self.run_id}/{self.reviewer_name}/docs"


@dataclass
class SandboxRunResult:
    returncode: int
    stdout_lines: list[str]
    stderr: str
    timed_out: bool
    duration_s: float
    proxy_entries: list[tuple[str, str, int, bool]]


def bwrap_available() -> bool:
    return shutil.which("bwrap") is not None


def _write_synthetic_etc(etc_dir: Path) -> None:
    etc_dir.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    gid = os.getgid()
    (etc_dir / "passwd").write_text(
        f"root:x:0:0:root:/root:/bin/sh\nreviewer:x:{uid}:{gid}:reviewer:/home/reviewer:/bin/sh\n",
        encoding="utf-8",
    )
    (etc_dir / "group").write_text(f"root:x:0:\nreviewer:x:{gid}:\n", encoding="utf-8")
    (etc_dir / "hosts").write_text("127.0.0.1 localhost\n::1 localhost\n", encoding="utf-8")
    (etc_dir / "nsswitch.conf").write_text("hosts: files\n", encoding="utf-8")


def _sandbox_accessible_socat() -> str | None:
    """Return a socat path guaranteed to exist inside the allowlist bwrap mount, or None (F-006, F-014)."""
    found = shutil.which("socat")
    if not found:
        return None
    try:
        resolved = Path(found).resolve()
    except OSError:
        return None
    mounted_prefixes = ("/usr/bin/", "/bin/", "/usr/sbin/", "/sbin/")
    if any(str(resolved).startswith(p) for p in mounted_prefixes) and resolved.is_file():
        return str(resolved)
    return None


def build_bwrap_command(
    spec: SandboxSpec,
    proxy_socket_host: Path,
    etc_dir: Path,
    use_socat: bool = True,
    secret_fd: int | None = None,
    socat_path: str | None = None,
) -> list[str]:
    cmd: list[str] = ["bwrap"]
    for sys_dir in ("/usr", "/lib", "/lib64", "/bin", "/sbin"):
        if Path(sys_dir).exists():
            cmd.extend(["--ro-bind", sys_dir, sys_dir])

    # Mount ONLY public TLS CA bundles/certs — NEVER /etc/ssl/private!
    for tls_path in (
        "/etc/ssl/certs",
        "/etc/ssl/openssl.cnf",
        "/etc/ca-certificates",
        "/etc/ca-certificates.conf",
        "/etc/pki/tls/certs",
    ):
        if Path(tls_path).exists():
            cmd.extend(["--ro-bind", tls_path, tls_path])

    for extra in spec.extra_ro_paths:
        if Path(extra).exists():
            cmd.extend(["--ro-bind", extra, extra])

    cmd.extend([
        "--ro-bind", str(etc_dir / "passwd"), "/etc/passwd",
        "--ro-bind", str(etc_dir / "group"), "/etc/group",
        "--ro-bind", str(etc_dir / "hosts"), "/etc/hosts",
        "--ro-bind", str(etc_dir / "nsswitch.conf"), "/etc/nsswitch.conf",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/run",
        "--dir", "/run/xreview",
        "--bind", str(proxy_socket_host), "/run/xreview/proxy.sock",
    ])

    for host_p, sb_p in spec.harness_mounts:
        cmd.extend(["--ro-bind", str(host_p), sb_p])

    cmd.extend([
        "--bind", str(spec.state_host_dir), "/home/reviewer",
        "--ro-bind", str(spec.tree_host_dir), spec.tree_mount,
        "--ro-bind", str(spec.inputs_host_dir), spec.inputs_mount,
    ])
    if spec.docs_host_dir is not None and spec.docs_host_dir.exists():
        cmd.extend(["--ro-bind", str(spec.docs_host_dir), spec.docs_mount])

    if secret_fd is not None:
        cmd.extend(["--file", str(secret_fd), "/run/xreview/secret.env"])

    cmd.extend([
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--clearenv",
    ])

    base_env = {
        "HOME": "/home/reviewer",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HTTP_PROXY": "http://127.0.0.1:18080",
        "HTTPS_PROXY": "http://127.0.0.1:18080",
        "http_proxy": "http://127.0.0.1:18080",
        "https_proxy": "http://127.0.0.1:18080",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
    }
    base_env.update(spec.env)
    for k, v in sorted(base_env.items()):
        cmd.extend(["--setenv", k, v])

    target_cmd = " ".join(shlex.quote(a) for a in spec.command)
    bridge_idle = max(300, int(spec.timeout_s)) if spec.timeout_s > 0 else 3600
    if use_socat:
        socat_exe = shlex.quote(socat_path or "/usr/bin/socat")
        bridge_start = (
            f"{socat_exe} TCP-LISTEN:18080,bind=127.0.0.1,reuseaddr,fork "
            "UNIX-CONNECT:/run/xreview/proxy.sock 2>/dev/null &"
        )
    else:
        bridge_start = (
            f"python3 -c {shlex.quote(_PY_PROXY_BRIDGE)} "
            f"/run/xreview/proxy.sock 18080 {bridge_idle} &"
        )

    secret_loader = ""
    if secret_fd is not None:
        # Load secret exports written via `bwrap --file` into tmpfs, then delete before exec
        secret_loader = ". /run/xreview/secret.env && rm -f /run/xreview/secret.env\n"

    wrapper = (
        f"{secret_loader}"
        f"{bridge_start}\n"
        "BRIDGE_PID=$!\n"
        'trap "kill $BRIDGE_PID 2>/dev/null || true" EXIT\n'
        "python3 -c 'import socket,sys,time\n"
        "for _ in range(100):\n"
        "  try:\n"
        "    s=socket.create_connection((\"127.0.0.1\",18080),0.1); s.close(); sys.exit(0)\n"
        "  except OSError: time.sleep(0.02)\n"
        "sys.stderr.write(\"[xreview] sandbox proxy bridge failed to bind 127.0.0.1:18080\\n\")\n"
        "sys.exit(97)' || exit 97\n"
        f"cd {shlex.quote(spec.cwd_in_sandbox)}\n"
        f"{target_cmd}\n"
    )
    cmd.extend(["--", "/bin/sh", "-c", wrapper])
    return cmd


def _format_secret_exports(secret_env: dict[str, str]) -> bytes:
    lines = [f"export {k}={shlex.quote(v)}" for k, v in sorted(secret_env.items())]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _kill_group(proc: subprocess.Popen[str], *, already_exited: bool = False) -> None:
    """Terminate and SIGKILL the process group while proc.pid is still pinned (unreaped, F-005, F-015, F-022).

    Never calls proc.wait() so leader reaping stays exclusively with the main thread after waitid.
    """
    if proc.returncode is not None:
        # Leader PID was already reaped; never signal a potentially recycled PGID (F-015)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    time.sleep(0.05 if already_exited else 0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


_MAX_STDERR_BYTES = 262_144  # 256 KB bounded stderr retention (astra-71)
_MAX_EVENT_LINE_CHARS = 4_194_304  # 4 MB per line cap


def run_in_sandbox(
    spec: SandboxSpec,
    stdin_text: str | None,
    raw_log_path: Path,
    proxy_log_path: Path,
    on_stdout_line: Callable[[str], None] | None = None,
    capture_stdout: bool = False,
) -> SandboxRunResult:
    """Run a harness command inside the allowlist bwrap sandbox + vendor proxy.

    Streams stdout line-by-line to raw_log_path and on_stdout_line as it arrives
    without retaining unbounded stdout/stderr in memory (astra-71).
    """
    if spec.use_sandbox and not bwrap_available():
        raise SandboxUnavailableError(
            "bubblewrap (bwrap) is not installed. Install bubblewrap or pass --no-sandbox explicitly."
        )

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.state_host_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    stdout_lines: list[str] = []
    stderr_chunks: list[str] = []
    stderr_bytes = 0
    stderr_lock = threading.Lock()
    timed_out = False

    def _append_stderr(text: str) -> None:
        nonlocal stderr_bytes
        with stderr_lock:
            if stderr_bytes < _MAX_STDERR_BYTES:
                keep = text[: _MAX_STDERR_BYTES - stderr_bytes]
                stderr_chunks.append(keep)
                stderr_bytes += len(keep)

    def _stream_proc(proc: subprocess.Popen[str]) -> int:
        nonlocal timed_out
        state_lock = threading.Lock()
        finished = False

        def _mark_timeout() -> None:
            nonlocal timed_out
            with state_lock:
                if finished:
                    return
                timed_out = True
                _kill_group(proc)

        timer: threading.Timer | None = None
        if spec.timeout_s > 0:
            timer = threading.Timer(spec.timeout_s, _mark_timeout)
            timer.daemon = True
            timer.start()

        def _cancel_timer() -> None:
            if timer is not None:
                timer.cancel()

        def _write_stdin() -> None:
            if proc.stdin is None:
                return
            try:
                if stdin_text is not None:
                    proc.stdin.write(stdin_text)
                    proc.stdin.flush()
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        def _read_stderr() -> None:
            if proc.stderr is None:
                return
            try:
                while True:
                    chunk = proc.stderr.read(8192)
                    if not chunk:
                        break
                    _append_stderr(chunk)
            except OSError:
                pass

        in_thread = threading.Thread(target=_write_stdin, daemon=True)
        err_thread = threading.Thread(target=_read_stderr, daemon=True)
        in_thread.start()
        err_thread.start()

        # Monitor leader exit concurrently via waitid(WNOWAIT) so descendants holding stdout OR stderr
        # open in --no-sandbox mode are cleaned up immediately after leader exit without blocking
        # readline() until timeout, while keeping proc.pid pinned until proc.wait() reaps it (F-005, F-022).
        waitid_fn = getattr(os, "waitid", None)
        has_waitid = waitid_fn is not None and hasattr(os, "WNOWAIT")

        def _watch_leader_exit() -> None:
            nonlocal finished
            if not has_waitid:
                return
            try:
                waitid_fn(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
            except (ChildProcessError, OSError):
                return
            with state_lock:
                if not finished:
                    finished = True
                    _cancel_timer()
                    _kill_group(proc, already_exited=True)

        watch_thread = threading.Thread(target=_watch_leader_exit, daemon=True)
        if has_waitid:
            watch_thread.start()

        try:
            assert proc.stdout is not None
            with raw_log_path.open("a", encoding="utf-8") as log_fh:
                while True:
                    line = proc.stdout.readline(_MAX_EVENT_LINE_CHARS)
                    if not line:
                        break
                    log_fh.write(line)
                    if len(line) >= _MAX_EVENT_LINE_CHARS and not line.endswith("\n"):
                        # Drain oversized event line to log without passing split fragments to parsers (F-004, F-016)
                        dropped_len = len(line)
                        while True:
                            extra = proc.stdout.readline(_MAX_EVENT_LINE_CHARS)
                            if not extra:
                                break
                            log_fh.write(extra)
                            dropped_len += len(extra)
                            if extra.endswith("\n"):
                                break
                        log_fh.flush()
                        _append_stderr(
                            f"\n[xreview] dropped oversized stdout event line ({dropped_len} chars > {_MAX_EVENT_LINE_CHARS})\n"
                        )
                        continue
                    log_fh.flush()
                    stripped = line.rstrip("\r\n")
                    if capture_stdout:
                        stdout_lines.append(stripped)
                    if on_stdout_line is not None:
                        on_stdout_line(stripped)

            if has_waitid:
                watch_thread.join(timeout=5)
            with state_lock:
                if not finished:
                    finished = True
                    _cancel_timer()
                    _kill_group(proc, already_exited=True)
            proc.wait()
            err_thread.join(timeout=2)
            in_thread.join(timeout=2)
        except BaseException:
            with state_lock:
                finished = True
                _cancel_timer()
                _kill_group(proc)
            if has_waitid:
                watch_thread.join(timeout=2)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            with state_lock:
                finished = True
                _cancel_timer()
                if proc.poll() is None:
                    _kill_group(proc)
            if has_waitid:
                watch_thread.join(timeout=2)
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            err_thread.join(timeout=2)
            in_thread.join(timeout=2)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        return proc.returncode if proc.returncode is not None else -1

    if not spec.use_sandbox:
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL")}
        env.update(spec.env)
        env.update(spec.secret_env)
        env["HOME"] = str(spec.state_host_dir)
        proc = subprocess.Popen(
            list(spec.command),
            cwd=str(spec.tree_host_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        rc = _stream_proc(proc)
        return SandboxRunResult(
            returncode=rc,
            stdout_lines=stdout_lines,
            stderr="".join(stderr_chunks),
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            proxy_entries=[],
        )

    with tempfile.TemporaryDirectory(prefix="xreview-sb-") as tmp_str:
        tmp = Path(tmp_str)
        os.chmod(tmp, 0o700)
        sock_path = tmp / "proxy.sock"
        etc_dir = tmp / "etc"
        _write_synthetic_etc(etc_dir)

        secret_r: int | None = None
        secret_w: int | None = None
        pass_fds: tuple[int, ...] = ()
        if spec.secret_env:
            secret_r, secret_w = os.pipe()
            os.set_inheritable(secret_r, True)
            pass_fds = (secret_r,)

        try:
            proxy_idle = max(300, int(spec.timeout_s)) if spec.timeout_s > 0 else 3600
            with VendorProxy(
                sock_path,
                allowlist=spec.vendor_hosts,
                log_path=proxy_log_path,
                idle_timeout=proxy_idle,
            ) as proxy:
                socat_path = _sandbox_accessible_socat()
                bwrap_cmd = build_bwrap_command(
                    spec,
                    sock_path,
                    etc_dir,
                    use_socat=socat_path is not None,
                    secret_fd=secret_r,
                    socat_path=socat_path,
                )
                proc = subprocess.Popen(
                    bwrap_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    pass_fds=pass_fds,
                )
                if secret_r is not None:
                    os.close(secret_r)
                    secret_r = None
                if secret_w is not None:
                    os.write(secret_w, _format_secret_exports(spec.secret_env))
                    os.close(secret_w)
                    secret_w = None
                rc = _stream_proc(proc)
                proxy_entries = proxy.entries
        finally:
            for fd in (secret_r, secret_w):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    return SandboxRunResult(
        returncode=rc,
        stdout_lines=stdout_lines,
        stderr="".join(stderr_chunks),
        timed_out=timed_out,
        duration_s=time.monotonic() - start,
        proxy_entries=proxy_entries,
    )
