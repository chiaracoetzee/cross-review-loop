"""Codex CLI harness adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

from xreview.adapters.base import (
    HarnessAdapter,
    TurnRequest,
    TurnResult,
    detect_quota_exhausted,
    detect_resume_failed,
    parse_json_from_text,
)
from xreview.config import (
    TESTED_HARNESS_VERSIONS,
    AppConfig,
    ReviewerConfig,
    safe_read_regular_file_under,
)
from xreview.quota import parse_reset_hint_from_error, read_codex_rollout_quota
from xreview.sandbox import SandboxSpec, run_in_sandbox


def _find_codex_mounts() -> tuple[tuple[tuple[Path, str], ...], str, Path, str]:
    """Resolve codex binary and return (sandbox_mounts, sandbox_exe, host_exe, sandbox_path_env).

    Only mounts a directory when it is verified as a dedicated Codex standalone release
    (contains codex-package.json alongside bin/codex). Otherwise mounts ONLY individual
    required codex executables, never exposing an entire shared prefix like /usr/local (astra-59).
    """
    exe = shutil.which("codex")
    if not exe:
        raise FileNotFoundError("codex executable not found in PATH")
    real = Path(os.path.realpath(exe))
    if (
        real.parent.name == "bin"
        and (real.parent.parent / "codex-package.json").is_file()
        and (real.parent / "codex-code-mode-host").is_file()
    ):
        pkg_root = real.parent.parent
        return (
            ((pkg_root, "/harness/codex-pkg"),),
            f"/harness/codex-pkg/bin/{real.name}",
            real,
            "/harness/codex-pkg/bin:/harness/codex-pkg:/usr/bin:/bin",
        )
    if real.is_file():
        mounts: list[tuple[Path, str]] = [(real, f"/harness/codex-bin/{real.name}")]
        for helper in ("codex-code-mode-host", "codex-exec-server"):
            hpath = real.parent / helper
            if hpath.is_file() and not hpath.is_symlink():
                mounts.append((hpath, f"/harness/codex-bin/{helper}"))
        return (
            tuple(mounts),
            f"/harness/codex-bin/{real.name}",
            real,
            "/harness/codex-bin:/usr/bin:/bin",
        )
    raise FileNotFoundError(f"Unsupported codex executable layout at {real}")


def _find_safe_rollout_relpaths(codex_home: Path, thread_id: str | None) -> list[str]:
    """Walk codex_home/sessions without following symlinks and return matching relative paths."""
    sessions_root = codex_home / "sessions"
    if not sessions_root.exists() or sessions_root.is_symlink():
        return []
    matches: list[tuple[float, str]] = []
    for root, dirs, files in os.walk(sessions_root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            if thread_id and thread_id not in fname:
                continue
            if not thread_id and not fname.startswith("rollout-"):
                continue
            full = Path(root) / fname
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            rel = str(full.relative_to(codex_home))
            matches.append((st.st_mtime, rel))
    matches.sort(reverse=True)
    return [rel for _, rel in matches]


def _read_codex_rollout_rate_limit(codex_home: Path, thread_id: str | None) -> tuple[float | None, bool]:
    """Safely read the LATEST primary/secondary rate_limits sample from the tail of Codex's rollout (astra-54)."""
    relpaths = _find_safe_rollout_relpaths(codex_home, thread_id)
    if not relpaths:
        return None, False

    raw = safe_read_regular_file_under(codex_home, relpaths[0], max_bytes=2_097_152, tail=True)
    if not raw:
        return None, False

    latest_ratio: float | None = None
    latest_reached = False
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if "rate_limits" not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        rl = (ev.get("payload") or {}).get("rate_limits")
        if not isinstance(rl, dict):
            continue
        latest_reached = bool(rl.get("rate_limit_reached_type"))
        sample_max: float | None = None
        for window in ("primary", "secondary"):
            w = rl.get(window)
            if isinstance(w, dict) and isinstance(w.get("used_percent"), (int, float)):
                ratio = float(w["used_percent"]) / 100.0
                if sample_max is None or ratio > sample_max:
                    sample_max = ratio
        if sample_max is not None:
            latest_ratio = sample_max
    return latest_ratio, latest_reached


class CodexAdapter(HarnessAdapter):
    name = "codex"
    tested_version = TESTED_HARNESS_VERSIONS["codex"]

    def version(self) -> str | None:
        raw = self._run_version_cmd("codex")
        if not raw:
            return None
        m = re.search(r"(\d+\.\d+\.\d+)", raw)
        return m.group(1) if m else raw

    def prepare_state(self, reviewer: ReviewerConfig, state_dir: Path, config: AppConfig) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(state_dir, 0o700)
        codex_home = state_dir / ".codex"
        if codex_home.is_symlink():
            raise PermissionError(f"Refusing symlinked .codex dir in reviewer state: {codex_home}")
        codex_home.mkdir(parents=True, exist_ok=True)
        auth_dest = codex_home / "auth.json"
        if not auth_dest.exists() and not auth_dest.is_symlink():
            src = config.auth.codex_auth_file or (Path.home() / ".codex" / "auth.json")
            if not src.exists():
                raise FileNotFoundError(
                    f"Codex auth.json not found at {src}. Sign in with `codex login` or set auth.codex_auth_file."
                )
            data = src.read_bytes()
            fd = os.open(str(auth_dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)

    def run_turn(self, req: TurnRequest, config: AppConfig) -> TurnResult:
        self.prepare_state(req.reviewer, req.state_host_dir, config)
        use_sandbox = not req.allow_no_sandbox
        sb_mounts, sb_bin, host_exe, sb_path_env = _find_codex_mounts()

        if use_sandbox:
            exe_path = sb_bin
            tree_path = f"/review/{req.run_id}/{req.reviewer.name}/tree"
            inputs_path = f"/review/{req.run_id}/{req.reviewer.name}/inputs"
            codex_home_env = "/home/reviewer/.codex"
            path_env = sb_path_env
            harness_mounts = sb_mounts
        else:
            exe_path = str(host_exe)
            tree_path = str(req.tree_host_dir)
            inputs_path = str(req.inputs_host_dir)
            codex_home_env = str(req.state_host_dir / ".codex")
            path_env = f"{host_exe.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            harness_mounts = ()

        schema_path = f"{inputs_path}/{req.schema_host_path.name}"

        if req.session_id:
            cmd = [
                exe_path,
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "-m",
                req.reviewer.model,
            ]
            if req.reviewer.effort:
                cmd.extend(["-c", f'model_reasoning_effort="{req.reviewer.effort}"'])
            cmd.extend([
                "-c",
                'sandbox_mode="read-only"',
                "--output-schema",
                schema_path,
                req.session_id,
                "-",
            ])
        else:
            cmd = [
                exe_path,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "-m",
                req.reviewer.model,
            ]
            if req.reviewer.effort:
                cmd.extend(["-c", f'model_reasoning_effort="{req.reviewer.effort}"'])
            cmd.extend([
                "-C",
                tree_path,
                "--sandbox",
                "read-only",
                "--output-schema",
                schema_path,
                "-",
            ])

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
                "CODEX_HOME": codex_home_env,
                "PATH": path_env,
            },
            vendor_hosts=config.vendor_hosts_for(req.reviewer.vendor),
            extra_ro_paths=config.profile.extra_ro_paths,
            timeout_s=req.timeout_s,
            use_sandbox=use_sandbox,
        )

        session_id = req.session_id
        last_agent_text = ""
        usage: dict[str, Any] = {}
        stream_errors: list[str] = []
        turn_completed = False
        turn_failed = False

        def _on_line(line: str) -> None:
            nonlocal session_id, last_agent_text, usage, turn_completed, turn_failed
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                return
            ev_type = ev.get("type")
            if ev_type == "thread.started" and ev.get("thread_id"):
                session_id = str(ev["thread_id"])
                if req.on_session_id:
                    req.on_session_id(session_id)
            elif ev_type == "turn.completed":
                turn_completed = True
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
            elif ev_type == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    last_agent_text = item["text"]
                elif item.get("type") == "error" and item.get("message"):
                    stream_errors.append(str(item["message"]))
            elif ev_type in ("error", "turn.failed"):
                turn_failed = True
                msg = ev.get("message") or (ev.get("error") or {}).get("message") or json.dumps(ev)
                stream_errors.append(str(msg))

        sb_res = run_in_sandbox(
            spec,
            stdin_text=req.prompt,
            raw_log_path=req.raw_log_path,
            proxy_log_path=req.proxy_log_path,
            on_stdout_line=_on_line,
        )

        # Require terminal turn.completed event and clean exit before accepting any reply (astra-37)
        raw_text = last_agent_text
        succeeded = sb_res.returncode == 0 and not sb_res.timed_out and not turn_failed and turn_completed
        reply = parse_json_from_text(raw_text) if succeeded else None

        rollout_q = read_codex_rollout_quota(req.state_host_dir / ".codex", session_id, plan_id=req.reviewer.plan_id)
        allowance_ratio, rl_reached = _read_codex_rollout_rate_limit(
            req.state_host_dir / ".codex", session_id
        )
        diag_err = "\n".join(stream_errors + ([sb_res.stderr] if sb_res.stderr else []))
        # Never let a historical rollout event discard a successful turn after quota reset (astra-54)
        quota = not succeeded and (rl_reached or detect_quota_exhausted(diag_err))
        malformed = bool(succeeded and reply is None)
        resume_failed = bool(
            req.session_id and not succeeded and not quota and not sb_res.timed_out and detect_resume_failed(diag_err)
        )
        quota_windows = [w.to_dict() for w in rollout_q.windows] if rollout_q else []
        resets_at = (rollout_q.exhausted_until or (rollout_q.next_reset_at(rollout_q.updated_at) if rollout_q else None))
        if resets_at is None and quota:
            resets_at = parse_reset_hint_from_error(diag_err)

        error = None
        if sb_res.timed_out:
            error = f"codex timed out after {req.timeout_s}s"
        elif quota:
            error = "codex quota/rate limit reached"
        elif not succeeded:
            error = f"codex failed (rc={sb_res.returncode}): {diag_err[:400].strip()}"
        elif malformed:
            error = "codex produced malformed non-JSON reply"

        return TurnResult(
            session_id=session_id,
            reply=reply,
            raw_text=raw_text,
            usage=usage,
            allowance_used_ratio=allowance_ratio,
            quota_windows=quota_windows,
            quota_resets_at=resets_at,
            quota_exhausted=quota,
            resume_failed=resume_failed,
            malformed_output=malformed,
            timed_out=sb_res.timed_out,
            error=error,
            duration_s=sb_res.duration_s,
            returncode=sb_res.returncode,
        )

    def cleanup(self, reviewer: ReviewerConfig, state_dir: Path, session_id: str | None) -> None:
        if session_id and state_dir.exists() and not state_dir.is_symlink():
            codex_home = state_dir / ".codex"
            if codex_home.is_dir() and not codex_home.is_symlink():
                exe = shutil.which("codex")
                if exe:
                    subprocess.run(
                        [exe, "delete", "--force", session_id],
                        env={**os.environ, "CODEX_HOME": str(codex_home)},
                        capture_output=True,
                        timeout=15,
                        check=False,
                    )
        super().cleanup(reviewer, state_dir, session_id)
