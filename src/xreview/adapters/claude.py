"""Claude Code CLI harness adapter."""

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
    detect_resume_failed,
    parse_json_from_text,
)
from xreview.config import TESTED_HARNESS_VERSIONS, AppConfig, ReviewerConfig
from xreview.quota import parse_reset_hint_from_error
from xreview.sandbox import SandboxSpec, run_in_sandbox


def _find_claude_binary() -> Path:
    exe = shutil.which("claude")
    if not exe:
        raise FileNotFoundError("claude executable not found in PATH")
    return Path(os.path.realpath(exe))


class ClaudeAdapter(HarnessAdapter):
    name = "claude"
    tested_version = TESTED_HARNESS_VERSIONS["claude"]

    def version(self) -> str | None:
        raw = self._run_version_cmd("claude")
        if not raw:
            return None
        m = re.search(r"(\d+\.\d+\.\d+)", raw)
        return m.group(1) if m else raw

    def _read_oauth_token(self, config: AppConfig) -> str | None:
        if os.environ.get("XREVIEW_CLAUDE_OAUTH_TOKEN"):
            return os.environ["XREVIEW_CLAUDE_OAUTH_TOKEN"].strip()
        src = config.auth.claude_token_file
        if src is None:
            candidate = Path.home() / ".config" / "xreview" / "claude-oauth-token"
            if candidate.exists():
                src = candidate
        if src and src.exists():
            return src.read_text(encoding="utf-8").strip()
        return None

    def prepare_state(self, reviewer: ReviewerConfig, state_dir: Path, config: AppConfig) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(state_dir, 0o700)
        cfg_dir = state_dir / ".claude"
        if cfg_dir.is_symlink():
            raise PermissionError(f"Refusing symlinked .claude in reviewer state: {cfg_dir}")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        if self._read_oauth_token(config):
            return
        cred_dest = cfg_dir / ".credentials.json"
        if not cred_dest.exists() and not cred_dest.is_symlink():
            src = Path.home() / ".claude" / ".credentials.json"
            if not src.exists():
                raise FileNotFoundError(
                    "Claude Code auth not found. Generate a token with `claude setup-token` "
                    "and save it to auth.claude_token_file (mode 600)."
                )
            fd = os.open(str(cred_dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                os.write(fd, src.read_bytes())
            finally:
                os.close(fd)

    def run_turn(self, req: TurnRequest, config: AppConfig) -> TurnResult:
        self.prepare_state(req.reviewer, req.state_host_dir, config)
        use_sandbox = not req.allow_no_sandbox
        claude_bin = _find_claude_binary()

        if use_sandbox:
            exe_path = "/harness/claude"
            tree_path = f"/review/{req.run_id}/{req.reviewer.name}/tree"
            inputs_path = f"/review/{req.run_id}/{req.reviewer.name}/inputs"
            docs_path = f"/review/{req.run_id}/{req.reviewer.name}/docs" if req.docs_host_dir else None
            cfg_dir_env = "/home/reviewer/.claude"
            path_env = "/harness:/usr/bin:/bin"
            harness_mounts = ((claude_bin, "/harness/claude"),)
        else:
            exe_path = str(claude_bin)
            tree_path = str(req.tree_host_dir)
            inputs_path = str(req.inputs_host_dir)
            docs_path = str(req.docs_host_dir) if req.docs_host_dir else None
            cfg_dir_env = str(req.state_host_dir / ".claude")
            path_env = f"{claude_bin.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            harness_mounts = ()

        schema_str = req.schema_host_path.read_text(encoding="utf-8")

        cmd = [
            exe_path,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            req.reviewer.model,
            "--restricted",
            "--tools",
            "Read,Glob,Grep",
            "--strict-mcp-config",
            "--permission-prompts",
            "none",
            "--disable-slash-commands",
            "--add-dir",
            inputs_path,
            "--json-schema",
            schema_str,
        ]
        if docs_path:
            cmd.extend(["--add-dir", docs_path])
        if req.reviewer.effort:
            cmd.extend(["--effort", req.reviewer.effort])
        if req.session_id:
            cmd.extend(["--resume", req.session_id])

        env = {
            "CLAUDE_CONFIG_DIR": cfg_dir_env,
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "PATH": path_env,
        }
        # Pass OAuth token via secret_env (inherited pipe FD, NEVER in host bwrap argv! astra-18)
        secret_env: dict[str, str] = {}
        token = self._read_oauth_token(config)
        if token:
            secret_env["CLAUDE_CODE_OAUTH_TOKEN"] = token

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
            env=env,
            secret_env=secret_env,
            vendor_hosts=config.vendor_hosts_for(req.reviewer.vendor),
            extra_ro_paths=config.profile.extra_ro_paths,
            timeout_s=req.timeout_s,
            use_sandbox=use_sandbox,
        )

        session_id = req.session_id
        structured_output: dict[str, Any] | None = None
        last_structured_tool_input: dict[str, Any] | None = None
        result_text = ""
        usage: dict[str, Any] = {}
        stream_errors: list[str] = []
        is_error_result = False
        saw_result = False
        rate_limit_windows: dict[str, dict[str, Any]] = {}
        rate_limit_rejected = False
        rate_limit_resets_at: float | None = None

        def _on_line(line: str) -> None:
            nonlocal session_id, structured_output, last_structured_tool_input, result_text, usage
            nonlocal is_error_result, saw_result, rate_limit_rejected, rate_limit_resets_at
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                return
            if ev.get("session_id") and not session_id:
                session_id = str(ev["session_id"])
                if req.on_session_id:
                    req.on_session_id(session_id)

            ev_type = ev.get("type")
            if ev_type == "rate_limit_event":
                info = ev.get("rate_limit_info")
                if isinstance(info, dict):
                    top_status = str(info.get("status", "")).lower()
                    if top_status in ("rejected", "rate_limited", "exhausted"):
                        rate_limit_rejected = True
                    if isinstance(info.get("resetsAt"), (int, float)):
                        rate_limit_resets_at = float(info["resetsAt"])
                    uwin = info.get("unifiedWindows")
                    if isinstance(uwin, dict):
                        for wname, wbody in uwin.items():
                            if isinstance(wbody, dict) and isinstance(wbody.get("utilization"), (int, float)):
                                rate_limit_windows[str(wname)] = {
                                    "name": str(wname),
                                    "used_ratio": float(wbody["utilization"]),
                                    "resets_at": float(wbody["resetsAt"]) if isinstance(wbody.get("resetsAt"), (int, float)) else None,
                                    "status": str(wbody.get("status", "")),
                                }
            elif ev_type == "assistant":
                msg = ev.get("message") or {}
                for block in msg.get("content") or ():
                    if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                        inp = block.get("input")
                        if isinstance(inp, dict):
                            last_structured_tool_input = inp
            elif ev_type == "result":
                saw_result = True
                if ev.get("is_error"):
                    is_error_result = True
                if isinstance(ev.get("structured_output"), dict):
                    structured_output = ev["structured_output"]
                if isinstance(ev.get("result"), str):
                    result_text = ev["result"]
                if isinstance(ev.get("usage"), dict):
                    usage = dict(ev["usage"])
                if isinstance(ev.get("total_cost_usd"), (int, float)):
                    usage["total_cost_usd"] = ev["total_cost_usd"]
                if ev.get("is_error") and ev.get("result"):
                    stream_errors.append(str(ev["result"]))
            elif ev_type == "error":
                stream_errors.append(json.dumps(ev))

        sb_res = run_in_sandbox(
            spec,
            stdin_text=req.prompt,
            raw_log_path=req.raw_log_path,
            proxy_log_path=req.proxy_log_path,
            on_stdout_line=_on_line,
        )

        # Require successful exit AND non-error terminal result before accepting any reply (astra-37)
        succeeded = sb_res.returncode == 0 and not sb_res.timed_out and saw_result and not is_error_result
        reply = (structured_output or last_structured_tool_input or parse_json_from_text(result_text)) if succeeded else None

        # Only scan error diagnostics (never a successful review's text!) for quota exhaustion (astra-37)
        err_only = "\n".join(stream_errors + ([sb_res.stderr] if sb_res.stderr else []))
        diag_err = err_only
        if is_error_result and result_text:
            diag_err += ("\n" if diag_err else "") + result_text
        quota = not succeeded and (rate_limit_rejected or detect_quota_exhausted(diag_err))
        malformed = bool(succeeded and reply is None)
        resume_failed = bool(
            req.session_id and not succeeded and not quota and not sb_res.timed_out and detect_resume_failed(err_only)
        )

        quota_windows_list = list(rate_limit_windows.values())
        allowance_ratio = (
            max((float(w["used_ratio"]) for w in quota_windows_list), default=None)
            if quota_windows_list
            else None
        )
        resets_at = rate_limit_resets_at
        if resets_at is None and quota:
            resets_at = parse_reset_hint_from_error(diag_err)

        error = None
        if sb_res.timed_out:
            error = f"claude timed out after {req.timeout_s}s"
        elif quota:
            error = "claude quota/rate limit reached"
        elif not succeeded:
            error = f"claude failed (rc={sb_res.returncode}): {diag_err[:400].strip()}"
        elif malformed:
            error = "claude produced malformed non-JSON reply"

        return TurnResult(
            session_id=session_id,
            reply=reply,
            raw_text=result_text,
            usage=usage,
            allowance_used_ratio=allowance_ratio,
            quota_windows=quota_windows_list,
            quota_resets_at=resets_at,
            quota_exhausted=quota,
            resume_failed=resume_failed,
            malformed_output=malformed,
            timed_out=sb_res.timed_out,
            error=error,
            duration_s=sb_res.duration_s,
            returncode=sb_res.returncode,
        )
