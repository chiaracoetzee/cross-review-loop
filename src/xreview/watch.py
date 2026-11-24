"""Per-reviewer live log formatter and follower for `xreview watch`."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from xreview.config import ConfigError, validate_reviewer_name
from xreview.ledger import load_ledger

# Escape C0, DEL, C1 controls, and unpaired UTF-16 surrogates (which can crash UTF-8 or re-encode under surrogateescape).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\ud800-\udfff]")


def attempt_log_path(raw_dir: Path, reviewer: str, suffix: str = "") -> Path:
    """Return the canonical per-attempt raw log path under `raw_dir`.

    Primary attempt uses `<reviewer>.jsonl`; continuations (`_cold`, `_retry`) live under
    `<reviewer>/cold.jsonl` and `<reviewer>/retry.jsonl` so they can never collide with
    another reviewer whose name ends in `_cold` or `_retry`.
    """
    validate_reviewer_name(reviewer)
    clean_sfx = suffix.lstrip("_").strip()
    if not clean_sfx:
        return raw_dir / f"{reviewer}.jsonl"
    return raw_dir / reviewer / f"{clean_sfx}.jsonl"


def attempt_log_relpaths(reviewer: str) -> tuple[tuple[str, ...], ...]:
    """Return ordered relative path component tuples beneath `r<N>/raw` for a reviewer's attempts."""
    validate_reviewer_name(reviewer)
    return (
        (f"{reviewer}.jsonl",),
        (reviewer, "cold.jsonl"),
        (reviewer, "retry.jsonl"),
    )


def _escape_ctrl(match: re.Match[str]) -> str:
    code = ord(match.group(0))
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def _sanitize_terminal(text: str) -> str:
    """Collapse whitespace and escape C0/DEL/C1 control chars and unpaired surrogates."""
    compact = " ".join(text.split())
    return _CONTROL_CHAR_RE.sub(_escape_ctrl, compact)


def _brief(text: str, max_len: int = 160) -> str:
    safe = _sanitize_terminal(text)
    if len(safe) <= max_len:
        return safe
    return safe[: max_len - 3].rstrip() + "..."


def _as_int(val: Any) -> int | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if not isinstance(val, float):
        return None
    return int(val) if math.isfinite(val) else None


def _as_float(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        f = float(val)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


@dataclass
class WatchState:
    reviewer: str
    harness: str
    last_thinking_tokens: int = 0
    seen_terminal: bool = False


def format_raw_event(state: WatchState, raw_line: str) -> list[str]:
    """Parse one raw JSONL line for `state.harness` and return 0+ terminal-safe status lines."""
    text = raw_line.strip()
    if not text:
        return []
    try:
        ev = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(ev, dict):
        return []

    harness = state.harness.lower()
    if harness == "agy":
        return _format_agy_event(state, ev)
    if harness == "claude":
        return _format_claude_event(state, ev)
    if harness == "codex":
        return _format_codex_event(state, ev)
    return []


def _format_agy_event(state: WatchState, ev: dict[str, Any]) -> list[str]:
    event_name = ev.get("event")
    if event_name == "step_update":
        s = ev.get("step_update")
        if not isinstance(s, dict):
            return []
        st = s.get("state")
        stype = s.get("step_type")
        tname = _sanitize_terminal(str(s.get("tool_name") or ""))
        idx = _sanitize_terminal(str(s.get("step_index", "?")))
        if st == "ACTIVE" and stype == "tool":
            tinfo = s.get("tool_info")
            params = tinfo.get("parameters") if isinstance(tinfo, dict) else None
            if isinstance(params, dict):
                raw_target = (
                    params.get("Path")
                    or params.get("CommandLine")
                    or params.get("Query")
                    or params.get("SearchPath")
                    or json.dumps(params, sort_keys=True)
                )
            else:
                raw_target = ""
            return [f"[step {idx}] -> {tname}: {_brief(str(raw_target), 140)}"]
        if st == "DONE":
            dur = _as_float(s.get("duration_seconds"))
            dur_str = f"{dur:.1f}s" if dur is not None else "?s"
            if stype == "agent_response":
                u = s.get("usage") if isinstance(s.get("usage"), dict) else {}
                out_tok = _as_int(u.get("output_tokens"))
                think_tok = _as_int(u.get("thinking_tokens"))
                metrics = [dur_str]
                if out_tok is not None:
                    metrics.append(f"out={out_tok}")
                if think_tok is not None:
                    metrics.append(f"think={think_tok}")
                return [f"[step {idx}] responded ({', '.join(metrics)})"]
            if stype == "tool":
                return [f"[step {idx}] <- {tname} done ({dur_str})"]
    elif event_name == "result":
        r = ev.get("result") if isinstance(ev.get("result"), dict) else {}
        state.seen_terminal = True
        status_s = _sanitize_terminal(str(r.get("status", "unknown")))
        return [f"[attempt done] {_sanitize_terminal(state.reviewer)}: status={status_s}"]
    return []


def _format_claude_event(state: WatchState, ev: dict[str, Any]) -> list[str]:
    ev_type = ev.get("type")
    if ev_type == "system" and ev.get("subtype") == "thinking_tokens":
        tok = _as_int(ev.get("estimated_tokens"))
        if tok is None or tok < 0:
            return []
        if tok == 0:
            if state.last_thinking_tokens > 0:
                state.last_thinking_tokens = 0
                return ["[thinking] 0 tokens"]
            return []
        if tok < state.last_thinking_tokens or (tok - state.last_thinking_tokens) >= 1000:
            state.last_thinking_tokens = tok
            return [f"[thinking] {tok:,} tokens"]
        return []
    if ev_type == "assistant":
        msg = ev.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            return []
        lines: list[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "tool_use":
                name = _sanitize_terminal(str(b.get("name") or "tool"))
                if name == "StructuredOutput":
                    lines.append("[tool] StructuredOutput (submitting reply)")
                else:
                    inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                    target = inp.get("file_path") or inp.get("pattern") or inp.get("command") or inp.get("path") or inp
                    lines.append(f"[tool] {name}: {_brief(str(target), 140)}")
            elif btype == "text":
                txt = str(b.get("text") or "").strip()
                if txt:
                    lines.append(f"[text] {_brief(txt, 180)}")
        return lines
    if ev_type == "rate_limit_event":
        info = ev.get("rate_limit_info") if isinstance(ev.get("rate_limit_info"), dict) else {}
        status = _sanitize_terminal(str(info.get("status") or "").lower())
        if status in ("rejected", "rate_limited", "exhausted"):
            return [f"[quota] Claude rate_limit_event: status={status}"]
        return []
    if ev_type == "result":
        state.seen_terminal = True
        cost = _as_float(ev.get("total_cost_usd"))
        cost_part = f" cost=${cost:.4f}" if cost is not None else ""
        return [
            f"[attempt done] {_sanitize_terminal(state.reviewer)}: "
            f"is_error={bool(ev.get('is_error'))}{cost_part}"
        ]
    if ev_type == "error":
        return [f"[error] {_brief(json.dumps(ev), 180)}"]
    return []


def _format_codex_event(state: WatchState, ev: dict[str, Any]) -> list[str]:
    ev_type = ev.get("type")
    if ev_type == "thread.started":
        tid = _sanitize_terminal(str(ev.get("thread_id") or ""))
        return [f"[started] thread {tid}"] if tid else []
    if ev_type == "item.started":
        item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        if item.get("type") == "command_execution":
            cmd = str(item.get("command") or "")
            return [f"[cmd] -> {_brief(cmd, 140)}"]
        return []
    if ev_type == "item.completed":
        item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        itype = item.get("type")
        if itype == "command_execution":
            cmd = str(item.get("command") or "")
            rc = _sanitize_terminal(str(item.get("exit_code")))
            return [f"[cmd] <- exit={rc}: {_brief(cmd, 120)}"]
        if itype == "reasoning":
            txt = str(item.get("text") or "").strip()
            return [f"[reasoning] {_brief(txt, 180)}"] if txt else []
        if itype == "agent_message":
            txt = str(item.get("text") or "").strip()
            if not txt:
                return []
            if txt.startswith("{"):
                return ["[message] submitting JSON reply"]
            return [f"[message] {_brief(txt, 180)}"]
        if itype == "error":
            return [f"[error] {_brief(str(item.get('message') or item), 180)}"]
        return []
    if ev_type == "turn.completed":
        state.seen_terminal = True
        u = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
        in_tok = _as_int(u.get("input_tokens"))
        out_tok = _as_int(u.get("output_tokens"))
        parts: list[str] = []
        if in_tok is not None:
            parts.append(f"in={in_tok}")
        if out_tok is not None:
            parts.append(f"out={out_tok}")
        usage_part = f" ({', '.join(parts)})" if parts else ""
        return [f"[attempt done] {_sanitize_terminal(state.reviewer)}{usage_part}"]
    if ev_type == "turn.failed":
        state.seen_terminal = True
        err = ev.get("error")
        msg = (err.get("message") if isinstance(err, dict) else None) or ev.get("message") or "turn failed"
        return [f"[attempt done] {_sanitize_terminal(state.reviewer)} failed: {_brief(str(msg), 180)}"]
    if ev_type == "error":
        return [f"[error] {_brief(str(ev.get('message') or ev), 180)}"]
    return []


def _open_safe_raw_log(run_dir: Path, target_round: int, rel_components: Sequence[str]) -> TextIO | None:
    """Safely open `<run_dir>/r<N>/raw/<rel_components...>` rejecting symlinks and FIFOs."""
    if not rel_components:
        return None
    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

    try:
        cur_fd = os.open(str(run_dir), flags_dir)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise PermissionError(f"Refusing unsafe run directory {run_dir}: {exc}") from exc

    dir_parts = (f"r{target_round}", "raw", *rel_components[:-1])
    leaf_name = rel_components[-1]
    try:
        for comp in dir_parts:
            try:
                next_fd = os.open(comp, flags_dir, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    return None
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise PermissionError(f"Refusing symlinked or non-directory component {comp!r} under {run_dir}") from exc
                raise
            os.close(cur_fd)
            cur_fd = next_fd

        try:
            leaf_fd = os.open(leaf_name, flags_file, dir_fd=cur_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return None
            if exc.errno == errno.ELOOP:
                raise PermissionError(f"Refusing symlinked raw log {leaf_name!r} under {run_dir}") from exc
            raise
    finally:
        os.close(cur_fd)

    try:
        st = os.fstat(leaf_fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(leaf_fd)
            raise PermissionError(f"Refusing non-regular raw log {leaf_name!r} under {run_dir}")
        return os.fdopen(leaf_fd, "r", encoding="utf-8", errors="replace")
    except BaseException:
        try:
            os.close(leaf_fd)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class _LedgerProgress:
    round_completed: bool
    reviewer_checkpointed: bool
    round_active_with_reviewer: bool
    checkpoint_entry: dict[str, Any] | None


def _check_ledger_progress(
    run_dir: Path,
    target_round: int,
    reviewer: str,
    warned_error: list[bool],
    out: TextIO,
) -> _LedgerProgress:
    try:
        ledger = load_ledger(run_dir)
    except Exception as exc:
        if not warned_error[0]:
            warned_error[0] = True
            print(f"[warn] Could not read ledger.json ({_brief(str(exc), 120)})", file=out, flush=True)
        return _LedgerProgress(False, False, False, None)
    round_completed = ledger.current_round >= target_round or (
        str(target_round) in ledger.round_snapshots and ledger.in_progress_round != target_round
    )
    in_prog = ledger.in_progress_round == target_round
    cp = ledger.pending_round_results.get(reviewer) if in_prog else None
    reviewer_checkpointed = in_prog and reviewer in ledger.pending_round_results
    in_roster = (not ledger.pending_round_roster) or (reviewer in ledger.pending_round_roster)
    round_active_with_reviewer = in_prog and in_roster and not reviewer_checkpointed
    return _LedgerProgress(
        round_completed=round_completed,
        reviewer_checkpointed=reviewer_checkpointed,
        round_active_with_reviewer=round_active_with_reviewer,
        checkpoint_entry=cp if isinstance(cp, dict) else None,
    )


def _format_checkpoint_summary(reviewer: str, target_round: int, cp: dict[str, Any] | None) -> str:
    safe_rev = _sanitize_terminal(reviewer)
    if not isinstance(cp, dict):
        return f"[DONE] {safe_rev} round {target_round} finished."
    err = cp.get("error")
    rc = cp.get("returncode")
    if err:
        rc_part = f" (rc={_sanitize_terminal(str(rc))})" if rc is not None else ""
        return f"[DONE] {safe_rev} round {target_round} failed{rc_part}: {_brief(str(err), 180)}"
    if cp.get("quota_exhausted"):
        return f"[DONE] {safe_rev} round {target_round} stopped: quota exhausted."
    if cp.get("timed_out"):
        return f"[DONE] {safe_rev} round {target_round} timed out."
    return f"[DONE] {safe_rev} round {target_round} finished."


def _drain_remaining_attempts(
    run_dir: Path,
    target_round: int,
    attempt_relpaths: Sequence[tuple[str, ...]],
    start_idx: int,
    state: WatchState,
    out: TextIO,
) -> None:
    for idx in range(start_idx, len(attempt_relpaths)):
        rel = attempt_relpaths[idx]
        fh = _open_safe_raw_log(run_dir, target_round, rel)
        if fh is None:
            continue
        disp = "/".join(rel)
        print(f"--- Continuing with {_sanitize_terminal(disp)} ---", file=out, flush=True)
        with fh:
            for line in fh:
                for formatted in format_raw_event(state, line):
                    print(formatted, file=out, flush=True)


def _sleep_capped(
    poll_interval: float,
    deadlines: Sequence[float | None],
    sleep_fn: Callable[[float], None],
    clock_fn: Callable[[], float],
) -> None:
    now = clock_fn()
    active = [d - now for d in deadlines if d is not None]
    if not active:
        sleep_fn(poll_interval)
        return
    rem = min(active)
    if rem <= 0:
        return
    sleep_fn(min(poll_interval, rem))


def watch_reviewer(
    run_dir: Path,
    reviewer: str,
    *,
    round_num: int | None = None,
    follow: bool = True,
    poll_interval: float = 0.5,
    start_timeout_s: float = 60.0,
    timeout_s: float | None = None,
    attempt_grace_polls: int = 4,
    stream: TextIO | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> int:
    """Format and stream one reviewer's raw JSONL logs (including cold/retry) until completion."""
    validate_reviewer_name(reviewer)
    ledger = load_ledger(run_dir)
    if reviewer not in ledger.reviewers:
        known = ", ".join(sorted(ledger.reviewers.keys())) or "(none)"
        raise ConfigError(f"Reviewer {reviewer!r} not in run {ledger.run_id}. Active reviewers: {known}")

    out = stream if stream is not None else sys.stdout
    if round_num is not None:
        if round_num < 1:
            raise ConfigError("--round must be >= 1")
        target_round = round_num
    elif ledger.in_progress_round is not None:
        target_round = ledger.in_progress_round
    elif not follow:
        target_round = max(1, ledger.current_round)
    else:
        if ledger.status in ("converged", "paused"):
            print(
                f"Run {_sanitize_terminal(ledger.run_id)} is {_sanitize_terminal(ledger.status)}; no new round in progress.",
                file=out,
                flush=True,
            )
            return 0
        target_round = ledger.current_round + 1

    harness = ledger.reviewers[reviewer].config.harness
    attempt_relpaths = attempt_log_relpaths(reviewer)
    raw_dir_display = run_dir / f"r{target_round}" / "raw"
    primary_display = _sanitize_terminal(str(raw_dir_display / "/".join(attempt_relpaths[0])))
    state = WatchState(reviewer=reviewer, harness=harness)
    warned_error = [False]
    started_at = clock_fn()
    overall_deadline = (started_at + timeout_s) if timeout_s is not None else None
    pre_dispatch_deadline = (started_at + start_timeout_s) if start_timeout_s is not None else None

    print(
        f"Watching {_sanitize_terminal(reviewer)} ({_sanitize_terminal(harness)}) "
        f"round {target_round}: {primary_display}",
        file=out,
        flush=True,
    )

    if not follow:
        found_any = False
        for idx, rel in enumerate(attempt_relpaths):
            fh = _open_safe_raw_log(run_dir, target_round, rel)
            if fh is None:
                continue
            found_any = True
            if idx > 0:
                print(f"--- Continuation log: {_sanitize_terminal('/'.join(rel))} ---", file=out, flush=True)
            with fh:
                for line in fh:
                    for formatted in format_raw_event(state, line):
                        print(formatted, file=out, flush=True)
        if not found_any:
            print(f"No raw log found yet at {primary_display}", file=out, flush=True)
        return 0

    # Follow mode: wait for initial log file; start_timeout_s only applies while round is NOT active with this reviewer.
    attempt_idx = 0
    fh = _open_safe_raw_log(run_dir, target_round, attempt_relpaths[attempt_idx])
    while fh is None:
        prog = _check_ledger_progress(run_dir, target_round, reviewer, warned_error, out)
        if prog.reviewer_checkpointed:
            # Reviewer finished/failed before creating a raw log (e.g. launch error) — show checkpoint outcome!
            print(_format_checkpoint_summary(reviewer, target_round, prog.checkpoint_entry), file=out, flush=True)
            return 0
        if prog.round_completed:
            print(
                f"[DONE] Round {target_round} finished ({_sanitize_terminal(reviewer)} was not invoked).",
                file=out,
                flush=True,
            )
            return 0
        now = clock_fn()
        if overall_deadline is not None and now >= overall_deadline:
            print(f"[TIMEOUT] Watch timed out after {timeout_s:.0f}s.", file=out, flush=True)
            return 0
        if not prog.round_active_with_reviewer and pre_dispatch_deadline is not None and now >= pre_dispatch_deadline:
            print(
                f"[TIMEOUT] Raw log did not appear within {start_timeout_s:.0f}s "
                f"(round {target_round} may have exited before dispatch).",
                file=out,
                flush=True,
            )
            return 0
        deadlines: list[float | None] = [overall_deadline]
        if not prog.round_active_with_reviewer:
            deadlines.append(pre_dispatch_deadline)
        _sleep_capped(poll_interval, deadlines, sleep_fn, clock_fn)
        fh = _open_safe_raw_log(run_dir, target_round, attempt_relpaths[attempt_idx])

    pending = ""
    grace_waited = 0
    try:
        while True:
            if overall_deadline is not None and clock_fn() >= overall_deadline:
                print(f"[TIMEOUT] Watch timed out after {timeout_s:.0f}s.", file=out, flush=True)
                return 0

            chunk = fh.readline()
            if chunk:
                if chunk.endswith("\n"):
                    full_line = pending + chunk
                    pending = ""
                    for formatted in format_raw_event(state, full_line):
                        print(formatted, file=out, flush=True)
                else:
                    pending += chunk
                continue

            # Check if a continuation attempt (cold/retry) has appeared.
            switched = False
            for next_idx in range(attempt_idx + 1, len(attempt_relpaths)):
                rel = attempt_relpaths[next_idx]
                next_fh = _open_safe_raw_log(run_dir, target_round, rel)
                if next_fh is not None:
                    if pending:
                        for formatted in format_raw_event(state, pending):
                            print(formatted, file=out, flush=True)
                        pending = ""
                    fh.close()
                    fh = next_fh
                    attempt_idx = next_idx
                    state.seen_terminal = False
                    state.last_thinking_tokens = 0
                    grace_waited = 0
                    print(f"--- Continuing with {_sanitize_terminal('/'.join(rel))} ---", file=out, flush=True)
                    switched = True
                    break
            if switched:
                continue

            prog = _check_ledger_progress(run_dir, target_round, reviewer, warned_error, out)
            if prog.round_completed or prog.reviewer_checkpointed:
                tail = fh.read()
                if tail or pending:
                    for ln in (pending + tail).splitlines():
                        for formatted in format_raw_event(state, ln):
                            print(formatted, file=out, flush=True)
                    pending = ""
                # Rescan and drain any continuation attempts written between our last check and checkpoint.
                _drain_remaining_attempts(run_dir, target_round, attempt_relpaths, attempt_idx + 1, state, out)
                print(_format_checkpoint_summary(reviewer, target_round, prog.checkpoint_entry), file=out, flush=True)
                return 0

            # While the round is actively running this reviewer, wait for the authoritative ledger checkpoint
            # (or explicit --timeout) so delayed retries are never missed. Only use grace polls if no active round.
            if state.seen_terminal and not prog.round_active_with_reviewer:
                if grace_waited >= attempt_grace_polls:
                    print(f"[DONE] {_sanitize_terminal(reviewer)} round {target_round} finished.", file=out, flush=True)
                    return 0
                grace_waited += 1

            if overall_deadline is not None and clock_fn() >= overall_deadline:
                print(f"[TIMEOUT] Watch timed out after {timeout_s:.0f}s.", file=out, flush=True)
                return 0

            _sleep_capped(poll_interval, [overall_deadline], sleep_fn, clock_fn)
    finally:
        fh.close()
