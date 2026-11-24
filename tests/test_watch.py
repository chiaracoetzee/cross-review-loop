"""Tests for `xreview watch` per-reviewer log streaming, sanitization, safety, and formatting."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from xreview.cli import main
from xreview.config import ConfigError, ReviewerConfig
from xreview.ledger import RunLedger, ReviewerRuntimeState, save_ledger
from xreview.watch import WatchState, format_raw_event, watch_reviewer

_RAW_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _make_run(
    state_dir: Path,
    run_id: str,
    *,
    current_round: int = 0,
    in_progress_round: int | None = 1,
    status: str = "initialized",
    pending_round_results: dict | None = None,
    pending_round_roster: list[str] | None = None,
) -> Path:
    run_dir = state_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    revs = {
        "sol": ReviewerRuntimeState(config=ReviewerConfig("sol", "codex", "gpt-6.2-sol", "gpt", "openai", effort="xhigh")),
        "sol_retry": ReviewerRuntimeState(config=ReviewerConfig("sol_retry", "codex", "gpt-6.2-sol", "gpt", "openai", effort="xhigh")),
        "hydrogen": ReviewerRuntimeState(config=ReviewerConfig("hydrogen", "agy", "gemini-4-hydrogen-xhigh", "gemini", "google")),
        "sonnet": ReviewerRuntimeState(config=ReviewerConfig("sonnet", "claude", "sonnet", "claude", "anthropic", effort="xhigh")),
    }
    ledger = RunLedger(
        run_id=run_id,
        created_at="2026-01-01T00:00:00Z",
        artifact="watch test",
        purpose="",
        issue="",
        in_scope="",
        out_of_scope="",
        repos=[],
        status=status,
        current_round=current_round,
        in_progress_round=in_progress_round,
        pending_round_results=pending_round_results or {},
        pending_round_roster=pending_round_roster if pending_round_roster is not None else [],
        reviewers=revs,
    )
    save_ledger(run_dir, ledger)
    return run_dir


class WatchTests(unittest.TestCase):
    def test_format_raw_event_harnesses_malformed_fields_and_terminal_sanitization(self) -> None:
        # 1) agy formatting + tool DONE + malformed duration/tokens + control char escaping
        st_agy = WatchState("hydrogen", "agy")
        lines_agy = format_raw_event(
            st_agy,
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "step_index": 3,
                    "state": "ACTIVE",
                    "step_type": "tool",
                    "tool_name": "read_file\x1b[31m",
                    "tool_info": {"parameters": {"Path": "/tree/a.py\x07\x1b]0;pwn\x07"}},
                },
            }),
        )
        self.assertEqual(len(lines_agy), 1)
        self.assertIsNone(_RAW_CTRL_RE.search(lines_agy[0]))
        self.assertIn("\\x1b[31m", lines_agy[0])
        self.assertIn("\\x07", lines_agy[0])

        self.assertEqual(
            format_raw_event(
                st_agy,
                json.dumps({
                    "event": "step_update",
                    "step_update": {
                        "step_index": 3,
                        "state": "DONE",
                        "step_type": "tool",
                        "tool_name": "read_file",
                        "duration_seconds": "unknown",
                    },
                }),
            ),
            ["[step 3] <- read_file done (?s)"],
        )
        self.assertEqual(
            format_raw_event(
                st_agy,
                json.dumps({
                    "event": "step_update",
                    "step_update": {
                        "step_index": 4,
                        "state": "DONE",
                        "step_type": "agent_response",
                        "duration_seconds": 12.34,
                        "usage": {"output_tokens": "bad", "thinking_tokens": True},
                    },
                }),
            ),
            ["[step 4] responded (12.3s)"],
        )
        self.assertEqual(
            format_raw_event(st_agy, json.dumps({"event": "result", "result": {"status": "SUCCESS"}})),
            ["[attempt done] hydrogen: status=SUCCESS"],
        )
        self.assertTrue(st_agy.seen_terminal)

        # 2) Claude formatting: no [thinking] 0 tokens spam, malformed token ignored, escape stripping
        st_cl = WatchState("sonnet", "claude")
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 0})),
            [],
        )
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": "NaN"})),
            [],
        )
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 1200})),
            ["[thinking] 1,200 tokens"],
        )
        # Reset back to 0 from >0 emits once, then stays silent on repeated 0s
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 0})),
            ["[thinking] 0 tokens"],
        )
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 0})),
            [],
        )
        cl_assist = format_raw_event(
            st_cl,
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hi \x1b[2J\x9b31m\ud800x\udc9b"},
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "/tree/x.py"}},
                        {"type": "tool_use", "name": "StructuredOutput", "input": {"findings": []}},
                    ]
                },
            }),
        )
        self.assertEqual(len(cl_assist), 3)
        self.assertIsNone(_RAW_CTRL_RE.search(cl_assist[0]))
        self.assertIn("\\x1b[2J\\x9b31m", cl_assist[0])
        self.assertIn("\\ud800x\\udc9b", cl_assist[0])
        # Must encode strictly as valid UTF-8 with no surrogate crash
        cl_assist[0].encode("utf-8")
        self.assertEqual(cl_assist[2], "[tool] StructuredOutput (submitting reply)")
        self.assertEqual(
            format_raw_event(st_cl, json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0.42})),
            ["[attempt done] sonnet: is_error=False cost=$0.4200"],
        )
        self.assertTrue(st_cl.seen_terminal)

        # 3) Codex formatting: thread.started, cmd, reasoning, agent_message, turn.completed, turn.failed, error
        st_cx = WatchState("sol", "codex")
        tid_line = format_raw_event(st_cx, json.dumps({"type": "thread.started", "thread_id": "th\x1b[32m-1"}))
        self.assertEqual(tid_line, ["[started] thread th\\x1b[32m-1"])
        self.assertEqual(
            format_raw_event(st_cx, json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "git diff"}})),
            ["[cmd] -> git diff"],
        )
        self.assertEqual(
            format_raw_event(st_cx, json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "git diff", "exit_code": 0}})),
            ["[cmd] <- exit=0: git diff"],
        )
        self.assertEqual(
            format_raw_event(st_cx, json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"findings":[]}'}})),
            ["[message] submitting JSON reply"],
        )
        self.assertEqual(
            format_raw_event(st_cx, json.dumps({"type": "error", "message": "boom\x07"})),
            ["[error] boom\\x07"],
        )
        self.assertEqual(
            format_raw_event(st_cx, json.dumps({"type": "turn.failed", "error": {"message": "quota\x1b[0m"}})),
            ["[attempt done] sol failed: quota\\x1b[0m"],
        )
        self.assertTrue(st_cx.seen_terminal)

    def test_watch_follows_retry_attempts_drains_on_round_end_and_bounds_startup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"

            # 1) Continuation retry (delayed > attempt_grace_polls) uses <reviewer>/retry.jsonl,
            #    does NOT pick up unrelated reviewer `sol_retry.jsonl`, and waits for checkpoint!
            run_dir = _make_run(state_dir, "w-retry", current_round=0, in_progress_round=1, pending_round_roster=["sol", "sol_retry"])
            raw_dir = run_dir / "r1" / "raw"
            raw_path = raw_dir / "sol.jsonl"
            other_rev_path = raw_dir / "sol_retry.jsonl"
            retry_path = raw_dir / "sol" / "retry.jsonl"
            step = {"n": 0}

            def fake_sleep_retry(_dt: float) -> None:
                step["n"] += 1
                if step["n"] == 1:
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    other_rev_path.write_text('{"type": "item.completed", "item": {"type": "reasoning", "text": "OTHER REVIEWER"}}\n', encoding="utf-8")
                    raw_path.write_text('{"type": "thread.started", "thread_id": "t-1"}\n', encoding="utf-8")
                elif step["n"] == 2:
                    with raw_path.open("a", encoding="utf-8") as fh:
                        fh.write('{"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}\n')
                elif step["n"] == 8:
                    # Delayed retry appearing AFTER more than attempt_grace_polls=2 polls
                    retry_path.parent.mkdir(parents=True, exist_ok=True)
                    retry_path.write_text(
                        '{"type": "item.completed", "item": {"type": "reasoning", "text": "fixing JSON"}}\n'
                        '{"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 8}}\n',
                        encoding="utf-8",
                    )
                elif step["n"] == 9:
                    _make_run(
                        state_dir,
                        "w-retry",
                        current_round=0,
                        in_progress_round=1,
                        pending_round_roster=["sol", "sol_retry"],
                        pending_round_results={"sol": {"status": "ok"}},
                    )

            buf = io.StringIO()
            rc = watch_reviewer(
                run_dir, "sol", follow=True, attempt_grace_polls=2, stream=buf, sleep_fn=fake_sleep_retry, clock_fn=lambda: 0.0
            )
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("[started] thread t-1", out)
            self.assertNotIn("OTHER REVIEWER", out)
            self.assertIn("Continuing with sol/retry.jsonl", out)
            self.assertIn("[reasoning] fixing JSON", out)
            self.assertIn("[DONE] sol round 1 finished.", out)

            # --no-follow replays BOTH initial and retry logs
            buf_replay = io.StringIO()
            self.assertEqual(watch_reviewer(run_dir, "sol", round_num=1, follow=False, stream=buf_replay), 0)
            self.assertIn("Continuation log: sol/retry.jsonl", buf_replay.getvalue())
            self.assertIn("[reasoning] fixing JSON", buf_replay.getvalue())

            # 1b) Queued reviewer in an active round MUST NOT time out at start_timeout_s=60s!
            run_q = _make_run(state_dir, "w-queued", current_round=0, in_progress_round=1, pending_round_roster=["hydrogen", "sol"])
            q_clock = [0.0]
            q_raw = run_q / "r1" / "raw" / "sol.jsonl"

            def q_sleep(dt: float) -> None:
                q_clock[0] += 30.0  # jumps past 60s start_timeout_s
                if q_clock[0] >= 120.0 and not q_raw.exists():
                    q_raw.parent.mkdir(parents=True, exist_ok=True)
                    q_raw.write_text('{"type": "turn.completed"}\n', encoding="utf-8")
                elif q_clock[0] >= 150.0:
                    _make_run(
                        state_dir,
                        "w-queued",
                        current_round=0,
                        in_progress_round=1,
                        pending_round_roster=["hydrogen", "sol"],
                        pending_round_results={"sol": {"status": "ok"}},
                    )

            buf_q = io.StringIO()
            self.assertEqual(
                watch_reviewer(
                    run_q, "sol", round_num=1, follow=True, start_timeout_s=60.0, stream=buf_q, sleep_fn=q_sleep, clock_fn=lambda: q_clock[0]
                ),
                0,
            )
            self.assertNotIn("[TIMEOUT]", buf_q.getvalue())
            self.assertIn("[DONE] sol round 1 finished.", buf_q.getvalue())

            # 1c) Checkpointed launch error without raw log reports error (not "was not invoked")
            run_lerr = _make_run(
                state_dir,
                "w-lerr",
                current_round=0,
                in_progress_round=1,
                pending_round_roster=["sol"],
                pending_round_results={"sol": {"error": "adapter launch error: missing bin", "returncode": -1}},
            )
            buf_lerr = io.StringIO()
            self.assertEqual(watch_reviewer(run_lerr, "sol", round_num=1, follow=True, stream=buf_lerr), 0)
            self.assertIn("failed (rc=-1): adapter launch error: missing bin", buf_lerr.getvalue())
            self.assertNotIn("was not invoked", buf_lerr.getvalue())

            # 1d) Explicit --timeout caps sleep duration so it never overshoots deadline
            run_cap = _make_run(state_dir, "w-cap", current_round=0, in_progress_round=1, pending_round_roster=["sol"])
            cap_clock = [0.0]
            sleeps: list[float] = []

            def cap_sleep(dt: float) -> None:
                sleeps.append(dt)
                cap_clock[0] += dt

            buf_cap = io.StringIO()
            self.assertEqual(
                watch_reviewer(
                    run_cap,
                    "sol",
                    round_num=1,
                    follow=True,
                    poll_interval=1.0,
                    timeout_s=1.3,
                    stream=buf_cap,
                    sleep_fn=cap_sleep,
                    clock_fn=lambda: cap_clock[0],
                ),
                0,
            )
            self.assertIn("[TIMEOUT]", buf_cap.getvalue())
            self.assertAlmostEqual(sum(sleeps), 1.3, places=4)

            # 2) Round finishes WITHOUT terminal event in reviewer log -> drains partial/tail and exits cleanly
            run_drain = _make_run(state_dir, "w-drain", current_round=0, in_progress_round=1)
            drain_raw = run_drain / "r1" / "raw"
            drain_raw.mkdir(parents=True)
            drain_log = drain_raw / "hydrogen.jsonl"
            drain_log.write_text('{"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "tool", "tool_name": "read", "duration_seconds": 0.2}}', encoding="utf-8")
            drain_step = {"n": 0}

            def fake_sleep_drain(_dt: float) -> None:
                drain_step["n"] += 1
                if drain_step["n"] == 1:
                    _make_run(state_dir, "w-drain", current_round=1, in_progress_round=None, status="needs_triage")

            buf_drain = io.StringIO()
            self.assertEqual(
                watch_reviewer(run_drain, "hydrogen", follow=True, stream=buf_drain, sleep_fn=fake_sleep_drain, clock_fn=lambda: 0.0),
                0,
            )
            self.assertIn("[step 1] <- read done (0.2s)", buf_drain.getvalue())
            self.assertIn("[DONE] hydrogen round 1 finished.", buf_drain.getvalue())

            # 3) Pre-dispatch failure: log never appears -> bounded start_timeout_s exits cleanly
            run_nodisp = _make_run(state_dir, "w-nodisp", current_round=0, in_progress_round=None)
            t_now = [0.0]

            def adv_sleep(dt: float) -> None:
                t_now[0] += dt

            buf_to = io.StringIO()
            self.assertEqual(
                watch_reviewer(
                    run_nodisp,
                    "sol",
                    round_num=1,
                    follow=True,
                    poll_interval=1.0,
                    start_timeout_s=2.0,
                    stream=buf_to,
                    sleep_fn=adv_sleep,
                    clock_fn=lambda: t_now[0],
                ),
                0,
            )
            self.assertIn("[TIMEOUT]", buf_to.getvalue())

            # 4) Converged run with no round_num exits immediately without polling
            run_conv = _make_run(state_dir, "w-conv", current_round=2, in_progress_round=None, status="converged")
            buf_conv = io.StringIO()
            self.assertEqual(
                watch_reviewer(run_conv, "sol", follow=True, stream=buf_conv, sleep_fn=lambda _: self.fail("should not poll")),
                0,
            )
            self.assertIn("is converged", buf_conv.getvalue())

            # 5) Historical round 1 during later in-progress round 2 finishes immediately without hanging
            run_hist = _make_run(state_dir, "w-hist", current_round=1, in_progress_round=2)
            buf_hist = io.StringIO()
            self.assertEqual(
                watch_reviewer(run_hist, "sol", round_num=1, follow=True, stream=buf_hist, sleep_fn=lambda _: self.fail("should not poll")),
                0,
            )
            self.assertIn("was not invoked", buf_hist.getvalue())

    def test_watch_rejects_parent_symlinks_dangling_symlinks_fifos_and_bad_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            run_dir = _make_run(state_dir, "w-sec", current_round=0, in_progress_round=1)

            # Unknown reviewer raises ConfigError (clean CLI formatting), invalid --round raises ConfigError
            with self.assertRaises(ConfigError):
                watch_reviewer(run_dir, "ghost", follow=False, stream=io.StringIO())
            with self.assertRaises(ConfigError):
                watch_reviewer(run_dir, "sol", round_num=0, follow=False, stream=io.StringIO())

            # Dangling leaf symlink refused
            raw_dir = run_dir / "r1" / "raw"
            raw_dir.mkdir(parents=True)
            os.symlink(root / "does_not_exist.jsonl", raw_dir / "sol.jsonl")
            with self.assertRaises(PermissionError):
                watch_reviewer(run_dir, "sol", round_num=1, follow=False, stream=io.StringIO())

            # FIFO refused without blocking
            fifo_path = raw_dir / "hydrogen.jsonl"
            os.mkfifo(fifo_path)
            with self.assertRaises(PermissionError):
                watch_reviewer(run_dir, "hydrogen", round_num=1, follow=False, stream=io.StringIO())

            # Symlinked parent directory (`r2/raw` -> external dir) refused
            ext_dir = root / "ext"
            ext_dir.mkdir()
            (ext_dir / "sonnet.jsonl").write_text('{"type": "result"}\n', encoding="utf-8")
            (run_dir / "r2").mkdir()
            os.symlink(ext_dir, run_dir / "r2" / "raw")
            with self.assertRaises(PermissionError):
                watch_reviewer(run_dir, "sonnet", round_num=2, follow=False, stream=io.StringIO())

            # CLI integration (`xreview watch --no-follow` and invalid `--poll-interval`)
            cfg_file = root / "config.toml"
            cfg_file.write_text(f'state_dir = "{state_dir}"\n', encoding="utf-8")
            (raw_dir / "sonnet.jsonl").write_text(
                '{"type": "result", "is_error": false, "total_cost_usd": 0.1}\n', encoding="utf-8"
            )
            cli_buf = io.StringIO()
            with redirect_stdout(cli_buf):
                self.assertEqual(
                    main(["--config", str(cfg_file), "watch", "w-sec", "--reviewer", "sonnet", "--round", "1", "--no-follow"]),
                    0,
                )
            self.assertIn("[attempt done] sonnet: is_error=False cost=$0.1000", cli_buf.getvalue())

    def test_watch_huge_ints_and_ledger_lone_surrogate(self) -> None:
        # F-002: arbitrarily large JSON integers must not raise OverflowError in _as_int / _as_float
        huge_int = 10**310
        st_cl = WatchState("sonnet", "claude")
        out_think = format_raw_event(
            st_cl,
            json.dumps({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": huge_int}),
        )
        self.assertEqual(len(out_think), 1)
        self.assertIn("[thinking]", out_think[0])

        out_res = format_raw_event(
            st_cl,
            json.dumps({"type": "result", "is_error": False, "total_cost_usd": huge_int}),
        )
        self.assertEqual(len(out_res), 1)
        self.assertIn("[attempt done] sonnet: is_error=False", out_res[0])
        self.assertNotIn("cost=", out_res[0])

        out_codex = format_raw_event(
            WatchState("sol", "codex"),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": huge_int, "output_tokens": 5}}),
        )
        self.assertEqual(len(out_codex), 1)
        self.assertIn("in=", out_codex[0])

        # F-049: lone UTF-16 surrogates in finding text must not crash LEDGER.md atomic write
        from xreview.ledger import Finding

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "surr-run"
            run_dir.mkdir()
            ledger = RunLedger.from_dict({"run_id": "surr-run", "created_at": "2026-01-01T00:00:00Z"})
            ledger.findings.append(Finding(
                ledger_id="F-001",
                title="bad \ud800 surrogate",
                file="a.py",
                line=1,
                evidence="ev \udfff here",
                failure_scenario="imp",
                suggested_fix="fix",
                round_opened=1,
            ))
            save_ledger(run_dir, ledger)
            md = (run_dir / "LEDGER.md").read_text(encoding="utf-8")
            self.assertIn("F-001", md)
            self.assertIn("\\ud800", md)


if __name__ == "__main__":
    unittest.main()
