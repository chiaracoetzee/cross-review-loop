"""Tests for pre-flight quota inspection, core->extra fallback, exhausted plan dropping, and quota cache."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from xreview.adapters.base import TurnResult
from xreview.cli import main
from xreview.config import AppConfig, AuthConfig, PolicyConfig, ReviewerConfig
from xreview.ledger import load_ledger, render_ledger_markdown
from xreview.policy import PolicyViolation
from xreview.protocol import ProtocolError, init_run, run_round
from xreview.quota import (
    PlanQuotaStatus,
    QuotaProbeError,
    QuotaWindow,
    apply_pre_flight_quota_policy,
    inspect_plan_quotas,
    load_quota_cache,
    parse_agy_usage_json,
    parse_reset_hint_from_error,
    probe_agy_plan_quota,
    probe_claude_plan_quota,
    read_codex_rollout_quota,
    record_plan_quota,
    refresh_plan_quotas,
)
from xreview.sandbox import SandboxRunResult
from xreview.snapshot import RepoSpec
from test_protocol import FakeAdapter, FakeClock, _make_git_repo, _reply


def _six_reviewers() -> tuple[ReviewerConfig, ...]:
    return (
        ReviewerConfig("astra", "codex", "gpt-6.2-astra", "gpt", "openai", effort="high", extra=False),
        ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google", extra=False),
        ReviewerConfig("fable", "claude", "claude-fable-5-2", "claude", "anthropic", effort="high", extra=False),
        ReviewerConfig("sol", "codex", "gpt-6.2-sol", "gpt", "openai", effort="xhigh", extra=True),
        ReviewerConfig("hydrogen", "agy", "gemini-4-hydrogen-xhigh", "gemini", "google", extra=True),
        ReviewerConfig("sonnet", "claude", "sonnet", "claude", "anthropic", effort="max", extra=True),
    )


class QuotaFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep every test hermetic from ambient host auth/Codex environment variables.
        clean_env = {k: v for k, v in os.environ.items() if k not in ("XREVIEW_CLAUDE_OAUTH_TOKEN", "CODEX_HOME")}
        patcher = mock.patch.dict(os.environ, clean_env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_low_quota_backs_out_of_heavyweight_core_and_promotes_extra(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            now = time.time()
            future = now + 3600
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=3, fallback_to_extras_at=0.75),
                state_dir=root / "state",
                implementor_family="claude",
            )
            # Claude seven_day at 78% (>= 75%), Codex & agy healthy
            record_plan_quota(
                cfg.state_dir,
                PlanQuotaStatus(
                    plan_id="claude",
                    harness="claude",
                    windows=(
                        QuotaWindow("five_hour", 0.10, future, "allowed"),
                        QuotaWindow("seven_day", 0.78, future + 86400, "allowed"),
                    ),
                    source="quota_cache",
                    updated_at=now,
                ),
            )
            clock = FakeClock()
            adapters = {
                h: FakeAdapter(h, lambda req: TurnResult(session_id=f"s-{req.reviewer.name}", reply=_reply()))
                for h in ("codex", "agy", "claude")
            }
            run_dir, ledger = init_run(
                cfg, "r-low", [RepoSpec("repo", repo)], "Low quota fallback",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            # Heavyweight fable dropped, cheaper sonnet promoted to extra=False
            self.assertNotIn("fable", ledger.reviewers)
            self.assertIn("sonnet", ledger.reviewers)
            self.assertFalse(ledger.reviewers["sonnet"].config.extra, "promoted extra must not be shed as extra")
            self.assertIn("astra", ledger.reviewers)
            self.assertIn("carbon", ledger.reviewers)
            self.assertEqual(len(ledger.quota_adjustments), 1)
            self.assertIn("fable", ledger.quota_adjustments[0])
            self.assertIn("sonnet", ledger.quota_adjustments[0])
            md = render_ledger_markdown(ledger)
            self.assertIn("Quota adjustments:", md)

    def test_exhausted_plan_auto_drops_to_two_family_floor_and_round_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            now = time.time()
            future = now + 3600
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=3, fallback_to_extras_at=0.75, auto_drop_exhausted=True),
                state_dir=root / "state",
                implementor_family="gemini",
            )
            # Codex 100% exhausted; Claude at 80% (low -> fallback to sonnet)
            record_plan_quota(
                cfg.state_dir,
                PlanQuotaStatus(
                    plan_id="codex",
                    harness="codex",
                    windows=(QuotaWindow("primary", 1.0, future, "exhausted"),),
                    exhausted=True,
                    exhausted_until=future,
                    source="codex_rollout",
                    updated_at=now,
                ),
            )
            record_plan_quota(
                cfg.state_dir,
                PlanQuotaStatus(
                    plan_id="claude",
                    harness="claude",
                    windows=(QuotaWindow("seven_day", 0.80, future, "allowed"),),
                    source="quota_cache",
                    updated_at=now,
                ),
            )
            clock = FakeClock()
            adapters = {
                h: FakeAdapter(h, lambda req: TurnResult(session_id=f"s-{req.reviewer.name}", reply=_reply()))
                for h in ("codex", "agy", "claude")
            }
            run_dir, ledger = init_run(
                cfg, "r-exh", [RepoSpec("repo", repo)], "Exhausted plan drop",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            # Codex completely removed, Claude fell back to sonnet, agy kept both
            self.assertEqual(sorted(ledger.reviewers.keys()), ["carbon", "hydrogen", "sonnet"])
            self.assertEqual(ledger.min_families_override, 2)
            self.assertEqual(len(ledger.quota_adjustments), 2)

            # Running a round MUST succeed using the degraded floor of 2 families (gemini + claude)
            out = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(out.status, "converged")

    def test_exhausted_plans_never_drop_below_hard_floor_of_two_families(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            now = time.time()
            future = now + 3600
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=3, auto_drop_exhausted=True),
                state_dir=root / "state",
                implementor_family="gemini",
            )
            # Both Codex and Claude exhausted -> only gemini (implementor!) left -> MUST refuse!
            for pid in ("codex", "claude"):
                record_plan_quota(
                    cfg.state_dir,
                    PlanQuotaStatus(
                        plan_id=pid,
                        harness=pid,
                        windows=(QuotaWindow("primary", 1.0, future, "exhausted"),),
                        exhausted=True,
                        exhausted_until=future,
                        source="quota_cache",
                        updated_at=now,
                    ),
                )
            adapters = {
                h: FakeAdapter(h, lambda req: TurnResult(session_id="s", reply=_reply()))
                for h in ("codex", "agy", "claude")
            }
            with self.assertRaises(PolicyViolation) as ctx:
                init_run(
                    cfg, "r-noquorum", [RepoSpec("repo", repo)], "Refuse <2 families",
                    no_sandbox=True, adapter_lookup=adapters.__getitem__,
                )
            self.assertIn("codex", str(ctx.exception))
            self.assertIn("claude", str(ctx.exception))

    def test_expired_window_ignored_and_explicit_reviewers_never_widened(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            past = now - 60
            future = now + 3600
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=2, fallback_to_extras_at=0.75),
                state_dir=root / "state",
            )
            # Expired codex exhaustion -> MUST be ignored (astra & sol both kept)
            expired_st = PlanQuotaStatus(
                plan_id="codex",
                harness="codex",
                windows=(QuotaWindow("primary", 1.0, past, "exhausted"),),
                exhausted=True,
                exhausted_until=past,
                source="codex_rollout",
                updated_at=past - 10,
            )
            # Active claude low quota, but candidate pool only has fable (explicit --reviewers) -> MUST NOT inject sonnet!
            low_claude = PlanQuotaStatus(
                plan_id="claude",
                harness="claude",
                windows=(QuotaWindow("seven_day", 0.85, future, "allowed"),),
                source="quota_cache",
                updated_at=now,
            )
            adj = apply_pre_flight_quota_policy(
                [r for r in cfg.reviewers if r.name in ("astra", "fable")],
                cfg,
                {"codex": expired_st, "claude": low_claude},
                now_ts=now,
            )
            self.assertEqual([r.name for r in adj.reviewers], ["astra", "fable"])
            self.assertTrue(any("no cheaper extra reviewer is available" in n for n in adj.notices))

    def test_codex_rollout_reader_handles_premium_null_after_codex_windows_and_turn_caches_quota(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            codex_home = root / "codex_home"
            sess_dir = codex_home / "sessions" / "2026" / "11" / "23"
            sess_dir.mkdir(parents=True)
            future = time.time() + 7200
            rollout = sess_dir / "rollout-2026-11-23-test.jsonl"
            rollout.write_text(
                "\n".join([
                    json.dumps({
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "rate_limits": {
                                "limit_id": "codex",
                                "primary": {"used_percent": 100.0, "resets_at": future},
                                "secondary": {"used_percent": 16.0, "resets_at": future + 86400},
                            },
                        },
                    }),
                    # Trailing premium event with null windows must NOT wipe codex windows!
                    json.dumps({
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "rate_limits": {
                                "limit_id": "premium",
                                "primary": None,
                                "secondary": None,
                            },
                        },
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            st = read_codex_rollout_quota(codex_home)
            assert st is not None
            self.assertTrue(st.is_exhausted(time.time()))
            self.assertAlmostEqual(st.effective_used_ratio(time.time()) or 0.0, 1.0)
            self.assertEqual(st.next_reset_at(time.time()), future)

            # Verify run_round persists TurnResult quota_windows into <state_dir>/quota_cache.json
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("carbon", "agy", "m1", "gemini", "google"),
                    ReviewerConfig("sonnet", "claude", "m2", "claude", "anthropic"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
                auth=AuthConfig(codex_auth_file=codex_home / "auth.json"),
            )
            clock = FakeClock()
            adapters = {
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
                "claude": FakeAdapter(
                    "claude",
                    lambda req: TurnResult(
                        session_id="s-s",
                        reply=_reply(),
                        allowance_used_ratio=0.77,
                        quota_windows=[{"name": "seven_day", "used_ratio": 0.77, "resets_at": future, "status": "allowed"}],
                        quota_resets_at=future,
                    ),
                ),
            }
            run_dir, _ = init_run(
                cfg, "r-cache", [RepoSpec("repo", repo)], "Cache test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            cached = load_quota_cache(cfg.state_dir)
            self.assertIn("claude", cached)
            self.assertAlmostEqual(cached["claude"].effective_used_ratio(time.time()) or 0.0, 0.77)

            # Verify parse_reset_hint_from_error parses Codex's human reset hint
            hint = parse_reset_hint_from_error("You've hit your usage limit... try again at 7:50 PM.")
            self.assertIsNotNone(hint)
            assert hint is not None
            self.assertGreater(hint, time.time() - 1)

            # Verify `xreview quota --json` runs cleanly with isolated codex_home
            cfg_file = root / "config.toml"
            cfg_file.write_text(
                f'state_dir = "{cfg.state_dir}"\n'
                'reviewers = [\n'
                '  { name = "astra", harness = "codex", model = "m1", family = "gpt" },\n'
                '  { name = "carbon", harness = "agy", model = "m2", family = "gemini" },\n'
                '  { name = "fable", harness = "claude", model = "m3", family = "claude" },\n'
                '  { name = "sonnet", harness = "claude", model = "m4", family = "claude", extra = true },\n'
                ']\n'
                '[policy]\n'
                'live_quota_probe = false\n'
                '[auth]\n'
                f'codex_auth_file = "{codex_home / "auth.json"}"\n',
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(cfg_file), "quota", "--cached", "--json"]), 0)
            # `--refresh` forces live check even when `live_quota_probe = false`
            refresh_seen: list[bool] = []

            def _rec_refresh(_c, *, force=False, **_kw):
                refresh_seen.append(force)
                return {}, []

            with mock.patch("xreview.cli.refresh_plan_quotas", side_effect=_rec_refresh), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(cfg_file), "quota", "--refresh", "--json"]), 0)
            self.assertEqual(refresh_seen, [True])

    def test_refresh_plan_quotas_runs_in_parallel_filters_plans_and_isolates_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            future = now + 7200
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=3),
                state_dir=root / "state",
            )
            barrier = threading.Barrier(2, timeout=5.0)
            agy_calls: list[tuple[str, bool]] = []
            claude_calls: list[tuple[str, bool]] = []

            def fake_agy_probe(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                agy_calls.append((pid, no_sb))
                barrier.wait()
                st = PlanQuotaStatus(
                    plan_id=pid,
                    harness="agy",
                    windows=(QuotaWindow("gemini-5h", 0.09, future, "allowed"),),
                    source="agy_usage",
                    updated_at=now,
                )
                record_plan_quota(_cfg.state_dir, st)
                return st

            def fake_claude_probe(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                claude_calls.append((pid, no_sb))
                barrier.wait()
                st = PlanQuotaStatus(
                    plan_id=pid,
                    harness="claude",
                    windows=(QuotaWindow("seven_day", 0.76, future, "allowed"),),
                    source="claude_probe",
                    updated_at=now,
                )
                record_plan_quota(_cfg.state_dir, st)
                return st

            # Concurrent execution proven via Barrier(2)
            statuses, warns = refresh_plan_quotas(
                cfg,
                include_host_codex_rollouts=False,
                now_ts=now,
                claude_probe_fn=fake_claude_probe,
                agy_probe_fn=fake_agy_probe,
            )
            self.assertEqual(agy_calls, [("agy", False)])
            self.assertEqual(claude_calls, [("claude", False)])
            self.assertEqual(warns, [])
            self.assertAlmostEqual(statuses["agy"].effective_used_ratio(now) or 0.0, 0.09)
            self.assertAlmostEqual(statuses["claude"].effective_used_ratio(now) or 0.0, 0.76)

            # plan_ids filter must NOT probe excluded plans (e.g. Claude excluded by --reviewers)
            agy_calls.clear()
            claude_calls.clear()

            def non_blocking_agy(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                agy_calls.append((pid, no_sb))
                return PlanQuotaStatus(
                    plan_id=pid,
                    harness="agy",
                    windows=(QuotaWindow("gemini-5h", 0.22, future, "allowed"),),
                    source="agy_usage",
                    updated_at=now + 60,
                )

            filtered_statuses, filtered_warns = refresh_plan_quotas(
                cfg,
                plan_ids={"codex", "agy"},
                include_host_codex_rollouts=False,
                now_ts=now + 60,
                claude_probe_fn=fake_claude_probe,
                agy_probe_fn=non_blocking_agy,
            )
            self.assertEqual(agy_calls, [("agy", False)])
            self.assertEqual(claude_calls, [], "excluded claude plan must not be probed")
            self.assertNotIn("claude", filtered_statuses)
            self.assertEqual(filtered_warns, [])

            # Failure isolation: Claude error emits warning with exception class while agy still updates
            def failing_claude(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                raise QuotaProbeError("network\noffline")

            iso_statuses, warns_err = refresh_plan_quotas(
                cfg,
                include_host_codex_rollouts=False,
                now_ts=now + 120,
                claude_probe_fn=failing_claude,
                agy_probe_fn=non_blocking_agy,
            )
            self.assertEqual(len(warns_err), 1)
            self.assertIn("QuotaProbeError", warns_err[0])
            self.assertIn("network offline", warns_err[0])
            self.assertNotIn("\n", warns_err[0])
            self.assertAlmostEqual(iso_statuses["agy"].effective_used_ratio(now + 120) or 0.0, 0.22)

    def test_parse_agy_usage_json_classification_edge_cases_and_plan_pool_isolation(self) -> None:
        now = time.time()
        payload = {
            "status": "SUCCESS",
            "command": {
                "name": "usage",
                "data": {
                    "groups": [
                        {
                            "name": "Gemini Models",
                            "buckets": [
                                {"id": "gemini-weekly", "window": "weekly", "remaining_fraction": 0.92, "reset_time": "2099-11-30T20:53:26Z"},
                                {"id": "gemini-5h", "window": "5h", "remaining_fraction": 0.20, "reset_time": "2099-11-24T06:53:26Z"},
                                # Bool remaining_fraction must be ignored
                                {"id": "gemini-bad", "window": "bad", "remaining_fraction": True, "reset_time": "2099-11-24T06:53:26Z"},
                            ],
                        },
                        {
                            "name": "Claude and GPT models",
                            "buckets": [
                                {"id": "3p-weekly", "window": "weekly", "remaining_fraction": 1.0, "reset_time": "2099-12-01T03:11:54Z"},
                                {"id": "3p-5h", "window": "5h", "remaining_fraction": 0.0, "reset_time": "2099-11-24T08:11:54Z"},
                            ],
                        },
                    ]
                },
            },
        }
        st_gem = parse_agy_usage_json(payload, families={"gemini"}, now_ts=now)
        self.assertIsNotNone(st_gem)
        if st_gem is not None:
            self.assertEqual(st_gem.source, "agy_usage")
            self.assertEqual([w.name for w in st_gem.windows], ["gemini-weekly", "gemini-5h"])
            self.assertAlmostEqual(st_gem.effective_used_ratio(now) or 0.0, 0.80, places=4)
            self.assertFalse(st_gem.is_exhausted(now))

        st_3p = parse_agy_usage_json(payload, families={"claude"}, plan_id="agy-3p", now_ts=now)
        self.assertIsNotNone(st_3p)
        if st_3p is not None:
            self.assertTrue(st_3p.is_exhausted(now))
            self.assertAlmostEqual(st_3p.effective_used_ratio(now) or 0.0, 1.0, places=4)

        # Single merged group: ID prefix must cleanly separate gemini- vs 3p- without leaking across families
        merged_group = {
            "status": "SUCCESS",
            "command": {
                "name": "usage",
                "data": {
                    "groups": [
                        {
                            "name": "All Models",
                            "buckets": [
                                {"id": "gemini-5h", "remaining_fraction": 0.9, "reset_time": "2099-11-24T06:53:26Z"},
                                {"id": "3p-5h", "remaining_fraction": 0.0, "reset_time": "2099-11-24T06:53:26Z"},
                            ],
                        }
                    ]
                },
            },
        }
        st_gem_only = parse_agy_usage_json(merged_group, families={"gemini"}, now_ts=now)
        self.assertIsNotNone(st_gem_only)
        if st_gem_only is not None:
            self.assertEqual([w.name for w in st_gem_only.windows], ["gemini-5h"])
            self.assertFalse(st_gem_only.is_exhausted(now))

        # Missing matching group MUST return None (never widen to unrelated 3P buckets!)
        only_3p = {
            "status": "SUCCESS",
            "command": {
                "name": "usage",
                "data": {
                    "groups": [
                        {
                            "name": "Claude and GPT models",
                            "buckets": [{"id": "3p-5h", "remaining_fraction": 0.0, "reset_time": "2099-11-24T06:53:26Z"}],
                        }
                    ]
                },
            },
        }
        self.assertIsNone(parse_agy_usage_json(only_3p, families={"gemini"}, now_ts=now))

        # Non-SUCCESS status must be rejected
        bad_status = dict(payload, status="ERROR")
        self.assertIsNone(parse_agy_usage_json(bad_status, families={"gemini"}, now_ts=now))

        # Separate default plan_id for agy Gemini ("agy") vs agy 3P ("agy-3p") keeps pools decoupled
        r_gem = ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google")
        r_3p = ReviewerConfig("agy-claude", "agy", "claude-opus-5-5-high", "claude", "google", extra=True)
        self.assertEqual(r_gem.plan_id, "agy")
        self.assertEqual(r_3p.plan_id, "agy-3p")
        adj = apply_pre_flight_quota_policy(
            [r_gem, r_3p],
            AppConfig(reviewers=(r_gem, r_3p), policy=PolicyConfig(auto_drop_exhausted=True)),
            {"agy": st_gem, "agy-3p": st_3p} if (st_gem and st_3p) else {},
            now_ts=now,
        )
        self.assertEqual([r.name for r in adj.reviewers], ["carbon"])

    def test_probe_agy_and_claude_plan_quota_command_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            future = now + 3600
            tok = root / "claude.tok"
            tok.write_text("sk-ant-test-secret\n", encoding="utf-8")
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(credits="never"),
                auth=AuthConfig(claude_token_file=tok),
                state_dir=root / "state",
            )

            captured_specs: list[object] = []
            valid_agy_payload = json.dumps({
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [{
                            "name": "Gemini Models",
                            "buckets": [{"id": "gemini-5h", "remaining_fraction": 0.85, "reset_time": "2099-11-24T06:53:26Z"}],
                        }]
                    },
                },
            })

            def fake_agy_sandbox(spec, **_kwargs):
                captured_specs.append(spec)
                return SandboxRunResult(returncode=0, stdout_lines=[valid_agy_payload], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[])

            with (
                mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=fake_agy_sandbox),
            ):
                st = probe_agy_plan_quota(cfg, "agy", no_sandbox=False, now_ts=now)
            self.assertAlmostEqual(st.effective_used_ratio(now) or 0.0, 0.15, places=4)
            agy_spec = captured_specs[-1]
            self.assertIn("-p", agy_spec.command)
            self.assertIn("/usage", agy_spec.command)
            self.assertIn("--settings", agy_spec.command)
            settings_arg = agy_spec.command[agy_spec.command.index("--settings") + 1]
            self.assertEqual(json.loads(settings_arg), {"useG1Credits": False})
            self.assertEqual(agy_spec.env["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/nonexistent")

            # agy probe failure surfaces returncode + stderr diagnostics via QuotaProbeError
            with (
                mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                mock.patch(
                    "xreview.sandbox.run_in_sandbox",
                    return_value=SandboxRunResult(returncode=1, stdout_lines=[], stderr="oauth token expired", timed_out=False, duration_s=0.1, proxy_entries=[]),
                ),
            ):
                with self.assertRaises(QuotaProbeError) as ctx:
                    probe_agy_plan_quota(cfg, "agy", no_sandbox=True, now_ts=now)
                self.assertIn("exited 1", str(ctx.exception))
                self.assertIn("oauth token expired", str(ctx.exception))

            # Non-matching group error includes observed groups & bucket IDs for diagnostics
            only_3p_json = json.dumps({
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [{
                            "name": "Claude and GPT models",
                            "buckets": [{"id": "3p-5h", "remaining_fraction": 0.5, "reset_time": "2099-11-24T06:53:26Z"}],
                        }]
                    },
                },
            })
            with (
                mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                mock.patch(
                    "xreview.sandbox.run_in_sandbox",
                    return_value=SandboxRunResult(returncode=0, stdout_lines=[only_3p_json], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[]),
                ),
            ):
                with self.assertRaises(QuotaProbeError) as ctx_nomatch:
                    probe_agy_plan_quota(cfg, "agy", no_sandbox=True, now_ts=now)
                self.assertIn("Claude and GPT models", str(ctx_nomatch.exception))
                self.assertIn("3p-5h", str(ctx_nomatch.exception))

            # Mixed agy Gemini + agy-3p config fetches `/usage` ONLY ONCE and populates both plans
            mixed_cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google"),
                    ReviewerConfig("agy-claude", "agy", "claude-opus-5-5-high", "claude", "google", extra=True),
                ),
                policy=PolicyConfig(credits="never"),
                state_dir=root / "mixed_state",
            )
            mixed_payload = json.dumps({
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [
                            {"name": "Gemini Models", "buckets": [{"id": "gemini-5h", "remaining_fraction": 0.90, "reset_time": "2099-11-24T06:53:26Z"}]},
                            {"name": "Claude and GPT models", "buckets": [{"id": "3p-5h", "remaining_fraction": 0.20, "reset_time": "2099-11-24T06:53:26Z"}]},
                        ]
                    },
                },
            })
            mixed_calls = 0

            def fake_mixed_agy(spec, **_kwargs):
                nonlocal mixed_calls
                mixed_calls += 1
                return SandboxRunResult(returncode=0, stdout_lines=[mixed_payload], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[])

            with (
                mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=fake_mixed_agy),
            ):
                m_statuses, m_warns = refresh_plan_quotas(mixed_cfg, include_host_codex_rollouts=False, now_ts=now)
            self.assertEqual(mixed_calls, 1, "mixed Gemini + 3P agy config must only invoke agy /usage once")
            self.assertEqual(m_warns, [])
            self.assertAlmostEqual(m_statuses["agy"].effective_used_ratio(now) or 0.0, 0.10, places=4)
            self.assertAlmostEqual(m_statuses["agy-3p"].effective_used_ratio(now) or 0.0, 0.80, places=4)

            # Claude probe: verify argv, secret_env token delivery, and aligned exhaustion semantics
            claude_specs: list[object] = []

            def make_claude_sandbox(lines: list[str], rc: int = 0, stderr: str = "", timed_out: bool = False):
                def _run(spec, **kwargs):
                    claude_specs.append(spec)
                    cb = kwargs.get("on_stdout_line")
                    if cb:
                        for ln in lines:
                            cb(ln)
                    return SandboxRunResult(returncode=rc, stdout_lines=[], stderr=stderr, timed_out=timed_out, duration_s=0.2, proxy_entries=[])
                return _run

            ok_lines = [
                json.dumps({
                    "type": "rate_limit_event",
                    "rate_limit_info": {
                        "status": "allowed",
                        "unifiedWindows": {
                            "five_hour": {"utilization": 0.12, "resetsAt": future, "status": "allowed"},
                            "seven_day": {"utilization": 0.71, "resetsAt": future + 86400, "status": "allowed"},
                        },
                    },
                }),
                json.dumps({"type": "result", "is_error": False, "result": "ok"}),
            ]
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=make_claude_sandbox(ok_lines)),
            ):
                st_c = probe_claude_plan_quota(cfg, "claude", no_sandbox=False, now_ts=now)
            self.assertAlmostEqual(st_c.effective_used_ratio(now) or 0.0, 0.71, places=4)
            self.assertFalse(st_c.is_exhausted(now))
            cspec = claude_specs[-1]
            self.assertIn("--tools", cspec.command)
            self.assertEqual(cspec.command[cspec.command.index("--tools") + 1], "")
            self.assertIn("--system-prompt", cspec.command)
            self.assertIn("--no-session-persistence", cspec.command)
            self.assertEqual(cspec.secret_env.get("CLAUDE_CODE_OAUTH_TOKEN"), "sk-ant-test-secret")
            self.assertNotIn("sk-ant-test-secret", " ".join(cspec.command))
            self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", cspec.env)

            # A rejected event on a SUCCEEDED turn with healthy windows must NOT falsely drop Claude
            soft_rej_lines = [
                json.dumps({
                    "type": "rate_limit_event",
                    "rate_limit_info": {
                        "status": "rejected",
                        "unifiedWindows": {
                            "five_hour": {"utilization": 0.25, "resetsAt": future, "status": "allowed"},
                        },
                    },
                }),
                json.dumps({"type": "result", "is_error": False, "result": "ok"}),
            ]
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=make_claude_sandbox(soft_rej_lines, rc=0)),
            ):
                st_soft = probe_claude_plan_quota(cfg, "claude", no_sandbox=True, now_ts=now)
            self.assertFalse(st_soft.is_exhausted(now))

            # Failed turn with rejected event or usage-limit text marks exhausted with reset timestamp
            rej_fail_lines = [
                json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": future}}),
                json.dumps({"type": "result", "is_error": True, "result": "You've hit your usage limit"}),
            ]
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=make_claude_sandbox(rej_fail_lines, rc=1)),
            ):
                st_exh = probe_claude_plan_quota(cfg, "claude", no_sandbox=True, now_ts=now)
            self.assertTrue(st_exh.is_exhausted(now))
            self.assertEqual(st_exh.exhausted_until, future)

            # Non-quota Claude probe failure raises QuotaProbeError with diagnostics
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=make_claude_sandbox([], rc=2, stderr="bad auth")),
            ):
                with self.assertRaises(QuotaProbeError) as ctx:
                    probe_claude_plan_quota(cfg, "claude", no_sandbox=True, now_ts=now)
                self.assertIn("bad auth", str(ctx.exception))

    def test_init_run_live_wiring_filters_plan_ids_checks_bwrap_first_and_separates_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=2, live_quota_probe=True),
                auth=AuthConfig(codex_auth_file=root / "empty_codex_home" / "auth.json"),
                state_dir=root / "state",
                implementor_family="gemini",
            )
            seen_calls: list[dict] = []

            def fake_refresh(c, *, plan_ids=None, no_sandbox=False, include_host_codex_rollouts=True, **_kw):
                seen_calls.append({"plan_ids": set(plan_ids or ()), "no_sandbox": no_sandbox})
                return {}, ["Live quota check for plan 'agy' (agy) failed (QuotaProbeError): timeout"]

            # 1) Missing bwrap MUST fail before invoking refresh_plan_quotas
            with (
                mock.patch("xreview.protocol.bwrap_available", return_value=False),
                mock.patch("xreview.protocol.refresh_plan_quotas", side_effect=fake_refresh),
            ):
                with self.assertRaises(ProtocolError):
                    init_run(cfg, "r-nobwrap", [RepoSpec("repo", repo)], "No bwrap", no_sandbox=False)
            self.assertEqual(seen_calls, [], "must not probe before verifying bwrap_available")

            # 2) Explicit --reviewers astra,carbon must pass ONLY {'codex', 'agy'} (never 'claude'),
            #    and probe warnings land in quota_warnings (not quota_adjustments!)
            fake_adapter = FakeAdapter("fake", lambda req: TurnResult(session_id="s", reply=_reply()))
            with (
                mock.patch("xreview.protocol.bwrap_available", return_value=True),
                mock.patch("xreview.protocol.refresh_plan_quotas", side_effect=fake_refresh),
                mock.patch("xreview.protocol.get_adapter", return_value=fake_adapter),
            ):
                # Re-bind default adapter_lookup check by importing protocol.get_adapter via mock target
                from xreview import protocol as proto_mod
                run_dir, ledger = init_run(
                    cfg,
                    "r-live-filter",
                    [RepoSpec("repo", repo)],
                    "Filter plans",
                    explicit_reviewers=["astra", "carbon"],
                    no_sandbox=True,
                    adapter_lookup=proto_mod.get_adapter,
                )
            self.assertEqual(len(seen_calls), 1)
            self.assertEqual(seen_calls[0]["plan_ids"], {"codex", "agy"})
            self.assertTrue(seen_calls[0]["no_sandbox"])
            self.assertEqual(ledger.quota_adjustments, [])
            self.assertEqual(len(ledger.quota_warnings), 1)
            self.assertIn("QuotaProbeError", ledger.quota_warnings[0])
            md = render_ledger_markdown(load_ledger(run_dir))
            self.assertNotIn("Quota adjustments:", md)
            self.assertIn("Quota probe warnings:", md)

            # 3) refresh_quota=False (--no-refresh-quota) skips live probes completely
            seen_calls.clear()
            with (
                mock.patch("xreview.protocol.bwrap_available", return_value=True),
                mock.patch("xreview.protocol.refresh_plan_quotas", side_effect=fake_refresh),
                mock.patch("xreview.protocol.get_adapter", return_value=fake_adapter),
            ):
                from xreview import protocol as proto_mod
                init_run(
                    cfg,
                    "r-norefresh",
                    [RepoSpec("repo", repo)],
                    "No refresh",
                    explicit_reviewers=["astra", "carbon"],
                    no_sandbox=True,
                    refresh_quota=False,
                    adapter_lookup=proto_mod.get_adapter,
                )
            self.assertEqual(seen_calls, [])

    def test_cli_quota_cached_live_init_flags_and_multi_plan_scheduling(self) -> None:
        from xreview.ledger import RunLedger

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg_file = root / "config.toml"
            cfg_file.write_text(
                f'state_dir = "{root / "state"}"\n'
                'reviewers = [\n'
                '  { name = "carbon", harness = "agy", model = "m1", family = "gemini" },\n'
                '  { name = "fable", harness = "claude", model = "m2", family = "claude" },\n'
                ']\n'
                '[auth]\n'
                f'codex_auth_file = "{root / "empty_codex" / "auth.json"}"\n',
                encoding="utf-8",
            )

            # 1) `quota --cached --json` with live probing enabled MUST NOT call refresh_plan_quotas
            def _forbid(*_a, **_kw):
                raise AssertionError("refresh_plan_quotas must not run with --cached")

            with (
                mock.patch("xreview.cli.refresh_plan_quotas", side_effect=_forbid),
                redirect_stdout(io.StringIO()) as out_buf,
                redirect_stderr(io.StringIO()) as err_buf,
            ):
                self.assertEqual(main(["--config", str(cfg_file), "quota", "--cached", "--json"]), 0)
            self.assertEqual(err_buf.getvalue(), "")
            self.assertIn("plans", json.loads(out_buf.getvalue()))

            # 2) Plain `quota --no-sandbox` calls refresh_plan_quotas(no_sandbox=True) and prints stderr progress;
            #    `quota --json` suppresses the stderr progress line.
            recorded: list[dict] = []

            def _fake_refresh(c, *, no_sandbox=False, now_ts=None, **_kw):
                recorded.append({"no_sandbox": no_sandbox})
                return (
                    {
                        "agy": PlanQuotaStatus("agy", "agy", source="agy_usage", updated_at=now_ts or 0.0),
                        "claude": PlanQuotaStatus("claude", "claude", source="claude_probe", updated_at=now_ts or 0.0),
                    },
                    [],
                )

            with (
                mock.patch("xreview.cli.refresh_plan_quotas", side_effect=_fake_refresh),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as err_plain,
            ):
                self.assertEqual(main(["--config", str(cfg_file), "quota", "--no-sandbox"]), 0)
            self.assertEqual(recorded, [{"no_sandbox": True}])
            self.assertIn("Checking plan quotas...", err_plain.getvalue())

            recorded.clear()
            with (
                mock.patch("xreview.cli.refresh_plan_quotas", side_effect=_fake_refresh),
                redirect_stdout(io.StringIO()) as out_json,
                redirect_stderr(io.StringIO()) as err_json,
            ):
                self.assertEqual(main(["--config", str(cfg_file), "quota", "--json"]), 0)
            self.assertEqual(recorded, [{"no_sandbox": False}])
            self.assertEqual(err_json.getvalue(), "")
            self.assertIn("agy", json.loads(out_json.getvalue())["plans"])

            # 3) `init --no-refresh-quota` maps to refresh_quota=False; omitted maps to None;
            #    quota_warnings appear in JSON output and stderr WARNING (quota) lines.
            init_calls: list[dict] = []

            def _fake_init(**kwargs):
                init_calls.append(kwargs)
                lg = RunLedger(
                    run_id=kwargs["run_id"],
                    created_at="2026-01-01T00:00:00Z",
                    artifact="t",
                    purpose="",
                    issue="",
                    in_scope="",
                    out_of_scope="",
                    repos=[],
                    sandbox_mode="bwrap",
                    quota_warnings=["Live quota check for plan 'claude' (claude) failed (QuotaProbeError): boom"],
                )
                return root / "state" / kwargs["run_id"], lg

            with (
                mock.patch("xreview.cli.init_run", side_effect=_fake_init),
                redirect_stdout(io.StringIO()) as out_i1,
                redirect_stderr(io.StringIO()) as err_i1,
            ):
                self.assertEqual(
                    main(["--config", str(cfg_file), "init", "--repo", str(repo), "--no-refresh-quota", "--json"]),
                    0,
                )
            self.assertIs(init_calls[-1]["refresh_quota"], False)
            self.assertNotIn("Checking plan quotas...", err_i1.getvalue())
            self.assertIn("WARNING (quota): Live quota check", err_i1.getvalue())
            parsed_i1 = json.loads(out_i1.getvalue())
            self.assertEqual(len(parsed_i1["quota_warnings"]), 1)

            with (
                mock.patch("xreview.cli.init_run", side_effect=_fake_init),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as err_i2,
            ):
                self.assertEqual(main(["--config", str(cfg_file), "init", "--repo", str(repo)]), 0)
            self.assertIsNone(init_calls[-1]["refresh_quota"])
            self.assertIn("Checking plan quotas...", err_i2.getvalue())
            self.assertIn("WARNING (quota): Live quota check", err_i2.getvalue())

            # 4) Multi-plan Claude fan-out vs serialized batch, and multi-plan injected agy sequential execution
            now = time.time()
            future = now + 3600
            two_claude_cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("f1", "claude", "m1", "claude", "anthropic", plan="claude-a"),
                    ReviewerConfig("f2", "claude", "m2", "claude", "anthropic", plan="claude-b"),
                ),
                policy=PolicyConfig(serialize_same_harness={"codex": True, "agy": True, "claude": False}),
                state_dir=root / "s_par",
            )
            c_barrier = threading.Barrier(2, timeout=5.0)
            seen_c_par: list[str] = []

            def _par_claude(_c, pid, _ns):
                seen_c_par.append(pid)
                c_barrier.wait()
                return PlanQuotaStatus(pid, "claude", windows=(QuotaWindow("5h", 0.1, future, "allowed"),), source="claude_probe", updated_at=now)

            st_par, w_par = refresh_plan_quotas(two_claude_cfg, include_host_codex_rollouts=False, now_ts=now, claude_probe_fn=_par_claude)
            self.assertEqual(w_par, [])
            self.assertEqual(set(seen_c_par), {"claude-a", "claude-b"})
            self.assertEqual(set(st_par.keys()), {"claude-a", "claude-b"})

            ser_claude_cfg = AppConfig(
                reviewers=two_claude_cfg.reviewers,
                policy=PolicyConfig(serialize_same_harness={"codex": True, "agy": True, "claude": True}),
                state_dir=root / "s_ser",
            )
            active = 0
            max_active = 0
            lock = threading.Lock()
            seen_c_ser: list[str] = []

            def _ser_claude(_c, pid, _ns):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    seen_c_ser.append(pid)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return PlanQuotaStatus(pid, "claude", windows=(QuotaWindow("5h", 0.1, future, "allowed"),), source="claude_probe", updated_at=now)

            _, w_ser = refresh_plan_quotas(ser_claude_cfg, include_host_codex_rollouts=False, now_ts=now, claude_probe_fn=_ser_claude)
            self.assertEqual(w_ser, [])
            self.assertEqual(set(seen_c_ser), {"claude-a", "claude-b"})
            self.assertEqual(max_active, 1)

            two_agy_cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("carbon", "agy", "m1", "gemini", "google"),
                    ReviewerConfig("agy-claude", "agy", "m2", "claude", "google", extra=True),
                ),
                state_dir=root / "s_agy",
            )
            agy_active = 0
            agy_max = 0
            seen_agy: list[str] = []

            def _ser_agy(_c, pid, _ns):
                nonlocal agy_active, agy_max
                with lock:
                    agy_active += 1
                    agy_max = max(agy_max, agy_active)
                    seen_agy.append(pid)
                time.sleep(0.02)
                with lock:
                    agy_active -= 1
                return PlanQuotaStatus(pid, "agy", windows=(QuotaWindow("5h", 0.1, future, "allowed"),), source="agy_usage", updated_at=now)

            _, w_agy = refresh_plan_quotas(two_agy_cfg, include_host_codex_rollouts=False, now_ts=now, agy_probe_fn=_ser_agy)
            self.assertEqual(w_agy, [])
            self.assertEqual(set(seen_agy), {"agy", "agy-3p"})
            self.assertEqual(agy_max, 1)

    def test_quota_hardening_regressions(self) -> None:
        from xreview.adapters.claude import parse_claude_rate_limit_info

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            future = now + 3600

            # 1) F-004, F-012 & F-038: top-level resetsAt falls back ONLY for matching rateLimitType,
            #    never applying seven_day's reset to five_hour; top-level utilization keyed by rateLimitType
            wins_match, rej, r_at = parse_claude_rate_limit_info({
                "status": "rejected",
                "resetsAt": future,
                "rateLimitType": "five_hour",
                "unifiedWindows": {
                    "five_hour": {"utilization": 1.0, "status": "rejected"},
                },
            })
            self.assertTrue(rej)
            self.assertEqual(r_at, future)
            self.assertEqual(wins_match["five_hour"]["resets_at"], future)

            wins_diff, _, _ = parse_claude_rate_limit_info({
                "status": "rejected",
                "resetsAt": future + 86400,
                "rateLimitType": "seven_day",
                "utilization": 0.65,
                "unifiedWindows": {
                    "five_hour": {"utilization": 1.0, "status": "rejected"},
                },
            })
            self.assertIsNone(wins_diff["five_hour"]["resets_at"], "seven_day reset must not bleed onto five_hour")
            self.assertAlmostEqual(wins_diff["seven_day"]["used_ratio"], 0.65)
            self.assertEqual(wins_diff["seven_day"]["resets_at"], future + 86400)

            # 2) F-011: stale cache entry with wrong harness is ignored & never selects wrong live probe
            cfg_h = AppConfig(
                reviewers=(ReviewerConfig("rev1", "agy", "gemini-4-carbon-high", "gemini", "google", plan="shared"),),
                state_dir=root / "s_harness",
            )
            record_plan_quota(
                cfg_h.state_dir,
                PlanQuotaStatus("shared", "claude", windows=(QuotaWindow("5h", 0.9, future, "allowed"),), source="claude_probe", updated_at=now),
            )
            inspected = inspect_plan_quotas(cfg_h, include_host_codex_rollouts=False, now_ts=now)
            self.assertEqual(inspected["shared"].harness, "agy")
            self.assertEqual(inspected["shared"].source, "unknown")
            agy_seen: list[str] = []
            refresh_plan_quotas(
                cfg_h,
                include_host_codex_rollouts=False,
                now_ts=now,
                claude_probe_fn=lambda *_a: self.fail("stale cache must not dispatch Claude probe"),
                agy_probe_fn=lambda _c, pid, _ns: (agy_seen.append(pid) or PlanQuotaStatus(pid, "agy", source="agy_usage", updated_at=now)),
            )
            self.assertEqual(agy_seen, ["shared"])

            # 3) F-010 & F-024: legacy unpooled agy cache with mixed/3p windows ignored for Gemini,
            #    and custom family 'google' with gemini-* model still maps to plan 'agy'
            custom_gem = ReviewerConfig("c1", "agy", "gemini-4-carbon-high", "google", "google")
            self.assertEqual(custom_gem.plan_id, "agy")
            cfg_leg = AppConfig(reviewers=(custom_gem,), state_dir=root / "s_legacy")
            record_plan_quota(
                cfg_leg.state_dir,
                PlanQuotaStatus("agy", "agy", windows=(QuotaWindow("3p-5h", 1.0, future, "exhausted"),), exhausted=True, exhausted_until=future, source="agy_usage", updated_at=now, pool=None),
            )
            self.assertEqual(inspect_plan_quotas(cfg_leg, include_host_codex_rollouts=False, now_ts=now)["agy"].source, "unknown")

            # 4) F-013: failed/timed-out Claude probe after partial telemetry MUST NOT overwrite usable cache
            cfg_c = AppConfig(
                reviewers=(
                    ReviewerConfig("f1", "claude", "m1", "claude", "anthropic", plan="claude-a"),
                    ReviewerConfig("f2", "claude", "m2", "claude", "anthropic", plan="claude-b"),
                ),
                policy=PolicyConfig(credits="never"),
                state_dir=root / "s_claude",
            )
            record_plan_quota(
                cfg_c.state_dir,
                PlanQuotaStatus("claude-a", "claude", windows=(QuotaWindow("seven_day", 0.55, future, "allowed"),), source="claude_probe", updated_at=now - 30),
            )

            def _timeout_after_partial(spec, **kwargs):
                cb = kwargs.get("on_stdout_line")
                if cb:
                    cb(json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.1, "resetsAt": future, "status": "allowed"}}}}))
                return SandboxRunResult(returncode=-1, stdout_lines=[], stderr="timed out", timed_out=True, duration_s=30.0, proxy_entries=[])

            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=_timeout_after_partial),
            ):
                with self.assertRaises(QuotaProbeError):
                    probe_claude_plan_quota(cfg_c, "claude-a", no_sandbox=True, now_ts=now)
            # Prior cached 55% window must remain intact!
            self.assertAlmostEqual(load_quota_cache(cfg_c.state_dir)["claude-a"].effective_used_ratio(now) or 0.0, 0.55)

            # 5) F-016/F-025 & F-009: real multi-plan Claude refresh runs ONE Haiku probe and fans out;
            #    cached-exhausted Claude plan is skipped under credits="never" unless force=True
            claude_calls = 0

            def _ok_claude(spec, **kwargs):
                nonlocal claude_calls
                claude_calls += 1
                cb = kwargs.get("on_stdout_line")
                if cb:
                    cb(json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour", "utilization": 0.33, "resetsAt": future}}))
                    cb(json.dumps({"type": "result", "is_error": False, "result": "ok"}))
                return SandboxRunResult(returncode=0, stdout_lines=[], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[])

            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=_ok_claude),
            ):
                st_multi, w_multi = refresh_plan_quotas(cfg_c, include_host_codex_rollouts=False, now_ts=now)
            self.assertEqual(claude_calls, 1, "multiple Claude plans sharing one credential must probe once")
            self.assertEqual(w_multi, [])
            self.assertAlmostEqual(st_multi["claude-a"].effective_used_ratio(now) or 0.0, 0.33)
            self.assertAlmostEqual(st_multi["claude-b"].effective_used_ratio(now) or 0.0, 0.33)

            # Cached exhausted Claude plan `claude-a` skips probing ITSELF under credits="never"
            # with an actionable warning, without contaminating distinct plan `claude-b` (F-045/F-048/F-050).
            record_plan_quota(
                cfg_c.state_dir,
                PlanQuotaStatus("claude-a", "claude", windows=(QuotaWindow("five_hour", 1.0, future, "exhausted"),), exhausted=True, exhausted_until=future, source="claude_probe", updated_at=now),
            )
            claude_calls = 0
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=_ok_claude),
            ):
                st_skip_a, w_skip_a = refresh_plan_quotas(cfg_c, plan_ids={"claude-a"}, force=False, include_host_codex_rollouts=False, now_ts=now + 5)
            self.assertEqual(claude_calls, 0, "cached-exhausted Claude plan must skip probe under credits=never")
            self.assertTrue(st_skip_a["claude-a"].is_exhausted(now + 5))
            self.assertTrue(any("Skipped live Claude check" in w and "xreview quota --refresh" in w for w in w_skip_a))

            claude_calls = 0
            with (
                mock.patch("xreview.adapters.claude._find_claude_binary", return_value=Path("/usr/bin/claude")),
                mock.patch("xreview.adapters.claude.ClaudeAdapter.prepare_state", return_value=None),
                mock.patch("xreview.sandbox.run_in_sandbox", side_effect=_ok_claude),
            ):
                st_probe_b, w_probe_b = refresh_plan_quotas(cfg_c, plan_ids={"claude-b"}, force=False, include_host_codex_rollouts=False, now_ts=now + 5)
            self.assertEqual(claude_calls, 1, "exhaustion on claude-a must not block probing distinct plan claude-b")
            self.assertEqual(w_probe_b, [])
            self.assertFalse(st_probe_b["claude-b"].is_exhausted(now + 5))
            self.assertAlmostEqual(st_probe_b["claude-b"].effective_used_ratio(now + 5) or 0.0, 0.33)

            # F-038/F-050: QuotaWindow with resets_at=None is bounded by DEFAULT_EXHAUSTED_COOLDOWN_S
            from xreview.quota import DEFAULT_EXHAUSTED_COOLDOWN_S
            no_reset_status = PlanQuotaStatus(
                "claude-no-reset",
                "claude",
                windows=(QuotaWindow("five_hour", 1.0, None, "exhausted"),),
                exhausted=True,
                exhausted_until=None,
                source="claude_probe",
                updated_at=now,
            )
            self.assertTrue(no_reset_status.is_exhausted(now + 60))
            self.assertFalse(no_reset_status.is_exhausted(now + DEFAULT_EXHAUSTED_COOLDOWN_S + 1))

            # 6) F-032/F-036/F-042: mid-round agy exhaustion (empty windows + pool='gemini') persists & reloads
            cfg_mid = AppConfig(
                reviewers=(ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google"),),
                state_dir=root / "s_mid_agy",
            )
            record_plan_quota(
                cfg_mid.state_dir,
                PlanQuotaStatus("agy", "agy", windows=(), exhausted=True, exhausted_until=future, source="quota_cache", updated_at=now, pool="gemini"),
            )
            self.assertTrue(inspect_plan_quotas(cfg_mid, include_host_codex_rollouts=False, now_ts=now)["agy"].is_exhausted(now))

            # 7) F-010: legacy v2 ledger with shed_plans=['agy'] preserves plan='agy' on 3P agy reviewer
            from xreview.ledger import RunLedger
            legacy_dict = {
                "ledger_version": 2,
                "run_id": "leg-1",
                "created_at": "2026-01-01T00:00:00Z",
                "shed_plans": ["agy"],
                "reviewers": {
                    "agy-claude": {
                        "config": {"name": "agy-claude", "harness": "agy", "model": "claude-opus", "family": "claude", "vendor": "google", "extra": True, "plan": None},
                        "status": "active",
                    }
                },
            }
            loaded_leg = RunLedger.from_dict(legacy_dict)
            self.assertEqual(loaded_leg.reviewers["agy-claude"].config.plan_id, "agy")
            self.assertIn(loaded_leg.reviewers["agy-claude"].config.plan_id, loaded_leg.shed_plans)

            # 8) F-037/F-044: persistent .qprobe dir resyncs when source token or credits policy changes
            from xreview.quota import _probe_state_dir
            tok1 = root / "agy_tok_1"
            tok1.write_text("tok-1", encoding="utf-8")
            tok2 = root / "agy_tok_2"
            tok2.write_text("tok-2-different-len", encoding="utf-8")
            cfg_p1 = AppConfig(auth=AuthConfig(agy_token_file=tok1), policy=PolicyConfig(credits="never"), state_dir=root / "s_probe")
            with _probe_state_dir(cfg_p1, "agy") as d1:
                (d1 / "sentinel.txt").write_text("v1", encoding="utf-8")
                (d1.parent / ".quota_probe_agy.agy_init").write_text("1", encoding="utf-8")
            # Same source + policy preserves persistent state
            with _probe_state_dir(cfg_p1, "agy") as d1_again:
                self.assertTrue((d1_again / "sentinel.txt").exists())
            # Switching token file clears stale state & init marker so prepare_state resyncs
            cfg_p2 = AppConfig(auth=AuthConfig(agy_token_file=tok2), policy=PolicyConfig(credits="never"), state_dir=root / "s_probe")
            with _probe_state_dir(cfg_p2, "agy") as d2:
                self.assertFalse((d2 / "sentinel.txt").exists())
                self.assertFalse((d2.parent / ".quota_probe_agy.agy_init").exists())

            # 9) F-005: agy timeout, empty stdout, invalid JSON, non-dict JSON all raise QuotaProbeError
            for sb_ret, expected in [
                (SandboxRunResult(returncode=-1, stdout_lines=[], stderr="", timed_out=True, duration_s=45.0, proxy_entries=[]), "timed out"),
                (SandboxRunResult(returncode=0, stdout_lines=[], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[]), "empty output"),
                (SandboxRunResult(returncode=0, stdout_lines=["{bad"], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[]), "invalid JSON"),
                (SandboxRunResult(returncode=0, stdout_lines=["[]"], stderr="", timed_out=False, duration_s=0.1, proxy_entries=[]), "non-object JSON"),
            ]:
                with (
                    mock.patch("xreview.adapters.agy._find_agy_binary", return_value=Path("/usr/bin/agy")),
                    mock.patch("xreview.adapters.agy.AgyAdapter.prepare_state", return_value=None),
                    mock.patch("xreview.sandbox.run_in_sandbox", return_value=sb_ret),
                ):
                    with self.assertRaises(QuotaProbeError) as ctx_a:
                        probe_agy_plan_quota(cfg_h, "shared", no_sandbox=True, now_ts=now)
                    self.assertIn(expected, str(ctx_a.exception))


if __name__ == "__main__":
    unittest.main()
