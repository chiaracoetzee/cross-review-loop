"""Tests for pre-flight quota inspection, core->extra fallback, exhausted plan dropping, and quota cache."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from xreview.adapters.base import TurnResult
from xreview.cli import main
from xreview.config import AppConfig, AuthConfig, PolicyConfig, ReviewerConfig
from xreview.ledger import load_ledger, render_ledger_markdown
from xreview.policy import PolicyViolation
from xreview.protocol import init_run, run_round
from xreview.quota import (
    PlanQuotaStatus,
    QuotaWindow,
    apply_pre_flight_quota_policy,
    inspect_plan_quotas,
    load_quota_cache,
    parse_reset_hint_from_error,
    read_codex_rollout_quota,
    record_plan_quota,
    refresh_plan_quotas,
)
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
                '[auth]\n'
                f'codex_auth_file = "{codex_home / "auth.json"}"\n',
                encoding="utf-8",
            )
            self.assertEqual(main(["--config", str(cfg_file), "quota", "--json"]), 0)

    def test_refresh_plan_quotas_probes_only_missing_or_stale_claude_plans(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            future = now + 7200
            cfg = AppConfig(
                reviewers=_six_reviewers(),
                policy=PolicyConfig(min_families=3, quota_cache_max_age_seconds=1800),
                state_dir=root / "state",
            )
            calls: list[tuple[str, bool]] = []

            def fake_probe(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                calls.append((pid, no_sb))
                st = PlanQuotaStatus(
                    plan_id=pid,
                    harness="claude",
                    windows=(QuotaWindow("seven_day", 0.76, future, "allowed"),),
                    source="claude_probe",
                    updated_at=now,
                )
                record_plan_quota(_cfg.state_dir, st)
                return st

            # Cache empty -> probes claude once
            statuses, warns = refresh_plan_quotas(
                cfg, force=False, include_host_codex_rollouts=False, now_ts=now, claude_probe_fn=fake_probe
            )
            self.assertEqual(calls, [("claude", False)])
            self.assertEqual(warns, [])
            self.assertAlmostEqual(statuses["claude"].effective_used_ratio(now) or 0.0, 0.76)

            # Immediately calling again with fresh cache -> NO probe!
            statuses2, _ = refresh_plan_quotas(
                cfg, force=False, include_host_codex_rollouts=False, now_ts=now + 60, claude_probe_fn=fake_probe
            )
            self.assertEqual(len(calls), 1, "fresh cache must not trigger redundant probe")
            self.assertAlmostEqual(statuses2["claude"].effective_used_ratio(now + 60) or 0.0, 0.76)

            # Force=True OR stale cache (>1800s) -> probes again
            refresh_plan_quotas(
                cfg, force=True, include_host_codex_rollouts=False, now_ts=now + 60, claude_probe_fn=fake_probe
            )
            self.assertEqual(len(calls), 2)

            # Probe error degrades gracefully with warning instead of crashing
            def failing_probe(_cfg: AppConfig, pid: str, no_sb: bool) -> PlanQuotaStatus:
                raise RuntimeError("network offline")

            _, warns_err = refresh_plan_quotas(
                cfg, force=True, include_host_codex_rollouts=False, now_ts=now + 120, claude_probe_fn=failing_probe
            )
            self.assertEqual(len(warns_err), 1)
            self.assertIn("network offline", warns_err[0])


if __name__ == "__main__":
    unittest.main()
