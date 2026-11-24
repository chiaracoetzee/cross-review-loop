"""Regression tests for Astra Round 2 findings (astra-47..astra-60 and Round 1 incomplete items)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from xreview.adapters.codex import _read_codex_rollout_rate_limit
from xreview.config import (
    AppConfig,
    ConfigError,
    PolicyConfig,
    ProfileConfig,
    ReviewerConfig,
    TagRule,
    safe_open_user_regular_file,
    validate_reviewer_name,
)
from xreview.ledger import RunLedger, load_ledger, render_ledger_markdown, run_lock, save_ledger
from xreview.protocol import (
    ProtocolError,
    init_run,
    merge_findings,
    run_round,
    substitute_reviewer,
    triage_finding,
)
from xreview.snapshot import RepoSpec, SnapshotError, create_round_snapshot, verify_repo_is_worktree_root
from test_protocol import FakeAdapter, FakeClock, _finding, _make_git_repo, _reply


class Round2RegressionTests(unittest.TestCase):
    def test_astra_6_and_52_and_53_and_57_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # astra-6: symlinked parent directory MUST be refused by safe_open_user_regular_file
            real_dir = root / "real"
            real_dir.mkdir()
            (real_dir / "file.txt").write_text("secret", encoding="utf-8")
            sym_parent = root / "sym_parent"
            os.symlink(real_dir, sym_parent)
            with self.assertRaises(ConfigError):
                safe_open_user_regular_file(sym_parent / "file.txt")

            # astra-52: reviewer name traversal rejected
            for bad in ("../evil", "/abs", "a/b", "..", "_docs"):
                with self.assertRaises(ConfigError):
                    validate_reviewer_name(bad)
                with self.assertRaises(ConfigError):
                    ReviewerConfig.from_dict({"name": bad, "harness": "codex", "model": "m", "family": "gpt"})

            # astra-53: --repo subdirectory refused
            repo = _make_git_repo(root / "repo")
            sub = repo / "subdir"
            sub.mkdir()
            self.assertEqual(verify_repo_is_worktree_root(repo), repo.resolve())
            with self.assertRaises(SnapshotError):
                verify_repo_is_worktree_root(sub)

            # astra-57: run_lock on nonexistent run MUST NOT create run_dir
            nonexistent = root / "state" / "ghost-run"
            with run_lock(nonexistent):
                self.assertFalse(nonexistent.exists())

    def test_astra_49_legacy_ledger_fixed_finding_requires_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
            )
            clock = FakeClock()
            adapters = {h: FakeAdapter(h, lambda req: TurnResult_ok(req)) for h in ("codex", "agy")}
            from xreview.adapters.base import TurnResult

            def TurnResult_ok(req):
                return TurnResult(session_id=f"s-{req.reviewer.name}", reply=_reply())

            run_dir, ledger = init_run(
                cfg, "r-leg", [RepoSpec("repo", repo)], "Legacy test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock
            )
            # Inject a legacy-format ledger JSON with fixed finding, NO fix_revision, and verified_fix=false
            raw = ledger.to_dict()
            raw["current_round"] = 1
            raw["status"] = "needs_triage"
            raw["findings"] = [
                {
                    "ledger_id": "F-001",
                    "title": "Legacy bug",
                    "file": "calc.py",
                    "line": 2,
                    "evidence": "e",
                    "failure_scenario": "f",
                    "suggested_fix": "s",
                    "round_opened": 1,
                    "found_by": [
                        {
                            "reviewer": "astra",
                            "family": "gpt",
                            "key": "a-1",
                            "round_raised": 1,
                            "severity": "major",
                            "confidence": "high",
                            "conceded": False,
                            "verified_fix": False,
                        }
                    ],
                    "state": "open",
                    "disposition": "fixed",
                    "fix_note": "fixed",
                }
            ]
            (run_dir / "ledger.json").write_text(json.dumps(raw), encoding="utf-8")

            migrated = load_ledger(run_dir)
            self.assertEqual(migrated.get_finding("F-001").fix_revision, 1)
            self.assertEqual(migrated.get_finding("F-001").found_by[0].verified_fix_revision, 0)

            # Running round 2 with empty responses MUST NOT close F-001 (astra must be marked incomplete)!
            out2 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertNotEqual(out2.status, "converged")
            self.assertEqual(load_ledger(run_dir).get_finding("F-001").state, "open")

    def test_astra_50_major_deferral_stays_open_when_minor_finder_maintains(self) -> None:
        from xreview.adapters.base import TurnResult

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
            )
            clock = FakeClock()

            def handle(req):
                if req.round_num == 1:
                    if req.reviewer.name == "astra":
                        return TurnResult(session_id="sa", reply=_reply(findings=[_finding("a-1", "Issue", "major")]))
                    return TurnResult(session_id="sc", reply=_reply(findings=[_finding("c-1", "Issue dup", "minor")]))
                if req.round_num == 2:
                    # First carbon settles merge acceptance; astra concedes major deferral
                    if req.reviewer.name == "astra":
                        return TurnResult(session_id="sa", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "concede", "argument": "ok defer"}]))
                    return TurnResult(session_id="sc", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "concede", "argument": "merge ok"}]))
                if req.round_num == 3 and req.reviewer.name == "carbon":
                    # Now carbon (minor severity!) MAINTAINS objection to the major deferral!
                    return TurnResult(session_id="sc", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "maintain", "argument": "Should not defer"}]))
                return TurnResult(session_id=req.session_id, reply=_reply())

            adapters = {h: FakeAdapter(h, handle) for h in ("codex", "agy")}
            run_dir, _ = init_run(cfg, "r-def", [RepoSpec("repo", repo)], "Defer test", no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock)
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            merge_findings(run_dir, "F-002", "F-001", reason="dup", now_fn=clock)
            triage_finding(run_dir, "F-001", "deferred", note="defer to next sprint", now_fn=clock)

            out2 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(load_ledger(run_dir).get_finding("F-001").state, "open")
            out3 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            f1 = load_ledger(run_dir).get_finding("F-001")
            self.assertEqual(f1.state, "open")
            self.assertEqual(f1.disagreement_rounds, 1)
            self.assertEqual(out3.status, "needs_triage")

    def test_astra_34_and_51_quota_paused_core_preserves_finding_and_blocks_active_sub(self) -> None:
        from xreview.adapters.base import TurnResult

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                    ReviewerConfig("fable", "claude", "m3", "claude", "anthropic"),
                    ReviewerConfig("backup", "codex", "m4", "gpt", "openai"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
                implementor_family="claude",
            )
            clock = FakeClock()
            r2_hit = False

            def handle(req):
                nonlocal r2_hit
                if req.round_num == 1 and req.reviewer.name == "astra":
                    return TurnResult(session_id="sa", reply=_reply(findings=[_finding("a-1", "Core bug", "major")]))
                if req.round_num == 2 and req.reviewer.name == "astra" and not r2_hit:
                    r2_hit = True
                    return TurnResult(session_id="sa", quota_exhausted=True, error="usage limit", returncode=1)
                if req.round_num == 3 and req.reviewer.name == "astra":
                    return TurnResult(session_id="sa", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "concede", "argument": "after reset, ok"}]))
                return TurnResult(session_id=f"s-{req.reviewer.name}", reply=_reply())

            adapters = {h: FakeAdapter(h, handle) for h in ("codex", "agy", "claude")}
            run_dir, _ = init_run(
                cfg, "r-qp", [RepoSpec("repo", repo)], "QP test",
                explicit_reviewers=["astra", "carbon", "fable"],
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock
            )
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            triage_finding(run_dir, "F-001", "rejected", reason="safe", citation="calc.py:2", now_fn=clock)
            out2 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(out2.status, "paused")
            # F-001 MUST remain open (NOT terminally unverified) while core is temporarily quota_paused (astra-34)!
            self.assertEqual(load_ledger(run_dir).get_finding("F-001").state, "open")

            # Cannot substitute an active reviewer (astra-51)
            with self.assertRaises(ProtocolError):
                substitute_reviewer(run_dir, cfg, "carbon", "backup", adapter_lookup=adapters.__getitem__, now_fn=clock)

            # After quota resets, --resume-pause reactivates astra and F-001 closes cleanly!
            out3 = run_round(run_dir, cfg, resume_pause=True, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(out3.status, "converged")
            self.assertEqual(load_ledger(run_dir).get_finding("F-001").state, "closed")

    def test_astra_54_codex_rollout_uses_latest_sample_not_historical_max(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            codex_home = Path(td)
            sdir = codex_home / "sessions" / "2026" / "11" / "23"
            sdir.mkdir(parents=True)
            rollout = sdir / "rollout-2026-11-23-tid1.jsonl"
            lines = [
                json.dumps({"payload": {"rate_limits": {"primary": {"used_percent": 100.0}, "rate_limit_reached_type": "primary"}}}),
                json.dumps({"payload": {"rate_limits": {"primary": {"used_percent": 12.0}, "rate_limit_reached_type": None}}}),
            ]
            rollout.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ratio, reached = _read_codex_rollout_rate_limit(codex_home, "tid1")
            self.assertAlmostEqual(ratio or 0.0, 0.12)
            self.assertFalse(reached)

    def test_astra_56_and_58_doc_round_diff_and_same_tree_submodule_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            doc = base / "design.md"
            doc.write_text("# Design v1\nline 1\n", encoding="utf-8")
            run_dir = base / "run-doc"
            r1 = create_round_snapshot(run_dir, 1, [], set(), set(), {}, docs=[doc])
            doc.write_text("# Design v1\nline 1 updated\n", encoding="utf-8")
            r2 = create_round_snapshot(run_dir, 2, [], set(), set(), {}, docs=[doc])
            # Standalone document edit MUST appear in ROUND.diff (astra-56)!
            self.assertIn("+line 1 updated", r2.round_diff)
            self.assertTrue((run_dir / "r2" / "diffs" / "ROUND.diff").is_file())

            # astra-58: updating submodule between two distinct commits with IDENTICAL tree
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "T",
                "GIT_AUTHOR_EMAIL": "t@e.com",
                "GIT_COMMITTER_NAME": "T",
                "GIT_COMMITTER_EMAIL": "t@e.com",
                "GIT_ALLOW_PROTOCOL": "file",
            }
            sub_orig = base / "sub_orig"
            sub_orig.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=sub_orig, env=env, capture_output=True, check=True)
            (sub_orig / "f.txt").write_text("same\n")
            subprocess.run(["git", "add", "."], cwd=sub_orig, env=env, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "c1"], cwd=sub_orig, env=env, capture_output=True, check=True)

            parent = base / "parent"
            parent.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=parent, env=env, capture_output=True, check=True)
            subprocess.run(["git", "submodule", "add", str(sub_orig), "sub"], cwd=parent, env=env, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "add sub"], cwd=parent, env=env, capture_output=True, check=True)

            # Empty commit inside submodule (different SHA, identical tree!)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "c2 empty"], cwd=parent / "sub", env=env, capture_output=True, check=True)
            new_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=parent / "sub", env=env, capture_output=True, text=True, check=True).stdout.strip()

            snap = create_round_snapshot(base / "run-sub", 1, [RepoSpec("parent", parent)], set(), set(), {})
            self.assertEqual(snap.repo_snapshots["parent"].submodules["sub"][1], new_head)
            self.assertIn(new_head[:8], snap.base_diff)

    def test_astra_60_footer_counts_accepted_and_escalated_separately(self) -> None:
        from xreview.adapters.base import TurnResult

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
            )
            clock = FakeClock()

            def handle(req):
                if req.round_num == 1 and req.reviewer.name == "astra":
                    return TurnResult(session_id="sa", reply=_reply(findings=[_finding("a-1", "Disputed", "major")]))
                if req.round_num == 2:
                    if req.reviewer.name == "astra":
                        return TurnResult(session_id="sa", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "maintain", "argument": "real"}]))
                    return TurnResult(session_id="sc", reply=_reply(responses=[{"ledger_id": "F-001", "verdict": "second", "argument": "agree"}]))
                return TurnResult(session_id=f"s-{req.reviewer.name}", reply=_reply())

            adapters = {h: FakeAdapter(h, handle) for h in ("codex", "agy")}
            run_dir, _ = init_run(cfg, "r-ftr", [RepoSpec("repo", repo)], "Footer test", no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock)
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            triage_finding(run_dir, "F-001", "rejected", reason="nope", citation="calc.py:2", now_fn=clock)
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            md = render_ledger_markdown(load_ledger(run_dir))
            # Escalated rejection -> Raised=1, Accepted=0, Escalated=1!
            self.assertIn("| `astra` | `m1` | gpt / openai | codex 1.0.0 | `active` | 1 | 0 | 1 |", md)


if __name__ == "__main__":
    unittest.main()
