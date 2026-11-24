"""Regression tests for bugs caught during live dogfooding (F-001..F-017 + --base EMPTY)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xreview.adapters.base import TurnResult, detect_resume_failed
from xreview.config import (
    AppConfig,
    PolicyConfig,
    ProfileConfig,
    ReviewerConfig,
    TagRule,
)
from xreview.ledger import load_ledger, save_ledger
from xreview.protocol import (
    attach_evidence,
    init_run,
    merge_findings,
    run_round,
    triage_finding,
)
from xreview.sandbox import (
    SandboxSpec,
    _sandbox_accessible_socat,
    build_bwrap_command,
    run_in_sandbox,
)
from xreview.snapshot import EMPTY_TREE_SHA, RepoSpec, create_round_snapshot
from test_protocol import FakeAdapter, FakeClock, _finding, _make_git_repo, _reply


class DogfoodRegressionTests(unittest.TestCase):
    def test_empty_base_rev_reviews_root_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            # Revert working-tree modification so working tree == root commit
            subprocess.run(["git", "checkout", "--", "calc.py"], cwd=repo, check=True)
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
                policy=PolicyConfig(min_families=2),
                state_dir=root / "state",
            )
            run_dir, ledger = init_run(cfg, "r-root", [RepoSpec("repo", repo, base_rev="EMPTY")], "Root review", no_sandbox=True)
            self.assertEqual(ledger.repos[0].base_rev, EMPTY_TREE_SHA)
            snap = create_round_snapshot(run_dir, 1, list(ledger.repos), set(), set(), {})
            self.assertIn("calc.py", snap.base_diff)

    def test_f001_failed_round1_reviewer_reviews_base_diff_in_round2(self) -> None:
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
            astra_prompts: list[tuple[int, str | None, str]] = []

            def astra_fn(req):
                astra_prompts.append((req.round_num, req.session_id, req.prompt))
                if req.round_num == 1:
                    # Emit session_id, then fail with quota before producing valid reply!
                    return TurnResult(session_id="broken-r1-sid", error="quota", quota_exhausted=True, returncode=1)
                return TurnResult(session_id="fresh-r2-sid", reply=_reply())

            adapters = {
                "codex": FakeAdapter("codex", astra_fn),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg, "r-f1", [RepoSpec("repo", repo)], "F-001 test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            out1 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(out1.status, "paused")
            ledger = load_ledger(run_dir)
            self.assertIsNone(ledger.reviewers["astra"].session_id)
            self.assertFalse(ledger.reviewers["astra"].initial_review_completed)

            # Resume in Round 2 -> astra MUST run fresh (session_id=None) with BASE.diff & BRIEF.md!
            run_round(run_dir, cfg, resume_pause=True, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(len(astra_prompts), 2)
            self.assertIsNone(astra_prompts[1][1])
            self.assertIn("BASE.diff", astra_prompts[1][2])
            self.assertIn("BRIEF.md", astra_prompts[1][2])
            ledger = load_ledger(run_dir)
            self.assertTrue(ledger.reviewers["astra"].initial_review_completed)

    def test_f002_evidence_revocation_clears_evidence_exposed_no_repeat_wipes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg_all = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
                policy=PolicyConfig(min_families=2),
                profile=ProfileConfig(tags={"evtag": TagRule(name="evtag", vendors=("openai", "google"))}),
                state_dir=root / "state",
            )
            clock = FakeClock()
            seen_sids: list[tuple[int, str | None]] = []

            def astra_fn(req):
                seen_sids.append((req.round_num, req.session_id))
                if req.round_num == 1:
                    return TurnResult(session_id="s-a-1", reply=_reply(findings=[_finding("a-1", "Bug 1")]))
                if req.round_num == 2:
                    return TurnResult(
                        session_id="s-a-2",
                        reply=_reply(responses=[{"ledger_id": f1_id, "verdict": "fix_incomplete", "argument": "try again"}]),
                    )
                return TurnResult(
                    session_id="s-a-2",
                    reply=_reply(responses=[{"ledger_id": f1_id, "verdict": "fix_ok", "argument": "good"}]),
                )

            adapters = {
                "codex": FakeAdapter("codex", astra_fn),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg_all, "r-f2", [RepoSpec("repo", repo)], "F-002 test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            ev = root / "ev.txt"
            ev.write_text("trace", encoding="utf-8")
            attach_evidence(run_dir, ev, "cmd", tags=["evtag"])
            out1 = run_round(run_dir, cfg_all, adapter_lookup=adapters.__getitem__, now_fn=clock)
            f1_id = out1.new_findings[0]
            triage_finding(run_dir, f1_id, "fixed", note="fix1", now_fn=clock)

            # Revoke evtag from openai before R2
            cfg_google = AppConfig(
                reviewers=cfg_all.reviewers,
                policy=cfg_all.policy,
                profile=ProfileConfig(tags={"evtag": TagRule(name="evtag", vendors=("google",))}),
                state_dir=root / "state",
            )
            run_round(run_dir, cfg_google, adapter_lookup=adapters.__getitem__, now_fn=clock)
            ledger = load_ledger(run_dir)
            self.assertEqual(ledger.reviewers["astra"].evidence_exposed, [])

            # Round 3 MUST keep astra's warm session s-a-2 (NOT wipe it again!)
            triage_finding(run_dir, f1_id, "fixed", note="fix2", now_fn=clock)
            run_round(run_dir, cfg_google, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(seen_sids, [(1, None), (2, None), (3, "s-a-2")])

    def test_f008_and_f009_fixed_clears_proposed_severity_and_replacement_verifies_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = AppConfig(
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                    ReviewerConfig("sol", "codex", "m3", "gpt", "openai", extra=True),
                ),
                policy=PolicyConfig(min_families=2),
                implementor_family="gemini",
                state_dir=root / "state",
            )
            clock = FakeClock()

            def astra_fn(req):
                if req.round_num == 1:
                    return TurnResult(session_id="s-a", reply=_reply(findings=[_finding("a-1", "Bug 1", severity="major")]))
                return TurnResult(
                    session_id="s-a",
                    reply=_reply(responses=[{"ledger_id": f1_id, "verdict": "fix_ok", "argument": "verified"}]),
                )

            def sol_fn(req):
                if req.round_num == 1:
                    return TurnResult(session_id="s-s", reply=_reply(findings=[_finding("s-1", "Bug 1 dup", severity="major")]))
                return TurnResult(session_id="s-s", reply=_reply())

            adapters = {
                "codex": FakeAdapter("codex", lambda req: astra_fn(req) if req.reviewer.name == "astra" else sol_fn(req)),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg, "r-f89", [RepoSpec("repo", repo)], "F-008/F-009 test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            ledger = load_ledger(run_dir)
            by_key = {f.found_by[0].key: f.ledger_id for f in ledger.findings}
            f1_id = by_key["a-1"]

            # F-008: lower_severity then retriage fixed MUST clear proposed_severity
            triage_finding(run_dir, f1_id, "lower_severity", new_severity="minor", reason="minor", now_fn=clock)
            self.assertEqual(load_ledger(run_dir).get_finding(f1_id).proposed_severity, "minor")
            triage_finding(run_dir, f1_id, "fixed", note="fixed it", now_fn=clock)
            self.assertIsNone(load_ledger(run_dir).get_finding(f1_id).proposed_severity)

            # F-009: merge sol into f1_id (accepted), drop sol, assign astra as replacement_verifier -> closes in ONE round!
            merge_findings(run_dir, by_key["s-1"], f1_id, now_fn=clock)
            ledger = load_ledger(run_dir)
            f1 = ledger.get_finding(f1_id)
            for fr in f1.found_by:
                fr.merge_accepted = True
                fr.pending_merges.clear()
            ledger.reviewers["sol"].status = "shed_allowance"
            f1.replacement_verifier = "astra"
            save_ledger(run_dir, ledger)

            out2 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertIn(f1_id, out2.closed_findings, "live finder acting as replacement_verifier must close in 1 round")

    def test_f004_f005_f006_f014_f015_f016_sandbox_hardening(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree, inputs, state = root / "tree", root / "inputs", root / "state"
            tree.mkdir(); inputs.mkdir(); state.mkdir()

            # F-006 / F-014: socat outside mounted system dirs must NOT be selected
            with mock.patch("xreview.sandbox.shutil.which", return_value="/usr/local/bin/socat"):
                self.assertIsNone(_sandbox_accessible_socat())

            spec_bwrap = SandboxSpec(
                run_id="r", reviewer_name="rev", tree_host_dir=tree, inputs_host_dir=inputs,
                state_host_dir=state, harness_mounts=(), command=("true",),
                cwd_in_sandbox="/review/r/rev/tree", env={}, secret_env={}, vendor_hosts=(),
                timeout_s=900, use_sandbox=True,
            )
            cmd = build_bwrap_command(spec_bwrap, root / "p.sock", root / "etc", use_socat=False)
            wrapper = cmd[-1]
            self.assertIn("900", wrapper, "Python proxy bridge must receive turn timeout as idle_timeout")
            self.assertIn("exit 97", wrapper, "bridge readiness loop must fail fast if port never binds")

            # F-004 / F-016: oversized stdout line (>cap) is drained without splitting into on_stdout_line
            seen_lines: list[str] = []
            spec_big = SandboxSpec(
                run_id="r", reviewer_name="rev", tree_host_dir=tree, inputs_host_dir=inputs,
                state_host_dir=state, harness_mounts=(),
                command=("python3", "-c", "print('A'*500); print('OK_AFTER')"),
                cwd_in_sandbox=str(tree), env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                secret_env={}, vendor_hosts=(), timeout_s=15, use_sandbox=False,
            )
            with mock.patch("xreview.sandbox._MAX_EVENT_LINE_CHARS", 100):
                res = run_in_sandbox(spec_big, None, root / "raw.log", root / "p.log", on_stdout_line=seen_lines.append)
            self.assertEqual(res.returncode, 0)
            self.assertEqual(seen_lines, ["OK_AFTER"], "oversized line must not be split into parser fragments")
            self.assertIn("dropped oversized stdout event line", res.stderr)

            # F-005 & F-022: descendant holding stderr OR stdout open must NOT hang or time out run_in_sandbox
            spec_desc = SandboxSpec(
                run_id="r", reviewer_name="rev", tree_host_dir=tree, inputs_host_dir=inputs,
                state_host_dir=state, harness_mounts=(),
                command=(
                    "python3",
                    "-c",
                    "import subprocess; subprocess.Popen(['sleep', '60']); print('DONE', flush=True)",
                ),
                cwd_in_sandbox=str(tree), env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                secret_env={}, vendor_hosts=(), timeout_s=15, use_sandbox=False,
            )
            res_desc = run_in_sandbox(spec_desc, None, root / "raw2.log", root / "p2.log", capture_stdout=True)
            self.assertEqual(res_desc.returncode, 0)
            self.assertFalse(res_desc.timed_out)
            self.assertEqual(res_desc.stdout_lines, ["DONE"])
            self.assertLess(res_desc.duration_s, 5.0)

    def test_f007_f011_f012_snapshot_fixes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub_origin = _make_git_repo(root / "sub_origin")
            parent = _make_git_repo(root / "parent")
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), "sub"],
                cwd=parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["git", "commit", "-m", "add sub"], cwd=parent, check=True, stdout=subprocess.DEVNULL)
            # Edit sub for Round 1 (so it's included even though excluded by profile)
            (parent / "sub" / "calc.py").write_text("def div(a, b):\n    return a / b  # sub edit\n", encoding="utf-8")
            doc = root / "spec.md"
            doc.write_text("# Spec\nline1\n", encoding="utf-8")

            run_dir = root / "run"
            run_dir.mkdir()
            snap1 = create_round_snapshot(
                run_dir, 1, [RepoSpec("parent", parent)], set(), set(),
                profile_excluded_submodules={"parent": {"sub"}}, docs=[doc],
            )
            self.assertIn("sub", snap1.repo_snapshots["parent"].submodules)

            # Revert sub back to base AND modify doc AND simulate crashed partial r2 publish (F-011)
            subprocess.run(["git", "checkout", "--", "calc.py"], cwd=parent / "sub", check=True)
            doc.write_text("# Spec\nline2\n", encoding="utf-8")
            (run_dir / "r2" / "diffs").mkdir(parents=True)
            (run_dir / "r2" / "diffs" / "stale.diff").write_text("stale", encoding="utf-8")
            (run_dir / "r2" / ".staging_orphan").mkdir(parents=True)

            snap2 = create_round_snapshot(
                run_dir, 2, [RepoSpec("parent", parent)], set(), set(),
                profile_excluded_submodules={"parent": {"sub"}},
                prev_snapshots=snap1.repo_snapshots, docs=[doc],
            )
            # F-011: orphan cleaned, r2 succeeded on first attempt
            self.assertFalse((run_dir / "r2" / ".staging_orphan").exists())
            # F-007: reverted excluded submodule emits revert diff, NOT 'removed since last round'
            self.assertNotIn("removed since last round", snap2.round_diff)
            self.assertIn("reverted to base", snap2.round_diff)
            # F-012: doc diff has valid a/docs/spec.md & b/docs/spec.md headers and changed_lines recorded
            self.assertIn("+++ b/docs/spec.md", snap2.round_diff)
            self.assertNotIn("bdocs/", snap2.round_diff)
            self.assertIn("docs/spec.md", snap2.changed_lines_in_round)

    def test_f017_f019_f021_f023_detect_resume_failed_distinguishes_transient_errors(self) -> None:
        self.assertTrue(detect_resume_failed("Error: conversation not found"))
        self.assertTrue(detect_resume_failed("Session abc-123 not found"))
        self.assertTrue(detect_resume_failed("Conversation conv_xyz does not exist"))
        self.assertTrue(detect_resume_failed("Thread 019a0ae9-58ff-7a30 expired"))
        self.assertTrue(detect_resume_failed("Session ID: 3f2a-9999 not found"))
        self.assertTrue(detect_resume_failed("Invalid session id abc"))
        self.assertTrue(detect_resume_failed("failed to resume thread 123"))
        self.assertFalse(detect_resume_failed("OAuth session expired; please re-login"))
        self.assertFalse(detect_resume_failed("502 Bad Gateway: upstream connection reset"))
        self.assertFalse(detect_resume_failed("HTTP 503 Service Unavailable"))

    def test_f020_nested_submodule_under_excluded_parent_reverts_with_content_diff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inner_origin = _make_git_repo(root / "inner_origin")
            outer_origin = _make_git_repo(root / "outer_origin")
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(inner_origin), "b"],
                cwd=outer_origin, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["git", "commit", "-m", "add inner b"], cwd=outer_origin, check=True, stdout=subprocess.DEVNULL)

            parent = _make_git_repo(root / "parent")
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(outer_origin), "a"],
                cwd=parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["git", "commit", "-m", "add outer a"], cwd=parent, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive"],
                cwd=parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            # Dirty nested submodule a/b in Round 1 (parent a is excluded by profile)
            (parent / "a" / "b" / "calc.py").write_text("def div(a, b):\n    return a / b  # nested edit\n", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            snap1 = create_round_snapshot(
                run_dir, 1, [RepoSpec("parent", parent)], set(), set(),
                profile_excluded_submodules={"parent": {"a"}},
            )
            self.assertIn("a/b", snap1.repo_snapshots["parent"].submodules)
            self.assertIn("a/b", snap1.repo_snapshots["parent"].submodule_obj_dirs)

            # Revert a/b in Round 2 -> ROUND.diff MUST emit a/b's content revert diff, NOT deletion!
            subprocess.run(["git", "checkout", "--", "calc.py"], cwd=parent / "a" / "b", check=True)
            snap2 = create_round_snapshot(
                run_dir, 2, [RepoSpec("parent", parent)], set(), set(),
                profile_excluded_submodules={"parent": {"a"}},
                prev_snapshots=snap1.repo_snapshots,
            )
            self.assertNotIn("removed since last round", snap2.round_diff)
            self.assertIn("reverted to base", snap2.round_diff)
            self.assertIn("# nested edit", snap2.round_diff)


if __name__ == "__main__":
    unittest.main()
