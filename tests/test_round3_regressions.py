"""Regression tests for Astra Round 3 findings (astra-61..astra-74 and Round 2 incomplete items)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from xreview.adapters.agy import AgyAdapter
from xreview.adapters.base import TurnResult
from xreview.config import (
    AppConfig,
    AuthConfig,
    PolicyConfig,
    ProfileConfig,
    ReviewerConfig,
    TagRule,
)
from xreview.ledger import load_ledger, save_ledger
from xreview.policy import PolicyViolation
from xreview.protocol import (
    ProtocolError,
    _compute_snapshot_hash,
    _handle_dropouts_and_replacements,
    attach_evidence,
    clean_run,
    init_run,
    merge_findings,
    run_pending_hooks,
    run_round,
    substitute_reviewer,
    triage_finding,
)
from xreview.sandbox import SandboxSpec, run_in_sandbox
from xreview.snapshot import RepoSpec, SnapshotError, create_round_snapshot
from test_protocol import FakeAdapter, FakeClock, _finding, _make_git_repo, _reply


def _make_cfg(
    state_dir: Path,
    *,
    reviewers=None,
    policy=None,
    profile=None,
    auth=None,
) -> AppConfig:
    if reviewers is None:
        reviewers = (
            ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
            ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
        )
    if policy is None:
        policy = PolicyConfig(min_families=2)
    return AppConfig(
        reviewers=reviewers,
        policy=policy,
        state_dir=state_dir,
        profile=profile or ProfileConfig(),
        auth=auth or AuthConfig(),
    )


class Round3RegressionTests(unittest.TestCase):
    def test_astra_61_and_69_state_dir_inside_repo_and_init_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            nested_state = repo / ".xreview-state"
            cfg = _make_cfg(nested_state)
            with self.assertRaises(ProtocolError):
                init_run(cfg, "r-nested", [RepoSpec("repo", repo)], "Nested state test", no_sandbox=True)
            self.assertFalse(nested_state.exists(), "must not create state_dir inside reviewed repo")

            # astra-69: unknown harness in init_run must not leave run_dir behind
            good_state = root / "state"
            bad_cfg = _make_cfg(
                good_state,
                reviewers=(
                    ReviewerConfig("astra", "nonexistent_harness", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                ),
            )
            with self.assertRaises(Exception):
                init_run(bad_cfg, "r-fail", [RepoSpec("repo", repo)], "Fail init", no_sandbox=True)
            self.assertFalse((good_state / "r-fail").exists(), "failed init_run must clean up reserved run_dir")

    def test_astra_62_fixed_to_ask_tag_change_not_pre_approved_by_unrelated_ask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            profile_v1 = ProfileConfig(
                tags={
                    "tag_a": TagRule(name="tag_a", vendors="ask"),
                    "tag_b": TagRule(name="tag_b", vendors=("openai", "google")),
                }
            )
            cfg_v1 = _make_cfg(root / "state", profile=profile_v1)
            clock = FakeClock()
            adapters = {
                "codex": FakeAdapter("codex", lambda req: TurnResult(session_id="s-a", reply=_reply())),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, ledger = init_run(
                cfg_v1,
                "r-tag",
                [RepoSpec("repo", repo)],
                "Tag test",
                cli_tags=["tag_a", "tag_b"],
                explicit_reviewers=["astra", "carbon"],
                raw_allows=["tag_a=openai,google"],
                no_sandbox=True,
                adapter_lookup=adapters.__getitem__,
                now_fn=clock,
            )
            self.assertEqual(ledger.approved_ask_tags, ["tag_a"])

            # Profile changes tag_b from fixed to ask -> run_round MUST fail without tag_b approval
            profile_v2 = ProfileConfig(
                tags={
                    "tag_a": TagRule(name="tag_a", vendors="ask"),
                    "tag_b": TagRule(name="tag_b", vendors="ask"),
                }
            )
            cfg_v2 = _make_cfg(root / "state", profile=profile_v2)
            with self.assertRaises(PolicyViolation) as ctx:
                run_round(run_dir, cfg_v2, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertIn("tag_b", str(ctx.exception))

    def test_astra_63_65_and_50_dropped_finder_and_retriage_concession(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = _make_cfg(
                root / "state",
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                    ReviewerConfig("opus", "claude", "m3", "claude", "anthropic"),
                ),
                policy=PolicyConfig(min_families=2),
            )
            clock = FakeClock()

            def astra_fn(req):
                if req.round_num == 1:
                    return TurnResult(
                        session_id="s-a",
                        reply=_reply(findings=[
                            _finding("a-1", severity="major", title="Bug 1"),
                            _finding("a-2", severity="minor", title="Bug 2"),
                            _finding("a-3", severity="major", title="Bug 3"),
                        ]),
                    )
                return TurnResult(session_id="s-a", reply=_reply())

            def carbon_fn(req):
                if req.round_num == 1:
                    return TurnResult(
                        session_id="s-c",
                        reply=_reply(findings=[
                            _finding("c-1", severity="minor", title="Bug 1 dup"),
                            _finding("c-2", severity="major", title="Bug 2 dup"),
                        ]),
                    )
                return TurnResult(session_id="s-c", reply=_reply())

            def opus_fn(req):
                if req.round_num == 1:
                    return TurnResult(
                        session_id="s-o",
                        reply=_reply(findings=[_finding("o-3", severity="major", title="Bug 3 dup")]),
                    )
                return TurnResult(session_id="s-o", reply=_reply())

            adapters = {
                "codex": FakeAdapter("codex", astra_fn),
                "agy": FakeAdapter("agy", carbon_fn),
                "claude": FakeAdapter("claude", opus_fn),
            }
            run_dir, _ = init_run(
                cfg, "r-state", [RepoSpec("repo", repo)], "State test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            ledger = load_ledger(run_dir)
            by_key = {f.found_by[0].key: f.ledger_id for f in ledger.findings}

            # Merge src -> dst: c-1 into a-1, c-2 into a-2, o-3 into a-3
            merge_findings(run_dir, by_key["c-1"], by_key["a-1"], now_fn=clock)
            merge_findings(run_dir, by_key["c-2"], by_key["a-2"], now_fn=clock)
            merge_findings(run_dir, by_key["o-3"], by_key["a-3"], now_fn=clock)

            triage_finding(run_dir, by_key["a-1"], "deferred", reason="later", now_fn=clock)
            triage_finding(run_dir, by_key["a-2"], "fixed", note="fixed a-2", now_fn=clock)
            triage_finding(run_dir, by_key["a-3"], "rejected", reason="nope", citation="calc.py:1", now_fn=clock)

            # Simulate major finder astra having conceded a-1 & a-3 after merge settled
            ledger = load_ledger(run_dir)
            f1 = ledger.get_finding(by_key["a-1"])
            f3 = ledger.get_finding(by_key["a-3"])
            for fr in f1.found_by:
                fr.merge_accepted = True
                fr.pending_merges.clear()
            f1.found_by[0].conceded = True  # major finder astra conceded
            f1.deferral_requires_concession = True
            for fr in f3.found_by:
                fr.merge_accepted = True
                fr.pending_merges.clear()
            f3.found_by[0].conceded = True  # astra conceded rejection
            save_ledger(run_dir, ledger)

            # astra-50: retriaging a-1 as deferred MUST preserve deferral_requires_concession=True
            triage_finding(run_dir, by_key["a-1"], "deferred", reason="updated deferral note", now_fn=clock)
            ledger = load_ledger(run_dir)
            f1 = ledger.get_finding(by_key["a-1"])
            self.assertTrue(f1.deferral_requires_concession)

            # astra-65: switching a-3 to fixed MUST reset astra's earlier conceded flag
            triage_finding(run_dir, by_key["a-3"], "fixed", note="actually fixed it", now_fn=clock)
            ledger = load_ledger(run_dir)
            f3 = ledger.get_finding(by_key["a-3"])
            self.assertFalse(f3.found_by[0].conceded)

            # astra-63: permanently dropping carbon while carbon has pending merge on a-2 marks a-2 unverified
            f2 = ledger.get_finding(by_key["a-2"])
            carbon_fr = next(fr for fr in f2.found_by if fr.reviewer == "carbon")
            self.assertFalse(carbon_fr.merge_accepted)
            ledger.reviewers["carbon"].status = "dropped_quota"
            unverified = _handle_dropouts_and_replacements(ledger, clock, round_num=2)
            self.assertIn(by_key["a-2"], unverified)
            self.assertEqual(f2.state, "unverified")

    def test_astra_64_crash_resume_when_quorum_already_broken_mid_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = _make_cfg(root / "state")
            clock = FakeClock()
            adapters = {
                "codex": FakeAdapter("codex", lambda req: TurnResult(session_id="s-a", reply=_reply())),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, ledger = init_run(
                cfg, "r-crash", [RepoSpec("repo", repo)], "Crash test",
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            # Create the canonical r1 snapshot as it would exist before mid-round crash
            create_round_snapshot(run_dir, 1, list(ledger.repos), set(), set(), {})
            ledger.in_progress_round = 1
            ledger.pending_round_roster = ["astra", "carbon"]
            ledger.pending_round_obligations = {"astra": {}, "carbon": {}}
            ledger.pending_round_results = {
                "astra": {
                    "session_id": None,
                    "reply": None,
                    "returncode": 1,
                    "duration_s": 1.0,
                    "error": "429 rate limit",
                    "quota_exhausted": True,
                    "timed_out": False,
                    "malformed_output": False,
                    "resume_failed": False,
                    "token_usage": None,
                    "quota_fraction": None,
                    "val_errors": ["429 rate limit"],
                },
                "carbon": {
                    "session_id": "s-c",
                    "reply": _reply(),
                    "returncode": 0,
                    "duration_s": 1.0,
                    "error": "",
                    "quota_exhausted": False,
                    "timed_out": False,
                    "malformed_output": False,
                    "resume_failed": False,
                    "token_usage": None,
                    "quota_fraction": None,
                    "val_errors": [],
                },
            }
            ledger.reviewers["astra"].status = "quota_paused"
            save_ledger(run_dir, ledger)

            # Resuming run_round must finalize the checkpointed work into a durable quorum pause
            outcome = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(outcome.status, "paused")
            ledger = load_ledger(run_dir)
            self.assertIsNone(ledger.in_progress_round)
            self.assertEqual(ledger.current_round, 1)
            self.assertTrue(any(c.startswith("quorum:") for c in ledger.pause_causes))

    def test_astra_66_simultaneous_quota_pause_and_soft_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            cfg = _make_cfg(
                root / "state",
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                    ReviewerConfig("opus", "claude", "m3", "claude", "anthropic", extra=True),
                ),
                policy=PolicyConfig(min_families=2, soft_round_ceiling=2),
            )
            clock = FakeClock()

            def astra_fn(req):
                if req.round_num == 1:
                    return TurnResult(session_id="s-a", reply=_reply(findings=[_finding("a-1", "Bug 1")]))
                return TurnResult(session_id="s-a", error="quota", quota_exhausted=True, returncode=1)

            adapters = {
                "codex": FakeAdapter("codex", astra_fn),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
                "claude": FakeAdapter("claude", lambda req: TurnResult(session_id="s-o", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg, "r-ceil", [RepoSpec("repo", repo)], "Ceiling test",
                explicit_reviewers=["astra", "carbon"],
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            out1 = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            triage_finding(run_dir, out1.new_findings[0], "fixed", note="fixed", now_fn=clock)
            # Round 2 hits BOTH astra quota pause AND soft_round_ceiling=2
            outcome = run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(outcome.status, "paused")
            ledger = load_ledger(run_dir)
            self.assertTrue(any(c.startswith("quota:") for c in ledger.pause_causes))
            self.assertTrue(any(c.startswith("ceiling:") for c in ledger.pause_causes))

            # Substituting astra with opus resolves quota/quorum, MUST preserve ceiling pause!
            ledger = substitute_reviewer(run_dir, cfg, "astra", "opus", adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(ledger.status, "paused")
            self.assertEqual(ledger.pause_causes, ["ceiling:2"])
            self.assertIn("Soft round ceiling", ledger.pause_reason)

    def test_astra_67_init_pins_base_rev_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            orig_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            cfg = _make_cfg(root / "state")
            run_dir, ledger = init_run(
                cfg, "r-pin", [RepoSpec("repo", repo, base_rev="HEAD")], "Pin test", no_sandbox=True
            )
            self.assertEqual(ledger.repos[0].base_rev, orig_head)

            # Advance HEAD in source repo -> BASE.diff must still diff against orig_head
            (repo / "calc.py").write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, stdout=subprocess.DEVNULL)

            snap = create_round_snapshot(run_dir, 1, list(ledger.repos), set(), set(), {})
            self.assertIn("a + b + 1", snap.base_diff)

    def test_astra_68_agy_prepare_state_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tok = root / "agy-token"
            tok.write_text('{"access_token":"abc"}', encoding="utf-8")
            state_dir = root / "state" / "carbon"
            reviewer = ReviewerConfig("carbon", "agy", "m2", "gemini", "google")
            cfg = _make_cfg(root / "state", auth=AuthConfig(agy_token_file=tok))
            adapter = AgyAdapter()
            adapter.prepare_state(reviewer, state_dir, cfg)
            token_dst = state_dir / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
            self.assertTrue(token_dst.is_file())
            self.assertEqual(token_dst.read_text(encoding="utf-8"), '{"access_token":"abc"}')
            # Calling prepare_state a second time on the same state_dir MUST succeed idempotently
            adapter.prepare_state(reviewer, state_dir, cfg)
            self.assertTrue((state_dir.parent / ".carbon.agy_init").is_file())

    def test_astra_70_and_41_hooks_recheck_policy_and_never_rerun_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = _make_git_repo(root / "repo")
            out_file = root / "hook_out.txt"
            hook_cmd = f"python3 -c \"open({str(out_file)!r}, 'a').write('ran\\n')\""
            profile_v1 = ProfileConfig(
                tags={"security": TagRule(name="security", vendors=("openai", "google"), post_run_hooks=True)},
                post_run_hooks=(hook_cmd,),
            )
            cfg_v1 = _make_cfg(root / "state", profile=profile_v1)
            clock = FakeClock()
            adapters = {
                "codex": FakeAdapter("codex", lambda req: TurnResult(session_id="s-a", reply=_reply())),
                "agy": FakeAdapter("agy", lambda req: TurnResult(session_id="s-c", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg_v1, "r-hook", [RepoSpec("repo", repo)], "Hook test",
                cli_tags=["security"], no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            run_round(run_dir, cfg_v1, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(out_file.read_text(encoding="utf-8"), "ran\n")

            # Simulate an uncertain crash mid-hook ("running" status)
            ledger = load_ledger(run_dir)
            ledger.hooks_executed[0]["status"] = "running"
            save_ledger(run_dir, ledger)

            # Default run_pending_hooks MUST NOT auto-rerun a 'running' hook
            res = run_pending_hooks(run_dir, cfg_v1)
            self.assertEqual(res[0]["status"], "running")
            self.assertEqual(out_file.read_text(encoding="utf-8"), "ran\n")

            # If current profile disables hooks for 'security', recovery MUST raise PolicyViolation
            cfg_disallowed = _make_cfg(
                root / "state",
                profile=ProfileConfig(
                    tags={"security": TagRule(name="security", vendors=("openai", "google"), post_run_hooks=False)},
                    post_run_hooks=(hook_cmd,),
                ),
            )
            with self.assertRaises(PolicyViolation):
                run_pending_hooks(run_dir, cfg_disallowed, retry_uncertain=True)
            self.assertEqual(out_file.read_text(encoding="utf-8"), "ran\n")

            # If command was removed from profile.post_run_hooks, recovery marks it blocked_policy
            cfg_removed = _make_cfg(
                root / "state",
                profile=ProfileConfig(tags=profile_v1.tags, post_run_hooks=()),
            )
            res2 = run_pending_hooks(run_dir, cfg_removed, retry_uncertain=True)
            self.assertEqual(res2[0]["status"], "blocked_policy")
            self.assertEqual(out_file.read_text(encoding="utf-8"), "ran\n")

            # Reset to running and retry with current trusted profile allowing it -> succeeds
            ledger = load_ledger(run_dir)
            ledger.hooks_executed[0]["status"] = "running"
            save_ledger(run_dir, ledger)
            res3 = run_pending_hooks(run_dir, cfg_v1, retry_uncertain=True)
            self.assertEqual(res3[0]["status"], "ok")
            self.assertEqual(out_file.read_text(encoding="utf-8"), "ran\nran\n")

    def test_astra_71_streaming_stdout_and_bounded_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree, inputs, state = root / "tree", root / "inputs", root / "state"
            tree.mkdir(); inputs.mkdir(); state.mkdir()
            raw_log = root / "raw.log"
            spec = SandboxSpec(
                run_id="r71",
                reviewer_name="rev",
                tree_host_dir=tree,
                inputs_host_dir=inputs,
                state_host_dir=state,
                harness_mounts=(),
                command=(
                    "python3",
                    "-c",
                    "import sys; print('LINE_ONE'); print('LINE_TWO'); sys.stderr.write('E' * 300000)",
                ),
                cwd_in_sandbox=str(tree),
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                secret_env={},
                vendor_hosts=(),
                timeout_s=15,
                use_sandbox=False,
            )
            res = run_in_sandbox(spec, None, raw_log, root / "proxy.log", capture_stdout=False)
            self.assertEqual(res.returncode, 0)
            self.assertEqual(res.stdout_lines, [])
            self.assertIn("LINE_ONE", raw_log.read_text(encoding="utf-8"))
            self.assertEqual(len(res.stderr), 262144, "stderr must be bounded at 256 KiB")

    def test_astra_72_73_74_submodule_untracked_non_utf8_and_descriptor_staging(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub_origin = _make_git_repo(root / "sub_origin")
            parent = _make_git_repo(root / "parent")
            subprocess.run(
                ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub_origin), "sub"],
                cwd=parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(["git", "commit", "-m", "add sub"], cwd=parent, check=True, stdout=subprocess.DEVNULL)
            # astra-72: status.showUntrackedFiles=no inside submodule
            subprocess.run(["git", "config", "status.showUntrackedFiles", "no"], cwd=parent / "sub", check=True)
            (parent / "sub" / "extra.txt").write_text("hello from sub\n", encoding="utf-8")

            # astra-73: non-UTF-8 standalone doc filename
            non_utf8_doc = os.fsdecode(os.fsencode(str(root / "doc_")) + b"\xff.md")
            Path(non_utf8_doc).write_text("# Doc\nline1\n", encoding="utf-8")

            run_dir = root / "run"
            run_dir.mkdir()
            snap = create_round_snapshot(
                run_dir,
                1,
                [RepoSpec("parent", parent)],
                included_files={"sub/extra.txt"},
                excluded_files=set(),
                profile_excluded_submodules={},
                docs=[Path(non_utf8_doc)],
            )
            self.assertTrue((snap.tree_dir / "sub" / "extra.txt").is_file())
            self.assertEqual((snap.tree_dir / "sub" / "extra.txt").read_text(encoding="utf-8"), "hello from sub\n")
            h1 = _compute_snapshot_hash(snap.repo_snapshots, snap.docs_dir)
            self.assertEqual(len(h1), 24)

            # Round 2 with modified non-UTF-8 doc -> diff and hash must work
            Path(non_utf8_doc).write_text("# Doc\nline2\n", encoding="utf-8")
            snap2 = create_round_snapshot(
                run_dir,
                2,
                [RepoSpec("parent", parent)],
                included_files={"sub/extra.txt"},
                excluded_files=set(),
                profile_excluded_submodules={},
                prev_snapshots=snap.repo_snapshots,
                docs=[Path(non_utf8_doc)],
            )
            h2 = _compute_snapshot_hash(snap2.repo_snapshots, snap2.docs_dir)
            self.assertNotEqual(h1, h2)
            self.assertIn("line2", snap2.round_diff)

            # astra-74: symlink included file refused by descriptor-based staging
            os.symlink(parent / "calc.py", parent / "sym_untracked.py")
            with self.assertRaises(SnapshotError):
                create_round_snapshot(
                    run_dir,
                    3,
                    [RepoSpec("parent", parent)],
                    included_files={"sym_untracked.py", "sub/extra.txt"},
                    excluded_files=set(),
                    profile_excluded_submodules={},
                )

    def test_astra_36_45_47_51_52(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # astra-45: length-delimited snapshot hash distinguishes shifted doc boundaries
            d1 = root / "d1"
            d2 = root / "d2"
            d1.mkdir(); d2.mkdir()
            (d1 / "a.md").write_text("ab", encoding="utf-8")
            (d1 / "b.md").write_text("c", encoding="utf-8")
            (d2 / "a.md").write_text("a", encoding="utf-8")
            (d2 / "b.md").write_text("bc", encoding="utf-8")
            self.assertNotEqual(_compute_snapshot_hash({}, d1), _compute_snapshot_hash({}, d2))

            repo = _make_git_repo(root / "repo")
            profile_ev_all = ProfileConfig(
                tags={"evtag": TagRule(name="evtag", vendors=("openai", "google"))}
            )
            cfg = _make_cfg(
                root / "state",
                reviewers=(
                    ReviewerConfig("astra", "codex", "m1", "gpt", "openai"),
                    ReviewerConfig("carbon", "agy", "m2", "gemini", "google"),
                    ReviewerConfig("opus", "claude", "m3", "claude", "anthropic", extra=True),
                    ReviewerConfig("luna", "codex", "m4", "gpt", "openai", extra=True),
                ),
                policy=PolicyConfig(min_families=2),
                profile=profile_ev_all,
            )
            clock = FakeClock()

            def astra_fn(req):
                if req.round_num == 1:
                    return TurnResult(session_id="s-a", reply=_reply(findings=[_finding("a-1", "Bug 1")]))
                return TurnResult(session_id="s-a", reply=_reply())

            def carbon_fn(req):
                if req.round_num == 1:
                    return TurnResult(
                        session_id="s-c",
                        reply=_reply(findings=[
                            _finding("c-1", title="dup1"),
                            _finding("c-2", title="dup2"),
                        ]),
                    )
                return TurnResult(
                    session_id="s-c",
                    reply=_reply(responses=[{"ledger_id": f1_id, "verdict": "concede", "argument": "yes"}]),
                )

            adapters = {
                "codex": FakeAdapter("codex", astra_fn),
                "agy": FakeAdapter("agy", carbon_fn),
                "claude": FakeAdapter("claude", lambda req: TurnResult(session_id="s-o", reply=_reply())),
            }
            run_dir, _ = init_run(
                cfg, "r-multi", [RepoSpec("repo", repo)], "Multi merge",
                explicit_reviewers=["astra", "carbon"],
                no_sandbox=True, adapter_lookup=adapters.__getitem__, now_fn=clock,
            )
            ev = root / "ev.txt"
            ev.write_text("secret trace", encoding="utf-8")
            attach_evidence(run_dir, ev, "unit test query", tags=["evtag"])

            run_round(run_dir, cfg, adapter_lookup=adapters.__getitem__, now_fn=clock)
            ledger = load_ledger(run_dir)
            by_key = {f.found_by[0].key: f.ledger_id for f in ledger.findings}
            f1_id = by_key["a-1"]

            # astra-36: merging two findings from same reviewer preserves both in pending_merges
            merge_findings(run_dir, by_key["c-1"], f1_id, now_fn=clock)
            merge_findings(run_dir, by_key["c-2"], f1_id, now_fn=clock)
            ledger = load_ledger(run_dir)
            f1 = ledger.get_finding(f1_id)
            carbon_fr = next(fr for fr in f1.found_by if fr.reviewer == "carbon")
            self.assertEqual(len(carbon_fr.pending_merges), 2)
            triage_finding(run_dir, f1_id, "fixed", note="fixed", now_fn=clock)

            # astra-47: narrow evtag visibility to google-only (revoking openai/astra) before R2
            # -> run_round must reset astra's session_id and wipe astra's state_dir before invoking R2
            (run_dir / "state" / "astra").mkdir(parents=True, exist_ok=True)
            sentinel = run_dir / "state" / "astra" / "stale_cache.txt"
            sentinel.write_text("sensitive", encoding="utf-8")
            cfg_narrow = _make_cfg(
                root / "state",
                reviewers=cfg.reviewers,
                policy=cfg.policy,
                profile=ProfileConfig(tags={"evtag": TagRule(name="evtag", vendors=("google",))}),
            )
            saw_session_ids = []
            orig_astra_fn = adapters["codex"]._handler

            def tracking_astra(req):
                saw_session_ids.append((req.round_num, req.session_id, sentinel.exists()))
                return TurnResult(
                    session_id="s-a-fresh",
                    reply=_reply(responses=[{"ledger_id": f1_id, "verdict": "fix_ok", "argument": "ok"}]),
                )

            adapters["codex"]._handler = tracking_astra
            run_round(run_dir, cfg_narrow, adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(saw_session_ids, [(2, None, False)], "revoked evidence must clear session_id & wipe state")
            adapters["codex"]._handler = orig_astra_fn

            # astra-51: incremental substitution across two quota-paused cores
            ledger = load_ledger(run_dir)
            ledger.status = "paused"
            ledger.pause_causes = ["quota:astra", "quota:carbon", "quorum:need 2"]
            ledger.pause_reason = "quota + quorum"
            ledger.reviewers["astra"].status = "quota_paused"
            ledger.reviewers["carbon"].status = "quota_paused"
            save_ledger(run_dir, ledger)

            # First substitution: astra -> luna (still paused because carbon is quota_paused and quorum=1<2)
            ledger = substitute_reviewer(run_dir, cfg, "astra", "luna", adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(ledger.status, "paused")
            self.assertTrue(any(c.startswith("quorum:") for c in ledger.pause_causes))
            # Substituting already-substituted astra again must fail
            with self.assertRaises(ProtocolError):
                substitute_reviewer(run_dir, cfg, "astra", "opus", adapter_lookup=adapters.__getitem__, now_fn=clock)
            # Second substitution: carbon -> opus (luna[gpt] + opus[claude] = 2 families -> unpaused!)
            ledger = substitute_reviewer(run_dir, cfg, "carbon", "opus", adapter_lookup=adapters.__getitem__, now_fn=clock)
            self.assertEqual(ledger.status, "needs_triage")

            # astra-52: symlinked run_dir/state refused in clean_run
            real_outside = root / "outside_precious"
            real_outside.mkdir()
            (real_outside / "keep.txt").write_text("precious", encoding="utf-8")
            state_sub = run_dir / "state"
            if state_sub.exists():
                shutil.rmtree(state_sub)
            os.symlink(real_outside, state_sub)
            with self.assertRaises(ProtocolError):
                clean_run(cfg, "r-multi", adapter_lookup=adapters.__getitem__)
            self.assertTrue((real_outside / "keep.txt").exists(), "clean_run must never follow symlinked state dir")


if __name__ == "__main__":
    unittest.main()
