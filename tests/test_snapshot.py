"""Tests for the read-only git snapshot engine (submodules, symlinks, worktrees, non-UTF-8, staging)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from xreview.config import ConfigError
from xreview.snapshot import (
    RepoSpec,
    SnapshotError,
    UnclassifiedFilesError,
    create_round_snapshot,
    find_overlapping_fix_lines,
    parse_diff_line_ranges,
)


def _git(args: list[str], cwd: Path) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_ALLOW_PROTOCOL": "file",
    }
    res = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return res.stdout


def _list_objects(git_dir: Path) -> list[str]:
    obj = git_dir / "objects"
    return sorted(str(p.relative_to(obj)) for p in obj.rglob("*") if p.is_file())


class SnapshotTests(unittest.TestCase):
    def _make_repo_with_submodule(self, base: Path) -> tuple[Path, Path]:
        sub_origin = base / "sub-origin"
        sub_origin.mkdir()
        _git(["init", "-b", "main"], sub_origin)
        (sub_origin / "sub.py").write_text("x = 1\n", encoding="utf-8")
        _git(["add", "."], sub_origin)
        _git(["commit", "-m", "sub init"], sub_origin)

        repo = base / "repo"
        repo.mkdir()
        _git(["init", "-b", "main"], repo)
        (repo / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (repo / "obsolete.py").write_text("old = True\n", encoding="utf-8")
        (repo / ".gitignore").write_text("build.out\n", encoding="utf-8")
        _git(["submodule", "add", str(sub_origin), "sub"], repo)
        _git(["add", "."], repo)
        _git(["commit", "-m", "init"], repo)
        return repo, sub_origin

    def test_zero_writes_submodule_diff_and_deleted_file_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo, _ = self._make_repo_with_submodule(base)
            run_dir = base / "run"
            run_dir.mkdir()

            main_gd = repo / ".git"
            sub_gd = repo / ".git" / "modules" / "sub"
            objs_before = _list_objects(main_gd)
            sub_objs_before = _list_objects(sub_gd)
            idx_before = (main_gd / "index").read_bytes()
            sub_idx_before = (sub_gd / "index").read_bytes()

            # Clean repo with initialized submodule must FAIL empty-diff guard (astra-19)!
            spec = [RepoSpec(name="repo", path=repo, base_rev="HEAD")]
            with self.assertRaises(SnapshotError):
                create_round_snapshot(run_dir, 1, spec, set(), set(), {})
            # Staging rollback must allow retrying round 1 cleanly in the SAME run_dir (astra-24)!
            self.assertFalse((run_dir / "r1" / "tree").exists())

            # Round 1 changes
            (repo / "app.py").write_text("def f():\n    return 2\n", encoding="utf-8")
            (repo / "new_mod.py").write_text("a = 10\n", encoding="utf-8")
            (repo / "build.out").write_text("ignored\n", encoding="utf-8")
            (repo / "sub" / "sub.py").write_text("x = 2\n", encoding="utf-8")

            with self.assertRaises(UnclassifiedFilesError) as ctx:
                create_round_snapshot(run_dir, 1, spec, set(), set(), {})
            self.assertEqual(ctx.exception.files, ["new_mod.py"])

            r1 = create_round_snapshot(
                run_dir, 1, spec, {"new_mod.py"}, set(), {"repo": {"sub"}}
            )
            self.assertIn("x = 2", r1.base_diff)
            self.assertTrue((r1.tree_dir / "sub" / "sub.py").exists())
            self.assertEqual((r1.tree_dir / "sub" / "sub.py").read_text(), "x = 2\n")
            self.assertTrue((r1.tree_dir / "obsolete.py").exists())
            self.assertFalse((r1.tree_dir / "build.out").exists())

            # Round 2: delete obsolete.py AND delete previously included new_mod.py (astra-11)!
            (repo / "obsolete.py").unlink()
            (repo / "new_mod.py").unlink()
            (repo / "app.py").write_text("def f():\n    return 3\n", encoding="utf-8")
            # Submodule unchanged in round 2 -> should NOT reappear in ROUND.diff (astra-19)!
            r2 = create_round_snapshot(
                run_dir, 2, spec, {"new_mod.py"}, set(), {"repo": {"sub"}}, prev_snapshots=r1.repo_snapshots
            )
            self.assertFalse((r2.tree_dir / "obsolete.py").exists())
            self.assertFalse((r2.tree_dir / "new_mod.py").exists())
            self.assertTrue((r1.tree_dir / "obsolete.py").exists())
            self.assertNotIn("sub/sub.py", r2.round_diff)
            self.assertIn("obsolete.py", r2.round_diff)

            # Verify zero writes to source repo & submodule
            self.assertEqual(_list_objects(main_gd), objs_before)
            self.assertEqual(_list_objects(sub_gd), sub_objs_before)
            self.assertEqual((main_gd / "index").read_bytes(), idx_before)
            self.assertEqual((sub_gd / "index").read_bytes(), sub_idx_before)

    def test_intermediate_symlink_exclusion_cannot_delete_host_files(self) -> None:
        """astra-1: .xreview.toml exclude_paths must filter in git index, never unlink via host symlinks."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            victim_dir = base / "victim"
            victim_dir.mkdir()
            victim_file = victim_dir / "precious.txt"
            victim_file.write_text("DO NOT DELETE", encoding="utf-8")

            repo = base / "repo"
            repo.mkdir()
            _git(["init", "-b", "main"], repo)
            os.symlink(str(victim_dir), repo / "linkdir")
            (repo / "a.py").write_text("a = 1\n")
            (repo / ".xreview.toml").write_text('exclude_paths = ["linkdir/precious.txt"]\n', encoding="utf-8")
            _git(["add", "."], repo)
            _git(["commit", "-m", "init"], repo)
            (repo / "a.py").write_text("a = 2\n")

            create_round_snapshot(base / "run", 1, [RepoSpec("repo", repo)], set(), set(), {})
            self.assertTrue(victim_file.exists(), "Host file outside snapshot was deleted!")

    def test_standalone_docs_use_separate_mount_and_refuse_symlinks(self) -> None:
        """astra-5 & astra-6: docs go to r1/docs (never tree/_docs) and symlinks are refused."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "repo"
            repo.mkdir()
            _git(["init", "-b", "main"], repo)
            (repo / "a.py").write_text("a = 1\n")
            (repo / "_docs").mkdir()
            (repo / "_docs" / "design.md").write_text("repo file\n")
            _git(["add", "."], repo)
            _git(["commit", "-m", "init"], repo)
            (repo / "a.py").write_text("a = 2\n")

            real_doc = base / "design.md"
            real_doc.write_text("standalone doc\n")
            sym_doc = base / "sym.md"
            os.symlink(real_doc, sym_doc)

            with self.assertRaises(SnapshotError):
                create_round_snapshot(base / "run1", 1, [RepoSpec("repo", repo)], set(), set(), {}, docs=[sym_doc])

            r1 = create_round_snapshot(base / "run2", 1, [RepoSpec("repo", repo)], set(), set(), {}, docs=[real_doc])
            self.assertIsNotNone(r1.docs_dir)
            assert r1.docs_dir is not None
            self.assertEqual((r1.docs_dir / "design.md").read_text(), "standalone doc\n")
            self.assertEqual((r1.tree_dir / "_docs" / "design.md").read_text(), "repo file\n")

    def test_linked_worktree_and_non_utf8_diff(self) -> None:
        """astra-21 & astra-46: linked git worktree uses common-dir objects and handles non-UTF-8 bytes."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "main-repo"
            repo.mkdir()
            _git(["init", "-b", "main"], repo)
            (repo / "legacy.txt").write_bytes(b"hello \xff\xfe v1\n")
            _git(["add", "."], repo)
            _git(["commit", "-m", "init"], repo)

            wt = base / "linked-wt"
            _git(["worktree", "add", "-b", "feat", str(wt)], repo)
            (wt / "legacy.txt").write_bytes(b"hello \xff\xfe v2\n")

            r1 = create_round_snapshot(base / "run", 1, [RepoSpec("wt", wt)], set(), set(), {})
            self.assertIn(b"v2", r1.base_diff_bytes)

    def test_refuses_secret_and_oversized_untracked_even_when_included(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "repo"
            repo.mkdir()
            _git(["init", "-b", "main"], repo)
            (repo / "a.py").write_text("a = 1\n")
            _git(["add", "."], repo)
            _git(["commit", "-m", "init"], repo)

            (repo / ".envrc").write_text("export SECRET=1\n")
            spec = [RepoSpec(name="repo", path=repo)]
            with self.assertRaises(SnapshotError):
                create_round_snapshot(base / "run1", 1, spec, {".envrc"}, set(), {})

            (repo / ".envrc").unlink()
            (repo / "big.bin").write_bytes(b"0" * (1_048_576 + 1))
            with self.assertRaises(SnapshotError):
                create_round_snapshot(base / "run2", 1, spec, {"big.bin"}, set(), {})

    def test_refuses_symlink_in_project_toml_and_never_excludes_changed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo = base / "repo"
            repo.mkdir()
            _git(["init", "-b", "main"], repo)
            (repo / "a.py").write_text("a = 1\n")
            (repo / "unchanged.txt").write_text("stay\n")
            os.symlink("/etc/passwd", repo / "evil.md")
            (repo / ".xreview.toml").write_text(
                'exclude_paths = ["a.py", "unchanged.txt"]\n'
                'instruction_files = ["evil.md"]\n',
                encoding="utf-8",
            )
            _git(["add", "."], repo)
            _git(["commit", "-m", "init"], repo)

            (repo / "a.py").write_text("a = 2\n")
            spec = [RepoSpec(name="repo", path=repo)]

            run_dir = base / "run1"
            with self.assertRaises(ConfigError):
                create_round_snapshot(run_dir, 1, spec, set(), set(), {})

            # Fix .xreview.toml and retry in the SAME run_dir (astra-24)!
            (repo / ".xreview.toml").write_text(
                'notes = "repo note"\nexclude_paths = ["a.py", "unchanged.txt"]\n',
                encoding="utf-8",
            )
            r1 = create_round_snapshot(run_dir, 1, spec, set(), set(), {})
            self.assertTrue((r1.tree_dir / "a.py").exists())
            self.assertFalse((r1.tree_dir / "unchanged.txt").exists())
            self.assertIn("repo:unchanged.txt", r1.project_exclusions)

    def test_overlapping_fix_line_hints(self) -> None:
        d1 = "--- a/app.py\n+++ b/app.py\n@@ -1,3 +10,5 @@\n+x\n"
        d2 = "--- a/app.py\n+++ b/app.py\n@@ -1,3 +12,2 @@\n+y\n"
        hints = find_overlapping_fix_lines([parse_diff_line_ranges(d1)], parse_diff_line_ranges(d2))
        self.assertEqual(len(hints), 1)
        self.assertIn("app.py:12-13 overlaps lines 10-14 changed in round 1", hints[0])


if __name__ == "__main__":
    unittest.main()
