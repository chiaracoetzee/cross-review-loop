"""Tests for adapter parsing helpers, strict CLI parsing (no abbrev, no CWD impersonation), and setup-gates."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from xreview.adapters.base import detect_quota_exhausted, parse_json_from_text
from xreview.adapters.codex import _read_codex_rollout_rate_limit
from xreview.cli import build_parser, main


class AdapterAndCliTests(unittest.TestCase):
    def test_detect_quota_exhausted(self) -> None:
        self.assertTrue(detect_quota_exhausted("Error: usage limit reached for your plan"))
        self.assertTrue(detect_quota_exhausted("RESOURCE_EXHAUSTED: quota exceeded"))
        self.assertTrue(detect_quota_exhausted("HTTP 429 rate limit hit"))
        self.assertFalse(detect_quota_exhausted("SyntaxError: invalid syntax"))

    def test_parse_json_from_text(self) -> None:
        self.assertEqual(parse_json_from_text('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_from_text("```json\n{\"a\": 2}\n```"), {"a": 2})
        self.assertIsNone(parse_json_from_text("not json"))

    def test_read_codex_rollout_rate_limit_ignores_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            codex_home = Path(td)
            sess_dir = codex_home / "sessions" / "2026" / "11" / "23"
            sess_dir.mkdir(parents=True)
            rollout = sess_dir / "rollout-2026-11-23-abc-123.jsonl"
            rollout.write_text(
                json.dumps({
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "rate_limits": {
                            "primary": {"used_percent": 42.0},
                            "secondary": {"used_percent": 85.0},
                            "rate_limit_reached_type": None,
                        },
                    },
                }) + "\n",
                encoding="utf-8",
            )
            ratio, reached = _read_codex_rollout_rate_limit(codex_home, "abc-123")
            self.assertAlmostEqual(ratio or 0.0, 0.85)
            self.assertFalse(reached)

            # Symlinked rollout must be ignored (astra-4)
            os.symlink(rollout, sess_dir / "rollout-2026-11-23-sym-999.jsonl")
            ratio_sym, _ = _read_codex_rollout_rate_limit(codex_home, "sym-999")
            self.assertIsNone(ratio_sym)

    def test_cli_disables_flag_abbreviations_and_cwd_impersonation_and_evidence_symlinks(self) -> None:
        """astra-2, astra-6, astra-13."""
        parser = build_parser()
        # Abbreviated --rev or --al MUST be rejected so harness ask rules always match literal flags!
        with self.assertRaises(SystemExit):
            parser.parse_args(["init", "--rev", "astra"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["init", "--al", "security=openai"])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "T",
                "GIT_AUTHOR_EMAIL": "t@e.com",
                "GIT_COMMITTER_NAME": "T",
                "GIT_COMMITTER_EMAIL": "t@e.com",
            }
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, capture_output=True, check=True)
            (repo / "a.py").write_text("x = 1\n")
            subprocess.run(["git", "add", "."], cwd=repo, env=env, capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, env=env, capture_output=True, check=True)
            (repo / "a.py").write_text("x = 2\n")
            (repo / "untracked.py").write_text("y = 1\n")

            # Fake run directory inside CWD MUST NOT impersonate a real run (astra-2)!
            fake_run = repo / "impostor"
            fake_run.mkdir()
            (fake_run / "ledger.json").write_text('{"run_id": "impostor"}', encoding="utf-8")

            cfg = root / "config.toml"
            cfg.write_text(
                f'state_dir = "{root / "state"}"\n'
                'reviewers = [\n'
                '  { name = "astra", harness = "codex", model = "m1", family = "gpt" },\n'
                '  { name = "carbon", harness = "agy", model = "m2", family = "gemini" },\n'
                ']\n',
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(repo)
                self.assertEqual(main(["--config", str(cfg), "status", "impostor"]), 1)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(main(["--config", str(cfg), "doctor", "--json"]), 0)
            self.assertEqual(
                main(["--config", str(cfg), "init", "--run-id", "cli-1", "--repo", str(repo), "--no-sandbox", "--json"]),
                0,
            )
            # Evidence symlink MUST be rejected (astra-6)
            secret = root / "secret.txt"
            secret.write_text("topsecret")
            sym_ev = root / "ev_link.txt"
            os.symlink(secret, sym_ev)
            self.assertEqual(
                main(["--config", str(cfg), "evidence", "cli-1", "--file", str(sym_ev), "--source", "cmd"]),
                1,
            )

            self.assertEqual(main(["--config", str(cfg), "classify", "cli-1", "--include", "untracked.py"]), 0)
            self.assertEqual(main(["--config", str(cfg), "status", "cli-1", "--json"]), 0)
            self.assertEqual(main(["--config", str(cfg), "ledger", "cli-1"]), 0)
            self.assertEqual(main(["--config", str(cfg), "clean", "cli-1"]), 0)
            self.assertFalse((root / "state" / "cli-1").exists())

            # Test setup-gates check + install
            settings = root / "claude_settings.json"
            self.assertEqual(main(["setup-gates", "--claude-settings", str(settings), "--check"]), 1)
            self.assertEqual(main(["setup-gates", "--claude-settings", str(settings)]), 0)
            self.assertEqual(main(["setup-gates", "--claude-settings", str(settings), "--check"]), 0)


if __name__ == "__main__":
    unittest.main()
