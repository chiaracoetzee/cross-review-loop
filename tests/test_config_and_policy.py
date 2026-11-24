"""Tests for layered config, untrusted .xreview.toml, sensitivity tags, gates, and quorum."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from xreview.config import (
    AppConfig,
    ConfigError,
    PolicyConfig,
    ProfileConfig,
    RepoProfileRule,
    ReviewerConfig,
    TagRule,
    is_secret_path,
    load_config,
    parse_untrusted_project_toml,
    safe_open_user_regular_file,
    safe_read_regular_file_under,
    validate_repo_alias,
    validate_run_id,
)
from xreview.policy import (
    EvidenceItem,
    PolicyViolation,
    evidence_visible_to_reviewer,
    parse_allow_flags,
    resolve_run_policy,
    should_redact_finding_for_recipient,
    visible_evidence_ids,
)


def _sample_config() -> AppConfig:
    reviewers = (
        ReviewerConfig("astra", "codex", "gpt-6.2-astra", "gpt", "openai", "high"),
        ReviewerConfig("carbon", "agy", "gemini-4-carbon-high", "gemini", "google"),
        ReviewerConfig("fable", "claude", "claude-fable-5-2", "claude", "anthropic", "high"),
        ReviewerConfig("agy-claude", "agy", "claude-opus-5-5-high", "claude", "google", extra=True),
    )
    profile = ProfileConfig(
        tags={
            "security": TagRule("security", vendors="ask", post_run_hooks=False),
            "anthropic-only": TagRule("anthropic-only", vendors=("anthropic",), post_run_hooks=False),
            "personal-data": TagRule("personal-data", vendors="ask", post_run_hooks=False),
            "private": TagRule("private", vendors="ask", post_run_hooks=False),
        },
        repo_rules=(
            RepoProfileRule(path="/repos/private-tools", tags=("private",)),
            RepoProfileRule(path="/repos/workspace/secret-sub", tags=("private",)),
        ),
    )
    return AppConfig(
        reviewers=reviewers,
        policy=PolicyConfig(min_families=2),
        profile=profile,
    )


class ConfigAndPolicyTests(unittest.TestCase):
    def test_load_toml_config_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prof = root / "profile.toml"
            prof.write_text(
                'brief_rules = ["Rule 1"]\n'
                'extra_secret_patterns = ["*.vault"]\n'
                '[tags.security]\nvendors = "ask"\npost_run_hooks = false\n',
                encoding="utf-8",
            )
            cfg = root / "config.toml"
            cfg.write_text(
                f'profile = "{prof}"\n'
                'implementor_family = "claude"\n'
                'reviewers = [\n'
                '  { name = "astra", harness = "codex", model = "m1", family = "gpt" },\n'
                '  { name = "carbon", harness = "agy", model = "m2", family = "gemini" },\n'
                ']\n'
                '[policy]\nmin_families = 2\nsoft_round_ceiling = 5\n',
                encoding="utf-8",
            )
            loaded = load_config(cfg)
            self.assertEqual(len(loaded.reviewers), 2)
            self.assertEqual(loaded.reviewers[0].vendor, "openai")
            self.assertEqual(loaded.reviewers[1].vendor, "google")
            self.assertEqual(loaded.policy.soft_round_ceiling, 5)
            self.assertEqual(loaded.implementor_family, "claude")
            self.assertEqual(loaded.profile.brief_rules, ("Rule 1",))
            self.assertTrue(is_secret_path("foo/prod.vault", loaded.all_secret_patterns()))

    def test_validate_run_id_and_repo_alias(self) -> None:
        self.assertEqual(validate_run_id("run-2026_1.0"), "run-2026_1.0")
        for bad in ("", ".", "..", "../escape", "/abs", "a/b"):
            with self.assertRaises(ConfigError):
                validate_run_id(bad)
        for bad_alias in ("", ".", "..", "_docs", "tree", "a/b"):
            with self.assertRaises(ConfigError):
                validate_repo_alias(bad_alias)

    def test_safe_file_helpers_refuse_symlinks_and_fifos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            state = root / "state"
            state.mkdir()
            (state / "sub").mkdir()
            os.symlink(outside, state / "sub" / "link.txt")
            os.symlink(root, state / "dirlink")
            os.mkfifo(state / "pipe")

            # Leaf symlink, parent symlink, and FIFO are all refused/None
            self.assertIsNone(safe_read_regular_file_under(state, "sub/link.txt"))
            self.assertIsNone(safe_read_regular_file_under(state, "dirlink/outside.txt"))
            self.assertIsNone(safe_read_regular_file_under(state, "pipe"))

            (state / "ok.txt").write_text("good", encoding="utf-8")
            self.assertEqual(safe_read_regular_file_under(state, "ok.txt"), b"good")

            with self.assertRaises(ConfigError):
                safe_open_user_regular_file(state / "sub" / "link.txt")

    def test_untrusted_project_toml_rejects_disallowed_keys_and_traversal(self) -> None:
        valid = parse_untrusted_project_toml(
            b'notes = "Check perf"\ninstruction_files = ["docs/GUIDE.md"]\nexclude_paths = ["vendor/gen"]\n'
        )
        self.assertEqual(valid.notes, "Check perf")
        self.assertEqual(valid.instruction_files, ("docs/GUIDE.md",))

        with self.assertRaises(ConfigError):
            parse_untrusted_project_toml(b'hooks = { post_run = ["evil"] }\n')
        with self.assertRaises(ConfigError):
            parse_untrusted_project_toml(b'instruction_files = ["../secret.txt"]\n')
        with self.assertRaises(ConfigError):
            parse_untrusted_project_toml(b'exclude_paths = ["/etc/passwd"]\n')

    def test_secret_patterns_include_full_env_prefix(self) -> None:
        for bad in (
            ".env",
            ".envrc",
            ".env-production",
            "dir/.env.local",
            "tls.pem",
            "priv.key",
            "id_rsa",
            "id_ed25519.pub",
            "key.properties",
            "my_credentials.json",
        ):
            self.assertTrue(is_secret_path(bad), f"Expected secret: {bad}")
        for ok in ("src/app.py", "README.md", "tests/test_auth.py"):
            self.assertFalse(is_secret_path(ok), f"Unexpected secret: {ok}")

    def test_ask_tag_requires_explicit_reviewers_and_records_gate_lift(self) -> None:
        cfg = _sample_config()
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg, ["/repos/pub"], ["security"], None, {})

        resolved = resolve_run_policy(cfg, ["/repos/pub"], ["security"], ["astra", "fable"], {})
        self.assertEqual([r.name for r in resolved.selected_reviewers], ["astra", "fable"])
        self.assertFalse(resolved.post_run_hooks_enabled)
        self.assertEqual(len(resolved.gates_lifted), 1)
        self.assertEqual(resolved.gates_lifted[0].kind, "reviewers")

    def test_path_sensitivity_matches_parent_repo_containing_protected_subtree_and_docs(self) -> None:
        cfg = _sample_config()
        # Parent repo contains /repos/workspace/secret-sub -> MUST trigger `private` ask gate (astra-9)
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg, ["/repos/workspace"], [], None, {})
        # Standalone doc inside protected repo -> MUST trigger `private` ask gate (astra-9)
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg, ["/repos/private-tools/design.md"], [], None, {})

    def test_fixed_vendor_list_cannot_be_widened(self) -> None:
        cfg = _sample_config()
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg, ["/repos/pub"], ["anthropic-only"], ["astra", "fable"], {})

    def test_quorum_counts_model_family_not_data_vendor_and_requires_non_implementor(self) -> None:
        cfg = _sample_config()
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg, ["/repos/pub"], [], ["fable", "agy-claude"], {})

        cfg1 = AppConfig(reviewers=cfg.reviewers, policy=PolicyConfig(min_families=1), profile=cfg.profile)
        with self.assertRaises(PolicyViolation):
            resolve_run_policy(cfg1, ["/repos/pub"], [], ["astra"], {}, implementor_family="gpt")

    def test_evidence_visibility_by_vendor_and_historical_redaction(self) -> None:
        cfg = _sample_config()
        ev_pii = EvidenceItem("E-01", "E-01_extract.txt", "/tmp/x", "psql ...", tags=("personal-data",))
        ev_pub = EvidenceItem("E-02", "E-02_log.txt", "/tmp/y", "tail ...", tags=())
        allow = parse_allow_flags(["personal-data=anthropic"])

        fable = next(r for r in cfg.reviewers if r.name == "fable")
        agy_claude = next(r for r in cfg.reviewers if r.name == "agy-claude")
        astra = next(r for r in cfg.reviewers if r.name == "astra")

        self.assertTrue(evidence_visible_to_reviewer(ev_pii, fable, cfg.profile.tags, allow))
        self.assertFalse(evidence_visible_to_reviewer(ev_pii, agy_claude, cfg.profile.tags, allow))
        self.assertFalse(evidence_visible_to_reviewer(ev_pii, astra, cfg.profile.tags, allow))
        self.assertTrue(evidence_visible_to_reviewer(ev_pub, astra, cfg.profile.tags, allow))

        astra_vis = visible_evidence_ids([ev_pii, ev_pub], astra, cfg.profile.tags, allow)
        fable_vis = visible_evidence_ids([ev_pii, ev_pub], fable, cfg.profile.tags, allow)

        # Even if profile later removes E-01 from everyone, a finding raised with source_evidence_ids=["E-01"]
        # stays redacted for any recipient whose current visibility lacks E-01 (astra-10)!
        self.assertTrue(should_redact_finding_for_recipient(["E-01", "E-02"], astra_vis))
        self.assertFalse(should_redact_finding_for_recipient(["E-02"], astra_vis))
        self.assertFalse(should_redact_finding_for_recipient(["E-01", "E-02"], fable_vis))
        self.assertTrue(should_redact_finding_for_recipient(["E-01"], frozenset()))


if __name__ == "__main__":
    unittest.main()
