"""Sensitivity tags, gates, vendor/family separation, and evidence visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from xreview.config import AppConfig, ConfigError, ReviewerConfig, TagRule


class PolicyViolation(ValueError):
    """Raised when a run violates sensitivity tags, gates, or quorum."""


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    filename: str
    host_path: str
    source: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "host_path": self.host_path,
            "source": self.source,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceItem:
        return cls(
            id=str(data["id"]),
            filename=str(data["filename"]),
            host_path=str(data["host_path"]),
            source=str(data["source"]),
            tags=tuple(str(t) for t in data.get("tags", ())),
        )


@dataclass(frozen=True)
class GateLift:
    kind: str  # "reviewers" or "allow"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> GateLift:
        return cls(kind=str(data["kind"]), detail=str(data["detail"]))


@dataclass(frozen=True)
class ResolvedRunPolicy:
    selected_reviewers: tuple[ReviewerConfig, ...]
    run_tags: tuple[str, ...]
    post_run_hooks_enabled: bool
    gates_lifted: tuple[GateLift, ...] = ()
    approved_ask_tags: tuple[str, ...] = ()
    allow_map: dict[str, tuple[str, ...]] = field(default_factory=dict)


def parse_allow_flags(raw_allows: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Parse --allow <tag>=<vendor1>,<vendor2> entries."""
    result: dict[str, list[str]] = {}
    for entry in raw_allows:
        if "=" not in entry:
            raise ConfigError(f"Invalid --allow format {entry!r}; expected <tag>=<vendor1>,<vendor2>")
        tag, _, vendors_s = entry.partition("=")
        tag = tag.strip()
        if not tag:
            raise ConfigError(f"Empty tag in --allow {entry!r}")
        vendors = [v.strip().lower() for v in vendors_s.split(",") if v.strip()]
        if not vendors:
            raise ConfigError(f"No vendors given in --allow {entry!r}")
        result.setdefault(tag, [])
        for v in vendors:
            if v not in result[tag]:
                result[tag].append(v)
    return {k: tuple(v) for k, v in result.items()}


def _tag_rule_for(tag: str, rules: dict[str, TagRule]) -> TagRule:
    if tag in rules:
        return rules[tag]
    # Unknown sensitivity tag defaults to strictest safe policy: ask before sharing.
    return TagRule(name=tag, vendors="ask", post_run_hooks=False)


def paths_overlap(artifact_path: str, rule_path: str) -> bool:
    """Return True if artifact_path equals, lies within, or contains rule_path.

    Checks both directions so reviewing a parent checkout that contains a protected
    subtree (or reviewing a doc inside a protected directory) triggers the rule.
    """
    a = str(Path(artifact_path)).rstrip("/") or "/"
    r = str(Path(rule_path)).rstrip("/") or "/"
    if a == r:
        return True
    if r == "/" or a == "/":
        return True
    return a.startswith(r + "/") or r.startswith(a + "/")


def tags_for_paths(config: AppConfig, artifact_paths: Iterable[str]) -> list[str]:
    """Return all profile tags matched by any artifact path (repos, submodules, docs)."""
    tag_list: list[str] = []
    for ap in artifact_paths:
        for rule in config.profile.repo_rules:
            if paths_overlap(ap, rule.path):
                for t in rule.tags:
                    if t not in tag_list:
                        tag_list.append(t)
    return tag_list


def enforce_tags_on_active_reviewers(
    config: AppConfig,
    run_tags: Iterable[str],
    reviewers: Iterable[ReviewerConfig],
    gates_lifted_for_tags: set[str],
) -> tuple[bool, list[str]]:
    """Re-verify that a set of tags (including newly discovered paths) is allowed for active reviewers.

    Returns (post_run_hooks_enabled, new_unapproved_ask_tags).
    Raises PolicyViolation if any reviewer violates a fixed vendor list.
    """
    fixed_vendor_intersection: set[str] | None = None
    unapproved_ask: list[str] = []
    hooks_enabled = True

    for tag in run_tags:
        rule = _tag_rule_for(tag, config.profile.tags)
        if not rule.post_run_hooks:
            hooks_enabled = False
        if rule.vendors == "ask":
            if tag not in gates_lifted_for_tags:
                unapproved_ask.append(tag)
        else:
            allowed_set = set(rule.vendors)
            if fixed_vendor_intersection is None:
                fixed_vendor_intersection = allowed_set
            else:
                fixed_vendor_intersection &= allowed_set

    if fixed_vendor_intersection is not None:
        disallowed = [r for r in reviewers if r.vendor not in fixed_vendor_intersection]
        if disallowed:
            names = ", ".join(f"{r.name} (vendor={r.vendor})" for r in disallowed)
            raise PolicyViolation(
                f"Reviewer(s) forbidden by fixed tag vendor policy ({sorted(fixed_vendor_intersection)}): {names}"
            )

    return hooks_enabled, unapproved_ask


def resolve_run_policy(
    config: AppConfig,
    artifact_paths: list[str],
    cli_tags: Iterable[str],
    explicit_reviewers: list[str] | None,
    allow_map: dict[str, tuple[str, ...]],
    implementor_family: str | None = None,
) -> ResolvedRunPolicy:
    """Resolve active reviewers, tags, gates, hooks, and quorum before starting a run."""
    enabled = [r for r in config.reviewers if r.enabled]
    if not enabled:
        raise PolicyViolation("No enabled reviewers configured.")

    tag_list: list[str] = []
    for t in cli_tags:
        t = t.strip()
        if t and t not in tag_list:
            tag_list.append(t)
    for t in tags_for_paths(config, artifact_paths):
        if t not in tag_list:
            tag_list.append(t)

    fixed_vendor_intersection: set[str] | None = None
    requires_ask = False
    ask_tags: list[str] = []
    hooks_enabled = True

    for tag in tag_list:
        rule = _tag_rule_for(tag, config.profile.tags)
        if not rule.post_run_hooks:
            hooks_enabled = False
        if rule.vendors == "ask":
            requires_ask = True
            ask_tags.append(tag)
        else:
            allowed_set = set(rule.vendors)
            if fixed_vendor_intersection is None:
                fixed_vendor_intersection = allowed_set
            else:
                fixed_vendor_intersection &= allowed_set

    gates_lifted: list[GateLift] = []

    if explicit_reviewers is not None:
        by_name = {r.name: r for r in enabled}
        unknown = [n for n in explicit_reviewers if n not in by_name]
        if unknown:
            raise PolicyViolation(f"Unknown or disabled reviewer(s): {', '.join(unknown)}")
        selected = [by_name[n] for n in explicit_reviewers]
        if requires_ask:
            gates_lifted.append(
                GateLift(
                    kind="reviewers",
                    detail=f"tags={','.join(ask_tags)} reviewers={','.join(r.name for r in selected)}",
                )
            )
    else:
        if requires_ask:
            raise PolicyViolation(
                f"Run carries gated tag(s) ({', '.join(ask_tags)}) with vendors='ask'. "
                "A person must approve and pass --reviewers <names> explicitly."
            )
        selected = list(enabled)

    # Fixed vendor lists can NEVER be widened by --reviewers or --allow
    if fixed_vendor_intersection is not None:
        disallowed = [r for r in selected if r.vendor not in fixed_vendor_intersection]
        if disallowed:
            if explicit_reviewers is not None:
                names = ", ".join(f"{r.name} (vendor={r.vendor})" for r in disallowed)
                raise PolicyViolation(
                    f"Reviewer(s) forbidden by fixed tag vendor policy ({sorted(fixed_vendor_intersection)}): {names}"
                )
            selected = [r for r in selected if r.vendor in fixed_vendor_intersection]

    for tag, vendors in sorted(allow_map.items()):
        gates_lifted.append(GateLift(kind="allow", detail=f"{tag}={','.join(vendors)}"))

    if not selected:
        raise PolicyViolation("No reviewers remain after applying tag vendor rules.")

    check_quorum(selected, config.policy.min_families, implementor_family or config.implementor_family)

    return ResolvedRunPolicy(
        selected_reviewers=tuple(selected),
        run_tags=tuple(tag_list),
        post_run_hooks_enabled=hooks_enabled,
        gates_lifted=tuple(gates_lifted),
        approved_ask_tags=tuple(ask_tags) if (explicit_reviewers is not None and requires_ask) else (),
        allow_map=dict(allow_map),
    )


def check_quorum(
    reviewers: Iterable[ReviewerConfig],
    min_families: int,
    implementor_family: str | None = None,
) -> None:
    """Verify distinct model families meet min_families and include a non-implementor family."""
    families = {r.family for r in reviewers}
    if len(families) < min_families:
        raise PolicyViolation(
            f"Quorum not met: need at least {min_families} distinct model families, "
            f"have {len(families)} ({sorted(families)})."
        )
    if implementor_family:
        impl = implementor_family.strip().lower()
        other_families = families - {impl}
        if not other_families:
            raise PolicyViolation(
                f"Quorum not met: all active reviewers share the implementor's model family ({impl!r})."
            )


def evidence_visible_to_reviewer(
    evidence: EvidenceItem,
    reviewer: ReviewerConfig,
    tag_rules: dict[str, TagRule],
    allow_map: dict[str, tuple[str, ...]],
) -> bool:
    """Decide whether a reviewer's data vendor may see a given evidence file."""
    vendor = reviewer.vendor.lower()
    for tag in evidence.tags:
        rule = _tag_rule_for(tag, tag_rules)
        if rule.vendors == "ask":
            allowed = allow_map.get(tag, ())
            if vendor not in allowed:
                return False
        else:
            if vendor not in rule.vendors:
                return False
    return True


def visible_evidence_ids(
    evidence_items: Iterable[EvidenceItem],
    reviewer: ReviewerConfig,
    tag_rules: dict[str, TagRule],
    allow_map: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    return frozenset(
        ev.id
        for ev in evidence_items
        if evidence_visible_to_reviewer(ev, reviewer, tag_rules, allow_map)
    )


def should_redact_finding_for_recipient(
    source_evidence_ids: Iterable[str],
    recipient_current_visibility: frozenset[str],
) -> bool:
    """Return True if ANY historical evidence exposed to the finding's source(s)
    is NOT currently authorized for the recipient reviewer.
    """
    src_set = frozenset(source_evidence_ids)
    return not src_set.issubset(recipient_current_visibility)
