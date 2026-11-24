"""Atomic ledger persistence (ledger.json), interprocess run locking, and Markdown rendering (LEDGER.md)."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from xreview.config import ReviewerConfig
from xreview.policy import EvidenceItem, GateLift
from xreview.schema import severity_max
from xreview.snapshot import RepoRoundSnapshot, RepoSpec


def default_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Acquire an exclusive interprocess flock outside run_dir so locking never creates run_dir (astra-26, astra-57)."""
    locks_dir = run_dir.parent / ".locks"
    locks_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(locks_dir, 0o700)
    lock_path = locks_dir / f"{run_dir.name}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@dataclass
class HistoryEvent:
    event: str
    round_num: int
    actor: str
    timestamp: str
    detail: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event": self.event,
            "round": self.round_num,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "detail": dict(self.detail),
        }
        if self.evidence_ids:
            d["evidence_ids"] = sorted(set(self.evidence_ids))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEvent:
        return cls(
            event=str(data["event"]),
            round_num=int(data["round"]),
            actor=str(data["actor"]),
            timestamp=str(data["timestamp"]),
            detail=dict(data.get("detail", {})),
            evidence_ids=list(data.get("evidence_ids", ())),
        )


@dataclass
class FinderRecord:
    reviewer: str
    family: str
    key: str
    round_raised: int
    severity: str
    confidence: str
    conceded: bool = False
    verified_fix_revision: int = 0
    merged_from: str | None = None
    merge_accepted: bool = True
    pending_merges: list[str] = field(default_factory=list)  # Ordered list of unresolved merged source IDs (astra-36)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "family": self.family,
            "key": self.key,
            "round_raised": self.round_raised,
            "severity": self.severity,
            "confidence": self.confidence,
            "conceded": self.conceded,
            "verified_fix_revision": self.verified_fix_revision,
            "merged_from": self.merged_from,
            "merge_accepted": self.merge_accepted,
            "pending_merges": list(self.pending_merges),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinderRecord:
        vfr = data.get("verified_fix_revision")
        if vfr is None:
            vfr = 1 if data.get("verified_fix") else 0
        merged_from = str(data["merged_from"]) if data.get("merged_from") else None
        merge_accepted = bool(data.get("merge_accepted", True))
        if "pending_merges" in data:
            pending = [str(x) for x in data.get("pending_merges", ())]
        elif not merge_accepted and merged_from:
            pending = [merged_from]
        else:
            pending = []
        return cls(
            reviewer=str(data["reviewer"]),
            family=str(data["family"]),
            key=str(data["key"]),
            round_raised=int(data["round_raised"]),
            severity=str(data["severity"]),
            confidence=str(data["confidence"]),
            conceded=bool(data.get("conceded", False)),
            verified_fix_revision=int(vfr),
            merged_from=merged_from,
            merge_accepted=merge_accepted and not bool(pending),
            pending_merges=pending,
        )


@dataclass
class Finding:
    ledger_id: str
    title: str
    file: str
    line: int
    evidence: str
    failure_scenario: str
    suggested_fix: str
    round_opened: int
    found_by: list[FinderRecord] = field(default_factory=list)
    seconded_by: list[dict[str, Any]] = field(default_factory=list)
    source_evidence_ids: list[str] = field(default_factory=list)  # Cumulative evidence exposed to finders (astra-10)
    state: str = "open"  # open, closed, escalated, unverified, merged
    disposition: str | None = None  # fixed, rejected, deferred, question, conflicts_with, nit_recorded
    disposition_reason: str = ""
    disposition_citation: str = ""
    fix_note: str = ""
    fix_revision: int = 0
    fix_snapshot_hash: str = ""
    deferral_requires_concession: bool = False
    question_to: str | None = None
    question_text: str = ""
    question_used: bool = False
    conflicts_with_id: str | None = None
    proposed_severity: str | None = None
    proposed_severity_reason: str = ""
    merged_into: str | None = None
    replacement_verifier: str | None = None
    replacement_verified_revision: int = 0
    disagreement_rounds: int = 0
    escalation_reason: str = ""
    round_closed: int | None = None
    latest_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[HistoryEvent] = field(default_factory=list)

    @property
    def effective_severity(self) -> str:
        active = [f.severity for f in self.found_by if not f.conceded]
        if not active:
            active = [f.severity for f in self.found_by]
        return severity_max(active)

    def finder_names(self) -> list[str]:
        return [f.reviewer for f in self.found_by]

    def rounds_open(self, current_round: int) -> int:
        end = self.round_closed if self.round_closed is not None else current_round
        return max(1, end - self.round_opened + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "title": self.title,
            "file": self.file,
            "line": self.line,
            "evidence": self.evidence,
            "failure_scenario": self.failure_scenario,
            "suggested_fix": self.suggested_fix,
            "round_opened": self.round_opened,
            "effective_severity": self.effective_severity,
            "found_by": [f.to_dict() for f in self.found_by],
            "seconded_by": list(self.seconded_by),
            "source_evidence_ids": sorted(set(self.source_evidence_ids)),
            "state": self.state,
            "disposition": self.disposition,
            "disposition_reason": self.disposition_reason,
            "disposition_citation": self.disposition_citation,
            "fix_note": self.fix_note,
            "fix_revision": self.fix_revision,
            "fix_snapshot_hash": self.fix_snapshot_hash,
            "deferral_requires_concession": self.deferral_requires_concession,
            "question_to": self.question_to,
            "question_text": self.question_text,
            "question_used": self.question_used,
            "conflicts_with_id": self.conflicts_with_id,
            "proposed_severity": self.proposed_severity,
            "proposed_severity_reason": self.proposed_severity_reason,
            "merged_into": self.merged_into,
            "replacement_verifier": self.replacement_verifier,
            "replacement_verified_revision": self.replacement_verified_revision,
            "disagreement_rounds": self.disagreement_rounds,
            "escalation_reason": self.escalation_reason,
            "round_closed": self.round_closed,
            "latest_responses": dict(self.latest_responses),
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        default_evidence_ids: list[str] | None = None,
        legacy_provenance: bool = False,
    ) -> Finding:
        found_by = [FinderRecord.from_dict(x) for x in data.get("found_by", ())]
        disposition = str(data["disposition"]) if data.get("disposition") else None
        raw_fix_rev = data.get("fix_revision")
        if raw_fix_rev is None:
            fix_rev = 1 if disposition == "fixed" else 0
        else:
            fix_rev = int(raw_fix_rev)
            if disposition == "fixed" and fix_rev == 0:
                fix_rev = 1

        # For legacy ledgers (< v2), conservatively attach all run evidence IDs to findings & responses (astra-10)
        if legacy_provenance and default_evidence_ids:
            src_ev = sorted(set(data.get("source_evidence_ids", ())) | set(default_evidence_ids))
        elif "source_evidence_ids" in data:
            src_ev = list(data.get("source_evidence_ids", ()))
        else:
            src_ev = list(default_evidence_ids or ())

        history = [HistoryEvent.from_dict(h) for h in data.get("history", ())]
        if legacy_provenance and default_evidence_ids:
            for h in history:
                if h.event == "responded" and not h.evidence_ids:
                    h.evidence_ids = list(default_evidence_ids)

        if "deferral_requires_concession" in data:
            defer_req = bool(data["deferral_requires_concession"])
        else:
            defer_req = disposition == "deferred" and severity_max([fr.severity for fr in found_by]) == "major"

        return cls(
            ledger_id=str(data["ledger_id"]),
            title=str(data["title"]),
            file=str(data["file"]),
            line=int(data["line"]),
            evidence=str(data["evidence"]),
            failure_scenario=str(data["failure_scenario"]),
            suggested_fix=str(data["suggested_fix"]),
            round_opened=int(data["round_opened"]),
            found_by=found_by,
            seconded_by=list(data.get("seconded_by", ())),
            source_evidence_ids=src_ev,
            state=str(data.get("state", "open")),
            disposition=disposition,
            disposition_reason=str(data.get("disposition_reason", "")),
            disposition_citation=str(data.get("disposition_citation", "")),
            fix_note=str(data.get("fix_note", "")),
            fix_revision=fix_rev,
            fix_snapshot_hash=str(data.get("fix_snapshot_hash", "")),
            deferral_requires_concession=defer_req,
            question_to=str(data["question_to"]) if data.get("question_to") else None,
            question_text=str(data.get("question_text", "")),
            question_used=bool(data.get("question_used", False)),
            conflicts_with_id=str(data["conflicts_with_id"]) if data.get("conflicts_with_id") else None,
            proposed_severity=str(data["proposed_severity"]) if data.get("proposed_severity") else None,
            proposed_severity_reason=str(data.get("proposed_severity_reason", "")),
            merged_into=str(data["merged_into"]) if data.get("merged_into") else None,
            replacement_verifier=str(data["replacement_verifier"]) if data.get("replacement_verifier") else None,
            replacement_verified_revision=int(data.get("replacement_verified_revision", 0)),
            disagreement_rounds=int(data.get("disagreement_rounds", 0)),
            escalation_reason=str(data.get("escalation_reason", "")),
            round_closed=int(data["round_closed"]) if data.get("round_closed") is not None else None,
            latest_responses=dict(data.get("latest_responses", {})),
            history=history,
        )


@dataclass
class ReviewerRuntimeState:
    config: ReviewerConfig
    status: str = "active"  # active, quota_paused, dropped_quota, shed_allowance, substituted
    session_id: str | None = None
    harness_version: str | None = None
    version_warning: str | None = None
    consecutive_incomplete: int = 0
    total_duration_s: float = 0.0
    findings_raised: int = 0
    conceded_count: int = 0
    maintained_count: int = 0
    last_usage: dict[str, Any] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_allowance_ratio: float | None = None
    shared_state_note: str | None = None
    evidence_exposed: list[str] = field(default_factory=list)  # Cumulative evidence IDs seen by this session
    initial_review_completed: bool = False  # True once reviewer completed a valid initial BASE.diff review (F-001)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status,
            "session_id": self.session_id,
            "harness_version": self.harness_version,
            "version_warning": self.version_warning,
            "consecutive_incomplete": self.consecutive_incomplete,
            "total_duration_s": round(self.total_duration_s, 3),
            "findings_raised": self.findings_raised,
            "conceded_count": self.conceded_count,
            "maintained_count": self.maintained_count,
            "last_usage": dict(self.last_usage),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "last_allowance_ratio": self.last_allowance_ratio,
            "shared_state_note": self.shared_state_note,
            "evidence_exposed": sorted(set(self.evidence_exposed)),
            "initial_review_completed": self.initial_review_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerRuntimeState:
        sid = str(data["session_id"]) if data.get("session_id") else None
        init_done = (
            bool(data["initial_review_completed"])
            if "initial_review_completed" in data
            else bool(sid)
        )
        return cls(
            config=ReviewerConfig.from_dict(data["config"]),
            status=str(data.get("status", "active")),
            session_id=sid,
            harness_version=str(data["harness_version"]) if data.get("harness_version") else None,
            version_warning=str(data["version_warning"]) if data.get("version_warning") else None,
            consecutive_incomplete=int(data.get("consecutive_incomplete", 0)),
            total_duration_s=float(data.get("total_duration_s", 0.0)),
            findings_raised=int(data.get("findings_raised", 0)),
            conceded_count=int(data.get("conceded_count", 0)),
            maintained_count=int(data.get("maintained_count", 0)),
            last_usage=dict(data.get("last_usage", {})),
            total_input_tokens=int(data.get("total_input_tokens", 0)),
            total_output_tokens=int(data.get("total_output_tokens", 0)),
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            last_allowance_ratio=float(data["last_allowance_ratio"]) if data.get("last_allowance_ratio") is not None else None,
            shared_state_note=str(data["shared_state_note"]) if data.get("shared_state_note") else None,
            evidence_exposed=list(data.get("evidence_exposed", ())),
            initial_review_completed=init_done,
        )


@dataclass
class RunLedger:
    run_id: str
    created_at: str
    artifact: str
    purpose: str
    issue: str
    in_scope: str
    out_of_scope: str
    repos: list[RepoSpec]
    ledger_version: int = 3
    docs: list[str] = field(default_factory=list)
    run_tags: list[str] = field(default_factory=list)
    gates_lifted: list[GateLift] = field(default_factory=list)
    approved_ask_tags: list[str] = field(default_factory=list)
    allow_map: dict[str, list[str]] = field(default_factory=dict)
    post_run_hooks_enabled: bool = True
    sandbox_mode: str = "bwrap"
    implementor_family: str | None = None
    current_round: int = 0
    in_progress_round: int | None = None
    pending_round_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_round_obligations: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    pending_round_roster: list[str] = field(default_factory=list)
    pending_overlap_hints: list[str] = field(default_factory=list)
    status: str = "initialized"  # initialized, needs_triage, converged, paused
    pause_reason: str = ""
    pause_causes: list[str] = field(default_factory=list)
    soft_ceiling_override: int | None = None
    min_families_override: int | None = None
    quota_adjustments: list[str] = field(default_factory=list)
    quota_warnings: list[str] = field(default_factory=list)
    shed_plans: list[str] = field(default_factory=list)  # Permanently shed plans (astra-44)
    included_files: list[str] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    project_exclusions: list[str] = field(default_factory=list)
    reviewers: dict[str, ReviewerRuntimeState] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    next_finding_num: int = 1
    round_snapshots: dict[str, dict[str, RepoRoundSnapshot]] = field(default_factory=dict)
    round_changed_lines: list[dict[str, list[list[int]]]] = field(default_factory=list)
    fix_overlap_hints: list[str] = field(default_factory=list)
    hooks_executed: list[dict[str, Any]] = field(default_factory=list)

    def get_finding(self, ledger_id: str) -> Finding:
        for f in self.findings:
            if f.ledger_id == ledger_id:
                return f
        raise KeyError(f"Finding not found: {ledger_id}")

    def allocate_finding_id(self) -> str:
        fid = f"F-{self.next_finding_num:03d}"
        self.next_finding_num += 1
        return fid

    def live_reviewers(self) -> list[ReviewerRuntimeState]:
        return [r for r in self.reviewers.values() if r.status == "active"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "artifact": self.artifact,
            "purpose": self.purpose,
            "issue": self.issue,
            "in_scope": self.in_scope,
            "out_of_scope": self.out_of_scope,
            "repos": [r.to_dict() for r in self.repos],
            "docs": list(self.docs),
            "run_tags": list(self.run_tags),
            "gates_lifted": [g.to_dict() for g in self.gates_lifted],
            "approved_ask_tags": sorted(set(self.approved_ask_tags)),
            "allow_map": {k: list(v) for k, v in self.allow_map.items()},
            "post_run_hooks_enabled": self.post_run_hooks_enabled,
            "sandbox_mode": self.sandbox_mode,
            "implementor_family": self.implementor_family,
            "current_round": self.current_round,
            "in_progress_round": self.in_progress_round,
            "pending_round_results": dict(self.pending_round_results),
            "pending_round_obligations": dict(self.pending_round_obligations),
            "pending_round_roster": list(self.pending_round_roster),
            "pending_overlap_hints": list(self.pending_overlap_hints),
            "status": self.status,
            "pause_reason": self.pause_reason,
            "pause_causes": list(self.pause_causes),
            "soft_ceiling_override": self.soft_ceiling_override,
            "min_families_override": self.min_families_override,
            "quota_adjustments": list(self.quota_adjustments),
            "quota_warnings": list(self.quota_warnings),
            "shed_plans": sorted(set(self.shed_plans)),
            "included_files": list(self.included_files),
            "excluded_files": list(self.excluded_files),
            "evidence": [e.to_dict() for e in self.evidence],
            "project_exclusions": list(self.project_exclusions),
            "reviewers": {k: v.to_dict() for k, v in self.reviewers.items()},
            "findings": [f.to_dict() for f in self.findings],
            "next_finding_num": self.next_finding_num,
            "round_snapshots": {
                rk: {rn: rs.to_dict() for rn, rs in rmap.items()}
                for rk, rmap in self.round_snapshots.items()
            },
            "round_changed_lines": list(self.round_changed_lines),
            "fix_overlap_hints": list(self.fix_overlap_hints),
            "hooks_executed": list(self.hooks_executed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunLedger:
        hooks_raw = data.get("hooks_executed", ())
        hooks_norm = [
            h if isinstance(h, dict) else {"command": str(h), "status": "ok"}
            for h in hooks_raw
        ]
        evidence_items = [EvidenceItem.from_dict(e) for e in data.get("evidence", ())]
        all_ev_ids = [e.id for e in evidence_items]
        lver = int(data.get("ledger_version", 1))
        legacy_prov = lver < 2
        raw_reviewers = {k: ReviewerRuntimeState.from_dict(v) for k, v in data.get("reviewers", {}).items()}
        if lver < 3:
            # Preserve pre-split legacy plan_id ("agy") on agy reviewers with no explicit plan
            # so persisted `shed_plans: ["agy"]` continues matching consistently on old ledgers.
            for rstate in raw_reviewers.values():
                if rstate.config.harness == "agy" and rstate.config.plan is None:
                    rc = rstate.config
                    rstate.config = ReviewerConfig(
                        name=rc.name,
                        harness=rc.harness,
                        model=rc.model,
                        family=rc.family,
                        vendor=rc.vendor,
                        effort=rc.effort,
                        extra=rc.extra,
                        enabled=rc.enabled,
                        serialize_group=rc.serialize_group,
                        plan="agy",
                    )
        return cls(
            ledger_version=3,
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            artifact=str(data.get("artifact", "")),
            purpose=str(data.get("purpose", "")),
            issue=str(data.get("issue", "")),
            in_scope=str(data.get("in_scope", "")),
            out_of_scope=str(data.get("out_of_scope", "")),
            repos=[RepoSpec.from_dict(r) for r in data.get("repos", ())],
            docs=list(data.get("docs", ())),
            run_tags=list(data.get("run_tags", ())),
            gates_lifted=[GateLift.from_dict(g) for g in data.get("gates_lifted", ())],
            approved_ask_tags=list(data.get("approved_ask_tags", ())),
            allow_map={k: list(v) for k, v in data.get("allow_map", {}).items()},
            post_run_hooks_enabled=bool(data.get("post_run_hooks_enabled", True)),
            sandbox_mode=str(data.get("sandbox_mode", "bwrap")),
            implementor_family=str(data["implementor_family"]) if data.get("implementor_family") else None,
            current_round=int(data.get("current_round", 0)),
            in_progress_round=int(data["in_progress_round"]) if data.get("in_progress_round") is not None else None,
            pending_round_results=dict(data.get("pending_round_results", {})),
            pending_round_obligations=dict(data.get("pending_round_obligations", {})),
            pending_round_roster=list(data.get("pending_round_roster", ())),
            pending_overlap_hints=list(data.get("pending_overlap_hints", ())),
            status=str(data.get("status", "initialized")),
            pause_reason=str(data.get("pause_reason", "")),
            pause_causes=list(data.get("pause_causes", ())),
            soft_ceiling_override=int(data["soft_ceiling_override"]) if data.get("soft_ceiling_override") is not None else None,
            min_families_override=int(data["min_families_override"]) if data.get("min_families_override") is not None else None,
            quota_adjustments=list(data.get("quota_adjustments", ())),
            quota_warnings=list(data.get("quota_warnings", ())),
            shed_plans=list(data.get("shed_plans", ())),
            included_files=list(data.get("included_files", ())),
            excluded_files=list(data.get("excluded_files", ())),
            evidence=evidence_items,
            project_exclusions=list(data.get("project_exclusions", ())),
            reviewers=raw_reviewers,
            findings=[
                Finding.from_dict(f, default_evidence_ids=all_ev_ids, legacy_provenance=legacy_prov)
                for f in data.get("findings", ())
            ],
            next_finding_num=int(data.get("next_finding_num", 1)),
            round_snapshots={
                rk: {rn: RepoRoundSnapshot.from_dict(rs) for rn, rs in rmap.items()}
                for rk, rmap in data.get("round_snapshots", {}).items()
            },
            round_changed_lines=list(data.get("round_changed_lines", ())),
            fix_overlap_hints=list(data.get("fix_overlap_hints", ())),
            hooks_executed=hooks_norm,
        )


def atomic_write_text(path: Path, content: str, mode: int = 0o600) -> None:
    """Write a file atomically with private permissions via temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.chmod(tmp_name, mode)
        with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_ledger(run_dir: Path, ledger: RunLedger) -> None:
    """Atomically write ledger.json and render LEDGER.md."""
    atomic_write_text(run_dir / "ledger.json", json.dumps(ledger.to_dict(), indent=2) + "\n", mode=0o600)
    atomic_write_text(run_dir / "LEDGER.md", render_ledger_markdown(ledger), mode=0o600)


def load_ledger(run_dir: Path) -> RunLedger:
    path = run_dir / "ledger.json"
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(f"Ledger not found: {path}")
    return RunLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_ledger_markdown(ledger: RunLedger) -> str:
    """Render LEDGER.md deterministically from ledger.json."""
    lines: list[str] = [
        f"# Review Ledger: {ledger.run_id}",
        "",
        f"- **Artifact:** {ledger.artifact or '(unspecified)'}",
        f"- **Status:** `{ledger.status}` (round {ledger.current_round})",
    ]
    if ledger.issue:
        lines.append(f"- **Issue:** {ledger.issue}")
    if ledger.pause_reason:
        lines.append(f"- **Pause reason:** {ledger.pause_reason}")
    lines.append(f"- **Sandbox:** `{ledger.sandbox_mode}`")
    if ledger.implementor_family:
        lines.append(f"- **Implementor family:** `{ledger.implementor_family}`")
    lines.append(
        f"- **Run tags:** {', '.join(f'`{t}`' for t in ledger.run_tags) if ledger.run_tags else 'none'}"
    )
    if ledger.gates_lifted:
        g_str = "; ".join(f"`--{g.kind}` ({g.detail})" for g in ledger.gates_lifted)
        lines.append(f"- **Gates lifted:** {g_str}")
    else:
        lines.append("- **Gates lifted:** none")
    if ledger.quota_adjustments:
        lines.append("- **Quota adjustments:** " + "; ".join(ledger.quota_adjustments))
    if ledger.quota_warnings:
        lines.append("- **Quota probe warnings:** " + "; ".join(ledger.quota_warnings))
    if ledger.project_exclusions:
        lines.append(
            "- **Project file exclusions:** "
            + ", ".join(f"`{e}`" for e in sorted(set(ledger.project_exclusions)))
        )
    else:
        lines.append("- **Project file exclusions:** none")

    warnings = [
        f"`{r.config.name}` ({r.version_warning})"
        for r in ledger.reviewers.values()
        if r.version_warning
    ]
    if warnings:
        lines.append(f"- **Harness version warnings:** {'; '.join(warnings)}")

    shared_notes = [
        f"`{r.config.name}` ({r.shared_state_note})"
        for r in ledger.reviewers.values()
        if r.shared_state_note
    ]
    if shared_notes:
        lines.append(f"- **Shared-state notes:** {'; '.join(shared_notes)}")

    lines.extend(["", "## Findings", ""])
    if not ledger.findings:
        lines.append("No findings recorded.")
    else:
        lines.append(
            "| ID & Title | Severity (per finder / effective) | Found by & Seconded by | "
            "Disposition & Reasons | Fix | Responses | Final state | Rounds open |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in ledger.findings:
            id_title = _md_escape(f"**{f.ledger_id}** {f.title} (`{f.file}:{f.line}`)")
            per_finder = ", ".join(
                f"{fr.reviewer}:{fr.severity}{'(conceded)' if fr.conceded else ''}"
                for fr in f.found_by
            )
            sev_col = _md_escape(f"{per_finder} / **{f.effective_severity}**")
            fb_str = ", ".join(f"{fr.reviewer} (r{fr.round_raised})" for fr in f.found_by)
            if f.seconded_by:
                sec_str = "; seconded by " + ", ".join(
                    f"{s['reviewer']} (r{s['round']})" for s in f.seconded_by
                )
            else:
                sec_str = ""
            found_col = _md_escape(fb_str + sec_str)

            disp_parts: list[str] = []
            if f.disposition:
                disp_parts.append(f"`{f.disposition}`")
            if f.disposition_reason:
                disp_parts.append(f.disposition_reason)
            if f.disposition_citation:
                disp_parts.append(f"cite: {f.disposition_citation}")
            if f.proposed_severity:
                disp_parts.append(f"lower_severity->{f.proposed_severity}: {f.proposed_severity_reason}")
            if f.conflicts_with_id:
                disp_parts.append(f"conflicts_with {f.conflicts_with_id}")
            if f.merged_into:
                disp_parts.append(f"merged into {f.merged_into}")
            disp_col = _md_escape("; ".join(disp_parts) or "—")

            fix_col = _md_escape(f.fix_note or "—")
            resp_parts = [
                f"{rev}: `{r['verdict']}` (r{r['round']}) {r.get('argument', '')[:80]}"
                for rev, r in sorted(f.latest_responses.items())
            ]
            resp_col = _md_escape("; ".join(resp_parts) or "—")
            state_col = _md_escape(
                f"`{f.state}`" + (f" ({f.escalation_reason})" if f.escalation_reason else "")
            )
            lines.append(
                f"| {id_title} | {sev_col} | {found_col} | {disp_col} | {fix_col} | {resp_col} | {state_col} | {f.rounds_open(ledger.current_round)} |"
            )

    escalated = [f for f in ledger.findings if f.state == "escalated"]
    if escalated:
        lines.extend(["", "## Escalations for the user", ""])
        for f in escalated:
            lines.append(
                f"- **{f.ledger_id}** (`{f.effective_severity}`, `{f.file}:{f.line}`): "
                f"{f.title} — {f.escalation_reason}"
            )

    unverified = [f for f in ledger.findings if f.state == "unverified"]
    if unverified:
        lines.extend(["", "## Unverified findings (reviewer dropped out)", ""])
        for f in unverified:
            lines.append(
                f"- **{f.ledger_id}** (`{f.effective_severity}`, `{f.file}:{f.line}`): "
                f"{f.title} — {f.escalation_reason or f'disposition {f.disposition}'}"
            )

    lines.extend(["", "## Reviewers", ""])
    lines.append(
        "| Reviewer | Model | Family / Vendor | Harness & Version | Status | Raised | Accepted | Escalated | Conceded | Maintained | Unique | Run time | Usage |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for name, r in ledger.reviewers.items():
        raised = 0
        accepted = 0
        escalated_cnt = 0
        unique = 0
        for f in ledger.findings:
            if f.state == "merged":
                continue
            finders = {fr.reviewer for fr in f.found_by}
            if name in finders:
                raised += 1
                # Accepted = terminal verified fix only (astra-60)
                if f.disposition == "fixed" and f.state == "closed":
                    accepted += 1
                elif f.state == "escalated":
                    escalated_cnt += 1
                if len(finders) == 1:
                    unique += 1
        hv = f"{r.config.harness} {r.harness_version or '?'}"
        usage_bits: list[str] = []
        if r.total_input_tokens or r.total_output_tokens:
            usage_bits.append(f"{r.total_input_tokens}in/{r.total_output_tokens}out")
        if r.total_cost_usd > 0:
            usage_bits.append(f"${r.total_cost_usd:.4f}")
        if r.last_allowance_ratio is not None:
            usage_bits.append(f"plan:{r.last_allowance_ratio * 100:.0f}%")
        usage_str = ", ".join(usage_bits) or "—"
        lines.append(
            f"| `{name}`{' (extra)' if r.config.extra else ''} | `{r.config.model}` | "
            f"{r.config.family} / {r.config.vendor} | {_md_escape(hv)} | `{r.status}` | "
            f"{raised} | {accepted} | {escalated_cnt} | {r.conceded_count} | {r.maintained_count} | "
            f"{unique} | {r.total_duration_s:.1f}s | {_md_escape(usage_str)} |"
        )

    lines.append("")
    return "\n".join(lines)
