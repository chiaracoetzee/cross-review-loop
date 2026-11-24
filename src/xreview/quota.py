"""Pre-flight quota inspection, shared quota cache, and automatic core->extra fallback."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

from xreview.config import AppConfig, ReviewerConfig, safe_read_regular_file_under
from xreview.ledger import atomic_write_text

# Default cooldown when a harness hits quota without reporting a reset timestamp.
DEFAULT_EXHAUSTED_COOLDOWN_S = 1800.0  # 30 minutes

# Ignore rollout files older than 8 days (older than a 7-day quota window).
_MAX_ROLLOUT_AGE_S = 8 * 86400

_TRY_AGAIN_TIME_RE = re.compile(
    r"try again (?:at|after)\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuotaWindow:
    name: str  # e.g. "primary" / "five_hour", "secondary" / "seven_day"
    used_ratio: float  # 0.0..1.0+
    resets_at: float | None = None  # Unix epoch seconds
    status: str = ""  # e.g. "allowed", "rejected", "rate_limited"

    def is_active(self, now_ts: float) -> bool:
        if self.resets_at is not None and now_ts >= self.resets_at:
            return False
        return True

    def is_exhausted(self, now_ts: float) -> bool:
        if not self.is_active(now_ts):
            return False
        if self.used_ratio >= 1.0:
            return True
        return self.status.lower() in ("rejected", "exhausted", "rate_limited", "blocked")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "used_ratio": round(self.used_ratio, 4),
            "resets_at": self.resets_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuotaWindow:
        return cls(
            name=str(data.get("name", "window")),
            used_ratio=float(data.get("used_ratio", 0.0)),
            resets_at=float(data["resets_at"]) if data.get("resets_at") is not None else None,
            status=str(data.get("status", "")),
        )


@dataclass(frozen=True)
class PlanQuotaStatus:
    plan_id: str
    harness: str
    windows: tuple[QuotaWindow, ...] = ()
    exhausted: bool = False
    exhausted_until: float | None = None
    source: str = "unknown"  # "codex_rollout", "quota_cache", "unknown"
    updated_at: float = 0.0

    def active_windows(self, now_ts: float) -> list[QuotaWindow]:
        return [w for w in self.windows if w.is_active(now_ts)]

    def is_exhausted(self, now_ts: float) -> bool:
        if any(w.is_exhausted(now_ts) for w in self.windows):
            return True
        if self.exhausted:
            until = self.exhausted_until
            if until is None:
                until = self.updated_at + DEFAULT_EXHAUSTED_COOLDOWN_S
            return now_ts < until
        return False

    def worst_window(self, now_ts: float) -> QuotaWindow | None:
        active = self.active_windows(now_ts)
        if not active:
            return None
        return max(active, key=lambda w: (w.is_exhausted(now_ts), w.used_ratio))

    def effective_used_ratio(self, now_ts: float) -> float | None:
        if self.is_exhausted(now_ts):
            w = self.worst_window(now_ts)
            return max(1.0, w.used_ratio) if w else 1.0
        w = self.worst_window(now_ts)
        return w.used_ratio if w is not None else None

    def next_reset_at(self, now_ts: float) -> float | None:
        cands: list[float] = []
        for w in self.active_windows(now_ts):
            if w.resets_at is not None and w.resets_at > now_ts:
                if w.is_exhausted(now_ts) or w.used_ratio >= 0.5:
                    cands.append(w.resets_at)
        if self.exhausted and self.exhausted_until is not None and self.exhausted_until > now_ts:
            cands.append(self.exhausted_until)
        if not cands:
            for w in self.active_windows(now_ts):
                if w.resets_at is not None and w.resets_at > now_ts:
                    cands.append(w.resets_at)
        return min(cands) if cands else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "harness": self.harness,
            "windows": [w.to_dict() for w in self.windows],
            "exhausted": self.exhausted,
            "exhausted_until": self.exhausted_until,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanQuotaStatus:
        return cls(
            plan_id=str(data.get("plan_id", "")),
            harness=str(data.get("harness", "")),
            windows=tuple(QuotaWindow.from_dict(w) for w in data.get("windows", ()) if isinstance(w, dict)),
            exhausted=bool(data.get("exhausted", False)),
            exhausted_until=float(data["exhausted_until"]) if data.get("exhausted_until") is not None else None,
            source=str(data.get("source", "quota_cache")),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class QuotaAdjustmentResult:
    reviewers: list[ReviewerConfig]
    notices: list[str] = field(default_factory=list)
    dropped_exhausted_plans: list[str] = field(default_factory=list)
    fallback_plans: list[str] = field(default_factory=list)
    plan_statuses: dict[str, PlanQuotaStatus] = field(default_factory=dict)


def format_reset_time(resets_at: float | None) -> str:
    """Format a Unix epoch reset timestamp into readable local time."""
    if resets_at is None:
        return "reset time unknown"
    try:
        dt = datetime.fromtimestamp(resets_at).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M %Z").strip()
    except (OSError, OverflowError, ValueError):
        return "reset time unknown"


def parse_reset_hint_from_error(text: str, now_ts: float | None = None) -> float | None:
    """Best-effort parse of a reset time like 'try again at 7:50 PM' from error diagnostics."""
    if not text:
        return None
    m = _TRY_AGAIN_TIME_RE.search(text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = m.group(3).upper()
    if hour < 1 or hour > 12 or minute > 59:
        return None
    if ampm == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    now_dt = datetime.fromtimestamp(now_ts if now_ts is not None else time.time()).astimezone()
    cand = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand <= now_dt:
        cand += timedelta(days=1)
    return cand.timestamp()


@contextmanager
def _quota_cache_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = state_dir / ".quota_cache.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _load_quota_cache_unlocked(state_dir: Path) -> dict[str, PlanQuotaStatus]:
    cache_path = state_dir / "quota_cache.json"
    if not cache_path.exists() or cache_path.is_symlink():
        return {}
    raw = safe_read_regular_file_under(state_dir, "quota_cache.json", max_bytes=1_048_576)
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    plans_raw = data.get("plans", {}) if isinstance(data, dict) else {}
    out: dict[str, PlanQuotaStatus] = {}
    if isinstance(plans_raw, dict):
        for pid, entry in plans_raw.items():
            if isinstance(entry, dict):
                try:
                    out[str(pid)] = PlanQuotaStatus.from_dict(entry)
                except (TypeError, ValueError):
                    continue
    return out


def load_quota_cache(state_dir: Path) -> dict[str, PlanQuotaStatus]:
    """Safely load cached per-plan quota snapshots from <state_dir>/quota_cache.json."""
    if not state_dir.is_dir():
        return {}
    return _load_quota_cache_unlocked(state_dir)


def record_plan_quota(
    state_dir: Path,
    status: PlanQuotaStatus,
) -> None:
    """Atomically persist a plan's latest quota snapshot under flock."""
    with _quota_cache_lock(state_dir):
        existing = _load_quota_cache_unlocked(state_dir)
        prev = existing.get(status.plan_id)
        # Preserve still-active windows if the new update only reports exhaustion with no windows
        merged_windows = status.windows
        if not merged_windows and prev is not None:
            merged_windows = tuple(prev.active_windows(status.updated_at or time.time()))
        merged = PlanQuotaStatus(
            plan_id=status.plan_id,
            harness=status.harness,
            windows=merged_windows,
            exhausted=status.exhausted,
            exhausted_until=status.exhausted_until,
            source=status.source,
            updated_at=status.updated_at or time.time(),
        )
        existing[status.plan_id] = merged
        payload = {
            "version": 1,
            "plans": {pid: st.to_dict() for pid, st in sorted(existing.items())},
        }
        atomic_write_text(state_dir / "quota_cache.json", json.dumps(payload, indent=2) + "\n", mode=0o600)


def _codex_home_for_config(config: AppConfig) -> Path:
    if config.auth.codex_auth_file is not None:
        return config.auth.codex_auth_file.parent
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return Path(os.path.expanduser(env_home))
    return Path.home() / ".codex"


def read_codex_rollout_quota(
    codex_home: Path,
    thread_id: str | None = None,
    plan_id: str = "codex",
    now_ts: float | None = None,
) -> PlanQuotaStatus | None:
    """Safely inspect Codex rollout JSONL files for the latest active rate_limits snapshot."""
    now = now_ts if now_ts is not None else time.time()
    sessions_root = codex_home / "sessions"
    if not sessions_root.exists() or sessions_root.is_symlink():
        return None

    matches: list[tuple[float, str]] = []
    for root, dirs, files in os.walk(sessions_root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            if thread_id and thread_id not in fname:
                continue
            if not thread_id and not fname.startswith("rollout-"):
                continue
            full = Path(root) / fname
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not thread_id and (now - st.st_mtime) > _MAX_ROLLOUT_AGE_S:
                continue
            matches.append((st.st_mtime, str(full.relative_to(codex_home))))

    if not matches:
        return None
    matches.sort(reverse=True)

    # Check up to the 5 most recent rollout files until we find one with rate_limits windows
    for mtime, rel in matches[:5]:
        raw = safe_read_regular_file_under(codex_home, rel, max_bytes=2_097_152, tail=True)
        if not raw:
            continue
        windows_by_name: dict[str, QuotaWindow] = {}
        latest_reached = False
        saw_any = False
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if "rate_limits" not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            rl = (ev.get("payload") or {}).get("rate_limits")
            if not isinstance(rl, dict):
                continue
            has_window_data = False
            for wname in ("primary", "secondary"):
                w = rl.get(wname)
                if isinstance(w, dict) and isinstance(w.get("used_percent"), (int, float)):
                    ratio = float(w["used_percent"]) / 100.0
                    resets = float(w["resets_at"]) if isinstance(w.get("resets_at"), (int, float)) else None
                    windows_by_name[wname] = QuotaWindow(
                        name=wname,
                        used_ratio=ratio,
                        resets_at=resets,
                        status="exhausted" if ratio >= 1.0 else "allowed",
                    )
                    has_window_data = True
            if has_window_data:
                saw_any = True
                latest_reached = bool(rl.get("rate_limit_reached_type"))
            elif rl.get("rate_limit_reached_type"):
                saw_any = True
                latest_reached = True

        if saw_any:
            win_tuple = tuple(windows_by_name.values())
            exhausted_win_resets = [
                w.resets_at for w in win_tuple if w.is_exhausted(now) and w.resets_at is not None
            ]
            exhausted_until = min(exhausted_win_resets) if exhausted_win_resets else None
            return PlanQuotaStatus(
                plan_id=plan_id,
                harness="codex",
                windows=win_tuple,
                exhausted=latest_reached or any(w.is_exhausted(now) for w in win_tuple),
                exhausted_until=exhausted_until,
                source="codex_rollout",
                updated_at=mtime,
            )
    return None


def inspect_plan_quotas(
    config: AppConfig,
    *,
    include_host_codex_rollouts: bool = True,
    now_ts: float | None = None,
) -> dict[str, PlanQuotaStatus]:
    """Collect the latest known quota status for every configured plan (zero API calls)."""
    now = now_ts if now_ts is not None else time.time()
    cached = load_quota_cache(config.state_dir)
    statuses: dict[str, PlanQuotaStatus] = {}

    plan_harnesses: dict[str, str] = {}
    for rc in config.reviewers:
        if rc.enabled:
            plan_harnesses.setdefault(rc.plan_id, rc.harness)

    codex_home = _codex_home_for_config(config) if include_host_codex_rollouts else None

    for pid, harness in plan_harnesses.items():
        best = cached.get(pid)
        if harness == "codex" and codex_home is not None:
            rollout_st = read_codex_rollout_quota(codex_home, thread_id=None, plan_id=pid, now_ts=now)
            if rollout_st is not None:
                if best is None or rollout_st.updated_at >= best.updated_at:
                    best = rollout_st
        if best is not None:
            statuses[pid] = best
        else:
            statuses[pid] = PlanQuotaStatus(plan_id=pid, harness=harness, source="unknown", updated_at=0.0)

    return statuses


def apply_pre_flight_quota_policy(
    candidates: list[ReviewerConfig],
    config: AppConfig,
    plan_statuses: dict[str, PlanQuotaStatus],
    *,
    now_ts: float | None = None,
) -> QuotaAdjustmentResult:
    """Filter/adjust candidate reviewers based on pre-flight plan quotas.

    - Exhausted plans (`100%` or active exhaustion): drop all reviewers on that plan when
      `config.policy.auto_drop_exhausted` is True.
    - Low-quota plans (`used_ratio >= config.policy.fallback_to_extras_at`): back out of
      heavyweight core models (`extra=False`) and promote cheaper extra models (`extra=True`)
      that are ALREADY in `candidates` (never adding unapproved reviewers or widening gates).
    """
    now = now_ts if now_ts is not None else time.time()
    threshold = config.policy.fallback_to_extras_at
    auto_drop = config.policy.auto_drop_exhausted

    by_plan: dict[str, list[ReviewerConfig]] = {}
    for rc in candidates:
        by_plan.setdefault(rc.plan_id, []).append(rc)

    kept: list[ReviewerConfig] = []
    notices: list[str] = []
    dropped_exhausted: list[str] = []
    fallback_plans: list[str] = []

    for pid, members in by_plan.items():
        st = plan_statuses.get(pid)
        if st is None:
            kept.extend(members)
            continue

        if st.is_exhausted(now):
            reset_str = format_reset_time(st.next_reset_at(now))
            worst = st.worst_window(now)
            win_detail = (
                f"{worst.name} {worst.used_ratio * 100:.0f}%"
                if worst is not None
                else "quota exhausted"
            )
            names = ", ".join(r.name for r in members)
            if auto_drop:
                dropped_exhausted.append(pid)
                notices.append(
                    f"Plan {pid!r} is out of quota ({win_detail}; resets {reset_str}): "
                    f"removed {names} from this run."
                )
                continue
            notices.append(
                f"WARNING: Plan {pid!r} appears out of quota ({win_detail}; resets {reset_str}), "
                f"but auto_drop_exhausted is disabled; keeping {names}."
            )
            kept.extend(members)
            continue

        used = st.effective_used_ratio(now)
        if threshold is not None and used is not None and used >= threshold:
            cores = [r for r in members if not r.extra]
            extras = [r for r in members if r.extra]
            worst = st.worst_window(now)
            win_name = worst.name if worst is not None else "quota"
            reset_str = format_reset_time(st.next_reset_at(now))
            if cores and extras:
                fallback_plans.append(pid)
                core_names = ", ".join(r.name for r in cores)
                extra_names = ", ".join(r.name for r in extras)
                # Promote extras to non-extra for this run so they carry the family without being shed.
                promoted = [
                    ReviewerConfig(
                        name=r.name,
                        harness=r.harness,
                        model=r.model,
                        family=r.family,
                        vendor=r.vendor,
                        effort=r.effort,
                        extra=False,
                        plan=r.plan,
                        enabled=r.enabled,
                        serialize_group=r.serialize_group,
                    )
                    for r in extras
                ]
                kept.extend(promoted)
                notices.append(
                    f"Plan {pid!r} is low on quota ({win_name} {used * 100:.0f}% >= {threshold * 100:.0f}%; "
                    f"resets {reset_str}): backed out of heavyweight core ({core_names}), "
                    f"falling back to {extra_names}."
                )
                continue
            if cores and not extras:
                core_names = ", ".join(r.name for r in cores)
                notices.append(
                    f"Plan {pid!r} is low on quota ({win_name} {used * 100:.0f}% >= {threshold * 100:.0f}%; "
                    f"resets {reset_str}), but no cheaper extra reviewer is available; keeping {core_names}."
                )

        kept.extend(members)

    # Preserve original relative ordering from candidates
    order = {rc.name: idx for idx, rc in enumerate(candidates)}
    kept.sort(key=lambda r: order.get(r.name, 9999))

    return QuotaAdjustmentResult(
        reviewers=kept,
        notices=notices,
        dropped_exhausted_plans=dropped_exhausted,
        fallback_plans=fallback_plans,
        plan_statuses=plan_statuses,
    )


def probe_claude_plan_quota(
    config: AppConfig,
    plan_id: str = "claude",
    *,
    no_sandbox: bool = False,
    now_ts: float | None = None,
) -> PlanQuotaStatus | None:
    """Run a tiny 1-turn sandboxed haiku probe to capture live Claude unifiedWindows quota telemetry."""
    from xreview.adapters.base import TurnRequest
    from xreview.adapters.claude import ClaudeAdapter
    from xreview.schema import REVIEWER_REPLY_SCHEMA

    now = now_ts if now_ts is not None else time.time()
    probe_rc = ReviewerConfig(
        name=f"quota_probe_{plan_id}",
        harness="claude",
        model="haiku",
        family="claude",
        vendor="anthropic",
        effort="low",
        plan=plan_id,
    )
    with tempfile.TemporaryDirectory(prefix="xreview-qprobe-") as td:
        root = Path(td)
        tree, inputs, state = root / "tree", root / "inputs", root / "state"
        tree.mkdir(mode=0o700)
        inputs.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        schema_p = inputs / "schema.json"
        schema_p.write_text(json.dumps(REVIEWER_REPLY_SCHEMA), encoding="utf-8")
        req = TurnRequest(
            run_id="qprobe",
            round_num=1,
            reviewer=probe_rc,
            prompt='Return valid JSON with empty findings, empty responses, and summary "ok". Do not read any files.',
            schema_host_path=schema_p,
            tree_host_dir=tree,
            inputs_host_dir=inputs,
            state_host_dir=state,
            raw_log_path=root / "raw.jsonl",
            proxy_log_path=root / "proxy.log",
            timeout_s=45,
            allow_no_sandbox=no_sandbox,
        )
        res = ClaudeAdapter().run_turn(req, config)

    q_wins = tuple(QuotaWindow.from_dict(w) for w in res.quota_windows if isinstance(w, dict))
    if not q_wins and not res.quota_exhausted and res.allowance_used_ratio is None:
        return None
    if not q_wins and res.allowance_used_ratio is not None:
        q_wins = (
            QuotaWindow(
                name="plan",
                used_ratio=res.allowance_used_ratio,
                resets_at=res.quota_resets_at,
                status="exhausted" if res.quota_exhausted else "allowed",
            ),
        )
    status = PlanQuotaStatus(
        plan_id=plan_id,
        harness="claude",
        windows=q_wins,
        exhausted=res.quota_exhausted,
        exhausted_until=res.quota_resets_at if res.quota_exhausted else None,
        source="claude_probe",
        updated_at=now,
    )
    try:
        record_plan_quota(config.state_dir, status)
    except OSError:
        pass
    return status


def refresh_plan_quotas(
    config: AppConfig,
    *,
    force: bool = False,
    no_sandbox: bool = False,
    include_host_codex_rollouts: bool = True,
    now_ts: float | None = None,
    claude_probe_fn: Callable[[AppConfig, str, bool], PlanQuotaStatus | None] | None = None,
) -> tuple[dict[str, PlanQuotaStatus], list[str]]:
    """Inspect plan quotas and refresh stale/missing Claude quota snapshots via a tiny haiku probe."""
    now = now_ts if now_ts is not None else time.time()
    max_age = max(60, int(config.policy.quota_cache_max_age_seconds))
    statuses = inspect_plan_quotas(
        config,
        include_host_codex_rollouts=include_host_codex_rollouts,
        now_ts=now,
    )
    warnings: list[str] = []

    for pid, st in list(statuses.items()):
        if st.harness != "claude":
            continue
        age = (now - st.updated_at) if st.updated_at > 0 else float("inf")
        has_active_data = bool(st.active_windows(now)) or st.is_exhausted(now)
        needs_refresh = force or (not has_active_data) or (age > max_age)
        if not needs_refresh:
            continue
        try:
            if claude_probe_fn is not None:
                fresh = claude_probe_fn(config, pid, no_sandbox)
            else:
                fresh = probe_claude_plan_quota(config, pid, no_sandbox=no_sandbox, now_ts=now)
            if fresh is not None:
                statuses[pid] = fresh
            else:
                warnings.append(f"Live quota probe for plan {pid!r} returned no rate_limit_event data.")
        except Exception as exc:
            warnings.append(f"Live quota probe for plan {pid!r} failed: {exc}")

    return statuses, warnings

