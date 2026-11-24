"""Pre-flight quota inspection, shared quota cache, and automatic core->extra fallback."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Collection, Iterator

from xreview.config import AppConfig, ReviewerConfig, agy_pool_for_reviewer, safe_read_regular_file_under
from xreview.ledger import atomic_write_text


class QuotaProbeError(RuntimeError):
    """Raised when a live quota probe fails with diagnostic details."""


def _collapse_ws(text: str, max_len: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."

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

    def is_active(self, now_ts: float, updated_at: float | None = None) -> bool:
        if self.resets_at is not None:
            return now_ts < self.resets_at
        if updated_at is not None and updated_at > 0:
            return now_ts < (updated_at + DEFAULT_EXHAUSTED_COOLDOWN_S)
        return True

    def is_exhausted(self, now_ts: float, updated_at: float | None = None) -> bool:
        if not self.is_active(now_ts, updated_at=updated_at):
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
    pool: str | None = None  # e.g. "gemini" or "3p" for Antigravity pools

    def active_windows(self, now_ts: float) -> list[QuotaWindow]:
        return [w for w in self.windows if w.is_active(now_ts, updated_at=self.updated_at)]

    def is_exhausted(self, now_ts: float) -> bool:
        if any(w.is_exhausted(now_ts, updated_at=self.updated_at) for w in self.windows):
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
        out: dict[str, Any] = {
            "plan_id": self.plan_id,
            "harness": self.harness,
            "windows": [w.to_dict() for w in self.windows],
            "exhausted": self.exhausted,
            "exhausted_until": self.exhausted_until,
            "source": self.source,
            "updated_at": self.updated_at,
        }
        if self.pool is not None:
            out["pool"] = self.pool
        return out

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
            pool=str(data["pool"]) if data.get("pool") is not None else None,
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
        # Preserve still-active windows only when harness and pool match.
        same_bucket = (
            prev is not None
            and prev.harness == status.harness
            and (status.pool is None or prev.pool == status.pool)
        )
        merged_windows = status.windows
        if not merged_windows and same_bucket and prev is not None:
            merged_windows = tuple(prev.active_windows(status.updated_at or time.time()))
        merged = PlanQuotaStatus(
            plan_id=status.plan_id,
            harness=status.harness,
            windows=merged_windows,
            exhausted=status.exhausted,
            exhausted_until=status.exhausted_until,
            source=status.source,
            updated_at=status.updated_at or time.time(),
            pool=status.pool if status.pool is not None else (prev.pool if same_bucket and prev is not None else None),
        )
        existing[status.plan_id] = merged
        payload = {
            "version": 2,
            "plans": {pid: st.to_dict() for pid, st in sorted(existing.items())},
        }
        atomic_write_text(state_dir / "quota_cache.json", json.dumps(payload, indent=2) + "\n", mode=0o600)


def _credential_source_stamp(config: AppConfig, harness: str) -> dict[str, Any]:
    """Build a fingerprint of the configured credential file and credit policy for probe resync."""
    src: Path | None = None
    if harness == "agy":
        src = config.auth.agy_token_file or (Path.home() / ".config" / "xreview" / "agy-oauth-token")
    elif harness == "claude":
        src = config.auth.claude_token_file or (Path.home() / ".claude" / ".credentials.json")
    mtime_ns: int | None = None
    size: int | None = None
    if src is not None:
        try:
            st = src.stat()
            mtime_ns = st.st_mtime_ns
            size = st.st_size
        except OSError:
            pass
    return {
        "src": str(src) if src is not None else None,
        "mtime_ns": mtime_ns,
        "size": size,
        "credits": config.policy.credits,
    }


@contextmanager
def _probe_state_dir(config: AppConfig, harness: str) -> Iterator[Path]:
    """Return a persistent private probe state dir under <state_dir>/.qprobe/<harness>, resyncing on source/policy change."""
    import shutil

    state_dir = config.state_dir
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    qprobe_root = state_dir / ".qprobe"
    if qprobe_root.is_symlink():
        raise PermissionError(f"Refusing symlinked probe state dir: {qprobe_root}")
    qprobe_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(qprobe_root, 0o700)
    lock_path = qprobe_root / f".{harness}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        hdir = qprobe_root / harness
        if hdir.is_symlink():
            raise PermissionError(f"Refusing symlinked probe harness dir: {hdir}")
        stamp_path = qprobe_root / f".{harness}.stamp.json"
        desired_stamp = _credential_source_stamp(config, harness)
        prev_stamp: dict[str, Any] | None = None
        if stamp_path.exists() and not stamp_path.is_symlink():
            try:
                prev_stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prev_stamp = None
        if prev_stamp != desired_stamp:
            # Credential source or credit policy changed: clear cached copy & init markers so prepare_state resyncs.
            if hdir.exists():
                shutil.rmtree(hdir)
            for marker in qprobe_root.glob(f".quota_probe_*.{harness}_init"):
                if not marker.is_symlink():
                    try:
                        marker.unlink()
                    except OSError:
                        pass
            atomic_write_text(stamp_path, json.dumps(desired_stamp, sort_keys=True) + "\n", mode=0o600)
        hdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(hdir, 0o700)
        yield hdir
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


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


def _expected_agy_pools_for_plan(config: AppConfig, plan_id: str) -> set[str]:
    return {agy_pool_for_reviewer(r) for r in config.reviewers if r.enabled and r.plan_id == plan_id and r.harness == "agy"}


def _agy_cache_matches_pools(st: PlanQuotaStatus, expected_pools: set[str]) -> bool:
    """Validate an agy cache entry's pool metadata / window prefixes so legacy mixed entries are rejected."""
    if not expected_pools:
        return True
    if st.pool is not None:
        return st.pool in expected_pools or (st.pool == "mixed" and len(expected_pools) > 1)
    # Legacy v1 cache entry without explicit pool: accept only if windows unambiguously match expected_pools.
    if not st.windows:
        return False
    inferred: set[str] = set()
    for w in st.windows:
        wn = w.name.strip().lower()
        if wn.startswith("gemini-"):
            inferred.add("gemini")
        elif wn.startswith("3p-"):
            inferred.add("3p")
        else:
            return False
    return inferred.issubset(expected_pools)


def inspect_plan_quotas(
    config: AppConfig,
    *,
    plan_ids: Collection[str] | None = None,
    include_host_codex_rollouts: bool = True,
    now_ts: float | None = None,
) -> dict[str, PlanQuotaStatus]:
    """Collect the latest known quota status for configured (or filtered) plans (zero API calls)."""
    now = now_ts if now_ts is not None else time.time()
    cached = load_quota_cache(config.state_dir)
    statuses: dict[str, PlanQuotaStatus] = {}
    allowed_plans = {p.strip().lower() for p in plan_ids} if plan_ids is not None else None

    plan_harnesses: dict[str, str] = {}
    for rc in config.reviewers:
        if rc.enabled:
            if allowed_plans is not None and rc.plan_id not in allowed_plans:
                continue
            plan_harnesses.setdefault(rc.plan_id, rc.harness)

    codex_home = _codex_home_for_config(config) if include_host_codex_rollouts else None

    for pid, harness in plan_harnesses.items():
        best = cached.get(pid)
        if best is not None and best.harness != harness:
            best = None
        if best is not None and harness == "agy":
            expected_pools = _expected_agy_pools_for_plan(config, pid)
            if not _agy_cache_matches_pools(best, expected_pools):
                best = None
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


def _parse_iso_timestamp(raw: str | None) -> float | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        s = raw.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _classify_agy_bucket(group_name: str, bucket_id: str) -> str:
    """Classify an Antigravity `/usage` bucket once: explicit ID prefix first, then group name."""
    bid = bucket_id.strip().lower()
    if bid.startswith("gemini-"):
        return "gemini"
    if bid.startswith("3p-"):
        return "3p"
    gname = group_name.strip().lower()
    if "gemini" in gname:
        return "gemini"
    if any(tok in gname for tok in ("claude", "gpt", "3p", "third")):
        return "3p"
    return "unknown"


def parse_agy_usage_json(
    payload: dict[str, Any],
    *,
    families: set[str] | None = None,
    plan_id: str = "agy",
    now_ts: float | None = None,
) -> PlanQuotaStatus | None:
    """Parse structured JSON output from `agy --output-format json -p /usage`."""
    now = now_ts if now_ts is not None else time.time()
    top_status = payload.get("status")
    if top_status is not None and str(top_status).strip().upper() != "SUCCESS":
        return None
    cmd = payload.get("command")
    if not isinstance(cmd, dict) or cmd.get("name") != "usage":
        return None
    data = cmd.get("data")
    if not isinstance(data, dict):
        return None
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        return None

    fams = {f.strip().lower() for f in (families or set()) if f and f.strip()}
    want_gemini = (not fams) or ("gemini" in fams)
    want_3p = (not fams) or any(f != "gemini" for f in fams)

    selected_buckets: list[dict[str, Any]] = []
    all_buckets: list[dict[str, Any]] = []
    for grp in groups:
        if not isinstance(grp, dict):
            continue
        gname = str(grp.get("name", ""))
        buckets = grp.get("buckets")
        if not isinstance(buckets, list):
            continue
        for b in buckets:
            if not isinstance(b, dict):
                continue
            all_buckets.append(b)
            kind = _classify_agy_bucket(gname, str(b.get("id", "")))
            if (kind == "gemini" and want_gemini) or (kind == "3p" and want_3p):
                selected_buckets.append(b)

    # Only fall back to all_buckets when caller did not filter by family.
    use_buckets = selected_buckets if fams else (selected_buckets or all_buckets)
    if not use_buckets:
        return None
    windows: list[QuotaWindow] = []
    for b in use_buckets:
        rem = b.get("remaining_fraction")
        if isinstance(rem, bool) or not isinstance(rem, (int, float)):
            continue
        used = max(0.0, min(1.0, 1.0 - float(rem)))
        resets_at = _parse_iso_timestamp(b.get("reset_time"))
        wname = str(b.get("id") or b.get("window") or b.get("name") or "window")
        windows.append(
            QuotaWindow(
                name=wname,
                used_ratio=used,
                resets_at=resets_at,
                status="exhausted" if used >= 1.0 else "allowed",
            )
        )
    if not windows:
        return None

    exhausted_resets = [w.resets_at for w in windows if w.is_exhausted(now) and w.resets_at is not None]
    exhausted = any(w.is_exhausted(now) for w in windows)
    pool_tag: str | None = None
    if fams:
        if want_gemini and not want_3p:
            pool_tag = "gemini"
        elif want_3p and not want_gemini:
            pool_tag = "3p"
        else:
            pool_tag = "mixed"
    return PlanQuotaStatus(
        plan_id=plan_id,
        harness="agy",
        windows=tuple(windows),
        exhausted=exhausted,
        exhausted_until=min(exhausted_resets) if exhausted_resets else None,
        source="agy_usage",
        updated_at=now,
        pool=pool_tag,
    )


def _summarize_agy_groups(payload: dict[str, Any]) -> str:
    cmd = payload.get("command")
    data = cmd.get("data") if isinstance(cmd, dict) else None
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list):
        return "[]"
    parts: list[str] = []
    for grp in groups:
        if not isinstance(grp, dict):
            continue
        gname = str(grp.get("name", "?"))
        buckets = grp.get("buckets")
        bids = [str(b.get("id", "?")) for b in buckets if isinstance(b, dict)] if isinstance(buckets, list) else []
        parts.append(f"{gname}: {','.join(bids)}")
    return "[" + _collapse_ws("; ".join(parts), 160) + "]"


def _fetch_agy_usage_payload(
    config: AppConfig,
    *,
    no_sandbox: bool = False,
) -> dict[str, Any]:
    """Run the documented zero-token `agy --output-format json -p /usage` once and return its JSON dict."""
    from xreview.adapters.agy import AgyAdapter, _find_agy_binary
    from xreview.sandbox import SandboxSpec, run_in_sandbox

    agy_bin = _find_agy_binary()
    probe_rc = ReviewerConfig(
        name="quota_probe_agy",
        harness="agy",
        model="usage",
        family="gemini",
        vendor="google",
        plan="agy",
    )
    timeout_s = 45
    with _probe_state_dir(config, "agy") as state, tempfile.TemporaryDirectory(prefix="xreview-agy-qprobe-") as td:
        root = Path(td)
        tree, inputs = root / "tree", root / "inputs"
        tree.mkdir(mode=0o700)
        inputs.mkdir(mode=0o700)
        AgyAdapter().prepare_state(probe_rc, state, config)

        use_sandbox = not no_sandbox
        if use_sandbox:
            exe_path = "/harness/agy"
            tree_path = f"/review/qprobe/{probe_rc.name}/tree"
            path_env = "/harness:/usr/bin:/bin"
            harness_mounts = ((agy_bin, "/harness/agy"),)
        else:
            exe_path = str(agy_bin)
            tree_path = str(tree)
            path_env = f"{agy_bin.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            harness_mounts = ()

        settings_json = json.dumps({"useG1Credits": config.policy.credits != "never"})
        cmd = (
            exe_path,
            "--output-format",
            "json",
            "--print-timeout",
            "30s",
            "--settings",
            settings_json,
            "-p",
            "/usage",
        )
        spec = SandboxSpec(
            run_id="qprobe",
            reviewer_name=probe_rc.name,
            tree_host_dir=tree,
            inputs_host_dir=inputs,
            state_host_dir=state,
            harness_mounts=harness_mounts,
            command=cmd,
            cwd_in_sandbox=tree_path,
            env={
                "PATH": path_env,
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/nonexistent",
                "XDG_RUNTIME_DIR": "/nonexistent",
            },
            vendor_hosts=config.vendor_hosts_for("google"),
            extra_ro_paths=config.profile.extra_ro_paths,
            timeout_s=timeout_s,
            use_sandbox=use_sandbox,
        )
        sb_res = run_in_sandbox(
            spec,
            stdin_text=None,
            raw_log_path=root / "raw.json",
            proxy_log_path=root / "proxy.log",
            capture_stdout=True,
        )

    if sb_res.timed_out:
        raise QuotaProbeError(f"agy /usage timed out after {timeout_s}s")
    raw_out = "\n".join(sb_res.stdout_lines).strip()
    if sb_res.returncode != 0:
        diag_parts = [p for p in (sb_res.stderr.strip(), raw_out) if p]
        diag = _collapse_ws(" | ".join(diag_parts) or "no output")
        raise QuotaProbeError(f"agy /usage exited {sb_res.returncode}: {diag}")
    if not raw_out:
        diag = _collapse_ws(sb_res.stderr or "empty stdout")
        raise QuotaProbeError(f"agy /usage returned empty output ({diag})")
    try:
        payload = json.loads(raw_out)
    except json.JSONDecodeError as exc:
        raise QuotaProbeError(f"agy /usage returned invalid JSON ({exc.msg}): {_collapse_ws(raw_out, 120)}") from exc
    if not isinstance(payload, dict):
        raise QuotaProbeError(f"agy /usage returned non-object JSON ({type(payload).__name__})")
    return payload


def _status_from_agy_payload(
    config: AppConfig,
    payload: dict[str, Any],
    plan_id: str,
    now_ts: float,
) -> PlanQuotaStatus:
    pools = _expected_agy_pools_for_plan(config, plan_id)
    families = {"gemini" if p == "gemini" else "3p" for p in pools}
    status = parse_agy_usage_json(payload, families=families, plan_id=plan_id, now_ts=now_ts)
    if status is None:
        top_st = payload.get("status")
        groups_summary = _summarize_agy_groups(payload)
        raise QuotaProbeError(
            f"agy /usage payload had no matching quota buckets "
            f"(status={top_st!r}, pools={sorted(pools)}, groups={groups_summary})"
        )
    try:
        record_plan_quota(config.state_dir, status)
    except OSError:
        pass
    return status


def probe_agy_plan_quota(
    config: AppConfig,
    plan_id: str = "agy",
    *,
    no_sandbox: bool = False,
    now_ts: float | None = None,
) -> PlanQuotaStatus:
    """Run the documented zero-token `agy --output-format json -p /usage` command inside bwrap."""
    now = now_ts if now_ts is not None else time.time()
    payload = _fetch_agy_usage_payload(config, no_sandbox=no_sandbox)
    return _status_from_agy_payload(config, payload, plan_id, now)


def probe_claude_plan_quota(
    config: AppConfig,
    plan_id: str = "claude",
    *,
    no_sandbox: bool = False,
    now_ts: float | None = None,
) -> PlanQuotaStatus:
    """Run an ultra-tiny sandboxed haiku turn (--tools "" --system-prompt) to read live Claude quota."""
    from xreview.adapters.base import detect_quota_exhausted
    from xreview.adapters.claude import (
        ClaudeAdapter,
        _find_claude_binary,
        build_claude_base_command,
        build_claude_env,
        parse_claude_rate_limit_info,
        read_claude_oauth_token,
    )
    from xreview.sandbox import SandboxSpec, run_in_sandbox

    now = now_ts if now_ts is not None else time.time()
    claude_bin = _find_claude_binary()
    adapter = ClaudeAdapter()
    probe_rc = ReviewerConfig(
        name=f"quota_probe_{plan_id}",
        harness="claude",
        model="haiku",
        family="claude",
        vendor="anthropic",
        effort="low",
        plan=plan_id,
    )

    rate_limit_windows: dict[str, QuotaWindow] = {}
    rate_limit_rejected = False
    rate_limit_resets_at: float | None = None
    stream_errors: list[str] = []
    saw_result = False
    is_error_result = False

    def _on_line(line: str) -> None:
        nonlocal rate_limit_rejected, rate_limit_resets_at, saw_result, is_error_result
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        ev_type = ev.get("type")
        if ev_type == "rate_limit_event":
            info = ev.get("rate_limit_info")
            if isinstance(info, dict):
                wins, rej, r_at = parse_claude_rate_limit_info(info)
                if rej:
                    rate_limit_rejected = True
                if r_at is not None:
                    rate_limit_resets_at = r_at
                for wname, wdict in wins.items():
                    rate_limit_windows[wname] = QuotaWindow.from_dict(wdict)
        elif ev_type == "result":
            saw_result = True
            if ev.get("is_error"):
                is_error_result = True
                if ev.get("result"):
                    stream_errors.append(str(ev["result"]))
        elif ev_type == "error":
            stream_errors.append(json.dumps(ev))

    timeout_s = 30
    with _probe_state_dir(config, "claude") as state, tempfile.TemporaryDirectory(prefix="xreview-claude-qprobe-") as td:
        root = Path(td)
        tree, inputs = root / "tree", root / "inputs"
        tree.mkdir(mode=0o700)
        inputs.mkdir(mode=0o700)
        adapter.prepare_state(probe_rc, state, config)

        use_sandbox = not no_sandbox
        if use_sandbox:
            exe_path = "/harness/claude"
            tree_path = f"/review/qprobe/{probe_rc.name}/tree"
            cfg_dir_env = "/home/reviewer/.claude"
            path_env = "/harness:/usr/bin:/bin"
            harness_mounts = ((claude_bin, "/harness/claude"),)
        else:
            exe_path = str(claude_bin)
            tree_path = str(tree)
            cfg_dir_env = str(state / ".claude")
            path_env = f"{claude_bin.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            harness_mounts = ()

        cmd = tuple(
            build_claude_base_command(exe_path, "haiku", "low")
            + [
                "--tools",
                "",
                "--system-prompt",
                "Reply with ok.",
                "--no-session-persistence",
            ]
        )
        env = build_claude_env(cfg_dir_env, path_env)
        secret_env: dict[str, str] = {}
        token = read_claude_oauth_token(config)
        if token:
            secret_env["CLAUDE_CODE_OAUTH_TOKEN"] = token

        spec = SandboxSpec(
            run_id="qprobe",
            reviewer_name=probe_rc.name,
            tree_host_dir=tree,
            inputs_host_dir=inputs,
            state_host_dir=state,
            harness_mounts=harness_mounts,
            command=cmd,
            cwd_in_sandbox=tree_path,
            env=env,
            secret_env=secret_env,
            vendor_hosts=config.vendor_hosts_for("anthropic"),
            extra_ro_paths=config.profile.extra_ro_paths,
            timeout_s=timeout_s,
            use_sandbox=use_sandbox,
        )
        sb_res = run_in_sandbox(
            spec,
            stdin_text="ok",
            raw_log_path=root / "raw.jsonl",
            proxy_log_path=root / "proxy.log",
            on_stdout_line=_on_line,
        )

    diag_err = "\n".join(stream_errors + ([sb_res.stderr] if sb_res.stderr else []))
    succeeded = sb_res.returncode == 0 and not sb_res.timed_out and saw_result and not is_error_result
    windows = tuple(rate_limit_windows.values())
    confirmed_exhausted = (not succeeded and (rate_limit_rejected or detect_quota_exhausted(diag_err))) or (
        succeeded and any(w.is_exhausted(now) for w in windows)
    )
    # Never overwrite usable cached quota with a failed/timed-out probe unless it was a confirmed quota rejection.
    if not succeeded and not confirmed_exhausted:
        if sb_res.timed_out:
            raise QuotaProbeError(f"claude haiku probe timed out after {timeout_s}s")
        diag = _collapse_ws(diag_err or "probe failed without quota rejection")
        raise QuotaProbeError(f"claude haiku probe exited {sb_res.returncode}: {diag}")
    if succeeded and not windows and not confirmed_exhausted:
        raise QuotaProbeError("claude haiku probe completed without emitting a rate_limit_event")

    exhausted_resets = [w.resets_at for w in windows if w.is_exhausted(now) and w.resets_at is not None]
    exhausted_until = (
        min(exhausted_resets)
        if exhausted_resets
        else (rate_limit_resets_at if confirmed_exhausted else None)
    )
    if confirmed_exhausted and exhausted_until is None:
        exhausted_until = parse_reset_hint_from_error(diag_err, now_ts=now)

    status = PlanQuotaStatus(
        plan_id=plan_id,
        harness="claude",
        windows=windows,
        exhausted=confirmed_exhausted,
        exhausted_until=exhausted_until,
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
    plan_ids: Collection[str] | None = None,
    force: bool = False,
    no_sandbox: bool = False,
    include_host_codex_rollouts: bool = True,
    now_ts: float | None = None,
    claude_probe_fn: Callable[[AppConfig, str, bool], PlanQuotaStatus | None] | None = None,
    agy_probe_fn: Callable[[AppConfig, str, bool], PlanQuotaStatus | None] | None = None,
) -> tuple[dict[str, PlanQuotaStatus], list[str]]:
    """Inspect plan quotas and run live `agy /usage` (0 tokens) and tiny Claude probes in parallel."""
    now = now_ts if now_ts is not None else time.time()
    allowed_plans = {p.strip().lower() for p in plan_ids} if plan_ids is not None else None
    statuses = inspect_plan_quotas(
        config,
        plan_ids=allowed_plans,
        include_host_codex_rollouts=include_host_codex_rollouts,
        now_ts=now,
    )
    warnings: list[str] = []

    # Derive probe targets strictly from configured reviewers, never from stale cached harness tags.
    configured_harnesses: dict[str, str] = {}
    for rc in config.reviewers:
        if rc.enabled:
            if allowed_plans is not None and rc.plan_id not in allowed_plans:
                continue
            configured_harnesses.setdefault(rc.plan_id, rc.harness)

    agy_plans = [pid for pid, h in configured_harnesses.items() if h == "agy"]
    claude_plans_all = [pid for pid, h in configured_harnesses.items() if h == "claude"]
    # Under credits="never", skip probing a selected Claude plan that is ALREADY cached as actively
    # exhausted (bounded by resets_at or DEFAULT_EXHAUSTED_COOLDOWN_S) unless explicitly forced.
    # Never broadcast exhaustion across unrelated plan IDs or accounts.
    claude_plans: list[str] = []
    for pid in claude_plans_all:
        existing_st = statuses.get(pid)
        if (
            not force
            and config.policy.credits == "never"
            and existing_st is not None
            and existing_st.is_exhausted(now)
        ):
            reset_str = format_reset_time(existing_st.next_reset_at(now))
            warnings.append(
                f"Skipped live Claude check for plan {pid!r} (cached exhausted; resets {reset_str}). "
                f"Run `xreview quota --refresh` to re-check."
            )
            continue
        claude_plans.append(pid)

    def _run_agy_batch() -> list[tuple[str, str, PlanQuotaStatus | None, Exception | None]]:
        if not agy_plans:
            return []
        if agy_probe_fn is not None:
            batch_out: list[tuple[str, str, PlanQuotaStatus | None, Exception | None]] = []
            for pid in agy_plans:
                try:
                    batch_out.append((pid, "agy", agy_probe_fn(config, pid, no_sandbox), None))
                except Exception as exc:
                    batch_out.append((pid, "agy", None, exc))
            return batch_out
        # Single `agy /usage` call serves all agy plans (e.g. `agy` Gemini + `agy-3p`) without concurrent agy runs.
        try:
            payload = _fetch_agy_usage_payload(config, no_sandbox=no_sandbox)
        except Exception as exc:
            return [(pid, "agy", None, exc) for pid in agy_plans]
        out_list: list[tuple[str, str, PlanQuotaStatus | None, Exception | None]] = []
        for pid in agy_plans:
            try:
                out_list.append((pid, "agy", _status_from_agy_payload(config, payload, pid, now), None))
            except Exception as exc:
                out_list.append((pid, "agy", None, exc))
        return out_list

    def _run_claude_shared_real() -> list[tuple[str, str, PlanQuotaStatus | None, Exception | None]]:
        # All Claude plans share one global Claude credential, so probe once and fan out to each plan_id.
        primary_pid = claude_plans[0]
        try:
            base = probe_claude_plan_quota(config, primary_pid, no_sandbox=no_sandbox, now_ts=now)
        except Exception as exc:
            return [(pid, "claude", None, exc) for pid in claude_plans]
        out_list: list[tuple[str, str, PlanQuotaStatus | None, Exception | None]] = [
            (primary_pid, "claude", base, None)
        ]
        for pid in claude_plans[1:]:
            cloned = PlanQuotaStatus(
                plan_id=pid,
                harness="claude",
                windows=base.windows,
                exhausted=base.exhausted,
                exhausted_until=base.exhausted_until,
                source=base.source,
                updated_at=base.updated_at,
            )
            try:
                record_plan_quota(config.state_dir, cloned)
            except OSError:
                pass
            out_list.append((pid, "claude", cloned, None))
        return out_list

    def _run_claude_injected_one(pid: str) -> list[tuple[str, str, PlanQuotaStatus | None, Exception | None]]:
        assert claude_probe_fn is not None
        try:
            return [(pid, "claude", claude_probe_fn(config, pid, no_sandbox), None)]
        except Exception as exc:
            return [(pid, "claude", None, exc)]

    def _run_claude_injected_batch() -> list[tuple[str, str, PlanQuotaStatus | None, Exception | None]]:
        out_list: list[tuple[str, str, PlanQuotaStatus | None, Exception | None]] = []
        for pid in claude_plans:
            out_list.extend(_run_claude_injected_one(pid))
        return out_list

    tasks: list[Callable[[], list[tuple[str, str, PlanQuotaStatus | None, Exception | None]]]] = []
    if agy_plans:
        tasks.append(_run_agy_batch)
    if claude_plans:
        if claude_probe_fn is None:
            tasks.append(_run_claude_shared_real)
        elif config.policy.serialize_same_harness.get("claude", False):
            tasks.append(_run_claude_injected_batch)
        else:
            for pid in claude_plans:
                tasks.append(lambda p=pid: _run_claude_injected_one(p))

    results: list[tuple[str, str, PlanQuotaStatus | None, Exception | None]] = []
    if len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            for chunk in pool.map(lambda fn: fn(), tasks):
                results.extend(chunk)
    else:
        for fn in tasks:
            results.extend(fn())

    for pid, harness, fresh, exc in results:
        if exc is not None:
            detail = _collapse_ws(str(exc)) or "(no message)"
            warnings.append(f"Live quota check for plan {pid!r} ({harness}) failed ({type(exc).__name__}): {detail}")
        elif fresh is not None:
            statuses[pid] = fresh
        else:
            warnings.append(f"Live quota check for plan {pid!r} ({harness}) returned no quota data.")

    return statuses, warnings

