"""Base interface and registry for pluggable reviewer harness adapters."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from xreview.config import AppConfig, ReviewerConfig

_QUOTA_PATTERNS = (
    re.compile(r"rate[_\s-]?limit[_\s-]?reached", re.IGNORECASE),
    re.compile(r"usage limit", re.IGNORECASE),
    re.compile(r"quota exceeded|out of quota|RESOURCE_EXHAUSTED", re.IGNORECASE),
    re.compile(r"insufficient[_\s-]?quota|credit balance|purchase credits", re.IGNORECASE),
    re.compile(r"hit your .*limit|plan limit reached|weekly limit", re.IGNORECASE),
    re.compile(r"\b429\b.*(?:rate|quota|limit)", re.IGNORECASE),
)


_RESUME_FAILURE_PATTERNS = (
    re.compile(
        r"\b(?:session|conversation|thread)\b[^\n]{0,80}?"
        r"\b(?:not found|missing|unknown|invalid|expired|corrupt(?:ed)?|does not exist|no such)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:could not|failed to|unable to|cannot)\s+(?:resume|load|find)\b[^\n]{0,80}?"
        r"\b(?:session|conversation|thread)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|unknown|invalid|expired|missing)\s+(?:session|conversation|thread)\b",
        re.IGNORECASE,
    ),
)


def detect_quota_exhausted(text: str) -> bool:
    """Return True if error diagnostics indicate subscription quota exhaustion."""
    if not text:
        return False
    return any(p.search(text) for p in _QUOTA_PATTERNS)


_AUTH_SESSION_EXCLUSION = re.compile(
    r"\b(?:oauth|auth|login|token|credential|cookie)\s+session\b",
    re.IGNORECASE,
)


def detect_resume_failed(text: str) -> bool:
    """Return True if harness error diagnostics indicate a missing/invalid/expired warm session (F-017, F-023)."""
    if not text:
        return False
    cleaned = _AUTH_SESSION_EXCLUSION.sub("", text)
    return any(p.search(cleaned) for p in _RESUME_FAILURE_PATTERNS)


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw text (stripping optional markdown fences)."""
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            s = "\n".join(lines[1:-1]).strip()
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class TurnRequest:
    run_id: str
    round_num: int
    reviewer: ReviewerConfig
    prompt: str
    schema_host_path: Path
    tree_host_dir: Path
    inputs_host_dir: Path
    state_host_dir: Path
    raw_log_path: Path
    proxy_log_path: Path
    docs_host_dir: Path | None = None
    prompt_filename: str = "PROMPT.md"
    session_id: str | None = None
    timeout_s: int = 1200
    allow_no_sandbox: bool = False
    on_session_id: Callable[[str], None] | None = None


@dataclass
class TurnResult:
    session_id: str | None = None
    reply: dict[str, Any] | None = None
    raw_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    allowance_used_ratio: float | None = None
    quota_windows: list[dict[str, Any]] = field(default_factory=list)
    quota_resets_at: float | None = None
    quota_exhausted: bool = False
    resume_failed: bool = False
    malformed_output: bool = False
    timed_out: bool = False
    error: str | None = None
    duration_s: float = 0.0
    returncode: int = 0


class HarnessAdapter(ABC):
    name: str
    tested_version: str = ""

    @abstractmethod
    def version(self) -> str | None:
        """Return installed harness version string, or None if unavailable."""

    @abstractmethod
    def prepare_state(self, reviewer: ReviewerConfig, state_dir: Path, config: AppConfig) -> None:
        """Prepare isolated per-reviewer state directory (auth files, credit-off settings)."""

    @abstractmethod
    def run_turn(self, req: TurnRequest, config: AppConfig) -> TurnResult:
        """Run one prompt turn (fresh or resumed) inside the sandbox."""

    def cleanup(self, reviewer: ReviewerConfig, state_dir: Path, session_id: str | None) -> None:
        """Clean up reviewer state directory."""
        if state_dir.exists() and not state_dir.is_symlink():
            shutil.rmtree(state_dir, ignore_errors=True)

    @staticmethod
    def _run_version_cmd(binary: str, args: list[str] | None = None) -> str | None:
        exe = shutil.which(binary)
        if not exe:
            return None
        try:
            res = subprocess.run(
                [exe, *(args or ["--version"])],
                capture_output=True,
                text=True,
                timeout=10,
            )
            out = (res.stdout or res.stderr).strip()
            return out.splitlines()[0].strip() if out else None
        except (OSError, subprocess.SubprocessError):
            return None


_ADAPTERS: dict[str, HarnessAdapter] = {}


def register_adapter(adapter: HarnessAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> HarnessAdapter:
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown harness adapter: {name!r} (registered: {sorted(_ADAPTERS)})")
    return _ADAPTERS[name]


def list_adapters() -> dict[str, HarnessAdapter]:
    return dict(_ADAPTERS)
