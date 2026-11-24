"""Layered configuration for xreview (built-in defaults, user config, profile, untrusted project file)."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SECRET_PATTERNS: tuple[str, ...] = (
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*credentials*",
    "key.properties",
)

DEFAULT_HARNESS_VENDOR: dict[str, str] = {
    "codex": "openai",
    "agy": "google",
    "claude": "anthropic",
}

DEFAULT_VENDOR_HOSTS: dict[str, tuple[str, ...]] = {
    "openai": (
        "api.openai.com",
        "chatgpt.com",
        "ab.chatgpt.com",
        "auth.openai.com",
    ),
    "google": (
        "daily-cloudcode-pa.googleapis.com",
        "aicode.googleapis.com",
        "oauth2.googleapis.com",
        "www.googleapis.com",
        "play.googleapis.com",
    ),
    "anthropic": (
        "api.anthropic.com",
        "claude.ai",
    ),
}

TESTED_HARNESS_VERSIONS: dict[str, str] = {
    "codex": "0.171.0",
    "agy": "1.5.15",
    "claude": "2.1.322",
}

MAX_UNTRACKED_FILE_BYTES = 1_048_576  # 1 MB
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_REPO_NAMES = frozenset({".", "..", "_docs", "docs", "inputs", "tree", "raw", "state", "objects"})


class ConfigError(ValueError):
    """Raised when configuration is invalid or violates trust rules."""


def validate_run_id(run_id: str) -> str:
    """Validate a run_id is a single safe path component (no slashes, no '..')."""
    if not run_id or run_id in (".", "..") or not _SAFE_ID_RE.match(run_id):
        raise ConfigError(
            f"Invalid run_id {run_id!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ with no path separators."
        )
    return run_id


def validate_repo_alias(name: str) -> str:
    """Validate a repository alias is a unique, safe, non-reserved single path component."""
    if not name or name in _RESERVED_REPO_NAMES or not _SAFE_ID_RE.match(name) or len(name) > 64:
        raise ConfigError(
            f"Invalid repository alias {name!r}: must be a single safe identifier (max 64 chars, non-reserved)."
        )
    return name


def validate_reviewer_name(name: str) -> str:
    """Validate a reviewer name is a safe single path component (astra-52)."""
    if not name or name in _RESERVED_REPO_NAMES or not _SAFE_ID_RE.match(name) or len(name) > 64:
        raise ConfigError(
            f"Invalid reviewer name {name!r}: must be a safe single identifier (max 64 chars, no path separators)."
        )
    return name


def safe_read_regular_file_under(
    root_dir: Path,
    rel_path: str,
    max_bytes: int = 8_388_608,
    tail: bool = False,
) -> bytes | None:
    """Safely read a regular file strictly beneath root_dir without following any symlinks.

    Walks each directory component with O_DIRECTORY | O_NOFOLLOW and opens the leaf
    with O_RDONLY | O_NOFOLLOW | O_NONBLOCK. If tail=True and the file exceeds max_bytes,
    reads the last max_bytes bytes (for long rollout logs, astra-54).
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ConfigError(f"Unsafe relative path: {rel_path!r}")

    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

    try:
        cur_fd = os.open(str(root_dir), flags_dir)
    except OSError:
        return None

    try:
        for comp in parts[:-1]:
            try:
                next_fd = os.open(comp, flags_dir, dir_fd=cur_fd)
            except OSError:
                return None
            os.close(cur_fd)
            cur_fd = next_fd

        try:
            leaf_fd = os.open(parts[-1], flags_file, dir_fd=cur_fd)
        except OSError:
            return None
        try:
            st = os.fstat(leaf_fd)
            if not stat.S_ISREG(st.st_mode):
                return None
            if tail and st.st_size > max_bytes:
                os.lseek(leaf_fd, st.st_size - max_bytes, os.SEEK_SET)
            chunks: list[bytes] = []
            total = 0
            while total < max_bytes:
                chunk = os.read(leaf_fd, min(65536, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            return b"".join(chunks)
        finally:
            os.close(leaf_fd)
    finally:
        os.close(cur_fd)


def safe_open_user_regular_file(path: Path, max_bytes: int | None = None) -> bytes:
    """Open a user-supplied regular file refusing symlinks on EVERY traversed component (astra-6).

    Walks directory components with O_DIRECTORY | O_NOFOLLOW and reads directly from the
    verified file descriptor so TOCTOU replacement cannot swap the file.
    """
    raw_str = os.path.expanduser(str(path))
    p = Path(raw_str)
    if not raw_str or "\x00" in raw_str:
        raise ConfigError(f"Invalid file path: {path!r}")

    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)

    if p.is_absolute():
        cur_fd = os.open("/", flags_dir)
        parts = [c for c in p.parts[1:] if c not in ("", ".")]
    else:
        cur_fd = os.open(".", flags_dir)
        parts = [c for c in p.parts if c not in ("", ".")]

    if not parts or any(c == ".." for c in parts):
        os.close(cur_fd)
        raise ConfigError(f"Refusing path traversal or empty path: {path}")

    try:
        for comp in parts[:-1]:
            try:
                next_fd = os.open(comp, flags_dir, dir_fd=cur_fd)
            except OSError as exc:
                raise ConfigError(f"Cannot safely traverse directory {comp!r} in {path}: {exc}") from exc
            os.close(cur_fd)
            cur_fd = next_fd

        try:
            fd = os.open(parts[-1], flags_file, dir_fd=cur_fd)
        except OSError as exc:
            raise ConfigError(f"Cannot safely open regular file {path}: {exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ConfigError(f"Path is not a regular file: {path}")
            if max_bytes is not None and st.st_size > max_bytes:
                raise ConfigError(f"File exceeds size limit ({st.st_size} > {max_bytes} bytes): {path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise ConfigError(f"File exceeds size limit ({max_bytes} bytes): {path}")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(cur_fd)


def agy_pool_for_reviewer(rc: ReviewerConfig) -> str:
    """Return the Antigravity quota pool ('gemini' vs '3p') for an agy reviewer."""
    if rc.model.strip().lower().startswith("gemini") or rc.family.strip().lower() == "gemini":
        return "gemini"
    return "3p"


@dataclass(frozen=True)
class ReviewerConfig:
    name: str
    harness: str
    model: str
    family: str
    vendor: str
    effort: str | None = None
    extra: bool = False
    enabled: bool = True
    serialize_group: str | None = None
    plan: str | None = None

    @property
    def plan_id(self) -> str:
        if self.plan:
            return self.plan.strip().lower()
        h = self.harness.strip().lower()
        if h == "agy" and agy_pool_for_reviewer(self) != "gemini":
            return "agy-3p"
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "harness": self.harness,
            "model": self.model,
            "family": self.family,
            "vendor": self.vendor,
            "effort": self.effort,
            "extra": self.extra,
            "enabled": self.enabled,
            "serialize_group": self.serialize_group,
            "plan": self.plan,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewerConfig:
        name = validate_reviewer_name(str(data["name"]).strip())
        harness = str(data["harness"]).strip()
        vendor = str(data.get("vendor") or DEFAULT_HARNESS_VENDOR.get(harness, harness)).strip().lower()
        return cls(
            name=name,
            harness=harness,
            model=str(data["model"]).strip(),
            family=str(data["family"]).strip().lower(),
            vendor=vendor,
            effort=str(data["effort"]).strip() if data.get("effort") else None,
            extra=bool(data.get("extra", False)),
            enabled=bool(data.get("enabled", True)),
            serialize_group=str(data["serialize_group"]).strip() if data.get("serialize_group") else None,
            plan=str(data["plan"]).strip().lower() if data.get("plan") else None,
        )


@dataclass(frozen=True)
class PolicyConfig:
    credits: str = "never"
    min_families: int = 2
    soft_round_ceiling: int = 6
    shed_extras_at: float = 0.8
    fallback_to_extras_at: float | None = 0.75
    auto_drop_exhausted: bool = True
    min_families_on_exhaustion: int = 2
    quota_preflight: bool = True
    live_quota_probe: bool = True
    timeout_seconds: int = 0
    serialize_same_harness: dict[str, bool] = field(
        default_factory=lambda: {"codex": True, "agy": True, "claude": False}
    )

    @property
    def refresh_stale_quota(self) -> bool:
        """Backward-compatible alias for `live_quota_probe`."""
        return self.live_quota_probe


@dataclass(frozen=True)
class TagRule:
    name: str
    vendors: str | tuple[str, ...] = "ask"  # "ask" or tuple of allowed vendor names
    post_run_hooks: bool = True


@dataclass(frozen=True)
class RepoProfileRule:
    path: str
    exclude_submodules: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthConfig:
    claude_token_file: Path | None = None
    codex_auth_file: Path | None = None
    agy_token_file: Path | None = None


@dataclass(frozen=True)
class ProfileConfig:
    path: Path | None = None
    brief_rules: tuple[str, ...] = ()
    extra_secret_patterns: tuple[str, ...] = ()
    extra_ro_paths: tuple[str, ...] = ()
    vendor_hosts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    repo_rules: tuple[RepoProfileRule, ...] = ()
    tags: dict[str, TagRule] = field(default_factory=dict)
    post_run_hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectConfig:
    """Untrusted .xreview.toml read only from snapshot git objects."""

    instruction_files: tuple[str, ...] = ()
    notes: str = ""
    exclude_paths: tuple[str, ...] = ()
    exclude_submodules: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    reviewers: tuple[ReviewerConfig, ...] = ()
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    state_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "state" / "xreview")
    implementor_family: str | None = None

    def all_secret_patterns(self) -> tuple[str, ...]:
        return DEFAULT_SECRET_PATTERNS + self.profile.extra_secret_patterns

    def vendor_hosts_for(self, vendor: str) -> tuple[str, ...]:
        v = vendor.lower()
        base = DEFAULT_VENDOR_HOSTS.get(v, ())
        extra = self.profile.vendor_hosts.get(v, ())
        return tuple(dict.fromkeys((*base, *extra)))


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "xreview" / "config.toml"


def _expand(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(os.path.expanduser(p)).resolve()


def _parse_tag_rules(raw_tags: dict[str, Any]) -> dict[str, TagRule]:
    rules: dict[str, TagRule] = {}
    for name, body in raw_tags.items():
        if not isinstance(body, dict):
            raise ConfigError(f"[tags.{name}] must be a table")
        raw_vendors = body.get("vendors", "ask")
        if raw_vendors == "ask":
            vendors: str | tuple[str, ...] = "ask"
        elif isinstance(raw_vendors, list) and all(isinstance(v, str) for v in raw_vendors):
            vendors = tuple(v.strip().lower() for v in raw_vendors if v.strip())
        else:
            raise ConfigError(f"[tags.{name}].vendors must be 'ask' or a list of vendor names")
        hooks = bool(body.get("post_run_hooks", True))
        rules[name.strip()] = TagRule(name=name.strip(), vendors=vendors, post_run_hooks=hooks)
    return rules


def load_profile(profile_path: Path) -> ProfileConfig:
    if not profile_path.exists():
        raise ConfigError(f"Profile file not found: {profile_path}")
    data = tomllib.loads(profile_path.read_text(encoding="utf-8"))

    brief_rules = tuple(str(x) for x in data.get("brief_rules", ()))
    extra_secret_patterns = tuple(str(x) for x in data.get("extra_secret_patterns", ()))
    extra_ro_paths = tuple(str(x) for x in data.get("extra_ro_paths", ()))

    vendor_hosts: dict[str, tuple[str, ...]] = {}
    for v, hosts in data.get("vendor_hosts", {}).items():
        if not isinstance(hosts, list):
            raise ConfigError(f"vendor_hosts.{v} must be a list")
        vendor_hosts[v.strip().lower()] = tuple(str(h).strip().lower() for h in hosts if str(h).strip())

    repo_rules: list[RepoProfileRule] = []
    for item in data.get("repo_rules", ()):
        if not isinstance(item, dict) or "path" not in item:
            raise ConfigError("Each [[repo_rules]] entry requires 'path'")
        repo_rules.append(
            RepoProfileRule(
                path=str(_expand(item["path"])),
                exclude_submodules=tuple(str(s) for s in item.get("exclude_submodules", ())),
                tags=tuple(str(t) for t in item.get("tags", ())),
            )
        )

    tags = _parse_tag_rules(data.get("tags", {}))
    hooks_tbl = data.get("hooks", {})
    post_run = tuple(str(c) for c in hooks_tbl.get("post_run", ())) if isinstance(hooks_tbl, dict) else ()

    return ProfileConfig(
        path=profile_path,
        brief_rules=brief_rules,
        extra_secret_patterns=extra_secret_patterns,
        extra_ro_paths=extra_ro_paths,
        vendor_hosts=vendor_hosts,
        repo_rules=tuple(repo_rules),
        tags=tags,
        post_run_hooks=post_run,
    )


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or default_config_path()
    if not path.exists():
        return AppConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    raw_reviewers = data.get("reviewers", ())
    reviewers: list[ReviewerConfig] = []
    seen_names: set[str] = set()
    for entry in raw_reviewers:
        if not isinstance(entry, dict):
            raise ConfigError("Each entry in 'reviewers' must be a table")
        for req in ("name", "harness", "model", "family"):
            if req not in entry:
                raise ConfigError(f"Reviewer entry missing required field {req!r}")
        rc = ReviewerConfig.from_dict(entry)
        if rc.name in seen_names:
            raise ConfigError(f"Duplicate reviewer name: {rc.name!r}")
        seen_names.add(rc.name)
        reviewers.append(rc)

    pol_raw = data.get("policy", {})
    ser_default = {"codex": True, "agy": True, "claude": False}
    if isinstance(pol_raw.get("serialize_same_harness"), dict):
        for k, v in pol_raw["serialize_same_harness"].items():
            ser_default[str(k)] = bool(v)

    raw_fallback = pol_raw.get("fallback_to_extras_at", 0.75)
    fallback_at: float | None = None if raw_fallback in (None, False, 0, 0.0, "") else float(raw_fallback)

    raw_timeout = pol_raw.get("timeout_seconds", 0)
    if raw_timeout is False or raw_timeout in (None, ""):
        timeout_seconds = 0
    elif isinstance(raw_timeout, bool):
        raise ConfigError("policy.timeout_seconds must be a non-negative integer (0 = no timeout)")
    else:
        timeout_seconds = int(raw_timeout)
    if timeout_seconds < 0:
        raise ConfigError("policy.timeout_seconds must be >= 0 (0 = no timeout)")

    policy = PolicyConfig(
        credits=str(pol_raw.get("credits", "never")),
        min_families=int(pol_raw.get("min_families", 2)),
        soft_round_ceiling=int(pol_raw.get("soft_round_ceiling", 6)),
        shed_extras_at=float(pol_raw.get("shed_extras_at", 0.8)),
        fallback_to_extras_at=fallback_at,
        auto_drop_exhausted=bool(pol_raw.get("auto_drop_exhausted", True)),
        min_families_on_exhaustion=int(pol_raw.get("min_families_on_exhaustion", 2)),
        quota_preflight=bool(pol_raw.get("quota_preflight", True)),
        live_quota_probe=bool(pol_raw.get("live_quota_probe", pol_raw.get("refresh_stale_quota", True))),
        timeout_seconds=timeout_seconds,
        serialize_same_harness=ser_default,
    )
    if policy.credits not in ("never", "allow"):
        raise ConfigError(f"policy.credits must be 'never' or 'allow', got {policy.credits!r}")
    if policy.min_families < 1:
        raise ConfigError("policy.min_families must be >= 1")
    if policy.min_families_on_exhaustion < 2:
        raise ConfigError("policy.min_families_on_exhaustion must be >= 2")
    if policy.soft_round_ceiling < 1:
        raise ConfigError("policy.soft_round_ceiling must be >= 1")
    if not (0.0 < policy.shed_extras_at <= 1.0):
        raise ConfigError("policy.shed_extras_at must be in (0.0, 1.0]")
    if policy.fallback_to_extras_at is not None and not (0.0 < policy.fallback_to_extras_at <= 1.0):
        raise ConfigError("policy.fallback_to_extras_at must be in (0.0, 1.0] or disabled")

    profile = ProfileConfig()
    if data.get("profile"):
        profile = load_profile(_expand(str(data["profile"])) or Path(str(data["profile"])))

    auth_raw = data.get("auth", {})
    auth = AuthConfig(
        claude_token_file=_expand(auth_raw.get("claude_token_file")),
        codex_auth_file=_expand(auth_raw.get("codex_auth_file")),
        agy_token_file=_expand(auth_raw.get("agy_token_file")),
    )

    state_dir = _expand(data.get("state_dir")) or (Path.home() / ".local" / "state" / "xreview")
    impl_family = str(data["implementor_family"]).strip().lower() if data.get("implementor_family") else None

    return AppConfig(
        reviewers=tuple(reviewers),
        policy=policy,
        profile=profile,
        auth=auth,
        state_dir=state_dir,
        implementor_family=impl_family,
    )


def validate_safe_relpath(raw: str) -> str:
    """Reject absolute paths, '..' segments, NUL bytes, or empty paths."""
    if not raw or "\x00" in raw:
        raise ConfigError(f"Invalid repository path: {raw!r}")
    norm = raw.replace("\\", "/")
    if norm.startswith("/"):
        raise ConfigError(f"Absolute path forbidden in .xreview.toml: {raw!r}")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ConfigError(f"Path traversal forbidden in .xreview.toml: {raw!r}")
    return "/".join(parts)


def parse_untrusted_project_toml(raw_bytes: bytes) -> ProjectConfig:
    """Parse untrusted .xreview.toml content read from git objects only."""
    data = tomllib.loads(raw_bytes.decode("utf-8"))
    allowed_keys = {"instruction_files", "notes", "exclude_paths", "exclude_submodules"}
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        raise ConfigError(
            f"Untrusted .xreview.toml contains disallowed keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    instr = tuple(validate_safe_relpath(str(p)) for p in data.get("instruction_files", ()))
    notes = str(data.get("notes", ""))
    excl_paths = tuple(validate_safe_relpath(str(p)) for p in data.get("exclude_paths", ()))
    excl_subs = tuple(validate_safe_relpath(str(p)) for p in data.get("exclude_submodules", ()))

    return ProjectConfig(
        instruction_files=instr,
        notes=notes,
        exclude_paths=excl_paths,
        exclude_submodules=excl_subs,
    )


def is_secret_path(rel_path: str, patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS) -> bool:
    """Return True if any path segment or the full path matches a secret filename pattern."""
    norm = rel_path.replace("\\", "/")
    name = norm.rsplit("/", 1)[-1]
    lower_norm = norm.lower()
    lower_name = name.lower()
    for pat in patterns:
        lp = pat.lower()
        if fnmatch.fnmatch(lower_name, lp) or fnmatch.fnmatch(lower_norm, lp):
            return True
    return False
