"""Read-only git snapshot engine (zero writes to source repos, submodule support, symlink-safe)."""

from __future__ import annotations

import configparser
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from xreview.config import (
    DEFAULT_SECRET_PATTERNS,
    MAX_UNTRACKED_FILE_BYTES,
    ConfigError,
    ProjectConfig,
    is_secret_path,
    parse_untrusted_project_toml,
    safe_open_user_regular_file,
    validate_safe_relpath,
)

EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


class SnapshotError(RuntimeError):
    """Raised when snapshotting fails or violates safety checks."""


class UnclassifiedFilesError(SnapshotError):
    """Raised when untracked, non-ignored files need --include or --exclude."""

    def __init__(self, files: list[str]) -> None:
        self.files = files
        super().__init__(
            "Unclassified untracked files found (pass --include or --exclude):\n  "
            + "\n  ".join(files)
        )


@dataclass(frozen=True)
class RepoSpec:
    name: str
    path: Path
    base_rev: str = "HEAD"
    target_rev: str | None = None  # None means working tree

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "base_rev": self.base_rev,
            "target_rev": self.target_rev,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepoSpec:
        return cls(
            name=str(data["name"]),
            path=Path(data["path"]),
            base_rev=str(data.get("base_rev", "HEAD")),
            target_rev=str(data["target_rev"]) if data.get("target_rev") else None,
        )


@dataclass
class RepoRoundSnapshot:
    repo_name: str
    repo_path: str
    base_sha: str
    commit_sha: str
    tree_sha: str
    submodules: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # sp -> (base_sha, snap_sha, tree_sha)
    submodule_git_dirs: dict[str, str] = field(default_factory=dict)  # sp -> verified git_dir (astra-20)
    submodule_obj_dirs: dict[str, str] = field(default_factory=dict)  # sp -> run-local objects dir (F-020)
    included_submodule_host_paths: list[str] = field(default_factory=list)
    excluded_submodules: list[str] = field(default_factory=list)
    project_exclusions: list[str] = field(default_factory=list)
    project_notes: str = ""
    project_instruction_contents: dict[str, str] = field(default_factory=dict)
    agents_md_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repo_name": self.repo_name,
            "repo_path": self.repo_path,
            "base_sha": self.base_sha,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "submodules": {k: list(v) for k, v in self.submodules.items()},
            "submodule_git_dirs": dict(self.submodule_git_dirs),
            "submodule_obj_dirs": dict(self.submodule_obj_dirs),
            "included_submodule_host_paths": list(self.included_submodule_host_paths),
            "excluded_submodules": list(self.excluded_submodules),
            "project_exclusions": list(self.project_exclusions),
            "project_notes": self.project_notes,
            "project_instruction_contents": dict(self.project_instruction_contents),
            "agents_md_paths": list(self.agents_md_paths),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepoRoundSnapshot:
        subs: dict[str, tuple[str, str, str]] = {}
        for k, v in data.get("submodules", {}).items():
            if len(v) >= 3:
                subs[k] = (str(v[0]), str(v[1]), str(v[2]))
            else:
                subs[k] = (str(v[0]), str(v[1]), "")
        return cls(
            repo_name=str(data["repo_name"]),
            repo_path=str(data["repo_path"]),
            base_sha=str(data["base_sha"]),
            commit_sha=str(data["commit_sha"]),
            tree_sha=str(data["tree_sha"]),
            submodules=subs,
            submodule_git_dirs={k: str(v) for k, v in data.get("submodule_git_dirs", {}).items()},
            submodule_obj_dirs={k: str(v) for k, v in data.get("submodule_obj_dirs", {}).items()},
            included_submodule_host_paths=list(data.get("included_submodule_host_paths", ())),
            excluded_submodules=list(data.get("excluded_submodules", ())),
            project_exclusions=list(data.get("project_exclusions", ())),
            project_notes=str(data.get("project_notes", "")),
            project_instruction_contents=dict(data.get("project_instruction_contents", {})),
            agents_md_paths=list(data.get("agents_md_paths", ())),
        )


@dataclass
class RoundSnapshotResult:
    round_num: int
    tree_dir: Path
    docs_dir: Path | None
    repo_snapshots: dict[str, RepoRoundSnapshot]
    base_diff_bytes: bytes
    round_diff_bytes: bytes
    changed_lines_in_round: dict[str, list[tuple[int, int]]]
    project_exclusions: list[str]
    covered_host_paths: list[str]

    @property
    def base_diff(self) -> str:
        return self.base_diff_bytes.decode("utf-8", errors="replace")

    @property
    def round_diff(self) -> str:
        return self.round_diff_bytes.decode("utf-8", errors="replace")


# Safe Git config flags that prevent hooks, external diff drivers, fsmonitor,
# or LFS filters from executing external commands or writing to the source repo.
_SAFE_GIT_CONFIG_ARGS: tuple[str, ...] = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.autocrlf=false",
    "-c", "core.quotepath=false",
    "-c", "core.attributesFile=/dev/null",
    "-c", "core.sshCommand=/bin/false",
    "-c", "diff.external=",
    "-c", "filter.lfs.smudge=cat",
    "-c", "filter.lfs.clean=cat",
    "-c", "filter.lfs.process=",
    "-c", "filter.lfs.required=false",
)


def _dynamic_neutralize_repo_config_args(repo_path: Path, env: dict[str, str]) -> list[str]:
    """Query ALL effective repo/included/worktree config keys via git config --list --includes --null
    and neutralize every custom filter/diff driver case-insensitively (astra-8)."""
    extra: list[str] = []
    res = subprocess.run(
        ["git", "config", "--list", "--includes", "--null"],
        cwd=repo_path,
        env=env,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout:
        return extra

    filters: set[str] = set()
    diffs: set[str] = set()
    for entry in res.stdout.split(b"\x00"):
        if not entry:
            continue
        key_b = entry.split(b"\n", 1)[0]
        key = key_b.decode("utf-8", errors="replace")
        low = key.lower()
        if low.startswith("filter."):
            rest = key[len("filter.") :]
            if "." in rest:
                filters.add(rest.rsplit(".", 1)[0])
        elif low.startswith("diff."):
            rest = key[len("diff.") :]
            if "." in rest:
                diffs.add(rest.rsplit(".", 1)[0])

    for name in sorted(filters):
        extra.extend([
            "-c", f"filter.{name}.clean=cat",
            "-c", f"filter.{name}.smudge=cat",
            "-c", f"filter.{name}.process=",
            "-c", f"filter.{name}.required=false",
        ])
    for name in sorted(diffs):
        extra.extend([
            "-c", f"diff.{name}.command=",
            "-c", f"diff.{name}.textconv=",
        ])
    return extra


def _git(
    args: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    base_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_EXTERNAL_DIFF": "",
        "GIT_AUTHOR_NAME": "xreview",
        "GIT_AUTHOR_EMAIL": "xreview@local",
        "GIT_COMMITTER_NAME": "xreview",
        "GIT_COMMITTER_EMAIL": "xreview@local",
    }
    if env:
        base_env.update(env)
    dyn_args = _dynamic_neutralize_repo_config_args(cwd, base_env)
    res = subprocess.run(
        ["git", *_SAFE_GIT_CONFIG_ARGS, *dyn_args, *args],
        cwd=cwd,
        env=base_env,
        capture_output=True,
        text=text,
    )
    if check and res.returncode != 0:
        err = res.stderr if text else res.stderr.decode("utf-8", errors="replace")
        raise SnapshotError(f"git {' '.join(args)} failed in {cwd} ({res.returncode}): {err.strip()}")
    return res


def resolve_commit_sha(repo_path: Path, rev: str, *, allow_empty_tree: bool = False) -> str:
    """Resolve and pin a git revision to its 40-char commit SHA (or EMPTY_TREE_SHA for root commits, astra-67)."""
    if allow_empty_tree and rev.strip() in (EMPTY_TREE_SHA, "EMPTY", "empty"):
        return EMPTY_TREE_SHA
    return _git(["rev-parse", f"{rev}^{{commit}}"], cwd=repo_path).stdout.strip()


def git_dirs_for_repo(repo_path: Path) -> tuple[Path, Path]:
    """Return (worktree_git_dir, common_git_dir) for a repository."""
    return _git_dirs(repo_path)


def verify_repo_is_worktree_root(repo_path: Path) -> Path:
    """Verify repo_path is a Git worktree top-level root, not a subdirectory (astra-53)."""
    resolved = repo_path.resolve()
    top = _git(["rev-parse", "--show-toplevel"], cwd=resolved).stdout.strip()
    if Path(top).resolve() != resolved:
        raise SnapshotError(
            f"--repo must point to the Git repository root ({top}), not a subdirectory ({resolved})."
        )
    return resolved


def _global_excludes_file(repo_path: Path) -> Path | None:
    """Discover user's trusted global/conditional Git ignore file as data in repo context (astra-55)."""
    try:
        res = subprocess.run(
            ["git", "config", "--includes", "--path", "--get", "core.excludesFile"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            p = Path(os.path.expanduser(res.stdout.strip()))
            if p.is_file():
                return p
    except (OSError, subprocess.SubprocessError):
        pass
    xdg = os.environ.get("XDG_CONFIG_HOME")
    default_p = (Path(xdg) if xdg else Path.home() / ".config") / "git" / "ignore"
    return default_p if default_p.is_file() else None


def _git_dirs(repo_path: Path) -> tuple[Path, Path]:
    """Return (worktree_git_dir, common_git_dir) separately so linked worktrees work."""
    gd_out = _git(["rev-parse", "--git-dir"], cwd=repo_path).stdout.strip()
    cd_out = _git(["rev-parse", "--git-common-dir"], cwd=repo_path).stdout.strip()
    gd = Path(gd_out)
    if not gd.is_absolute():
        gd = (repo_path / gd).resolve()
    cd = Path(cd_out)
    if not cd.is_absolute():
        cd = (repo_path / cd).resolve()
    return gd, cd


def _obj_env(objects_dir: Path, common_git_dir: Path, index_file: Path | None = None) -> dict[str, str]:
    env = {
        "GIT_OBJECT_DIRECTORY": str(objects_dir),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(common_git_dir / "objects"),
    }
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    return env


def _has_symlink_component(root: Path, rel_path: str) -> bool:
    """Return True if any path component from root through rel_path is a symlink."""
    cur = root
    for comp in [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".")]:
        cur = cur / comp
        try:
            st = os.lstat(cur)
        except OSError:
            return False
        if stat.S_ISLNK(st.st_mode):
            return True
    return False


def read_blob_from_git_object(
    repo_path: Path,
    commit_sha: str,
    rel_path: str,
    env: dict[str, str],
) -> bytes | None:
    """Read a regular file from a snapshot commit strictly via git objects.

    Refuses symlinks (mode 120000), submodules (160000), directories, and '..'.
    """
    safe = validate_safe_relpath(rel_path)
    ls = _git(["ls-tree", "-z", commit_sha, "--", safe], cwd=repo_path, env=env, check=False, text=False)
    if ls.returncode != 0 or not ls.stdout:
        return None
    entry = ls.stdout.split(b"\x00")[0]
    if not entry:
        return None
    meta_b, _, _path_b = entry.partition(b"\t")
    parts = meta_b.decode("ascii", errors="replace").split()
    if len(parts) < 3:
        return None
    mode, obj_type, obj_sha = parts[0], parts[1], parts[2]
    if mode == "120000":
        raise ConfigError(f"Refusing to read symlink from snapshot: {safe!r}")
    if obj_type != "blob" or mode not in ("100644", "100755"):
        raise ConfigError(f"Refusing to read non-regular git object ({mode} {obj_type}): {safe!r}")
    cat = _git(["cat-file", "blob", obj_sha], cwd=repo_path, env=env, text=False)
    return cat.stdout


def list_untracked_files(repo_path: Path) -> list[str]:
    args = ["ls-files", "--others", "--exclude-standard", "-z"]
    global_ignore = _global_excludes_file(repo_path)
    if global_ignore is not None:
        args.append(f"--exclude-from={global_ignore}")
    out = _git(args, cwd=repo_path, text=False).stdout
    if not out:
        return []
    return sorted(p.decode("utf-8", errors="surrogateescape") for p in out.split(b"\x00") if p)


def _discover_submodules(repo_path: Path, rev: str, env: dict[str, str]) -> dict[str, str]:
    """Return {sub_path: gitlink_sha} for all 160000 entries in rev using NUL-delimited ls-tree."""
    out = _git(["ls-tree", "-r", "-z", rev], cwd=repo_path, env=env, text=False).stdout
    subs: dict[str, str] = {}
    for entry in out.split(b"\x00"):
        if not entry:
            continue
        meta_b, _, path_b = entry.partition(b"\t")
        parts = meta_b.decode("ascii", errors="replace").split()
        if len(parts) >= 3 and parts[0] == "160000":
            subs[path_b.decode("utf-8", errors="surrogateescape")] = parts[2]
    return subs


def _expected_submodule_name(
    parent_repo: Path, sub_path: str, gitmodules_bytes: bytes | None = None
) -> tuple[str | None, str]:
    """Return (explicit_logical_name_from_gitmodules_or_None, fallback_sub_path)."""
    raw_text: str | None = None
    if gitmodules_bytes is not None:
        raw_text = gitmodules_bytes.decode("utf-8", errors="replace")
    else:
        gm = parent_repo / ".gitmodules"
        if gm.is_file() and not gm.is_symlink():
            try:
                raw_text = gm.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw_text = None
    if raw_text:
        try:
            cp = configparser.ConfigParser(interpolation=None)
            cp.read_string(raw_text)
            for sec in cp.sections():
                m = re.match(r'^submodule\s+"(.+)"$', sec.strip())
                if m and cp.get(sec, "path", fallback="").strip() == sub_path:
                    return m.group(1), sub_path
        except configparser.Error:
            pass
    return None, sub_path


def _confined_module_dir(parent_common_dir: Path, mod_name: str) -> Path | None:
    """Resolve <parent_common_dir>/modules/<mod_name> ensuring strict confinement beneath modules/ (astra-7)."""
    try:
        safe_name = validate_safe_relpath(mod_name)
    except ConfigError:
        return None
    modules_root = (parent_common_dir / "modules").resolve()
    cand = (modules_root / safe_name).resolve()
    if cand == modules_root or not str(cand).startswith(str(modules_root) + os.sep):
        return None
    return cand


def _is_submodule_initialized_safely(parent_repo: Path, parent_common_dir: Path, sub_path: str) -> bool:
    """Verify sub_path is a non-symlink dir whose git & common dirs strictly match THIS module (astra-7)."""
    if _has_symlink_component(parent_repo, sub_path):
        return False
    sub_dir = parent_repo / sub_path
    if not sub_dir.is_dir():
        return False
    dot_git = sub_dir / ".git"
    if not dot_git.exists() or dot_git.is_symlink():
        return False
    try:
        sub_gd, sub_cd = _git_dirs(sub_dir)
    except SnapshotError:
        return False

    sub_gd_res = sub_gd.resolve()
    sub_cd_res = sub_cd.resolve()
    if sub_gd_res != sub_cd_res:
        return False

    explicit_name, fallback = _expected_submodule_name(parent_repo, sub_path)
    if explicit_name is not None:
        # When .gitmodules declares an explicit logical name, require THAT exact confined module dir!
        expected_mod_dir = _confined_module_dir(parent_common_dir, explicit_name)
        if expected_mod_dir is not None and sub_gd_res == expected_mod_dir:
            return True
    else:
        fallback_mod_dir = _confined_module_dir(parent_common_dir, fallback)
        if fallback_mod_dir is not None and sub_gd_res == fallback_mod_dir:
            return True

    if dot_git.is_dir() and sub_gd_res == dot_git.resolve():
        return True
    return False


def _submodule_has_changes(
    parent_repo: Path,
    parent_common_dir: Path,
    sub_path: str,
    base_sub_sha: str | None,
    target_rev: str | None,
    target_sub_sha: str | None,
    has_included_files: bool = False,
) -> bool:
    if has_included_files or base_sub_sha != target_sub_sha:
        return True
    if target_rev is not None:
        return False
    if not _is_submodule_initialized_safely(parent_repo, parent_common_dir, sub_path):
        return False
    sub_dir = parent_repo / sub_path
    head_sha = _git(["rev-parse", "HEAD"], cwd=sub_dir, check=False).stdout.strip()
    if base_sub_sha and head_sha and head_sha != base_sub_sha:
        return True
    # Check tracked changes + recursive submodule status regardless of repo's showUntrackedFiles setting (astra-72)
    st = _git(["status", "--porcelain", "--untracked-files=normal"], cwd=sub_dir).stdout.strip()
    return bool(st)


def validate_included_file(
    repo_path: Path,
    rel_path: str,
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS,
) -> None:
    """Refuse secret-like files, symlinks, non-regular files, or files > 1 MB even when --include is passed."""
    if is_secret_path(rel_path, secret_patterns):
        raise SnapshotError(f"Refusing to include secret-like file: {rel_path}")
    if _has_symlink_component(repo_path, rel_path):
        raise SnapshotError(f"Refusing to include symlinked untracked path: {rel_path}")
    full = repo_path / rel_path
    try:
        st = os.lstat(full)
    except OSError as exc:
        raise SnapshotError(f"Included untracked file is unreadable: {rel_path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise SnapshotError(f"Refusing to include non-regular untracked file: {rel_path}")
    if st.st_size > MAX_UNTRACKED_FILE_BYTES:
        raise SnapshotError(f"Refusing to include file over 1 MB ({st.st_size} bytes): {rel_path}")


def _stage_untracked_via_descriptor(
    repo_path: Path,
    rel_path: str,
    env: dict[str, str],
    secret_patterns: tuple[str, ...],
) -> None:
    """Capture an untracked regular file via verified O_NOFOLLOW FD, write blob, and stage exact cacheinfo (astra-74)."""
    if is_secret_path(rel_path, secret_patterns):
        raise SnapshotError(f"Refusing to include secret-like file: {rel_path}")
    # Open leaf via descriptor-relative O_DIRECTORY|O_NOFOLLOW walk and verify S_ISREG & size <= 1 MB
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise SnapshotError(f"Unsafe untracked path: {rel_path!r}")
    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    cur_fd = os.open(str(repo_path), flags_dir)
    try:
        for comp in parts[:-1]:
            try:
                nxt = os.open(comp, flags_dir, dir_fd=cur_fd)
            except OSError as exc:
                raise SnapshotError(f"Refusing symlinked or unreadable directory in {rel_path}: {exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt
        try:
            leaf_fd = os.open(parts[-1], flags_file, dir_fd=cur_fd)
        except OSError as exc:
            raise SnapshotError(f"Refusing symlinked or unreadable untracked file {rel_path}: {exc}") from exc
        try:
            st = os.fstat(leaf_fd)
            if not stat.S_ISREG(st.st_mode):
                raise SnapshotError(f"Refusing non-regular untracked file: {rel_path}")
            if st.st_size > MAX_UNTRACKED_FILE_BYTES:
                raise SnapshotError(f"Refusing to include file over 1 MB ({st.st_size} bytes): {rel_path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(leaf_fd, 65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UNTRACKED_FILE_BYTES:
                    raise SnapshotError(f"Refusing to include file over 1 MB: {rel_path}")
                chunks.append(chunk)
            data = b"".join(chunks)
            mode = "100755" if (st.st_mode & 0o111) else "100644"
        finally:
            os.close(leaf_fd)
    finally:
        os.close(cur_fd)

    # Write exact verified bytes to run-local GIT_OBJECT_DIRECTORY via git hash-object
    full_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        **env,
    }
    ho = subprocess.run(
        ["git", *_SAFE_GIT_CONFIG_ARGS, "hash-object", "-w", "--stdin"],
        cwd=repo_path,
        env=full_env,
        input=data,
        capture_output=True,
    )
    if ho.returncode != 0:
        raise SnapshotError(f"git hash-object failed for {rel_path}: {ho.stderr.decode('utf-8', 'replace')}")
    blob_sha = ho.stdout.decode("ascii").strip()

    # Stage exact (mode, blob_sha, path) in temporary index without reopening path on disk!
    ui_input = f"{mode} {blob_sha}\t".encode("ascii") + os.fsencode(rel_path) + b"\x00"
    ui = subprocess.run(
        ["git", *_SAFE_GIT_CONFIG_ARGS, "update-index", "-z", "--index-info"],
        cwd=repo_path,
        env=full_env,
        input=ui_input,
        capture_output=True,
    )
    if ui.returncode != 0:
        raise SnapshotError(f"git update-index failed for {rel_path}: {ui.stderr.decode('utf-8', 'replace')}")


def _snapshot_single_git_dir(
    repo_path: Path,
    objects_dir: Path,
    base_rev: str | None,
    target_rev: str | None,
    include_untracked: list[str],
    secret_patterns: tuple[str, ...],
    sub_commits: dict[str, str] | None = None,
    reuse_if_tree_matches: list[tuple[str, str]] | None = None,
) -> tuple[str, str, str]:
    """Create a snapshot commit in objects_dir without touching repo_path/.git."""
    objects_dir.mkdir(parents=True, exist_ok=True)
    real_gd, common_gd = _git_dirs(repo_path)
    base_sha = (
        resolve_commit_sha(repo_path, base_rev, allow_empty_tree=True)
        if base_rev is not None
        else EMPTY_TREE_SHA
    )

    if target_rev is not None:
        env = _obj_env(objects_dir, common_gd)
        commit_sha = _git(["rev-parse", f"{target_rev}^{{commit}}"], cwd=repo_path, env=env).stdout.strip()
        tree_sha = _git(["rev-parse", f"{commit_sha}^{{tree}}"], cwd=repo_path, env=env).stdout.strip()
        return base_sha, commit_sha, tree_sha

    tmp_index = objects_dir / f"tmp_index_{time.time_ns()}_{os.getpid()}"
    real_index = real_gd / "index"
    if real_index.exists():
        shutil.copy2(real_index, tmp_index)
    env = _obj_env(objects_dir, common_gd, tmp_index)

    try:
        _git(["add", "-u"], cwd=repo_path, env=env)
        current_untracked = set(list_untracked_files(repo_path))
        exact_includes = [rel for rel in include_untracked if rel in current_untracked]
        for rel in exact_includes:
            _stage_untracked_via_descriptor(repo_path, rel, env, secret_patterns)

        if sub_commits:
            for sp, sub_sha in sub_commits.items():
                _git(["update-index", "--cacheinfo", f"160000,{sub_sha},{sp}"], cwd=repo_path, env=env)

        tree_sha = _git(["write-tree"], cwd=repo_path, env=env).stdout.strip()

        if reuse_if_tree_matches:
            for cand_tree, cand_commit in reuse_if_tree_matches:
                if cand_tree and cand_commit and tree_sha == cand_tree:
                    return base_sha, cand_commit, tree_sha

        parent_args = ["-p", base_sha] if base_sha != EMPTY_TREE_SHA else []
        commit_sha = _git(
            ["commit-tree", tree_sha, *parent_args, "-m", "xreview snapshot"],
            cwd=repo_path,
            env=env,
        ).stdout.strip()
        return base_sha, commit_sha, tree_sha
    finally:
        tmp_index.unlink(missing_ok=True)


def _checkout_tree_to_dir(
    repo_path: Path,
    objects_dir: Path,
    tree_or_commit: str,
    dest_dir: Path,
    exclude_unchanged_paths: set[str] | None = None,
) -> list[str]:
    """Checkout tree_or_commit into dest_dir, removing exclude_unchanged_paths in the Git index BEFORE checkout.

    Returns list of paths actually removed from the checkout index (zero host filesystem traversal!).
    """
    _gd, common_gd = _git_dirs(repo_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    co_index = objects_dir / f"co_index_{time.time_ns()}_{os.getpid()}"
    env = _obj_env(objects_dir, common_gd, co_index)
    removed_entries: list[str] = []
    try:
        _git(["read-tree", tree_or_commit], cwd=repo_path, env=env)
        if exclude_unchanged_paths:
            ls_out = _git(["ls-files", "-z"], cwd=repo_path, env=env, text=False).stdout
            indexed = [p.decode("utf-8", errors="surrogateescape") for p in ls_out.split(b"\x00") if p]
            to_drop: list[str] = []
            for ep in sorted(exclude_unchanged_paths):
                matched = [p for p in indexed if p == ep or p.startswith(ep + "/")]
                if matched:
                    to_drop.extend(matched)
                    removed_entries.append(ep)
            if to_drop:
                # Pass paths via stdin NUL-delimited to update-index --force-remove
                proc = subprocess.run(
                    ["git", *_SAFE_GIT_CONFIG_ARGS, "update-index", "--force-remove", "-z", "--stdin"],
                    cwd=repo_path,
                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", **env},
                    input=b"\x00".join(p.encode("utf-8", errors="surrogateescape") for p in to_drop) + b"\x00",
                    capture_output=True,
                )
                if proc.returncode != 0:
                    raise SnapshotError(f"git update-index --force-remove failed: {proc.stderr.decode('utf-8', 'replace')}")
        _git(["checkout-index", "-a", f"--prefix={dest_dir.resolve()}/"], cwd=repo_path, env=env)
        return removed_entries
    finally:
        co_index.unlink(missing_ok=True)


def _diff_commits_bytes(
    repo_path: Path,
    objects_dir: Path,
    rev_a: str,
    rev_b: str,
    prefix: str = "",
) -> bytes:
    _gd, common_gd = _git_dirs(repo_path)
    env = _obj_env(objects_dir, common_gd)
    args = ["diff", "--no-ext-diff", "--no-textconv"]
    if prefix:
        args.extend([f"--src-prefix=a/{prefix}/", f"--dst-prefix=b/{prefix}/"])
    args.extend([rev_a, rev_b])
    return _git(args, cwd=repo_path, env=env, text=False).stdout


def _diff_bare_git_dir_bytes(
    git_dir: Path,
    objects_dir: Path | None,
    rev_a: str,
    rev_b: str,
    prefix: str = "",
) -> bytes:
    """Compute a diff directly from a bare module git dir (for deleted submodules, astra-20)."""
    env = {"GIT_DIR": str(git_dir)}
    if objects_dir is not None and objects_dir.exists():
        env["GIT_OBJECT_DIRECTORY"] = str(objects_dir)
        env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(git_dir / "objects")
    args = ["diff", "--no-ext-diff", "--no-textconv"]
    if prefix:
        args.extend([f"--src-prefix=a/{prefix}/", f"--dst-prefix=b/{prefix}/"])
    args.extend([rev_a, rev_b])
    res = _git(args, cwd=git_dir, env=env, check=False, text=False)
    return res.stdout if res.returncode == 0 else b""


def _changed_files_between(
    repo_path: Path,
    objects_dir: Path,
    rev_a: str,
    rev_b: str,
) -> set[str]:
    _gd, common_gd = _git_dirs(repo_path)
    env = _obj_env(objects_dir, common_gd)
    out = _git(
        ["diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z", rev_a, rev_b],
        cwd=repo_path,
        env=env,
        text=False,
    ).stdout
    return {p.decode("utf-8", errors="surrogateescape") for p in out.split(b"\x00") if p}


def parse_diff_line_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff hunks into {file_path: [(start_line, end_line), ...]} in the new file."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("+++ /dev/null"):
            current_file = None
        elif current_file and line.startswith("@@ "):
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                end = start + max(count - 1, 0)
                ranges.setdefault(current_file, []).append((start, end))
    return ranges


def find_overlapping_fix_lines(
    previous_rounds_ranges: list[dict[str, list[tuple[int, int]]]],
    current_ranges: dict[str, list[tuple[int, int]]],
) -> list[str]:
    """Return human-readable hints for files/lines modified by an earlier round and modified again."""
    hints: list[str] = []
    for file_path, cur_spans in sorted(current_ranges.items()):
        for r_idx, prev_map in enumerate(previous_rounds_ranges, start=1):
            prev_spans = prev_map.get(file_path, [])
            for cs, ce in cur_spans:
                for ps, pe in prev_spans:
                    if cs <= pe and ps <= ce:
                        hints.append(
                            f"{file_path}:{cs}-{ce} overlaps lines {ps}-{pe} changed in round {r_idx}"
                        )
                        break
    return hints


def _discover_working_tree_gitlinks(repo_path: Path) -> dict[str, str]:
    """Discover gitlinks (160000) in the working repo's index using NUL-delimited ls-files."""
    out = _git(["ls-files", "--stage", "-z"], cwd=repo_path, text=False).stdout
    subs: dict[str, str] = {}
    for entry in out.split(b"\x00"):
        if not entry:
            continue
        meta_b, _, path_b = entry.partition(b"\t")
        parts = meta_b.decode("ascii", errors="replace").split()
        if len(parts) >= 2 and parts[0] == "160000":
            subs[path_b.decode("utf-8", errors="surrogateescape")] = parts[1]
    return subs


def check_unclassified_files(
    repos: list[RepoSpec],
    included_files: set[str],
    excluded_files: set[str],
    profile_excluded_submodules: dict[str, set[str]],
) -> list[str]:
    """Return list of unclassified untracked files across repos and active initialized submodules."""
    multi = len(repos) > 1
    unclassified: list[str] = []
    for repo in repos:
        if repo.target_rev is not None:
            continue
        for rel in list_untracked_files(repo.path):
            key = f"{repo.name}/{rel}" if multi else rel
            if key not in included_files and key not in excluded_files:
                unclassified.append(key)

        _gd, common_gd = _git_dirs(repo.path)
        base_subs = _discover_submodules(repo.path, repo.base_rev, _obj_env(common_gd / "objects", common_gd))
        wt_subs = _discover_working_tree_gitlinks(repo.path)
        excl_subs = profile_excluded_submodules.get(repo.name, set())

        def _check_sub_tree(parent_dir: Path, parent_common: Path, prefix: str, b_subs: dict[str, str], t_subs: dict[str, str]) -> None:
            for sp, t_sha in sorted(t_subs.items()):
                full_sp = f"{prefix}{sp}"
                if not _is_submodule_initialized_safely(parent_dir, parent_common, sp):
                    continue
                b_sha = b_subs.get(sp)
                changed = _submodule_has_changes(parent_dir, parent_common, sp, b_sha, None, t_sha)
                if full_sp in excl_subs and not changed:
                    continue
                sub_dir = parent_dir / sp
                for rel in list_untracked_files(sub_dir):
                    full_rel = f"{full_sp}/{rel}"
                    key = f"{repo.name}/{full_rel}" if multi else full_rel
                    if key not in included_files and key not in excluded_files:
                        unclassified.append(key)
                # Recurse nested submodules
                _sgd, sub_common = _git_dirs(sub_dir)
                nested_t = _discover_working_tree_gitlinks(sub_dir)
                nested_b = (
                    _discover_submodules(sub_dir, b_sha, _obj_env(sub_common / "objects", sub_common))
                    if b_sha
                    else {}
                )
                if nested_t:
                    _check_sub_tree(sub_dir, sub_common, f"{full_sp}/", nested_b, nested_t)

        _check_sub_tree(repo.path, common_gd, "", base_subs, wt_subs)
    return unclassified


def _write_file_exclusive_nofollow(dest: Path, data: bytes) -> None:
    """Create a regular file exclusively without following symlinks."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(dest), flags, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _diff_docs_dirs(prev_docs: Path | None, cur_docs: Path | None) -> bytes:
    """Compute byte-preserving unified diffs across standalone doc snapshots via git diff --no-index (astra-56)."""
    if prev_docs is None and cur_docs is None:
        return b""
    prev_files = {p.name: p for p in prev_docs.iterdir() if p.is_file() and not p.is_symlink()} if (prev_docs and prev_docs.is_dir()) else {}
    cur_files = {p.name: p for p in cur_docs.iterdir() if p.is_file() and not p.is_symlink()} if (cur_docs and cur_docs.is_dir()) else {}
    parts: list[bytes] = []
    null_dev = "/dev/null"
    for name in sorted(set(prev_files.keys()) | set(cur_files.keys())):
        p_a = prev_files.get(name)
        p_b = cur_files.get(name)
        if p_a and p_b and p_a.read_bytes() == p_b.read_bytes():
            continue
        arg_a = str(p_a.resolve()) if p_a else null_dev
        arg_b = str(p_b.resolve()) if p_b else null_dev
        res = subprocess.run(
            ["git", *_SAFE_GIT_CONFIG_ARGS, "diff", "--no-index", "--no-ext-diff", "--no-textconv", "--binary", "--", arg_a, arg_b],
            capture_output=True,
            check=False,
        )
        if res.stdout:
            # git diff --no-index /abs/a /abs/b emits a/abs/a and b/abs/b; replace /abs/... with /docs/<name>
            # so headers become a/docs/<name> and b/docs/<name> for parse_diff_line_ranges (F-012)
            out = res.stdout
            name_b = os.fsencode(name)
            if p_a:
                out = out.replace(os.fsencode(arg_a), b"/docs/" + name_b)
            if p_b:
                out = out.replace(os.fsencode(arg_b), b"/docs/" + name_b)
            parts.append(out)
    return b"\n".join(parts)


def create_round_snapshot(
    run_dir: Path,
    round_num: int,
    repos: list[RepoSpec],
    included_files: set[str],
    excluded_files: set[str],
    profile_excluded_submodules: dict[str, set[str]],
    secret_patterns: tuple[str, ...] = DEFAULT_SECRET_PATTERNS,
    prev_snapshots: dict[str, RepoRoundSnapshot] | None = None,
    docs: list[Path] | None = None,
    before_publish: Callable[[list[str]], None] | None = None,
) -> RoundSnapshotResult:
    """Create a brand-new read-only snapshot tree and diffs for round_num atomically via staging."""
    # Re-verify every repository is still a Git worktree root before snapshotting (astra-53)
    for repo in repos:
        verify_repo_is_worktree_root(repo.path)

    unclassified = check_unclassified_files(
        repos, included_files, excluded_files, profile_excluded_submodules
    )
    if unclassified:
        raise UnclassifiedFilesError(unclassified)

    round_dir = run_dir / f"r{round_num}"
    final_tree_dir = round_dir / "tree"
    final_docs_dir = round_dir / "docs"
    final_diffs_dir = round_dir / "diffs"
    if final_tree_dir.exists():
        raise SnapshotError(f"Round tree already exists (never reuse a round tree): {final_tree_dir}")

    # Clean up partial publish artifacts or leftover staging dirs from an earlier crash before tree publish (F-011)
    if round_dir.is_dir():
        for orphan in (final_docs_dir, final_diffs_dir):
            if orphan.exists() or orphan.is_symlink():
                shutil.rmtree(orphan, ignore_errors=True)
        for child in round_dir.iterdir():
            if child.name.startswith(".staging_"):
                shutil.rmtree(child, ignore_errors=True)

    round_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = round_dir / f".staging_{time.time_ns()}_{os.getpid()}"
    staging_tree = staging_dir / "tree"
    staging_docs = staging_dir / "docs"
    staging_diffs = staging_dir / "diffs"
    staging_tree.mkdir(parents=True)
    staging_diffs.mkdir(parents=True)

    multi = len(repos) > 1
    repo_snapshots: dict[str, RepoRoundSnapshot] = {}
    base_diff_parts: list[bytes] = []
    round_diff_parts: list[bytes] = []
    all_project_exclusions: list[str] = []
    covered_host_paths: list[str] = [str(r.path.resolve()) for r in repos]

    try:
        for repo in repos:
            objs_dir = run_dir / "objects" / repo.name
            objs_dir.mkdir(parents=True, exist_ok=True)
            _gd, common_gd = _git_dirs(repo.path)
            repo_tree_dest = (staging_tree / repo.name) if multi else staging_tree

            def _strip_prefix(p: str) -> str | None:
                if not multi:
                    return p
                pref = repo.name + "/"
                return p[len(pref):] if p.startswith(pref) else None

            base_sha = resolve_commit_sha(repo.path, repo.base_rev, allow_empty_tree=True)
            base_subs = _discover_submodules(repo.path, base_sha, _obj_env(objs_dir, common_gd))
            if repo.target_rev is not None:
                target_subs = _discover_submodules(repo.path, repo.target_rev, _obj_env(objs_dir, common_gd))
            else:
                target_subs = _discover_working_tree_gitlinks(repo.path)

            # Separate parent includes from submodule includes
            repo_includes: list[str] = []
            sub_includes: dict[str, list[str]] = {}
            all_sub_keys = sorted(set(base_subs.keys()) | set(target_subs.keys()), key=len, reverse=True)
            for inc in sorted(included_files):
                rel = _strip_prefix(inc)
                if rel is None:
                    continue
                matched_sub = None
                for sp in all_sub_keys:
                    if rel.startswith(sp + "/"):
                        matched_sub = sp
                        sub_includes.setdefault(sp, []).append(rel[len(sp) + 1 :])
                        break
                if matched_sub is None:
                    repo_includes.append(rel)

            _, prelim_commit, _ = _snapshot_single_git_dir(
                repo.path, objs_dir, repo.base_rev, repo.target_rev, repo_includes, secret_patterns
            )
            snap_env = _obj_env(objs_dir, common_gd)

            # Read untrusted .xreview.toml strictly from git objects
            proj_cfg = ProjectConfig()
            raw_proj = read_blob_from_git_object(repo.path, prelim_commit, ".xreview.toml", snap_env)
            if raw_proj is not None:
                proj_cfg = parse_untrusted_project_toml(raw_proj)

            prof_excl = profile_excluded_submodules.get(repo.name, set())
            proj_excl_subs = set(proj_cfg.exclude_submodules)
            prev_repo_snap = prev_snapshots.get(repo.name) if prev_snapshots else None

            included_sub_commits: dict[str, str] = {}
            sub_records: dict[str, tuple[str, str, str]] = {}
            sub_git_dirs: dict[str, str] = {}
            sub_obj_dirs: dict[str, Path] = {}
            sub_host_dirs: dict[str, Path] = {}
            present_excluded_subs: dict[str, tuple[Path, Path, str | None]] = {}
            excluded_subs: list[str] = []
            repo_proj_exclusions: list[str] = []
            included_sub_host_paths: list[str] = []
            deleted_sub_diffs: list[bytes] = []

            base_gitmodules_bytes = read_blob_from_git_object(repo.path, base_sha, ".gitmodules", snap_env)

            def _resolve_deleted_sub_git_dir(
                parent_path: Path, parent_common: Path, sp: str, full_sp: str, b_gm: bytes | None
            ) -> Path | None:
                if prev_repo_snap and full_sp in prev_repo_snap.submodule_git_dirs:
                    p = Path(prev_repo_snap.submodule_git_dirs[full_sp])
                    if p.is_dir():
                        return p
                exp_name, fallback = _expected_submodule_name(parent_path, sp, gitmodules_bytes=b_gm)
                for cand_name in ([exp_name] if exp_name else [fallback]):
                    c = _confined_module_dir(parent_common, cand_name)
                    if c and c.is_dir():
                        return c
                return None

            def _process_submodules(
                parent_path: Path,
                parent_common: Path,
                parent_objs: Path,
                b_subs: dict[str, str],
                t_subs: dict[str, str],
                prefix: str,
                target_is_commit: bool,
                inc_map: dict[str, list[str]],
                b_gm: bytes | None,
            ) -> dict[str, str]:
                direct_updates: dict[str, str] = {}
                # Handle submodules deleted in target (resolve logical name from base .gitmodules, astra-20)
                for sp, b_sha in sorted(b_subs.items()):
                    if sp not in t_subs:
                        full_sp = f"{prefix}{sp}"
                        sp_pref = f"{repo.name}/{full_sp}" if multi else full_sp
                        sub_objs = parent_objs / "submodules" / sp
                        d_bytes = b""
                        if _is_submodule_initialized_safely(parent_path, parent_common, sp):
                            d_bytes = _diff_commits_bytes(parent_path / sp, sub_objs, b_sha, EMPTY_TREE_SHA, prefix=sp_pref)
                        else:
                            cand_gd = _resolve_deleted_sub_git_dir(parent_path, parent_common, sp, full_sp, b_gm)
                            if cand_gd is not None:
                                d_bytes = _diff_bare_git_dir_bytes(cand_gd, sub_objs, b_sha, EMPTY_TREE_SHA, prefix=sp_pref)
                        if d_bytes:
                            deleted_sub_diffs.append(
                                b"\n# Submodule " + os.fsencode(sp_pref) + f" removed ({b_sha[:8]})\n".encode("ascii") + d_bytes
                            )
                        else:
                            deleted_sub_diffs.append(
                                b"\n# Submodule " + os.fsencode(sp_pref) + f" removed ({b_sha[:8]}; objects unavailable)\n".encode("ascii")
                            )

                for sp, t_sha in sorted(t_subs.items()):
                    full_sp = f"{prefix}{sp}"
                    b_sha = b_subs.get(sp)
                    if not _is_submodule_initialized_safely(parent_path, parent_common, sp):
                        if b_sha != t_sha:
                            raise SnapshotError(
                                f"Changed submodule {full_sp!r} in {repo.name} is not initialized or has a symlinked path."
                            )
                        excluded_subs.append(f"{full_sp} (uninitialized)")
                        continue

                    sub_has_inc = bool(inc_map.get(sp))
                    changed = _submodule_has_changes(
                        parent_path,
                        parent_common,
                        sp,
                        b_sha,
                        "commit" if target_is_commit else None,
                        t_sha,
                        has_included_files=sub_has_inc,
                    )
                    if not changed and (full_sp in prof_excl or full_sp in proj_excl_subs):
                        present_excluded_subs[full_sp] = (
                            parent_path / sp,
                            parent_objs / "submodules" / sp,
                            b_sha,
                        )
                        reason = "profile" if full_sp in prof_excl else ".xreview.toml"
                        excluded_subs.append(f"{full_sp} ({reason})")
                        if full_sp in proj_excl_subs and full_sp not in prof_excl:
                            repo_proj_exclusions.append(f"{repo.name}:{full_sp} (submodule)")
                        continue

                    sub_repo = parent_path / sp
                    sub_objs = parent_objs / "submodules" / sp
                    sub_objs.mkdir(parents=True, exist_ok=True)
                    sub_gd, sub_common = _git_dirs(sub_repo)
                    sub_git_dirs[full_sp] = str(sub_gd.resolve())
                    included_sub_host_paths.append(str(sub_repo.resolve()))
                    covered_host_paths.append(str(sub_repo.resolve()))

                    nested_b = (
                        _discover_submodules(sub_repo, b_sha, _obj_env(sub_objs, sub_common))
                        if b_sha
                        else {}
                    )
                    nested_b_gm = (
                        read_blob_from_git_object(sub_repo, b_sha, ".gitmodules", _obj_env(sub_objs, sub_common))
                        if b_sha
                        else None
                    )
                    nested_t = (
                        _discover_submodules(sub_repo, t_sha, _obj_env(sub_objs, sub_common))
                        if target_is_commit
                        else _discover_working_tree_gitlinks(sub_repo)
                    )
                    nested_inc: dict[str, list[str]] = {}
                    own_inc: list[str] = []
                    for rel in inc_map.get(sp, ()):
                        matched_n = None
                        for nsp in nested_t:
                            if rel.startswith(nsp + "/"):
                                matched_n = nsp
                                nested_inc.setdefault(nsp, []).append(rel[len(nsp) + 1 :])
                                break
                        if matched_n is None:
                            own_inc.append(rel)

                    nested_updates = _process_submodules(
                        sub_repo, sub_common, sub_objs, nested_b, nested_t, f"{full_sp}/", target_is_commit, nested_inc, nested_b_gm
                    )

                    # Prefer checked-out HEAD / target gitlink BEFORE base commit so intentional
                    # commit-pointer changes between same-tree commits are preserved (astra-58)!
                    reuse_candidates: list[tuple[str, str]] = []
                    head_sha = _git(["rev-parse", "HEAD"], cwd=sub_repo, check=False).stdout.strip()
                    if head_sha:
                        head_tree = _git(["rev-parse", f"{head_sha}^{{tree}}"], cwd=sub_repo, check=False).stdout.strip()
                        if head_tree:
                            reuse_candidates.append((head_tree, head_sha))
                    if t_sha and t_sha != head_sha:
                        t_tree = _git(["rev-parse", f"{t_sha}^{{tree}}"], cwd=sub_repo, check=False).stdout.strip()
                        if t_tree:
                            reuse_candidates.append((t_tree, t_sha))
                    if prev_repo_snap and full_sp in prev_repo_snap.submodules:
                        _pb, p_snap, p_tree = prev_repo_snap.submodules[full_sp]
                        if p_tree and p_snap:
                            reuse_candidates.append((p_tree, p_snap))
                    if b_sha and b_sha not in (head_sha, t_sha):
                        b_tree = _git(["rev-parse", f"{b_sha}^{{tree}}"], cwd=sub_repo, check=False).stdout.strip()
                        if b_tree:
                            reuse_candidates.append((b_tree, b_sha))

                    sb, sc, st = _snapshot_single_git_dir(
                        sub_repo,
                        sub_objs,
                        b_sha,
                        t_sha if target_is_commit else None,
                        own_inc,
                        secret_patterns,
                        sub_commits=nested_updates if not target_is_commit else None,
                        reuse_if_tree_matches=reuse_candidates,
                    )
                    direct_updates[sp] = sc
                    sub_records[full_sp] = (sb, sc, st)
                    sub_obj_dirs[full_sp] = sub_objs
                    sub_host_dirs[full_sp] = sub_repo
                return direct_updates

            included_sub_commits = _process_submodules(
                repo.path,
                common_gd,
                objs_dir,
                base_subs,
                target_subs,
                "",
                repo.target_rev is not None,
                sub_includes,
                base_gitmodules_bytes,
            )

            # Reuse parent base/prev commit if unchanged? Keep parent commit fresh if changed, reuse if identical to base.
            base_tree = _git(["rev-parse", f"{base_sha}^{{tree}}"], cwd=repo.path).stdout.strip()
            reuse_parent = [(base_tree, base_sha)]
            if prev_repo_snap:
                reuse_parent.append((prev_repo_snap.tree_sha, prev_repo_snap.commit_sha))

            base_sha, commit_sha, tree_sha = _snapshot_single_git_dir(
                repo.path,
                objs_dir,
                repo.base_rev,
                repo.target_rev,
                repo_includes,
                secret_patterns,
                sub_commits=included_sub_commits if repo.target_rev is None else None,
                reuse_if_tree_matches=reuse_parent,
            )

            # Compute unchanged .xreview.toml exclude_paths BEFORE checkout (astra-1 & astra-22)
            changed_files = _changed_files_between(repo.path, objs_dir, base_sha, commit_sha)
            safe_excl_paths: set[str] = set()
            for ep in proj_cfg.exclude_paths:
                if any(cf == ep or cf.startswith(ep + "/") or ep.startswith(cf + "/") for cf in changed_files):
                    continue
                safe_excl_paths.add(ep)

            removed = _checkout_tree_to_dir(
                repo.path, objs_dir, tree_sha, repo_tree_dest, exclude_unchanged_paths=safe_excl_paths
            )
            for ep in removed:
                repo_proj_exclusions.append(f"{repo.name}:{ep}")

            # Checkout included submodules in depth order (parent submodules before nested)
            for full_sp in sorted(sub_records.keys(), key=lambda s: s.count("/")):
                _sb, sc, _st = sub_records[full_sp]
                _checkout_tree_to_dir(
                    sub_host_dirs[full_sp],
                    sub_obj_dirs[full_sp],
                    sc,
                    repo_tree_dest / full_sp,
                )

            # Read untrusted instruction files named in .xreview.toml strictly via git objects
            instr_contents: dict[str, str] = {}
            for ifile in proj_cfg.instruction_files:
                blob = read_blob_from_git_object(repo.path, commit_sha, ifile, snap_env)
                if blob is not None:
                    instr_contents[ifile] = blob.decode("utf-8", errors="replace")

            # Locate AGENTS.md paths via NUL-delimited ls-tree
            agents_out = _git(["ls-tree", "-r", "--name-only", "-z", commit_sha], cwd=repo.path, env=snap_env, text=False).stdout
            agents_paths = [
                f"{repo.name}/{p}" if multi else p
                for p in (x.decode("utf-8", errors="surrogateescape") for x in agents_out.split(b"\x00") if x)
                if p == "AGENTS.md" or p.endswith("/AGENTS.md")
            ]

            # Compute BASE.diff as raw bytes using os.fsencode for non-UTF-8 paths (astra-73)
            diff_prefix = repo.name if multi else ""
            b_diff = _diff_commits_bytes(repo.path, objs_dir, base_sha, commit_sha, prefix=diff_prefix)
            for full_sp in sorted(sub_records.keys()):
                sb, sc, _st = sub_records[full_sp]
                if sb != sc:
                    sp_pref = f"{repo.name}/{full_sp}" if multi else full_sp
                    sub_d = _diff_commits_bytes(sub_host_dirs[full_sp], sub_obj_dirs[full_sp], sb, sc, prefix=sp_pref)
                    if sub_d:
                        hdr = b"\n# Submodule " + os.fsencode(sp_pref) + f" content diff ({sb[:8]}..{sc[:8]})\n".encode("ascii")
                        b_diff += hdr + sub_d
            for del_d in deleted_sub_diffs:
                b_diff += del_d
            if b_diff.strip():
                base_diff_parts.append(b_diff)

            # Compute ROUND.diff as raw bytes
            if prev_repo_snap:
                r_diff = _diff_commits_bytes(repo.path, objs_dir, prev_repo_snap.commit_sha, commit_sha, prefix=diff_prefix)
                for full_sp in sorted(sub_records.keys()):
                    sb, sc, _st = sub_records[full_sp]
                    prev_sub = prev_repo_snap.submodules.get(full_sp)
                    prev_sc = prev_sub[1] if prev_sub else sb
                    if prev_sc != sc:
                        sp_pref = f"{repo.name}/{full_sp}" if multi else full_sp
                        sub_rd = _diff_commits_bytes(
                            sub_host_dirs[full_sp], sub_obj_dirs[full_sp], prev_sc, sc, prefix=sp_pref
                        )
                        if sub_rd:
                            hdr = b"\n# Submodule " + os.fsencode(sp_pref) + f" round diff ({prev_sc[:8]}..{sc[:8]})\n".encode("ascii")
                            r_diff += hdr + sub_rd
                # Emit content diff for submodules deleted OR reverted-to-excluded since previous round (astra-20, F-007)
                for prev_sp, (_pb, prev_sc, _pt) in sorted(prev_repo_snap.submodules.items()):
                    if prev_sp not in sub_records:
                        sp_pref = f"{repo.name}/{prev_sp}" if multi else prev_sp
                        sub_objs = (
                            Path(prev_repo_snap.submodule_obj_dirs[prev_sp])
                            if prev_sp in prev_repo_snap.submodule_obj_dirs
                            else (objs_dir / "submodules" / prev_sp)
                        )
                        if prev_sp in present_excluded_subs or any(
                            prev_sp.startswith(ex + "/") for ex in present_excluded_subs
                        ):
                            if prev_sp in present_excluded_subs:
                                ex_host, ex_objs, ex_base = present_excluded_subs[prev_sp]
                                target_revert = ex_base or _pb
                                if target_revert and prev_sc != target_revert:
                                    rev_rd = _diff_commits_bytes(ex_host, ex_objs, prev_sc, target_revert, prefix=sp_pref)
                                    if rev_rd:
                                        r_diff += (
                                            b"\n# Submodule "
                                            + os.fsencode(sp_pref)
                                            + f" reverted to base ({prev_sc[:8]}..{target_revert[:8]})\n".encode("ascii")
                                            + rev_rd
                                        )
                            else:
                                # Nested submodule beneath an excluded parent submodule that reverted to base (F-020)
                                cand_gd = _resolve_deleted_sub_git_dir(repo.path, common_gd, prev_sp, prev_sp, base_gitmodules_bytes)
                                if cand_gd is not None and _pb and prev_sc != _pb:
                                    rev_rd = _diff_bare_git_dir_bytes(cand_gd, sub_objs, prev_sc, _pb, prefix=sp_pref)
                                    if rev_rd:
                                        r_diff += (
                                            b"\n# Submodule "
                                            + os.fsencode(sp_pref)
                                            + f" reverted to base ({prev_sc[:8]}..{_pb[:8]})\n".encode("ascii")
                                            + rev_rd
                                        )
                            continue
                        cand_gd = _resolve_deleted_sub_git_dir(repo.path, common_gd, prev_sp, prev_sp, base_gitmodules_bytes)
                        del_rd = (
                            _diff_bare_git_dir_bytes(cand_gd, sub_objs, prev_sc, EMPTY_TREE_SHA, prefix=sp_pref)
                            if cand_gd is not None
                            else b""
                        )
                        if del_rd:
                            r_diff += b"\n# Submodule " + os.fsencode(sp_pref) + f" removed since last round ({prev_sc[:8]})\n".encode("ascii") + del_rd
                        else:
                            r_diff += b"\n# Submodule " + os.fsencode(sp_pref) + f" removed since last round ({prev_sc[:8]}; objects unavailable)\n".encode("ascii")
                if r_diff.strip():
                    round_diff_parts.append(r_diff)

            all_project_exclusions.extend(repo_proj_exclusions)
            repo_snapshots[repo.name] = RepoRoundSnapshot(
                repo_name=repo.name,
                repo_path=str(repo.path),
                base_sha=base_sha,
                commit_sha=commit_sha,
                tree_sha=tree_sha,
                submodules=sub_records,
                submodule_git_dirs=sub_git_dirs,
                submodule_obj_dirs={k: str(v) for k, v in sub_obj_dirs.items()},
                included_submodule_host_paths=included_sub_host_paths,
                excluded_submodules=excluded_subs,
                project_exclusions=repo_proj_exclusions,
                project_notes=proj_cfg.notes,
                project_instruction_contents=instr_contents,
                agents_md_paths=agents_paths,
            )

        # Copy standalone design documents into separate orchestrator-owned docs/ dir (astra-5 & astra-6)
        has_docs = False
        if docs:
            staging_docs.mkdir(parents=True, exist_ok=True)
            seen_names: set[str] = set()
            for doc in docs:
                if is_secret_path(doc.name, secret_patterns):
                    raise SnapshotError(f"Refusing secret-like --doc file: {doc.name}")
                if doc.name in seen_names:
                    raise SnapshotError(f"Duplicate --doc filename: {doc.name!r}")
                seen_names.add(doc.name)
                try:
                    data = safe_open_user_regular_file(doc, max_bytes=MAX_UNTRACKED_FILE_BYTES)
                except ConfigError as exc:
                    raise SnapshotError(str(exc)) from exc
                _write_file_exclusive_nofollow(staging_docs / doc.name, data)
                covered_host_paths.append(os.path.abspath(os.path.expanduser(str(doc))))
                has_docs = True

        # Include standalone document diffs in ROUND.diff across rounds (astra-56)
        if round_num > 1 and (has_docs or (run_dir / f"r{round_num - 1}" / "docs").is_dir()):
            prev_docs_dir = run_dir / f"r{round_num - 1}" / "docs"
            doc_rd = _diff_docs_dirs(prev_docs_dir if prev_docs_dir.is_dir() else None, staging_docs if has_docs else None)
            if doc_rd:
                round_diff_parts.append(doc_rd)

        base_diff_bytes = b"\n".join(base_diff_parts)
        round_diff_bytes = b"\n".join(round_diff_parts)

        if round_num == 1 and not base_diff_bytes.strip() and not has_docs:
            raise SnapshotError("Base and target do not differ; nothing to review.")

        # Run caller pre-publish checks (e.g., tag enforcement) BEFORE publishing tree (astra-24)
        if before_publish is not None:
            before_publish(covered_host_paths)

        # Write canonical diffs into staging_diffs so recovery never relies on reviewer inputs (astra-25)
        (staging_diffs / "BASE.diff").write_bytes(base_diff_bytes)
        (staging_diffs / "ROUND.diff").write_bytes(round_diff_bytes)

        # Publish staging directories atomically (docs + diffs first, tree last as commit point!)
        published_docs: Path | None = None
        if has_docs:
            os.replace(staging_docs, final_docs_dir)
            published_docs = final_docs_dir
        os.replace(staging_diffs, final_diffs_dir)
        os.replace(staging_tree, final_tree_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(final_docs_dir, ignore_errors=True)
        shutil.rmtree(final_diffs_dir, ignore_errors=True)
        shutil.rmtree(final_tree_dir, ignore_errors=True)
        raise

    diff_for_lines = (round_diff_bytes if round_num > 1 else base_diff_bytes).decode("utf-8", errors="replace")
    changed_lines = parse_diff_line_ranges(diff_for_lines)

    return RoundSnapshotResult(
        round_num=round_num,
        tree_dir=final_tree_dir,
        docs_dir=published_docs,
        repo_snapshots=repo_snapshots,
        base_diff_bytes=base_diff_bytes,
        round_diff_bytes=round_diff_bytes,
        changed_lines_in_round=changed_lines,
        project_exclusions=all_project_exclusions,
        covered_host_paths=covered_host_paths,
    )
