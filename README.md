# xreview (`cross-review-loop`)

A cross-vendor, multi-model code and design review loop built on subscription CLIs (Codex CLI, Antigravity `agy`, and Claude Code) using Python 3.11+ standard library only and an allowlist `bubblewrap` sandbox.

## How it works

1. **Snapshot without touching your repo:** `xreview` creates a fresh read-only tree per round using temporary git indices and a run-local object store (`GIT_OBJECT_DIRECTORY` + `GIT_ALTERNATE_OBJECT_DIRECTORIES`), including initialized submodules (with full submodule content diffs).
2. **Allowlist sandbox + vendor-only proxy:** Each reviewer runs in its own `bwrap` mount, PID, and network namespace. Only read-only system dirs, the round's snapshot tree, that reviewer's inputs, and its isolated state folder are mounted. Outbound traffic flows over a per-reviewer Unix socket to a host-side HTTPS `CONNECT` proxy restricted strictly to that reviewer's data vendor hosts.
3. **Blind Round 1, warm Round 2+:** Round 1 runs reviewers independently so nobody anchors on another model. The implementor verifies each finding against the code and records dispositions (`fixed`, `rejected`, `deferred`, `question`, `conflicts_with`, `lower_severity`, `merge`). Subsequent rounds resume each reviewer warmly with `ROUND.diff` until consensus, with bounded disputes escalating to a human.
4. **Deterministic ledger:** `ledger.json` (written atomically after every step) and `LEDGER.md` track every finding, gate lifted, project exclusion, escalation, and per-reviewer summary stats.

## Requirements

- Python 3.11+ (no third-party Python dependencies)
- `bubblewrap` (`bwrap`) on Linux (optional `socat`; falls back to a built-in Python bridge if `socat` is absent)
- One or more supported subscription CLIs (`codex`, `agy`, `claude`)

## Quick start

```bash
pip install -e . --no-build-isolation
xreview doctor
xreview quota --refresh
```

`xreview quota` inspects cached plan utilization (from Codex session rollouts and Claude `rate_limit_event` telemetry stored in `<state_dir>/quota_cache.json`). Passing `--refresh` (or letting `xreview init` auto-refresh when Claude's cache is missing or older than `quota_cache_max_age_seconds`, default 30 min; opt out with `--no-refresh-quota`) runs a tiny sandboxed `haiku` probe to fetch live Claude `five_hour`/`seven_day` utilization before choosing reviewers. During `xreview init`, plans at or above `fallback_to_extras_at` (default `0.75`) automatically back out of heavyweight core models and promote their cheaper extra models, and plans that are 100% exhausted are removed (down to a hard floor of `2` distinct model families including at least one non-implementor family), printing a notice and recording the adjustment in `LEDGER.md`.

Copy `examples/config.toml` to `~/.config/xreview/config.toml` and customize your reviewers and optional environment profile (see `examples/profile.toml`).

### Authentication for isolated sandboxes

- **Claude Code:** run `claude setup-token` and store the token in `~/.config/xreview/claude-oauth-token` (`chmod 600`).
- **Codex CLI:** sign in with `codex login`; `xreview` copies `~/.codex/auth.json` into each Codex reviewer's isolated `CODEX_HOME`.
- **Antigravity (`agy`):** create a file-based OAuth token (used automatically when no D-Bus session bus is present inside the sandbox):
  ```bash
  HOME=~/.config/xreview/agy-auth DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent XDG_RUNTIME_DIR=/nonexistent agy acp --login
  ```
  Then point `auth.agy_token_file` in `config.toml` to `~/.config/xreview/agy-auth/.gemini/antigravity-cli/antigravity-oauth-token`.

### Typical loop

```bash
# 1. Start a run
xreview init --repo . --base HEAD --artifact "Fix token refresh race" --implementor-family claude

# 2. Classify any untracked files (if prompted)
xreview classify <run-id> --include src/new_file.py --exclude notes.tmp

# 3. Run Round 1 (blind)
xreview round <run-id>

# 4. Inspect findings and record triage decisions
xreview status <run-id>
xreview triage <run-id> F-001 fixed --note "Added mutex around refresh"
xreview triage <run-id> F-002 rejected --reason "Bounded by caller" --cite "src/auth.py:42"

# 5. Run Round 2 (warm resume) until converged (exit code 0)
xreview round <run-id>

# 6. View final ledger and clean up reviewer state when done
xreview ledger <run-id>
xreview clean <run-id>
```

### Exit codes for `xreview round`

| Code | Meaning |
| --- | --- |
| `0` | Converged (no open findings above `nit`, no new findings above `nit`) |
| `10` | Needs implementor triage before next round |
| `20` | Paused for human decision (soft round ceiling, core quota exhausted, two consecutive incomplete rounds, or quorum lost) |
| `2` | Unclassified untracked files need `xreview classify` |
| `1` | Error or protocol guard refusal |

## Implementor skill and permission gates

`skills/xreview-implementor/SKILL.md` provides instructions for driving `xreview` from Claude Code, Antigravity, or Codex.

**Only a person may lift a gate:** configure your implementor harness so any `xreview` command containing `--allow` or `--reviewers` prompts the user for explicit approval (for example, `ask` rules in Claude Code's `~/.claude/settings.json` or Antigravity's permission settings).

## Running tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

All unit tests run offline with fake harness adapters and fake clocks (no model calls, no sleeps).
