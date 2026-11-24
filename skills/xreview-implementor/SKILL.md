---
name: xreview-implementor
description: Drive a cross-vendor multi-model code or design review loop using xreview, triaging every finding against the code until convergence or escalation.
---

# xreview Implementor Skill

Use `xreview` to run independent reviewers from multiple model families over uncommitted changes, commits, or design documents. `xreview` handles snapshots, sandboxing, resume, validation, and the ledger; **you (the implementor) make every technical judgment call.**

## Non-negotiable gate rule (only a person lifts a gate)

- **STOP and ask the user** before running any `xreview` command with `--allow` or with `--reviewers` on a gated run (`--tag security`, `--tag private`, or any tag configured with `vendors = "ask"`).
- Never pass `--no-sandbox` without explicit user approval.
- Never pass `--resume-pause` without presenting the pause reason and getting the user's explicit approval to continue.
- Never commit or push reviewed changes without explicit user approval.

### Recommended harness permission rules

Configure your implementor harness so lifting a gate always prompts the user:

- **Claude Code (`~/.claude/settings.json`):** run `xreview setup-gates` (or `xreview setup-gates --check` to verify), which installs:
  ```json
  {
    "permissions": {
      "ask": [
        "Bash(xreview *--allow*)",
        "Bash(xreview *--reviewers*)",
        "Bash(xreview *--no-sandbox*)",
        "Bash(xreview *--resume-pause*)",
        "Bash(xreview *substitute*)",
        "Bash(python3 -m xreview *--allow*)",
        "Bash(python3 -m xreview *--reviewers*)",
        "Bash(python3 -m xreview *--no-sandbox*)",
        "Bash(python3 -m xreview *--resume-pause*)",
        "Bash(python3 -m xreview *substitute*)"
      ]
    }
  }
  ```
- **Antigravity (`agy` settings):** add `ask` rules for `run_command` matching `xreview *--allow*`, `xreview *--reviewers*`, `xreview *--no-sandbox*`, `xreview *--resume-pause*`, and `xreview *substitute*`.

## Workflow

1. **Initialize the run:**
   ```bash
   xreview init --repo <path> --base HEAD \
     --artifact "<short title>" \
     --purpose "<why this change exists>" \
     --issue "<issue ref>" \
     --in-scope "<scope>" \
     --out-of-scope "<non-goals>" \
     --implementor-family <your-family> \
     --json
   ```
   - Add `--tag security` if the work touches a security vulnerability.
   - Add `--repo <path>` once per repository when changes span multiple repos.
   - Attach minimal, pseudonymized evidence files (if needed) with `xreview evidence <run-id> --file <path> --source "<cmd/query>" --tag <tag>`.

2. **Run Round 1 (blind):**
   ```bash
   xreview round <run-id>
   ```
   - If exit code is `2`, classify every listed untracked file with `xreview classify <run-id> --include <path> --exclude <path>` and re-run `xreview round <run-id>`.
   - Run `xreview round` in the background if your harness supports it; a round typically takes 5–20 minutes.

3. **Triage every open finding (exit code `10`):**
   - Run `xreview status <run-id> --json` and inspect `ledger.json`.
   - **Treat every finding as a claim, not a fact.** Read the cited file and lines before acting.
   - Merge duplicates first so each finder keeps its vote:
     `xreview merge <run-id> F-002 --into F-001 --reason "Same null check"`
   - Record a disposition for every open non-nit finding:
     - **Fix it** in the working tree, run targeted tests, then:
       `xreview triage <run-id> F-001 fixed --note "<what changed>"`
     - **Reject it** with concrete evidence:
       `xreview triage <run-id> F-002 rejected --reason "<why>" --cite "<file:line or scope section>"`
     - **Defer it** (never allowed for `blocker`; majors require finder concession):
       `xreview triage <run-id> F-003 deferred --note "<follow-up plan/issue>"`
     - **Ask one question** (at most once per finding):
       `xreview triage <run-id> F-004 question --to <reviewer> --note "<specific question>"`
     - **Lower severity** (dispute handled like a rejection):
       `xreview triage <run-id> F-005 lower_severity --severity minor --reason "<why>"`
     - **Flag a conflict** (escalates both findings to the user immediately):
       `xreview triage <run-id> F-006 conflicts_with --conflicts-with F-001 --reason "<why>"`

4. **Run Round 2+ (warm resume):**
   ```bash
   xreview round <run-id>
   ```
   - Watch for `HINT (overlapping fix lines)`: verify you did not undo an earlier fix.
   - Repeat steps 3–4 until:
     - **Exit `0` (converged):** present `LEDGER.md` (and any escalated or unverified items) to the user alongside your commit proposal.
     - **Exit `20` (paused):** stop and present the pause reason and current ledger to the user before continuing.

5. **Clean up** after the user finishes reviewing the ledger:
   ```bash
   xreview clean <run-id>
   ```
