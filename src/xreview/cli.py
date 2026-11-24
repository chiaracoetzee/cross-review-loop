"""Command-line interface for xreview."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from xreview.adapters import list_adapters
from xreview.config import ConfigError, load_config, validate_repo_alias
from xreview.ledger import load_ledger, render_ledger_markdown
from xreview.policy import PolicyViolation
from xreview.quota import (
    apply_pre_flight_quota_policy,
    format_reset_time,
    inspect_plan_quotas,
    refresh_plan_quotas,
)
from xreview.protocol import (
    ProtocolError,
    attach_evidence,
    classify_files,
    clean_run,
    drop_reviewer,
    init_run,
    merge_findings,
    resolve_trusted_run_dir,
    run_pending_hooks,
    run_round,
    substitute_reviewer,
    triage_finding,
)
from xreview.sandbox import SandboxUnavailableError
from xreview.snapshot import RepoSpec, SnapshotError, UnclassifiedFilesError, check_unclassified_files
from xreview.watch import watch_reviewer

GATE_ASK_PATTERNS: tuple[str, ...] = (
    "Bash(xreview *--allow*)",
    "Bash(xreview *--reviewers*)",
    "Bash(xreview *--no-sandbox*)",
    "Bash(xreview *--resume-pause*)",
    "Bash(xreview *substitute*)",
    "Bash(xreview *drop*)",
    "Bash(python3 -m xreview *--allow*)",
    "Bash(python3 -m xreview *--reviewers*)",
    "Bash(python3 -m xreview *--no-sandbox*)",
    "Bash(python3 -m xreview *--resume-pause*)",
    "Bash(python3 -m xreview *substitute*)",
    "Bash(python3 -m xreview *drop*)",
)


def _parse_repo_spec(raw: str, base_rev: str, target_rev: str | None) -> RepoSpec:
    if "=" in raw:
        name, _, p_str = raw.partition("=")
        path = Path(p_str).expanduser().resolve()
        name = name.strip() or path.name
    else:
        path = Path(raw).expanduser().resolve()
        name = path.name
    validate_repo_alias(name)
    if not path.is_dir():
        raise ConfigError(f"Repository path is not a directory: {path}")
    return RepoSpec(name=name, path=path, base_rev=base_rev, target_rev=target_rev)


def _default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def _parse_reviewer_list(raw_list: list[str] | None) -> list[str] | None:
    if raw_list is None:
        return None
    out: list[str] = []
    for item in raw_list:
        for part in item.split(","):
            s = part.strip()
            if s and s not in out:
                out.append(s)
    return out


class _StrictArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with allow_abbrev=False enforced on every parser and subparser (astra-13)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]


def build_parser() -> argparse.ArgumentParser:
    p = _StrictArgumentParser(
        prog="xreview",
        description="Cross-vendor multi-model code and design review loop on subscription CLIs.",
    )
    p.add_argument("--config", type=Path, default=None, help="Path to user config.toml")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_StrictArgumentParser)

    # init
    p_init = sub.add_parser("init", help="Initialize a new review run")
    p_init.add_argument("--run-id", default=None, help="Custom run ID")
    p_init.add_argument("--repo", action="append", default=[], help="Repository path (or name=path); repeatable")
    p_init.add_argument("--base", default="HEAD", help="Base git revision (default: HEAD)")
    p_init.add_argument("--target", default=None, help="Target git revision (default: working tree)")
    p_init.add_argument("--doc", action="append", default=[], help="Standalone design document file; repeatable")
    p_init.add_argument("--artifact", default="", help="Short description of the artifact under review")
    p_init.add_argument("--purpose", default="", help="Purpose of the change")
    p_init.add_argument("--issue", default="", help="Issue reference or link")
    p_init.add_argument("--in-scope", default="", help="What is in scope")
    p_init.add_argument("--out-of-scope", default="", help="What is out of scope")
    p_init.add_argument("--tag", action="append", default=[], help="Run sensitivity tag; repeatable")
    p_init.add_argument("--reviewers", action="append", default=None, help="Explicit reviewer subset (comma-separated or repeatable)")
    p_init.add_argument("--allow", action="append", default=[], help="Allow vendor(s) for an evidence tag: <tag>=<v1,v2>")
    p_init.add_argument("--implementor-family", default=None, help="Model family of the implementor (e.g., claude, gemini, gpt)")
    p_init.add_argument("--no-sandbox", action="store_true", help="Explicit opt-in to run without bubblewrap sandbox")
    p_init.add_argument("--no-refresh-quota", action="store_true", help="Skip live agy/Claude quota checks during init (use cached/rollout data only)")
    p_init.add_argument("--json", action="store_true", help="Output JSON")

    # classify
    p_cls = sub.add_parser("classify", help="Classify untracked files as included or excluded")
    p_cls.add_argument("run_id")
    p_cls.add_argument("--include", action="append", default=[], help="Include untracked relative path")
    p_cls.add_argument("--exclude", action="append", default=[], help="Exclude untracked relative path")

    # evidence
    p_ev = sub.add_parser("evidence", help="Attach a tagged evidence file to a run")
    p_ev.add_argument("run_id")
    p_ev.add_argument("--file", required=True, type=Path, help="Path to evidence file")
    p_ev.add_argument("--source", required=True, help="Query or command that produced this evidence")
    p_ev.add_argument("--tag", action="append", default=[], help="Sensitivity tag on this evidence file")

    # round
    p_rnd = sub.add_parser("round", help="Execute the next review round")
    p_rnd.add_argument("run_id")
    p_rnd.add_argument("--resume-pause", action="store_true", help="Continue past a pause after user approval")
    p_rnd.add_argument("--json", action="store_true", help="Output JSON")

    # triage
    p_tri = sub.add_parser("triage", help="Record implementor disposition for an open finding")
    p_tri.add_argument("run_id")
    p_tri.add_argument("ledger_id")
    p_tri.add_argument(
        "action",
        choices=("fixed", "rejected", "deferred", "question", "conflicts_with", "lower_severity"),
    )
    p_tri.add_argument("--reason", default="", help="Reason for rejection, conflict, or severity change")
    p_tri.add_argument("--cite", default="", help="Required citation (file:line, quote, or scope section) for rejection")
    p_tri.add_argument("--note", default="", help="Fix summary, deferral follow-up note, or question text")
    p_tri.add_argument("--to", default="", help="Target finder reviewer name (for question)")
    p_tri.add_argument("--conflicts-with", dest="conflicts_with_id", default="", help="Conflicting finding ID")
    p_tri.add_argument("--severity", default="", help="Proposed lower severity (for lower_severity)")

    # merge
    p_mrg = sub.add_parser("merge", help="Merge a duplicate finding into another")
    p_mrg.add_argument("run_id")
    p_mrg.add_argument("src_id")
    p_mrg.add_argument("--into", required=True, dest="dst_id", help="Target finding ID")
    p_mrg.add_argument("--reason", default="", help="Reason for merging")

    # substitute
    p_sub = sub.add_parser("substitute", help="Replace a quota-paused/dropped reviewer with an approved substitute")
    p_sub.add_argument("run_id")
    p_sub.add_argument("old_reviewer")
    p_sub.add_argument("--with", required=True, dest="new_reviewer", help="Name of enabled substitute reviewer")

    # drop
    p_drop = sub.add_parser(
        "drop",
        help="Permanently drop a reviewer (reassigns its open fixes to another non-implementor reviewer)",
    )
    p_drop.add_argument("run_id")
    p_drop.add_argument("reviewer", help="Reviewer name to permanently drop from the run")

    # status
    p_st = sub.add_parser("status", help="Show run status, open findings, and unclassified files")
    p_st.add_argument("run_id")
    p_st.add_argument("--json", action="store_true", help="Output JSON")

    # ledger
    p_led = sub.add_parser("ledger", help="Print rendered LEDGER.md for a run")
    p_led.add_argument("run_id")

    # watch
    p_w = sub.add_parser("watch", help="Stream a single reviewer's live progress log for a round")
    p_w.add_argument("run_id")
    p_w.add_argument("--reviewer", required=True, help="Reviewer name to watch (one watcher per reviewer)")
    p_w.add_argument("--round", dest="round_num", type=int, default=None, help="Round number (default: in-progress or next round)")
    p_w.add_argument("--no-follow", action="store_true", help="Print existing events and exit immediately without waiting")
    p_w.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval in seconds (default: 0.5)")
    p_w.add_argument("--start-timeout", type=float, default=60.0, help="Max seconds to wait for initial log before exiting (default: 60)")
    p_w.add_argument("--timeout", type=float, default=None, help="Optional overall max seconds to watch before exiting")

    # clean
    p_cln = sub.add_parser("clean", help="Delete reviewer state folders and run folder")
    p_cln.add_argument("run_id")

    # hooks
    p_hk = sub.add_parser("hooks", help="Run pending (or retry failed/uncertain) post-run hooks on a converged run")
    p_hk.add_argument("run_id")
    p_hk.add_argument("--retry-failed", action="store_true", help="Also retry hooks that previously failed")
    p_hk.add_argument("--retry-uncertain", action="store_true", help="Also retry hooks left in uncertain running state")

    # setup-gates
    p_sg = sub.add_parser("setup-gates", help="Check or install harness ask permission rules for gate-lifting commands")
    p_sg.add_argument("--claude-settings", type=Path, default=Path.home() / ".claude" / "settings.json")
    p_sg.add_argument("--check", action="store_true", help="Check whether gate rules are installed without modifying files")

    # doctor
    p_doc = sub.add_parser("doctor", help="Check sandbox, harnesses, versions, and config")
    p_doc.add_argument("--json", action="store_true", help="Output JSON")

    # quota
    p_q = sub.add_parser("quota", help="Show plan quotas and automatic fallback actions")
    q_mode = p_q.add_mutually_exclusive_group()
    q_mode.add_argument("--cached", action="store_true", help="Read cached/rollout quota data only (skip live agy/Claude probes)")
    q_mode.add_argument("--refresh", action="store_true", help="Force live agy/Claude quota checks even if live_quota_probe is false")
    p_q.add_argument("--no-sandbox", action="store_true", help="Run live quota checks without bwrap sandbox (explicit opt-in)")
    p_q.add_argument("--json", action="store_true", help="Output JSON")

    return p


def _handle_setup_gates(settings_path: Path, check_only: bool) -> int:
    data: dict = {}
    if settings_path.exists():
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    perms = data.setdefault("permissions", {})
    ask_list = list(perms.get("ask", []))
    missing = [pat for pat in GATE_ASK_PATTERNS if pat not in ask_list]
    if check_only:
        if missing:
            print(f"Missing {len(missing)} gate ask rule(s) in {settings_path}:")
            for m in missing:
                print(f"  {m}")
            return 1
        print(f"All {len(GATE_ASK_PATTERNS)} gate ask rules are installed in {settings_path}.")
        return 0
    if missing:
        ask_list.extend(missing)
        perms["ask"] = ask_list
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Installed {len(missing)} gate ask rule(s) into {settings_path}.")
    else:
        print(f"All {len(GATE_ASK_PATTERNS)} gate ask rules already present in {settings_path}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)

        if args.command == "setup-gates":
            return _handle_setup_gates(args.claude_settings.expanduser(), args.check)

        if args.command == "quota":
            now_ts = datetime.now(timezone.utc).timestamp()
            probe_warnings: list[str] = []
            do_live = args.refresh or (config.policy.live_quota_probe and not args.cached)
            if do_live:
                if not args.json:
                    print("Checking plan quotas...", file=sys.stderr)
                statuses, probe_warnings = refresh_plan_quotas(
                    config,
                    force=bool(args.refresh),
                    no_sandbox=args.no_sandbox,
                    now_ts=now_ts,
                )
            else:
                statuses = inspect_plan_quotas(config, now_ts=now_ts)
            enabled = [r for r in config.reviewers if r.enabled]
            adj = apply_pre_flight_quota_policy(enabled, config, statuses, now_ts=now_ts)
            plans_out: dict[str, dict] = {}
            for pid, st in sorted(statuses.items()):
                used = st.effective_used_ratio(now_ts)
                exhausted = st.is_exhausted(now_ts)
                reset_ts = st.next_reset_at(now_ts)
                action = "ok"
                if pid in adj.dropped_exhausted_plans:
                    action = "drop_exhausted"
                elif pid in adj.fallback_plans:
                    action = "fallback_to_extras"
                elif exhausted:
                    action = "exhausted_kept"
                plans_out[pid] = {
                    "harness": st.harness,
                    "source": st.source,
                    "exhausted": exhausted,
                    "used_ratio": round(used, 4) if used is not None else None,
                    "resets_at": reset_ts,
                    "resets_local": format_reset_time(reset_ts) if reset_ts is not None else None,
                    "windows": [w.to_dict() for w in st.active_windows(now_ts)],
                    "planned_action": action,
                }
            payload = {
                "fallback_to_extras_at": config.policy.fallback_to_extras_at,
                "auto_drop_exhausted": config.policy.auto_drop_exhausted,
                "plans": plans_out,
                "planned_reviewers": [r.name for r in adj.reviewers],
                "notices": adj.notices,
                "warnings": probe_warnings,
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                for pid, pinfo in plans_out.items():
                    pct = f"{pinfo['used_ratio'] * 100:.0f}%" if pinfo["used_ratio"] is not None else "unknown"
                    rst = f", resets {pinfo['resets_local']}" if pinfo["resets_local"] else ""
                    print(
                        f"  {pid:10s} ({pinfo['harness']:6s}): {pct:7s} "
                        f"[{pinfo['planned_action']}] (source={pinfo['source']}{rst})"
                    )
                for n in adj.notices:
                    print(f"NOTICE: {n}")
                for w in probe_warnings:
                    print(f"WARNING: {w}", file=sys.stderr)
                print(f"Planned reviewers: {', '.join(payload['planned_reviewers']) or '(none)'}")
            return 0

        if args.command == "doctor":
            info = {
                "python": sys.version.split()[0],
                "bwrap": shutil.which("bwrap"),
                "socat": shutil.which("socat"),
                "reviewers_configured": len(config.reviewers),
                "profile": str(config.profile.path) if config.profile.path else None,
                "harnesses": {},
            }
            for name, adapter in sorted(list_adapters().items()):
                ver = adapter.version()
                warn = (
                    f"differs from tested {adapter.tested_version}"
                    if ver and adapter.tested_version and ver != adapter.tested_version
                    else None
                )
                info["harnesses"][name] = {
                    "version": ver,
                    "tested_version": adapter.tested_version,
                    "warning": warn,
                }
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print(f"Python:   {info['python']}")
                print(f"bwrap:    {info['bwrap'] or 'MISSING'}")
                print(f"socat:    {info['socat'] or 'not installed (will use built-in Python bridge)'}")
                print(f"Config:   {len(config.reviewers)} reviewer(s), profile={info['profile'] or 'none'}")
                for hname, hinfo in info["harnesses"].items():
                    w = f"  [WARN: {hinfo['warning']}]" if hinfo["warning"] else ""
                    print(f"  {hname:7s}: {hinfo['version'] or 'NOT FOUND'}{w}")
            return 0

        if args.command == "init":
            run_id = args.run_id or _default_run_id()
            repos = [_parse_repo_spec(r, args.base, args.target) for r in args.repo]
            # Do NOT call .resolve() on --doc before symlink validation (astra-6)
            docs = [Path(os.path.expanduser(d)) for d in args.doc]
            rev_subset = _parse_reviewer_list(args.reviewers)
            if config.policy.quota_preflight and config.policy.live_quota_probe and not args.no_refresh_quota and not args.json:
                print("Checking plan quotas...", file=sys.stderr)
            run_dir, ledger = init_run(
                config=config,
                run_id=run_id,
                repos=repos,
                artifact=args.artifact,
                purpose=args.purpose,
                issue=args.issue,
                in_scope=args.in_scope,
                out_of_scope=args.out_of_scope,
                cli_tags=args.tag,
                explicit_reviewers=rev_subset,
                raw_allows=args.allow,
                docs=docs,
                implementor_family=args.implementor_family,
                no_sandbox=args.no_sandbox,
                refresh_quota=False if args.no_refresh_quota else None,
            )
            out = {
                "run_id": ledger.run_id,
                "run_dir": str(run_dir),
                "reviewers": list(ledger.reviewers.keys()),
                "run_tags": ledger.run_tags,
                "sandbox_mode": ledger.sandbox_mode,
                "gates_lifted": [g.to_dict() for g in ledger.gates_lifted],
                "quota_adjustments": list(ledger.quota_adjustments),
                "quota_warnings": list(ledger.quota_warnings),
                "version_warnings": {
                    k: v.version_warning for k, v in ledger.reviewers.items() if v.version_warning
                },
            }
            if args.json:
                print(json.dumps(out, indent=2))
                for note in ledger.quota_adjustments:
                    print(f"NOTICE (quota): {note}", file=sys.stderr)
                for warn in ledger.quota_warnings:
                    print(f"WARNING (quota): {warn}", file=sys.stderr)
            else:
                print(f"Initialized run {ledger.run_id} at {run_dir}")
                print(f"Reviewers: {', '.join(out['reviewers'])}")
                print(f"Sandbox:   {ledger.sandbox_mode}")
                for note in ledger.quota_adjustments:
                    print(f"NOTICE (quota): {note}", file=sys.stderr)
                for warn in ledger.quota_warnings:
                    print(f"WARNING (quota): {warn}", file=sys.stderr)
                if out["version_warnings"]:
                    for r, w in out["version_warnings"].items():
                        print(f"WARNING ({r}): {w}", file=sys.stderr)
            return 0

        if args.command == "clean":
            clean_run(config, args.run_id)
            print(f"Cleaned run {args.run_id}")
            return 0

        # Resolve run_id strictly under trusted config.state_dir (astra-2)
        run_dir = resolve_trusted_run_dir(config, args.run_id)

        if args.command == "classify":
            ledger = classify_files(run_dir, include=args.include, exclude=args.exclude)
            print(
                f"Run {ledger.run_id}: included={len(ledger.included_files)}, excluded={len(ledger.excluded_files)}"
            )
            return 0

        if args.command == "evidence":
            # Do NOT call .resolve() before attach_evidence validates symlinks (astra-6)
            ledger, item = attach_evidence(run_dir, Path(os.path.expanduser(str(args.file))), args.source, args.tag)
            print(f"Attached evidence {item.id} ({item.filename}, tags={list(item.tags)}) to run {ledger.run_id}")
            return 0

        if args.command == "triage":
            ledger = triage_finding(
                run_dir,
                args.ledger_id,
                args.action,
                reason=args.reason,
                citation=args.cite,
                note=args.note,
                to_reviewer=args.to,
                conflicts_with_id=args.conflicts_with_id,
                new_severity=args.severity,
            )
            f = ledger.get_finding(args.ledger_id)
            print(f"{f.ledger_id}: state={f.state} disposition={f.disposition}")
            return 0

        if args.command == "merge":
            ledger = merge_findings(run_dir, args.src_id, args.dst_id, reason=args.reason)
            print(f"Merged {args.src_id} into {args.dst_id}")
            return 0

        if args.command == "substitute":
            ledger = substitute_reviewer(run_dir, config, args.old_reviewer, args.new_reviewer)
            print(f"Substituted {args.old_reviewer} -> {args.new_reviewer} in run {ledger.run_id}")
            return 0

        if args.command == "drop":
            ledger = drop_reviewer(run_dir, config, args.reviewer)
            live_names = [r.config.name for r in ledger.live_reviewers()]
            print(
                f"Dropped {args.reviewer} from run {ledger.run_id} "
                f"(status={ledger.status}; active={', '.join(live_names)})"
            )
            return 0

        if args.command == "round":
            outcome = run_round(run_dir, config, resume_pause=args.resume_pause)
            if args.json:
                print(
                    json.dumps(
                        {
                            "round": outcome.round_num,
                            "status": outcome.status,
                            "pause_reason": outcome.pause_reason,
                            "new_findings": outcome.new_findings,
                            "closed_findings": outcome.closed_findings,
                            "escalated_findings": outcome.escalated_findings,
                            "unverified_findings": outcome.unverified_findings,
                            "fix_overlap_hints": outcome.fix_overlap_hints,
                            "warnings": outcome.warnings,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"Round {outcome.round_num} finished: status={outcome.status}")
                if outcome.pause_reason:
                    print(f"Pause reason: {outcome.pause_reason}")
                if outcome.new_findings:
                    print(f"New findings:       {', '.join(outcome.new_findings)}")
                if outcome.closed_findings:
                    print(f"Closed findings:    {', '.join(outcome.closed_findings)}")
                if outcome.escalated_findings:
                    print(f"Escalated findings: {', '.join(outcome.escalated_findings)}")
                if outcome.unverified_findings:
                    print(f"Unverified:         {', '.join(outcome.unverified_findings)}")
                for h in outcome.fix_overlap_hints:
                    print(f"HINT (overlapping fix lines): {h}")
                for w in outcome.warnings:
                    print(f"WARNING: {w}", file=sys.stderr)
                print(f"Ledger: {run_dir / 'LEDGER.md'}")

            if outcome.status == "converged":
                return 0
            if outcome.status == "needs_triage":
                return 10
            if outcome.status == "paused":
                return 20
            return 0

        if args.command == "status":
            ledger = load_ledger(run_dir)
            prof_excl: dict[str, set[str]] = {}
            for repo in ledger.repos:
                rpath = str(repo.path.resolve())
                for rule in config.profile.repo_rules:
                    if rpath == rule.path or rpath.startswith(rule.path.rstrip("/") + "/"):
                        prof_excl.setdefault(repo.name, set()).update(rule.exclude_submodules)
            unclassified = check_unclassified_files(
                ledger.repos, set(ledger.included_files), set(ledger.excluded_files), prof_excl
            )
            open_findings = [f.to_dict() for f in ledger.findings if f.state == "open"]
            payload = {
                "run_id": ledger.run_id,
                "status": ledger.status,
                "round": ledger.current_round,
                "in_progress_round": ledger.in_progress_round,
                "pause_reason": ledger.pause_reason,
                "unclassified_files": unclassified,
                "open_findings": open_findings,
                "escalated_count": sum(1 for f in ledger.findings if f.state == "escalated"),
                "closed_count": sum(1 for f in ledger.findings if f.state == "closed"),
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Run {ledger.run_id} | status={ledger.status} | round={ledger.current_round}")
                if ledger.pause_reason:
                    print(f"Pause: {ledger.pause_reason}")
                if unclassified:
                    print("Unclassified untracked files:")
                    for uf in unclassified:
                        print(f"  {uf}")
                print(f"Open findings: {len(open_findings)} | Closed: {payload['closed_count']} | Escalated: {payload['escalated_count']}")
                for f in ledger.findings:
                    if f.state == "open":
                        print(
                            f"  {f.ledger_id} [{f.effective_severity}] {f.file}:{f.line} "
                            f"(disp={f.disposition or 'NONE'}): {f.title}"
                        )
            return 0

        if args.command == "ledger":
            ledger = load_ledger(run_dir)
            print(render_ledger_markdown(ledger))
            return 0

        if args.command == "watch":
            poll_iv = float(args.poll_interval)
            if not math.isfinite(poll_iv) or poll_iv <= 0:
                raise ConfigError("--poll-interval must be a positive finite number")
            start_to = float(args.start_timeout)
            if not math.isfinite(start_to) or start_to < 0:
                raise ConfigError("--start-timeout must be a non-negative finite number")
            total_to = None
            if args.timeout is not None:
                total_to = float(args.timeout)
                if not math.isfinite(total_to) or total_to <= 0:
                    raise ConfigError("--timeout must be a positive finite number")
            try:
                return watch_reviewer(
                    run_dir,
                    args.reviewer,
                    round_num=args.round_num,
                    follow=not args.no_follow,
                    poll_interval=max(0.05, poll_iv),
                    start_timeout_s=start_to,
                    timeout_s=total_to,
                )
            except KeyboardInterrupt:
                return 0

        if args.command == "hooks":
            results = run_pending_hooks(
                run_dir, config, retry_failed=args.retry_failed, retry_uncertain=args.retry_uncertain
            )
            print(json.dumps(results, indent=2))
            return 0 if all(h.get("status") == "ok" for h in results) else 1

    except UnclassifiedFilesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ConfigError, PolicyViolation, ProtocolError, SnapshotError, SandboxUnavailableError, FileNotFoundError, KeyError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
