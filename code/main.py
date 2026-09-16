#!/usr/bin/env python3
"""
Buy or Wait? - entry point.

Stage 1 scope (default, no subcommand): load the dataset, build indexes,
run referential-integrity and sanity validation, and print a summary
report. Unchanged from Stage 1 - still does not compute any affordability
decision or write output.csv.

Stage 2 scope (`forecast` subcommand): reconcile one user's events as of a
request's request_date and print the deterministic 90-day cash forecast -
detected recurring series, the balance timeline, and whether the minimum
balance is ever breached. This is financial-state reconstruction only; it
does NOT decide amount_safe_to_pay or any other planner output.

Stage 3 scope (`plan` subcommand): run the deterministic affordability +
payment planner for a request - full/partial/installment/wait candidates,
ranked per the challenge's explicit rule order, plus any spending changes
needed. An INTERNAL decision object; does not write output.csv.

Stage 4 scope (`run` subcommand): the PRODUCTION entry point. Runs the
full pipeline - dataset, validated evidence facts, resolved amounts
applied to blank events, reconciliation, 90-day forecast (with Stage 4c
series amendments), Stage 3 planner - and writes output.csv.

Evidence is read from the on-disk evidence cache by default, so every
command here runs fully offline with no API key and no network. Live
model resolution is opt-in via `run --resolve-evidence`, and is the only
path that ever constructs a model client or spends API quota.

Usage:
    python3 code/main.py [--dataset-dir PATH]
    python3 code/main.py forecast --request-id REQUEST_ID [--dataset-dir PATH]
    python3 code/main.py forecast --all [--dataset-dir PATH]
    python3 code/main.py plan --request-id REQUEST_ID [--dataset-dir PATH]
    python3 code/main.py plan --all [--dataset-dir PATH]
    python3 code/main.py run [--output PATH] [--evidence-cache PATH]
    python3 code/main.py run --no-evidence        # structured data only
    python3 code/main.py run --resolve-evidence   # live model resolution

Every subcommand accepts --no-evidence to reproduce the pre-evidence
(structured-data-only) behavior for comparison.

By default, --dataset-dir resolves to <repo_root>/dataset, where repo_root
is the parent of the directory this file lives in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loader import load_dataset
from data.indexes import build_indexes
from data.validate import validate_dataset

from engine.forecast import build_forecast
from engine.reconciliation import reconcile_user_events

from planner import plan_request

import pipeline


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "dataset"


def _run_validate(dataset_dir: Path) -> int:
    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_dataset(dataset_dir)
    indexes = build_indexes(dataset)
    report = validate_dataset(dataset, indexes)

    print("\n--- Facts ---")
    for key, value in report.facts.items():
        print(f"{key}: {value}")

    print(f"\n--- Validation: {len(report.errors)} error(s), {len(report.warnings)} warning(s) ---")
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code}: {issue.message}")

    if report.ok:
        print("\nStage 1 data foundation: OK (no errors).")
        return 0
    else:
        print("\nStage 1 data foundation: FAILED - fix errors above before proceeding.")
        return 1


def _forecast_one(
    request_id: str, dataset, indexes, *, verbose: bool, evidence_facts: tuple = ()
) -> dict:
    request = indexes.requests_by_id.get(request_id)
    if request is None:
        raise SystemExit(f"ERROR: request_id {request_id!r} not found in requests.csv")

    profile = indexes.profiles_by_user[request.user_id]
    user_events = indexes.events_by_user.get(request.user_id, [])

    state = reconcile_user_events(
        request.user_id, user_events, request.request_date,
        indexes.events_by_id, profile.home_currency, indexes,
    )
    result = build_forecast(
        state,
        starting_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        evidence_facts=evidence_facts,
    )

    if verbose:
        print(f"\n=== {request_id} (user={request.user_id}, "
              f"request_date={request.request_date.isoformat()}) ===")
        print(f"Starting balance: {profile.current_available_balance} {profile.home_currency}")
        print(f"Minimum balance to keep: {profile.minimum_balance_to_keep} {profile.home_currency}")
        print(f"Recurring series detected: {len(result.recurring_series)}")
        for s in result.recurring_series:
            print(f"  - {s.key.category}/{s.key.event_type}/{s.key.direction}: "
                  f"{s.cadence} (last={s.last_date.isoformat()}, "
                  f"n_history={len(s.occurrences)}, projected_amount={s.projected_amount})")
        print(f"Generated occurrences overridden by known future events: "
              f"{result.overridden_generated_count}")
        print(f"Excluded duplicate pending events: {len(state.excluded_duplicate_ids)}")
        print(f"Unresolved blank-amount events: {len(state.unresolved_blank_ids)}")
        print(f"Checkpoints in 90-day window: {len(result.checkpoints)}")
        print(f"Minimum balance reached: {result.min_balance_reached}")
        print(f"Breaches minimum_balance_to_keep: {result.breaches_minimum()}")
        for cp in result.checkpoints:
            print(f"  {cp.date.isoformat()}  {cp.event.direction:6s} {cp.event.amount:>14} "
                  f"{cp.event.category:20s} src={cp.event.source:10s} -> balance={cp.balance_after}")

    return {
        "request_id": request_id,
        "user_id": request.user_id,
        "recurring_series_count": len(result.recurring_series),
        "excluded_duplicate_count": len(state.excluded_duplicate_ids),
        "unresolved_blank_count": len(state.unresolved_blank_ids),
        "min_balance_reached": result.min_balance_reached,
        "breaches_minimum": result.breaches_minimum(),
    }


def _run_forecast(
    dataset_dir: Path, request_id: str | None, run_all: bool,
    *, use_evidence: bool = True, cache_path: Path | None = None,
) -> int:
    prepared = pipeline.load_and_prepare(
        dataset_dir, cache_path=cache_path, use_evidence=use_evidence
    )
    dataset, indexes = prepared.dataset, prepared.indexes
    _print_evidence_banner(prepared, use_evidence)

    if run_all:
        total_series = 0
        total_duplicates = 0
        total_unresolved = 0
        total_breaches = 0
        for request in dataset.requests:
            summary = _forecast_one(
                request.request_id, dataset, indexes, verbose=False,
                evidence_facts=prepared.facts_for(request.user_id),
            )
            total_series += summary["recurring_series_count"]
            total_duplicates += summary["excluded_duplicate_count"]
            total_unresolved += summary["unresolved_blank_count"]
            if summary["breaches_minimum"]:
                total_breaches += 1
        print(f"Requests processed: {len(dataset.requests)}")
        print(f"Total recurring series detected (summed per request): {total_series}")
        print(f"Total excluded duplicate-pending events (summed per request): {total_duplicates}")
        print(f"Total unresolved blank-amount events (summed per request): {total_unresolved}")
        print(f"Requests whose forecast breaches minimum_balance_to_keep: {total_breaches}")
        return 0

    if request_id is None:
        print("ERROR: forecast requires either --request-id or --all", file=sys.stderr)
        return 2

    request = indexes.requests_by_id.get(request_id)
    facts = prepared.facts_for(request.user_id) if request is not None else ()
    _forecast_one(request_id, dataset, indexes, verbose=True, evidence_facts=facts)
    return 0


def _plan_one(
    request_id: str, dataset, indexes, *, verbose: bool, evidence_facts: tuple = ()
) -> dict:
    request = indexes.requests_by_id.get(request_id)
    if request is None:
        raise SystemExit(f"ERROR: request_id {request_id!r} not found in requests.csv")

    profile = indexes.profiles_by_user[request.user_id]
    result = plan_request(request, profile, dataset, indexes, evidence_facts=evidence_facts)

    if verbose:
        print(f"\n=== {request_id} (user={request.user_id}, "
              f"request_date={request.request_date.isoformat()}, "
              f"requested_amount={request.requested_amount}) ===")
        print(f"amount_safe_to_pay: {result.amount_safe_to_pay}")
        print(f"affordability_status: {result.affordability_status}")
        print(f"recommended_payment_method: {result.recommended_payment_method}")
        if result.chosen_plan is not None:
            for p in result.chosen_plan.payments:
                print(f"  payment: {p.date.isoformat()}  {p.amount}")
        print(f"earliest_date_for_full_payment: {result.earliest_date_for_full_payment}")
        for c in result.spending_changes:
            print(f"  spending change: {c.kind} {c.event_id} ({c.category}) "
                  f"{c.original_amount} -> {c.adjusted_amount}")

    return {
        "request_id": request_id,
        "affordability_status": result.affordability_status,
        "recommended_payment_method": result.recommended_payment_method,
        "uses_spending_changes": len(result.spending_changes) > 0,
    }


def _run_plan(
    dataset_dir: Path, request_id: str | None, run_all: bool,
    *, use_evidence: bool = True, cache_path: Path | None = None,
) -> int:
    prepared = pipeline.load_and_prepare(
        dataset_dir, cache_path=cache_path, use_evidence=use_evidence
    )
    dataset, indexes = prepared.dataset, prepared.indexes
    _print_evidence_banner(prepared, use_evidence)

    if run_all:
        counts = {
            "affordable_now": 0, "affordable_with_plan": 0,
            "affordable_later": 0, "not_affordable": 0,
        }
        with_spending_changes = 0
        for request in dataset.requests:
            summary = _plan_one(
                request.request_id, dataset, indexes, verbose=False,
                evidence_facts=prepared.facts_for(request.user_id),
            )
            counts[summary["affordability_status"]] = counts.get(summary["affordability_status"], 0) + 1
            if summary["uses_spending_changes"]:
                with_spending_changes += 1
        print(f"Requests processed: {len(dataset.requests)}")
        for status, count in counts.items():
            print(f"  {status}: {count}")
        print(f"Plans requiring spending changes: {with_spending_changes}")
        return 0

    if request_id is None:
        print("ERROR: plan requires either --request-id or --all", file=sys.stderr)
        return 2

    request = indexes.requests_by_id.get(request_id)
    facts = prepared.facts_for(request.user_id) if request is not None else ()
    _plan_one(request_id, dataset, indexes, verbose=True, evidence_facts=facts)
    return 0

def _print_evidence_banner(prepared, use_evidence: bool) -> None:
    """One line describing what evidence actually reached the engine, so a
    run is never ambiguous about whether facts were applied."""
    if not use_evidence:
        print("Evidence: DISABLED (--no-evidence) - structured dataset only")
        return
    report = prepared.apply_report
    print(
        f"Evidence: {prepared.total_facts} validated fact(s); "
        f"{len(report.applied)} blank amount(s) filled "
        f"({prepared.image_derived_count} image-derived); "
        f"{len(prepared.facts_by_user)} user(s) with series-level facts"
    )
    for skipped in report.skipped:
        print(f"  skipped {skipped.fact_id}: {skipped.reason}")


def _run_production(
    dataset_dir: Path,
    output_path: Path | None,
    *,
    use_evidence: bool = True,
    cache_path: Path | None = None,
    resolve_live: bool = False,
) -> int:
    """Stage 4 production run: full pipeline -> output.csv.

    Delegates entirely to `pipeline`, which `final_submission.py` also
    uses, so the CLI and the submission runner can never diverge.
    """
    print(f"Loading dataset from: {dataset_dir}")
    if resolve_live:
        print("Evidence: resolving LIVE via the model (consumes API quota)...")

    try:
        prepared = pipeline.load_and_prepare(
            dataset_dir,
            cache_path=cache_path,
            use_evidence=use_evidence,
            resolve_live=resolve_live,
        )
    except RuntimeError as exc:
        # Raised by GeminiAIClient when the SDK or GEMINI_API_KEY is absent.
        print(f"ERROR: live evidence resolution unavailable: {exc}", file=sys.stderr)
        print(
            "Re-run without --resolve-evidence to use the cached evidence facts.",
            file=sys.stderr,
        )
        return 2

    _print_evidence_banner(prepared, use_evidence)

    rows = pipeline.plan_all(prepared)
    out = pipeline.write_output_csv(rows, output_path)

    print(f"Wrote {len(rows)} predictions to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? - CLI and production entry point")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to the dataset/ directory (default: <repo_root>/dataset)",
    )
    subparsers = parser.add_subparsers(dest="command")

    forecast_parser = subparsers.add_parser(
        "forecast", help="Run the Stage 2 deterministic 90-day forecast for a request"
    )
    forecast_group = forecast_parser.add_mutually_exclusive_group(required=True)
    forecast_group.add_argument("--request-id", help="A request_id from requests.csv")
    forecast_group.add_argument(
        "--all", action="store_true", help="Run for every request in requests.csv and print aggregate counts"
    )

    plan_parser = subparsers.add_parser(
        "plan", help="Run the Stage 3 deterministic affordability + payment planner for a request"
    )
    plan_group = plan_parser.add_mutually_exclusive_group(required=True)
    plan_group.add_argument("--request-id", help="A request_id from requests.csv")
    plan_group.add_argument(
        "--all", action="store_true", help="Run for every request in requests.csv and print aggregate counts"
    )

    run_parser = subparsers.add_parser(
        "run", help="Stage 4 production run: full pipeline -> output.csv"
    )
    run_parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write predictions (default: <repo_root>/output.csv)",
    )
    run_parser.add_argument(
        "--resolve-evidence", action="store_true",
        help="Resolve evidence live via the model instead of using the cache "
             "(requires google-genai and GEMINI_API_KEY; consumes quota)",
    )

    # Shared evidence flags - every subcommand can turn evidence off or
    # point at a different cache, so runs stay comparable and offline.
    for sub in (forecast_parser, plan_parser, run_parser):
        sub.add_argument(
            "--no-evidence", action="store_true",
            help="Ignore evidence facts entirely (structured dataset only)",
        )
        sub.add_argument(
            "--evidence-cache", type=Path, default=None,
            help="Path to the evidence cache JSON "
                 "(default: code/evidence/cache/evidence_cache.json)",
        )

    args = parser.parse_args(argv)

    dataset_dir = getattr(args, "dataset_dir", None) or default_dataset_dir()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    use_evidence = not getattr(args, "no_evidence", False)
    cache_path = getattr(args, "evidence_cache", None)

    if args.command == "forecast":
        return _run_forecast(
            dataset_dir, args.request_id, args.all,
            use_evidence=use_evidence, cache_path=cache_path,
        )

    if args.command == "plan":
        return _run_plan(
            dataset_dir, args.request_id, args.all,
            use_evidence=use_evidence, cache_path=cache_path,
        )

    if args.command == "run":
        return _run_production(
            dataset_dir, args.output,
            use_evidence=use_evidence, cache_path=cache_path,
            resolve_live=args.resolve_evidence,
        )

    return _run_validate(dataset_dir)


if __name__ == "__main__":
    sys.exit(main())