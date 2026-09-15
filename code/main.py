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
needed. Still an INTERNAL decision object; does NOT write output.csv.

Usage:
    python3 code/main.py [--dataset-dir PATH]
    python3 code/main.py forecast --request-id REQUEST_ID [--dataset-dir PATH]
    python3 code/main.py forecast --all [--dataset-dir PATH]
    python3 code/main.py plan --request-id REQUEST_ID [--dataset-dir PATH]
    python3 code/main.py plan --all [--dataset-dir PATH]

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


def _forecast_one(request_id: str, dataset, indexes, *, verbose: bool) -> dict:
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


def _run_forecast(dataset_dir: Path, request_id: str | None, run_all: bool) -> int:
    dataset = load_dataset(dataset_dir)
    indexes = build_indexes(dataset)

    if run_all:
        total_series = 0
        total_duplicates = 0
        total_unresolved = 0
        total_breaches = 0
        for request in dataset.requests:
            summary = _forecast_one(request.request_id, dataset, indexes, verbose=False)
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

    _forecast_one(request_id, dataset, indexes, verbose=True)
    return 0


def _plan_one(request_id: str, dataset, indexes, *, verbose: bool) -> dict:
    request = indexes.requests_by_id.get(request_id)
    if request is None:
        raise SystemExit(f"ERROR: request_id {request_id!r} not found in requests.csv")

    profile = indexes.profiles_by_user[request.user_id]
    result = plan_request(request, profile, dataset, indexes)

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


def _run_plan(dataset_dir: Path, request_id: str | None, run_all: bool) -> int:
    dataset = load_dataset(dataset_dir)
    indexes = build_indexes(dataset)

    if run_all:
        counts = {
            "affordable_now": 0, "affordable_with_plan": 0,
            "affordable_later": 0, "not_affordable": 0,
        }
        with_spending_changes = 0
        for request in dataset.requests:
            summary = _plan_one(request.request_id, dataset, indexes, verbose=False)
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

    _plan_one(request_id, dataset, indexes, verbose=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? - Stage 1/2 CLI")
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

    args = parser.parse_args(argv)

    dataset_dir = getattr(args, "dataset_dir", None) or default_dataset_dir()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    if args.command == "forecast":
        return _run_forecast(dataset_dir, args.request_id, args.all)

    if args.command == "plan":
        return _run_plan(dataset_dir, args.request_id, args.all)

    return _run_validate(dataset_dir)


if __name__ == "__main__":
    sys.exit(main())