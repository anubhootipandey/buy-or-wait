"""
Stage 3 planner: deterministic affordability + payment planning.

This package sits between Stage 2 (financial-state reconstruction / 90-day
forecast) and Stage 4 (final `output.csv` generation). It never touches
`code/data/` or `code/engine/`, never calls an LLM, and never guesses a
number - every affordability decision is derived from Stage 2's forecast
using exact `Decimal` arithmetic.

Entry point: `plan_request(request, profile, dataset, indexes)` returns a
`PlanningResult` - the internal, typed decision object Stage 4 will later
translate into the seven `output.csv` prediction columns. Stage 3 does not
build `output.csv` itself.
"""

from __future__ import annotations

from data.models import Dataset, FinancialProfile, Request
from data.indexes import Indexes

from .affordability import (
    build_user_forecast,
    earliest_date_for_full_payment,
    safe_amount_no_changes,
)
from .models import PlanningResult
from .payment_plans import (
    build_full_payment_candidates,
    build_installment_candidates,
    build_partial_payment_candidate,
    build_wait_candidate,
)
from .ranking import choose_best

__all__ = ["plan_request", "PlanningResult"]


def plan_request(
    request: Request,
    profile: FinancialProfile,
    dataset: Dataset,
    indexes: Indexes,
    evidence_facts: tuple = (),
) -> PlanningResult:
    """Run the full Stage 3 planner for one request.

    Reuses Stage 2's `reconcile_user_events` + `build_forecast` (via
    `affordability.build_user_forecast`) to get the request's 90-day
    forecast, then builds every eligible plan candidate (full payment now,
    full payment enabled by spending changes, partial payment, each
    supplied installment option, and wait), ranks them per the challenge's
    explicit lexicographic rule order, and returns the chosen plan as a
    typed `PlanningResult`.
    """
    state, forecast = build_user_forecast(request, profile, dataset, indexes, evidence_facts=evidence_facts)

    amount_safe_to_pay = safe_amount_no_changes(forecast, request.requested_amount)
    earliest_full_date = earliest_date_for_full_payment(forecast, request.requested_amount)

    candidates = []
    candidates.extend(
        build_full_payment_candidates(
            request, profile, forecast, forecast.recurring_series, indexes.events_by_id
        )
    )
    partial = build_partial_payment_candidate(
        request, profile, forecast, amount_safe_to_pay, earliest_full_date
    )
    if partial is not None:
        candidates.append(partial)
    candidates.extend(
        build_installment_candidates(
            request, profile, indexes.payment_options_by_request.get(request.request_id, []), forecast
        )
    )
    wait = build_wait_candidate(request, profile, earliest_full_date)
    if wait is not None:
        candidates.append(wait)

    chosen = choose_best(candidates, request.desired_completion_date)

    if chosen is None:
        return PlanningResult(
            request_id=request.request_id,
            user_id=request.user_id,
            requested_amount=request.requested_amount,
            amount_safe_to_pay=amount_safe_to_pay,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            chosen_plan=None,
            earliest_date_for_full_payment=earliest_full_date,
            spending_changes=[],
        )

    return PlanningResult(
        request_id=request.request_id,
        user_id=request.user_id,
        requested_amount=request.requested_amount,
        amount_safe_to_pay=amount_safe_to_pay,
        affordability_status=chosen.affordability_status,
        recommended_payment_method=chosen.method,
        chosen_plan=chosen,
        earliest_date_for_full_payment=earliest_full_date,
        spending_changes=chosen.spending_changes,
    )
