"""
Stage 3 candidate plan builders.

Each function here returns zero or more `PlanCandidate` objects for one
payment method, already filtered down to only the candidates that are
independently safe and eligible per the challenge rules. `ranking.py` then
picks the best one; nothing here decides between candidates of different
methods itself.

Shared assumption (see PROJECT_STATE.md / this stage's docs): an immediate
method (`full_payment`, `partial_payment`, `installments`) is only
considered eligible if its last payment lands on or before
`desired_completion_date`, and `wait` is only eligible if the base
(no-spending-change) `earliest_date_for_full_payment` is on or before
`desired_completion_date` too - matching the problem statement's "a
recommendation is safe only if ... the user can complete the full request
by its deadline."
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from data.models import FinancialEvent, FinancialProfile, PaymentOption, Request

from engine.forecast import ForecastResult
from engine.recurrence import RecurringSeries

from .affordability import is_safe_with_debits, safe_amount_on
from .models import PlanCandidate, PlannedPayment
from .spending_changes import find_minimal_spending_change_combo


def build_full_payment_candidates(
    request: Request,
    profile: FinancialProfile,
    forecast: ForecastResult,
    recurring_series: list[RecurringSeries],
    events_by_id: dict[str, FinancialEvent],
) -> list[PlanCandidate]:
    """Full payment on `request_date`: `affordable_now` if already safe
    without any spending change; otherwise `affordable_with_plan` if the
    minimal spending-change combination makes it safe. Per the challenge
    rules ("if full payment is safe on the request date, prefer it over
    unnecessary installments"), the no-change candidate is returned alone
    when it already works - there is never a reason to also propose a
    spending-change variant of the same method/date/amount."""
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return []
    if request.request_date > request.desired_completion_date:
        return []

    safe_now = safe_amount_on(forecast, request.request_date, request.requested_amount)
    if safe_now >= request.requested_amount:
        return [
            PlanCandidate(
                method="full_payment",
                affordability_status="affordable_now",
                payments=[PlannedPayment(request.request_date, request.requested_amount)],
                spending_changes=[],
            )
        ]

    combo_result = find_minimal_spending_change_combo(
        forecast, request.requested_amount, request.request_date, profile, recurring_series, events_by_id
    )
    if combo_result is None:
        return []
    combo, _modified_forecast = combo_result
    return [
        PlanCandidate(
            method="full_payment",
            affordability_status="affordable_with_plan",
            payments=[PlannedPayment(request.request_date, request.requested_amount)],
            spending_changes=combo,
        )
    ]


def build_wait_candidate(
    request: Request,
    profile: FinancialProfile,
    earliest_date_for_full_payment: date | None,
) -> PlanCandidate | None:
    """`affordable_later`: full payment becomes safe on a later date
    (computed without spending changes, per the field's own definition),
    provided that date is still on or before `desired_completion_date`."""
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None
    if earliest_date_for_full_payment is None:
        return None
    if earliest_date_for_full_payment <= request.request_date:
        # Already safe today - the full_payment/affordable_now candidate
        # covers this; a "wait" for the same date would be a duplicate.
        return None
    if earliest_date_for_full_payment > request.desired_completion_date:
        return None
    return PlanCandidate(
        method="wait",
        affordability_status="affordable_later",
        payments=[PlannedPayment(earliest_date_for_full_payment, request.requested_amount)],
        spending_changes=[],
    )


def build_partial_payment_candidate(
    request: Request,
    profile: FinancialProfile,
    forecast: ForecastResult,
    amount_safe_to_pay: Decimal,
    earliest_date_for_full_payment: date | None,
) -> PlanCandidate | None:
    """`affordable_with_plan` via exactly two payments: `amount_safe_to_pay`
    today, then the remainder on `earliest_date_for_full_payment`. Eligible
    only when every stated condition holds; never invents a plan otherwise.
    """
    if not request.allows_partial_payment:
        return None
    if "partial_payment" not in profile.payment_methods_user_will_consider:
        return None
    if not (Decimal("0") < amount_safe_to_pay < request.requested_amount):
        return None
    if earliest_date_for_full_payment is None:
        return None
    if earliest_date_for_full_payment > request.desired_completion_date:
        return None

    remaining = request.requested_amount - amount_safe_to_pay
    payments = [
        PlannedPayment(request.request_date, amount_safe_to_pay),
        PlannedPayment(earliest_date_for_full_payment, remaining),
    ]
    if not is_safe_with_debits(forecast, [(p.date, p.amount) for p in payments]):
        return None

    return PlanCandidate(
        method="partial_payment",
        affordability_status="affordable_with_plan",
        payments=payments,
        spending_changes=[],
    )


def build_installment_candidates(
    request: Request,
    profile: FinancialProfile,
    payment_options: list[PaymentOption],
    forecast: ForecastResult,
) -> list[PlanCandidate]:
    """One candidate per supplied `installments` payment option that (a)
    respects `max_installment_months`, (b) completes on or before
    `desired_completion_date`, and (c) is safe for every one of its
    payments against the base forecast (exactly matching the supplied
    schedule - never a Stage-3-invented schedule)."""
    if "installments" not in profile.payment_methods_user_will_consider:
        return []
    if profile.max_installment_months is None:
        return []

    candidates: list[PlanCandidate] = []
    for po in payment_options:
        if po.payment_method != "installments":
            continue
        if po.number_of_payments > profile.max_installment_months:
            continue

        payments: list[PlannedPayment] = []
        current_date = po.first_payment_date
        for _ in range(po.number_of_payments):
            payments.append(PlannedPayment(date=current_date, amount=po.payment_amount))
            if po.payment_frequency_days:
                current_date = current_date + timedelta(days=po.payment_frequency_days)

        if payments[-1].date > request.desired_completion_date:
            continue
        if not is_safe_with_debits(forecast, [(p.date, p.amount) for p in payments]):
            continue

        candidates.append(
            PlanCandidate(
                method="installments",
                affordability_status="affordable_with_plan",
                payments=payments,
                spending_changes=[],
                payment_option_id=po.payment_option_id,
            )
        )
    return candidates
