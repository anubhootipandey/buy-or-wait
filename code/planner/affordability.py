"""
Stage 3 affordability primitives.

Everything here reads a Stage 2 `ForecastResult` and derives exact,
deterministic financial facts from it - it never re-implements
reconciliation or recurrence detection (that stays Stage 2's job), and it
never coerces a decision from anything other than `Decimal` arithmetic.

Core idea: paying an amount `X` on some date `d` is a single extra debit
that permanently reduces every subsequent balance checkpoint by `X`. So the
largest safe `X` at `d` is bounded by the *lowest* balance the forecast
ever reaches from `d` onward (including the instant just before `d`'s own
events, since a payment on `d` is assumed to happen before same-day
events - the conservative assumption, matching how Stage 2 itself treats
`current_available_balance` as the balance strictly BEFORE any event dated
on `request_date`).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Iterable

from data.indexes import Indexes
from data.models import Dataset, FinancialProfile, Request

from engine.forecast import ForecastResult, build_forecast
from engine.reconciliation import reconcile_user_events
from engine.state import ReconciledState


def build_user_forecast(
    request: Request,
    profile: FinancialProfile,
    dataset: Dataset,
    indexes: Indexes,
    evidence_facts: tuple = (),
) -> tuple[ReconciledState, ForecastResult]:
    """Reconcile + forecast one request's user as of `request.request_date`,
    using exactly the Stage 2 engine (no re-derivation)."""
    user_events = indexes.events_by_user.get(request.user_id, [])
    state = reconcile_user_events(
        request.user_id,
        user_events,
        request.request_date,
        indexes.events_by_id,
        profile.home_currency,
        indexes,
    )
    forecast = build_forecast(
        state,
        starting_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        evidence_facts=evidence_facts,
    )
    return state, forecast


def min_forward_balance(forecast: ForecastResult, anchor: date) -> Decimal:
    """The lowest balance the forecast ever reaches from `anchor` onward.

    `anchor == request_date` is a special case matching Stage 2's own rule
    that `current_available_balance` is the balance BEFORE any event dated
    on `request_date` (so a same-day credit is not assumed already
    available for today's payment). Every later anchor date uses the
    standard forecast convention already established by
    `ForecastResult.balance_on` - the balance AFTER that date's own events
    - since by definition every event on a future date is already known
    and forecast-scheduled, not a same-day surprise.
    """
    if anchor <= forecast.request_date:
        reference = forecast.starting_balance
        future_checkpoints = [cp for cp in forecast.checkpoints if cp.date >= anchor]
    else:
        reference = forecast.balance_on(anchor)
        future_checkpoints = [cp for cp in forecast.checkpoints if cp.date > anchor]
    values = [reference] + [cp.balance_after for cp in future_checkpoints]
    return min(values)


def safe_amount_on(
    forecast: ForecastResult,
    anchor: date,
    requested_amount: Decimal,
) -> Decimal:
    """The largest amount safely payable on `anchor`, capped at
    `requested_amount` and never negative."""
    headroom = min_forward_balance(forecast, anchor) - forecast.minimum_balance_to_keep
    return max(Decimal("0"), min(requested_amount, headroom))


def safe_amount_no_changes(forecast: ForecastResult, requested_amount: Decimal) -> Decimal:
    """`amount_safe_to_pay`: the largest amount safe on `request_date`
    before any optional spending change, capped at `requested_amount`."""
    return safe_amount_on(forecast, forecast.request_date, requested_amount)


def earliest_date_for_full_payment(
    forecast: ForecastResult,
    requested_amount: Decimal,
) -> date | None:
    """`earliest_date_for_full_payment`: the first date (within the 90-day
    forecast horizon) on which paying the full `requested_amount` as a
    single payment stays safe for the rest of the forecast, computed
    WITHOUT any spending change (per the problem statement's explicit
    definition of this field). `safe_amount_on` is non-decreasing in the
    anchor date (later anchors only drop earlier, already-passed
    constraints), so it is enough to test `request_date` plus every
    distinct checkpoint date and return the first one that qualifies.
    """
    if requested_amount <= 0:
        return forecast.request_date

    candidate_dates = sorted({forecast.request_date} | {cp.date for cp in forecast.checkpoints})
    for d in candidate_dates:
        if safe_amount_on(forecast, d, requested_amount) >= requested_amount:
            return d
    return None


def simulate_min_balance(
    forecast: ForecastResult,
    extra_debits: Iterable[tuple[date, Decimal]],
) -> Decimal:
    """General-purpose safety simulator: merge one or more externally-applied
    one-time debits (e.g. installment payments, or a partial-payment's two
    payments) into the base forecast's chronological event ledger and
    return the minimum balance reached anywhere in the timeline (including
    the instant before `request_date`'s own events).

    Same-date tie-break: a debit dated `request_date` is applied before the
    forecast's own same-day events (conservative - matches
    `min_forward_balance`'s treatment of `request_date`); a debit dated any
    later day is applied after that day's own forecast events (consistent
    with `min_forward_balance` treating a future date's events, including
    a same-day credit, as already resolved by the time a scheduled payment
    on that date lands). Ties within the forecast's own checkpoints keep
    their existing (already deterministic) relative order via a stable
    sort.
    """
    events: list[tuple[date, int, Decimal]] = []
    for d, amount in extra_debits:
        # Same-day tie-break: a debit on `request_date` is conservatively
        # applied BEFORE that day's forecast events (rank 0, matching how
        # `min_forward_balance` treats `request_date` - see its docstring).
        # A debit on any later date is applied AFTER that day's forecast
        # events (rank 2) - consistent with `min_forward_balance` treating
        # a future date's own events (e.g. a same-day salary credit) as
        # already resolved by the time a scheduled payment on that date is
        # made.
        rank = 0 if d <= forecast.request_date else 2
        events.append((d, rank, -amount))
    for cp in forecast.checkpoints:
        events.append((cp.date, 1, cp.event.signed_amount))

    events.sort(key=lambda t: (t[0], t[1]))

    balance = forecast.starting_balance
    min_balance = balance
    for _d, _rank, delta in events:
        balance += delta
        if balance < min_balance:
            min_balance = balance
    return min_balance


def is_safe_with_debits(
    forecast: ForecastResult,
    extra_debits: Iterable[tuple[date, Decimal]],
) -> bool:
    return simulate_min_balance(forecast, extra_debits) >= forecast.minimum_balance_to_keep