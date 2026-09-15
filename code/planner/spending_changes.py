"""
Stage 3 spending-change planning.

Builds a small, deterministic set of candidate spending changes from a
user's already-detected Stage 2 recurring series, then searches - smallest
and gentlest combination first - for the minimal combination (at most 3)
that makes an otherwise-unsafe full payment safe on `request_date`.

Eligibility mirrors the challenge rules exactly:

  * only DEBIT recurring series can be changed (never income)
  * a category the user's profile protects (`expense_categories_to_protect`)
    - or one of a small set of categories that are always protected
      regardless of what the profile says - can never be changed
  * "reduce_to" requires the category to be in
    `expense_categories_user_is_willing_to_reduce` AND the series'
    flexibility to be "reducible" or "reducible_or_stoppable" AND a
    `minimum_allowed_amount` to exist on the template event (the reduction
    target - the maximum permitted saving)
  * "stop" requires the category to be in
    `expense_categories_user_is_willing_to_stop` AND the series'
    flexibility to be "stoppable" or "reducible_or_stoppable"
  * stopping and reducing the *same* event are mutually exclusive within
    one combination (enforced by construction: a combination can use at
    most one action per `event_id`)

Search order (deterministic, gentlest-first): within each combination size
(1, 2, 3), combinations are tried in order of ascending total severity
(reduce = 0, stop = 1 per action) and then by `event_id`, so a solution
that only needs a partial reduction is always preferred over one that
stops something outright, and the first safe combination found is used -
matching the real dataset's worked examples (see BUILD_LOG.md / FACTS.md).
"""

from __future__ import annotations

import dataclasses
import itertools
from datetime import date
from decimal import Decimal

from data.models import FinancialEvent, FinancialProfile
from engine.forecast import ForecastCheckpoint, ForecastResult
from engine.recurrence import RecurringSeries

from .affordability import safe_amount_on
from .models import SpendingChangeAction

# Categories that must never be reduced or stopped, regardless of what a
# profile's own `expense_categories_to_protect` says (defense in depth -
# on the real dataset every event in these categories is already
# `flexibility=fixed`, so this never actually changes an outcome, but the
# challenge rules call these out explicitly as always-protected).
ALWAYS_PROTECTED_CATEGORIES = frozenset(
    {"rent", "housing", "utilities", "debt_repayment", "insurance", "healthcare"}
)

MAX_SPENDING_CHANGES = 3


def eligible_series_actions(
    profile: FinancialProfile,
    recurring_series: list[RecurringSeries],
    events_by_id: dict[str, FinancialEvent],
) -> list[SpendingChangeAction]:
    """Every atomic spending-change candidate available for this user's
    detected recurring series, sorted gentlest-first (reduce before stop),
    then by `event_id` for full determinism."""
    protect = set(profile.expense_categories_to_protect) | ALWAYS_PROTECTED_CATEGORIES
    can_reduce = set(profile.expense_categories_user_is_willing_to_reduce)
    can_stop = set(profile.expense_categories_user_is_willing_to_stop)

    actions: list[SpendingChangeAction] = []
    for series in recurring_series:
        if series.key.direction != "debit":
            continue
        category = series.key.category
        if category in protect:
            continue

        template = series.occurrences[-1]
        event_id = template.event_id
        if event_id is None:
            continue
        original_amount = series.projected_amount
        series_key = (series.key.category, series.key.event_type, series.key.direction)

        if series.flexibility in ("reducible", "reducible_or_stoppable") and category in can_reduce:
            fin_event = events_by_id.get(event_id)
            min_allowed = fin_event.minimum_allowed_amount if fin_event else None
            if min_allowed is not None and min_allowed < original_amount:
                actions.append(
                    SpendingChangeAction(
                        kind="reduce_to",
                        event_id=event_id,
                        category=category,
                        series_key=series_key,
                        original_amount=original_amount,
                        adjusted_amount=min_allowed,
                        severity=0,
                    )
                )

        if series.flexibility in ("stoppable", "reducible_or_stoppable") and category in can_stop:
            actions.append(
                SpendingChangeAction(
                    kind="stop",
                    event_id=event_id,
                    category=category,
                    series_key=series_key,
                    original_amount=original_amount,
                    adjusted_amount=Decimal("0"),
                    severity=1,
                )
            )

    actions.sort(key=lambda a: (a.severity, a.event_id))
    return actions


def apply_changes_to_forecast(
    forecast: ForecastResult, changes: list[SpendingChangeAction]
) -> ForecastResult:
    """Return a new `ForecastResult` with every future (`known_future` or
    `recurring`) checkpoint matching one of `changes`' series adjusted to
    that change's amount, and the balance walk redone from scratch.
    Historical `settled` checkpoints (today's already-committed cash) are
    never touched - only forward-looking events can be changed."""
    changes_by_key = {c.series_key: c for c in changes}

    new_checkpoints: list[ForecastCheckpoint] = []
    balance = forecast.starting_balance
    for cp in forecast.checkpoints:
        event = cp.event
        key = (event.category, event.event_type, event.direction)
        change = changes_by_key.get(key)
        if change is not None and event.source in ("known_future", "recurring"):
            new_amount = min(event.amount, change.adjusted_amount)
            event = dataclasses.replace(event, amount=new_amount)
        balance = balance + event.signed_amount
        new_checkpoints.append(ForecastCheckpoint(date=cp.date, event=event, balance_after=balance))

    return ForecastResult(
        user_id=forecast.user_id,
        request_date=forecast.request_date,
        starting_balance=forecast.starting_balance,
        minimum_balance_to_keep=forecast.minimum_balance_to_keep,
        checkpoints=new_checkpoints,
        recurring_series=forecast.recurring_series,
        overridden_generated_count=forecast.overridden_generated_count,
    )


def find_minimal_spending_change_combo(
    forecast: ForecastResult,
    requested_amount: Decimal,
    anchor: date,
    profile: FinancialProfile,
    recurring_series: list[RecurringSeries],
    events_by_id: dict[str, FinancialEvent],
) -> tuple[list[SpendingChangeAction], ForecastResult] | None:
    """Search combinations of up to `MAX_SPENDING_CHANGES` eligible actions,
    smallest size first and gentlest-total-severity first within a size,
    for the first one that makes the full `requested_amount` safe to pay
    on `anchor`. Returns `(chosen_actions, modified_forecast)` or `None` if
    no combination of size <= MAX_SPENDING_CHANGES suffices."""
    actions = eligible_series_actions(profile, recurring_series, events_by_id)
    if not actions:
        return None

    for size in range(1, MAX_SPENDING_CHANGES + 1):
        combos = list(itertools.combinations(actions, size))
        # Exclude any combination that touches the same event_id twice
        # (mutual exclusivity between "stop" and "reduce_to" on one event).
        combos = [c for c in combos if len({a.event_id for a in c}) == len(c)]
        combos.sort(key=lambda c: (sum(a.severity for a in c), tuple(a.event_id for a in c)))
        for combo in combos:
            modified = apply_changes_to_forecast(forecast, list(combo))
            if safe_amount_on(modified, anchor, requested_amount) >= requested_amount:
                return list(combo), modified
    return None
