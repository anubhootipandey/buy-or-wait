"""
Stage 2 reconciliation: turn raw `FinancialEvent` rows for one user into a
`ReconciledState` - the normalized settled history + known future events a
forecast is allowed to use.

Rules implemented here (see PROJECT_STATE.md / DECISIONS.md for the
approved Stage 2 rule set this mirrors exactly):

  * settled            -> counts normally (history if <= request_date,
                           known_future if a settled event is somehow
                           dated after request_date)
  * cancelled          -> excluded
  * failed             -> excluded
  * pending, credit    -> NOT confirmed income -> excluded
  * pending, debit     -> known_future (a future forecast may need it)
  * scheduled          -> known_future (either direction; this is how the
                           dataset's "next confirmed salary" is represented)
  * unrealized/non_cash-> never cash, always excluded
  * blank amount       -> unresolved; NEVER coerced to 0; excluded from
                           every cash calculation until a later stage
                           resolves it from an image (out of scope here)
  * duplicate pending  -> a pending event that is a structural duplicate of
                           an already-settled linked event (same user,
                           category, direction, amount) is excluded from
                           both the cash ledger and recurrence detection

Currency: every amount is converted to the user's home currency using the
existing Stage 1 exact-date exchange-rate lookup (`data.currency.convert`),
using the event's settlement_date (falling back to event_date only for the
handful of statuses where settlement_date might be blank - none are, per
FACTS.md, but the fallback keeps this robust rather than crashing).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from data.currency import MissingExchangeRateError, convert
from data.indexes import Indexes
from data.models import EventDirection, EventStatus, FinancialEvent

from .state import CashEvent, ReconciledState


def _is_duplicate_pending(event: FinancialEvent, events_by_id: dict[str, FinancialEvent]) -> bool:
    """Rule 3: pending + linked settled event + same user/category/direction/amount."""
    if event.status != EventStatus.PENDING.value:
        return False
    if event.linked_event_id is None:
        return False
    linked = events_by_id.get(event.linked_event_id)
    if linked is None:
        return False
    if linked.status != EventStatus.SETTLED.value:
        return False
    return (
        linked.user_id == event.user_id
        and linked.category == event.category
        and linked.direction == event.direction
        and linked.amount is not None
        and event.amount is not None
        and linked.amount == event.amount
    )


def reconcile_user_events(
    user_id: str,
    user_events: Iterable[FinancialEvent],
    request_date: date,
    events_by_id: dict[str, FinancialEvent],
    home_currency: str,
    indexes: Indexes,
) -> ReconciledState:
    """Reconcile every event belonging to `user_id` as of `request_date`.

    `user_events` should already be filtered to this user (callers
    typically pass `indexes.events_by_user.get(user_id, [])`); `events_by_id`
    is the full cross-user lookup, needed to resolve `linked_event_id`.
    """
    settled_history: list[CashEvent] = []
    known_future: list[CashEvent] = []
    excluded_duplicate_ids: list[str] = []
    unresolved_blank_ids: list[str] = []
    excluded_ids: list[str] = []

    for e in user_events:
        # Blank amount: NEVER treat as zero. Stays unresolved regardless of
        # status - Stage 2 does no image interpretation (that is Stage 3).
        if e.amount is None:
            unresolved_blank_ids.append(e.event_id)
            continue

        if e.direction == EventDirection.NON_CASH.value:
            excluded_ids.append(e.event_id)
            continue

        if e.status in (EventStatus.CANCELLED.value, EventStatus.FAILED.value):
            excluded_ids.append(e.event_id)
            continue

        if e.status == EventStatus.UNREALIZED.value:
            excluded_ids.append(e.event_id)
            continue

        if _is_duplicate_pending(e, events_by_id):
            excluded_duplicate_ids.append(e.event_id)
            continue

        if e.status == EventStatus.PENDING.value and e.direction == EventDirection.CREDIT.value:
            # Rule 2: pending credit is NOT confirmed income.
            excluded_ids.append(e.event_id)
            continue

        cash_date = e.settlement_date or e.event_date

        try:
            home_amount = convert(e.amount, e.currency, home_currency, cash_date, indexes)
        except MissingExchangeRateError:
            # Should not happen on this dataset (FACTS.md: 0 missing rates),
            # but fail safe rather than silently mis-valuing a cash amount.
            excluded_ids.append(e.event_id)
            continue

        cash_event = CashEvent(
            date=cash_date,
            amount=home_amount,
            direction=e.direction,
            category=e.category,
            event_type=e.event_type,
            flexibility=e.flexibility,
            source="settled" if e.status == EventStatus.SETTLED.value else "known_future",
            event_id=e.event_id,
        )

        if e.status == EventStatus.SETTLED.value:
            if cash_date <= request_date:
                settled_history.append(cash_event)
            else:
                # A settled event dated after request_date is still a real,
                # already-realized cash movement - include it directly, not
                # via recurrence projection.
                known_future.append(cash_event)
            continue

        if e.status in (EventStatus.PENDING.value, EventStatus.SCHEDULED.value):
            # pending-debit or scheduled(any direction): a real known future
            # cash movement already on file.
            known_future.append(cash_event)
            continue

        # Any other/unrecognized status: exclude rather than silently
        # assume it is cash (mirrors Stage 1's "surface, don't guess" stance).
        excluded_ids.append(e.event_id)

    return ReconciledState(
        user_id=user_id,
        request_date=request_date,
        settled_history=settled_history,
        known_future=known_future,
        excluded_duplicate_ids=excluded_duplicate_ids,
        unresolved_blank_ids=unresolved_blank_ids,
        excluded_ids=excluded_ids,
    )
