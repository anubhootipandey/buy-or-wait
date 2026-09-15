"""
Normalized financial-state types shared across the Stage 2 engine.

`CashEvent` is the engine's own currency-neutral (already-converted-to-
home-currency), amount-resolved, sign-agnostic representation of a single
cash movement. It intentionally throws away everything the forecast layer
doesn't need (raw status strings, linkage) once reconciliation has already
made the keep/exclude/duplicate decision - so the forecast/recurrence code
never has to re-litigate reconciliation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional


class CashDirection(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class EventSource(str, Enum):
    # An actual settled historical cash movement (from financial_events.csv).
    SETTLED = "settled"
    # An actual pending/scheduled (or future-dated settled) cash movement
    # already on file - never generated, always traceable to one event_id.
    KNOWN_FUTURE = "known_future"
    # A projected occurrence of a detected recurring series. Never has an
    # event_id, because it does not exist in financial_events.csv.
    RECURRING = "recurring"


@dataclass(frozen=True)
class CashEvent:
    """One normalized, home-currency, amount-resolved cash movement."""

    date: date
    amount: Decimal  # always >= 0; sign comes from `direction`
    direction: str  # CashDirection value
    category: str
    event_type: str
    flexibility: str
    source: str  # EventSource value
    event_id: Optional[str] = None  # None only for source == "recurring"
    series_key: Optional[tuple] = None  # set for recurring-related events

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction == CashDirection.CREDIT.value else -self.amount


@dataclass
class ReconciledState:
    """Output of `reconciliation.reconcile_user_events` for one user.

    Everything here has already been amount-resolved (no blanks), currency-
    converted to the user's home currency, and status-filtered per the
    Stage 2 reconciliation rules. Downstream code (recurrence detection,
    forecasting) never needs to look at raw event status again.
    """

    user_id: str
    request_date: date

    # Settled, resolved-amount cash movements dated on/before request_date.
    # This - and only this - is the history recurrence detection is allowed
    # to use (Rule: trailing pending/scheduled events never establish
    # cadence, because they are never in this list).
    settled_history: list[CashEvent] = field(default_factory=list)

    # Real (non-generated) future cash movements already on file: pending
    # debits, scheduled events (any direction), and settled events dated
    # after request_date. Each still carries its own event_id.
    known_future: list[CashEvent] = field(default_factory=list)

    # event_ids excluded as structural duplicate pending charges.
    excluded_duplicate_ids: list[str] = field(default_factory=list)

    # event_ids with a blank amount - remain unresolved, NEVER coerced to 0.
    unresolved_blank_ids: list[str] = field(default_factory=list)

    # event_ids excluded for any other reason (cancelled, failed, non_cash /
    # unrealized, pending credit not yet confirmed, unknown status) - kept
    # for reporting/explainability, not for any cash calculation.
    excluded_ids: list[str] = field(default_factory=list)
