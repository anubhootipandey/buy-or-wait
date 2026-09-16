"""
Typed internal models produced by the Stage 3 planner.

Nothing here is written to `output.csv` directly - Stage 4 consumes
`PlanningResult` and formats it into the seven required prediction columns.
Keeping this boundary explicit is what lets Stage 3 stay focused on
*deciding* rather than *formatting*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class SpendingChangeAction:
    """One candidate spending change against a single flexible recurring
    series, identified (per the required `output.csv` format) by the
    `event_id` of that series' most recent settled occurrence - the same
    occurrence `engine.recurrence.RecurringSeries.occurrences[-1]` uses as
    its projection template.
    """

    kind: str  # "stop" | "reduce_to"
    event_id: str
    category: str
    series_key: tuple  # (category, event_type, direction) - matches CashEvent.series_key
    original_amount: Decimal
    adjusted_amount: Decimal  # 0 for "stop"
    severity: int  # 0 = reduce (gentler), 1 = stop (more disruptive) - search order only
    # Identity of the exact `RecurringSeries` object this action was built
    # from (id(series) - see `CashEvent.series_instance_id`). Lets
    # `apply_changes_to_forecast` target exactly the series the action was
    # generated for, even when another series shares the same series_key
    # (an alternating-series split, Rule 9). None only for actions built
    # directly outside `eligible_series_actions` (e.g. hand-written tests).
    series_instance_id: Optional[int] = None

    @property
    def reduction_amount(self) -> Decimal:
        return self.original_amount - self.adjusted_amount


@dataclass(frozen=True)
class PlannedPayment:
    date: date
    amount: Decimal


@dataclass
class PlanCandidate:
    """One fully-specified, independently-safe candidate plan.

    `affordability_status` and `method` are already resolved here so the
    ranking step only has to compare candidates, never re-derive meaning
    from them.
    """

    method: str  # full_payment | partial_payment | installments | wait
    affordability_status: str
    payments: list[PlannedPayment]
    spending_changes: list[SpendingChangeAction] = field(default_factory=list)
    payment_option_id: str | None = None

    @property
    def total_amount_paid(self) -> Decimal:
        return sum((p.amount for p in self.payments), Decimal("0"))

    @property
    def uses_spending_changes(self) -> bool:
        return len(self.spending_changes) > 0

    @property
    def starts_on(self) -> date:
        return self.payments[0].date

    @property
    def completes_on(self) -> date:
        return self.payments[-1].date

    @property
    def num_payments(self) -> int:
        return len(self.payments)


@dataclass
class PlanningResult:
    """The Stage 3 output for one request - everything Stage 4 needs to
    render the seven `output.csv` prediction columns, without Stage 4
    having to re-derive any financial decision itself."""

    request_id: str
    user_id: str
    requested_amount: Decimal
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    chosen_plan: PlanCandidate | None
    earliest_date_for_full_payment: date | None
    spending_changes: list[SpendingChangeAction]
