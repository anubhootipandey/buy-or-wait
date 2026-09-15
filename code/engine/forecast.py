"""
Stage 2 forecast: anchor at `request_date`, apply the starting balance
(Rule 1: `current_available_balance` is the balance BEFORE any event on
request_date), then walk chronologically through 90 days of known future
events and projected recurring occurrences, checking the minimum-balance
safety condition at every checkpoint.

Rule 7 (known future events override generated recurrence) is enforced
here: for every generated occurrence of a detected series, if a known
future event of the same (category, event_type, direction) falls within
that series' cadence tolerance of the generated date, the generated
occurrence is dropped - the known event (already in `known_future`) is
what actually appears in the ledger. This guarantees exactly one cash
movement per expected occurrence, never two.

This module does not decide affordability - it only produces the balance
timeline that a later stage (the planner) will need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from .evidence_integration import EvidenceApplicationReport, apply_evidence_to_series
from .recurrence import OVERRIDE_TOLERANCE, RecurringSeries, detect_recurring_series, project_series
from .state import CashEvent, ReconciledState

if TYPE_CHECKING:  # pragma: no cover - type-checking only, no runtime import cycle
    from evidence.models import NormalizedFact

FORECAST_DAYS = 90


@dataclass(frozen=True)
class ForecastCheckpoint:
    date: date
    event: CashEvent
    balance_after: Decimal


@dataclass
class ForecastResult:
    user_id: str
    request_date: date
    starting_balance: Decimal
    minimum_balance_to_keep: Decimal
    checkpoints: list[ForecastCheckpoint] = field(default_factory=list)
    recurring_series: list[RecurringSeries] = field(default_factory=list)
    overridden_generated_count: int = 0
    # Stage 4c only: set when `build_forecast` was called with non-empty
    # `evidence_facts`; `None` whenever evidence integration wasn't used
    # (including every pre-Stage-4c call site), so nothing downstream can
    # mistake "no evidence was supplied" for "evidence found nothing to do".
    evidence_report: EvidenceApplicationReport | None = None

    @property
    def min_balance_reached(self) -> Decimal:
        if not self.checkpoints:
            return self.starting_balance
        return min(cp.balance_after for cp in self.checkpoints)

    def balance_on(self, as_of: date) -> Decimal:
        """Balance at the end of `as_of`, after applying every event with
        date <= as_of (and none after)."""
        balance = self.starting_balance
        for cp in self.checkpoints:
            if cp.date <= as_of:
                balance = cp.balance_after
            else:
                break
        return balance

    def breaches_minimum(self) -> bool:
        return any(cp.balance_after < self.minimum_balance_to_keep for cp in self.checkpoints)


def build_forecast(
    state: ReconciledState,
    starting_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    forecast_days: int = FORECAST_DAYS,
    evidence_facts: tuple["NormalizedFact", ...] = (),
) -> ForecastResult:
    """Build the 90-day deterministic forecast.

    `evidence_facts` is Stage 4c only, additive, and defaults to empty: it
    is `()` at every pre-Stage-4c call site (`main.py`,
    `planner/spending_changes.py`, and every existing test), so passing
    nothing here reproduces Stage 2's forecast exactly as before - no
    behavior change unless a caller explicitly opts in. Only
    `future_amount_change` / `series_terminated` facts that safely match
    exactly one detected `RecurringSeries` (see `evidence_integration.py`)
    are ever applied; Gemini is never called here and no fact is ever
    guessed onto a series.
    """
    window_start = state.request_date
    window_end = state.request_date + timedelta(days=forecast_days)

    series_list = detect_recurring_series(state.settled_history, state.user_id)
    cadence_by_key = {
        (s.key.category, s.key.event_type, s.key.direction): s.cadence for s in series_list
    }

    # Generate each series' occurrences via the untouched Stage 2
    # `project_series`, keyed by the series *object* identity (not its
    # (category, event_type, direction) key, which two Rule-9 split
    # halves can share) so Stage 4c can target one split half without
    # ever confusing it with the other.
    per_series_generated: dict[int, list[CashEvent]] = {
        id(series): project_series(series, window_start, window_end) for series in series_list
    }

    evidence_report: EvidenceApplicationReport | None = None
    if evidence_facts:
        per_series_generated, evidence_report = apply_evidence_to_series(
            series_list, per_series_generated, evidence_facts
        )

    generated: list[CashEvent] = []
    for series in series_list:
        generated.extend(per_series_generated[id(series)])

    known_by_key: dict[tuple, list[CashEvent]] = {}
    for k in state.known_future:
        skey = (k.category, k.event_type, k.direction)
        known_by_key.setdefault(skey, []).append(k)

    kept_generated: list[CashEvent] = []
    overridden = 0
    for g in generated:
        skey = g.series_key
        tol = OVERRIDE_TOLERANCE.get(cadence_by_key.get(skey), 3)
        candidates = known_by_key.get(skey, ())
        matched = any(abs((k.date - g.date).days) <= tol for k in candidates)
        if matched:
            overridden += 1
            continue
        kept_generated.append(g)

    ledger: list[CashEvent] = []
    for e in state.settled_history:
        if window_start <= e.date <= window_end:
            ledger.append(e)
    for k in state.known_future:
        if window_start <= k.date <= window_end:
            ledger.append(k)
    ledger.extend(kept_generated)

    # Chronological order (Rule 9: process events chronologically). Ties on
    # the same date are broken with settled-before-projected, a deterministic
    # and conservative default (real, already-known movements first).
    ledger.sort(key=lambda e: (e.date, 0 if e.source != "recurring" else 1, e.event_id or ""))

    balance = starting_balance
    checkpoints: list[ForecastCheckpoint] = []
    for e in ledger:
        balance = balance + e.signed_amount
        checkpoints.append(ForecastCheckpoint(date=e.date, event=e, balance_after=balance))

    return ForecastResult(
        user_id=state.user_id,
        request_date=state.request_date,
        starting_balance=starting_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        checkpoints=checkpoints,
        recurring_series=series_list,
        overridden_generated_count=overridden,
        evidence_report=evidence_report,
    )