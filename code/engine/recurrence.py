"""
Stage 2 recurrence detection.

Groups a user's settled history by (category, event_type, direction) and
decides whether it is a recurring series, following the approved rule
order exactly:

  1. duplicates are already excluded upstream (reconciliation.py)
  2. only settled history is used to establish cadence
  3. group by (user_id, category, event_type, direction)
  4. monthly-scale tolerance: +/-3 days (nominal 30-day gap)
  5. weekly-scale tolerance: +/-2 days (nominal 7-day gap)
  6. biweekly-scale tolerance: +/-4 days (nominal 14-day gap)
  7. normally require >= 3 occurrences
  8. allow 2 only with strict monthly evidence: same calendar day within
     2 days AND gap 28-31 days
  9. alternating-series splitting only when BOTH resulting sub-series
     independently satisfy the recurrence rule (interleaved series, e.g.
     two salaries, must stay separate)
  10. trailing pending/scheduled events never establish cadence - guaranteed
      structurally, since they are never present in `settled_history`

One-off events (a group that doesn't qualify under this rule) are never
projected.

Variable essential spending (dining/groceries/transport) projects at the
maximum historical settled amount (Rule 6) - no other estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from .state import CashEvent

VARIABLE_ESSENTIAL_CATEGORIES = frozenset({"dining", "groceries", "transport"})

# name -> (min_gap_days, max_gap_days, nominal_interval_days)
CADENCES: dict[str, tuple[int, int, int]] = {
    "weekly": (5, 9, 7),
    "biweekly": (10, 18, 14),
    "monthly": (27, 33, 30),
}

# Tolerance (days) used when deciding whether a known future event overrides
# a generated occurrence of a series with this cadence (Rule 7).
OVERRIDE_TOLERANCE: dict[str, int] = {"weekly": 2, "biweekly": 4, "monthly": 3}


@dataclass(frozen=True)
class SeriesKey:
    user_id: str
    category: str
    event_type: str
    direction: str


@dataclass
class RecurringSeries:
    key: SeriesKey
    cadence: str  # "weekly" | "biweekly" | "monthly"
    interval_days: int
    occurrences: list[CashEvent]  # the settled occurrences that established this series
    flexibility: str

    @property
    def last_date(self) -> date:
        return self.occurrences[-1].date

    @property
    def projected_amount(self) -> Decimal:
        if self.key.category in VARIABLE_ESSENTIAL_CATEGORIES:
            return max(o.amount for o in self.occurrences)
        return self.occurrences[-1].amount


def _cadence_for_gap(gap_days: int) -> Optional[str]:
    for name, (lo, hi, _nominal) in CADENCES.items():
        if lo <= gap_days <= hi:
            return name
    return None


def _gaps(events_sorted: list[CashEvent]) -> list[int]:
    return [
        (events_sorted[i + 1].date - events_sorted[i].date).days
        for i in range(len(events_sorted) - 1)
    ]


def _is_strict_monthly_pair(a: CashEvent, b: CashEvent) -> bool:
    gap = (b.date - a.date).days
    if not (28 <= gap <= 31):
        return False
    return abs(a.date.day - b.date.day) <= 2


def _add_months(d: date, months: int) -> date:
    """Advance `d` by `months` calendar months, keeping the same day-of-month
    where possible and clamping to the target month's last day otherwise
    (e.g. Jan 31 + 1 month -> Feb 28/29). This is what lets a monthly series
    anchored on a specific day (rent due the 1st, salary the 15th) project
    correctly, instead of drifting under a fixed 30-day step."""
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day_of_month = (next_month_first - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day_of_month))


def _qualifying_cadence(events_sorted: list[CashEvent]) -> Optional[str]:
    """Rule 7/8: >=3 occurrences with consistent single-cadence gaps, or
    exactly 2 occurrences with strict monthly evidence. Anything else does
    not qualify (including a single occurrence: a one-off)."""
    if len(events_sorted) >= 3:
        gaps = _gaps(events_sorted)
        cadences = [_cadence_for_gap(g) for g in gaps]
        if cadences[0] is not None and all(c == cadences[0] for c in cadences):
            return cadences[0]
        return None
    if len(events_sorted) == 2 and _is_strict_monthly_pair(events_sorted[0], events_sorted[1]):
        return "monthly"
    return None


def detect_recurring_series(settled_history: list[CashEvent], user_id: str) -> list[RecurringSeries]:
    """Detect recurring series within one user's settled history.

    Only events already classified as `source == "settled"` should be
    passed in (that filtering happens in `ReconciledState.settled_history`).
    """
    groups: dict[SeriesKey, list[CashEvent]] = {}
    for e in settled_history:
        key = SeriesKey(user_id, e.category, e.event_type, e.direction)
        groups.setdefault(key, []).append(e)

    series_list: list[RecurringSeries] = []
    for key, events in groups.items():
        events_sorted = sorted(events, key=lambda e: e.date)

        cadence = _qualifying_cadence(events_sorted)
        if cadence is not None:
            series_list.append(
                RecurringSeries(
                    key=key,
                    cadence=cadence,
                    interval_days=CADENCES[cadence][2],
                    occurrences=events_sorted,
                    flexibility=events_sorted[-1].flexibility,
                )
            )
            continue

        # Rule 9: alternating-series splitting, only if BOTH halves qualify
        # independently. This is what keeps two interleaved series (e.g.
        # two salaries paid on different cadences) separate rather than
        # being treated as one noisy series or dropped entirely.
        if len(events_sorted) >= 4:
            even = events_sorted[0::2]
            odd = events_sorted[1::2]
            cadence_even = _qualifying_cadence(even)
            cadence_odd = _qualifying_cadence(odd)
            if cadence_even is not None and cadence_odd is not None:
                series_list.append(
                    RecurringSeries(
                        key=key,
                        cadence=cadence_even,
                        interval_days=CADENCES[cadence_even][2],
                        occurrences=even,
                        flexibility=even[-1].flexibility,
                    )
                )
                series_list.append(
                    RecurringSeries(
                        key=key,
                        cadence=cadence_odd,
                        interval_days=CADENCES[cadence_odd][2],
                        occurrences=odd,
                        flexibility=odd[-1].flexibility,
                    )
                )
        # Otherwise: a one-off (or an irregular group) - never projected.

    return series_list


def project_series(series: RecurringSeries, window_start: date, window_end: date) -> list[CashEvent]:
    """Generate future occurrences of an already-qualified `series`, strictly
    after its last known settled occurrence, within [window_start, window_end].

    Never called on a one-off group - only on a `RecurringSeries` that has
    already passed `_qualifying_cadence`.
    """
    out: list[CashEvent] = []
    template = series.occurrences[-1]
    amount = series.projected_amount

    def _step(d: date) -> date:
        if series.cadence == "monthly":
            return _add_months(d, 1)
        return d + timedelta(days=series.interval_days)

    next_date = _step(series.last_date)
    while next_date <= window_end:
        if next_date >= window_start:
            out.append(
                CashEvent(
                    date=next_date,
                    amount=amount,
                    direction=template.direction,
                    category=template.category,
                    event_type=template.event_type,
                    flexibility=template.flexibility,
                    source="recurring",
                    event_id=None,
                    series_key=(series.key.category, series.key.event_type, series.key.direction),
                )
            )
        next_date = _step(next_date)
    return out
