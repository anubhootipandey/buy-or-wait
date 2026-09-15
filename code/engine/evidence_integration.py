"""
Stage 4c: apply Stage 4a/4b's validated `NormalizedFact` evidence to
Stage 2's already-detected recurring series, deterministically, in
Python. Gemini never runs here and is never imported here - this module
only consumes facts that have already been resolved and validated.

This module is additive: `engine/forecast.py` calls it only when the
caller passes a non-empty `evidence_facts` tuple to `build_forecast`. It
never changes `recurrence.py`'s cadence-detection/projection algorithm,
and it never changes which events land in `settled_history` or
`known_future` - it only adjusts the *generated* (projected) occurrences
of a series that evidence has safely identified.

Series identity (the core safety rule)
---------------------------------------
A fact's `target_event_id` (when present) is the preferred identity: it
points at one real, concrete event that belongs to exactly one detected
`RecurringSeries` (via that series' own `occurrences`), so it survives
Stage 2's alternating-series split (Rule 9) even when two split halves
share an identical `(category, event_type, direction)` key.

A fact with only `target_series_key` (a bare `(category, event_type,
direction)` triple - the standalone-message case) is applied ONLY when
that triple identifies exactly one `RecurringSeries` for the user. If it
matches more than one (the split-series case), or none, the fact is left
unresolved here - never guessed.

Conflict order (per-series, once a fact is safely matched)
------------------------------------------------------------
1. `series_terminated` (an explicit cancellation) always beats a
   `future_amount_change` for any date on/after the termination's
   `effective_date` - a cancelled series does not keep "changing amount"
   after it has ended.
2. Among multiple `future_amount_change` facts whose active window
   ([effective_date, end_date or open-ended]) covers the same date,
   the one with the latest `effective_date` wins (most specific/most
   recent instruction); ties break on newer `provenance.sent_at`
   (newer same-source evidence), then on higher `confidence`
   ("high" > "medium" > "low", i.e. settled-sounding evidence over an
   estimate), then - if still tied - on the financially safer amount
   (for a debit series, the larger amount; for a credit series, the
   smaller amount), so an unresolved tie never overstates money
   available to the user.
3. Among multiple `series_terminated` facts for the same series, the
   earliest `effective_date` wins (stopping sooner is the safer,
   more conservative reading of "this series ended").

`end_date` on a `future_amount_change`
----------------------------------------
ASSUMPTION (documented in DECISIONS.md, tested in
`test_evidence_integration.py`): a `future_amount_change` with a
non-null `end_date` describes a *temporary* change - once a date passes
`end_date`, and no other active change covers it, the occurrence resumes
the series' own original projected amount. This directly matches the
`NormalizedFact` docstring's field list for `future_amount_change`
(`effective_date`, optional `end_date`) and Stage 4b's schema, where
`end_date` has no meaning other than bounding a change's own window.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from .recurrence import RecurringSeries
from .state import CashEvent

try:  # pragma: no cover - import path depends on caller's sys.path setup
    from evidence.models import NormalizedFact
except ImportError:  # pragma: no cover - keeps this module importable in isolation
    NormalizedFact = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class AmountChange:
    effective_date: date
    new_amount: Decimal
    end_date: date | None
    fact_id: str
    sent_at: object  # datetime | None, used only for conflict tie-breaks
    confidence: str | None


@dataclass(frozen=True)
class SeriesAmendment:
    """The fully-resolved (conflict-free) instruction set for one matched
    `RecurringSeries`: zero or more time-bounded amount changes, plus an
    optional date from which no further occurrences should be generated."""

    changes: tuple[AmountChange, ...] = ()
    terminated_from: date | None = None
    applied_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppliedEvidenceRecord:
    fact_id: str
    fact_type: str
    series_category: str
    series_event_type: str
    series_direction: str
    effective_date: date | None
    detail: str


@dataclass(frozen=True)
class UnresolvedEvidenceRecord:
    fact_id: str
    fact_type: str
    reason: str


@dataclass(frozen=True)
class EvidenceApplicationReport:
    """Summary of what Stage 4c did with the evidence it was given, for
    honest reporting (never silently applied, never silently dropped)."""

    applied: tuple[AppliedEvidenceRecord, ...] = field(default_factory=tuple)
    unresolved: tuple[UnresolvedEvidenceRecord, ...] = field(default_factory=tuple)
    occurrences_amount_changed: int = 0
    occurrences_dropped_by_termination: int = 0


_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0, None: -1}


def _series_key_of(series: RecurringSeries) -> tuple[str, str, str]:
    return (series.key.category, series.key.event_type, series.key.direction)


def match_series_for_fact(
    fact: "NormalizedFact", series_list: list[RecurringSeries]
) -> tuple[RecurringSeries | None, str | None]:
    """Resolve a fact's target series using the strongest identity
    available. Returns `(series, None)` on a safe unambiguous match, or
    `(None, reason)` when the fact must be left unresolved rather than
    guessed."""

    if fact.target_event_id is not None:
        matches = [
            s
            for s in series_list
            if any(o.event_id == fact.target_event_id for o in s.occurrences)
        ]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) == 0:
            return None, (
                f"target_event_id {fact.target_event_id!r} is not an occurrence of "
                "any detected recurring series for this user"
            )
        return None, (
            f"target_event_id {fact.target_event_id!r} matched more than one "
            "recurring series - refusing to guess"
        )

    if fact.target_series_key is not None:
        key = tuple(fact.target_series_key)
        matches = [s for s in series_list if _series_key_of(s) == key]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) == 0:
            return None, f"no recurring series with key {key!r} for this user"
        return None, (
            f"key {key!r} matches {len(matches)} recurring series for this user "
            "(e.g. an alternating-split pair) - a bare (category, event_type, "
            "direction) match is not a safe identity here; refusing to guess"
        )

    return None, "fact has neither target_event_id nor target_series_key"


def _is_safer_amount(candidate: Decimal, incumbent: Decimal, direction: str) -> bool:
    """True if `candidate` is the more financially conservative choice than
    `incumbent` for a series with this `direction` - larger debit (spend
    more than expected, never less) or smaller credit (expect less income
    than claimed, never more)."""
    if direction == "debit":
        return candidate > incumbent
    return candidate < incumbent


def _pick_winning_change(
    candidates: list["NormalizedFact"], direction: str
) -> "NormalizedFact":
    """Tie-break order: latest effective_date, then newest sent_at, then
    highest confidence, then the financially safer amount."""

    def sent_at_key(f: "NormalizedFact"):
        sent_at = f.provenance.sent_at
        return sent_at if sent_at is not None else _MIN_DATETIME

    best = candidates[0]
    for f in candidates[1:]:
        if f.effective_date != best.effective_date:
            if f.effective_date > best.effective_date:
                best = f
            continue
        if sent_at_key(f) != sent_at_key(best):
            if sent_at_key(f) > sent_at_key(best):
                best = f
            continue
        rank_f = _CONFIDENCE_RANK.get(f.confidence, -1)
        rank_best = _CONFIDENCE_RANK.get(best.confidence, -1)
        if rank_f != rank_best:
            if rank_f > rank_best:
                best = f
            continue
        if f.resolved_amount != best.resolved_amount and _is_safer_amount(
            f.resolved_amount, best.resolved_amount, direction
        ):
            best = f
    return best


_MIN_DATETIME = datetime.min


def build_amendment(
    series: RecurringSeries, facts: list["NormalizedFact"]
) -> SeriesAmendment:
    """Combine every fact already matched to one series into a single,
    conflict-free `SeriesAmendment`."""

    terminations = [f for f in facts if f.fact_type == "series_terminated"]
    changes_facts = [f for f in facts if f.fact_type == "future_amount_change"]

    applied_ids: list[str] = []

    terminated_from: date | None = None
    if terminations:
        # Rule: among multiple terminations, the earliest effective_date is
        # the safer (stop-sooner) reading.
        winner = min(terminations, key=lambda f: f.effective_date)
        terminated_from = winner.effective_date
        applied_ids.append(winner.fact_id)

    # Group amount-change facts by effective_date so exact-date conflicts
    # (two different facts both claiming the same effective_date) are
    # resolved to a single winner; distinct effective_dates are each kept
    # as their own step in the amount's timeline (not a conflict).
    by_effective_date: dict[date, list["NormalizedFact"]] = {}
    for f in changes_facts:
        by_effective_date.setdefault(f.effective_date, []).append(f)

    changes: list[AmountChange] = []
    for eff_date, group in by_effective_date.items():
        winner = group[0] if len(group) == 1 else _pick_winning_change(group, series.key.direction)
        applied_ids.append(winner.fact_id)
        changes.append(
            AmountChange(
                effective_date=winner.effective_date,
                new_amount=winner.resolved_amount,
                end_date=winner.end_date,
                fact_id=winner.fact_id,
                sent_at=winner.provenance.sent_at,
                confidence=winner.confidence,
            )
        )

    if terminated_from is not None:
        # An explicit cancellation beats a weaker future estimate: no
        # amount change takes effect on/after the series' own end.
        changes = [c for c in changes if c.effective_date < terminated_from]

    changes.sort(key=lambda c: c.effective_date)
    return SeriesAmendment(
        changes=tuple(changes), terminated_from=terminated_from, applied_fact_ids=tuple(applied_ids)
    )


def _amount_for_date(original_amount: Decimal, changes: tuple[AmountChange, ...], d: date) -> Decimal:
    active: AmountChange | None = None
    for c in changes:  # already sorted ascending by effective_date
        if c.effective_date <= d and (c.end_date is None or d <= c.end_date):
            active = c  # a later-effective, still-active change overwrites an earlier one
    return active.new_amount if active is not None else original_amount


def apply_evidence_to_series(
    series_list: list[RecurringSeries],
    per_series_generated: dict[int, list[CashEvent]],
    evidence_facts: tuple["NormalizedFact", ...],
) -> tuple[dict[int, list[CashEvent]], EvidenceApplicationReport]:
    """Adjust each series' already-generated occurrences (produced by the
    untouched `project_series`) according to any `future_amount_change` /
    `series_terminated` facts that safely match it.

    `per_series_generated` is keyed by `id(series)` so that two Stage-2
    split series sharing an identical `(category, event_type, direction)`
    key are never confused with each other - each dict entry is exactly
    the occurrences `project_series` produced for that one series object.

    Returns a new dict (input is never mutated) plus a full report of what
    was applied and what was left unresolved.
    """

    matched: dict[int, list["NormalizedFact"]] = {}
    unresolved: list[UnresolvedEvidenceRecord] = []

    for fact in evidence_facts:
        if fact.fact_type not in ("future_amount_change", "series_terminated"):
            # Stage 4c only acts on these two fact types; anything else
            # (resolved_amount, status_correction, no_fact) is out of its
            # scope and is left untouched here.
            continue
        series, reason = match_series_for_fact(fact, series_list)
        if series is None:
            unresolved.append(
                UnresolvedEvidenceRecord(fact_id=fact.fact_id, fact_type=fact.fact_type, reason=reason or "unresolved")
            )
            continue
        matched.setdefault(id(series), []).append(fact)

    applied: list[AppliedEvidenceRecord] = []
    amount_changed_count = 0
    dropped_count = 0

    result: dict[int, list[CashEvent]] = {}
    for series in series_list:
        sid = id(series)
        occurrences = per_series_generated.get(sid, [])
        facts_for_series = matched.get(sid)
        if not facts_for_series:
            result[sid] = occurrences
            continue

        amendment = build_amendment(series, facts_for_series)
        skey = _series_key_of(series)

        new_occurrences: list[CashEvent] = []
        for e in occurrences:
            if amendment.terminated_from is not None and e.date >= amendment.terminated_from:
                dropped_count += 1
                continue
            new_amount = _amount_for_date(e.amount, amendment.changes, e.date)
            if new_amount != e.amount:
                amount_changed_count += 1
                e = dataclasses.replace(e, amount=new_amount)
            new_occurrences.append(e)
        result[sid] = new_occurrences

        for fact in facts_for_series:
            applied.append(
                AppliedEvidenceRecord(
                    fact_id=fact.fact_id,
                    fact_type=fact.fact_type,
                    series_category=skey[0],
                    series_event_type=skey[1],
                    series_direction=skey[2],
                    effective_date=fact.effective_date,
                    detail=(
                        "applied"
                        if fact.fact_id in amendment.applied_fact_ids
                        else "superseded by a higher-priority fact for the same series/date"
                    ),
                )
            )

    report = EvidenceApplicationReport(
        applied=tuple(applied),
        unresolved=tuple(unresolved),
        occurrences_amount_changed=amount_changed_count,
        occurrences_dropped_by_termination=dropped_count,
    )
    return result, report


__all__ = [
    "AmountChange",
    "SeriesAmendment",
    "AppliedEvidenceRecord",
    "UnresolvedEvidenceRecord",
    "EvidenceApplicationReport",
    "match_series_for_fact",
    "build_amendment",
    "apply_evidence_to_series",
]