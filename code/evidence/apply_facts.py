"""
Applies validated `resolved_amount` facts onto the dataset's events.

This is the deterministic bridge between evidence resolution and the
financial engine, and it is pure Python - no model, no heuristics, no
inference. Stage 4c (`engine/evidence_integration.py`) deliberately
handles only series-level facts (`future_amount_change`,
`series_terminated`), because those amend a *forecast*. A
`resolved_amount` fact is different in kind: it fills in a blank on an
event record that already exists, so it must be applied to the dataset
BEFORE reconciliation, recurrence detection, and forecasting ever see it.

Every rule here fails closed. An event is only ever filled in when:

  * the fact is a `resolved_amount` carrying an actual amount,
  * it names an event that exists,
  * that event's amount is genuinely blank (a known amount is NEVER
    overwritten by evidence - the structured record wins),
  * the fact's currency matches the event's own currency, and
  * no other fact disagrees about the same event.

Anything else is skipped and recorded in the report, leaving the amount
blank. A blank amount is handled safely downstream; a wrong amount would
quietly corrupt a real forecast, so "skip" is always the safer failure.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Optional

from data.models import Dataset

from .models import NormalizedFact


@dataclass(frozen=True)
class AppliedAmount:
    event_id: str
    amount: Decimal
    currency: str
    fact_id: str
    image_id: Optional[str] = None
    message_id: Optional[str] = None
    resolution_method: str = "deterministic"
    confidence: Optional[str] = None


@dataclass(frozen=True)
class SkippedAmount:
    fact_id: str
    event_id: Optional[str]
    reason: str


@dataclass(frozen=True)
class ApplyAmountsReport:
    applied: tuple[AppliedAmount, ...]
    skipped: tuple[SkippedAmount, ...]

    @property
    def applied_event_ids(self) -> frozenset[str]:
        return frozenset(a.event_id for a in self.applied)

    def summary(self) -> str:
        return f"{len(self.applied)} applied, {len(self.skipped)} skipped"


def apply_resolved_amounts(
    dataset: Dataset, facts: Iterable[NormalizedFact]
) -> tuple[Dataset, ApplyAmountsReport]:
    """Return a new `Dataset` with blank event amounts filled in from
    `facts`, plus a report of exactly what was and was not applied.

    `dataset` is never mutated - events are rebuilt with
    `dataclasses.replace`, matching how the rest of the pipeline treats
    dataset records as immutable.
    """
    events_by_id = {e.event_id: e for e in dataset.events}

    candidates: dict[str, list[NormalizedFact]] = {}
    skipped: list[SkippedAmount] = []

    for fact in facts:
        if fact.fact_type != "resolved_amount":
            continue
        if fact.resolved_amount is None:
            skipped.append(
                SkippedAmount(fact.fact_id, fact.target_event_id, "resolved_amount fact carries no amount")
            )
            continue
        if fact.target_event_id is None:
            skipped.append(SkippedAmount(fact.fact_id, None, "fact names no target_event_id"))
            continue
        event = events_by_id.get(fact.target_event_id)
        if event is None:
            skipped.append(
                SkippedAmount(fact.fact_id, fact.target_event_id, "target_event_id not present in dataset")
            )
            continue
        if event.amount is not None:
            skipped.append(
                SkippedAmount(
                    fact.fact_id,
                    fact.target_event_id,
                    "event already has a known amount; structured data wins over evidence",
                )
            )
            continue
        if fact.currency is not None and fact.currency != event.currency:
            skipped.append(
                SkippedAmount(
                    fact.fact_id,
                    fact.target_event_id,
                    f"fact currency {fact.currency!r} contradicts event currency {event.currency!r}",
                )
            )
            continue
        if fact.resolved_amount <= 0:
            skipped.append(
                SkippedAmount(fact.fact_id, fact.target_event_id, "resolved amount is not positive")
            )
            continue
        candidates.setdefault(fact.target_event_id, []).append(fact)

    resolved: dict[str, NormalizedFact] = {}
    for event_id, event_facts in candidates.items():
        amounts = {f.resolved_amount for f in event_facts}
        if len(amounts) > 1:
            # Two pieces of evidence disagree about the same blank event.
            # There is no safe deterministic tiebreak here, and guessing
            # would defeat the point of validating at all - leave blank.
            for f in event_facts:
                skipped.append(
                    SkippedAmount(
                        f.fact_id,
                        event_id,
                        "conflicting resolved_amount facts for this event; left unresolved",
                    )
                )
            continue
        # Identical amounts from multiple sources: deterministic pick by
        # fact_id so the result never depends on iteration order.
        resolved[event_id] = sorted(event_facts, key=lambda f: f.fact_id)[0]

    applied: list[AppliedAmount] = []
    new_events = []
    for event in dataset.events:
        fact = resolved.get(event.event_id)
        if fact is None:
            new_events.append(event)
            continue
        new_events.append(dataclasses.replace(event, amount=fact.resolved_amount))
        applied.append(
            AppliedAmount(
                event_id=event.event_id,
                amount=fact.resolved_amount,
                currency=event.currency,
                fact_id=fact.fact_id,
                image_id=fact.provenance.image_id,
                message_id=fact.provenance.message_id,
                resolution_method=fact.resolution_method,
                confidence=fact.confidence,
            )
        )

    report = ApplyAmountsReport(
        applied=tuple(sorted(applied, key=lambda a: a.event_id)),
        skipped=tuple(sorted(skipped, key=lambda s: (s.fact_id, s.reason))),
    )
    return dataclasses.replace(dataset, events=new_events), report


__all__ = [
    "apply_resolved_amounts",
    "ApplyAmountsReport",
    "AppliedAmount",
    "SkippedAmount",
]
