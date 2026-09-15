"""
Stage 4a: deterministic evidence pre-filter.

For every message and image, decide - using ONLY already-known structured
data (event status, amount presence, currency) plus a narrow, literal
number-format extraction - which of five buckets it falls into:

  * `no_fact`            - the event it's about is already closed
                            (settled/cancelled/failed) with a known
                            amount, so nothing about it can still change
                            what Stage 2/3 need. Historical settled facts
                            are not retroactively amendable, and
                            cancelled/failed events are already excluded
                            from cash flow - a message about either can
                            only be corroboration, never new information.
                            EXCEPTION: a `failed` event whose dataset
                            record is linked back to by another event
                            (via that other event's `linked_event_id`)
                            that is still `pending`/`scheduled` has a
                            live future cash-flow effect (a queued retry)
                            - that case escalates instead, see
                            `_find_live_forward_link`.
  * `deterministic_fact` - a blank-amount event's message contains
                            EXACTLY ONE `<CCY> <number>` pattern anywhere
                            in the text, using the event's own known
                            currency. More than one match anywhere in the
                            text makes it ambiguous and this bucket is
                            never used - it only fires when there is
                            genuinely nothing to guess between.
  * `needs_escalation`   - everything that requires reading the evidence
                            for meaning: a blank amount with no safe
                            single-number match, an image (Stage 4a never
                            attempts OCR), a standalone message with no
                            related_event_id, or a message about an event
                            still in a "live" status (pending/scheduled)
                            that could plausibly be amended or cancelled.
                            Actual escalation (Gemini) is Stage 4c/4d -
                            not implemented here.
  * `missing_evidence`   - the evidence cannot be resolved by any means:
                            an image file absent from disk, or a
                            `related_event_id` that doesn't exist in
                            `financial_events.csv`.
  * `unresolved_other`   - defensive catch-all for a shape this module
                            doesn't otherwise expect (e.g. an image with
                            no `related_event_id` at all).

This module NEVER calls a model, NEVER invents a number, and NEVER
classifies based on sentiment/urgency/scam keyword matching - only on
already-known CSV fields and the single narrow numeric extraction above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from data.models import Dataset, FinancialEvent, ImageRecord, Message

from .models import EvidenceRef, NormalizedFact

NO_FACT = "no_fact"
DETERMINISTIC_FACT = "deterministic_fact"
NEEDS_ESCALATION = "needs_escalation"
MISSING_EVIDENCE = "missing_evidence"
UNRESOLVED_OTHER = "unresolved_other"

BUCKETS = (NO_FACT, DETERMINISTIC_FACT, NEEDS_ESCALATION, MISSING_EVIDENCE, UNRESOLVED_OTHER)

# Statuses whose cash-flow treatment Stage 2 has already fully and
# permanently decided - see module docstring for why these are safe to
# call "no_fact" without reading the evidence text at all.
_CLOSED_STATUSES = frozenset({"settled", "cancelled", "failed"})

# Statuses Stage 2 still treats as "live" - i.e. still capable of
# affecting a future cash forecast (see engine/reconciliation.py).
_LIVE_STATUSES = frozenset({"pending", "scheduled"})

# A literal `<3-letter currency code>` optionally followed by a space,
# then a number (digits with optional thousands separators/decimal
# point). Deliberately simple and literal - this is a format match, not
# an attempt to parse natural language.
_CURRENCY_AMOUNT_RE = re.compile(r"\b([A-Z]{3})\s?([0-9][0-9.,]*)\b")


def _find_live_forward_link(
    event: FinancialEvent, events_by_id: dict[str, FinancialEvent]
) -> Optional[FinancialEvent]:
    """If `event` is `failed`, look for another event in the dataset that
    points BACK at it via `linked_event_id` (the dataset's convention: a
    retry carries `linked_event_id` = the failed attempt it retries, not
    the other way around) and whose OWN status is still `_LIVE_STATUSES`
    (pending/scheduled).

    A failed payment with such a linked retry still has a real, live
    future cash-flow effect (Stage 2 already puts that retry event into
    `known_future` on its own merits) even though the failed event itself
    is correctly excluded. Returns that live linked event, or `None` if
    there isn't one.

    This is a purely structural check - which event's `linked_event_id`
    points where, and what status that event currently has - never a
    reading of message text, so it stays fully deterministic.
    """
    if event.status != "failed":
        return None
    for candidate in events_by_id.values():
        if candidate.linked_event_id == event.event_id and candidate.status in _LIVE_STATUSES:
            return candidate
    return None


@dataclass(frozen=True)
class PrefilterDecision:
    """One evidence item's deterministic pre-filter classification."""

    ref: EvidenceRef
    bucket: str
    reason: str
    fact: Optional[NormalizedFact] = None

    def __post_init__(self) -> None:
        if self.bucket not in BUCKETS:
            raise ValueError(f"unknown bucket: {self.bucket!r}")
        if self.fact is not None and self.bucket not in (NO_FACT, DETERMINISTIC_FACT):
            raise ValueError(f"a fact was attached to a non-fact bucket: {self.bucket!r}")


def _extract_single_unambiguous_amount(text: str, expected_currency: str) -> Optional[Decimal]:
    """A `Decimal` only if `text` contains EXACTLY ONE currency-amount
    pattern anywhere, using `expected_currency`. Any second match anywhere
    (even a different currency, even the same amount repeated) makes the
    result ambiguous, so this returns `None` - it never guesses which
    number is the right one."""
    matches = _CURRENCY_AMOUNT_RE.findall(text)
    if len(matches) != 1:
        return None
    currency, raw_amount = matches[0]
    if currency != expected_currency:
        return None
    try:
        return Decimal(raw_amount.replace(",", ""))
    except InvalidOperation:
        return None


def classify_message(
    message: Message, events_by_id: dict[str, FinancialEvent]
) -> PrefilterDecision:
    ref = EvidenceRef(
        user_id=message.user_id,
        message_id=message.message_id,
        request_id=message.request_id,
        related_event_id=message.related_event_id,
        sent_at=message.sent_at,
    )

    if message.related_event_id is None:
        # Standalone/account-level notice (e.g. a general payroll change
        # notice with no request_id/related_event_id): there is no single
        # structured record to check deterministically against, so only
        # semantic interpretation can decide whether it describes
        # something Stage 2 needs to know.
        return PrefilterDecision(ref, NEEDS_ESCALATION, "standalone message, no related_event_id")

    event = events_by_id.get(message.related_event_id)
    if event is None:
        return PrefilterDecision(
            ref, MISSING_EVIDENCE, "related_event_id does not exist in financial_events.csv"
        )

    if event.amount is not None and event.status in _CLOSED_STATUSES:
        live_retry = _find_live_forward_link(event, events_by_id)
        if live_retry is not None:
            return PrefilterDecision(
                ref, NEEDS_ESCALATION,
                f"related event is status='failed' but {live_retry.event_id!r} links back "
                f"to it and is still status={live_retry.status!r}; the failed event has a "
                "live future cash-flow effect, so this is not inert corroboration",
            )
        fact = NormalizedFact(
            fact_id=f"fact:{message.message_id}:no_fact",
            user_id=message.user_id,
            fact_type="no_fact",
            provenance=ref,
            target_event_id=event.event_id,
        )
        return PrefilterDecision(
            ref, NO_FACT,
            f"related event is already status={event.status!r} with a known amount; "
            "closed facts are not retroactively amendable",
            fact,
        )

    if event.amount is None:
        extracted = _extract_single_unambiguous_amount(message.message_text, event.currency)
        if extracted is not None:
            fact = NormalizedFact(
                fact_id=f"fact:{message.message_id}:resolved_amount",
                user_id=message.user_id,
                fact_type="resolved_amount",
                provenance=ref,
                target_event_id=event.event_id,
                resolved_amount=extracted,
                currency=event.currency,
                resolution_method="deterministic",
                confidence="high",
            )
            return PrefilterDecision(
                ref, DETERMINISTIC_FACT,
                "single unambiguous currency-amount match in message text",
                fact,
            )
        return PrefilterDecision(
            ref, NEEDS_ESCALATION,
            "blank-amount event; message text has no single unambiguous amount match",
        )

    # amount present but status is "live" (pending/scheduled) - a message
    # could plausibly amend, confirm, or cancel it; only semantic
    # interpretation can decide, so this always escalates.
    return PrefilterDecision(
        ref, NEEDS_ESCALATION,
        f"related event status={event.status!r} is not closed; message may amend it",
    )


def classify_image(
    image: ImageRecord, events_by_id: dict[str, FinancialEvent]
) -> PrefilterDecision:
    ref = EvidenceRef(
        user_id=image.user_id,
        image_id=image.image_id,
        request_id=image.request_id,
        related_event_id=image.related_event_id,
    )

    if not Path(image.file_path).is_file():
        return PrefilterDecision(ref, MISSING_EVIDENCE, f"image file not found on disk: {image.file_path}")

    if image.related_event_id is None:
        return PrefilterDecision(ref, UNRESOLVED_OTHER, "image has no related_event_id to ground against")

    event = events_by_id.get(image.related_event_id)
    if event is None:
        return PrefilterDecision(
            ref, MISSING_EVIDENCE, "related_event_id does not exist in financial_events.csv"
        )

    if event.amount is not None:
        # Defensive - not expected in the current dataset (every image is
        # tied to a blank-amount event), but handled the same way a
        # closed-status message is: nothing left for Stage 2/3 to need,
        # UNLESS this is a failed event with a live linked retry (see
        # `_find_live_forward_link`) - same exception as classify_message.
        live_retry = _find_live_forward_link(event, events_by_id)
        if live_retry is not None:
            return PrefilterDecision(
                ref, NEEDS_ESCALATION,
                f"related event is status='failed' but {live_retry.event_id!r} links back "
                f"to it and is still status={live_retry.status!r}; the failed event has a "
                "live future cash-flow effect, so this is not inert corroboration",
            )
        fact = NormalizedFact(
            fact_id=f"fact:{image.image_id}:no_fact",
            user_id=image.user_id,
            fact_type="no_fact",
            provenance=ref,
            target_event_id=event.event_id,
        )
        return PrefilterDecision(ref, NO_FACT, "related event already has a known amount", fact)

    # Blank amount, image present on disk - genuinely requires visual
    # interpretation. Stage 4a never attempts OCR/vision.
    return PrefilterDecision(
        ref, NEEDS_ESCALATION, "blank-amount event; requires image interpretation (Stage 4d)"
    )


@dataclass(frozen=True)
class PrefilterReport:
    decisions: tuple[PrefilterDecision, ...]

    def by_bucket(self, bucket: str) -> tuple[PrefilterDecision, ...]:
        if bucket not in BUCKETS:
            raise ValueError(f"unknown bucket: {bucket!r}")
        return tuple(d for d in self.decisions if d.bucket == bucket)

    def counts(self) -> dict[str, int]:
        return {bucket: len(self.by_bucket(bucket)) for bucket in BUCKETS}


def run_prefilter(dataset: Dataset, events_by_id: dict[str, FinancialEvent]) -> PrefilterReport:
    """Classify every message and image in `dataset` deterministically.
    Takes `events_by_id` directly (rather than a full `Indexes`) so this
    module has no dependency beyond `data.models` + the standard library."""
    decisions = [classify_message(m, events_by_id) for m in dataset.messages]
    decisions += [classify_image(img, events_by_id) for img in dataset.images]
    return PrefilterReport(tuple(decisions))