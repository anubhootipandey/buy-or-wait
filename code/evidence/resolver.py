"""
Stage 4b orchestration.

`build_context` grounds one message against already-known structured data
(the exact event it's about, or - for a standalone message - the user's
own known recurring-series categories) without ever handing the model
anything beyond that single user's own records.

`resolve_message_with_ai` implements the retry policy required by the
task: call the model once; if the response is invalid, retry EXACTLY
once with a materially more corrective prompt (see
`prompts.build_retry_prompt`); if still invalid, return `unresolved`.
A rejected/invalid response is NEVER cached and NEVER silently repaired.

`run_stage4b` composes Stage 4a's deterministic prefilter with Stage 4b's
AI resolution for exactly the evidence Stage 4a could not resolve itself:
  * `no_fact` / `deterministic_fact` decisions pass through unchanged.
  * `missing_evidence` / `unresolved_other` decisions pass through as
    `UnresolvedEvidence` unchanged.
  * `needs_escalation` MESSAGES go through the AI resolver.
  * `needs_escalation` IMAGES are left unresolved - Stage 4b is
    text-only; vision is Stage 4d and explicitly out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.models import Dataset, Message
from data.indexes import Indexes

from .ai_client import AIClient
from .cache import EvidenceCache
from .deterministic import DETERMINISTIC_FACT, NEEDS_ESCALATION, NO_FACT, run_prefilter
from .models import EvidenceRef, EvidenceResolutionOutcome, NormalizedFact, UnresolvedEvidence
from .prompts import EvidenceContext, build_prompt, build_retry_prompt
from .validation import validate_response


def build_context(message: Message, indexes: Indexes) -> EvidenceContext:
    event = indexes.events_by_id.get(message.related_event_id) if message.related_event_id else None
    known_series_categories: tuple[tuple[str, str, str], ...] = ()
    if event is None:
        user_events = indexes.events_by_user.get(message.user_id, [])
        known_series_categories = tuple(
            sorted({(e.category, e.event_type, e.direction) for e in user_events})
        )
    return EvidenceContext(message=message, event=event, known_series_categories=known_series_categories)


def _ref_from_message(message: Message) -> EvidenceRef:
    return EvidenceRef(
        user_id=message.user_id,
        message_id=message.message_id,
        request_id=message.request_id,
        related_event_id=message.related_event_id,
        sent_at=message.sent_at,
    )


@dataclass(frozen=True)
class ResolutionOutcome:
    """One message's full resolution trace - used both to build the final
    `EvidenceResolutionOutcome` and to report real-experiment statistics
    (`run_text_experiment.py`)."""

    ref: EvidenceRef
    fact: NormalizedFact | None = None
    unresolved: UnresolvedEvidence | None = None
    api_calls: int = 0
    cache_hit: bool = False
    first_rejection_reason: str | None = None
    second_rejection_reason: str | None = None
    raw_outputs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def classification(self) -> str | None:
        return self.fact.fact_type if self.fact is not None else None


def resolve_message_with_ai(
    message: Message,
    indexes: Indexes,
    client: AIClient,
    cache: EvidenceCache | None = None,
) -> ResolutionOutcome:
    ctx = build_context(message, indexes)
    ref = _ref_from_message(message)

    if cache is not None:
        cached_fact = cache.get(ctx)
        if cached_fact is not None:
            return ResolutionOutcome(ref=ref, fact=cached_fact, cache_hit=True, api_calls=0)

    system1, user1 = build_prompt(ctx)
    raw1 = client.complete(system1, user1)
    result1 = validate_response(raw1, ctx)

    if result1.is_valid:
        if result1.fact is not None:
            if cache is not None:
                cache.set(ctx, result1.fact)
            return ResolutionOutcome(ref=ref, fact=result1.fact, api_calls=1, raw_outputs=(raw1,))
        unresolved = UnresolvedEvidence(provenance=ref, reason=f"model_unresolved: {result1.unresolved_reason}")
        return ResolutionOutcome(ref=ref, unresolved=unresolved, api_calls=1, raw_outputs=(raw1,))

    # Invalid first response - retry exactly once, materially corrective.
    system2, user2 = build_retry_prompt(ctx, raw1, result1.rejection_reason or "invalid response")
    raw2 = client.complete(system2, user2)
    result2 = validate_response(raw2, ctx)

    if result2.is_valid:
        if result2.fact is not None:
            if cache is not None:
                cache.set(ctx, result2.fact)
            return ResolutionOutcome(
                ref=ref, fact=result2.fact, api_calls=2,
                first_rejection_reason=result1.rejection_reason, raw_outputs=(raw1, raw2),
            )
        unresolved = UnresolvedEvidence(
            provenance=ref, reason=f"model_unresolved_after_retry: {result2.unresolved_reason}"
        )
        return ResolutionOutcome(
            ref=ref, unresolved=unresolved, api_calls=2,
            first_rejection_reason=result1.rejection_reason, raw_outputs=(raw1, raw2),
        )

    # Still invalid after the single allowed retry - unresolved, never cached.
    unresolved = UnresolvedEvidence(
        provenance=ref,
        reason=f"invalid_model_output_after_retry: {result2.rejection_reason}",
    )
    return ResolutionOutcome(
        ref=ref, unresolved=unresolved, api_calls=2,
        first_rejection_reason=result1.rejection_reason,
        second_rejection_reason=result2.rejection_reason,
        raw_outputs=(raw1, raw2),
    )


def run_stage4b(
    dataset: Dataset,
    indexes: Indexes,
    client: AIClient,
    cache: EvidenceCache | None = None,
) -> tuple[EvidenceResolutionOutcome, list[ResolutionOutcome]]:
    """Full Stage 4a + Stage 4b pipeline over every message/image in
    `dataset`. Calls `client` once (or twice, on a single retry) per
    escalated MESSAGE - for a real client this consumes API quota for
    every one of them, so real-API experiments should use
    `resolve_message_with_ai` directly on a small slice instead (see
    `run_text_experiment.py`)."""
    prefilter = run_prefilter(dataset, indexes.events_by_id)
    messages_by_id = {m.message_id: m for m in dataset.messages}

    facts: list[NormalizedFact] = [
        d.fact for d in prefilter.by_bucket(NO_FACT) if d.fact is not None
    ] + [
        d.fact for d in prefilter.by_bucket(DETERMINISTIC_FACT) if d.fact is not None
    ]
    unresolved: list[UnresolvedEvidence] = [
        UnresolvedEvidence(provenance=d.ref, reason=d.reason)
        for bucket in ("missing_evidence", "unresolved_other")
        for d in prefilter.by_bucket(bucket)
    ]

    ai_outcomes: list[ResolutionOutcome] = []
    for decision in prefilter.by_bucket(NEEDS_ESCALATION):
        if decision.ref.image_id is not None:
            unresolved.append(
                UnresolvedEvidence(
                    provenance=decision.ref,
                    reason="requires image interpretation (Stage 4d, not yet implemented)",
                )
            )
            continue
        message = messages_by_id[decision.ref.message_id]
        outcome = resolve_message_with_ai(message, indexes, client, cache)
        ai_outcomes.append(outcome)
        if outcome.fact is not None:
            facts.append(outcome.fact)
        elif outcome.unresolved is not None:
            unresolved.append(outcome.unresolved)

    return EvidenceResolutionOutcome(facts=tuple(facts), unresolved=tuple(unresolved)), ai_outcomes


__all__ = [
    "build_context",
    "resolve_message_with_ai",
    "run_stage4b",
    "ResolutionOutcome",
]
