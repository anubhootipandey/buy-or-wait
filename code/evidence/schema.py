"""
Stage 4b: the AI evidence-interpretation response contract.

This module defines ONLY the shape of the model's structured output and
the closed vocabularies used to validate it. No model/API client and no
validation logic live here - see `ai_client.py` for the client and
`validation.py` for turning a raw response into a `NormalizedFact` /
`UnresolvedEvidence` / rejection.

The contract is a superset of the task's required minimum fields
(evidence_id, classification, amount, currency, effective_date, end_date,
confidence, supporting_quote, negation_or_ambiguity_notes) plus a small
number of additional fields needed to ground a fact into Stage 4a's
existing `evidence.models.NormalizedFact` shape without ever letting the
model invent a `target_event_id` or a series key out of thin air:

  * `new_status`      - required only for classification="status_correction".
  * `target_category` / `target_event_type` / `target_direction` -
                        required only for a *standalone* message (no
                        `related_event_id`) classified as
                        "future_amount_change" or "series_terminated",
                        since Python has no event to derive a series key
                        from in that case. Application validation grounds
                        these against the user's OWN historical events
                        (see `validation.py`) - the model cannot invent a
                        category the user has no history in.

`classification` is intentionally a different (slightly richer) closed
vocabulary than `evidence.models.FACT_TYPES`: it adds "unresolved" as an
explicit "I looked, I can't safely decide" outcome distinct from
"no_fact" ("I looked, there is genuinely nothing to act on"). Application
validation maps classification -> fact_type/UnresolvedEvidence.
"""

from __future__ import annotations

CLASSIFICATIONS = (
    "resolved_amount",
    "status_correction",
    "future_amount_change",
    "series_terminated",
    "no_fact",
    "unresolved",
)

CONFIDENCE_LEVELS = ("high", "medium", "low")

# Statuses a message can plausibly assert as a correction. Deliberately
# excludes "unrealized" (never message-driven) and requires the value to
# differ from a bare guess - see validation.py for the "must actually
# change something" check.
CORRECTABLE_STATUSES = ("cancelled", "failed", "settled", "pending", "scheduled")

# Fields the model must always return (may be JSON null where not
# applicable to this item's classification - see validation.py for which
# combinations are actually required per classification).
RESPONSE_FIELDS = (
    "evidence_id",
    "classification",
    "amount",
    "currency",
    "effective_date",
    "end_date",
    "confidence",
    "supporting_quote",
    "negation_or_ambiguity_notes",
    "new_status",
    "target_category",
    "target_event_type",
    "target_direction",
)

# JSON Schema for the provider's structured-output / response-schema
# mechanism (Gemini's `response_json_schema`). All fields are present and
# nullable rather than "optional", so the model can never simply omit a
# field it doesn't want to think about - it must explicitly say null.
RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "evidence_id": {"type": "string"},
        "classification": {"type": "string", "enum": list(CLASSIFICATIONS)},
        "amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "supporting_quote": {"type": "string"},
        "negation_or_ambiguity_notes": {"type": ["string", "null"]},
        "new_status": {"type": ["string", "null"], "enum": list(CORRECTABLE_STATUSES) + [None]},
        "target_category": {"type": ["string", "null"]},
        "target_event_type": {"type": ["string", "null"]},
        "target_direction": {"type": ["string", "null"]},
    },
    "required": list(RESPONSE_FIELDS),
    "propertyOrdering": list(RESPONSE_FIELDS),
}


# ---------------------------------------------------------------------------
# Stage 4d: the IMAGE evidence-interpretation response contract.
# ---------------------------------------------------------------------------
#
# A separate, deliberately narrower contract than the text one above.
# Images in this pipeline exist for exactly one reason - a financial event
# whose `amount` is blank has a receipt/payslip/invoice attached - so the
# model is given a far smaller decision space than for free-text messages.
#
# Two differences from the text contract are load-bearing, not cosmetic:
#
#   1. There is NO `currency` field. The dataset's own event currency is
#      authoritative and is stamped on by Python (see validation.py).
#      Rather than asking the model for a currency and then discarding
#      its answer, the contract simply never offers it the slot - it
#      cannot override what it was never asked for. `currency_as_shown`
#      exists only to record what the document displayed, for audit and
#      for a contradiction check; it never becomes the fact's currency.
#
#   2. `amount_text_as_shown` is required whenever an amount is returned.
#      Text evidence is grounded by checking the model's quote and number
#      against the source message (`_amount_is_grounded`); an image has no
#      source text to check against. So the model must transcribe the
#      number exactly as printed, and Python re-parses that transcription
#      and requires it to match the numeric `amount` it returned. A model
#      that "reads" 8,528.10 but reports 852810 fails this check.

IMAGE_CLASSIFICATIONS = (
    "resolved_amount",  # the document states the event's actual amount
    "no_fact",          # document considered, nothing actionable in it
    "unresolved",       # cannot safely read/decide - explicitly safer than guessing
)

# Common document shapes, recorded for traceability. Closed vocabulary so
# it stays assertable in tests rather than drifting into free prose.
IMAGE_DOCUMENT_TYPES = (
    "receipt",
    "invoice",
    "payslip",
    "bank_statement",
    "bill",
    "screenshot",
    "handwritten_note",
    "other",
)

IMAGE_RESPONSE_FIELDS = (
    "evidence_id",
    "classification",
    "amount",
    "amount_text_as_shown",
    "amount_label",
    "currency_as_shown",
    "document_type",
    "confidence",
    "negation_or_ambiguity_notes",
)

IMAGE_RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "evidence_id": {"type": "string"},
        "classification": {"type": "string", "enum": list(IMAGE_CLASSIFICATIONS)},
        "amount": {"type": ["number", "null"]},
        "amount_text_as_shown": {"type": ["string", "null"]},
        "amount_label": {"type": ["string", "null"]},
        "currency_as_shown": {"type": ["string", "null"]},
        "document_type": {"type": ["string", "null"], "enum": list(IMAGE_DOCUMENT_TYPES) + [None]},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "negation_or_ambiguity_notes": {"type": ["string", "null"]},
    },
    "required": list(IMAGE_RESPONSE_FIELDS),
    "propertyOrdering": list(IMAGE_RESPONSE_FIELDS),
}
