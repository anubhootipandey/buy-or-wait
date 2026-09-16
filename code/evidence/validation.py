"""
Stage 4b application-level validation.

Turns the AI client's raw response text into exactly one of:

  * a `NormalizedFact` (classification was actionable and every check
    passed - `resolution_method="gemini"`),
  * an "unresolved" outcome (classification=="unresolved", or "no_fact"
    which is itself a valid-but-inert `NormalizedFact` per Stage 4a's own
    convention), or
  * a rejection (`rejection_reason` set) - the response is INVALID and
    must never be treated as a fact. `resolver.py` uses this to drive the
    "retry exactly once, then unresolved" policy.

This module NEVER repairs, coerces, or fills in a malformed/missing
field - see `_Reject` usage throughout. It also never trusts the model's
own claims about amounts or quotes without checking them against the
actual source text (`_quote_is_grounded`, `_amount_is_grounded`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

from data.models import KNOWN_CURRENCIES

from .models import EvidenceRef, NormalizedFact
from .prompts import EvidenceContext, ImageEvidenceContext
from .schema import (
    CLASSIFICATIONS,
    CONFIDENCE_LEVELS,
    CORRECTABLE_STATUSES,
    IMAGE_CLASSIFICATIONS,
    IMAGE_DOCUMENT_TYPES,
    IMAGE_RESPONSE_FIELDS,
    RESPONSE_FIELDS,
)


class _Reject(Exception):
    """Internal control-flow only - never escapes `validate_response`."""


@dataclass(frozen=True)
class ValidationResult:
    fact: Optional[NormalizedFact] = None
    unresolved_reason: Optional[str] = None
    rejection_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.rejection_reason is None

    @property
    def is_rejected(self) -> bool:
        return self.rejection_reason is not None


def _reject(reason: str) -> None:
    raise _Reject(reason)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _quote_is_grounded(source_text: str, quote: str) -> bool:
    """Exact-character substring match, tolerant only of whitespace
    differences (e.g. a stray double space) - never case-insensitive,
    per the prompt's own instruction that the quote must be verbatim."""
    return _normalize_whitespace(quote) in _normalize_whitespace(source_text)


_NUMBER_RE_SOURCE = r"[0-9][0-9,]*(?:\.[0-9]+)?"


def _numbers_in_text(text: str) -> list[Decimal]:
    import re

    out = []
    for raw in re.findall(_NUMBER_RE_SOURCE, text):
        try:
            out.append(Decimal(raw.replace(",", "")))
        except InvalidOperation:
            continue
    return out


def _amount_is_grounded(source_text: str, amount: Decimal) -> bool:
    return any(amount == candidate for candidate in _numbers_in_text(source_text))


def _parse_optional_date(value: object, field_name: str) -> Optional[date]:
    if value is None:
        return None
    if not isinstance(value, str):
        _reject(f"invalid {field_name}: not a string")
    try:
        return date.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        _reject(f"invalid {field_name}: not an ISO-8601 date ({value!r})")
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_optional_amount(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(f"invalid amount: not a number ({value!r})")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        _reject(f"invalid amount: could not parse {value!r}")
        raise AssertionError("unreachable")  # pragma: no cover
    if amount <= 0:
        _reject(f"invalid amount: must be positive, got {amount}")
    return amount


def validate_response(raw_text: str, ctx: EvidenceContext) -> ValidationResult:
    try:
        return _validate_response_inner(raw_text, ctx)
    except _Reject as exc:
        return ValidationResult(rejection_reason=str(exc))


def _validate_response_inner(raw_text: str, ctx: EvidenceContext) -> ValidationResult:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _reject(f"malformed JSON: {exc}")

    if not isinstance(parsed, dict):
        _reject(f"response is not a JSON object (got {type(parsed).__name__})")

    missing = [f for f in RESPONSE_FIELDS if f not in parsed]
    if missing:
        _reject(f"missing required field(s): {sorted(missing)}")
    unexpected = [k for k in parsed if k not in RESPONSE_FIELDS]
    if unexpected:
        _reject(f"unexpected field(s): {sorted(unexpected)}")

    evidence_id = parsed["evidence_id"]
    if not isinstance(evidence_id, str) or evidence_id != ctx.message.message_id:
        _reject(f"invalid evidence_id: expected {ctx.message.message_id!r}, got {evidence_id!r}")

    classification = parsed["classification"]
    if classification not in CLASSIFICATIONS:
        _reject(f"unsupported classification: {classification!r}")

    confidence = parsed["confidence"]
    if confidence not in CONFIDENCE_LEVELS:
        _reject(f"invalid confidence: {confidence!r}")

    quote = parsed["supporting_quote"]
    if not isinstance(quote, str) or not quote.strip():
        _reject("supporting_quote is empty or not a string")
    if not _quote_is_grounded(ctx.message.message_text, quote):
        _reject(f"ungrounded supporting_quote: {quote!r} does not appear verbatim in the source message text")

    amount = _parse_optional_amount(parsed["amount"])
    currency = parsed["currency"]
    if currency is not None:
        if not isinstance(currency, str) or currency not in KNOWN_CURRENCIES:
            _reject(f"invalid currency: {currency!r}")
        if ctx.event is not None and currency != ctx.event.currency:
            _reject(
                f"currency {currency!r} contradicts the related event's own "
                f"currency {ctx.event.currency!r}"
            )
    if amount is not None and not _amount_is_grounded(ctx.message.message_text, amount):
        _reject(f"ungrounded amount: {amount} does not appear in the source message text")

    effective_date = _parse_optional_date(parsed["effective_date"], "effective_date")
    end_date = _parse_optional_date(parsed["end_date"], "end_date")
    if effective_date is not None and end_date is not None and end_date < effective_date:
        _reject(f"end_date {end_date} precedes effective_date {effective_date}")

    new_status = parsed["new_status"]
    if new_status is not None and new_status not in CORRECTABLE_STATUSES:
        _reject(f"invalid new_status: {new_status!r}")

    target_category = parsed["target_category"]
    target_event_type = parsed["target_event_type"]
    target_direction = parsed["target_direction"]

    # --- classification-specific requirements -----------------------------
    if classification == "resolved_amount":
        if ctx.event is None:
            _reject("resolved_amount requires a related event; this message is standalone")
        if ctx.event.amount is not None:
            _reject("resolved_amount used on an event that already has a known amount")
        if amount is None or currency is None:
            _reject("resolved_amount requires non-null amount and currency")

    elif classification == "status_correction":
        if ctx.event is None:
            _reject("status_correction requires a related event; this message is standalone")
        if new_status is None:
            _reject("status_correction requires a non-null new_status")
        if new_status == ctx.event.status:
            _reject(f"new_status {new_status!r} does not actually change the event's current status")

    elif classification == "future_amount_change":
        if amount is None or currency is None or effective_date is None:
            _reject("future_amount_change requires non-null amount, currency, and effective_date")
        target_series_key = _resolve_series_key(
            ctx, target_category, target_event_type, target_direction
        )

    elif classification == "series_terminated":
        if effective_date is None:
            _reject("series_terminated requires a non-null effective_date")
        target_series_key = _resolve_series_key(
            ctx, target_category, target_event_type, target_direction
        )

    # no_fact / unresolved: no additional required fields.

    if classification == "unresolved":
        reason = parsed["negation_or_ambiguity_notes"] or "model classified evidence as unresolved"
        return ValidationResult(unresolved_reason=reason)

    provenance = EvidenceRef(
        user_id=ctx.message.user_id,
        message_id=ctx.message.message_id,
        request_id=ctx.message.request_id,
        related_event_id=ctx.message.related_event_id,
        sent_at=ctx.message.sent_at,
    )

    target_event_id = ctx.event.event_id if ctx.event is not None else None
    target_series_key = None
    if classification in ("future_amount_change", "series_terminated"):
        target_series_key = _resolve_series_key(ctx, target_category, target_event_type, target_direction)

    fact = NormalizedFact(
        fact_id=f"fact:{ctx.message.message_id}:{classification}",
        user_id=ctx.message.user_id,
        fact_type=classification,
        provenance=provenance,
        target_event_id=target_event_id,
        target_series_key=target_series_key,
        resolved_amount=amount,
        currency=currency,
        new_status=new_status,
        effective_date=effective_date,
        end_date=end_date,
        resolution_method="gemini",
        confidence=confidence,
        raw_model_output=parsed,
    )
    return ValidationResult(fact=fact)


def _resolve_series_key(
    ctx: EvidenceContext,
    target_category: object,
    target_event_type: object,
    target_direction: object,
) -> tuple[str, str, str]:
    """Ground a series-level fact's (category, event_type, direction) key.

    If the message is linked to a known event, Python derives the key
    itself from that event - the model's target_* fields are ignored (and
    must not be trusted) in that case. If the message is standalone, the
    model's target_* fields are used ONLY if they exactly match one of
    this user's own already-observed recurring-series keys - never a
    category the model invented.
    """
    if ctx.event is not None:
        return (ctx.event.category, ctx.event.event_type, ctx.event.direction)

    if not (target_category and target_event_type and target_direction):
        _reject(
            "standalone future_amount_change/series_terminated requires non-null "
            "target_category, target_event_type, and target_direction"
        )
    key = (target_category, target_event_type, target_direction)
    if key not in ctx.known_series_categories:
        _reject(
            f"target series {key!r} is not among this user's own known recurring "
            "series - refusing to trust an invented category"
        )
    return key  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stage 4d: image response validation.
# ---------------------------------------------------------------------------

# An amount larger than this is refused outright as implausible for a
# single consumer financial document, in ANY currency. Chosen generously
# (a trillion) so that legitimately large low-denomination-currency
# amounts - IDR, VND - are never wrongly rejected; the point is to catch
# a misread that concatenated digits or swallowed a decimal point, not to
# second-guess the dataset's own scale.
MAX_PLAUSIBLE_IMAGE_AMOUNT = Decimal("1e12")

# Amounts are money, not measurements: more than 2 decimal places means
# the model has misread a separator (e.g. "1.234.56") rather than found a
# genuinely sub-cent figure.
MAX_IMAGE_AMOUNT_DECIMAL_PLACES = 2


def _decimal_places(value: Decimal) -> int:
    exponent = value.normalize().as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def validate_image_response(raw_text: str, ctx: ImageEvidenceContext) -> ValidationResult:
    """Image-evidence twin of `validate_response`.

    Same three outcomes and the same absolute refusal to repair, coerce,
    or fill in anything the model got wrong. The two image-specific
    checks are:

      * amount grounding - an image has no source text to match a quote
        against, so instead the model's own verbatim transcription
        (`amount_text_as_shown`) is re-parsed by Python and must yield
        the numeric `amount` it reported. This catches separator
        misreads and digit concatenation, which are the realistic
        failure modes for document OCR.
      * currency authority - the fact's currency is ALWAYS the dataset
        event's own currency. The model is never asked for a currency and
        can never set one; `currency_as_shown` is recorded for audit and,
        if it happens to be a real ISO code that contradicts the event,
        is treated as a reason to reject rather than to overrule the
        dataset.
    """
    try:
        return _validate_image_response_inner(raw_text, ctx)
    except _Reject as exc:
        return ValidationResult(rejection_reason=str(exc))


def _validate_image_response_inner(raw_text: str, ctx: ImageEvidenceContext) -> ValidationResult:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _reject(f"malformed JSON: {exc}")

    if not isinstance(parsed, dict):
        _reject(f"response is not a JSON object (got {type(parsed).__name__})")

    missing = [f for f in IMAGE_RESPONSE_FIELDS if f not in parsed]
    if missing:
        _reject(f"missing required field(s): {sorted(missing)}")
    unexpected = [k for k in parsed if k not in IMAGE_RESPONSE_FIELDS]
    if unexpected:
        # Notably this is what refuses a model-supplied "currency": the
        # image contract has no such field, so offering one is invalid.
        _reject(f"unexpected field(s): {sorted(unexpected)}")

    evidence_id = parsed["evidence_id"]
    if not isinstance(evidence_id, str) or evidence_id != ctx.image.image_id:
        _reject(f"invalid evidence_id: expected {ctx.image.image_id!r}, got {evidence_id!r}")

    classification = parsed["classification"]
    if classification not in IMAGE_CLASSIFICATIONS:
        _reject(f"unsupported classification: {classification!r}")

    confidence = parsed["confidence"]
    if confidence not in CONFIDENCE_LEVELS:
        _reject(f"invalid confidence: {confidence!r}")

    document_type = parsed["document_type"]
    if document_type is not None and document_type not in IMAGE_DOCUMENT_TYPES:
        _reject(f"invalid document_type: {document_type!r}")

    # `currency_as_shown` is free-form (a symbol like "Rs." is fine and
    # common), but if the model reports an exact ISO code that the system
    # knows AND it contradicts the authoritative event currency, that is a
    # genuine conflict - reject rather than silently overrule the dataset.
    currency_as_shown = parsed["currency_as_shown"]
    if currency_as_shown is not None and not isinstance(currency_as_shown, str):
        _reject(f"invalid currency_as_shown: not a string ({currency_as_shown!r})")
    if (
        isinstance(currency_as_shown, str)
        and currency_as_shown.strip().upper() in KNOWN_CURRENCIES
        and currency_as_shown.strip().upper() != ctx.event.currency
    ):
        _reject(
            f"document currency {currency_as_shown!r} contradicts the related "
            f"event's own authoritative currency {ctx.event.currency!r}"
        )

    amount = _parse_optional_amount(parsed["amount"])
    amount_text = parsed["amount_text_as_shown"]
    amount_label = parsed["amount_label"]

    if classification == "unresolved":
        reason = parsed["negation_or_ambiguity_notes"] or "model classified image as unresolved"
        return ValidationResult(unresolved_reason=reason)

    if classification == "resolved_amount":
        if ctx.event.amount is not None:
            _reject("resolved_amount used on an event that already has a known amount")
        if amount is None:
            _reject("resolved_amount requires a non-null amount")
        if not isinstance(amount_text, str) or not amount_text.strip():
            _reject("resolved_amount requires a non-empty amount_text_as_shown")
        if not isinstance(amount_label, str) or not amount_label.strip():
            _reject("resolved_amount requires a non-empty amount_label")
        if amount > MAX_PLAUSIBLE_IMAGE_AMOUNT:
            _reject(
                f"implausible amount: {amount} exceeds the "
                f"{MAX_PLAUSIBLE_IMAGE_AMOUNT} ceiling for a single document"
            )
        if _decimal_places(amount) > MAX_IMAGE_AMOUNT_DECIMAL_PLACES:
            _reject(
                f"implausible amount: {amount} has more than "
                f"{MAX_IMAGE_AMOUNT_DECIMAL_PLACES} decimal places"
            )
        if not _amount_is_grounded(amount_text, amount):
            _reject(
                f"ungrounded amount: {amount} does not match the model's own "
                f"transcription of the document ({amount_text!r})"
            )
    else:
        # no_fact: must not smuggle an amount through an inert result.
        if amount is not None:
            _reject(f"classification {classification!r} must not carry an amount")

    provenance = EvidenceRef(
        user_id=ctx.image.user_id,
        image_id=ctx.image.image_id,
        request_id=ctx.image.request_id,
        related_event_id=ctx.image.related_event_id,
    )

    fact = NormalizedFact(
        fact_id=f"fact:{ctx.image.image_id}:{classification}",
        user_id=ctx.image.user_id,
        fact_type=classification,
        provenance=provenance,
        target_event_id=ctx.event.event_id,
        resolved_amount=amount,
        # The dataset's event currency is authoritative, always. The model
        # is never consulted for this value.
        currency=ctx.event.currency if amount is not None else None,
        resolution_method="gemini",
        confidence=confidence,
        raw_model_output=parsed,
    )
    return ValidationResult(fact=fact)


__all__ = ["ValidationResult", "validate_response", "validate_image_response"]
