"""
Stage 4b prompt construction.

Builds the (system_instructions, user_prompt) pair sent to the AI client
for one message-level piece of evidence that Stage 4a's deterministic
prefilter could not resolve. Kept in one place so the contract the model
is held to (schema.py) and the words describing that contract stay in
sync, and so retries can be built by extending the same base prompt
rather than re-deriving it.

This module does no I/O and calls no client - it only builds strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.models import FinancialEvent, ImageRecord, Message

from .image_input import ImagePayload
from .schema import CLASSIFICATIONS, CORRECTABLE_STATUSES, IMAGE_DOCUMENT_TYPES

SYSTEM_INSTRUCTIONS = """\
You are a strict evidence-interpretation component inside a deterministic \
personal-finance planner. You read ONE message and decide what, if \
anything, it establishes about a user's cash flow. You never make the \
financial decision yourself - a separate Python system does all financial \
math and safety checks. Your only job is semantic interpretation of text.

Rules you must follow exactly:
1. Base your answer ONLY on the message text and the structured context \
given to you. Never use outside knowledge about the merchant/employer \
named in the message.
2. Pay close attention to NEGATION. "You will NOT be charged", "payment \
has been cancelled", "we will not debit your account" must never become \
a future charge or a positive amount. Prefer classification="no_fact" or \
"status_correction" (new_status="cancelled") for these, never \
"future_amount_change" or "resolved_amount".
3. Prize, reward, lottery, or "pay a release/processing charge to claim \
your winnings" language is very likely a scam or, at best, unconfirmed. \
It must NEVER become income. If the money has not been confirmed as \
credited, use "no_fact" (nothing for the planner to act on yet) - do not \
invent a future income fact.
4. A "pending" credit (bonus, commission, refund, prize, investment gain) \
that has not yet been confirmed/credited is NOT confirmed income. Use \
"no_fact" unless the message explicitly confirms the money has landed.
5. A "failed" payment with live retry/replacement language is being \
handled separately by the deterministic system - do not fabricate a new \
amount for it; if the message only describes a retry with the SAME \
amount, classification is usually "no_fact".
6. `supporting_quote` MUST be an exact, verbatim substring of the message \
text below (same characters, same case) that most directly supports your \
classification. Never paraphrase it. Never invent a quote.
7. `amount`, when not null, MUST be a plain number that also appears \
(same digits, ignoring currency symbols/thousands separators) somewhere \
in the message text. Never compute, round, or convert an amount yourself.
8. If you are not confident the message safely resolves to one of the \
narrow classifications below, use "unresolved" and explain why in \
`negation_or_ambiguity_notes`. Returning "unresolved" is always safer \
than guessing.
9. Return ONLY the fields defined by the schema you were given. Do not \
add commentary outside the structured response.

Valid `classification` values and when to use them:
- "resolved_amount": the message states the single actual amount for a \
  financial event whose amount is not yet known. Requires `amount` and \
  `currency`.
- "status_correction": the message confirms the linked event's status \
  actually changed (e.g. cancelled, confirmed settled). Requires \
  `new_status`.
- "future_amount_change": the message describes a recurring series \
  (salary, subscription, rent, etc.) changing to a new amount from a \
  specific date onward (may be temporary, with an `end_date`). Requires \
  `amount`, `currency`, `effective_date`.
- "series_terminated": the message describes a recurring series ending \
  (contract ended, subscription cancelled going forward, no renewal). \
  Requires `effective_date`.
- "no_fact": evidence considered, but there is nothing actionable for \
  the planner (already reflected in known data, a scam, an unconfirmed \
  pending credit, a negated/cancelled charge, mere corroboration).
- "unresolved": you cannot safely classify this evidence.
"""


@dataclass(frozen=True)
class EvidenceContext:
    """Everything about the *already-known* structured record (if any)
    that the model is allowed to see, kept deliberately narrow - it never
    sees other users' data or unrelated events."""

    message: Message
    event: Optional[FinancialEvent]
    known_series_categories: tuple[tuple[str, str, str], ...] = ()

    @property
    def is_standalone(self) -> bool:
        return self.event is None


def _event_context_block(ctx: EvidenceContext) -> str:
    if ctx.event is None:
        if ctx.known_series_categories:
            series_lines = "\n".join(
                f"  - category={c!r}, event_type={t!r}, direction={d!r}"
                for c, t, d in ctx.known_series_categories
            )
        else:
            series_lines = "  (none on record)"
        return (
            "This message is STANDALONE: it has no related_event_id, so there is no "
            "single financial-event record it amends directly. If it describes a "
            "recurring series changing (e.g. a salary change), you must identify the "
            "series using ONLY one of this user's own known (category, event_type, "
            "direction) combinations below - never invent a new one:\n"
            f"{series_lines}\n"
            "If the message doesn't clearly match one of these known series, use "
            "\"no_fact\" or \"unresolved\" instead of guessing a category."
        )
    e = ctx.event
    return (
        "This message is linked to exactly one financial-event record (already "
        "known to the system, given here for grounding only - do not restate it "
        "as if you discovered it):\n"
        f"  event_id={e.event_id!r}\n"
        f"  event_type={e.event_type!r}\n"
        f"  category={e.category!r}\n"
        f"  direction={e.direction!r}\n"
        f"  amount={'<blank>' if e.amount is None else str(e.amount)}\n"
        f"  currency={e.currency!r}\n"
        f"  status={e.status!r}\n"
        f"  event_date={e.event_date.isoformat()}\n"
    )


def build_prompt(ctx: EvidenceContext) -> tuple[str, str]:
    """Return (system_instructions, user_prompt) for the first attempt."""
    user_prompt = (
        f"evidence_id: {ctx.message.message_id}\n"
        f"message sent_at: {ctx.message.sent_at.isoformat()}\n"
        f"message source_type: {ctx.message.source_type}\n\n"
        f"{_event_context_block(ctx)}\n"
        "Message text (verbatim, between the markers):\n"
        "-----BEGIN MESSAGE-----\n"
        f"{ctx.message.message_text}\n"
        "-----END MESSAGE-----\n\n"
        f"Respond with evidence_id exactly equal to {ctx.message.message_id!r}."
    )
    return SYSTEM_INSTRUCTIONS, user_prompt


def build_retry_prompt(ctx: EvidenceContext, previous_raw_output: str, rejection_reason: str) -> tuple[str, str]:
    """Return (system_instructions, user_prompt) for the single allowed
    retry after an invalid first response. Materially more corrective
    than the first attempt: it shows the model exactly what it returned
    and exactly why that was rejected, rather than just repeating the
    original prompt verbatim."""
    base_system, base_user = build_prompt(ctx)
    correction = (
        "\n\nYour previous response was REJECTED by strict validation and will "
        "never be accepted again in this form. Fix it.\n"
        f"Your previous response was:\n{previous_raw_output}\n\n"
        f"Rejection reason: {rejection_reason}\n\n"
        "Re-read the rules above. Return a corrected response that fixes exactly "
        "this problem. If you cannot produce a response that would pass, use "
        "classification=\"unresolved\" and explain why in "
        "negation_or_ambiguity_notes instead of repeating the same mistake."
    )
    return base_system, base_user + correction


# ---------------------------------------------------------------------------
# Stage 4d: image evidence prompts.
# ---------------------------------------------------------------------------

IMAGE_SYSTEM_INSTRUCTIONS = """\
You are a strict document-reading component inside a deterministic \
personal-finance planner. You are shown ONE image of a financial document \
(a receipt, invoice, payslip, bill, or similar) that is attached to ONE \
financial event whose amount is currently unknown to the system. Your only \
job is to read the single amount that this document establishes for that \
event. You never make the financial decision - a separate Python system \
does all financial math, forecasting, and safety checks.

Rules you must follow exactly:
1. Read ONLY what is visibly printed or written in the image. Never infer, \
compute, convert, or estimate an amount that is not shown. Never use \
outside knowledge about the merchant, employer, or document shown.
2. Return the amount that represents what this event actually costs or \
pays the user - the settlement figure, not a historical or informational \
one. Prefer, in order: an explicitly outstanding figure ("Balance Due", \
"Amount Payable", "Amount Due"), then the final settled figure ("Grand \
Total", "Total Paid", "Total Amount Received", "Net Pay"), then a plain \
"Total". If a document shows BOTH a larger historical total AND a smaller \
outstanding balance, the outstanding balance is the answer.
3. Do NOT return a subtotal, a line item, a tax line, a previous balance, \
a credit limit, an account number, a date, or a loyalty-points figure as \
the amount.
4. `amount_text_as_shown` MUST be the amount transcribed EXACTLY as it \
appears in the document, including its thousands separators and decimal \
point (for example "8,528.10" or "1 00 000"). Do not reformat it, do not \
strip separators, do not round. This transcription is checked against the \
numeric `amount` you return - if they disagree, your response is rejected.
5. `amount` MUST be the same number as `amount_text_as_shown`, expressed \
as a plain JSON number with no separators (for example 8528.10).
6. `amount_label` MUST be the label printed next to that amount in the \
document (for example "Grand Total", "Balance Due", "Net Pay"). If the \
amount genuinely has no printed label, describe its position in a few \
words (for example "handwritten figure at foot of note").
7. `currency_as_shown` is ONLY a record of what symbol or code the \
document displays (for example "Rs.", "INR", "$"). It is NOT used as the \
event's currency - the system already knows the correct currency and will \
use its own. Never try to change it. Report null if nothing is shown.
8. If the image is unreadable, ambiguous between multiple plausible \
settlement amounts, shows no amount at all, or is not a financial \
document, use classification="unresolved" and explain why in \
`negation_or_ambiguity_notes`. Returning "unresolved" is ALWAYS safer \
than guessing - a wrong amount corrupts a real financial forecast.
9. Use classification="no_fact" only when the document is genuinely inert \
for this event (for example it merely corroborates something with no \
amount to extract).
10. Return ONLY the fields defined by the schema you were given. Do not \
add commentary outside the structured response.

Valid `classification` values:
- "resolved_amount": the document establishes this event's amount. \
  Requires `amount`, `amount_text_as_shown`, and `amount_label`.
- "no_fact": document considered, nothing actionable to extract.
- "unresolved": you cannot safely read or decide this document.
"""


@dataclass(frozen=True)
class ImageEvidenceContext:
    """Everything about an image and the single event it is attached to
    that the model is allowed to see. As with `EvidenceContext`, it never
    carries another user's data or an unrelated event."""

    image: ImageRecord
    event: FinancialEvent
    payload: ImagePayload

    @property
    def evidence_id(self) -> str:
        return self.image.image_id


def _image_event_context_block(ctx: ImageEvidenceContext) -> str:
    e = ctx.event
    return (
        "This image is attached to exactly one financial-event record whose "
        "amount is currently BLANK. The record is given here for grounding "
        "only - do not restate it as if you discovered it:\n"
        f"  event_id={e.event_id!r}\n"
        f"  event_type={e.event_type!r}\n"
        f"  category={e.category!r}\n"
        f"  direction={e.direction!r}\n"
        f"  amount=<blank - this is what you are reading from the image>\n"
        f"  currency={e.currency!r}  (AUTHORITATIVE - the system uses this, "
        "not anything the document displays)\n"
        f"  status={e.status!r}\n"
        f"  event_date={e.event_date.isoformat()}\n"
    )


def build_image_prompt(ctx: ImageEvidenceContext) -> tuple[str, str]:
    """Return (system_instructions, user_prompt) for the first attempt at
    one image. The image bytes themselves are passed separately by the
    client (see `AIClient.complete_with_image`) - this builds only text."""
    doc_types = ", ".join(repr(t) for t in IMAGE_DOCUMENT_TYPES)
    user_prompt = (
        f"evidence_id: {ctx.image.image_id}\n"
        f"image media type: {ctx.payload.mime_type}\n\n"
        f"{_image_event_context_block(ctx)}\n"
        "The document image is attached to this request. Read it and report "
        "the single amount it establishes for the event above.\n"
        f"`document_type` must be one of: {doc_types}.\n\n"
        f"Respond with evidence_id exactly equal to {ctx.image.image_id!r}."
    )
    return IMAGE_SYSTEM_INSTRUCTIONS, user_prompt


def build_image_retry_prompt(
    ctx: ImageEvidenceContext, previous_raw_output: str, rejection_reason: str
) -> tuple[str, str]:
    """Return the single allowed corrective retry prompt for one image -
    same policy and same reasoning as `build_retry_prompt` for text."""
    base_system, base_user = build_image_prompt(ctx)
    correction = (
        "\n\nYour previous response was REJECTED by strict validation and will "
        "never be accepted again in this form. Fix it.\n"
        f"Your previous response was:\n{previous_raw_output}\n\n"
        f"Rejection reason: {rejection_reason}\n\n"
        "Re-read the rules above and look at the image again. Return a "
        "corrected response that fixes exactly this problem. If you cannot "
        "produce a response that would pass, use classification=\"unresolved\" "
        "and explain why in negation_or_ambiguity_notes instead of repeating "
        "the same mistake."
    )
    return base_system, base_user + correction


__all__ = [
    "SYSTEM_INSTRUCTIONS",
    "IMAGE_SYSTEM_INSTRUCTIONS",
    "ImageEvidenceContext",
    "build_image_prompt",
    "build_image_retry_prompt",
    "EvidenceContext",
    "build_prompt",
    "build_retry_prompt",
    "CLASSIFICATIONS",
    "CORRECTABLE_STATUSES",
]
