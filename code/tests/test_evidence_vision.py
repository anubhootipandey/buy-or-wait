"""
Stage 4d (image/vision evidence) tests.

No test here ever touches the network or the real Gemini SDK: every test
uses a fake client implementing `complete_with_image`, so the suite stays
fully deterministic and consumes no API quota - the same discipline the
Stage 4b text tests already follow.
"""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from data.models import Dataset, FinancialEvent, ImageRecord
from evidence.apply_facts import apply_resolved_amounts
from evidence.cache import EvidenceCache
from evidence.image_input import (
    ImagePayload,
    UnsupportedImageError,
    load_image,
    sniff_mime_type,
)
from evidence.models import NormalizedFact
from evidence.resolver import build_image_context, resolve_image_with_ai
from evidence.validation import validate_image_response

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff\xe0"


def make_event(
    event_id: str = "event_1",
    amount=None,
    currency: str = "INR",
    status: str = "pending",
) -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id="user_1",
        event_type="purchase",
        description="test event",
        category="shopping",
        direction="debit",
        amount=amount,
        currency=currency,
        event_date=date(2025, 6, 1),
        settlement_date=None,
        status=status,
        linked_event_id=None,
        flexibility="fixed",
        minimum_allowed_amount=None,
    )


def make_image(image_id: str = "image_01", related_event_id: str = "event_1") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        user_id="user_1",
        request_id="request_1",
        related_event_id=related_event_id,
        file_path="/nonexistent/not-actually-read.png",
    )


def make_payload(data: bytes = PNG_MAGIC + b"body", mime_type: str = "image/png") -> ImagePayload:
    import hashlib

    return ImagePayload(
        data=data,
        mime_type=mime_type,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


def image_response(**overrides) -> str:
    payload = {
        "evidence_id": "image_01",
        "classification": "resolved_amount",
        "amount": 8528.10,
        "amount_text_as_shown": "8,528.10",
        "amount_label": "Grand Total",
        "currency_as_shown": "Rs.",
        "document_type": "receipt",
        "confidence": "high",
        "negation_or_ambiguity_notes": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeVisionClient:
    """Fake multimodal client. Records every call so tests can assert on
    how many API calls were made and what bytes/MIME type were sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system_instructions: str, user_prompt: str) -> str:
        raise AssertionError("text completion should not be used for image evidence")

    def complete_with_image(self, system_instructions, user_prompt, image_bytes, mime_type):
        self.calls.append(
            {
                "system": system_instructions,
                "prompt": user_prompt,
                "bytes": image_bytes,
                "mime_type": mime_type,
            }
        )
        if not self.responses:
            raise AssertionError("FakeVisionClient ran out of scripted responses")
        return self.responses.pop(0)


class TextOnlyClient:
    """A client predating Stage 4d - has no vision support at all."""

    def complete(self, system_instructions: str, user_prompt: str) -> str:
        return "{}"


class ExplodingVisionClient:
    def complete(self, system_instructions: str, user_prompt: str) -> str:
        raise AssertionError("unused")

    def complete_with_image(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


# ---------------------------------------------------------------------------
# MIME sniffing / image loading
# ---------------------------------------------------------------------------


class SniffMimeTypeTests(unittest.TestCase):
    def test_png_detected_from_magic_bytes(self):
        self.assertEqual(sniff_mime_type(PNG_MAGIC + b"rest"), "image/png")

    def test_jpeg_detected_from_magic_bytes(self):
        self.assertEqual(sniff_mime_type(JPEG_MAGIC + b"rest"), "image/jpeg")

    def test_webp_detected_from_riff_container(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest"
        self.assertEqual(sniff_mime_type(data), "image/webp")

    def test_magic_bytes_beat_a_lying_extension(self):
        # A JPEG named .png is a JPEG. Sending it as image/png is exactly
        # the silent failure this sniffing exists to prevent.
        self.assertEqual(sniff_mime_type(JPEG_MAGIC + b"rest", "photo.png"), "image/jpeg")

    def test_extension_used_only_when_content_is_inconclusive(self):
        self.assertEqual(sniff_mime_type(b"\x00\x01\x02\x03", "scan.webp"), "image/webp")

    def test_unknown_content_and_extension_returns_none(self):
        self.assertIsNone(sniff_mime_type(b"\x00\x01\x02\x03", "notes.txt"))


class LoadImageTests(unittest.TestCase):
    def test_missing_file_raises_unsupported(self):
        with self.assertRaises(UnsupportedImageError):
            load_image("/nonexistent/missing.png")

    def test_real_dataset_image_loads_with_sniffed_type(self):
        candidate = (
            Path(__file__).resolve().parents[2] / "dataset" / "media" / "images" / "image_01.png"
        )
        if not candidate.is_file():
            self.skipTest("dataset image not available")
        payload = load_image(candidate)
        self.assertEqual(payload.mime_type, "image/png")
        self.assertEqual(len(payload.sha256), 64)
        self.assertGreater(payload.byte_size, 0)


# ---------------------------------------------------------------------------
# Image response validation
# ---------------------------------------------------------------------------


class ValidateImageResponseTests(unittest.TestCase):
    def setUp(self):
        self.ctx = build_image_context(make_image(), make_event(), make_payload())

    def test_valid_image_derived_amount(self):
        result = validate_image_response(image_response(), self.ctx)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.fact)
        self.assertEqual(result.fact.resolved_amount, Decimal("8528.10"))
        self.assertEqual(result.fact.fact_type, "resolved_amount")
        self.assertEqual(result.fact.target_event_id, "event_1")
        self.assertEqual(result.fact.resolution_method, "gemini")
        self.assertEqual(result.fact.provenance.image_id, "image_01")

    def test_malformed_json_is_rejected(self):
        result = validate_image_response("{not json at all", self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("malformed JSON", result.rejection_reason)

    def test_non_numeric_amount_is_rejected(self):
        result = validate_image_response(image_response(amount="8,528.10"), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("not a number", result.rejection_reason)

    def test_negative_amount_is_rejected(self):
        result = validate_image_response(image_response(amount=-5), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("must be positive", result.rejection_reason)

    def test_amount_must_match_the_models_own_transcription(self):
        # Reports 852810 but transcribed "8,528.10" - a separator misread.
        result = validate_image_response(
            image_response(amount=852810, amount_text_as_shown="8,528.10"), self.ctx
        )
        self.assertTrue(result.is_rejected)
        self.assertIn("ungrounded amount", result.rejection_reason)

    def test_implausibly_large_amount_is_rejected(self):
        result = validate_image_response(
            image_response(amount=5e12, amount_text_as_shown="5000000000000"), self.ctx
        )
        self.assertTrue(result.is_rejected)
        self.assertIn("implausible amount", result.rejection_reason)

    def test_too_many_decimal_places_is_rejected(self):
        result = validate_image_response(
            image_response(amount=1.2345, amount_text_as_shown="1.2345"), self.ctx
        )
        self.assertTrue(result.is_rejected)
        self.assertIn("implausible amount", result.rejection_reason)

    def test_missing_required_field_is_rejected(self):
        payload = json.loads(image_response())
        del payload["amount_label"]
        result = validate_image_response(json.dumps(payload), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("missing required field", result.rejection_reason)

    def test_wrong_evidence_id_is_rejected(self):
        result = validate_image_response(image_response(evidence_id="image_99"), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid evidence_id", result.rejection_reason)

    def test_resolved_amount_on_already_known_amount_is_rejected(self):
        ctx = build_image_context(make_image(), make_event(amount=Decimal("500")), make_payload())
        result = validate_image_response(image_response(), ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("already has a known amount", result.rejection_reason)

    def test_no_fact_must_not_carry_an_amount(self):
        result = validate_image_response(
            image_response(classification="no_fact"), self.ctx
        )
        self.assertTrue(result.is_rejected)
        self.assertIn("must not carry an amount", result.rejection_reason)

    def test_unresolved_classification_returns_unresolved_not_a_fact(self):
        result = validate_image_response(
            image_response(
                classification="unresolved",
                amount=None,
                amount_text_as_shown=None,
                amount_label=None,
                negation_or_ambiguity_notes="document is too blurry to read",
            ),
            self.ctx,
        )
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.fact)
        self.assertIn("blurry", result.unresolved_reason)

    # --- currency authority --------------------------------------------

    def test_model_cannot_override_event_currency(self):
        # The document displays a dollar sign, but the dataset says INR.
        # The dataset wins - always.
        result = validate_image_response(image_response(currency_as_shown="$"), self.ctx)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.fact.currency, "INR")

    def test_model_supplying_a_currency_field_is_rejected(self):
        # The image contract has no `currency` field at all, so a model
        # trying to set one is an invalid response, not an override.
        payload = json.loads(image_response())
        payload["currency"] = "USD"
        result = validate_image_response(json.dumps(payload), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("unexpected field", result.rejection_reason)

    def test_contradicting_iso_currency_code_is_rejected_not_applied(self):
        result = validate_image_response(image_response(currency_as_shown="USD"), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("contradicts", result.rejection_reason)

    def test_currency_symbol_is_recorded_but_never_used_as_the_currency(self):
        result = validate_image_response(image_response(currency_as_shown="Rs."), self.ctx)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.fact.currency, "INR")
        self.assertEqual(result.fact.raw_model_output["currency_as_shown"], "Rs.")


# ---------------------------------------------------------------------------
# Resolver behaviour (retry / failure / caching)
# ---------------------------------------------------------------------------


class ResolveImageWithAITests(unittest.TestCase):
    def test_valid_response_resolves_in_one_call(self):
        client = FakeVisionClient([image_response()])
        outcome = resolve_image_with_ai(
            make_image(), make_event(), client, payload=make_payload()
        )
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.api_calls, 1)
        self.assertEqual(outcome.fact.resolved_amount, Decimal("8528.10"))

    def test_image_bytes_and_mime_type_are_actually_sent(self):
        payload = make_payload(JPEG_MAGIC + b"body", "image/jpeg")
        client = FakeVisionClient([image_response()])
        resolve_image_with_ai(make_image(), make_event(), client, payload=payload)
        self.assertEqual(client.calls[0]["bytes"], payload.data)
        self.assertEqual(client.calls[0]["mime_type"], "image/jpeg")

    def test_invalid_response_retries_exactly_once_then_succeeds(self):
        client = FakeVisionClient(["{bad json", image_response()])
        outcome = resolve_image_with_ai(
            make_image(), make_event(), client, payload=make_payload()
        )
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.api_calls, 2)
        self.assertIsNotNone(outcome.first_rejection_reason)
        self.assertIn("REJECTED", client.calls[1]["prompt"])

    def test_invalid_twice_fails_safe_to_unresolved(self):
        client = FakeVisionClient(["{bad", "{also bad"])
        outcome = resolve_image_with_ai(
            make_image(), make_event(), client, payload=make_payload()
        )
        self.assertIsNone(outcome.fact)
        self.assertIsNotNone(outcome.unresolved)
        self.assertIn("invalid_model_output_after_retry", outcome.unresolved.reason)
        self.assertEqual(outcome.api_calls, 2)

    def test_never_retries_more_than_once(self):
        client = FakeVisionClient(["{bad", "{also bad"])
        resolve_image_with_ai(make_image(), make_event(), client, payload=make_payload())
        self.assertEqual(len(client.calls), 2)

    def test_provider_failure_fails_safe_to_unresolved(self):
        outcome = resolve_image_with_ai(
            make_image(), make_event(), ExplodingVisionClient(), payload=make_payload()
        )
        self.assertIsNone(outcome.fact)
        self.assertIn("vision_call_failed", outcome.unresolved.reason)

    def test_client_without_vision_support_fails_safe(self):
        outcome = resolve_image_with_ai(
            make_image(), make_event(), TextOnlyClient(), payload=make_payload()
        )
        self.assertIsNone(outcome.fact)
        self.assertEqual(outcome.unresolved.reason, "ai_client_has_no_vision_support")

    def test_unreadable_image_fails_safe_without_calling_the_model(self):
        client = FakeVisionClient([image_response()])
        outcome = resolve_image_with_ai(make_image(), make_event(), client)
        self.assertIsNone(outcome.fact)
        self.assertIn("unusable_image", outcome.unresolved.reason)
        self.assertEqual(client.calls, [])


class ImageCacheTests(unittest.TestCase):
    def test_cache_hit_avoids_a_second_api_call(self):
        cache = EvidenceCache(path=None)
        payload = make_payload()
        client1 = FakeVisionClient([image_response()])
        first = resolve_image_with_ai(
            make_image(), make_event(), client1, cache=cache, payload=payload
        )
        self.assertEqual(first.api_calls, 1)
        self.assertFalse(first.cache_hit)

        client2 = FakeVisionClient([])  # would raise if called
        second = resolve_image_with_ai(
            make_image(), make_event(), client2, cache=cache, payload=payload
        )
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.api_calls, 0)
        self.assertEqual(client2.calls, [])
        self.assertEqual(second.fact.resolved_amount, first.fact.resolved_amount)

    def test_different_image_content_produces_a_different_cache_key(self):
        cache = EvidenceCache(path=None)
        image = make_image()
        event = make_event()

        resolve_image_with_ai(
            image, event, FakeVisionClient([image_response()]),
            cache=cache, payload=make_payload(PNG_MAGIC + b"original"),
        )
        # Same image_id, same event, DIFFERENT bytes -> must not hit.
        client = FakeVisionClient([image_response(amount=99, amount_text_as_shown="99")])
        outcome = resolve_image_with_ai(
            image, event, client,
            cache=cache, payload=make_payload(PNG_MAGIC + b"edited"),
        )
        self.assertFalse(outcome.cache_hit)
        self.assertEqual(outcome.api_calls, 1)
        self.assertEqual(outcome.fact.resolved_amount, Decimal("99"))
        self.assertEqual(len(cache), 2)

    def test_same_image_on_a_different_event_is_not_served_a_stale_fact(self):
        cache = EvidenceCache(path=None)
        payload = make_payload()
        resolve_image_with_ai(
            make_image(), make_event("event_1"), FakeVisionClient([image_response()]),
            cache=cache, payload=payload,
        )
        client = FakeVisionClient([image_response()])
        outcome = resolve_image_with_ai(
            make_image(related_event_id="event_2"), make_event("event_2"), client,
            cache=cache, payload=payload,
        )
        self.assertFalse(outcome.cache_hit)

    def test_unresolved_outcomes_are_never_cached(self):
        cache = EvidenceCache(path=None)
        resolve_image_with_ai(
            make_image(), make_event(), FakeVisionClient(["{bad", "{bad"]),
            cache=cache, payload=make_payload(),
        )
        self.assertEqual(len(cache), 0)


# ---------------------------------------------------------------------------
# Applying image-derived amounts to the dataset (reaching the engine)
# ---------------------------------------------------------------------------


def make_dataset(events) -> Dataset:
    return Dataset(
        profiles=[], events=list(events), exchange_rates=[], payment_options=[],
        messages=[], images=[], requests=[], sample_requests=[],
    )


def amount_fact(
    event_id: str = "event_1",
    amount: str = "8528.10",
    currency: str = "INR",
    fact_id: str = "fact:image_01:resolved_amount",
    image_id: str = "image_01",
) -> NormalizedFact:
    from evidence.models import EvidenceRef

    return NormalizedFact(
        fact_id=fact_id,
        user_id="user_1",
        fact_type="resolved_amount",
        provenance=EvidenceRef(user_id="user_1", image_id=image_id, related_event_id=event_id),
        target_event_id=event_id,
        resolved_amount=Decimal(amount),
        currency=currency,
        resolution_method="gemini",
        confidence="high",
    )


class ApplyResolvedAmountsTests(unittest.TestCase):
    def test_image_evidence_fills_a_blank_amount(self):
        dataset = make_dataset([make_event("event_1", amount=None)])
        new_dataset, report = apply_resolved_amounts(dataset, [amount_fact()])
        self.assertEqual(new_dataset.events[0].amount, Decimal("8528.10"))
        self.assertEqual(len(report.applied), 1)
        self.assertEqual(report.applied[0].image_id, "image_01")

    def test_original_dataset_is_not_mutated(self):
        dataset = make_dataset([make_event("event_1", amount=None)])
        apply_resolved_amounts(dataset, [amount_fact()])
        self.assertIsNone(dataset.events[0].amount)

    def test_known_amount_is_never_overwritten(self):
        dataset = make_dataset([make_event("event_1", amount=Decimal("500"))])
        new_dataset, report = apply_resolved_amounts(dataset, [amount_fact()])
        self.assertEqual(new_dataset.events[0].amount, Decimal("500"))
        self.assertEqual(len(report.applied), 0)
        self.assertIn("already has a known amount", report.skipped[0].reason)

    def test_currency_mismatch_is_skipped_not_applied(self):
        dataset = make_dataset([make_event("event_1", amount=None, currency="INR")])
        new_dataset, report = apply_resolved_amounts(dataset, [amount_fact(currency="USD")])
        self.assertIsNone(new_dataset.events[0].amount)
        self.assertIn("contradicts event currency", report.skipped[0].reason)

    def test_unknown_event_is_skipped(self):
        dataset = make_dataset([make_event("event_1", amount=None)])
        _, report = apply_resolved_amounts(dataset, [amount_fact(event_id="event_999")])
        self.assertEqual(len(report.applied), 0)
        self.assertIn("not present in dataset", report.skipped[0].reason)

    def test_conflicting_facts_leave_the_amount_blank(self):
        dataset = make_dataset([make_event("event_1", amount=None)])
        facts = [
            amount_fact(amount="100", fact_id="fact:image_01:resolved_amount"),
            amount_fact(amount="200", fact_id="fact:image_02:resolved_amount", image_id="image_02"),
        ]
        new_dataset, report = apply_resolved_amounts(dataset, facts)
        self.assertIsNone(new_dataset.events[0].amount)
        self.assertEqual(len(report.applied), 0)
        self.assertTrue(all("conflicting" in s.reason for s in report.skipped))

    def test_agreeing_facts_apply_once_deterministically(self):
        dataset = make_dataset([make_event("event_1", amount=None)])
        facts = [
            amount_fact(amount="100", fact_id="fact:image_02:resolved_amount", image_id="image_02"),
            amount_fact(amount="100", fact_id="fact:image_01:resolved_amount", image_id="image_01"),
        ]
        new_dataset, report = apply_resolved_amounts(dataset, facts)
        self.assertEqual(new_dataset.events[0].amount, Decimal("100"))
        self.assertEqual(len(report.applied), 1)
        self.assertEqual(report.applied[0].fact_id, "fact:image_01:resolved_amount")

    def test_non_resolved_amount_facts_are_ignored_here(self):
        from evidence.models import EvidenceRef

        series_fact = NormalizedFact(
            fact_id="fact:msg_1:series_terminated",
            user_id="user_1",
            fact_type="series_terminated",
            provenance=EvidenceRef(user_id="user_1", message_id="msg_1"),
            target_series_key=("rent", "bill", "debit"),
            effective_date=date(2025, 7, 1),
        )
        dataset = make_dataset([make_event("event_1", amount=None)])
        new_dataset, report = apply_resolved_amounts(dataset, [series_fact])
        self.assertIsNone(new_dataset.events[0].amount)
        self.assertEqual(report.applied, ())
        self.assertEqual(report.skipped, ())


class EndToEndImageEvidenceTests(unittest.TestCase):
    """The full Stage 4d path: model reads an image -> validated fact ->
    deterministic applier -> the event the financial engine will see."""

    def test_image_derived_amount_reaches_the_dataset(self):
        event = make_event("event_1", amount=None, currency="INR")
        client = FakeVisionClient([image_response()])
        outcome = resolve_image_with_ai(make_image(), event, client, payload=make_payload())

        dataset = make_dataset([event])
        new_dataset, report = apply_resolved_amounts(dataset, [outcome.fact])

        self.assertEqual(new_dataset.events[0].amount, Decimal("8528.10"))
        self.assertEqual(new_dataset.events[0].currency, "INR")
        self.assertEqual(report.applied[0].resolution_method, "gemini")

    def test_unresolved_image_leaves_the_event_blank(self):
        event = make_event("event_1", amount=None)
        client = FakeVisionClient(["{bad", "{bad"])
        outcome = resolve_image_with_ai(make_image(), event, client, payload=make_payload())
        self.assertIsNone(outcome.fact)

        dataset = make_dataset([event])
        new_dataset, report = apply_resolved_amounts(
            dataset, [f for f in [outcome.fact] if f is not None]
        )
        self.assertIsNone(new_dataset.events[0].amount)
        self.assertEqual(report.applied, ())


if __name__ == "__main__":
    unittest.main()
