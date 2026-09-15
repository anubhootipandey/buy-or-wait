"""Tests for Stage 4b (evidence.ai_client / prompts / schema / validation /
cache / resolver). Uses ONLY a fake/mock AI client - no network call, no
API quota consumed, fully deterministic."""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime
from decimal import Decimal

from data.indexes import Indexes
from data.models import FinancialEvent, Message

from evidence.cache import EvidenceCache
from evidence.prompts import EvidenceContext
from evidence.resolver import build_context, resolve_message_with_ai
from evidence.schema import RESPONSE_FIELDS
from evidence.validation import validate_response


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_event(event_id="ev_1", amount="100", status="settled", currency="USD",
                category="dining", event_type="expense", direction="debit", **overrides):
    defaults = dict(
        event_id=event_id, user_id="user_1", event_type=event_type, description="test event",
        category=category, direction=direction, amount=Decimal(amount) if amount is not None else None,
        currency=currency, event_date=date(2026, 1, 1), settlement_date=date(2026, 1, 1),
        status=status, linked_event_id=None, flexibility="fixed", minimum_allowed_amount=None,
    )
    defaults.update(overrides)
    return FinancialEvent(**defaults)


def make_message(message_id="msg_1", related_event_id="ev_1", text="", request_id=None, **overrides):
    defaults = dict(
        message_id=message_id, user_id="user_1", request_id=request_id,
        related_event_id=related_event_id, sent_at=datetime(2026, 1, 2, 9, 0, 0),
        source_type="merchant", message_text=text,
    )
    defaults.update(overrides)
    return Message(**defaults)


def make_indexes(events: list[FinancialEvent]) -> Indexes:
    events_by_id = {e.event_id: e for e in events}
    events_by_user: dict[str, list[FinancialEvent]] = {}
    for e in events:
        events_by_user.setdefault(e.user_id, []).append(e)
    return Indexes(
        profiles_by_user={},
        events_by_id=events_by_id,
        events_by_user=events_by_user,
        payment_options_by_request={},
        messages_by_request_and_user={},
        messages_by_user={},
        messages_by_related_event={},
        images_by_related_event={},
        images_by_request={},
        exchange_rate_lookup={},
        requests_by_id={},
    )


def response(evidence_id: str, quote: str, classification: str = "no_fact", **overrides) -> str:
    """Build a raw JSON response string with every field defaulted to
    null/safe values, then apply `overrides`."""
    d = {f: None for f in RESPONSE_FIELDS}
    d.update(
        evidence_id=evidence_id,
        classification=classification,
        confidence="high",
        supporting_quote=quote,
    )
    d.update(overrides)
    return json.dumps(d)


class FakeAIClient:
    """Returns canned responses in order. Raises if called more times
    than responses were supplied (catches accidental extra API calls,
    e.g. a cache miss that should have been a hit)."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        self.call_count = 0

    def complete(self, system_instructions: str, user_prompt: str) -> str:
        self.call_count += 1
        if not self._queue:
            raise AssertionError("FakeAIClient called more times than responses were queued")
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# validate_response - field-level rejections
# ---------------------------------------------------------------------------


class TestValidateResponseRejections(unittest.TestCase):
    def setUp(self):
        self.event = make_event(event_id="ev_1", amount="100", status="pending", currency="USD")
        self.message = make_message(
            message_id="msg_1", related_event_id="ev_1",
            text="Your USD 100 payment is being processed and may change.",
        )
        self.ctx = EvidenceContext(message=self.message, event=self.event)

    def test_valid_no_fact_response_is_accepted(self):
        raw = response("msg_1", "may change", classification="no_fact")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.fact)
        self.assertEqual(result.fact.fact_type, "no_fact")
        self.assertEqual(result.fact.resolution_method, "gemini")

    def test_malformed_json_is_rejected(self):
        result = validate_response("{not valid json", self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("malformed JSON", result.rejection_reason)

    def test_missing_required_field_is_rejected(self):
        d = json.loads(response("msg_1", "may change"))
        del d["confidence"]
        result = validate_response(json.dumps(d), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("missing required field", result.rejection_reason)

    def test_unexpected_field_is_rejected(self):
        d = json.loads(response("msg_1", "may change"))
        d["extra_field"] = "surprise"
        result = validate_response(json.dumps(d), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("unexpected field", result.rejection_reason)

    def test_wrong_evidence_id_is_rejected(self):
        raw = response("msg_WRONG", "may change")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid evidence_id", result.rejection_reason)

    def test_invalid_amount_type_is_rejected(self):
        raw = response("msg_1", "may change", classification="resolved_amount",
                        amount="a lot", currency="USD")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid amount", result.rejection_reason)

    def test_negative_amount_is_rejected(self):
        raw = response("msg_1", "may change", classification="resolved_amount",
                        amount=-50, currency="USD")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid amount", result.rejection_reason)

    def test_invalid_currency_is_rejected(self):
        raw = response("msg_1", "may change", classification="resolved_amount",
                        amount=100, currency="XYZ")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid currency", result.rejection_reason)

    def test_currency_contradicting_known_event_is_rejected(self):
        raw = response("msg_1", "may change", classification="status_correction",
                        currency="EUR", new_status="cancelled")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("contradicts", result.rejection_reason)

    def test_invalid_date_is_rejected(self):
        raw = response("msg_1", "may change", classification="future_amount_change",
                        amount=100, currency="USD", effective_date="not-a-date")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid effective_date", result.rejection_reason)

    def test_end_date_before_effective_date_is_rejected(self):
        raw = response("msg_1", "may change", classification="future_amount_change",
                        amount=100, currency="USD",
                        effective_date="2026-06-01", end_date="2026-01-01")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("precedes", result.rejection_reason)

    def test_invalid_confidence_is_rejected(self):
        d = json.loads(response("msg_1", "may change"))
        d["confidence"] = "very-high"
        result = validate_response(json.dumps(d), self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid confidence", result.rejection_reason)

    def test_ungrounded_supporting_quote_is_rejected(self):
        raw = response("msg_1", "this text is not in the message at all")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("ungrounded supporting_quote", result.rejection_reason)

    def test_ungrounded_amount_is_rejected(self):
        # "500" never appears in the source text.
        raw = response("msg_1", "may change", classification="resolved_amount",
                        amount=500, currency="USD")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("ungrounded amount", result.rejection_reason)

    def test_resolved_amount_on_event_with_known_amount_is_rejected(self):
        # self.event already has amount=100 (not blank) - resolved_amount
        # must never be used to override an already-known amount.
        raw = response("msg_1", "USD 100", classification="resolved_amount",
                        amount=100, currency="USD")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("already has a known amount", result.rejection_reason)

    def test_status_correction_to_same_status_is_rejected(self):
        raw = response("msg_1", "may change", classification="status_correction",
                        new_status="pending")  # event is already "pending"
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("does not actually change", result.rejection_reason)

    def test_invalid_new_status_is_rejected(self):
        raw = response("msg_1", "may change", classification="status_correction",
                        new_status="unrealized")
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("invalid new_status", result.rejection_reason)


class TestStandaloneSeriesGrounding(unittest.TestCase):
    """Standalone messages (no related_event_id) must ground any series-
    level fact against the user's OWN known (category, event_type,
    direction) combinations - never an invented one."""

    def setUp(self):
        self.events = [make_event(event_id="ev_salary", amount="3000", status="settled",
                                   category="salary", event_type="income", direction="credit")]
        self.message = make_message(
            message_id="msg_standalone", related_event_id=None,
            text="Your monthly pay is USD 3500 starting 2026-03-01.",
        )
        self.ctx = EvidenceContext(
            message=self.message, event=None,
            known_series_categories=(("salary", "income", "credit"),),
        )

    def test_future_amount_change_with_known_series_is_accepted(self):
        raw = response(
            "msg_standalone", "USD 3500", classification="future_amount_change",
            amount=3500, currency="USD", effective_date="2026-03-01",
            target_category="salary", target_event_type="income", target_direction="credit",
        )
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.fact.target_series_key, ("salary", "income", "credit"))
        self.assertEqual(result.fact.resolved_amount, Decimal("3500"))

    def test_future_amount_change_with_invented_series_is_rejected(self):
        raw = response(
            "msg_standalone", "USD 3500", classification="future_amount_change",
            amount=3500, currency="USD", effective_date="2026-03-01",
            target_category="bonus", target_event_type="income", target_direction="credit",
        )
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("not among this user's own known recurring series", result.rejection_reason)

    def test_future_amount_change_missing_target_fields_is_rejected(self):
        raw = response(
            "msg_standalone", "USD 3500", classification="future_amount_change",
            amount=3500, currency="USD", effective_date="2026-03-01",
        )
        result = validate_response(raw, self.ctx)
        self.assertTrue(result.is_rejected)
        self.assertIn("requires non-null target_category", result.rejection_reason)


class TestUnresolvedClassification(unittest.TestCase):
    def test_unresolved_classification_produces_no_fact_object(self):
        event = make_event(status="pending")
        message = make_message(text="Something ambiguous is happening with this payment.")
        ctx = EvidenceContext(message=message, event=event)
        raw = response(
            "msg_1", "Something ambiguous", classification="unresolved",
            negation_or_ambiguity_notes="cannot tell if this amends the event",
        )
        result = validate_response(raw, ctx)
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.fact)
        self.assertEqual(result.unresolved_reason, "cannot tell if this amends the event")


# ---------------------------------------------------------------------------
# resolve_message_with_ai - retry policy, caching
# ---------------------------------------------------------------------------


class TestResolveMessageWithAI(unittest.TestCase):
    def setUp(self):
        self.event = make_event(event_id="ev_1", amount="100", status="pending", currency="USD")
        self.indexes = make_indexes([self.event])

    def test_valid_first_response_resolves_in_one_call(self):
        message = make_message(text="Confirming your USD 100 payment is still pending, no change.")
        client = FakeAIClient([response("msg_1", "no change", classification="no_fact")])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertEqual(outcome.api_calls, 1)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "no_fact")

    def test_negated_future_charge_resolves_to_no_fact_not_a_charge(self):
        message = make_message(
            text="Please note: you will NOT be charged for this pending payment.",
        )
        client = FakeAIClient([
            response("msg_1", "will NOT be charged", classification="no_fact")
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "no_fact")
        self.assertIsNone(outcome.fact.resolved_amount)

    def test_cancellation_resolves_to_status_correction(self):
        message = make_message(text="Your payment has been cancelled and will not be retried.")
        client = FakeAIClient([
            response("msg_1", "has been cancelled", classification="status_correction",
                      new_status="cancelled")
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "status_correction")
        self.assertEqual(outcome.fact.new_status, "cancelled")

    def test_scam_prize_language_resolves_to_no_fact(self):
        message = make_message(
            message_id="msg_scam", related_event_id=None,
            text="Congratulations! Pay a release charge today to claim your cash prize.",
        )
        client = FakeAIClient([
            response("msg_scam", "Pay a release charge today", classification="no_fact")
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "no_fact")

    def test_future_amount_change_is_resolved(self):
        salary_event = make_event(event_id="ev_salary", amount="3000", status="settled",
                                   category="salary", event_type="income", direction="credit")
        indexes = make_indexes([salary_event])
        message = make_message(
            message_id="msg_standalone", related_event_id=None,
            text="Your monthly pay is USD 3500 starting 2026-03-01.",
        )
        client = FakeAIClient([
            response("msg_standalone", "USD 3500", classification="future_amount_change",
                      amount=3500, currency="USD", effective_date="2026-03-01",
                      target_category="salary", target_event_type="income", target_direction="credit")
        ])
        outcome = resolve_message_with_ai(message, indexes, client)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "future_amount_change")
        self.assertEqual(outcome.fact.target_series_key, ("salary", "income", "credit"))

    def test_ambiguous_evidence_resolves_to_unresolved(self):
        message = make_message(text="This payment situation is unclear and may or may not change.")
        client = FakeAIClient([
            response("msg_1", "unclear", classification="unresolved",
                      negation_or_ambiguity_notes="cannot tell what changed")
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertIsNone(outcome.fact)
        self.assertIsNotNone(outcome.unresolved)
        self.assertIn("model_unresolved", outcome.unresolved.reason)

    def test_retry_after_invalid_output_then_succeeds(self):
        message = make_message(text="Confirming your USD 100 payment is still pending, no change.")
        client = FakeAIClient([
            "{not valid json",  # first attempt: malformed
            response("msg_1", "no change", classification="no_fact"),  # retry: valid
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertEqual(outcome.api_calls, 2)
        self.assertIsNotNone(outcome.first_rejection_reason)
        self.assertIsNotNone(outcome.fact)
        self.assertEqual(outcome.fact.fact_type, "no_fact")

    def test_second_invalid_response_resolves_to_unresolved(self):
        message = make_message(text="Confirming your USD 100 payment is still pending, no change.")
        client = FakeAIClient([
            "{not valid json",
            "still not valid json",
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client)
        self.assertEqual(outcome.api_calls, 2)
        self.assertIsNone(outcome.fact)
        self.assertIsNotNone(outcome.unresolved)
        self.assertIn("invalid_model_output_after_retry", outcome.unresolved.reason)
        self.assertIsNotNone(outcome.second_rejection_reason)

    def test_successful_result_is_cached_and_reused(self):
        message = make_message(text="Confirming your USD 100 payment is still pending, no change.")
        cache = EvidenceCache(path=None)
        client = FakeAIClient([response("msg_1", "no change", classification="no_fact")])

        first = resolve_message_with_ai(message, self.indexes, client, cache)
        self.assertFalse(first.cache_hit)
        self.assertEqual(client.call_count, 1)

        second = resolve_message_with_ai(message, self.indexes, client, cache)
        self.assertTrue(second.cache_hit)
        self.assertEqual(client.call_count, 1)  # no additional API call was made
        self.assertEqual(second.fact.fact_type, "no_fact")

    def test_invalid_result_is_not_cached(self):
        message = make_message(text="Confirming your USD 100 payment is still pending, no change.")
        cache = EvidenceCache(path=None)
        client = FakeAIClient(["{not valid", "still not valid"])

        outcome = resolve_message_with_ai(message, self.indexes, client, cache)
        self.assertIsNone(outcome.fact)
        ctx = build_context(message, self.indexes)
        self.assertIsNone(cache.get(ctx))  # nothing was cached for a rejected result

    def test_unresolved_result_is_not_cached(self):
        message = make_message(text="This is genuinely ambiguous.")
        cache = EvidenceCache(path=None)
        client = FakeAIClient([
            response("msg_1", "genuinely ambiguous", classification="unresolved",
                      negation_or_ambiguity_notes="unclear")
        ])
        outcome = resolve_message_with_ai(message, self.indexes, client, cache)
        self.assertIsNone(outcome.fact)
        ctx = build_context(message, self.indexes)
        self.assertIsNone(cache.get(ctx))


if __name__ == "__main__":
    unittest.main()
