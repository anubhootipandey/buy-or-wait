"""Tests for evidence.models and evidence.deterministic - the Stage 4a
deterministic pre-filter. No model/API calls anywhere in this file."""

from __future__ import annotations

import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from data.loader import load_dataset
from data.indexes import build_indexes
from data.models import Dataset, FinancialEvent, ImageRecord, Message

from evidence.deterministic import (
    DETERMINISTIC_FACT,
    MISSING_EVIDENCE,
    NEEDS_ESCALATION,
    NO_FACT,
    UNRESOLVED_OTHER,
    PrefilterDecision,
    classify_image,
    classify_message,
    run_prefilter,
)
from evidence.models import EvidenceRef, NormalizedFact


def make_event(event_id="ev_1", amount="100", status="settled", currency="USD", **overrides):
    defaults = dict(
        event_id=event_id, user_id="user_1", event_type="expense", description="test event",
        category="dining", direction="debit", amount=Decimal(amount) if amount is not None else None,
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


def make_image(image_id="img_1", related_event_id="ev_1", file_path="/nonexistent/path.png", **overrides):
    defaults = dict(
        image_id=image_id, user_id="user_1", request_id=None,
        related_event_id=related_event_id, file_path=file_path,
    )
    defaults.update(overrides)
    return ImageRecord(**defaults)


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

class TestEvidenceRef(unittest.TestCase):
    def test_requires_exactly_one_of_message_id_or_image_id(self):
        with self.assertRaises(ValueError):
            EvidenceRef(user_id="user_1")  # neither set
        with self.assertRaises(ValueError):
            EvidenceRef(user_id="user_1", message_id="m1", image_id="i1")  # both set

    def test_message_only_is_valid(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        self.assertEqual(ref.message_id, "m1")
        self.assertIsNone(ref.image_id)

    def test_image_only_is_valid(self):
        ref = EvidenceRef(user_id="user_1", image_id="i1")
        self.assertEqual(ref.image_id, "i1")


class TestNormalizedFact(unittest.TestCase):
    def test_rejects_unknown_fact_type(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        with self.assertRaises(ValueError):
            NormalizedFact(fact_id="f1", user_id="user_1", fact_type="not_a_real_type", provenance=ref)

    def test_rejects_unknown_resolution_method(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        with self.assertRaises(ValueError):
            NormalizedFact(
                fact_id="f1", user_id="user_1", fact_type="no_fact", provenance=ref,
                resolution_method="magic",
            )

    def test_valid_fact_constructs(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        fact = NormalizedFact(fact_id="f1", user_id="user_1", fact_type="no_fact", provenance=ref)
        self.assertEqual(fact.fact_type, "no_fact")


# ---------------------------------------------------------------------------
# deterministic.py - classify_message
# ---------------------------------------------------------------------------

class TestClosedStatusIsNoFact(unittest.TestCase):
    def test_settled_known_amount_is_no_fact(self):
        events = {"ev_1": make_event(status="settled", amount="100")}
        decision = classify_message(make_message(), events)
        self.assertEqual(decision.bucket, NO_FACT)
        self.assertIsNotNone(decision.fact)
        self.assertEqual(decision.fact.fact_type, "no_fact")

    def test_cancelled_known_amount_is_no_fact(self):
        events = {"ev_1": make_event(status="cancelled", amount="100")}
        decision = classify_message(make_message(), events)
        self.assertEqual(decision.bucket, NO_FACT)

    def test_failed_known_amount_is_no_fact(self):
        events = {"ev_1": make_event(status="failed", amount="100")}
        decision = classify_message(make_message(), events)
        self.assertEqual(decision.bucket, NO_FACT)


class TestFailedEventWithLiveForwardLinkEscalates(unittest.TestCase):
    """Regression coverage for the message_69/179/198/201 bug: a `failed`
    event whose retry is recorded as another event that links BACK at it
    (the dataset's convention) must escalate, not be filed as `no_fact`,
    because that retry can still affect the future cash forecast."""

    def test_failed_with_scheduled_retry_linking_back_needs_escalation(self):
        events = {
            "ev_1": make_event(event_id="ev_1", status="failed", amount="166"),
            "ev_2": make_event(
                event_id="ev_2", status="scheduled", amount="166", linked_event_id="ev_1"
            ),
        }
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)
        self.assertIsNone(decision.fact)

    def test_failed_with_pending_retry_linking_back_needs_escalation(self):
        events = {
            "ev_1": make_event(event_id="ev_1", status="failed", amount="73"),
            "ev_2": make_event(
                event_id="ev_2", status="pending", amount="73", linked_event_id="ev_1"
            ),
        }
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)

    def test_failed_with_settled_linked_event_stays_no_fact(self):
        """A failed event linked to by an already-SETTLED event (e.g. a
        corrective settled transaction unrelated to a future retry) is
        genuinely inert - only a live pending/scheduled link escalates."""
        events = {
            "ev_1": make_event(event_id="ev_1", status="failed", amount="100"),
            "ev_2": make_event(
                event_id="ev_2", status="settled", amount="100", linked_event_id="ev_1"
            ),
        }
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NO_FACT)

    def test_failed_with_no_linking_event_stays_no_fact(self):
        """A genuinely closed failed event (no retry anywhere in the
        dataset) keeps the original, correct no_fact behavior."""
        events = {"ev_1": make_event(event_id="ev_1", status="failed", amount="100")}
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NO_FACT)

    def test_settled_event_with_live_linked_event_is_unaffected(self):
        """The exception is scoped to status='failed' only - a settled or
        cancelled event stays no_fact even if something live happens to
        link back to it, since only failed->retry is a known dataset
        pattern with a real forward cash-flow effect."""
        events = {
            "ev_1": make_event(event_id="ev_1", status="settled", amount="100"),
            "ev_2": make_event(
                event_id="ev_2", status="scheduled", amount="100", linked_event_id="ev_1"
            ),
        }
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NO_FACT)

    def test_cancelled_event_with_live_linked_event_is_unaffected(self):
        events = {
            "ev_1": make_event(event_id="ev_1", status="cancelled", amount="100"),
            "ev_2": make_event(
                event_id="ev_2", status="scheduled", amount="100", linked_event_id="ev_1"
            ),
        }
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, NO_FACT)


class TestLiveStatusEscalates(unittest.TestCase):
    def test_pending_known_amount_needs_escalation(self):
        events = {"ev_1": make_event(status="pending", amount="100")}
        decision = classify_message(make_message(), events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)
        self.assertIsNone(decision.fact)

    def test_scheduled_known_amount_needs_escalation(self):
        events = {"ev_1": make_event(status="scheduled", amount="100")}
        decision = classify_message(make_message(), events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)


class TestStandaloneMessageEscalates(unittest.TestCase):
    def test_no_related_event_id_needs_escalation(self):
        events = {}
        decision = classify_message(make_message(related_event_id=None), events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)
        self.assertEqual(decision.ref.related_event_id, None)


class TestDanglingReferenceIsMissingEvidence(unittest.TestCase):
    def test_related_event_not_in_events_by_id(self):
        events = {}  # ev_1 doesn't exist
        decision = classify_message(make_message(related_event_id="ev_1"), events)
        self.assertEqual(decision.bucket, MISSING_EVIDENCE)


class TestBlankAmountDeterministicExtraction(unittest.TestCase):
    def test_single_unambiguous_amount_is_deterministic_fact(self):
        events = {"ev_1": make_event(amount=None, currency="USD")}
        msg = make_message(text="Your payment of USD 245.50 was received.")
        decision = classify_message(msg, events)
        self.assertEqual(decision.bucket, DETERMINISTIC_FACT)
        self.assertEqual(decision.fact.resolved_amount, Decimal("245.50"))
        self.assertEqual(decision.fact.currency, "USD")
        self.assertEqual(decision.fact.fact_type, "resolved_amount")
        self.assertEqual(decision.fact.resolution_method, "deterministic")

    def test_two_amounts_in_text_is_ambiguous_and_escalates(self):
        events = {"ev_1": make_event(amount=None, currency="USD")}
        msg = make_message(text="Total USD 500 received; balance due USD 245.50 remains.")
        decision = classify_message(msg, events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)
        self.assertIsNone(decision.fact)

    def test_wrong_currency_match_escalates(self):
        events = {"ev_1": make_event(amount=None, currency="USD")}
        msg = make_message(text="Your payment of EUR 245.50 was received.")
        decision = classify_message(msg, events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)

    def test_no_amount_mentioned_escalates(self):
        events = {"ev_1": make_event(amount=None, currency="USD")}
        msg = make_message(text="Your receipt shows the final amount for this transaction.")
        decision = classify_message(msg, events)
        self.assertEqual(decision.bucket, NEEDS_ESCALATION)

    def test_thousands_separator_is_parsed(self):
        events = {"ev_1": make_event(amount=None, currency="IDR")}
        msg = make_message(text="Amount charged: IDR 4,500,000 today.")
        decision = classify_message(msg, events)
        self.assertEqual(decision.bucket, DETERMINISTIC_FACT)
        self.assertEqual(decision.fact.resolved_amount, Decimal("4500000"))


# ---------------------------------------------------------------------------
# deterministic.py - classify_image
# ---------------------------------------------------------------------------

class TestImageMissingFile(unittest.TestCase):
    def test_file_not_on_disk_is_missing_evidence(self):
        img = make_image(file_path="/definitely/does/not/exist.png")
        decision = classify_image(img, {})
        self.assertEqual(decision.bucket, MISSING_EVIDENCE)


class TestImageNoRelatedEvent(unittest.TestCase):
    def test_no_related_event_id_is_unresolved_other(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            img = make_image(related_event_id=None, file_path=f.name)
            decision = classify_image(img, {})
            self.assertEqual(decision.bucket, UNRESOLVED_OTHER)


class TestImageDanglingReference(unittest.TestCase):
    def test_related_event_not_found_is_missing_evidence(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            img = make_image(related_event_id="ev_missing", file_path=f.name)
            decision = classify_image(img, {})
            self.assertEqual(decision.bucket, MISSING_EVIDENCE)


class TestImageKnownAmountIsNoFact(unittest.TestCase):
    def test_amount_already_known_is_no_fact(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            events = {"ev_1": make_event(amount="100")}
            img = make_image(file_path=f.name)
            decision = classify_image(img, events)
            self.assertEqual(decision.bucket, NO_FACT)
            self.assertIsNotNone(decision.fact)


class TestImageFailedEventWithLiveForwardLinkEscalates(unittest.TestCase):
    def test_failed_with_scheduled_retry_linking_back_needs_escalation(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            events = {
                "ev_1": make_event(event_id="ev_1", status="failed", amount="166"),
                "ev_2": make_event(
                    event_id="ev_2", status="scheduled", amount="166", linked_event_id="ev_1"
                ),
            }
            img = make_image(related_event_id="ev_1", file_path=f.name)
            decision = classify_image(img, events)
            self.assertEqual(decision.bucket, NEEDS_ESCALATION)
            self.assertIsNone(decision.fact)


class TestImageBlankAmountEscalates(unittest.TestCase):
    def test_blank_amount_with_file_present_needs_escalation(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            events = {"ev_1": make_event(amount=None)}
            img = make_image(file_path=f.name)
            decision = classify_image(img, events)
            self.assertEqual(decision.bucket, NEEDS_ESCALATION)
            self.assertIsNone(decision.fact)


# ---------------------------------------------------------------------------
# PrefilterDecision validation + PrefilterReport
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Real-dataset regression: message_69/179/198/201, the exact four messages
# reported as misclassified against event_8575/21101/23306/23855.
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"

_FIXED_MESSAGE_IDS = {
    "message_69": "event_8575",
    "message_179": "event_21101",
    "message_198": "event_23306",
    "message_201": "event_23855",
}


class TestRealDatasetFailedRetryMessagesEscalate(unittest.TestCase):
    """The exact four messages named in the bug report must now escalate
    instead of being filed as no_fact, and every other message's bucket
    must be unaffected (no collateral changes)."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATASET_DIR)
        cls.indexes = build_indexes(cls.dataset)

    def test_named_messages_now_need_escalation(self):
        for message in self.dataset.messages:
            if message.message_id in _FIXED_MESSAGE_IDS:
                self.assertEqual(message.related_event_id, _FIXED_MESSAGE_IDS[message.message_id])
                decision = classify_message(message, self.indexes.events_by_id)
                self.assertEqual(
                    decision.bucket, NEEDS_ESCALATION,
                    f"{message.message_id} (related to {message.related_event_id}) "
                    f"should escalate, got {decision.bucket!r}",
                )
                self.assertIsNone(decision.fact)

    def test_underlying_events_are_failed_with_a_live_scheduled_retry(self):
        for event_id in _FIXED_MESSAGE_IDS.values():
            event = self.indexes.events_by_id[event_id]
            self.assertEqual(event.status, "failed")
            retries = [
                e for e in self.indexes.events_by_id.values()
                if e.linked_event_id == event_id
            ]
            self.assertTrue(
                any(r.status == "scheduled" for r in retries),
                f"{event_id} should have a scheduled retry linking back to it",
            )

    def test_total_evidence_count_is_231_and_bucket_counts_shift_by_exactly_four(self):
        report = run_prefilter(self.dataset, self.indexes.events_by_id)
        counts = report.counts()
        self.assertEqual(sum(counts.values()), 231)
        self.assertEqual(len(self.dataset.messages), 215)
        self.assertEqual(len(self.dataset.images), 16)
        # Before this fix: no_fact=16, needs_escalation=215. The four
        # named messages move from no_fact -> needs_escalation and
        # nothing else changes.
        self.assertEqual(counts[NO_FACT], 12)
        self.assertEqual(counts[NEEDS_ESCALATION], 219)
        self.assertEqual(counts[DETERMINISTIC_FACT], 0)
        self.assertEqual(counts[MISSING_EVIDENCE], 0)
        self.assertEqual(counts[UNRESOLVED_OTHER], 0)


class TestPrefilterDecisionValidation(unittest.TestCase):
    def test_rejects_unknown_bucket(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        with self.assertRaises(ValueError):
            PrefilterDecision(ref=ref, bucket="not_a_bucket", reason="x")

    def test_rejects_fact_attached_to_non_fact_bucket(self):
        ref = EvidenceRef(user_id="user_1", message_id="m1")
        fact = NormalizedFact(fact_id="f1", user_id="user_1", fact_type="no_fact", provenance=ref)
        with self.assertRaises(ValueError):
            PrefilterDecision(ref=ref, bucket=NEEDS_ESCALATION, reason="x", fact=fact)


class TestRunPrefilterAndReport(unittest.TestCase):
    def test_counts_and_by_bucket_over_a_small_synthetic_dataset(self):
        events_by_id = {
            "ev_closed": make_event(event_id="ev_closed", status="settled", amount="50"),
            "ev_live": make_event(event_id="ev_live", status="pending", amount="50"),
            "ev_blank": make_event(event_id="ev_blank", amount=None, currency="USD"),
        }
        messages = [
            make_message(message_id="m_closed", related_event_id="ev_closed", text="closed"),
            make_message(message_id="m_live", related_event_id="ev_live", text="live"),
            make_message(message_id="m_standalone", related_event_id=None, text="standalone"),
            make_message(
                message_id="m_extract", related_event_id="ev_blank",
                text="Charged USD 10.00 today.",
            ),
        ]
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            images = [make_image(image_id="img_1", related_event_id="ev_blank", file_path=f.name)]
            dataset = Dataset(
                profiles=[], events=list(events_by_id.values()), exchange_rates=[],
                payment_options=[], messages=messages, images=images, requests=[], sample_requests=[],
            )
            report = run_prefilter(dataset, events_by_id)

        counts = report.counts()
        self.assertEqual(counts[NO_FACT], 1)
        self.assertEqual(counts[NEEDS_ESCALATION], 3)  # live message + standalone message + image
        self.assertEqual(counts[DETERMINISTIC_FACT], 1)
        self.assertEqual(sum(counts.values()), len(messages) + len(images))
        self.assertEqual(len(report.by_bucket(NO_FACT)), 1)

    def test_by_bucket_rejects_unknown_bucket_name(self):
        from evidence.deterministic import PrefilterReport
        report = PrefilterReport(())
        with self.assertRaises(ValueError):
            report.by_bucket("nonsense")


if __name__ == "__main__":
    unittest.main()