"""Tests for engine.reconciliation - Stage 2 status/duplicate/linkage rules."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from data.models import FinancialEvent
from engine.recurrence import detect_recurring_series
from engine.reconciliation import reconcile_user_events


@dataclass
class _FakeIndexes:
    """Minimal stand-in for data.indexes.Indexes - convert() only reads
    exchange_rate_lookup, and every test here uses same-currency events."""

    exchange_rate_lookup: dict = None

    def __post_init__(self):
        if self.exchange_rate_lookup is None:
            self.exchange_rate_lookup = {}


def make_event(
    event_id,
    user_id="user_1",
    event_type="expense",
    category="dining",
    direction="debit",
    amount="100",
    currency="INR",
    event_date=date(2026, 1, 1),
    settlement_date=None,
    status="settled",
    linked_event_id=None,
    flexibility="fixed",
    minimum_allowed_amount=None,
):
    return FinancialEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        description="test event",
        category=category,
        direction=direction,
        amount=None if amount is None else Decimal(amount),
        currency=currency,
        event_date=event_date,
        settlement_date=settlement_date if settlement_date is not None else event_date,
        status=status,
        linked_event_id=linked_event_id,
        flexibility=flexibility,
        minimum_allowed_amount=minimum_allowed_amount,
    )


def reconcile(events, request_date=date(2026, 2, 1), home_currency="INR"):
    events_by_id = {e.event_id: e for e in events}
    return reconcile_user_events(
        "user_1", events, request_date, events_by_id, home_currency, _FakeIndexes()
    )


class TestReconciliationRules(unittest.TestCase):
    def test_refund_pair_both_events_count(self):
        expense = make_event("e1", category="shopping", direction="debit", amount="500",
                              event_date=date(2026, 1, 5))
        refund = make_event("e2", event_type="refund", category="shopping", direction="credit",
                             amount="500", event_date=date(2026, 1, 10), linked_event_id="e1")
        state = reconcile([expense, refund])
        ids = {c.event_id for c in state.settled_history}
        self.assertEqual(ids, {"e1", "e2"})

    def test_cancelled_replacement_pair(self):
        cancelled = make_event("e1", status="cancelled", amount="200", event_date=date(2026, 1, 5))
        replacement = make_event("e2", status="settled", amount="210",
                                  event_date=date(2026, 1, 6), linked_event_id="e1")
        state = reconcile([cancelled, replacement])
        self.assertIn("e1", state.excluded_ids)
        self.assertEqual([c.event_id for c in state.settled_history], ["e2"])

    def test_failed_retry_pair(self):
        failed = make_event("e1", status="failed", amount="150", event_date=date(2026, 1, 5))
        retry = make_event("e2", status="settled", amount="150",
                            event_date=date(2026, 1, 6), linked_event_id="e1")
        state = reconcile([failed, retry])
        self.assertIn("e1", state.excluded_ids)
        self.assertEqual([c.event_id for c in state.settled_history], ["e2"])

    def test_investment_valuation_excluded(self):
        valuation = make_event(
            "e1", event_type="investment_valuation", category="investment",
            direction="non_cash", status="unrealized", amount="10000",
            event_date=date(2026, 1, 5), settlement_date=None,
        )
        state = reconcile([valuation])
        self.assertIn("e1", state.excluded_ids)
        self.assertEqual(state.settled_history, [])
        self.assertEqual(state.known_future, [])

    def test_settled_investment_sale_is_genuine_credit(self):
        sale = make_event("e1", event_type="investment_sale", category="investment",
                           direction="credit", amount="5000", event_date=date(2026, 1, 5))
        state = reconcile([sale])
        self.assertEqual(len(state.settled_history), 1)
        self.assertEqual(state.settled_history[0].direction, "credit")

    def test_duplicate_pending_charge_excluded(self):
        settled = make_event("e1", status="settled", amount="80", event_date=date(2026, 1, 20))
        duplicate = make_event("e2", status="pending", amount="80",
                                event_date=date(2026, 1, 20), linked_event_id="e1")
        state = reconcile([settled, duplicate])
        self.assertIn("e2", state.excluded_duplicate_ids)
        self.assertEqual([c.event_id for c in state.settled_history], ["e1"])
        self.assertEqual(state.known_future, [])

    def test_pending_credit_not_confirmed_income(self):
        pending_credit = make_event("e1", status="pending", direction="credit", amount="1000",
                                     event_date=date(2026, 2, 5))
        state = reconcile([pending_credit])
        self.assertIn("e1", state.excluded_ids)
        self.assertEqual(state.known_future, [])

    def test_pending_debit_is_known_future(self):
        pending_debit = make_event("e1", status="pending", direction="debit", amount="300",
                                    event_date=date(2026, 2, 10))
        state = reconcile([pending_debit])
        self.assertEqual([c.event_id for c in state.known_future], ["e1"])

    def test_scheduled_credit_is_known_future(self):
        scheduled_salary = make_event("e1", event_type="income", status="scheduled",
                                       category="salary", direction="credit", amount="50000",
                                       event_date=date(2026, 2, 15))
        state = reconcile([scheduled_salary])
        self.assertEqual([c.event_id for c in state.known_future], ["e1"])

    def test_blank_amount_remains_unresolved(self):
        blank = make_event("e1", amount=None, status="settled", event_date=date(2026, 1, 5))
        state = reconcile([blank])
        self.assertIn("e1", state.unresolved_blank_ids)
        self.assertEqual(state.settled_history, [])
        self.assertEqual(state.excluded_ids, [])
        self.assertEqual(state.excluded_duplicate_ids, [])

    def test_trailing_scheduled_blank_amount_does_not_establish_cadence(self):
        # Three genuine monthly settled rent payments...
        rents = [
            make_event(f"r{i}", category="rent", event_type="expense", direction="debit",
                       amount="1000", event_date=date(2026, m, 1))
            for i, m in enumerate([1, 2, 3], start=1)
        ]
        # ...plus a trailing SCHEDULED, BLANK-amount event in the same
        # category that must never affect cadence detection.
        trailing = make_event("r_future", category="rent", event_type="expense", direction="debit",
                               amount=None, status="scheduled", event_date=date(2026, 4, 1))
        state = reconcile(rents + [trailing], request_date=date(2026, 3, 15))

        self.assertIn("r_future", state.unresolved_blank_ids)
        self.assertNotIn("r_future", [c.event_id for c in state.settled_history])

        # Cadence still forms cleanly from the 3 real settled occurrences.
        series = detect_recurring_series(state.settled_history, "user_1")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].cadence, "monthly")
        self.assertEqual(len(series[0].occurrences), 3)


if __name__ == "__main__":
    unittest.main()
