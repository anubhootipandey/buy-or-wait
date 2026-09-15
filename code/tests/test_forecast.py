"""Tests for engine.forecast - balance anchoring, chronological processing,
and the Rule 7 known-future-overrides-generated-recurrence guarantee."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from engine.forecast import build_forecast
from engine.state import CashEvent, ReconciledState


def cash(d, amount, category="rent", event_type="expense", direction="debit",
         flexibility="fixed", source="settled", event_id=None):
    return CashEvent(
        date=d, amount=Decimal(amount), direction=direction, category=category,
        event_type=event_type, flexibility=flexibility, source=source,
        event_id=event_id or f"e-{d.isoformat()}-{category}-{source}",
    )


class TestRequestDateBalanceAnchoring(unittest.TestCase):
    def test_starting_balance_excludes_request_date_event_until_processed(self):
        request_date = date(2026, 3, 1)
        # An expense dated exactly on request_date must be applied WITHIN
        # the forecast, never pre-subtracted from the starting balance.
        on_request_date = cash(request_date, "200", source="known_future")
        state = ReconciledState(
            user_id="user_1", request_date=request_date,
            settled_history=[], known_future=[on_request_date],
        )
        result = build_forecast(state, starting_balance=Decimal("1000"),
                                 minimum_balance_to_keep=Decimal("0"))
        self.assertEqual(len(result.checkpoints), 1)
        self.assertEqual(result.checkpoints[0].balance_after, Decimal("800"))
        # Before any event is applied, the balance is exactly the starting
        # balance - the "before applying any event on request_date" rule.
        self.assertEqual(result.balance_on(date(2026, 2, 28)), Decimal("1000"))


class TestEmptyStateIsNoOp(unittest.TestCase):
    def test_no_events_leaves_balance_flat(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="user_1", request_date=request_date)
        result = build_forecast(state, starting_balance=Decimal("5000"),
                                 minimum_balance_to_keep=Decimal("1000"))
        self.assertEqual(result.checkpoints, [])
        self.assertEqual(result.min_balance_reached, Decimal("5000"))
        self.assertFalse(result.breaches_minimum())
        self.assertEqual(result.balance_on(request_date + __import__("datetime").timedelta(days=90)),
                          Decimal("5000"))


class TestKnownFutureOverridesGeneratedRecurrence(unittest.TestCase):
    """Regression test for Rule 7: a known future event that matches an
    already-detected recurring series must replace the generated occurrence
    - never both, never neither."""

    def test_known_future_rent_replaces_generated_occurrence_no_duplication(self):
        request_date = date(2026, 3, 15)
        settled_history = [
            cash(date(2026, 1, 1), "1000"),
            cash(date(2026, 2, 1), "1000"),
            cash(date(2026, 3, 1), "1000"),
        ]
        # The landlord already scheduled next month's rent, and it went UP.
        known_rent = cash(date(2026, 4, 1), "1100", source="known_future", event_id="rent_known_apr")
        state = ReconciledState(
            user_id="user_1", request_date=request_date,
            settled_history=settled_history, known_future=[known_rent],
        )
        result = build_forecast(state, starting_balance=Decimal("10000"),
                                 minimum_balance_to_keep=Decimal("0"), forecast_days=90)

        self.assertEqual(len(result.recurring_series), 1)
        self.assertEqual(result.overridden_generated_count, 1)

        april_rent_events = [cp.event for cp in result.checkpoints if cp.date == date(2026, 4, 1)]
        self.assertEqual(len(april_rent_events), 1, "must be exactly one cash movement for April rent")
        self.assertEqual(april_rent_events[0].amount, Decimal("1100"))
        self.assertEqual(april_rent_events[0].event_id, "rent_known_apr")

        # No May-onward generated rent should still be there? Actually May
        # rent has no known future counterpart, so it must still be
        # generated at the last-known amount (1000, per Rule "fixed
        # category projects at last settled amount").
        may_rent_events = [cp.event for cp in result.checkpoints if cp.date == date(2026, 5, 1)]
        self.assertEqual(len(may_rent_events), 1)
        self.assertEqual(may_rent_events[0].amount, Decimal("1000"))
        self.assertEqual(may_rent_events[0].source, "recurring")


class TestChronologicalProcessingAndMinimumBalanceCheck(unittest.TestCase):
    def test_breach_detected_mid_forecast(self):
        request_date = date(2026, 1, 1)
        known_future = [
            cash(date(2026, 1, 10), "9500", source="known_future", event_id="big_bill"),
        ]
        state = ReconciledState(
            user_id="user_1", request_date=request_date,
            settled_history=[], known_future=known_future,
        )
        result = build_forecast(state, starting_balance=Decimal("10000"),
                                 minimum_balance_to_keep=Decimal("1000"))
        self.assertTrue(result.breaches_minimum())
        self.assertEqual(result.min_balance_reached, Decimal("500"))


if __name__ == "__main__":
    unittest.main()
