"""Tests for planner.affordability - safe-amount computation, earliest safe
full-payment date, and the general debit simulator, all built directly on
`engine.forecast.build_forecast` (Stage 2) with hand-built synthetic
fixtures (matching the style of `test_forecast.py`)."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal

from engine.forecast import build_forecast
from engine.state import CashEvent, ReconciledState

from planner.affordability import (
    earliest_date_for_full_payment,
    is_safe_with_debits,
    safe_amount_no_changes,
    safe_amount_on,
    simulate_min_balance,
)


def cash(d, amount, category="rent", event_type="expense", direction="debit",
         flexibility="fixed", source="settled", event_id=None):
    return CashEvent(
        date=d, amount=Decimal(amount), direction=direction, category=category,
        event_type=event_type, flexibility=flexibility, source=source,
        event_id=event_id or f"e-{d.isoformat()}-{category}-{source}",
    )


class TestFullyAffordable(unittest.TestCase):
    """Case 1: a request with no future cash pressure is fully safe today."""

    def test_no_future_events_full_amount_is_safe(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("1000")
        )
        safe = safe_amount_no_changes(forecast, Decimal("5000"))
        self.assertEqual(safe, Decimal("5000"))
        self.assertEqual(
            earliest_date_for_full_payment(forecast, Decimal("5000")), request_date
        )


class TestExceedsImmediateSafeBalance(unittest.TestCase):
    """Case 2: the requested amount is larger than what is safe today."""

    def test_safe_amount_capped_below_requested(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("5000"), minimum_balance_to_keep=Decimal("1000")
        )
        # Only 4000 is ever safe (5000 - 1000 minimum), regardless of a
        # much larger request.
        safe = safe_amount_no_changes(forecast, Decimal("20000"))
        self.assertEqual(safe, Decimal("4000"))


class TestMinimumBalanceConstraint(unittest.TestCase):
    """Case 3: minimum_balance_to_keep directly bounds the safe amount."""

    def test_minimum_balance_reduces_headroom(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("9500")
        )
        safe = safe_amount_no_changes(forecast, Decimal("10000"))
        self.assertEqual(safe, Decimal("500"))


class TestFutureEssentialExpensePreventsImmediatePayment(unittest.TestCase):
    """Case 4: a large future known expense - not visible in today's balance
    alone - reduces how much is safe to pay today."""

    def test_future_known_future_debit_caps_safe_amount(self):
        request_date = date(2026, 3, 1)
        big_future_rent = cash(
            date(2026, 3, 20), "9000", category="rent", source="known_future",
            event_id="future_rent",
        )
        state = ReconciledState(
            user_id="u1", request_date=request_date, known_future=[big_future_rent]
        )
        forecast = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("500")
        )
        # Balance after the future rent: 10000 - 9000 = 1000. Safe headroom
        # over the whole horizon is min(10000, 1000) - 500 = 500, even
        # though today's balance alone (10000) looks fine.
        safe = safe_amount_no_changes(forecast, Decimal("2000"))
        self.assertEqual(safe, Decimal("500"))

    def test_earliest_date_for_full_payment_is_after_the_future_expense(self):
        request_date = date(2026, 3, 1)
        # A future debit, then future income big enough to cover the request.
        big_future_rent = cash(
            date(2026, 3, 10), "8000", category="rent", source="known_future",
            event_id="future_rent",
        )
        future_income = cash(
            date(2026, 3, 15), "8000", category="salary", direction="credit",
            source="known_future", event_id="future_salary",
        )
        state = ReconciledState(
            user_id="u1", request_date=request_date,
            known_future=[big_future_rent, future_income],
        )
        forecast = build_forecast(
            state, starting_balance=Decimal("9000"), minimum_balance_to_keep=Decimal("500")
        )
        # Not safe today (9000 - 8500 min-forward = 500 < 3000 requested).
        self.assertLess(
            safe_amount_no_changes(forecast, Decimal("3000")), Decimal("3000")
        )
        # But safe once the salary lands on the 15th.
        earliest = earliest_date_for_full_payment(forecast, Decimal("3000"))
        self.assertEqual(earliest, date(2026, 3, 15))


class TestSimulateMinBalanceAndSafety(unittest.TestCase):
    def test_extra_debit_reduces_every_subsequent_checkpoint(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("1000"), minimum_balance_to_keep=Decimal("0")
        )
        self.assertTrue(is_safe_with_debits(forecast, [(request_date, Decimal("1000"))]))
        self.assertFalse(is_safe_with_debits(forecast, [(request_date, Decimal("1000.01"))]))

    def test_multiple_debits_at_different_dates(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("1000"), minimum_balance_to_keep=Decimal("100")
        )
        debits = [(request_date, Decimal("500")), (request_date + timedelta(days=10), Decimal("400"))]
        self.assertTrue(is_safe_with_debits(forecast, debits))
        min_balance = simulate_min_balance(forecast, debits)
        self.assertEqual(min_balance, Decimal("100"))


class TestNoFullPaymentSafeWithinHorizon(unittest.TestCase):
    def test_earliest_date_is_none_when_never_safe(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("100"), minimum_balance_to_keep=Decimal("50")
        )
        self.assertIsNone(earliest_date_for_full_payment(forecast, Decimal("1000000")))


class TestDecimalArithmeticOnly(unittest.TestCase):
    """Case 20: every planner money computation stays in Decimal - never
    float - so a 90-day forecast never accumulates binary rounding error."""

    def test_results_are_decimal_not_float(self):
        request_date = date(2026, 3, 1)
        state = ReconciledState(user_id="u1", request_date=request_date)
        forecast = build_forecast(
            state, starting_balance=Decimal("1000.33"), minimum_balance_to_keep=Decimal("100.11")
        )
        safe = safe_amount_no_changes(forecast, Decimal("500.5"))
        self.assertIsInstance(safe, Decimal)
        earliest = earliest_date_for_full_payment(forecast, Decimal("500.5"))
        self.assertEqual(earliest, request_date)
        min_balance = simulate_min_balance(forecast, [(request_date, Decimal("1.5"))])
        self.assertIsInstance(min_balance, Decimal)


if __name__ == "__main__":
    unittest.main()
