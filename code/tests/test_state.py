"""Tests for engine.state - CashEvent sign handling and ReconciledState
defaults (empty normalized adjustments are a no-op)."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from engine.state import CashEvent, ReconciledState


class TestCashEventSignedAmount(unittest.TestCase):
    def test_credit_is_positive(self):
        e = CashEvent(date=date(2026, 1, 1), amount=Decimal("100"), direction="credit",
                       category="salary", event_type="income", flexibility="fixed", source="settled")
        self.assertEqual(e.signed_amount, Decimal("100"))

    def test_debit_is_negative(self):
        e = CashEvent(date=date(2026, 1, 1), amount=Decimal("100"), direction="debit",
                       category="rent", event_type="expense", flexibility="fixed", source="settled")
        self.assertEqual(e.signed_amount, Decimal("-100"))


class TestEmptyReconciledStateIsNoOp(unittest.TestCase):
    def test_default_state_has_no_adjustments(self):
        state = ReconciledState(user_id="user_1", request_date=date(2026, 1, 1))
        self.assertEqual(state.settled_history, [])
        self.assertEqual(state.known_future, [])
        self.assertEqual(state.excluded_duplicate_ids, [])
        self.assertEqual(state.unresolved_blank_ids, [])
        self.assertEqual(state.excluded_ids, [])


if __name__ == "__main__":
    unittest.main()
