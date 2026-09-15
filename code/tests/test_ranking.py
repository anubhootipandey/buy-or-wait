"""Tests for planner.ranking - the explicit lexicographic ranking order
(deadline completion, no spending changes, minimize total paid, start
earlier, fewer payments, lowest payment_option_id)."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from planner.models import PlanCandidate, PlannedPayment, SpendingChangeAction
from planner.ranking import choose_best


def payment(d, amount):
    return PlannedPayment(date=d, amount=Decimal(amount))


class TestNoSpendingChangeBeatsSpendingChange(unittest.TestCase):
    """Case 15: when both a no-change plan and a spending-change plan
    would work, the no-change plan must rank first."""

    def test_no_change_plan_wins(self):
        no_change = PlanCandidate(
            method="full_payment", affordability_status="affordable_now",
            payments=[payment(date(2026, 3, 1), "1000")], spending_changes=[],
        )
        with_change = PlanCandidate(
            method="full_payment", affordability_status="affordable_with_plan",
            payments=[payment(date(2026, 3, 1), "1000")],
            spending_changes=[
                SpendingChangeAction(
                    kind="stop", event_id="ev1", category="streaming",
                    series_key=("streaming", "subscription", "debit"),
                    original_amount=Decimal("20"), adjusted_amount=Decimal("0"), severity=1,
                )
            ],
        )
        chosen = choose_best([with_change, no_change], date(2026, 4, 1))
        self.assertIs(chosen, no_change)


class TestEarlierCompletionRanksHigher(unittest.TestCase):
    """Case 16: among otherwise-equal candidates, the one that starts (and
    completes) earlier wins."""

    def test_earlier_start_date_wins(self):
        earlier = PlanCandidate(
            method="full_payment", affordability_status="affordable_later",
            payments=[payment(date(2026, 3, 5), "1000")],
        )
        later = PlanCandidate(
            method="full_payment", affordability_status="affordable_later",
            payments=[payment(date(2026, 3, 20), "1000")],
        )
        chosen = choose_best([later, earlier], date(2026, 4, 1))
        self.assertIs(chosen, earlier)


class TestFewerPaymentsRanking(unittest.TestCase):
    """Case 17: with total paid and start date tied, fewer payments wins."""

    def test_fewer_payments_wins(self):
        two_payments = PlanCandidate(
            method="installments", affordability_status="affordable_with_plan",
            payments=[payment(date(2026, 3, 1), "500"), payment(date(2026, 3, 31), "500")],
            payment_option_id="payment_option_02",
        )
        one_payment = PlanCandidate(
            method="full_payment", affordability_status="affordable_now",
            payments=[payment(date(2026, 3, 1), "1000")],
        )
        chosen = choose_best([two_payments, one_payment], date(2026, 4, 1))
        self.assertIs(chosen, one_payment)


class TestLowerPaymentOptionIdTieBreak(unittest.TestCase):
    """Case 18: when everything else ties, the lower payment_option_id
    wins."""

    def test_lower_payment_option_id_wins(self):
        option_5 = PlanCandidate(
            method="installments", affordability_status="affordable_with_plan",
            payments=[payment(date(2026, 3, 1), "500"), payment(date(2026, 3, 31), "500")],
            payment_option_id="payment_option_05",
        )
        option_2 = PlanCandidate(
            method="installments", affordability_status="affordable_with_plan",
            payments=[payment(date(2026, 3, 1), "500"), payment(date(2026, 3, 31), "500")],
            payment_option_id="payment_option_02",
        )
        chosen = choose_best([option_5, option_2], date(2026, 4, 1))
        self.assertIs(chosen, option_2)


class TestMinimizeTotalAmountPaid(unittest.TestCase):
    def test_cheaper_total_wins_over_installments_with_fees(self):
        full_payment = PlanCandidate(
            method="full_payment", affordability_status="affordable_now",
            payments=[payment(date(2026, 3, 1), "1000")],
        )
        installments_with_fee = PlanCandidate(
            method="installments", affordability_status="affordable_with_plan",
            payments=[payment(date(2026, 3, 1), "550"), payment(date(2026, 3, 31), "550")],
            payment_option_id="payment_option_01",
        )
        chosen = choose_best([installments_with_fee, full_payment], date(2026, 4, 1))
        self.assertIs(chosen, full_payment)


class TestEmptyCandidateList(unittest.TestCase):
    def test_returns_none_for_no_candidates(self):
        self.assertIsNone(choose_best([], date(2026, 4, 1)))


if __name__ == "__main__":
    unittest.main()
