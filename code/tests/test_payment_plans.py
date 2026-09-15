"""Tests for planner.payment_plans - installment and partial-payment
candidate construction and eligibility filtering."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from data.models import FinancialProfile, PaymentOption, Request
from engine.forecast import build_forecast
from engine.state import CashEvent, ReconciledState

from planner.affordability import safe_amount_no_changes, earliest_date_for_full_payment
from planner.payment_plans import (
    build_installment_candidates,
    build_partial_payment_candidate,
    build_wait_candidate,
)


def make_profile(**overrides) -> FinancialProfile:
    defaults = dict(
        user_id="user_1",
        home_currency="USD",
        current_available_balance=Decimal("10000"),
        minimum_balance_to_keep=Decimal("1000"),
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment", "partial_payment", "installments"),
        max_installment_months=12,
    )
    defaults.update(overrides)
    return FinancialProfile(**defaults)


def make_request(**overrides) -> Request:
    defaults = dict(
        request_id="request_1",
        user_id="user_1",
        request_date=date(2026, 3, 1),
        request_type="purchase",
        requested_amount=Decimal("3000"),
        desired_completion_date=date(2026, 4, 1),
        allows_partial_payment=True,
        request_text="",
    )
    defaults.update(overrides)
    return Request(**defaults)


def make_option(**overrides) -> PaymentOption:
    defaults = dict(
        payment_option_id="payment_option_01",
        request_id="request_1",
        payment_method="installments",
        payment_amount=Decimal("1000"),
        number_of_payments=3,
        first_payment_date=date(2026, 3, 1),
        payment_frequency_days=30,
        financing_fee=Decimal("0"),
        total_payable_amount=Decimal("3000"),
    )
    defaults.update(overrides)
    return PaymentOption(**defaults)


def flat_forecast(starting_balance, minimum_balance_to_keep, request_date=date(2026, 3, 1)):
    state = ReconciledState(user_id="user_1", request_date=request_date)
    return build_forecast(
        state, starting_balance=starting_balance, minimum_balance_to_keep=minimum_balance_to_keep
    )


class TestInstallmentEverySafe(unittest.TestCase):
    """Case 5: every installment of a supplied option is safe."""

    def test_all_three_installments_safe(self):
        request = make_request(desired_completion_date=date(2026, 5, 1))
        profile = make_profile()
        forecast = flat_forecast(Decimal("10000"), Decimal("1000"))
        option = make_option()
        candidates = build_installment_candidates(request, profile, [option], forecast)
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertEqual(cand.method, "installments")
        self.assertEqual(len(cand.payments), 3)
        self.assertEqual(cand.payment_option_id, "payment_option_01")


class TestInstallmentLaterBreach(unittest.TestCase):
    """Case 6: a later installment breaches minimum_balance_to_keep."""

    def test_third_installment_breaches_minimum(self):
        request = make_request(desired_completion_date=date(2026, 6, 1))
        profile = make_profile()
        # Balance only 2900 - after two 1000 installments only 900 left,
        # already below the 1000 minimum before the third payment even
        # lands.
        forecast = flat_forecast(Decimal("2900"), Decimal("1000"))
        option = make_option(number_of_payments=3, payment_amount=Decimal("1000"))
        candidates = build_installment_candidates(request, profile, [option], forecast)
        self.assertEqual(candidates, [])


class TestInstallmentExceedsMaxMonths(unittest.TestCase):
    def test_option_rejected_when_exceeding_max_installment_months(self):
        request = make_request(desired_completion_date=date(2027, 1, 1))
        profile = make_profile(max_installment_months=2)
        forecast = flat_forecast(Decimal("100000"), Decimal("0"))
        option = make_option(number_of_payments=3)
        candidates = build_installment_candidates(request, profile, [option], forecast)
        self.assertEqual(candidates, [])

    def test_installments_rejected_when_not_in_accepted_methods(self):
        request = make_request()
        profile = make_profile(payment_methods_user_will_consider=("full_payment",))
        forecast = flat_forecast(Decimal("100000"), Decimal("0"))
        option = make_option()
        candidates = build_installment_candidates(request, profile, [option], forecast)
        self.assertEqual(candidates, [])


class TestPartialPaymentValid(unittest.TestCase):
    """Case 7: partial payment allowed and valid - exactly two payments."""

    def test_partial_plan_built_when_eligible(self):
        request = make_request(requested_amount=Decimal("3000"), allows_partial_payment=True,
                                desired_completion_date=date(2026, 4, 1))
        profile = make_profile()
        # Safe today only for 1000; future events make full payment safe
        # by 2026-03-20 (a future credit).
        rent = CashEvent(date=date(2026, 3, 10), amount=Decimal("0"), direction="debit",
                          category="rent", event_type="expense", flexibility="fixed",
                          source="known_future", event_id="noop")
        salary = CashEvent(date=date(2026, 3, 20), amount=Decimal("5000"), direction="credit",
                            category="salary", event_type="income", flexibility="fixed",
                            source="known_future", event_id="future_salary")
        state = ReconciledState(user_id="user_1", request_date=date(2026, 3, 1),
                                 known_future=[rent, salary])
        forecast = build_forecast(state, starting_balance=Decimal("2000"),
                                   minimum_balance_to_keep=Decimal("1000"))
        safe = safe_amount_no_changes(forecast, request.requested_amount)
        earliest = earliest_date_for_full_payment(forecast, request.requested_amount)
        candidate = build_partial_payment_candidate(request, profile, forecast, safe, earliest)
        self.assertIsNotNone(candidate)
        self.assertEqual(len(candidate.payments), 2)
        self.assertEqual(
            candidate.payments[0].amount + candidate.payments[1].amount,
            request.requested_amount,
        )
        self.assertEqual(candidate.payments[0].date, request.request_date)
        self.assertEqual(candidate.payments[1].date, earliest)


class TestPartialPaymentNotAllowed(unittest.TestCase):
    """Case 8: partial payment not allowed by the request or the profile."""

    def test_request_disallows_partial(self):
        request = make_request(allows_partial_payment=False)
        profile = make_profile()
        forecast = flat_forecast(Decimal("2000"), Decimal("1000"))
        candidate = build_partial_payment_candidate(
            request, profile, forecast, Decimal("1000"), date(2026, 3, 15)
        )
        self.assertIsNone(candidate)

    def test_profile_does_not_accept_partial_payment(self):
        request = make_request(allows_partial_payment=True)
        profile = make_profile(payment_methods_user_will_consider=("full_payment",))
        forecast = flat_forecast(Decimal("2000"), Decimal("1000"))
        candidate = build_partial_payment_candidate(
            request, profile, forecast, Decimal("1000"), date(2026, 3, 15)
        )
        self.assertIsNone(candidate)


class TestPartialPaymentMissesDeadline(unittest.TestCase):
    """Case 9: partial payment cannot finish by the desired completion date."""

    def test_earliest_full_payment_after_deadline(self):
        request = make_request(desired_completion_date=date(2026, 3, 10))
        profile = make_profile()
        forecast = flat_forecast(Decimal("2000"), Decimal("1000"))
        candidate = build_partial_payment_candidate(
            request, profile, forecast, Decimal("1000"), date(2026, 3, 15)
        )
        self.assertIsNone(candidate)

    def test_no_earliest_date_at_all(self):
        request = make_request()
        profile = make_profile()
        forecast = flat_forecast(Decimal("2000"), Decimal("1000"))
        candidate = build_partial_payment_candidate(
            request, profile, forecast, Decimal("1000"), None
        )
        self.assertIsNone(candidate)


class TestPartialPlanExactlyTwoPayments(unittest.TestCase):
    """Case 10: the partial plan must contain exactly two payments that
    sum to the full requested amount."""

    def test_two_payments_sum_to_requested_amount(self):
        request = make_request(requested_amount=Decimal("3000"))
        profile = make_profile()
        forecast = flat_forecast(Decimal("10000"), Decimal("1000"))
        candidate = build_partial_payment_candidate(
            request, profile, forecast, Decimal("1200"), date(2026, 3, 20)
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(len(candidate.payments), 2)
        total = candidate.payments[0].amount + candidate.payments[1].amount
        self.assertEqual(total, Decimal("3000"))


class TestNoValidPlanWithinDesiredDate(unittest.TestCase):
    """Case 19: nothing (wait, partial, installments) can complete by the
    deadline - no candidate should be produced by any builder."""

    def test_wait_rejected_when_earliest_date_past_deadline(self):
        request = make_request(desired_completion_date=date(2026, 3, 5))
        profile = make_profile()
        candidate = build_wait_candidate(request, profile, date(2026, 3, 20))
        self.assertIsNone(candidate)

    def test_installments_rejected_when_last_payment_past_deadline(self):
        request = make_request(desired_completion_date=date(2026, 3, 5))
        profile = make_profile()
        forecast = flat_forecast(Decimal("100000"), Decimal("0"))
        option = make_option(number_of_payments=3, payment_frequency_days=30)
        candidates = build_installment_candidates(request, profile, [option], forecast)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
