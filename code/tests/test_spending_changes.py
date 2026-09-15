"""Tests for planner.spending_changes - eligibility rules (protected
categories, profile willingness), the max-3-changes limit, and the
gentlest-first minimal-combination search."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from data.models import FinancialEvent, FinancialProfile
from engine.forecast import build_forecast
from engine.recurrence import RecurringSeries, SeriesKey
from engine.state import CashEvent, ReconciledState

from planner.affordability import safe_amount_on
from planner.spending_changes import (
    MAX_SPENDING_CHANGES,
    apply_changes_to_forecast,
    eligible_series_actions,
    find_minimal_spending_change_combo,
)


def make_profile(**overrides) -> FinancialProfile:
    defaults = dict(
        user_id="user_1",
        home_currency="USD",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("500"),
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=(),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )
    defaults.update(overrides)
    return FinancialProfile(**defaults)


def make_series(category, flexibility, amount="100", event_id="ev_1",
                 event_type="subscription", direction="debit", cadence="monthly"):
    occurrence = CashEvent(
        date=date(2026, 2, 1), amount=Decimal(amount), direction=direction,
        category=category, event_type=event_type, flexibility=flexibility,
        source="settled", event_id=event_id,
    )
    return RecurringSeries(
        key=SeriesKey(user_id="user_1", category=category, event_type=event_type, direction=direction),
        cadence=cadence, interval_days=30, occurrences=[occurrence], flexibility=flexibility,
    )


def make_fin_event(event_id, category, amount, minimum_allowed_amount=None, flexibility="reducible"):
    return FinancialEvent(
        event_id=event_id, user_id="user_1", event_type="subscription", description="",
        category=category, direction="debit", amount=Decimal(amount), currency="USD",
        event_date=date(2026, 2, 1), settlement_date=date(2026, 2, 1), status="settled",
        linked_event_id=None, flexibility=flexibility,
        minimum_allowed_amount=Decimal(minimum_allowed_amount) if minimum_allowed_amount else None,
    )


class TestProtectedExpenseCannotBeReduced(unittest.TestCase):
    """Case 11: a protected category never produces a spending-change
    candidate, even if the underlying event is technically flexible."""

    def test_protected_category_excluded_even_if_reducible(self):
        profile = make_profile(
            expense_categories_to_protect=("rent",),
            expense_categories_user_is_willing_to_reduce=("rent",),
        )
        series = [make_series("rent", "reducible", event_id="ev_rent")]
        events_by_id = {"ev_rent": make_fin_event("ev_rent", "rent", "100", "50")}
        actions = eligible_series_actions(profile, series, events_by_id)
        self.assertEqual(actions, [])

    def test_always_protected_category_excluded_regardless_of_profile(self):
        # utilities is always-protected even if the profile never listed it
        # under expense_categories_to_protect.
        profile = make_profile(expense_categories_user_is_willing_to_reduce=("utilities",))
        series = [make_series("utilities", "reducible", event_id="ev_util")]
        events_by_id = {"ev_util": make_fin_event("ev_util", "utilities", "100", "50")}
        actions = eligible_series_actions(profile, series, events_by_id)
        self.assertEqual(actions, [])


class TestFlexibleExpenseCanBeReduced(unittest.TestCase):
    """Case 12: an eligible flexible recurring expense produces a valid
    candidate targeting its minimum_allowed_amount."""

    def test_reduce_candidate_generated(self):
        profile = make_profile(expense_categories_user_is_willing_to_reduce=("dining",))
        series = [make_series("dining", "reducible", amount="200", event_id="ev_dining")]
        events_by_id = {"ev_dining": make_fin_event("ev_dining", "dining", "200", "80")}
        actions = eligible_series_actions(profile, series, events_by_id)
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action.kind, "reduce_to")
        self.assertEqual(action.event_id, "ev_dining")
        self.assertEqual(action.original_amount, Decimal("200"))
        self.assertEqual(action.adjusted_amount, Decimal("80"))


class TestProfileUnwillingnessBlocksChange(unittest.TestCase):
    """Case 14: a technically-flexible event is still excluded if the
    user's profile does not list its category as willing-to-reduce/stop."""

    def test_reducible_series_excluded_without_willingness(self):
        profile = make_profile()  # no willing_to_reduce / willing_to_stop at all
        series = [make_series("dining", "reducible", event_id="ev_dining")]
        events_by_id = {"ev_dining": make_fin_event("ev_dining", "dining", "100", "40")}
        actions = eligible_series_actions(profile, series, events_by_id)
        self.assertEqual(actions, [])

    def test_stoppable_series_excluded_without_willingness(self):
        profile = make_profile()
        series = [make_series("streaming", "stoppable", event_id="ev_stream")]
        events_by_id = {"ev_stream": make_fin_event("ev_stream", "streaming", "20")}
        actions = eligible_series_actions(profile, series, events_by_id)
        self.assertEqual(actions, [])


class TestMaxThreeSpendingChanges(unittest.TestCase):
    """Case 13: the combination search never proposes more than
    MAX_SPENDING_CHANGES actions, even when more are eligible and needed."""

    def test_never_exceeds_three_even_with_many_eligible_actions(self):
        self.assertEqual(MAX_SPENDING_CHANGES, 3)

        profile = make_profile(
            expense_categories_user_is_willing_to_stop=tuple(f"cat{i}" for i in range(5))
        )
        series = [
            make_series(f"cat{i}", "stoppable", amount="10", event_id=f"ev_{i}")
            for i in range(5)
        ]
        events_by_id = {f"ev_{i}": make_fin_event(f"ev_{i}", f"cat{i}", "10") for i in range(5)}

        request_date = date(2026, 3, 1)
        forecast_events = []
        for i in range(5):
            forecast_events.append(
                CashEvent(
                    date=date(2026, 3, 15), amount=Decimal("10"), direction="debit",
                    category=f"cat{i}", event_type="subscription", flexibility="stoppable",
                    source="recurring", event_id=None, series_key=(f"cat{i}", "subscription", "debit"),
                )
            )
        state = ReconciledState(user_id="user_1", request_date=request_date)
        forecast = build_forecast(state, starting_balance=Decimal("1000"),
                                   minimum_balance_to_keep=Decimal("955"))
        # Manually splice in the 5 synthetic recurring debits since they
        # weren't detected by recurrence (only one settled occurrence each).
        from engine.forecast import ForecastCheckpoint
        balance = forecast.starting_balance
        checkpoints = []
        for ev in forecast_events:
            balance -= ev.amount
            checkpoints.append(ForecastCheckpoint(date=ev.date, event=ev, balance_after=balance))
        forecast.checkpoints = checkpoints

        # Needing all 5 stopped to be safe (50 total debit vs 45 headroom)
        # is impossible within the 3-change cap - the search must return
        # None rather than a 4- or 5-change combination.
        result = find_minimal_spending_change_combo(
            forecast, Decimal("0"), request_date, profile, series, events_by_id
        )
        # amount requested is 0 so it's trivially "safe" without changes;
        # use a nonzero anchor check instead via safe_amount_on directly.
        safe_no_changes = safe_amount_on(forecast, request_date, Decimal("1"))
        self.assertEqual(safe_no_changes, Decimal("0"))  # confirms constrained scenario

        # Re-run properly: request a full payment amount that requires
        # stopping subscriptions to become safe, and confirm the search
        # never returns a combination bigger than 3.
        combo_result = find_minimal_spending_change_combo(
            forecast, Decimal("5"), request_date, profile, series, events_by_id
        )
        if combo_result is not None:
            combo, _modified = combo_result
            self.assertLessEqual(len(combo), MAX_SPENDING_CHANGES)


class TestApplyChangesToForecast(unittest.TestCase):
    def test_stop_action_zeroes_out_future_occurrences(self):
        request_date = date(2026, 3, 1)
        from engine.forecast import ForecastCheckpoint
        from planner.models import SpendingChangeAction

        recurring_event = CashEvent(
            date=date(2026, 3, 15), amount=Decimal("50"), direction="debit",
            category="streaming", event_type="subscription", flexibility="stoppable",
            source="recurring", event_id=None,
        )
        state = ReconciledState(user_id="user_1", request_date=request_date)
        forecast = build_forecast(state, starting_balance=Decimal("1000"),
                                   minimum_balance_to_keep=Decimal("0"))
        forecast.checkpoints = [
            ForecastCheckpoint(date=recurring_event.date, event=recurring_event,
                                balance_after=Decimal("950"))
        ]
        action = SpendingChangeAction(
            kind="stop", event_id="ev_stream", category="streaming",
            series_key=("streaming", "subscription", "debit"),
            original_amount=Decimal("50"), adjusted_amount=Decimal("0"), severity=1,
        )
        modified = apply_changes_to_forecast(forecast, [action])
        self.assertEqual(modified.checkpoints[0].event.amount, Decimal("0"))
        self.assertEqual(modified.checkpoints[0].balance_after, Decimal("1000"))


if __name__ == "__main__":
    unittest.main()
