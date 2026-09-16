"""
Reproduction test for the suspected split-series / spending-changes bug.

Investigation only - this file is NOT wired into any fix, and no
production code is modified alongside it.

Scenario: two interleaved debit recurring series share an identical
(category, event_type, direction) key after Stage 2's alternating-series
split (`engine/recurrence.py` Rule 9) - e.g. two different subscriptions
in the same category, paid on different cadences/amounts. A spending
change computed FOR one of them (identified by its own event_id) is
expected to affect ONLY that series' future occurrences. If it also
zeroes out or reduces the other split half's occurrences, that is
silent, unrequested overreach into a series the user never selected -
this test demonstrates exactly that.
"""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from data.models import FinancialProfile
from engine.forecast import ForecastCheckpoint, build_forecast
from engine.recurrence import RecurringSeries, SeriesKey, detect_recurring_series
from engine.state import CashEvent, ReconciledState

from planner.spending_changes import apply_changes_to_forecast, eligible_series_actions


def make_profile(**overrides) -> FinancialProfile:
    defaults = dict(
        user_id="user_1",
        home_currency="USD",
        current_available_balance=Decimal("1000"),
        minimum_balance_to_keep=Decimal("500"),
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=(),
        expense_categories_user_is_willing_to_stop=("streaming",),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )
    defaults.update(overrides)
    return FinancialProfile(**defaults)


def cash(d, amount, event_id, flexibility="stoppable"):
    return CashEvent(
        date=d, amount=Decimal(amount), direction="debit", category="streaming",
        event_type="subscription", flexibility=flexibility, source="settled",
        event_id=event_id,
    )


class SplitSeriesShareOneKeyTest(unittest.TestCase):
    """Step 0: confirm the premise - Rule 9 split halves really do share
    one (category, event_type, direction) key. If this fails, the bug
    theory itself is wrong and nothing below matters."""

    def test_two_interleaved_streaming_series_share_a_key(self):
        # Same construction as the existing, already-passing
        # TestInterleavedSalarySeries fixture (tests/test_recurrence.py),
        # just for a DEBIT category instead of income: two weekly series
        # offset by 3/4 days so the RAW merged sequence's gaps (3, 4) fit
        # no single cadence bucket - only the even/odd split (Rule 9)
        # reveals each half is independently a clean weekly cadence.
        from datetime import timedelta

        svc_a_dates = [date(2026, 1, 1) + i * timedelta(days=7) for i in range(4)]
        svc_b_dates = [date(2026, 1, 4) + i * timedelta(days=7) for i in range(4)]
        interleaved = sorted(svc_a_dates + svc_b_dates)
        history = [
            cash(d, "12" if d in svc_a_dates else "40", event_id=f"e-{d.isoformat()}")
            for d in interleaved
        ]
        gaps = [(interleaved[i + 1] - interleaved[i]).days for i in range(len(interleaved) - 1)]
        self.assertEqual(sorted(set(gaps)), [3, 4], "fixture sanity: raw gaps fit no single cadence")

        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 2, "must split into two independent series")
        keys = {(s.key.category, s.key.event_type, s.key.direction) for s in series}
        self.assertEqual(len(keys), 1, "both split halves must share one key")
        amounts = sorted(s.occurrences[-1].amount for s in series)
        self.assertEqual(amounts, [Decimal("12"), Decimal("40")])


class SpendingChangeCrossesSplitSeriesBoundaryTest(unittest.TestCase):
    """Steps 2-4: the actual bug, exercised through the REAL production
    path (detect_recurring_series -> project_series -> eligible_series_
    actions -> apply_changes_to_forecast) rather than hand-built
    checkpoints, so this test proves the fix on the exact code the
    planner runs - not just on a matching function in isolation."""

    def _two_split_debit_series_with_generated_checkpoints(self):
        from datetime import timedelta

        # Same interleaving pattern as the passing premise test above:
        # cheap ($12) and expensive ($40) streaming subscriptions,
        # weekly, offset so only the Rule 9 split reveals two series.
        cheap_dates = [date(2026, 1, 1) + i * timedelta(days=7) for i in range(4)]
        expensive_dates = [date(2026, 1, 4) + i * timedelta(days=7) for i in range(4)]
        interleaved = sorted(cheap_dates + expensive_dates)
        history = [
            cash(
                d, "12" if d in cheap_dates else "40",
                event_id=f"e-{d.isoformat()}", flexibility="stoppable",
            )
            for d in interleaved
        ]

        series_list = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series_list), 2, "fixture sanity: must split into two series")
        cheap_series = next(s for s in series_list if s.occurrences[-1].amount == Decimal("12"))
        expensive_series = next(s for s in series_list if s.occurrences[-1].amount == Decimal("40"))

        window_start = interleaved[-1] + timedelta(days=1)
        window_end = window_start + timedelta(days=30)
        generated = []
        for s in series_list:
            from engine.recurrence import project_series
            generated.extend(project_series(s, window_start, window_end))
        self.assertTrue(generated, "fixture sanity: must actually generate future occurrences")

        request_date = interleaved[-1]
        forecast = build_forecast(
            ReconciledState(user_id="user_1", request_date=request_date),
            starting_balance=Decimal("1000"), minimum_balance_to_keep=Decimal("0"),
        )
        balance = forecast.starting_balance
        checkpoints = []
        for ev in sorted(generated, key=lambda e: e.date):
            balance -= ev.amount
            checkpoints.append(ForecastCheckpoint(date=ev.date, event=ev, balance_after=balance))
        forecast.checkpoints = checkpoints
        forecast.recurring_series = series_list
        return forecast, cheap_series, expensive_series

    def test_stop_action_for_cheap_series_does_not_affect_the_expensive_one(self):
        """(1) affects the targeted series, (2) does not affect the other
        split half sharing the same (category, event_type, direction)."""
        forecast, cheap_series, expensive_series = self._two_split_debit_series_with_generated_checkpoints()
        profile = make_profile()

        # Built by the real eligibility function, targeting ONLY the cheap
        # series - exactly what the planner does.
        actions = eligible_series_actions(profile, [cheap_series], {})
        self.assertEqual(len(actions), 1)
        stop_action = actions[0]
        self.assertEqual(stop_action.series_instance_id, id(cheap_series))

        modified = apply_changes_to_forecast(forecast, [stop_action])

        cheap_checkpoints = [
            cp for cp in modified.checkpoints if cp.event.series_instance_id == id(cheap_series)
        ]
        expensive_checkpoints = [
            cp for cp in modified.checkpoints if cp.event.series_instance_id == id(expensive_series)
        ]
        self.assertTrue(cheap_checkpoints and expensive_checkpoints)

        # (1) targeted series affected.
        self.assertTrue(
            all(cp.event.amount == Decimal("0") for cp in cheap_checkpoints),
            "the targeted ($12) series should be stopped",
        )
        # (2) the other split half, sharing the same series_key, untouched.
        self.assertTrue(
            all(cp.event.amount == Decimal("40") for cp in expensive_checkpoints),
            "a spending change for one split-series half must not affect "
            "the OTHER split half sharing the same (category, event_type, "
            "direction) key",
        )

    def test_eligible_series_actions_tags_each_action_with_its_own_series(self):
        """Corroborating evidence at the layer above: two split series
        now produce actions carrying DIFFERENT series_instance_id values,
        even though they share one series_key."""
        forecast, cheap_series, expensive_series = self._two_split_debit_series_with_generated_checkpoints()
        profile = make_profile()

        actions = eligible_series_actions(profile, [cheap_series, expensive_series], {})
        self.assertEqual(len(actions), 2)
        instance_ids = {a.series_instance_id for a in actions}
        self.assertEqual(
            instance_ids, {id(cheap_series), id(expensive_series)},
            "each action must carry the identity of the exact series it came from",
        )
        # They still share one series_key - that premise hasn't changed.
        self.assertEqual(len({a.series_key for a in actions}), 1)


class KnownFutureAmbiguitySafetyTest(unittest.TestCase):
    """(4) Fail-closed safety: a known_future checkpoint (no series
    object to match by identity) must NOT receive a change when its
    (category, event_type, direction) key is ambiguous - i.e. claimed by
    more than one detected series - mirroring
    `evidence_integration.match_series_for_fact`'s refusal to guess."""

    def test_known_future_event_under_an_ambiguous_key_is_left_unchanged(self):
        forecast, cheap_series, expensive_series = (
            SpendingChangeCrossesSplitSeriesBoundaryTest()
            ._two_split_debit_series_with_generated_checkpoints()
        )
        profile = make_profile()
        actions = eligible_series_actions(profile, [cheap_series], {})
        self.assertEqual(len(actions), 1)

        known_future_event = CashEvent(
            date=forecast.request_date + __import__("datetime").timedelta(days=5),
            amount=Decimal("40"), direction="debit", category="streaming",
            event_type="subscription", flexibility="stoppable",
            source="known_future", event_id="kf_1",
        )
        balance = forecast.checkpoints[-1].balance_after - known_future_event.amount
        forecast.checkpoints = forecast.checkpoints + [
            ForecastCheckpoint(date=known_future_event.date, event=known_future_event, balance_after=balance)
        ]

        modified = apply_changes_to_forecast(forecast, actions)
        kf_after = next(cp for cp in modified.checkpoints if cp.event.event_id == "kf_1")
        self.assertEqual(
            kf_after.event.amount, Decimal("40"),
            "a known_future event under an AMBIGUOUS key must be left "
            "unchanged rather than guessed at",
        )


class SingleSeriesBehaviorUnchangedTest(unittest.TestCase):
    """(3) The normal, non-split case: exactly one series under a key,
    known_future AND generated occurrences both still get the change
    applied, unambiguously - unaffected by the fix."""

    def test_single_series_known_future_and_generated_both_still_change(self):
        request_date = date(2026, 3, 1)
        series = RecurringSeries(
            key=SeriesKey(user_id="user_1", category="streaming",
                          event_type="subscription", direction="debit"),
            cadence="monthly", interval_days=30,
            occurrences=[cash(date(2026, 2, 1), "50", "ev_template")],
            flexibility="stoppable",
        )
        generated_event = CashEvent(
            date=date(2026, 3, 15), amount=Decimal("50"), direction="debit",
            category="streaming", event_type="subscription", flexibility="stoppable",
            source="recurring", event_id=None, series_instance_id=id(series),
        )
        known_future_event = CashEvent(
            date=date(2026, 4, 1), amount=Decimal("50"), direction="debit",
            category="streaming", event_type="subscription", flexibility="stoppable",
            source="known_future", event_id="kf_solo",
        )

        state = ReconciledState(user_id="user_1", request_date=request_date)
        forecast = build_forecast(state, starting_balance=Decimal("1000"),
                                   minimum_balance_to_keep=Decimal("0"))
        balance = forecast.starting_balance
        checkpoints = []
        for ev in (generated_event, known_future_event):
            balance -= ev.amount
            checkpoints.append(ForecastCheckpoint(date=ev.date, event=ev, balance_after=balance))
        forecast.checkpoints = checkpoints
        forecast.recurring_series = [series]

        profile = make_profile()
        actions = eligible_series_actions(profile, [series], {})
        self.assertEqual(len(actions), 1)

        modified = apply_changes_to_forecast(forecast, actions)
        for cp in modified.checkpoints:
            self.assertEqual(
                cp.event.amount, Decimal("0"),
                f"unambiguous single-series case: both {cp.event.source} "
                "occurrences must still be stopped, exactly as before the fix",
            )


if __name__ == "__main__":
    unittest.main()
