"""Tests for engine.recurrence - cadence detection, alternating-series
splitting, and projection amount rules."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from engine.recurrence import detect_recurring_series, project_series
from engine.state import CashEvent


def cash(d, amount, category="rent", event_type="expense", direction="debit", flexibility="fixed"):
    return CashEvent(
        date=d, amount=Decimal(amount), direction=direction, category=category,
        event_type=event_type, flexibility=flexibility, source="settled",
        event_id=f"e-{d.isoformat()}-{category}",
    )


class TestFixedMonthlyRecurrence(unittest.TestCase):
    def test_detects_and_projects_at_last_amount(self):
        history = [
            cash(date(2026, 1, 1), "1000"),
            cash(date(2026, 2, 1), "1000"),
            cash(date(2026, 3, 1), "1000"),
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].cadence, "monthly")

        occurrences = project_series(series[0], date(2026, 3, 2), date(2026, 5, 31))
        dates = [o.date for o in occurrences]
        self.assertEqual(dates, [date(2026, 4, 1), date(2026, 5, 1)])
        self.assertTrue(all(o.amount == Decimal("1000") for o in occurrences))


class TestWeeklySalary(unittest.TestCase):
    def test_weekly_cadence_detected(self):
        history = [
            cash(date(2026, 1, 2), "2000", category="salary", event_type="income", direction="credit"),
            cash(date(2026, 1, 9), "2000", category="salary", event_type="income", direction="credit"),
            cash(date(2026, 1, 16), "2000", category="salary", event_type="income", direction="credit"),
            cash(date(2026, 1, 23), "2000", category="salary", event_type="income", direction="credit"),
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].cadence, "weekly")
        self.assertEqual(series[0].interval_days, 7)


class TestInterleavedSalarySeries(unittest.TestCase):
    def test_two_interleaved_weekly_salaries_stay_separate(self):
        # Two weekly-paid jobs, offset by 3/4 days from each other, so the
        # RAW merged sequence has alternating 3/4-day gaps that match no
        # single cadence bucket on their own - only the alternating split
        # (even-index events = job A, odd-index = job B) reveals that each
        # sub-series is independently a clean weekly cadence.
        job_a = [date(2026, 1, 1) + i * __import__("datetime").timedelta(days=7) for i in range(4)]
        job_b = [date(2026, 1, 4) + i * __import__("datetime").timedelta(days=7) for i in range(4)]
        interleaved_dates = sorted(job_a + job_b)
        history = [
            cash(d, "2000" if d in job_a else "2500", category="salary",
                 event_type="income", direction="credit")
            for d in interleaved_dates
        ]
        # Sanity check on the fixture itself: the raw merged gaps must NOT
        # already look like one consistent cadence.
        gaps = [(interleaved_dates[i + 1] - interleaved_dates[i]).days
                for i in range(len(interleaved_dates) - 1)]
        self.assertEqual(sorted(set(gaps)), [3, 4])

        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 2, "must split into two separate series, not one or zero")
        for s in series:
            self.assertEqual(s.cadence, "weekly")
            self.assertEqual(len(s.occurrences), 4)
        occurrence_date_sets = sorted(tuple(o.date for o in s.occurrences) for s in series)
        self.assertEqual(occurrence_date_sets, sorted([tuple(job_a), tuple(job_b)]))


class TestOneOffEventsNotProjected(unittest.TestCase):
    def test_single_occurrence_is_not_a_series(self):
        history = [cash(date(2026, 1, 5), "40000", category="electronics", event_type="expense")]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(series, [])

    def test_two_irregular_occurrences_do_not_qualify(self):
        # Gap is 10 days - not strict-monthly evidence, and only 2 events.
        history = [
            cash(date(2026, 1, 1), "500", category="electronics", event_type="expense"),
            cash(date(2026, 1, 11), "500", category="electronics", event_type="expense"),
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(series, [])

    def test_two_occurrences_with_strict_monthly_evidence_qualify(self):
        # Gap = 29 days (in [28,31]) and calendar day 5 vs 3 (diff 2, within
        # the +/-2 day strict-monthly tolerance).
        history = [
            cash(date(2026, 1, 5), "1200", category="rent", event_type="expense"),
            cash(date(2026, 2, 3), "1200", category="rent", event_type="expense"),
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].cadence, "monthly")


class TestMaxHistoryVariableEssentialSpending(unittest.TestCase):
    def test_dining_projects_at_max_not_last_or_average(self):
        history = [
            cash(date(2026, 1, 5), "20", category="dining", flexibility="reducible_or_stoppable"),
            cash(date(2026, 2, 5), "45", category="dining", flexibility="reducible_or_stoppable"),
            cash(date(2026, 3, 5), "15", category="dining", flexibility="reducible_or_stoppable"),
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0].projected_amount, Decimal("45"))

        occurrences = project_series(series[0], date(2026, 3, 6), date(2026, 5, 6))
        self.assertTrue(all(o.amount == Decimal("45") for o in occurrences))

    def test_fixed_category_projects_at_last_not_max(self):
        history = [
            cash(date(2026, 1, 1), "1000", category="rent"),
            cash(date(2026, 2, 1), "1000", category="rent"),
            cash(date(2026, 3, 1), "1200", category="rent"),  # rent increase
        ]
        series = detect_recurring_series(history, "user_1")
        self.assertEqual(series[0].projected_amount, Decimal("1200"))


if __name__ == "__main__":
    unittest.main()
