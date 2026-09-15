"""Stage 4c tests: applying validated evidence facts (future_amount_change,
series_terminated) to Stage 2's deterministic forecast.

These tests build `NormalizedFact` fixtures directly (never via Gemini /
AIClient - Stage 4c never calls a model) and drive `engine.forecast.
build_forecast` and `engine.evidence_integration` directly.
"""

from __future__ import annotations

import inspect
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from engine import evidence_integration as ei
from engine import forecast as forecast_module
from engine.forecast import build_forecast
from engine.state import CashEvent, ReconciledState
from evidence.models import EvidenceRef, NormalizedFact
from planner.affordability import safe_amount_no_changes


def cash(d, amount, category="rent", event_type="expense", direction="debit",
         flexibility="fixed", source="settled", event_id=None):
    return CashEvent(
        date=d, amount=Decimal(amount), direction=direction, category=category,
        event_type=event_type, flexibility=flexibility, source=source,
        event_id=event_id or f"e-{d.isoformat()}-{category}-{source}",
    )


def fact(
    fact_id,
    user_id="user_1",
    fact_type="future_amount_change",
    target_event_id=None,
    target_series_key=None,
    resolved_amount=None,
    effective_date=None,
    end_date=None,
    sent_at=None,
    confidence="high",
    message_id=None,
):
    return NormalizedFact(
        fact_id=fact_id,
        user_id=user_id,
        fact_type=fact_type,
        provenance=EvidenceRef(
            user_id=user_id,
            message_id=message_id or fact_id,
            sent_at=sent_at,
        ),
        target_event_id=target_event_id,
        target_series_key=target_series_key,
        resolved_amount=resolved_amount,
        currency="USD" if resolved_amount is not None else None,
        effective_date=effective_date,
        end_date=end_date,
        resolution_method="gemini",
        confidence=confidence,
    )


RENT_SETTLED = [
    cash(date(2026, 1, 1), "1000", event_id="rent_jan"),
    cash(date(2026, 2, 1), "1000", event_id="rent_feb"),
    cash(date(2026, 3, 1), "1000", event_id="rent_mar"),
]


def rent_state(request_date=date(2026, 3, 15), extra_settled=None, extra_known_future=None):
    return ReconciledState(
        user_id="user_1",
        request_date=request_date,
        settled_history=list(RENT_SETTLED) + list(extra_settled or []),
        known_future=list(extra_known_future or []),
    )


class TestAmountChangeApplication(unittest.TestCase):
    def test_amount_change_applies_on_and_after_effective_date(self):
        # Rent goes from 1000 to 1200 starting 2026-05-01. April rent (before
        # the change) must stay 1000; May rent (on/after) must become 1200.
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1200"), effective_date=date(2026, 5, 1),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(f,),
        )
        april = [cp for cp in result.checkpoints if cp.date == date(2026, 4, 1)][0]
        may = [cp for cp in result.checkpoints if cp.date == date(2026, 5, 1)][0]
        self.assertEqual(april.event.amount, Decimal("1000"))
        self.assertEqual(may.event.amount, Decimal("1200"))
        # May and June (both >= the 2026-05-01 effective_date) change; only
        # April (before it) keeps the old amount.
        self.assertEqual(result.evidence_report.occurrences_amount_changed, 2)

    def test_effective_date_exactly_on_a_recurrence_date(self):
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1300"), effective_date=date(2026, 4, 1),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=60, evidence_facts=(f,),
        )
        april = [cp for cp in result.checkpoints if cp.date == date(2026, 4, 1)][0]
        self.assertEqual(april.event.amount, Decimal("1300"))

    def test_effective_date_between_recurrence_dates(self):
        # Effective 2026-04-15 (between the Apr 1 and May 1 occurrences):
        # April keeps the old amount, May picks up the new one.
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1400"), effective_date=date(2026, 4, 15),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(f,),
        )
        april = [cp for cp in result.checkpoints if cp.date == date(2026, 4, 1)][0]
        may = [cp for cp in result.checkpoints if cp.date == date(2026, 5, 1)][0]
        self.assertEqual(april.event.amount, Decimal("1000"))
        self.assertEqual(may.event.amount, Decimal("1400"))

    def test_end_date_resumes_original_amount(self):
        # A temporary bump for May only (end_date = end of May); June must
        # resume the original 1000, per the documented end_date assumption.
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1500"), effective_date=date(2026, 5, 1),
            end_date=date(2026, 5, 31),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=100, evidence_facts=(f,),
        )
        may = [cp for cp in result.checkpoints if cp.date == date(2026, 5, 1)][0]
        june = [cp for cp in result.checkpoints if cp.date == date(2026, 6, 1)][0]
        self.assertEqual(may.event.amount, Decimal("1500"))
        self.assertEqual(june.event.amount, Decimal("1000"))

    def test_no_duplicate_future_event_created(self):
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1200"), effective_date=date(2026, 5, 1),
        )
        baseline = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90,
        )
        with_evidence = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(f,),
        )
        self.assertEqual(len(baseline.checkpoints), len(with_evidence.checkpoints))
        rent_dates = [cp.date for cp in with_evidence.checkpoints if cp.event.category == "rent"]
        self.assertEqual(len(rent_dates), len(set(rent_dates)), "no duplicate rent occurrence dates")


class TestSeriesTermination(unittest.TestCase):
    def test_termination_stops_future_occurrences_preserves_history(self):
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="series_terminated",
            effective_date=date(2026, 5, 1),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(f,),
        )
        rent_dates_in_ledger = sorted(
            cp.date for cp in result.checkpoints if cp.event.category == "rent"
        )
        # April (before effective_date) still generated; May onward dropped.
        self.assertIn(date(2026, 4, 1), rent_dates_in_ledger)
        self.assertNotIn(date(2026, 5, 1), rent_dates_in_ledger)
        self.assertNotIn(date(2026, 6, 1), rent_dates_in_ledger)
        # Historical settled occurrences are untouched (not in this window,
        # but confirm the underlying state was never mutated).
        self.assertEqual(len(RENT_SETTLED), 3)
        self.assertEqual(result.evidence_report.occurrences_dropped_by_termination, 2)

    def test_termination_does_not_affect_another_series(self):
        salary_settled = [
            cash(date(2026, 1, 5), "5000", category="salary", event_type="income",
                 direction="credit", event_id="salary_jan"),
            cash(date(2026, 2, 5), "5000", category="salary", event_type="income",
                 direction="credit", event_id="salary_feb"),
            cash(date(2026, 3, 5), "5000", category="salary", event_type="income",
                 direction="credit", event_id="salary_mar"),
        ]
        state = rent_state(extra_settled=salary_settled)
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="series_terminated",
            effective_date=date(2026, 5, 1),
        )
        result = build_forecast(
            state, starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(f,),
        )
        salary_dates = sorted(cp.date for cp in result.checkpoints if cp.event.category == "salary")
        self.assertIn(date(2026, 4, 5), salary_dates)
        self.assertIn(date(2026, 5, 5), salary_dates)
        self.assertIn(date(2026, 6, 5), salary_dates)


class TestAmbiguousSeriesIdentity(unittest.TestCase):
    """Two salary series sharing (category, event_type, direction) after
    Stage 2's Rule-9 alternating split - the exact case the task warns
    about."""

    # Raw merged gaps are 3/4/3/4/3 days - no single cadence (weekly needs
    # 5-9), so the whole group fails Stage 2's straight cadence check and
    # only qualifies via Rule 9's alternating split: even-indexed events
    # (7-day gaps) and odd-indexed events (7-day gaps) each independently
    # qualify as their own weekly series, sharing one identical
    # (category, event_type, direction) key - exactly the case the task
    # warns a bare-key match must not guess across.
    ALTERNATING_SALARY = [
        cash(date(2026, 1, 1), "3000", category="salary", event_type="income",
             direction="credit", event_id="sal_a1"),
        cash(date(2026, 1, 4), "4000", category="salary", event_type="income",
             direction="credit", event_id="sal_b1"),
        cash(date(2026, 1, 8), "3000", category="salary", event_type="income",
             direction="credit", event_id="sal_a2"),
        cash(date(2026, 1, 11), "4000", category="salary", event_type="income",
             direction="credit", event_id="sal_b2"),
        cash(date(2026, 1, 15), "3000", category="salary", event_type="income",
             direction="credit", event_id="sal_a3"),
        cash(date(2026, 1, 18), "4000", category="salary", event_type="income",
             direction="credit", event_id="sal_b3"),
    ]

    def _state(self):
        return ReconciledState(
            user_id="user_1", request_date=date(2026, 2, 10),
            settled_history=list(self.ALTERNATING_SALARY),
        )

    def test_confirms_two_series_share_the_bare_key(self):
        from engine.recurrence import detect_recurring_series
        series_list = detect_recurring_series(self.ALTERNATING_SALARY, "user_1")
        keys = [(s.key.category, s.key.event_type, s.key.direction) for s in series_list]
        self.assertEqual(len(series_list), 2)
        self.assertEqual(keys[0], keys[1], "precondition: both split halves share one bare key")

    def test_bare_key_only_fact_left_unresolved(self):
        f = fact(
            "f1", fact_type="future_amount_change",
            target_series_key=("salary", "income", "credit"),
            resolved_amount=Decimal("9999"), effective_date=date(2026, 3, 1),
        )
        baseline = build_forecast(
            self._state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=60,
        )
        result = build_forecast(
            self._state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=60, evidence_facts=(f,),
        )
        # No occurrence anywhere should have picked up 9999 - the fact must
        # be left unresolved, never guessed onto either half.
        amounts = {cp.event.amount for cp in result.checkpoints if cp.event.category == "salary"}
        self.assertNotIn(Decimal("9999"), amounts)
        self.assertEqual(len(result.evidence_report.unresolved), 1)
        self.assertEqual(result.evidence_report.unresolved[0].fact_id, "f1")
        self.assertIn("refusing to guess", result.evidence_report.unresolved[0].reason.lower())
        # And the forecast is otherwise identical to having no evidence at all.
        self.assertEqual(
            [(cp.date, cp.event.amount) for cp in baseline.checkpoints],
            [(cp.date, cp.event.amount) for cp in result.checkpoints],
        )

    def test_target_event_id_disambiguates_and_only_adjusts_that_half(self):
        f = fact(
            "f1", fact_type="future_amount_change", target_event_id="sal_a3",
            resolved_amount=Decimal("3500"), effective_date=date(2026, 2, 1),
        )
        result = build_forecast(
            self._state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=60, evidence_facts=(f,),
        )
        by_amount = sorted(
            (cp.date, cp.event.amount) for cp in result.checkpoints if cp.event.category == "salary"
        )
        # The 3000-paid-biweekly-ish half should now show 3500 going
        # forward; the 4000 half must be completely untouched.
        amounts_seen = {amt for _, amt in by_amount}
        self.assertIn(Decimal("3500"), amounts_seen)
        self.assertIn(Decimal("4000"), amounts_seen)
        self.assertNotIn(Decimal("3000"), amounts_seen)  # the changed half is now 3500, not 3000


class TestConflictResolution(unittest.TestCase):
    def test_newer_same_source_evidence_wins(self):
        older = fact(
            "older", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1100"), effective_date=date(2026, 5, 1),
            sent_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        newer = fact(
            "newer", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1250"), effective_date=date(2026, 5, 1),
            sent_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(older, newer),
        )
        may = [cp for cp in result.checkpoints if cp.date == date(2026, 5, 1)][0]
        self.assertEqual(may.event.amount, Decimal("1250"))

    def test_higher_confidence_wins_when_timestamps_tied(self):
        same_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
        low = fact(
            "low", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1100"), effective_date=date(2026, 5, 1),
            sent_at=same_time, confidence="low",
        )
        high = fact(
            "high", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1250"), effective_date=date(2026, 5, 1),
            sent_at=same_time, confidence="high",
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=90, evidence_facts=(low, high),
        )
        may = [cp for cp in result.checkpoints if cp.date == date(2026, 5, 1)][0]
        self.assertEqual(may.event.amount, Decimal("1250"))

    def test_cancellation_beats_weaker_future_estimate(self):
        termination = fact(
            "term", target_event_id="rent_mar", fact_type="series_terminated",
            effective_date=date(2026, 5, 1),
        )
        weaker_change = fact(
            "change", target_event_id="rent_mar", fact_type="future_amount_change",
            resolved_amount=Decimal("1200"), effective_date=date(2026, 6, 1),
        )
        result = build_forecast(
            rent_state(), starting_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("0"), forecast_days=100,
            evidence_facts=(termination, weaker_change),
        )
        rent_dates = [cp.date for cp in result.checkpoints if cp.event.category == "rent"]
        self.assertNotIn(date(2026, 6, 1), rent_dates)  # dropped by termination, not re-priced


class TestExistingBehaviorUnchangedWithoutEvidence(unittest.TestCase):
    def test_omitted_and_explicit_empty_evidence_produce_identical_forecasts(self):
        state = rent_state()
        omitted = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("0"), forecast_days=90,
        )
        explicit_empty = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("0"),
            forecast_days=90, evidence_facts=(),
        )
        self.assertEqual(
            [(cp.date, cp.event.amount, cp.balance_after) for cp in omitted.checkpoints],
            [(cp.date, cp.event.amount, cp.balance_after) for cp in explicit_empty.checkpoints],
        )
        self.assertIsNone(omitted.evidence_report)
        self.assertIsNone(explicit_empty.evidence_report)


class TestPlannerReactsToEvidenceDrivenForecastChanges(unittest.TestCase):
    """Confirms the integration point works end-to-end into the (untouched)
    Stage 3 planner: `safe_amount_no_changes` is a pure function of the
    `ForecastResult` it's given, so a terminated expense series must
    increase the safe-to-pay amount without any planner code changing."""

    def test_terminating_an_expense_series_increases_amount_safe_to_pay(self):
        state = rent_state(request_date=date(2026, 3, 15))
        baseline = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("0"), forecast_days=90,
        )
        f = fact(
            "f1", target_event_id="rent_mar", fact_type="series_terminated",
            effective_date=date(2026, 4, 1),
        )
        with_evidence = build_forecast(
            state, starting_balance=Decimal("10000"), minimum_balance_to_keep=Decimal("0"),
            forecast_days=90, evidence_facts=(f,),
        )
        baseline_safe = safe_amount_no_changes(baseline, Decimal("100000"))
        evidence_safe = safe_amount_no_changes(with_evidence, Decimal("100000"))
        self.assertGreater(evidence_safe, baseline_safe)


class TestNoGeminiInFinancialIntegration(unittest.TestCase):
    """Stage 4c must never call a model - it only ever consumes
    already-resolved `NormalizedFact`s."""

    def test_evidence_integration_module_has_no_model_client_dependency(self):
        source = inspect.getsource(ei)
        for forbidden in ("genai", "GEMINI_API_KEY", "AIClient", "requests", "urllib"):
            self.assertNotIn(forbidden, source)

    def test_forecast_module_has_no_model_client_dependency(self):
        source = inspect.getsource(forecast_module)
        for forbidden in ("genai", "GEMINI_API_KEY", "AIClient", "requests", "urllib"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()