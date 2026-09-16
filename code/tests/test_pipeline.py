"""
Production-pipeline wiring tests.

These cover the seam that `main.py` previously did not have: that
validated evidence facts actually reach the forecast and the planner, and
that the CLI and the submission runner execute the same code path.

Everything here is offline and deterministic - facts are constructed
directly or written to a temporary cache file. No test constructs a model
client, needs `google-genai`, or requires GEMINI_API_KEY.
"""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

import main as main_module
import pipeline
from data.models import Dataset, FinancialEvent
from evidence.cache import EvidenceCache
from evidence.models import EvidenceRef, NormalizedFact

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"


def make_event(event_id="event_1", amount=None, currency="INR") -> FinancialEvent:
    return FinancialEvent(
        event_id=event_id,
        user_id="user_1",
        event_type="purchase",
        description="test event",
        category="shopping",
        direction="debit",
        amount=amount,
        currency=currency,
        event_date=date(2025, 6, 1),
        settlement_date=None,
        status="pending",
        linked_event_id=None,
        flexibility="fixed",
        minimum_allowed_amount=None,
    )


def make_dataset(events) -> Dataset:
    return Dataset(
        profiles=[], events=list(events), exchange_rates=[], payment_options=[],
        messages=[], images=[], requests=[], sample_requests=[],
    )


def image_amount_fact(event_id="event_1", amount="1234.50") -> NormalizedFact:
    return NormalizedFact(
        fact_id=f"fact:image_01:resolved_amount",
        user_id="user_1",
        fact_type="resolved_amount",
        provenance=EvidenceRef(user_id="user_1", image_id="image_01", related_event_id=event_id),
        target_event_id=event_id,
        resolved_amount=Decimal(amount),
        currency="INR",
        resolution_method="gemini",
        confidence="high",
    )


def series_fact() -> NormalizedFact:
    return NormalizedFact(
        fact_id="fact:message_01:series_terminated",
        user_id="user_1",
        fact_type="series_terminated",
        provenance=EvidenceRef(user_id="user_1", message_id="message_01"),
        target_series_key=("rent", "bill", "debit"),
        effective_date=date(2025, 7, 1),
        resolution_method="gemini",
        confidence="high",
    )


class SplitFactsTests(unittest.TestCase):
    def test_event_level_and_series_level_are_separated(self):
        event_level, series_level = pipeline.split_facts([image_amount_fact(), series_fact()])
        self.assertEqual([f.fact_type for f in event_level], ["resolved_amount"])
        self.assertEqual([f.fact_type for f in series_level], ["series_terminated"])

    def test_a_fact_is_never_in_both_groups(self):
        facts = [image_amount_fact(), series_fact()]
        event_level, series_level = pipeline.split_facts(facts)
        ids = {f.fact_id for f in event_level} & {f.fact_id for f in series_level}
        self.assertEqual(ids, set())
        self.assertEqual(len(event_level) + len(series_level), len(facts))

    def test_grouping_by_user(self):
        grouped = pipeline.group_facts_by_user([series_fact()])
        self.assertEqual(set(grouped), {"user_1"})
        self.assertIsInstance(grouped["user_1"], tuple)


class PrepareTests(unittest.TestCase):
    def test_amount_facts_are_applied_before_indexing(self):
        dataset = make_dataset([make_event(amount=None)])
        prepared = pipeline.prepare(dataset, [image_amount_fact()])
        self.assertEqual(prepared.dataset.events[0].amount, Decimal("1234.50"))
        # The indexes must reflect the filled amount, not the blank one.
        self.assertEqual(
            prepared.indexes.events_by_id["event_1"].amount, Decimal("1234.50")
        )

    def test_series_facts_are_not_applied_to_events(self):
        dataset = make_dataset([make_event(amount=None)])
        prepared = pipeline.prepare(dataset, [series_fact()])
        self.assertIsNone(prepared.dataset.events[0].amount)
        self.assertEqual(prepared.facts_for("user_1"), (series_fact(),))

    def test_image_derived_count_is_reported(self):
        dataset = make_dataset([make_event(amount=None)])
        prepared = pipeline.prepare(dataset, [image_amount_fact()])
        self.assertEqual(prepared.image_derived_count, 1)

    def test_no_facts_reproduces_the_pre_evidence_dataset(self):
        dataset = make_dataset([make_event(amount=None)])
        prepared = pipeline.prepare(dataset, [])
        self.assertIsNone(prepared.dataset.events[0].amount)
        self.assertEqual(prepared.facts_by_user, {})
        self.assertEqual(prepared.total_facts, 0)


class CacheLoadingTests(unittest.TestCase):
    def test_missing_cache_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = pipeline.load_facts_from_cache(Path(tmp) / "nope.json")
            self.assertEqual(facts, [])

    def test_facts_round_trip_through_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            cache = EvidenceCache(path)
            # Write via the cache's own serializer so this test exercises
            # the same format the real pipeline reads.
            cache._store["image_01:abc"] = __import__(
                "evidence.cache", fromlist=["_fact_to_json"]
            )._fact_to_json(image_amount_fact())
            cache.save()

            facts = pipeline.load_facts_from_cache(path)
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].resolved_amount, Decimal("1234.50"))
            self.assertEqual(facts[0].provenance.image_id, "image_01")

    def test_all_facts_is_deterministically_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            cache = EvidenceCache(path)
            to_json = __import__("evidence.cache", fromlist=["_fact_to_json"])._fact_to_json
            cache._store["z_key"] = to_json(image_amount_fact("event_1"))
            cache._store["a_key"] = to_json(image_amount_fact("event_2"))
            self.assertEqual(
                [f.target_event_id for f in cache.all_facts()], ["event_2", "event_1"]
            )


class OfflineGuaranteeTests(unittest.TestCase):
    def test_loading_evidence_never_constructs_a_model_client(self):
        # `resolve_evidence_live` imports the SDK lazily; the default path
        # must never touch it, so the pipeline works with no key installed.
        import evidence.ai_client as ai_client

        calls = []
        original = ai_client.GeminiAIClient.__init__

        def spy(self, *a, **kw):  # pragma: no cover - should never run
            calls.append(1)
            return original(self, *a, **kw)

        ai_client.GeminiAIClient.__init__ = spy
        try:
            dataset = make_dataset([make_event(amount=None)])
            pipeline.prepare(dataset, [image_amount_fact()])
            with tempfile.TemporaryDirectory() as tmp:
                pipeline.load_facts_from_cache(Path(tmp) / "nope.json")
        finally:
            ai_client.GeminiAIClient.__init__ = original
        self.assertEqual(calls, [])


def run_cli(argv) -> int:
    """Invoke the CLI, swallowing its console output so the test suite's
    own output stays readable. Returns the exit code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return main_module.main(argv)


@unittest.skipUnless(DATASET_DIR.is_dir(), "dataset/ not available")
class ProductionRunTests(unittest.TestCase):
    """End-to-end over the real dataset, still fully offline."""

    def test_run_writes_every_request_with_the_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            rc = run_cli(["run", "--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())

            with out.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], pipeline.COLUMNS)

            prepared = pipeline.load_and_prepare()
            self.assertEqual(len(rows) - 1, len(prepared.dataset.requests))

    def test_run_output_respects_amount_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            run_cli(["run", "--output", str(out)])
            prepared = pipeline.load_and_prepare()
            requested = {r.request_id: r.requested_amount for r in prepared.dataset.requests}
            with out.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    amount = Decimal(row["amount_safe_to_pay"])
                    self.assertGreaterEqual(amount, 0)
                    self.assertLessEqual(amount, requested[row["request_id"]])

    def test_no_evidence_flag_changes_nothing_structurally(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            rc = run_cli(["run", "--output", str(out), "--no-evidence"])
            self.assertEqual(rc, 0)
            with out.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], pipeline.COLUMNS)
            self.assertGreater(len(rows) - 1, 0)

    def test_evidence_actually_reaches_the_planner(self):
        """With evidence on, at least one user must carry series-level
        facts into `plan_request` - otherwise the wiring is inert."""
        prepared = pipeline.load_and_prepare()
        without = pipeline.load_and_prepare(use_evidence=False)
        self.assertGreater(prepared.total_facts, 0)
        self.assertGreater(len(prepared.facts_by_user), 0)
        self.assertEqual(without.total_facts, 0)
        self.assertEqual(without.facts_by_user, {})

    def test_final_submission_uses_the_same_pipeline_as_main_run(self):
        import final_submission

        self.assertIs(final_submission.pipeline, pipeline)

    def test_empty_cache_path_still_produces_a_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output.csv"
            rc = run_cli(
                ["run", "--output", str(out), "--evidence-cache", str(Path(tmp) / "none.json")]
            )
            self.assertEqual(rc, 0)
            with out.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertGreater(len(rows) - 1, 0)

    def test_forecast_and_plan_subcommands_still_run(self):
        prepared = pipeline.load_and_prepare()
        request_id = prepared.dataset.requests[0].request_id
        self.assertEqual(run_cli(["forecast", "--request-id", request_id]), 0)
        self.assertEqual(run_cli(["plan", "--request-id", request_id]), 0)

    def test_plan_subcommand_accepts_no_evidence(self):
        prepared = pipeline.load_and_prepare()
        request_id = prepared.dataset.requests[0].request_id
        self.assertEqual(
            run_cli(["plan", "--request-id", request_id, "--no-evidence"]), 0
        )


if __name__ == "__main__":
    unittest.main()
