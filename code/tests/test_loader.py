import sys
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import (
    DatasetLoadError,
    load_dataset,
    parse_bool,
    parse_optional_date,
    parse_optional_decimal,
    parse_pipe_list,
)

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


class TestParsingHelpers(unittest.TestCase):
    def test_blank_amount_is_none_not_zero(self):
        self.assertIsNone(parse_optional_decimal(""))
        self.assertIsNone(parse_optional_decimal("   "))

    def test_decimal_parses_exactly(self):
        self.assertEqual(parse_optional_decimal("1852.11"), Decimal("1852.11"))

    def test_invalid_decimal_raises(self):
        with self.assertRaises(DatasetLoadError):
            parse_optional_decimal("not-a-number")

    def test_blank_date_is_none(self):
        self.assertIsNone(parse_optional_date(""))

    def test_date_parses(self):
        self.assertEqual(parse_optional_date("2024-03-03"), date(2024, 3, 3))

    def test_pipe_list_splits_and_ignores_blank(self):
        self.assertEqual(parse_pipe_list("education|debt_repayment"), ("education", "debt_repayment"))
        self.assertEqual(parse_pipe_list(""), ())
        self.assertEqual(parse_pipe_list("dining"), ("dining",))

    def test_bool_parsing(self):
        self.assertTrue(parse_bool("true", field_name="x"))
        self.assertFalse(parse_bool("false", field_name="x"))
        with self.assertRaises(DatasetLoadError):
            parse_bool("yes", field_name="x")


class TestLoadDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATASET_DIR)

    def test_row_counts_match_known_facts(self):
        d = self.dataset
        self.assertEqual(len(d.profiles), 275)
        self.assertEqual(len(d.events), 25342)
        self.assertEqual(len(d.exchange_rates), 134)
        self.assertEqual(len(d.payment_options), 790)
        self.assertEqual(len(d.messages), 215)
        self.assertEqual(len(d.images), 16)
        self.assertEqual(len(d.requests), 250)
        self.assertEqual(len(d.sample_requests), 25)

    def test_blank_amount_events_parse_to_none(self):
        blanks = [e for e in self.dataset.events if e.amount is None]
        self.assertEqual(len(blanks), 16)
        ids = {e.event_id for e in blanks}
        self.assertIn("event_7307", ids)

    def test_event_7307_is_blank_amount_usd_taxi(self):
        e = next(e for e in self.dataset.events if e.event_id == "event_7307")
        self.assertIsNone(e.amount)
        self.assertEqual(e.currency, "USD")
        self.assertEqual(e.category, "transport")
        self.assertEqual(e.status, "settled")

    def test_profile_pipe_fields_parsed(self):
        p = next(p for p in self.dataset.profiles if p.user_id == "user_01")
        self.assertEqual(p.financial_priorities, ("education", "debt_repayment"))
        self.assertEqual(p.expense_categories_user_is_willing_to_reduce, ("dining",))
        self.assertIsNone(p.max_installment_months)

    def test_message_datetime_parsed_with_utc(self):
        m = self.dataset.messages[0]
        self.assertIsInstance(m.sent_at, datetime)
        self.assertIsNotNone(m.sent_at.tzinfo)

    def test_sample_requests_carry_solved_output_fields(self):
        s = next(s for s in self.dataset.sample_requests if s.request_id == "request_01")
        self.assertEqual(s.amount_safe_to_pay, Decimal("25256"))
        self.assertEqual(s.affordability_status, "affordable_now")
        self.assertEqual(s.earliest_date_for_full_payment, date(2024, 3, 3))


if __name__ == "__main__":
    unittest.main()
