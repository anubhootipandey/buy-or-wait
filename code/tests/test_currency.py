import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.currency import MissingExchangeRateError, convert
from data.indexes import build_indexes
from data.loader import load_dataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


class TestCurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATASET_DIR)
        cls.indexes = build_indexes(cls.dataset)

    def test_same_currency_is_noop(self):
        amt = convert(Decimal("100"), "ZAR", "ZAR", date(2024, 1, 1), self.indexes)
        self.assertEqual(amt, Decimal("100"))

    def test_known_pair_converts(self):
        # 2023-10-15 EUR->ZAR rate is 20 per the raw CSV.
        amt = convert(Decimal("10"), "EUR", "ZAR", date(2023, 10, 15), self.indexes)
        self.assertEqual(amt, Decimal("200"))

    def test_missing_rate_raises(self):
        with self.assertRaises(MissingExchangeRateError):
            convert(Decimal("10"), "ZAR", "IDR", date(1999, 1, 1), self.indexes)

    def test_every_foreign_currency_event_has_exact_rate(self):
        """This is the fact that justifies the no-fallback design decision:
        re-verify it every run, not just once during planning."""
        missing = []
        for e in self.dataset.events:
            profile = self.indexes.profiles_by_user.get(e.user_id)
            if profile is None or e.settlement_date is None:
                continue
            if e.currency == profile.home_currency:
                continue
            try:
                convert(Decimal("1"), e.currency, profile.home_currency, e.settlement_date, self.indexes)
            except MissingExchangeRateError:
                missing.append(e.event_id)
        self.assertEqual(missing, [])

    def test_event_7307_pair_has_a_rate(self):
        e = next(e for e in self.dataset.events if e.event_id == "event_7307")
        profile = self.indexes.profiles_by_user[e.user_id]
        # Should not raise.
        convert(Decimal("1"), e.currency, profile.home_currency, e.settlement_date, self.indexes)


if __name__ == "__main__":
    unittest.main()
