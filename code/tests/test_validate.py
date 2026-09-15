import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.indexes import build_indexes
from data.loader import load_dataset
from data.validate import Severity, validate_dataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


class TestValidateRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATASET_DIR)
        cls.indexes = build_indexes(cls.dataset)
        cls.report = validate_dataset(cls.dataset, cls.indexes)

    def test_no_errors_on_real_dataset(self):
        self.assertEqual(self.report.errors, [], msg=[i.message for i in self.report.errors])

    def test_no_warnings_on_real_dataset(self):
        self.assertEqual(self.report.warnings, [], msg=[i.message for i in self.report.warnings])

    def test_facts_match_known_counts(self):
        f = self.report.facts
        self.assertEqual(f["num_profiles"], 275)
        self.assertEqual(f["num_events"], 25342)
        self.assertEqual(f["num_payment_options"], 790)
        self.assertEqual(f["num_messages"], 215)
        self.assertEqual(f["num_images"], 16)
        self.assertEqual(f["num_requests"], 250)
        self.assertEqual(f["num_sample_requests"], 25)
        self.assertEqual(f["num_blank_amount_events"], 16)
        self.assertEqual(f["num_foreign_currency_events"], 140)
        self.assertEqual(f["num_linked_events"], 58)
        self.assertEqual(f["num_missing_exchange_rates"], 0)


class TestValidateCatchesInjectedProblems(unittest.TestCase):
    """Mutate a copy of the real dataset to confirm each check actually
    fires, rather than only ever running against already-clean data."""

    def setUp(self):
        self.dataset = load_dataset(DATASET_DIR)

    def test_catches_event_with_unknown_user(self):
        broken = copy.deepcopy(self.dataset)
        bad_event = replace(broken.events[0], user_id="user_does_not_exist")
        broken.events[0] = bad_event
        indexes = build_indexes(broken)
        report = validate_dataset(broken, indexes)
        codes = {i.code for i in report.errors}
        self.assertIn("EVENT_UNKNOWN_USER", codes)

    def test_catches_dangling_linked_event(self):
        broken = copy.deepcopy(self.dataset)
        bad_event = replace(broken.events[0], linked_event_id="event_does_not_exist")
        broken.events[0] = bad_event
        indexes = build_indexes(broken)
        report = validate_dataset(broken, indexes)
        codes = {i.code for i in report.errors}
        self.assertIn("EVENT_UNKNOWN_LINK", codes)

    def test_catches_blank_amount_event_with_no_image(self):
        broken = copy.deepcopy(self.dataset)
        # Take a normal event and blank out its amount without adding an image.
        idx = next(i for i, e in enumerate(broken.events) if e.amount is not None)
        broken.events[idx] = replace(broken.events[idx], amount=None)
        indexes = build_indexes(broken)
        report = validate_dataset(broken, indexes)
        codes = {i.code for i in report.errors}
        self.assertIn("BLANK_AMOUNT_NO_IMAGE", codes)

    def test_catches_duplicate_request_id(self):
        broken = copy.deepcopy(self.dataset)
        dup = replace(broken.requests[1], request_id=broken.requests[0].request_id)
        broken.requests[1] = dup
        indexes = build_indexes(broken)
        report = validate_dataset(broken, indexes)
        codes = {i.code for i in report.errors}
        self.assertIn("DUPLICATE_REQUEST_ID", codes)

    def test_catches_payment_option_for_unknown_request(self):
        broken = copy.deepcopy(self.dataset)
        bad_po = replace(broken.payment_options[0], request_id="request_does_not_exist")
        broken.payment_options[0] = bad_po
        indexes = build_indexes(broken)
        report = validate_dataset(broken, indexes)
        codes = {i.code for i in report.errors}
        self.assertIn("PAYMENT_OPTION_UNKNOWN_REQUEST", codes)


if __name__ == "__main__":
    unittest.main()
