"""
Referential-integrity and sanity checks for the loaded dataset.

This is deliberately structural, not financial: it checks that IDs resolve,
enum-like fields are within the known vocabulary, files on disk exist where
referenced, and known row-count facts still hold. It does NOT check
business rules like "only flexible events may be stopped" - that belongs to
the decision-logic stage, once amount_safe_to_pay etc. are being computed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .indexes import Indexes
from .models import (
    KNOWN_CURRENCIES,
    Dataset,
    EventDirection,
    EventFlexibility,
    EventStatus,
    EventType,
    PaymentMethod,
    RequestType,
)


class Severity(str, Enum):
    ERROR = "ERROR"  # a downstream stage cannot safely proceed
    WARNING = "WARNING"  # worth a human's attention, not fatal


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def add(self, severity: Severity, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


def _enum_values(enum_cls: type[Enum]) -> set[str]:
    return {member.value for member in enum_cls}


def validate_dataset(dataset: Dataset, indexes: Indexes) -> ValidationReport:
    report = ValidationReport()

    report.facts["num_profiles"] = len(dataset.profiles)
    report.facts["num_events"] = len(dataset.events)
    report.facts["num_exchange_rates"] = len(dataset.exchange_rates)
    report.facts["num_payment_options"] = len(dataset.payment_options)
    report.facts["num_messages"] = len(dataset.messages)
    report.facts["num_images"] = len(dataset.images)
    report.facts["num_requests"] = len(dataset.requests)
    report.facts["num_sample_requests"] = len(dataset.sample_requests)

    _check_profiles(dataset, report)
    _check_events(dataset, indexes, report)
    _check_payment_options(dataset, indexes, report)
    _check_messages(dataset, indexes, report)
    _check_images(dataset, indexes, report)
    _check_requests(dataset, indexes, report)
    _check_currency_coverage(dataset, indexes, report)

    return report


def _check_profiles(dataset: Dataset, report: ValidationReport) -> None:
    seen: set[str] = set()
    for p in dataset.profiles:
        if p.user_id in seen:
            report.add(Severity.ERROR, "DUPLICATE_USER_ID", f"Duplicate profile for {p.user_id}")
        seen.add(p.user_id)
        if p.home_currency not in KNOWN_CURRENCIES:
            report.add(
                Severity.WARNING,
                "UNKNOWN_CURRENCY",
                f"{p.user_id}: unrecognized home_currency {p.home_currency!r}",
            )
    report.facts["num_unique_users"] = len(seen)


def _check_events(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    known_event_ids = set(indexes.events_by_id.keys())
    event_types = _enum_values(EventType)
    directions = _enum_values(EventDirection)
    statuses = _enum_values(EventStatus)
    flexibilities = _enum_values(EventFlexibility)

    blank_amount_count = 0
    foreign_currency_count = 0
    linked_count = 0

    for e in dataset.events:
        if e.user_id not in indexes.profiles_by_user:
            report.add(
                Severity.ERROR,
                "EVENT_UNKNOWN_USER",
                f"{e.event_id}: user_id {e.user_id!r} not found in financial_profiles.csv",
            )
        if e.event_type not in event_types:
            report.add(Severity.WARNING, "UNKNOWN_EVENT_TYPE", f"{e.event_id}: event_type {e.event_type!r}")
        if e.direction not in directions:
            report.add(Severity.WARNING, "UNKNOWN_DIRECTION", f"{e.event_id}: direction {e.direction!r}")
        if e.status not in statuses:
            report.add(Severity.WARNING, "UNKNOWN_STATUS", f"{e.event_id}: status {e.status!r}")
        if e.flexibility not in flexibilities:
            report.add(
                Severity.WARNING, "UNKNOWN_FLEXIBILITY", f"{e.event_id}: flexibility {e.flexibility!r}"
            )
        if e.currency not in KNOWN_CURRENCIES:
            report.add(Severity.WARNING, "UNKNOWN_CURRENCY", f"{e.event_id}: currency {e.currency!r}")

        if e.linked_event_id is not None:
            linked_count += 1
            if e.linked_event_id not in known_event_ids:
                report.add(
                    Severity.ERROR,
                    "EVENT_UNKNOWN_LINK",
                    f"{e.event_id}: linked_event_id {e.linked_event_id!r} does not exist",
                )
            elif e.linked_event_id == e.event_id:
                report.add(Severity.ERROR, "EVENT_SELF_LINK", f"{e.event_id}: links to itself")

        if e.amount is None:
            blank_amount_count += 1
            if e.event_id not in indexes.images_by_related_event:
                report.add(
                    Severity.ERROR,
                    "BLANK_AMOUNT_NO_IMAGE",
                    f"{e.event_id}: blank amount but no linked image in images.csv",
                )

        profile = indexes.profiles_by_user.get(e.user_id)
        if profile is not None and e.currency != profile.home_currency:
            foreign_currency_count += 1

        if e.status == EventStatus.UNREALIZED.value and e.settlement_date is not None:
            report.add(
                Severity.WARNING,
                "UNREALIZED_HAS_SETTLEMENT_DATE",
                f"{e.event_id}: unrealized event unexpectedly has a settlement_date",
            )
        if e.status != EventStatus.UNREALIZED.value and e.settlement_date is None:
            report.add(
                Severity.WARNING,
                "NON_UNREALIZED_MISSING_SETTLEMENT_DATE",
                f"{e.event_id}: status={e.status!r} has no settlement_date",
            )

    report.facts["num_blank_amount_events"] = blank_amount_count
    report.facts["num_foreign_currency_events"] = foreign_currency_count
    report.facts["num_linked_events"] = linked_count


def _check_payment_options(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    known_request_ids = set(indexes.requests_by_id.keys()) | {
        r.request_id for r in dataset.sample_requests
    }
    allowed_methods = {PaymentMethod.FULL_PAYMENT.value, PaymentMethod.INSTALLMENTS.value}

    per_request: dict[str, int] = {}
    for po in dataset.payment_options:
        per_request[po.request_id] = per_request.get(po.request_id, 0) + 1
        if po.request_id not in known_request_ids:
            report.add(
                Severity.ERROR,
                "PAYMENT_OPTION_UNKNOWN_REQUEST",
                f"{po.payment_option_id}: request_id {po.request_id!r} not found",
            )
        if po.payment_method not in allowed_methods:
            report.add(
                Severity.WARNING,
                "UNEXPECTED_PAYMENT_METHOD",
                f"{po.payment_option_id}: payment_method {po.payment_method!r} "
                "(expected only full_payment/installments to appear as supplied options)",
            )

    for request_id, count in per_request.items():
        if not (2 <= count <= 4):
            report.add(
                Severity.WARNING,
                "UNEXPECTED_OPTION_COUNT",
                f"{request_id}: has {count} payment options (spec says 2-4)",
            )


def _check_messages(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    known_request_ids = set(indexes.requests_by_id.keys()) | {
        r.request_id for r in dataset.sample_requests
    }
    for m in dataset.messages:
        if m.user_id not in indexes.profiles_by_user:
            report.add(
                Severity.ERROR, "MESSAGE_UNKNOWN_USER", f"{m.message_id}: user_id {m.user_id!r} not found"
            )
        if m.request_id is not None and m.request_id not in known_request_ids:
            report.add(
                Severity.ERROR,
                "MESSAGE_UNKNOWN_REQUEST",
                f"{m.message_id}: request_id {m.request_id!r} not found",
            )
        if m.related_event_id is not None and m.related_event_id not in indexes.events_by_id:
            report.add(
                Severity.ERROR,
                "MESSAGE_UNKNOWN_EVENT",
                f"{m.message_id}: related_event_id {m.related_event_id!r} not found",
            )
        # Cross-check: if a message names both a request and an event, the
        # event should belong to the same user as the request.
        if m.request_id is not None and m.related_event_id is not None:
            req = indexes.requests_by_id.get(m.request_id)
            evt = indexes.events_by_id.get(m.related_event_id)
            if req is not None and evt is not None and req.user_id != evt.user_id:
                report.add(
                    Severity.WARNING,
                    "MESSAGE_USER_MISMATCH",
                    f"{m.message_id}: request {m.request_id} user {req.user_id} != "
                    f"event {m.related_event_id} user {evt.user_id}",
                )


def _check_images(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    for img in dataset.images:
        if img.user_id not in indexes.profiles_by_user:
            report.add(
                Severity.ERROR, "IMAGE_UNKNOWN_USER", f"{img.image_id}: user_id {img.user_id!r} not found"
            )
        if img.related_event_id is not None and img.related_event_id not in indexes.events_by_id:
            report.add(
                Severity.ERROR,
                "IMAGE_UNKNOWN_EVENT",
                f"{img.image_id}: related_event_id {img.related_event_id!r} not found",
            )
        if not os.path.isfile(img.file_path):
            report.add(
                Severity.ERROR, "IMAGE_FILE_MISSING", f"{img.image_id}: file not found at {img.file_path}"
            )


def _check_requests(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    request_types = _enum_values(RequestType)
    all_requests = list(dataset.requests) + list(dataset.sample_requests)
    seen_ids: set[str] = set()
    for r in all_requests:
        if r.request_id in seen_ids:
            report.add(Severity.ERROR, "DUPLICATE_REQUEST_ID", f"Duplicate request_id {r.request_id}")
        seen_ids.add(r.request_id)
        if r.user_id not in indexes.profiles_by_user:
            report.add(
                Severity.ERROR, "REQUEST_UNKNOWN_USER", f"{r.request_id}: user_id {r.user_id!r} not found"
            )
        if r.request_type not in request_types:
            report.add(
                Severity.WARNING, "UNKNOWN_REQUEST_TYPE", f"{r.request_id}: request_type {r.request_type!r}"
            )
        if r.requested_amount < 0:
            report.add(
                Severity.ERROR,
                "NEGATIVE_REQUESTED_AMOUNT",
                f"{r.request_id}: requested_amount is negative",
            )
        if r.desired_completion_date < r.request_date:
            report.add(
                Severity.WARNING,
                "COMPLETION_BEFORE_REQUEST",
                f"{r.request_id}: desired_completion_date before request_date",
            )


def _check_currency_coverage(dataset: Dataset, indexes: Indexes, report: ValidationReport) -> None:
    """For every foreign-currency event, confirm an exact-date rate exists.

    This is the check that turned "prior recon claim" into FACT: as of this
    dataset, the count of missing rates is 0. Kept as an automated check so
    it re-verifies itself if the dataset ever changes.
    """
    missing = 0
    for e in dataset.events:
        profile = indexes.profiles_by_user.get(e.user_id)
        if profile is None or e.settlement_date is None:
            continue
        if e.currency == profile.home_currency:
            continue
        key = (e.settlement_date, e.currency, profile.home_currency)
        if key not in indexes.exchange_rate_lookup:
            missing += 1
            report.add(
                Severity.ERROR,
                "MISSING_EXCHANGE_RATE",
                f"{e.event_id}: no rate for {e.currency}->{profile.home_currency} "
                f"on {e.settlement_date.isoformat()}",
            )
    report.facts["num_missing_exchange_rates"] = missing
