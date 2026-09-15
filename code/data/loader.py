"""
CSV -> typed-record loading for the "Buy or Wait?" dataset.

Design decisions (see DECISIONS.md for the full rationale):
  * stdlib `csv` only - no pandas. The dataset is small (~25k events is the
    largest file) and stdlib csv + dataclasses is more than fast enough,
    while keeping the dependency footprint at zero for a 4GB-RAM laptop.
  * Blank amounts are parsed to `None`, never to `Decimal("0")`. The problem
    statement is explicit that a blank amount must be resolved from an
    image, not treated as zero - Stage 1 must not quietly destroy that
    distinction.
  * Dates are parsed to `datetime.date`; the one datetime field
    (`messages.sent_at`) is parsed to `datetime.datetime` (it carries a
    time-of-day and a `Z` UTC suffix).
  * Every loader function fails loudly (raises) on a malformed *required*
    field. Blank *optional* fields are just `None` / empty tuple, not
    an error - that distinction is what validate.py checks structurally
    afterwards.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import (
    Dataset,
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
    SampleRequest,
)


class DatasetLoadError(ValueError):
    """Raised when a required field is missing or malformed."""


# ---------------------------------------------------------------------------
# Small parsing helpers - centralised so every loader function treats a given
# CSV convention (blank string, pipe list, true/false) identically.
# ---------------------------------------------------------------------------


def _blank_to_none(value: str) -> str | None:
    value = value.strip()
    return value if value != "" else None


def parse_optional_decimal(value: str) -> Decimal | None:
    value = value.strip()
    if value == "":
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise DatasetLoadError(f"Invalid decimal amount: {value!r}") from exc


def parse_decimal(value: str, *, field_name: str) -> Decimal:
    parsed = parse_optional_decimal(value)
    if parsed is None:
        raise DatasetLoadError(f"Required decimal field {field_name!r} is blank")
    return parsed


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise DatasetLoadError(f"Invalid integer: {value!r}") from exc


def parse_int(value: str, *, field_name: str) -> int:
    parsed = parse_optional_int(value)
    if parsed is None:
        raise DatasetLoadError(f"Required integer field {field_name!r} is blank")
    return parsed


def parse_optional_date(value: str) -> date | None:
    value = value.strip()
    if value == "":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DatasetLoadError(f"Invalid date (expected YYYY-MM-DD): {value!r}") from exc


def parse_date(value: str, *, field_name: str) -> date:
    parsed = parse_optional_date(value)
    if parsed is None:
        raise DatasetLoadError(f"Required date field {field_name!r} is blank")
    return parsed


def parse_datetime_z(value: str, *, field_name: str) -> datetime:
    value = value.strip()
    if value == "":
        raise DatasetLoadError(f"Required datetime field {field_name!r} is blank")
    try:
        # Python 3.11+ handles a trailing 'Z' directly; done defensively
        # in case this ever runs under an older interpreter.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetLoadError(f"Invalid ISO-8601 datetime: {value!r}") from exc


def parse_bool(value: str, *, field_name: str) -> bool:
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise DatasetLoadError(f"Field {field_name!r} must be 'true'/'false', got {value!r}")


def parse_pipe_list(value: str) -> tuple[str, ...]:
    value = value.strip()
    if value == "":
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip() != "")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Per-file loaders
# ---------------------------------------------------------------------------


def load_profiles(dataset_dir: Path) -> list[FinancialProfile]:
    out = []
    for row in _rows(dataset_dir / "financial_profiles.csv"):
        out.append(
            FinancialProfile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                current_available_balance=parse_decimal(
                    row["current_available_balance"], field_name="current_available_balance"
                ),
                minimum_balance_to_keep=parse_decimal(
                    row["minimum_balance_to_keep"], field_name="minimum_balance_to_keep"
                ),
                financial_priorities=parse_pipe_list(row["financial_priorities"]),
                expense_categories_to_protect=parse_pipe_list(row["expense_categories_to_protect"]),
                expense_categories_user_is_willing_to_reduce=parse_pipe_list(
                    row["expense_categories_user_is_willing_to_reduce"]
                ),
                expense_categories_user_is_willing_to_stop=parse_pipe_list(
                    row["expense_categories_user_is_willing_to_stop"]
                ),
                payment_methods_user_will_consider=parse_pipe_list(
                    row["payment_methods_user_will_consider"]
                ),
                max_installment_months=parse_optional_int(row["max_installment_months"]),
            )
        )
    return out


def load_events(dataset_dir: Path) -> list[FinancialEvent]:
    out = []
    for row in _rows(dataset_dir / "financial_events.csv"):
        out.append(
            FinancialEvent(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"],
                description=row["description"],
                category=row["category"],
                direction=row["direction"],
                amount=parse_optional_decimal(row["amount"]),  # blank -> None, never 0
                currency=row["currency"],
                event_date=parse_date(row["event_date"], field_name="event_date"),
                settlement_date=parse_optional_date(row["settlement_date"]),
                status=row["status"],
                linked_event_id=_blank_to_none(row["linked_event_id"]),
                flexibility=row["flexibility"],
                minimum_allowed_amount=parse_optional_decimal(row["minimum_allowed_amount"]),
            )
        )
    return out


def load_exchange_rates(dataset_dir: Path) -> list[ExchangeRate]:
    out = []
    for row in _rows(dataset_dir / "exchange_rates.csv"):
        out.append(
            ExchangeRate(
                rate_date=parse_date(row["rate_date"], field_name="rate_date"),
                from_currency=row["from_currency"],
                to_currency=row["to_currency"],
                rate=parse_decimal(row["rate"], field_name="rate"),
            )
        )
    return out


def load_payment_options(dataset_dir: Path) -> list[PaymentOption]:
    out = []
    for row in _rows(dataset_dir / "request_payment_options.csv"):
        out.append(
            PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"],
                payment_amount=parse_decimal(row["payment_amount"], field_name="payment_amount"),
                number_of_payments=parse_int(row["number_of_payments"], field_name="number_of_payments"),
                first_payment_date=parse_date(row["first_payment_date"], field_name="first_payment_date"),
                payment_frequency_days=parse_optional_int(row["payment_frequency_days"]),
                financing_fee=parse_decimal(row["financing_fee"], field_name="financing_fee"),
                total_payable_amount=parse_decimal(
                    row["total_payable_amount"], field_name="total_payable_amount"
                ),
            )
        )
    return out


def load_messages(dataset_dir: Path) -> list[Message]:
    out = []
    for row in _rows(dataset_dir / "messages.csv"):
        out.append(
            Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=_blank_to_none(row["request_id"]),
                related_event_id=_blank_to_none(row["related_event_id"]),
                sent_at=parse_datetime_z(row["sent_at"], field_name="sent_at"),
                source_type=row["source_type"],
                message_text=row["message_text"],
            )
        )
    return out


def load_images(dataset_dir: Path) -> list[ImageRecord]:
    out = []
    media_dir = dataset_dir / "media" / "images"
    for row in _rows(dataset_dir / "images.csv"):
        image_id = row["image_id"]
        out.append(
            ImageRecord(
                image_id=image_id,
                user_id=row["user_id"],
                request_id=_blank_to_none(row["request_id"]),
                related_event_id=_blank_to_none(row["related_event_id"]),
                file_path=str(media_dir / f"{image_id}.png"),
            )
        )
    return out


def load_requests(dataset_dir: Path, filename: str = "requests.csv") -> list[Request]:
    out = []
    for row in _rows(dataset_dir / filename):
        out.append(
            Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=parse_date(row["request_date"], field_name="request_date"),
                request_type=row["request_type"],
                requested_amount=parse_decimal(row["requested_amount"], field_name="requested_amount"),
                desired_completion_date=parse_date(
                    row["desired_completion_date"], field_name="desired_completion_date"
                ),
                allows_partial_payment=parse_bool(
                    row["allows_partial_payment"], field_name="allows_partial_payment"
                ),
                request_text=row["request_text"],
            )
        )
    return out


def load_sample_requests(dataset_dir: Path) -> list[SampleRequest]:
    out = []
    for row in _rows(dataset_dir / "sample_requests.csv"):
        out.append(
            SampleRequest(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=parse_date(row["request_date"], field_name="request_date"),
                request_type=row["request_type"],
                requested_amount=parse_decimal(row["requested_amount"], field_name="requested_amount"),
                desired_completion_date=parse_date(
                    row["desired_completion_date"], field_name="desired_completion_date"
                ),
                allows_partial_payment=parse_bool(
                    row["allows_partial_payment"], field_name="allows_partial_payment"
                ),
                request_text=row["request_text"],
                amount_safe_to_pay=parse_decimal(
                    row["amount_safe_to_pay"], field_name="amount_safe_to_pay"
                ),
                affordability_status=row["affordability_status"],
                recommended_payment_method=row["recommended_payment_method"],
                payment_plan=row["payment_plan"],
                earliest_date_for_full_payment=parse_optional_date(
                    row["earliest_date_for_full_payment"]
                ),
                spending_changes_needed=row["spending_changes_needed"],
                decision_explanation=row["decision_explanation"],
            )
        )
    return out


def load_dataset(dataset_dir: Path) -> Dataset:
    """Load every CSV in `dataset_dir` into typed records.

    `dataset_dir` should point at the repo's `dataset/` folder.
    """
    dataset_dir = Path(dataset_dir)
    return Dataset(
        profiles=load_profiles(dataset_dir),
        events=load_events(dataset_dir),
        exchange_rates=load_exchange_rates(dataset_dir),
        payment_options=load_payment_options(dataset_dir),
        messages=load_messages(dataset_dir),
        images=load_images(dataset_dir),
        requests=load_requests(dataset_dir),
        sample_requests=load_sample_requests(dataset_dir),
    )
