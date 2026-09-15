"""
Typed data models for the "Buy or Wait?" dataset.

Stage 1 scope only: these classes hold parsed, typed data exactly as it
appears in the CSVs (plus currency-neutral parsing of dates/decimals/lists).
No financial decisions, forecasting, or AI/image interpretation happens here.

All money fields use `decimal.Decimal`, never `float`, so downstream stages
never accumulate binary floating-point rounding error on currency amounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Controlled vocabularies observed in the dataset (see FACTS.md for the
# verification that produced these exact sets). Kept as enums so a future
# stage gets an immediate, loud failure if the dataset ever contains a value
# nobody has accounted for, instead of a silent typo bug.
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    EXPENSE = "expense"
    INCOME = "income"
    SUBSCRIPTION = "subscription"
    DEBT_PAYMENT = "debt_payment"
    REFUND = "refund"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"


class EventDirection(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    # Used only by the 10 `unrealized` / `investment_valuation` events: a
    # non-cash mark-to-market change that must never be treated as
    # available cash (see FACTS.md).
    NON_CASH = "non_cash"


class EventStatus(str, Enum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNREALIZED = "unrealized"


class EventFlexibility(str, Enum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"


class PaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"


class RequestType(str, Enum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


# Currencies actually present in the dataset today. Not used to reject data
# (validate.py reports unknown currencies as issues rather than crashing),
# just documented here for reference.
KNOWN_CURRENCIES = frozenset({"INR", "ZAR", "IDR", "USD", "EUR"})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str  # validated against EventType, kept as str for forward-compat
    description: str
    category: str
    direction: str  # validated against EventDirection
    amount: Optional[Decimal]  # None means blank in the CSV; NEVER coerce to 0
    currency: str
    event_date: date
    settlement_date: Optional[date]  # None for unrealized events
    status: str  # validated against EventStatus
    linked_event_id: Optional[str]
    flexibility: str  # validated against EventFlexibility
    minimum_allowed_amount: Optional[Decimal]

    @property
    def amount_is_blank(self) -> bool:
        return self.amount is None


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # validated against PaymentMethod (full_payment/installments only)
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    file_path: str  # resolved dataset/media/images/<image_id>.png path


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str  # validated against RequestType
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class SampleRequest(Request):
    """A `sample_requests.csv` row: a Request plus the solved output columns.

    These are worked examples for understanding format/style only - the
    problem statement is explicit that they are not ground truth for the
    250 graded requests and must never be used as such.
    """

    amount_safe_to_pay: Decimal = field(default=Decimal("0"))
    affordability_status: str = ""
    recommended_payment_method: str = ""
    payment_plan: str = ""
    earliest_date_for_full_payment: Optional[date] = None
    spending_changes_needed: str = ""
    decision_explanation: str = ""


@dataclass
class Dataset:
    """Everything loaded from dataset/, untouched by any business logic."""

    profiles: list[FinancialProfile]
    events: list[FinancialEvent]
    exchange_rates: list[ExchangeRate]
    payment_options: list[PaymentOption]
    messages: list[Message]
    images: list[ImageRecord]
    requests: list[Request]
    sample_requests: list[SampleRequest]
