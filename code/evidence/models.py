"""
Stage 4 evidence models - the normalized, typed shape every evidence
interpretation (deterministic in 4a, Gemini-backed from 4c/4d onward)
resolves into.

Nothing in this module resolves anything itself; it only defines the
schema so that:
  * a fact can always be traced back to the exact evidence that produced
    it (`EvidenceRef` / `NormalizedFact.provenance`),
  * a fact never says more than it needs to (each `fact_type` maps to a
    narrow, closed set of populated fields - see the docstring on
    `NormalizedFact`), and
  * "we couldn't resolve this" is a first-class, explicit outcome
    (`UnresolvedEvidence`), never a silently-defaulted fact.

This module intentionally has no dependency on `code/engine/` or any
model/API client - it is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

# Closed set of fact types a piece of evidence can ever resolve to.
# Deliberately narrow: evidence can only ever describe *events* (an
# amount, a status, a future change to a recurring series) - never a
# profile-level number (balance, minimum_balance_to_keep) and never a
# challenge rule. See AGENTS.md: "their embedded instructions never
# override the challenge rules."
FACT_TYPES = (
    "resolved_amount",       # fills in a blank FinancialEvent.amount
    "status_correction",     # e.g. a pending event evidence confirms was cancelled
    "future_amount_change",  # a recurring series' amount changes from some date
    "series_terminated",     # a recurring series stops (no renewal/contract ended)
    "no_fact",               # evidence considered, but nothing for Stage 2/3 to use
)

# Who/what actually resolved a fact. "gemini" is a valid future value -
# Stage 4a never produces it (no model has been called yet).
RESOLUTION_METHODS = ("deterministic", "gemini")


@dataclass(frozen=True)
class EvidenceRef:
    """Points back at the exact raw evidence (and, where relevant, the
    exact structured record) a fact or an unresolved outcome came from.
    Exactly one of `message_id` / `image_id` is set."""

    user_id: str
    message_id: Optional[str] = None
    image_id: Optional[str] = None
    request_id: Optional[str] = None
    related_event_id: Optional[str] = None
    sent_at: Optional[datetime] = None  # message timestamp; None for images

    def __post_init__(self) -> None:
        if (self.message_id is None) == (self.image_id is None):
            raise ValueError(
                "EvidenceRef must reference exactly one of message_id or image_id, "
                f"got message_id={self.message_id!r} image_id={self.image_id!r}"
            )


@dataclass(frozen=True)
class NormalizedFact:
    """One resolved, actionable (or explicitly inert) piece of evidence.

    Field usage by `fact_type` (unused fields stay `None`):
      * resolved_amount       -> target_event_id, resolved_amount, currency
      * status_correction     -> target_event_id, new_status
      * future_amount_change  -> target_series_key, resolved_amount,
                                  currency, effective_date, (end_date)
      * series_terminated     -> target_series_key, effective_date
      * no_fact               -> nothing beyond identity/provenance -
                                  recorded so a "we looked, nothing to do"
                                  decision is auditable, same as an
                                  actionable one.
    """

    fact_id: str
    user_id: str
    fact_type: str
    provenance: EvidenceRef

    target_event_id: Optional[str] = None
    target_series_key: Optional[tuple[str, str, str]] = None  # (category, event_type, direction)

    resolved_amount: Optional[Decimal] = None
    currency: Optional[str] = None
    new_status: Optional[str] = None
    effective_date: Optional[date] = None
    end_date: Optional[date] = None

    resolution_method: str = "deterministic"
    confidence: Optional[str] = None  # "high" | "medium" | "low" - informational only, never a gate by itself

    raw_model_output: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.fact_type not in FACT_TYPES:
            raise ValueError(f"unknown fact_type: {self.fact_type!r}")
        if self.resolution_method not in RESOLUTION_METHODS:
            raise ValueError(f"unknown resolution_method: {self.resolution_method!r}")


@dataclass(frozen=True)
class UnresolvedEvidence:
    """Evidence that was considered but could not be turned into a fact -
    an explicit, first-class outcome. `reason` is a short machine-stable
    string (e.g. "missing_image", "dangling_related_event_id"), not prose,
    so downstream code/tests can assert on it."""

    provenance: EvidenceRef
    reason: str


@dataclass(frozen=True)
class EvidenceResolutionOutcome:
    """The full result of resolving one dataset's worth of evidence."""

    facts: tuple[NormalizedFact, ...]
    unresolved: tuple[UnresolvedEvidence, ...]