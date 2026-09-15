"""
Stage 3 plan ranking.

Implements the challenge's exact ranking order as an explicit lexicographic
comparison over concrete fields - never a weighted or fuzzy score:

  1. complete the full request by `desired_completion_date`
  2. require no spending changes
  3. minimize the total amount paid
  4. start payment earlier
  5. use fewer payments
  6. use the lowest `payment_option_id` as the final tie-breaker
"""

from __future__ import annotations

import re
from datetime import date

from .models import PlanCandidate

_TRAILING_DIGITS = re.compile(r"(\d+)$")


def _payment_option_sort_key(payment_option_id: str | None) -> tuple[int, int]:
    """Lower is better. A real `payment_option_id` sorts by its numeric
    suffix; candidates with no `payment_option_id` (full_payment,
    partial_payment, wait) only reach this field when every earlier rule
    is already tied, so their exact placement here is a formality - they
    are given a constant key so the comparison stays deterministic."""
    if payment_option_id is None:
        return (1, 0)
    match = _TRAILING_DIGITS.search(payment_option_id)
    return (0, int(match.group(1)) if match else 0)


def rank_key(candidate: PlanCandidate, desired_completion_date: date):
    completes_late = 0 if candidate.completes_on <= desired_completion_date else 1
    uses_changes = 1 if candidate.uses_spending_changes else 0
    return (
        completes_late,
        uses_changes,
        candidate.total_amount_paid,
        candidate.starts_on,
        candidate.num_payments,
        _payment_option_sort_key(candidate.payment_option_id),
    )


def choose_best(
    candidates: list[PlanCandidate], desired_completion_date: date
) -> PlanCandidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: rank_key(c, desired_completion_date))
