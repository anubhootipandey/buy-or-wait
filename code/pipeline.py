"""
The production pipeline for Buy or Wait?.

One place that owns the full path from raw dataset to final predictions:

    data
      -> deterministic prefilter + evidence resolution (Stage 4a/4b/4d)
      -> validated NormalizedFacts
      -> resolved_amount facts applied to blank events (Stage 4 apply)
      -> reconciliation + 90-day forecast, with series-level facts
         applied by Stage 4c inside the forecast
      -> Stage 3 planner
      -> output.csv rows

This module exists so that `main.py` (the CLI) and `final_submission.py`
(the submission runner) execute *the same* pipeline rather than two
similar-looking copies of it. It contains no new financial logic: every
step delegates to the module that already owned it.

Two properties are deliberate:

  * Evidence is read from the on-disk cache by default, so the whole
    pipeline runs fully offline, with no API key and no network. Live
    model resolution is opt-in (`resolve_evidence_live`).
  * `resolved_amount` facts are applied to the dataset BEFORE indexing,
    so reconciliation, recurrence detection and forecasting all see the
    real amounts. Series-level facts are passed through to the planner,
    which hands them to Stage 4c. Facts are never applied twice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Optional, Sequence

from data.indexes import Indexes, build_indexes
from data.loader import load_dataset
from data.models import Dataset, FinancialProfile

from evidence.apply_facts import ApplyAmountsReport, apply_resolved_amounts
from evidence.cache import EvidenceCache
from evidence.models import NormalizedFact

from planner import plan_request

COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# Fact types that fill in a blank amount on an existing event. These are
# applied to the dataset itself; everything else is series-level and is
# handed to the planner (and from there to Stage 4c).
EVENT_LEVEL_FACT_TYPES = ("resolved_amount",)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_dataset_dir() -> Path:
    return repo_root() / "dataset"


def default_cache_path() -> Path:
    return repo_root() / "code" / "evidence" / "cache" / "evidence_cache.json"


def default_output_path() -> Path:
    return repo_root() / "output.csv"


def fmt_amount(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def load_facts_from_cache(cache_path: Optional[Path] = None) -> list[NormalizedFact]:
    """Load every previously-validated fact from the evidence cache.

    Offline and dependency-free: no model client is constructed and no
    API key is needed. A missing cache file is not an error - it simply
    means no evidence is available yet, and the pipeline runs on the
    structured dataset alone.
    """
    path = cache_path or default_cache_path()
    if not path.is_file():
        return []
    return EvidenceCache(path).all_facts()


def resolve_evidence_live(
    dataset: Dataset, indexes: Indexes, cache_path: Optional[Path] = None
) -> tuple[list[NormalizedFact], dict[str, int]]:
    """Run the FULL evidence pipeline against the real model: Stage 4a's
    deterministic prefilter, then Stage 4b text resolution and Stage 4d
    image resolution for whatever the prefilter escalated.

    Opt-in only. Requires `google-genai` and `GEMINI_API_KEY`, consumes
    API quota, and is never reached by the default (cache-backed) path or
    by any test. Newly validated facts are written to the same cache the
    offline path reads, so a live run makes subsequent runs offline.
    """
    # Imported lazily so the offline pipeline never needs the SDK present.
    from evidence.ai_client import GeminiAIClient
    from evidence.resolver import run_stage4b

    path = cache_path or default_cache_path()
    cache = EvidenceCache(path)
    client = GeminiAIClient()

    outcome, ai_outcomes = run_stage4b(dataset, indexes, client, cache)
    cache.save()

    stats = {
        "facts": len(outcome.facts),
        "unresolved": len(outcome.unresolved),
        "model_calls": sum(o.api_calls for o in ai_outcomes),
        "cache_hits": sum(1 for o in ai_outcomes if o.cache_hit),
    }
    return list(outcome.facts), stats


def split_facts(
    facts: Iterable[NormalizedFact],
) -> tuple[list[NormalizedFact], list[NormalizedFact]]:
    """Split facts into (event-level amount facts, series-level facts).

    The two groups enter the engine at different points and must never
    both be applied to the same fact - see this module's docstring.
    """
    facts = list(facts)
    event_level = [f for f in facts if f.fact_type in EVENT_LEVEL_FACT_TYPES]
    series_level = [f for f in facts if f.fact_type not in EVENT_LEVEL_FACT_TYPES]
    return event_level, series_level


def group_facts_by_user(
    facts: Iterable[NormalizedFact],
) -> dict[str, tuple[NormalizedFact, ...]]:
    grouped: dict[str, list[NormalizedFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.user_id, []).append(fact)
    return {user: tuple(items) for user, items in grouped.items()}


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedPipeline:
    """A dataset with evidence applied, ready to forecast and plan."""

    dataset: Dataset
    indexes: Indexes
    facts_by_user: dict[str, tuple[NormalizedFact, ...]]
    apply_report: ApplyAmountsReport
    total_facts: int

    def facts_for(self, user_id: str) -> tuple[NormalizedFact, ...]:
        return self.facts_by_user.get(user_id, ())

    @property
    def image_derived_count(self) -> int:
        return sum(1 for a in self.apply_report.applied if a.image_id is not None)


def prepare(dataset: Dataset, facts: Sequence[NormalizedFact]) -> PreparedPipeline:
    """Apply evidence to `dataset` and build the indexes the engine uses.

    Event-level amount facts are applied here, before `build_indexes`, so
    every downstream stage sees the resolved amounts. Series-level facts
    are grouped per user and carried for the planner.
    """
    event_level, series_level = split_facts(facts)
    new_dataset, apply_report = apply_resolved_amounts(dataset, event_level)
    return PreparedPipeline(
        dataset=new_dataset,
        indexes=build_indexes(new_dataset),
        facts_by_user=group_facts_by_user(series_level),
        apply_report=apply_report,
        total_facts=len(facts),
    )


def load_and_prepare(
    dataset_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    use_evidence: bool = True,
    resolve_live: bool = False,
) -> PreparedPipeline:
    """Load the dataset and apply evidence to it.

    `use_evidence=False` reproduces the pre-evidence pipeline exactly (no
    facts loaded, nothing applied), which is useful for isolating whether
    a difference in output came from evidence or from the engine.
    """
    dataset = load_dataset(dataset_dir or default_dataset_dir())

    if not use_evidence:
        return prepare(dataset, [])

    if resolve_live:
        facts, _stats = resolve_evidence_live(dataset, build_indexes(dataset), cache_path)
    else:
        facts = load_facts_from_cache(cache_path)
    return prepare(dataset, facts)


# ---------------------------------------------------------------------------
# Planning + rendering
# ---------------------------------------------------------------------------


def render_result(result, profile: FinancialProfile) -> list[str]:
    """Render one `PlanningResult` as an output.csv row.

    Moved here unchanged from `final_submission.py` so the CLI and the
    submission runner cannot drift apart in column order or wording.
    """
    plan = result.chosen_plan
    if plan is None:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan = "none"
        changes = "none"
        if result.earliest_date_for_full_payment is None:
            earliest = ""
        else:
            earliest = result.earliest_date_for_full_payment.isoformat()
        explanation = (
            f"Do not proceed by the requested deadline. The forecast cannot safely "
            f"complete {fmt_amount(result.requested_amount)} {profile.home_currency} "
            f"while maintaining the {fmt_amount(profile.minimum_balance_to_keep)} "
            f"{profile.home_currency} minimum."
        )
    else:
        status = plan.affordability_status
        method = plan.method
        payment_plan = "|".join(
            f"{p.date.isoformat()}:{fmt_amount(p.amount)}" for p in plan.payments
        )
        earliest = (
            result.earliest_date_for_full_payment.isoformat()
            if result.earliest_date_for_full_payment else ""
        )
        if not plan.spending_changes:
            changes = "none"
        else:
            parts = []
            for c in plan.spending_changes:
                if c.kind == "stop":
                    parts.append(f"stop:{c.event_id}")
                else:
                    parts.append(f"reduce_to:{c.event_id}:{fmt_amount(c.adjusted_amount)}")
            changes = "|".join(parts)

        if method == "full_payment" and status == "affordable_now":
            lead = f"Pay {fmt_amount(result.requested_amount)} {profile.home_currency} in full today."
        elif method == "wait":
            lead = f"Wait until {plan.payments[0].date.isoformat()}, then pay in full."
        elif method == "installments":
            lead = (
                f"Use {len(plan.payments)} supplied installments of "
                f"{fmt_amount(plan.payments[0].amount)} {profile.home_currency}."
            )
        elif method == "partial_payment":
            lead = (
                f"Pay {fmt_amount(plan.payments[0].amount)} {profile.home_currency} today "
                f"and the remainder on {plan.payments[1].date.isoformat()}."
            )
        else:
            lead = f"Use the {method} plan."
        if plan.spending_changes:
            lead += " Required flexible-spending changes: " + changes + "."
        explanation = (
            lead + f" The selected plan keeps the forecast above the "
            f"{fmt_amount(profile.minimum_balance_to_keep)} {profile.home_currency} "
            f"minimum."
        )
    return [
        result.request_id,
        fmt_amount(result.amount_safe_to_pay),
        status,
        method,
        payment_plan,
        earliest,
        changes,
        explanation,
    ]


def plan_all(prepared: PreparedPipeline) -> list[list[str]]:
    """Plan every request in the prepared dataset, in dataset order."""
    rows = []
    for request in prepared.dataset.requests:
        profile = prepared.indexes.profiles_by_user[request.user_id]
        result = plan_request(
            request,
            profile,
            prepared.dataset,
            prepared.indexes,
            evidence_facts=prepared.facts_for(request.user_id),
        )
        rows.append(render_result(result, profile))
    return rows


def write_output_csv(rows: Sequence[Sequence[str]], path: Optional[Path] = None) -> Path:
    out = path or default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return out


__all__ = [
    "COLUMNS",
    "PreparedPipeline",
    "default_cache_path",
    "default_dataset_dir",
    "default_output_path",
    "fmt_amount",
    "group_facts_by_user",
    "load_and_prepare",
    "load_facts_from_cache",
    "plan_all",
    "prepare",
    "render_result",
    "resolve_evidence_live",
    "split_facts",
    "write_output_csv",
]
