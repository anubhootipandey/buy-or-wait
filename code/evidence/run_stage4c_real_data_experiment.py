#!/usr/bin/env python3
"""
Stage 4c real-data integration experiment - deterministic, offline, no
Gemini calls.

This script never calls a model. It:

  1. Runs Stage 4a's deterministic prefilter over the real dataset.
  2. For `needs_escalation` MESSAGES, looks each one up in the existing
     `code/evidence/cache/evidence_cache.json` (the file this project's
     verified real Gemini run already populated). A cache hit is used as
     already-resolved evidence, exactly as Stage 4b's own cache-first
     policy would use it. A cache MISS is left unresolved with an honest
     `offline_experiment_no_cache_entry` reason - this sandbox has no
     network and no `google-genai` package (see `STAGE_4B.md`), so it
     never invents a result for it.
  3. `needs_escalation` IMAGES are left unresolved (Stage 4d, unchanged).
  4. Feeds the resulting `future_amount_change` / `series_terminated`
     facts, grouped by user, into the new Stage 4c-aware
     `engine.forecast.build_forecast(..., evidence_facts=...)` for every
     one of the 250 real requests, alongside a baseline (no-evidence)
     forecast for the same request, and reports the difference.

Run from the repo root:
    python3 code/evidence/run_stage4c_real_data_experiment.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_dataset  # noqa: E402
from data.indexes import build_indexes  # noqa: E402
from engine.forecast import build_forecast  # noqa: E402
from engine.reconciliation import reconcile_user_events  # noqa: E402
from evidence.cache import EvidenceCache  # noqa: E402
from evidence.deterministic import (  # noqa: E402
    DETERMINISTIC_FACT,
    NEEDS_ESCALATION,
    NO_FACT,
    run_prefilter,
)
from evidence.models import NormalizedFact, UnresolvedEvidence  # noqa: E402
from evidence.image_input import UnsupportedImageError
from evidence.resolver import build_context, build_image_context  # noqa: E402

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "evidence_cache.json"

APPLICABLE_FACT_TYPES = ("future_amount_change", "series_terminated")


def resolve_evidence_offline(dataset, indexes, cache_path: Path):
    """Stage 4a (unchanged) + Stage 4b's cache lookup ONLY - never calls a
    model. Returns (facts, unresolved, stats)."""
    prefilter = run_prefilter(dataset, indexes.events_by_id)
    messages_by_id = {m.message_id: m for m in dataset.messages}
    cache = EvidenceCache(path=cache_path)

    facts: list[NormalizedFact] = [
        d.fact for d in prefilter.by_bucket(NO_FACT) if d.fact is not None
    ] + [
        d.fact for d in prefilter.by_bucket(DETERMINISTIC_FACT) if d.fact is not None
    ]
    unresolved: list[UnresolvedEvidence] = [
        UnresolvedEvidence(provenance=d.ref, reason=d.reason)
        for bucket in ("missing_evidence", "unresolved_other")
        for d in prefilter.by_bucket(bucket)
    ]

    images_by_id = {i.image_id: i for i in dataset.images}

    cache_hits = 0
    offline_unresolved = 0
    images_deferred = 0
    for decision in prefilter.by_bucket(NEEDS_ESCALATION):
        if decision.ref.image_id is not None:
            # Stage 4d exists now, but this script is deliberately OFFLINE -
            # it never makes a model call. So an image resolves here only if
            # a previously-validated fact is already in the cache; otherwise
            # it stays unresolved, exactly like an uncached message.
            image = images_by_id[decision.ref.image_id]
            event = (
                indexes.events_by_id.get(image.related_event_id)
                if image.related_event_id else None
            )
            cached_image_fact = None
            if event is not None:
                try:
                    image_ctx = build_image_context(image, event)
                    cached_image_fact = cache.get_image(image_ctx)
                except UnsupportedImageError:
                    cached_image_fact = None
            if cached_image_fact is not None:
                facts.append(cached_image_fact)
                cache_hits += 1
            else:
                unresolved.append(
                    UnresolvedEvidence(
                        provenance=decision.ref,
                        reason="offline_experiment_no_cache_entry (image; no vision call attempted)",
                    )
                )
                images_deferred += 1
            continue
        message = messages_by_id[decision.ref.message_id]
        ctx = build_context(message, indexes)
        cached_fact = cache.get(ctx)
        if cached_fact is not None:
            facts.append(cached_fact)
            cache_hits += 1
        else:
            unresolved.append(
                UnresolvedEvidence(
                    provenance=decision.ref,
                    reason="offline_experiment_no_cache_entry (no network/model call attempted)",
                )
            )
            offline_unresolved += 1

    stats = {
        "prefilter_counts": prefilter.counts(),
        "cache_hits_used": cache_hits,
        "escalated_messages_without_cache_entry": offline_unresolved,
        "escalated_images_deferred_to_4d": images_deferred,
    }
    return facts, unresolved, stats


def main() -> int:
    dataset = load_dataset(DEFAULT_DATASET_DIR)
    indexes = build_indexes(dataset)

    facts, unresolved, stats = resolve_evidence_offline(dataset, indexes, DEFAULT_CACHE_PATH)

    print("--- Offline evidence resolution (Stage 4a + cached Stage 4b only; no model calls) ---")
    for k, v in stats["prefilter_counts"].items():
        print(f"  prefilter[{k}] = {v}")
    print(f"  cache_hits_used = {stats['cache_hits_used']}")
    print(f"  escalated_messages_without_cache_entry = {stats['escalated_messages_without_cache_entry']}")
    print(f"  escalated_images_deferred_to_4d = {stats['escalated_images_deferred_to_4d']}")
    print(f"  total facts available = {len(facts)}")
    print(f"  total unresolved (all reasons) = {len(unresolved)}")

    applicable_facts = [f for f in facts if f.fact_type in APPLICABLE_FACT_TYPES]
    print(f"  facts applicable to Stage 4c (future_amount_change/series_terminated) = {len(applicable_facts)}")

    facts_by_user: dict[str, list[NormalizedFact]] = defaultdict(list)
    for f in applicable_facts:
        facts_by_user[f.user_id].append(f)

    print("\n--- Stage 4c integration across all real requests ---")
    total_requests = 0
    requests_with_applicable_evidence = 0
    total_applied = 0
    total_unresolved_in_engine = 0
    total_amount_changed_occurrences = 0
    total_dropped_by_termination = 0
    breach_changed_count = 0
    min_balance_delta_examples = []

    for request in dataset.requests:
        total_requests += 1
        profile = indexes.profiles_by_user[request.user_id]
        user_events = indexes.events_by_user.get(request.user_id, [])
        state = reconcile_user_events(
            request.user_id, user_events, request.request_date,
            indexes.events_by_id, profile.home_currency, indexes,
        )
        user_facts = tuple(facts_by_user.get(request.user_id, ()))

        baseline = build_forecast(
            state, starting_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
        )
        if not user_facts:
            continue

        requests_with_applicable_evidence += 1
        with_evidence = build_forecast(
            state, starting_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            evidence_facts=user_facts,
        )
        report = with_evidence.evidence_report
        total_applied += len(report.applied)
        total_unresolved_in_engine += len(report.unresolved)
        total_amount_changed_occurrences += report.occurrences_amount_changed
        total_dropped_by_termination += report.occurrences_dropped_by_termination

        if baseline.breaches_minimum() != with_evidence.breaches_minimum():
            breach_changed_count += 1

        delta = with_evidence.min_balance_reached - baseline.min_balance_reached
        if delta != 0:
            min_balance_delta_examples.append((request.request_id, request.user_id, delta))

    print(f"  total requests = {total_requests}")
    print(f"  requests with >=1 applicable evidence fact for that user = {requests_with_applicable_evidence}")
    print(f"  fact-application records emitted (applied, incl. superseded) = {total_applied}")
    print(f"  fact records left unresolved inside the engine (ambiguous/no match) = {total_unresolved_in_engine}")
    print(f"  generated occurrences with an amount changed = {total_amount_changed_occurrences}")
    print(f"  generated occurrences dropped by a termination = {total_dropped_by_termination}")
    print(f"  requests whose breaches_minimum() flipped due to evidence = {breach_changed_count}")
    print(f"  requests with a nonzero min-balance delta from evidence = {len(min_balance_delta_examples)}")
    for rid, uid, delta in min_balance_delta_examples[:10]:
        print(f"    {rid} (user={uid}): min_balance_reached delta = {delta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())