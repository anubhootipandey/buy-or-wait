#!/usr/bin/env python3
"""
Stage 4d real-API vision experiment runner.

Runs the image-evidence resolver against the real dataset's escalated
images using the real Gemini API (never a fake/mock client - that's what
the unit tests are for). Same staged usage as the text runner:

    python3 code/evidence/run_vision_experiment.py --limit 1
    # inspect the result for semantic correctness, THEN:
    python3 code/evidence/run_vision_experiment.py --limit 5
    # and finally, to populate the cache for the submission runner:
    python3 code/evidence/run_vision_experiment.py --all

Requires the GEMINI_API_KEY environment variable. Never reads a key from
a file or a hardcoded value, and never writes one anywhere.

Successfully validated facts are written to the shared evidence cache, so
`final_submission.py` can consume them without re-spending free-tier
quota. Rejected and unresolved outcomes are NEVER cached, so a transient
failure always gets a fresh attempt next run.

This script never touches Stage 2/3 and never writes output.csv.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import load_dataset
from data.indexes import build_indexes

from evidence.ai_client import DEFAULT_MODEL_ID, GeminiAIClient
from evidence.cache import EvidenceCache
from evidence.deterministic import NEEDS_ESCALATION, run_prefilter
from evidence.resolver import resolve_image_with_ai


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "dataset"


def default_cache_path() -> Path:
    return Path(__file__).resolve().parent / "cache" / "evidence_cache.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of escalated images to process.")
    parser.add_argument("--all", action="store_true", help="Process every escalated image.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Disable the evidence cache for this run.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir()
    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_dataset(dataset_dir)
    indexes = build_indexes(dataset)

    prefilter = run_prefilter(dataset, indexes.events_by_id)
    images_by_id = {i.image_id: i for i in dataset.images}
    escalated = [d for d in prefilter.by_bucket(NEEDS_ESCALATION) if d.ref.image_id is not None]
    print(f"Escalated images needing vision: {len(escalated)}")

    selected = escalated if args.all else escalated[: max(0, args.limit)]
    print(f"Processing: {len(selected)}")
    if not selected:
        return 0

    cache = None if args.no_cache else EvidenceCache(default_cache_path())
    client = GeminiAIClient()
    print(f"Model: {client.model_id} (default {DEFAULT_MODEL_ID})")

    stats: Counter[str] = Counter()
    total_api_calls = 0

    for decision in selected:
        image = images_by_id[decision.ref.image_id]
        event = indexes.events_by_id.get(image.related_event_id) if image.related_event_id else None
        if event is None:
            stats["no_related_event"] += 1
            print(f"\n{image.image_id}: SKIPPED - no resolvable related event")
            continue

        outcome = resolve_image_with_ai(image, event, client, cache)
        total_api_calls += outcome.api_calls

        print(f"\n{image.image_id} (event {event.event_id}, currency {event.currency}):")
        if outcome.cache_hit:
            stats["cache_hit"] += 1
            print(f"  CACHE HIT -> {outcome.fact.resolved_amount} {outcome.fact.currency}")
        elif outcome.fact is not None:
            stats[f"resolved:{outcome.fact.fact_type}"] += 1
            raw = outcome.fact.raw_model_output or {}
            print(f"  RESOLVED  -> {outcome.fact.resolved_amount} {outcome.fact.currency}")
            print(f"  label     : {raw.get('amount_label')!r}")
            print(f"  as shown  : {raw.get('amount_text_as_shown')!r}")
            print(f"  doc type  : {raw.get('document_type')!r}")
            print(f"  confidence: {outcome.fact.confidence}")
        else:
            stats["unresolved"] += 1
            print(f"  UNRESOLVED-> {outcome.unresolved.reason}")
        if outcome.first_rejection_reason:
            print(f"  (first attempt rejected: {outcome.first_rejection_reason})")
        if outcome.second_rejection_reason:
            print(f"  (retry also rejected: {outcome.second_rejection_reason})")

    if cache is not None:
        cache.save()
        print(f"\nCache saved to {default_cache_path()} ({len(cache)} entries)")

    print("\n--- summary ---")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    print(f"  total API calls: {total_api_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
