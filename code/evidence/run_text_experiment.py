#!/usr/bin/env python3
"""
Stage 4b real-API experiment runner.

Runs the AI evidence resolver against a SMALL, controlled slice of the
real dataset's escalated messages, using the real Gemini API (never a
fake/mock client - that's what the unit tests are for). Intended usage,
per the project's own instructions:

    python3 code/evidence/run_text_experiment.py --limit 1
    # inspect the result for semantic correctness, THEN:
    python3 code/evidence/run_text_experiment.py --limit 5

Requires the GEMINI_API_KEY environment variable to be set. Never reads
a key from a file or a hardcoded value.

This script NEVER touches Stage 2/3, never writes output.csv, and never
processes the full dataset - it exists purely to validate Stage 4b's
real-world behavior on a bounded number of real API calls.
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
from evidence.resolver import resolve_message_with_ai


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "dataset"


def default_cache_path() -> Path:
    return Path(__file__).resolve().parent / "cache" / "evidence_cache.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of escalated messages to process.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Disable the evidence cache for this run.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir()
    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_dataset(dataset_dir)
    indexes = build_indexes(dataset)

    prefilter = run_prefilter(dataset, indexes.events_by_id)
    messages_by_id = {m.message_id: m for m in dataset.messages}
    escalated_message_decisions = [
        d for d in prefilter.by_bucket(NEEDS_ESCALATION) if d.ref.message_id is not None
    ]
    print(f"Total messages needing escalation (dataset-wide): {len(escalated_message_decisions)}")

    slice_ = escalated_message_decisions[: args.limit]
    print(f"Processing {len(slice_)} of them this run (--limit {args.limit}).")

    try:
        client = GeminiAIClient(model_id=DEFAULT_MODEL_ID)
    except RuntimeError as exc:
        print(f"\nERROR: could not construct the Gemini client: {exc}", file=sys.stderr)
        return 2
    print(f"Using model: {client.model_id}")

    cache = None if args.no_cache else EvidenceCache(default_cache_path())

    classification_counts: Counter[str] = Counter()
    api_calls_total = 0
    retries = 0
    rejected_first_attempts = 0
    unresolved_count = 0
    successful_facts = 0
    cache_hits = 0
    errors: list[tuple[str, str]] = []

    for i, decision in enumerate(slice_, start=1):
        message = messages_by_id[decision.ref.message_id]
        print(f"\n--- [{i}/{len(slice_)}] {message.message_id} (user={message.user_id}) ---")
        print(f"  prefilter reason: {decision.reason}")
        print(f"  text: {message.message_text[:200]!r}")
        try:
            outcome = resolve_message_with_ai(message, indexes, client, cache)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole run
            print(f"  API/RUNTIME ERROR: {exc!r}")
            errors.append((message.message_id, repr(exc)))
            continue

        api_calls_total += outcome.api_calls
        if outcome.cache_hit:
            cache_hits += 1
            print("  CACHE HIT - no API call made.")
        if outcome.api_calls == 2:
            retries += 1
        if outcome.first_rejection_reason:
            rejected_first_attempts += 1
            print(f"  first attempt rejected: {outcome.first_rejection_reason}")
        if outcome.fact is not None:
            successful_facts += 1
            classification_counts[outcome.fact.fact_type] += 1
            print(f"  RESULT: fact_type={outcome.fact.fact_type!r} confidence={outcome.fact.confidence!r}")
            print(f"    amount={outcome.fact.resolved_amount} currency={outcome.fact.currency}")
            print(f"    effective_date={outcome.fact.effective_date} end_date={outcome.fact.end_date}")
            print(f"    new_status={outcome.fact.new_status} target_series_key={outcome.fact.target_series_key}")
        elif outcome.unresolved is not None:
            unresolved_count += 1
            print(f"  RESULT: unresolved - {outcome.unresolved.reason}")

    if cache is not None:
        cache.save()

    print("\n=== Summary ===")
    print(f"Messages processed:        {len(slice_)}")
    print(f"API calls made:            {api_calls_total}")
    print(f"Cache hits (no API call):  {cache_hits}")
    print(f"Retries (2nd attempt):     {retries}")
    print(f"Rejected first attempts:   {rejected_first_attempts}")
    print(f"Successful facts:          {successful_facts}")
    print(f"Unresolved (final):        {unresolved_count}")
    print(f"Errors (API/runtime):      {len(errors)}")
    if errors:
        for mid, err in errors:
            print(f"  {mid}: {err}")
    print(f"Classification breakdown:  {dict(classification_counts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
