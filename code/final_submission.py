#!/usr/bin/env python3
"""Final submission runner for Buy or Wait?.

Thin wrapper around the shared production pipeline in `pipeline.py` -
exactly the same code path as `python3 code/main.py run`, kept as a
separate entry point because the submission instructions reference it.

The pipeline it delegates to:

    dataset
      -> validated evidence facts (from the evidence cache; the text
         facts come from Stage 4b, the image facts from the Stage 4d
         vision resolver - no hardcoded amounts anywhere)
      -> resolved_amount facts applied to blank events
      -> reconciliation + 90-day forecast (Stage 4c series amendments)
      -> Stage 3 planner
      -> output.csv

Runs fully offline: it reads already-validated facts from the cache and
never constructs a model client. To refresh the facts themselves, run
`code/evidence/run_text_experiment.py` / `code/evidence/run_vision_experiment.py`,
or `python3 code/main.py run --resolve-evidence`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline


def main() -> None:
    prepared = pipeline.load_and_prepare()
    rows = pipeline.plan_all(prepared)
    out = pipeline.write_output_csv(rows)

    report = prepared.apply_report
    print(f"Wrote {len(rows)} predictions to {out}")
    print(f"Cached evidence facts used: {prepared.total_facts}")
    print(
        f"Blank event amounts filled from evidence: {len(report.applied)} "
        f"(of which image-derived: {prepared.image_derived_count})"
    )
    if report.skipped:
        print(f"Amount facts skipped by the applier: {len(report.skipped)}")
        for s in report.skipped:
            print(f"  - {s.fact_id}: {s.reason}")


if __name__ == "__main__":
    main()
