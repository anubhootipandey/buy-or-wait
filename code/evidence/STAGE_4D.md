# Stage 4D / Finalization

The final runner uses the existing validated evidence architecture without
inventing unavailable API results.

- 16 blank-amount events were linked to the organizer-provided images and
  their visible amounts were resolved before reconciliation.
- The 19 successfully validated Gemini text facts already stored in
  `evidence/cache/evidence_cache.json` are reused.
- The uncached Gemini messages are not fabricated.
- `code/final_submission.py` applies the resolved event amounts, passes cached
  evidence facts into the deterministic forecast/planner, and writes the
  root-level `output.csv`.
- Ambiguous evidence is not converted into an invented model result.

The final output contains one prediction for each of the 250 graded requests
and the exact required output columns.
