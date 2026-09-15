# AI usage report

## Gemini text experiment

Model: `gemini-3.1-flash-lite`.

A real local experiment was run against the challenge messages:
- 20 messages processed
- 15 Gemini API calls
- 5 cache hits
- 19 successfully validated facts
- 1 unresolved response
- 0 retries
- 0 rejected first attempts
- 0 API errors

The 19 successful results are preserved in
`code/evidence/cache/evidence_cache.json` and are reused by the final runner.

Exact token counts were not emitted by the experiment runner/SDK output, so no
token numbers are fabricated.

## Image evidence

The 16 organizer-provided linked images were inspected to resolve the blank
financial-event amounts. These are applied as dataset-evidence resolutions,
not represented as fabricated Gemini API calls.

## Cost

No paid API tier or billing setup is used by the final runner. The final
generation itself uses the cached Gemini results and deterministic Python.
