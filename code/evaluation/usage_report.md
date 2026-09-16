# AI usage report

## Gemini text experiment

Model: gemini-3.1-flash-lite.

A real local experiment was run against the challenge messages:
- 20 messages processed
- 15 Gemini API calls
- 5 cache hits
- 19 successfully validated facts
- 1 unresolved response
- 0 retries
- 0 rejected first attempts
- 0 API errors

Exact token counts were not emitted by the experiment runner/SDK output, so no token numbers are fabricated.

## Gemini image evidence

A real Gemini vision experiment was run against all 16 organizer-provided linked images:
- 16 images processed
- 16 Gemini API calls
- 16 successfully validated esolved_amount facts
- 0 unresolved images
- 0 API errors
- 0 retries

The validated image facts are cached and applied only to their corresponding blank financial events. Dataset-provided event currency remains authoritative.

## Cost

No paid API tier or billing setup is used by the final runner. The final generation itself uses cached Gemini results and deterministic Python.


