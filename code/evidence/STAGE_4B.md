# Stage 4b - AI evidence-interpretation layer

Scope: AI-backed interpretation of the message-level evidence Stage 4a's
deterministic prefilter (`deterministic.py`) could not resolve itself
(`needs_escalation` bucket, messages only - images stay unresolved,
deferred to Stage 4d/vision, out of scope here). Does not touch Stage 2,
Stage 3, or the financial engine. Does not wire facts back into Stage 2/3
(that's Stage 4c).

## Model/provider selection

**Selected: Google Gemini, model ID `gemini-3.1-flash-lite`** (Google's
current GA "workhorse" cost-efficiency model, released March 2026),
via the official `google-genai` Python SDK. Overridable without a code
change via the `EVIDENCE_MODEL_ID` environment variable.

Verified against official documentation on 2026-09-13 (`ai.google.dev`):

| Criterion | Gemini `gemini-3.1-flash-lite` | Groq (`openai/gpt-oss-120b`, free tier) | Mistral (La Plateforme free tier) |
|---|---|---|---|
| Exact model ID confirmed | Yes - `gemini-3.1-flash-lite`, stable/GA | Yes, but see limitations | Yes (e.g. `mistral-small-latest`), but see limitations |
| Currently available | Yes. (NB: `gemini-2.5-flash-lite` and all `2.0` models are **already deprecated/shutting down in 2026** - 3.1 was chosen specifically to avoid building on a model already scheduled for shutdown.) | Yes | Yes |
| Free tier | Yes, Flash/Flash-Lite family free within rate limits (Pro models moved to paid-only April 2026) | Yes, ~1,000 req/day, 30 RPM per model, no card | Yes ("Experiment" tier), but region-gated + identity verification required, and Mistral no longer publishes exact free-tier numbers |
| Rate/token limits | Not published as a static table anymore (moved to a per-account dashboard, `aistudio.google.com/rate-limit`) - historically Flash-Lite tier free limits are in the 15-30 RPM / ~1,000-1,500 RPD range | ~1,000 RPD / 30 RPM per model (published) | Not published; "conservative" per official docs |
| Structured JSON/schema support | Yes - native `response_json_schema` (JSON Schema) or Pydantic, enforced output | Yes - `json_schema` strict mode on select models | Docs explicitly warn: **"JSON mode does not guarantee adherence to a specific schema. Use function calling for structured outputs"** - weaker guarantee |
| Text support | Yes | Yes | Yes |
| Vision/image support (future Stage 4d) | Yes, natively, same model family/API (text, image, video, audio, PDF input) | No native vision on the free chat model list (GPT-OSS/Qwen/Compound) | Image input exists on some Mistral models but is a separate track from the free "Experiment" text tier |
| Reliability under time constraint | Official SDK, one dependency, one env var (`GEMINI_API_KEY`) | Also simple (OpenAI-compatible), but weaker future-vision story | Extra friction: region gating + identity verification |

**Why Gemini 3.1 Flash-Lite specifically, not 2.5 Flash-Lite or a Pro
model:** `gemini-2.5-flash-lite` is scheduled for shutdown on
2026-10-16/20 per Firebase's official model docs - not worth building a
fresh integration against a model already mid-deprecation. Pro models
have had free-tier access removed entirely since April 2026. 3.1
Flash-Lite is GA, stable, explicitly positioned by Google as the
efficiency/high-volume workhorse, and is multimodal from day one, which
directly serves the "future vision compatibility" requirement without a
provider change later.

**Important limitation to flag:** as of September 2026 Google no longer
publishes a static free-tier RPM/TPM/RPD table on the rate-limits doc
page - it directs developers to a live per-account dashboard instead.
`run_text_experiment.py` is therefore deliberately conservative (one
evidence item at a time, `--limit`-bounded, clear 429/error surfacing) so
whatever the account's actual current limit is, the script degrades to a
visible error rather than a silent partial run.

## Architecture

```
Stage 4a prefilter (unchanged)
  -> no_fact / deterministic_fact          -> pass through unchanged
  -> missing_evidence / unresolved_other   -> pass through as UnresolvedEvidence
  -> needs_escalation, image               -> UnresolvedEvidence("...Stage 4d...")
  -> needs_escalation, message             -> Stage 4b AI resolution:
       build_context()               ground the message against ONLY that
                                      user's own known event/series data
       build_prompt()                explicit schema + rules (schema.py)
       AIClient.complete()           one call to the model
       validate_response()           strict application-level validation
       (invalid?) -> build_retry_prompt() -> ONE retry -> validate again
       (still invalid?) -> UnresolvedEvidence, never cached
       (valid?)    -> NormalizedFact (resolution_method="gemini"), cached
```

Modules: `schema.py` (response contract + enums), `prompts.py` (prompt
text), `ai_client.py` (`AIClient` protocol + `GeminiAIClient`),
`validation.py` (parsing + strict checks), `cache.py` (JSON-file cache,
successful facts only), `resolver.py` (orchestration + retry policy),
`run_text_experiment.py` (small real-API CLI experiment).

## Validation / retry strategy

- The model must return every field in the schema, always (nullable, but
  never omitted) - "missing field" and "unexpected field" are both hard
  rejections, never silently patched.
- `supporting_quote` must be an exact (whitespace-tolerant) substring of
  the source message text. `amount`, when present, must also literally
  appear as a number in the source text. Both are checked in Python, not
  trusted from the model.
- `currency`, when the message is linked to a known event, must equal
  that event's own currency - a model-asserted currency can never
  override the dataset's.
- A standalone message's series-level fact (`future_amount_change` /
  `series_terminated`) must name a `(target_category, target_event_type,
  target_direction)` that already exists among *that user's own* events.
  The model can never invent a spending/income category from nothing.
- `resolved_amount` is only accepted for an event whose `amount` is
  genuinely blank; `status_correction` is only accepted if `new_status`
  actually differs from the event's current status.
- Exactly one retry is allowed, and it is materially different from the
  first attempt: the retry prompt includes the model's own previous
  (rejected) output and the exact rejection reason, then asks it to fix
  that specific problem or explicitly return `"unresolved"`.
- A rejected/invalid response and an "unresolved" outcome are NEVER
  cached - only a fully-validated fact is, keyed to the exact message
  text + related event snapshot the model saw (so a later dataset change
  can never serve a stale cached answer).

## Design decisions

- Only `needs_escalation` **messages** go to the AI; `needs_escalation`
  **images** are left unresolved (Stage 4d territory) - Stage 4b never
  attempts OCR/vision, per the task's explicit "do not implement 4c/4d".
- The response schema adds a few fields beyond the task's stated minimum
  (`new_status`, `target_category`/`target_event_type`/`target_direction`)
  so a `status_correction` or a standalone series-level fact can be
  produced without ever letting the model supply a raw `target_event_id`
  or series key itself - Python always derives/grounds those from data it
  already trusts.
- `classification` intentionally includes `"unresolved"` as distinct from
  `evidence.models.FACT_TYPES`' `"no_fact"`: `"no_fact"` means "nothing to
  act on" (itself a valid, cacheable fact, matching Stage 4a's own
  convention); `"unresolved"` means "I can't safely decide" and is never
  cached, so it gets a fresh attempt on the next run.
- `resolution_method="gemini"` is stored on every AI-produced fact
  (already a valid value in Stage 4a's `RESOLUTION_METHODS`), so a fact
  from Stage 4b is always distinguishable from a Stage 4a deterministic
  one in `raw_model_output`/provenance for later auditing.

## Real API experiment - status

**Not executed by the coding agent in this session.** The execution
sandbox this session runs in has outbound network access disabled
(`bash_tool` network is off) and does not have the `google-genai`
package installed, so `python3 code/evidence/run_text_experiment.py
--limit 1` fails immediately and honestly with:

```
ERROR: could not construct the Gemini client: The 'google-genai' package
is required for GeminiAIClient. Install it with: pip install google-genai
```

No experiment result is reported here because none was actually run -
inventing one would violate this project's own "do not claim success
merely because HTTP status is 200" / "do not fabricate experiment
results" rules. To actually run it:

```bash
pip install google-genai
export GEMINI_API_KEY=<your key from https://aistudio.google.com/apikey>
python3 code/evidence/run_text_experiment.py --limit 1
# inspect the single result's classification/amount/quote for semantic
# correctness, then:
python3 code/evidence/run_text_experiment.py --limit 5
```

The script prints, per item, the prefilter reason, the message text, the
resolved classification/fields (or rejection/retry/unresolved detail),
and a final summary of API calls, cache hits, retries, rejections,
unresolved counts, errors, and a classification breakdown - everything
needed to eyeball semantic correctness and confirm real (not just
HTTP-200) success before scaling up.
