"""
Stage 4b AI client.

`AIClient` is the narrow interface `resolver.py` depends on - anything
with a `.complete(system_instructions, user_prompt) -> str` method. Unit
tests use a fake implementation (see `code/tests/test_evidence_resolver.py`)
so they never touch the network or consume API quota.

`GeminiAIClient` is the real implementation, selected per DECISIONS.md /
`STAGE_4B.md`: Google Gemini, model `gemini-3.1-flash-lite` by default
(overridable via the `EVIDENCE_MODEL_ID` env var so a model swap never
requires a code change). It:

  * reads its API key ONLY from the `GEMINI_API_KEY` environment variable
    (never hardcoded, never logged, never written to any file here);
  * imports the `google-genai` SDK lazily, inside `__init__`, so nothing
    else in this package fails to import when the SDK isn't installed
    (e.g. when only running the mock-backed unit tests);
  * uses the provider's native structured-output mechanism
    (`response_json_schema`) so the model is constrained to the schema in
    `schema.py` rather than relying on prompt-only JSON discipline;
  * never retries internally - `resolver.py` owns the "retry exactly
    once" policy so it can make the retry materially more corrective
    (see `prompts.build_retry_prompt`), which a client-level retry could
    not do.
"""

from __future__ import annotations

import os
from typing import Protocol

from .schema import IMAGE_RESPONSE_JSON_SCHEMA, RESPONSE_JSON_SCHEMA

DEFAULT_MODEL_ID = "gemini-3.1-flash-lite"


class AIClient(Protocol):
    def complete(self, system_instructions: str, user_prompt: str) -> str:
        """Return the raw text of the model's structured-output response.
        Implementations should NOT catch/retry on transient errors -
        that's the caller's responsibility (see resolver.py), because
        only the caller knows how to build a materially better retry
        prompt."""
        ...

    def complete_with_image(
        self,
        system_instructions: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        """Return the raw text of the model's structured-output response
        for one IMAGE plus its text prompt (Stage 4d).

        Kept as a separate method rather than an optional argument on
        `complete` so that the text and image paths are constrained by
        their own distinct response schemas, and so an existing
        text-only client implementation stays valid. Callers must treat
        this method as optional (`hasattr`) and fail safe to unresolved
        when a client does not provide it - see `resolver.py`.
        """
        ...


class GeminiAIClient:
    """Real client backed by Google's Gemini API (`google-genai` SDK).

    Not imported/instantiated by the unit test suite - only by
    `run_text_experiment.py`, so `python3 -m unittest discover` never
    requires the SDK to be installed or a network connection to exist.
    """

    def __init__(self, model_id: str | None = None, api_key: str | None = None) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when SDK is missing
            raise RuntimeError(
                "The 'google-genai' package is required for GeminiAIClient. "
                "Install it with: pip install google-genai"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. Stage 4b never "
                "hardcodes API keys - export GEMINI_API_KEY before running the "
                "real-API experiment."
            )

        self._types = types
        self._client = genai.Client(api_key=key)
        self.model_id = model_id or os.environ.get("EVIDENCE_MODEL_ID", DEFAULT_MODEL_ID)

    def complete(self, system_instructions: str, user_prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_instructions,
                response_mime_type="application/json",
                response_json_schema=RESPONSE_JSON_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if text is None:
            raise RuntimeError(f"Gemini returned no text content (finish_reason may indicate why): {response!r}")
        return text

    def complete_with_image(
        self,
        system_instructions: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        """Stage 4d multimodal call.

        Uses the same model, the same temperature=0.0 determinism, and
        the same native structured-output mechanism as `complete` - only
        the payload (an inline image part alongside the text part) and
        the response schema differ. No extra provider, no OCR service,
        no additional dependency: `google-genai` is already required for
        the text path and is natively multimodal.

        The image is sent inline as bytes with its ACTUAL sniffed MIME
        type (see `image_input.load_image`); nothing is uploaded, stored,
        or retained anywhere by this client.
        """
        image_part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[image_part, user_prompt],
            config=self._types.GenerateContentConfig(
                system_instruction=system_instructions,
                response_mime_type="application/json",
                response_json_schema=IMAGE_RESPONSE_JSON_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if text is None:
            raise RuntimeError(
                f"Gemini returned no text content for image evidence "
                f"(finish_reason may indicate why): {response!r}"
            )
        return text


__all__ = ["AIClient", "GeminiAIClient", "DEFAULT_MODEL_ID"]
