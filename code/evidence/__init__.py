"""
Stage 4 evidence-interpretation layer.

Stage 4a (the deterministic pre-filter, `deterministic.py`) classifies
every message/image into one of five buckets using nothing but
already-known structured data.

Stage 4b (`ai_client.py`, `prompts.py`, `schema.py`, `validation.py`,
`cache.py`, `resolver.py`) adds AI-backed interpretation for the
`needs_escalation` MESSAGES Stage 4a could not resolve itself, with
strict application-level validation, a single corrective retry, and a
cache that only ever stores successfully-validated facts. Images remain
unresolved pending Stage 4d (vision) - out of scope here.

No wiring into Stage 2/3 exists yet - that is Stage 4c, a separate,
not-yet-approved sub-stage.
"""

from __future__ import annotations

from .ai_client import AIClient, DEFAULT_MODEL_ID, GeminiAIClient
from .cache import EvidenceCache
from .deterministic import (
    BUCKETS,
    DETERMINISTIC_FACT,
    MISSING_EVIDENCE,
    NEEDS_ESCALATION,
    NO_FACT,
    UNRESOLVED_OTHER,
    PrefilterDecision,
    PrefilterReport,
    classify_image,
    classify_message,
    run_prefilter,
)
from .models import (
    EvidenceRef,
    EvidenceResolutionOutcome,
    NormalizedFact,
    UnresolvedEvidence,
)
from .prompts import EvidenceContext, build_prompt, build_retry_prompt
from .resolver import ResolutionOutcome, build_context, resolve_message_with_ai, run_stage4b
from .validation import ValidationResult, validate_response

__all__ = [
    "BUCKETS",
    "NO_FACT",
    "DETERMINISTIC_FACT",
    "NEEDS_ESCALATION",
    "MISSING_EVIDENCE",
    "UNRESOLVED_OTHER",
    "PrefilterDecision",
    "PrefilterReport",
    "classify_message",
    "classify_image",
    "run_prefilter",
    "EvidenceRef",
    "NormalizedFact",
    "UnresolvedEvidence",
    "EvidenceResolutionOutcome",
    "AIClient",
    "GeminiAIClient",
    "DEFAULT_MODEL_ID",
    "EvidenceCache",
    "EvidenceContext",
    "build_prompt",
    "build_retry_prompt",
    "ResolutionOutcome",
    "build_context",
    "resolve_message_with_ai",
    "run_stage4b",
    "ValidationResult",
    "validate_response",
]