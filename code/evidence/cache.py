"""
Stage 4b evidence cache.

A small JSON-file-backed cache so re-running the experiment (e.g.
`--limit 1` followed by `--limit 5`) never re-spends free-tier API quota
on evidence already successfully resolved. Deliberately narrow:

  * ONLY a successfully-validated `NormalizedFact` is ever cached. An
    invalid (rejected) response and an "unresolved" outcome are NEVER
    cached, so a transient model mistake or a temporary rate-limit
    situation always gets a fresh attempt on the next run.
  * The cache key includes a hash of the exact inputs the model saw
    (message text + the related event's own amount/status/currency, when
    there is one), so if the underlying dataset ever changes, stale
    entries are silently bypassed rather than served incorrectly - this
    is a correctness property, not just an optimization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from .models import EvidenceRef, NormalizedFact
from .prompts import EvidenceContext


def _cache_key(ctx: EvidenceContext) -> str:
    parts = [ctx.message.message_id, ctx.message.message_text]
    if ctx.event is not None:
        e = ctx.event
        parts += [e.event_id, str(e.amount), e.currency, e.status]
    else:
        parts += ["standalone", ",".join("/".join(k) for k in ctx.known_series_categories)]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{ctx.message.message_id}:{digest}"


def _fact_to_json(fact: NormalizedFact) -> dict:
    d = asdict(fact)
    d["resolved_amount"] = None if fact.resolved_amount is None else str(fact.resolved_amount)
    d["effective_date"] = None if fact.effective_date is None else fact.effective_date.isoformat()
    d["end_date"] = None if fact.end_date is None else fact.end_date.isoformat()
    d["target_series_key"] = None if fact.target_series_key is None else list(fact.target_series_key)
    ref = fact.provenance
    d["provenance"] = {
        "user_id": ref.user_id,
        "message_id": ref.message_id,
        "image_id": ref.image_id,
        "request_id": ref.request_id,
        "related_event_id": ref.related_event_id,
        "sent_at": None if ref.sent_at is None else ref.sent_at.isoformat(),
    }
    return d


def _fact_from_json(d: dict) -> NormalizedFact:
    ref_d = d["provenance"]
    ref = EvidenceRef(
        user_id=ref_d["user_id"],
        message_id=ref_d["message_id"],
        image_id=ref_d["image_id"],
        request_id=ref_d["request_id"],
        related_event_id=ref_d["related_event_id"],
        sent_at=None if ref_d["sent_at"] is None else datetime.fromisoformat(ref_d["sent_at"]),
    )
    return NormalizedFact(
        fact_id=d["fact_id"],
        user_id=d["user_id"],
        fact_type=d["fact_type"],
        provenance=ref,
        target_event_id=d["target_event_id"],
        target_series_key=None if d["target_series_key"] is None else tuple(d["target_series_key"]),
        resolved_amount=None if d["resolved_amount"] is None else Decimal(d["resolved_amount"]),
        currency=d["currency"],
        new_status=d["new_status"],
        effective_date=None if d["effective_date"] is None else date.fromisoformat(d["effective_date"]),
        end_date=None if d["end_date"] is None else date.fromisoformat(d["end_date"]),
        resolution_method=d["resolution_method"],
        confidence=d["confidence"],
        raw_model_output=d["raw_model_output"],
    )


class EvidenceCache:
    """In-memory cache, optionally backed by a JSON file. `path=None`
    gives a pure in-memory cache (used by tests - no disk I/O)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self._store: dict[str, dict] = {}
        if path is not None and path.is_file():
            try:
                self._store = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def get(self, ctx: EvidenceContext) -> Optional[NormalizedFact]:
        entry = self._store.get(_cache_key(ctx))
        if entry is None:
            return None
        return _fact_from_json(entry)

    def set(self, ctx: EvidenceContext, fact: NormalizedFact) -> None:
        self._store[_cache_key(ctx)] = _fact_to_json(fact)

    def __len__(self) -> int:
        return len(self._store)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["EvidenceCache"]
