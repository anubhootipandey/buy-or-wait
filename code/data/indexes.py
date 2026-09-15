"""
Lookup indexes over a loaded `Dataset`.

Nothing here filters, judges, or interprets data - it only groups records
that later stages (financial engine, planner) will need to look up
repeatedly, so they aren't re-scanning 25k events per request.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .models import (
    Dataset,
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageRecord,
    Message,
    PaymentOption,
    Request,
)


@dataclass
class Indexes:
    profiles_by_user: dict[str, FinancialProfile]

    events_by_id: dict[str, FinancialEvent]
    events_by_user: dict[str, list[FinancialEvent]]

    payment_options_by_request: dict[str, list[PaymentOption]]

    # Per the problem statement, messages must be retrieved using BOTH
    # request_id and user_id together, and separately some messages carry
    # no request_id at all (e.g. a standalone payroll notice) but are still
    # relevant to that user's events. Both access paths are indexed.
    messages_by_request_and_user: dict[tuple[str, str], list[Message]]
    messages_by_user: dict[str, list[Message]]
    messages_by_related_event: dict[str, list[Message]]

    images_by_related_event: dict[str, ImageRecord]
    images_by_request: dict[str, list[ImageRecord]]

    # (rate_date, from_currency, to_currency) -> ExchangeRate
    exchange_rate_lookup: dict[tuple[date, str, str], ExchangeRate]

    requests_by_id: dict[str, Request]


def build_indexes(dataset: Dataset) -> Indexes:
    profiles_by_user = {p.user_id: p for p in dataset.profiles}

    events_by_id = {e.event_id: e for e in dataset.events}
    events_by_user: dict[str, list[FinancialEvent]] = defaultdict(list)
    for e in dataset.events:
        events_by_user[e.user_id].append(e)

    payment_options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
    for po in dataset.payment_options:
        payment_options_by_request[po.request_id].append(po)

    messages_by_request_and_user: dict[tuple[str, str], list[Message]] = defaultdict(list)
    messages_by_user: dict[str, list[Message]] = defaultdict(list)
    messages_by_related_event: dict[str, list[Message]] = defaultdict(list)
    for m in dataset.messages:
        messages_by_user[m.user_id].append(m)
        if m.request_id is not None:
            messages_by_request_and_user[(m.request_id, m.user_id)].append(m)
        if m.related_event_id is not None:
            messages_by_related_event[m.related_event_id].append(m)

    images_by_related_event: dict[str, ImageRecord] = {}
    images_by_request: dict[str, list[ImageRecord]] = defaultdict(list)
    for img in dataset.images:
        if img.related_event_id is not None:
            images_by_related_event[img.related_event_id] = img
        if img.request_id is not None:
            images_by_request[img.request_id].append(img)

    exchange_rate_lookup: dict[tuple[date, str, str], ExchangeRate] = {}
    for r in dataset.exchange_rates:
        exchange_rate_lookup[(r.rate_date, r.from_currency, r.to_currency)] = r

    requests_by_id = {r.request_id: r for r in dataset.requests}
    # sample_requests share the Request shape/id-space but are kept separate
    # per FACTS.md - they are not merged into requests_by_id so nothing can
    # accidentally treat a worked example as a graded request.

    return Indexes(
        profiles_by_user=profiles_by_user,
        events_by_id=events_by_id,
        events_by_user=dict(events_by_user),
        payment_options_by_request=dict(payment_options_by_request),
        messages_by_request_and_user=dict(messages_by_request_and_user),
        messages_by_user=dict(messages_by_user),
        messages_by_related_event=dict(messages_by_related_event),
        images_by_related_event=images_by_related_event,
        images_by_request=dict(images_by_request),
        exchange_rate_lookup=exchange_rate_lookup,
        requests_by_id=requests_by_id,
    )
