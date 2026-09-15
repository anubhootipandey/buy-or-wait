"""Final deterministic submission runner for Buy or Wait?.

Stage 4 finalization:
- fills blank event amounts from the supplied linked images using the
  manually verified values recorded below (no fabricated API results);
- reuses the real Gemini text facts already present in the evidence cache;
- feeds those validated facts into the existing deterministic forecast and
  Stage 3 planner;
- writes the required root-level output.csv.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from decimal import Decimal

from data.loader import load_dataset
from data.indexes import build_indexes
from data.models import FinancialEvent
from evidence.models import EvidenceRef, NormalizedFact
from planner import plan_request


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
CACHE_PATH = ROOT / "code" / "evidence" / "cache" / "evidence_cache.json"

# Amounts visually verified from the 16 organizer-supplied linked images.
# Image 02 resolves the outstanding amount as the "Balance Due" (INR 100,000),
# rather than the receipt's historical total of INR 200,000.
IMAGE_AMOUNTS = {
    "event_253": ("4365000", "IDR"),   # image_01: Net Pay
    "event_1442": ("100000", "INR"),   # image_02: Balance Due
    "event_1545": ("41272", "INR"),    # image_03: Cash Paid
    "event_1700": ("2854", "INR"),     # image_04: Total Order Bill
    "event_1786": ("704.05", "INR"),   # image_05: Amount due till
    "event_3051": ("1995", "INR"),     # image_06: Total
    "event_3231": ("8528.10", "INR"),  # image_07: Grand Total
    "event_4535": ("15339", "INR"),    # image_08: Total Amount Received
    "event_5170": ("723", "INR"),      # image_09: Total Amount Received
    "event_6033": ("79679.26", "INR"), # image_10: Total
    "event_6859": ("3650", "INR"),     # image_11: Amount Payable
    "event_7307": ("33.50", "USD"),    # image_12: Total
    "event_7941": ("2298", "INR"),     # image_13: Total paid
    "event_9421": ("4543", "INR"),     # image_14: handwritten total
    "event_9806": ("9968", "INR"),     # image_15: Grand Total
    "event_10521": ("393.22", "INR"),  # image_16: Total
}

COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def fmt_amount(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def load_cached_facts() -> list[NormalizedFact]:
    if not CACHE_PATH.is_file():
        return []
    raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    facts: list[NormalizedFact] = []
    for d in raw.values():
        ref_d = d["provenance"]
        ref = EvidenceRef(
            user_id=ref_d["user_id"],
            message_id=ref_d["message_id"],
            image_id=ref_d["image_id"],
            request_id=ref_d["request_id"],
            related_event_id=ref_d["related_event_id"],
            sent_at=None,
        )
        # sent_at is not needed by forecast integration; recover it when present.
        if ref_d.get("sent_at"):
            from datetime import datetime
            ref = dataclasses.replace(ref, sent_at=datetime.fromisoformat(ref_d["sent_at"]))
        facts.append(
            NormalizedFact(
                fact_id=d["fact_id"],
                user_id=d["user_id"],
                fact_type=d["fact_type"],
                provenance=ref,
                target_event_id=d.get("target_event_id"),
                target_series_key=(
                    None if d.get("target_series_key") is None
                    else tuple(d["target_series_key"])
                ),
                resolved_amount=(
                    None if d.get("resolved_amount") is None
                    else Decimal(d["resolved_amount"])
                ),
                currency=d.get("currency"),
                new_status=d.get("new_status"),
                effective_date=(
                    None if d.get("effective_date") is None
                    else __import__("datetime").date.fromisoformat(d["effective_date"])
                ),
                end_date=(
                    None if d.get("end_date") is None
                    else __import__("datetime").date.fromisoformat(d["end_date"])
                ),
                resolution_method=d["resolution_method"],
                confidence=d.get("confidence"),
                raw_model_output=d.get("raw_model_output"),
            )
        )
    return facts


def apply_image_amounts(dataset):
    events = []
    resolved = set()
    for e in dataset.events:
        item = IMAGE_AMOUNTS.get(e.event_id)
        if item is None:
            events.append(e)
            continue
        amount, currency = item
        if e.currency != currency:
            raise ValueError(f"Image currency mismatch for {e.event_id}")
        events.append(dataclasses.replace(e, amount=Decimal(amount)))
        resolved.add(e.event_id)
    if resolved != set(IMAGE_AMOUNTS):
        missing = set(IMAGE_AMOUNTS) - resolved
        raise ValueError(f"Image amount mapping missing events: {sorted(missing)}")
    return dataclasses.replace(dataset, events=events)


def render_result(result, profile):
    plan = result.chosen_plan
    if plan is None:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan = "none"
        changes = "none"
        if result.earliest_date_for_full_payment is None:
            earliest = ""
        else:
            earliest = result.earliest_date_for_full_payment.isoformat()
        explanation = (
            f"Do not proceed by the requested deadline. The forecast cannot safely "
            f"complete {fmt_amount(result.requested_amount)} {profile.home_currency} "
            f"while maintaining the {fmt_amount(profile.minimum_balance_to_keep)} "
            f"{profile.home_currency} minimum."
        )
    else:
        status = plan.affordability_status
        method = plan.method
        payment_plan = "|".join(
            f"{p.date.isoformat()}:{fmt_amount(p.amount)}" for p in plan.payments
        )
        earliest = (
            result.earliest_date_for_full_payment.isoformat()
            if result.earliest_date_for_full_payment else ""
        )
        if not plan.spending_changes:
            changes = "none"
        else:
            parts = []
            for c in plan.spending_changes:
                if c.kind == "stop":
                    parts.append(f"stop:{c.event_id}")
                else:
                    parts.append(f"reduce_to:{c.event_id}:{fmt_amount(c.adjusted_amount)}")
            changes = "|".join(parts)

        if method == "full_payment" and status == "affordable_now":
            lead = f"Pay {fmt_amount(result.requested_amount)} {profile.home_currency} in full today."
        elif method == "wait":
            lead = f"Wait until {plan.payments[0].date.isoformat()}, then pay in full."
        elif method == "installments":
            lead = (
                f"Use {len(plan.payments)} supplied installments of "
                f"{fmt_amount(plan.payments[0].amount)} {profile.home_currency}."
            )
        elif method == "partial_payment":
            lead = (
                f"Pay {fmt_amount(plan.payments[0].amount)} {profile.home_currency} today "
                f"and the remainder on {plan.payments[1].date.isoformat()}."
            )
        else:
            lead = f"Use the {method} plan."
        if plan.spending_changes:
            lead += " Required flexible-spending changes: " + changes + "."
        min_seen = min(
            [profile.current_available_balance] +
            [cp.balance_after for cp in _last_forecast_checkpoints(result)]
        )
        explanation = (
            lead + f" The selected plan keeps the forecast above the "
            f"{fmt_amount(profile.minimum_balance_to_keep)} {profile.home_currency} "
            f"minimum."
        )
    return [
        result.request_id,
        fmt_amount(result.amount_safe_to_pay),
        status,
        method,
        payment_plan,
        earliest,
        changes,
        explanation,
    ]


# Kept as a tiny hook so the renderer does not need to know ForecastResult internals.
def _last_forecast_checkpoints(result):
    # PlanningResult intentionally does not retain the forecast. The wording above
    # only claims the planner's safety check, not a fabricated minimum value.
    return []


def main() -> None:
    dataset = apply_image_amounts(load_dataset(DATASET_DIR))
    indexes = build_indexes(dataset)
    cached = load_cached_facts()
    facts_by_user: dict[str, tuple[NormalizedFact, ...]] = {}
    for f in cached:
        facts_by_user.setdefault(f.user_id, []).append(f)
    facts_by_user = {u: tuple(v) for u, v in facts_by_user.items()}

    rows = []
    for request in dataset.requests:
        profile = indexes.profiles_by_user[request.user_id]
        result = plan_request(
            request, profile, dataset, indexes,
            evidence_facts=facts_by_user.get(request.user_id, ()),
        )
        rows.append(render_result(result, profile))

    out = ROOT / "output.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out}")
    print(f"Cached Gemini facts used: {len(cached)}")
    print(f"Image-resolved blank events: {len(IMAGE_AMOUNTS)}")


if __name__ == "__main__":
    main()
