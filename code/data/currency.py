"""
Currency conversion for the "Buy or Wait?" dataset.

DECISION (see DECISIONS.md): use an exact `(date, from_currency,
to_currency)` lookup only - no multi-hop chaining through an intermediate
currency, and no nearest-date fallback. This was verified against the full
dataset (FACTS.md): every one of the 140 foreign-currency events has an
exact-date rate row for its own settlement date and exact currency pair.
Chaining/fallback logic would be complexity with zero payoff on this data,
and could silently paper over a genuinely missing rate in a way that's
worse than failing loudly.

If a later stage ever hits a `MissingExchangeRateError`, that is a signal
the dataset changed in a way Stage 1's assumption doesn't cover, and the
lookup logic should be revisited then - not preemptively generalized now.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from .indexes import Indexes


class MissingExchangeRateError(LookupError):
    def __init__(self, on_date: date, from_currency: str, to_currency: str) -> None:
        self.on_date = on_date
        self.from_currency = from_currency
        self.to_currency = to_currency
        super().__init__(
            f"No exchange rate for {from_currency}->{to_currency} on {on_date.isoformat()}"
        )


def convert(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    on_date: date,
    indexes: Indexes,
) -> Decimal:
    """Convert `amount` from `from_currency` to `to_currency` on `on_date`.

    Same-currency conversion is a no-op (returned as-is, no rate lookup).
    Otherwise requires an exact-date, exact-pair row in exchange_rates.csv;
    raises MissingExchangeRateError if none exists.
    """
    if from_currency == to_currency:
        return amount

    rate_row = indexes.exchange_rate_lookup.get((on_date, from_currency, to_currency))
    if rate_row is not None:
        return amount * rate_row.rate

    raise MissingExchangeRateError(on_date, from_currency, to_currency)
