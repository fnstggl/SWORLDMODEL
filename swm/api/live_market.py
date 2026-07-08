"""Live, KEYLESS market data — a real structured source for the markets domain.

The grounding router's structured tier needs real backends, not fixtures. Coinbase's public API is keyless and
returns both a spot price (state grounding) and daily candles (rate grounding), so it wires straight into a
`StructuredSource`. The measurement CI is the instrument's own 1-day volatility (from the recent candles), so a
calm asset grounds tight and a volatile one grounds wide — honest by construction. Every call is wrapped:
a network failure returns None (the router falls back to retrieval), never an exception.

Equities/indices/rates need a keyed vendor in practice (Yahoo/Stooq rate-limit hard); this covers crypto
live and leaves `products` injectable so a keyed feed slots in with the same interface.
"""
from __future__ import annotations

import json
import urllib.request

from swm.api.grounding_sources import StructuredSource

_SPOT = "https://api.coinbase.com/v2/prices/{p}/spot"
_CANDLES = "https://api.exchange.coinbase.com/products/{p}/candles?granularity=86400"
_PRODUCTS = {"btc_usd": "BTC-USD", "eth_usd": "ETH-USD", "sol_usd": "SOL-USD"}
_ALIASES = {"btc_usd": ["bitcoin price", "btc", "bitcoin"], "eth_usd": ["ethereum price", "eth", "ether"],
            "sol_usd": ["solana price", "sol"]}


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "swm-grounder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def coinbase_spot(product="BTC-USD"):
    try:
        return float(_get(_SPOT.format(p=product))["data"]["amount"])
    except Exception:
        return None


def coinbase_closes(product="BTC-USD", n=30):
    """Recent daily closes, oldest→newest (for rate grounding). None on any failure."""
    try:
        rows = sorted(_get(_CANDLES.format(p=product)), key=lambda r: r[0])   # [time,low,high,open,close,vol]
        return [float(r[4]) for r in rows][-n:]
    except Exception:
        return None


def _daily_vol_sd(closes, spot):
    """1-day price uncertainty = recent daily-return volatility × price (floored at 0.5%)."""
    if not closes or len(closes) < 5:
        return spot * 0.01
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1]]
    m = sum(rets) / len(rets)
    vol = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    return max(spot * 0.005, vol * spot)


def coinbase_source(products=None, aliases=None, **kw) -> StructuredSource:
    """A live StructuredSource over Coinbase. `fetch` returns (spot, 1-day-vol sd); `fetch_series` returns the
    recent daily closes for `TransitionOperator.ground_gain`."""
    products = products or _PRODUCTS
    aliases = aliases or {k: _ALIASES.get(k, [k]) for k in products}

    def fetch(key, as_of=None):
        p = products.get(key)
        if not p:
            return None
        spot = coinbase_spot(p)
        if spot is None:
            return None
        return (spot, _daily_vol_sd(coinbase_closes(p, 30) or [], spot))

    def fetch_series(key, as_of=None, window=6):
        p = products.get(key)
        return coinbase_closes(p, window) if p else None

    return StructuredSource("coinbase", "markets", aliases, fetch=fetch, fetch_series=fetch_series, **kw)
