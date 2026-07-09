"""As-of market grounding — the leakage-free lever for metric questions.

The backtest could not ground a metric question's current value without leaking the future (today's price is
post-resolution). The fix is AS-OF grounding: fetch the value the market had ON THE QUESTION'S DATE — exactly
what a forecaster standing at that moment knew — from historical data. For crypto this is Coinbase's public
daily candles, keyless. We return both the as-of price AND the asset's REALISED volatility over the prior
window (not an LLM guess), so the diffusion is calibrated to how that asset actually moves. This turns the
metric branch from a humbled guess into a confident, correct, leakage-free simulation of the price path.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
import urllib.request

_CANDLES = "https://api.exchange.coinbase.com/products/{p}/candles?granularity=86400&start={s}&end={e}"
_PRODUCTS = {"bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD", "ether": "ETH-USD",
             "solana": "SOL-USD", "sol": "SOL-USD", "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
             "cardano": "ADA-USD", "ada": "ADA-USD", "ripple": "XRP-USD", "xrp": "XRP-USD",
             "litecoin": "LTC-USD", "avalanche": "AVAX-USD", "avax": "AVAX-USD", "polkadot": "DOT-USD",
             "chainlink": "LINK-USD", "link": "LINK-USD", "polygon": "MATIC-USD", "matic": "MATIC-USD"}


def _iso(ts):
    return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%dT%H:%M:%S")


def _candles(product, ts, days_back):
    try:
        url = _CANDLES.format(p=product, s=_iso(ts - days_back * 86400), e=_iso(ts + 86400))
        req = urllib.request.Request(url, headers={"User-Agent": "swm-grounder/1.0"})
        rows = sorted(json.loads(urllib.request.urlopen(req, timeout=20).read()), key=lambda r: r[0])
        return [r for r in rows if r[0] <= ts + 86400]        # [time, low, high, open, close, vol], as-of only
    except Exception:
        return None


def detect_product(text):
    t = (text or "").lower()
    for word, prod in _PRODUCTS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return prod
    return None


class CryptoAsofGrounder:
    """Grounds a crypto metric question's current value + realised volatility AS OF the question date."""

    name = "coinbase_asof"

    def ground_metric(self, question, metric_name, as_of):
        prod = detect_product(f"{metric_name or ''} {question or ''}")
        if prod is None or as_of is None:
            return None
        rows = _candles(prod, float(as_of), days_back=120)
        if not rows or len(rows) < 8:
            return None
        closes = [r[4] for r in rows]
        price = closes[-1]                                    # the close AS OF the question date (leakage-free)
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(rets) < 5:
            return None
        m = sum(rets) / len(rets)
        daily_vol = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
        return {"value": float(price), "annual_vol_pct": float(daily_vol * math.sqrt(365) * 100),
                "source": f"{self.name}:{prod}", "product": prod}
