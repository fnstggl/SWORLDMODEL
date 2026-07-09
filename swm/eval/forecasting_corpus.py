"""Leakage-proof backtest corpus from public forecasting platforms — history as the test set.

Pulls RESOLVED binary questions from Manifold (clean open API; tens of thousands of markets) and Polymarket
(real-money), normalized to one schema with the CROWD probability reconstructed at a fair as-of lead (the
baseline the world-model must beat). The crowd forecast at as-of and the model forecast at as-of use the SAME
information time, so the comparison is fair. Each item is tagged `cutoff_clean` (resolved after the LLM's
training cutoff ⇒ the model cannot have memorized the outcome) and a coarse `category`, so Stage-1 can report
skill on the leakage-proof slice and by domain.

  BacktestItem: qid, platform, question, outcome (0/1), as_of, resolve_ts, crowd_prob (at as_of), n_crowd,
                category, cutoff_clean.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

CUTOFF_TS = 1719792000          # 2024-07-01 UTC — resolved after this ⇒ post-training-cutoff (clean)
AS_OF_FRAC = 0.4                # forecast at 40% of the market's life (real crowd info, still unresolved)

_CATS = [("election", "election|senate|president|primary|governor|parliament|vote|ballot|referendum"),
         ("geopolitics", "war|invade|ceasefire|treaty|sanction|nuclear|coup|border|hostage"),
         ("economy", "gdp|inflation|recession|unemployment|rate hike|fed |interest rate|stock|s&p|nasdaq"),
         ("crypto", "bitcoin|btc|ethereum|eth |crypto|token|coin|solana"),
         ("tech", "ai |gpt|openai|model|launch|release|chip|nvidia|tesla|spacex|apple"),
         ("sports", "nba|nfl|win the|championship|super bowl|world cup|match|game|playoff|cup final"),
         ("science", "covid|vaccine|climate|temperature|nasa|fusion|study|approve"),
         ("culture", "movie|oscar|award|album|box office|celebrity|twitter|elon")]


def _cat(q):
    ql = q.lower()
    for name, pat in _CATS:
        if re.search(pat, ql):
            return name
    return "other"


@dataclass
class BacktestItem:
    qid: str
    platform: str
    question: str
    outcome: int
    as_of: int
    resolve_ts: int
    crowd_prob: float
    n_crowd: int
    category: str
    cutoff_clean: bool


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "swm-backtest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ----------------------------------------------------------------------------- Manifold
def _manifold_crowd_prob(cid, as_of_ms):
    """The crowd probability at as_of, from bet history (last probAfter at or before as_of)."""
    try:
        bets = _get(f"https://api.manifold.markets/v0/bets?contractId={cid}&limit=2000")
    except Exception:
        return None
    prior = [b for b in bets if b.get("createdTime", 0) <= as_of_ms and b.get("probAfter") is not None]
    if not prior:
        return 0.5
    return float(max(prior, key=lambda b: b["createdTime"])["probAfter"])


def pull_manifold(target=3000, min_bettors=6, page=1000):
    """Resolved YES/NO binary markets with a real crowd, newest-first, paginated."""
    items, before = [], None
    seen = 0
    while len(items) < target and seen < target * 30:
        url = f"https://api.manifold.markets/v0/markets?limit={page}" + (f"&before={before}" if before else "")
        batch = _get(url)
        if not batch:
            break
        before = batch[-1]["id"]
        seen += len(batch)
        for m in batch:
            if (m.get("outcomeType") == "BINARY" and m.get("isResolved")
                    and m.get("resolution") in ("YES", "NO")
                    and m.get("uniqueBettorCount", 0) >= min_bettors
                    and m.get("resolutionTime") and m.get("createdTime")):
                created, resolved = m["createdTime"], m["resolutionTime"]
                if resolved <= created:
                    continue
                as_of = int(created + AS_OF_FRAC * (resolved - created))
                cp = _manifold_crowd_prob(m["id"], as_of)
                if cp is None:
                    continue
                items.append(BacktestItem(
                    qid=f"mf_{m['id']}", platform="manifold", question=m["question"].strip(),
                    outcome=1 if m["resolution"] == "YES" else 0, as_of=as_of // 1000,
                    resolve_ts=resolved // 1000, crowd_prob=round(cp, 4),
                    n_crowd=int(m["uniqueBettorCount"]), category=_cat(m["question"]),
                    cutoff_clean=(resolved // 1000) > CUTOFF_TS))
                if len(items) >= target:
                    break
    return items


# ----------------------------------------------------------------------------- Polymarket
def _poly_crowd_prob(clob_ids, as_of_ts):
    try:
        tid = json.loads(clob_ids)[0]
        hist = _get(f"https://clob.polymarket.com/prices-history?market={tid}&fidelity=180"
                    f"&startTs={as_of_ts - 86400 * 30}&endTs={as_of_ts}")
        pts = hist.get("history", [])
        prior = [p for p in pts if p.get("t", 0) <= as_of_ts]
        return float(prior[-1]["p"]) if prior else None
    except Exception:
        return None


def pull_polymarket(target=1500, min_volume=2000, page=500):
    items, offset = [], 0
    while len(items) < target and offset < target * 20:
        batch = _get(f"https://gamma-api.polymarket.com/markets?limit={page}&offset={offset}"
                     f"&closed=true&order=volume&ascending=false")
        if not batch:
            break
        offset += len(batch)
        for m in batch:
            try:
                outs, prices = json.loads(m.get("outcomes", "[]")), json.loads(m.get("outcomePrices", "[]"))
            except Exception:
                continue
            if (len(outs) != 2 or len(prices) != 2 or float(m.get("volume", 0) or 0) < min_volume
                    or not m.get("endDate") or not m.get("clobTokenIds")):
                continue
            yes_i = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "over")), 0)
            fp = float(prices[yes_i])
            if fp not in (0.0, 1.0):                          # keep only cleanly-resolved (0/1) markets
                continue
            import calendar
            import datetime as _dt
            try:
                end = calendar.timegm(_dt.datetime.strptime(m["endDate"][:19], "%Y-%m-%dT%H:%M:%S").timetuple())
                start = calendar.timegm(_dt.datetime.strptime((m.get("startDate") or m["endDate"])[:19],
                                                              "%Y-%m-%dT%H:%M:%S").timetuple())
            except Exception:
                continue
            as_of = int(start + AS_OF_FRAC * max(1, end - start))
            cp = _poly_crowd_prob(m["clobTokenIds"], as_of)
            if cp is None:
                continue
            items.append(BacktestItem(
                qid=f"pm_{m['id']}", platform="polymarket", question=m["question"].strip(),
                outcome=int(fp), as_of=as_of, resolve_ts=end, crowd_prob=round(cp, 4),
                n_crowd=int(float(m.get("volume", 0)) // 100), category=_cat(m["question"]),
                cutoff_clean=end > CUTOFF_TS))
            if len(items) >= target:
                break
    return items


_CONFLICT_TERMS = ["war", "ceasefire", "invade", "invasion", "coup", "sanctions", "military strike", "nuclear",
                   "annex", "hostage", "protest", "election", "regime", "airstrike", "troops", "border conflict",
                   "peace deal", "referendum", "assassination", "missile"]


def pull_conflict_corpus(target=140, min_bettors=6, path="experiments/results/conflict_corpus.json"):
    """A LARGER geopolitics/conflict slice — search Manifold for conflict terms, keep resolved binary markets
    with a real crowd that are cutoff-clean AND name a detectable country (the slice where GDELT/structural
    grounding could matter), reconstruct the crowd prob at a fair as-of lead. Cached to `path`."""
    import urllib.parse

    from pathlib import Path

    from swm.api.gdelt_social import detect_country
    seen, items = set(), []
    for term in _CONFLICT_TERMS:
        try:
            batch = _get(f"https://api.manifold.markets/v0/search-markets?term={urllib.parse.quote(term)}"
                         f"&sort=most-popular&filter=resolved&contractType=BINARY&limit=100")
        except Exception:
            continue
        for m in batch:
            if (m["id"] in seen or not m.get("isResolved") or m.get("resolution") not in ("YES", "NO")
                    or m.get("uniqueBettorCount", 0) < min_bettors or not m.get("resolutionTime")
                    or not m.get("createdTime")):
                continue
            seen.add(m["id"])
            created, resolved = m["createdTime"], m["resolutionTime"]
            if resolved <= created or (resolved // 1000) <= CUTOFF_TS:      # cutoff-clean only
                continue
            if detect_country(m["question"])[0] is None:                    # must name a country to ground it
                continue
            as_of = int(created + AS_OF_FRAC * (resolved - created))
            cp = _manifold_crowd_prob(m["id"], as_of)
            if cp is None:
                continue
            items.append(BacktestItem(
                qid=f"mf_{m['id']}", platform="manifold", question=m["question"].strip(),
                outcome=1 if m["resolution"] == "YES" else 0, as_of=as_of // 1000,
                resolve_ts=resolved // 1000, crowd_prob=round(cp, 4), n_crowd=int(m["uniqueBettorCount"]),
                category=_cat(m["question"]), cutoff_clean=True))
            if len(items) >= target:
                break
        if len(items) >= target:
            break
    Path(path).write_text(json.dumps([asdict(it) for it in items], indent=0))
    return items


def build_corpus(*, manifold=3000, polymarket=1500, path="experiments/results/backtest_corpus.json"):
    items = pull_manifold(target=manifold) + pull_polymarket(target=polymarket)
    seen, uniq = set(), []
    for it in items:
        k = it.question.lower()[:80]
        if k not in seen:
            seen.add(k)
            uniq.append(it)
    from pathlib import Path
    rows = [asdict(it) for it in uniq]
    Path(path).write_text(json.dumps(rows, indent=0))
    return rows


def load_corpus(path="experiments/results/backtest_corpus.json"):
    from pathlib import Path
    return [BacktestItem(**r) for r in json.loads(Path(path).read_text())]
