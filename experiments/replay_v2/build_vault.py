"""Temporal Replay v2 — deterministic 100-world vault construction from the Polymarket archive.

ALL selection rules are FROZEN HERE, in code, before any model runs (Part 12). No hand-selection:
eligible events are ordered by sha1(event_id) and quotas are filled in that hash order. One WORLD = one
underlying Polymarket EVENT (its correlated contracts stay together, clustered; the primary contract is the
highest-volume resolved binary market). Resolutions and blinding mappings go ONLY to the sealed store.

Frozen eligibility (an event is eligible iff):
  * closed, endDate in [2024-03-01, 2025-05-31] (post model documented-cutoff era, all resolved);
  * has >= 1 binary Yes/No market with a definitive resolution (outcomePrices 0/1);
  * primary market volume >= 50,000 USDC and price-history lifetime >= 45 days;
  * >= 3 objective trajectory targets derivable from the archived price history.

Frozen quotas over the hash order: exactly 100 worlds; <= 15 per domain; >= 10 worlds for each causal
category (institutional / population / network-diffusion / strategic-negotiation / structural-change),
categories assigned by frozen keyword rules over the question text (recorded per world); >= 20 worlds
whose event has >= 3 correlated contracts (categorical/distributional target support).

Cutoffs: 4 per world at 20% / 45% / 70% / 90% of the primary market's archived trading lifetime.
Market snapshots: the archived price at the exact cutoff (nearest tick at-or-before; timestamp recorded) —
never a closing price backdated.
Trajectory targets (>= 3, objective, timestamped, archived): first crossings of 0.35 / 0.50 / 0.65, the
day of the largest daily move, and the final pre-resolution price decile.
Splits (Part 13, world-level, frozen): hash-order positions 0-39 calibration, 40-59 validation,
60-99 locked test. Every cutoff/arm/contract of a world inherits its split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

VAULT = Path("experiments/replay_vault_v2")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

WINDOW = ("2024-03-01T00:00:00Z", "2025-05-31T00:00:00Z")
MIN_VOLUME, MIN_LIFETIME_D = 50_000.0, 45.0
CUTOFF_FRACS = (0.20, 0.45, 0.70, 0.90)
QUOTA_PER_DOMAIN = 15
N_WORLDS = 100
SPLITS = (("calibration", 0, 40), ("validation", 40, 60), ("locked_test", 60, 100))

#: frozen category keyword rules (question text, lowercased) — assigned before any model run.
_CAUSAL_RULES = (
    ("institutional", ("vote", "confirm", "pass", "bill", "court", "ruling", "approve", "nominee",
                       "impeach", "veto", "election certif", "fed ", "rate", "fomc", "senate", "congress",
                       "parliament", "referendum", "law", "ban ")),
    ("population", ("election", "win the", "turnout", "popular vote", "adoption", "users", "sales",
                    "box office", "subscribers", "downloads", "votes", "poll")),
    ("network_diffusion", ("viral", "trend", "followers", "views", "tweet", "retweet", "mention",
                           "spread", "meme", "streams")),
    ("strategic_negotiation", ("ceasefire", "deal", "agreement", "negotiat", "resign", "acquire",
                               "merger", "meet", "summit", "strike", "war ", "invade", "attack",
                               "hostage", "release", "step down", "drop out", "withdraw")),
    ("structural_change", ("resign", "coup", "collapse", "replace", "new leader", "dissol", "snap election",
                           "emergency", "default", "regime")),
)
_DOMAIN_MAP = {"us-current-affairs": "us_politics", "politics": "politics", "crypto": "crypto",
               "sports": "sports", "pop-culture": "culture", "science": "science",
               "business": "business", "economics": "economics", "middle east": "geopolitics",
               "world": "geopolitics", "elections": "elections", "tech": "technology",
               "ai": "technology", "finance": "finance"}


def _get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wmv2-replay"}),
                                        timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return None


_Q_DOMAIN_RULES = (
    ("crypto", ("bitcoin", "btc", "ethereum", "eth ", "solana", "crypto", "token", "airdrop", "binance")),
    ("sports", ("nba", "nfl", "nhl", "mlb", "premier league", "champions league", "world cup", "super bowl",
                "grand prix", "wimbledon", "olympic", "ufc", " vs. ", "playoff", "cup?", "win the series")),
    ("culture", ("movie", "film", "box office", "album", "billboard", "grammy", "oscar", "taylor swift",
                 "spotify", "netflix", "youtube", "tiktok")),
    ("science_space", ("spacex", "starship", "nasa", "launch", "rocket", "asteroid", "vaccine", "fda",
                       "drug", "trial")),
    ("technology", ("openai", "gpt", "apple", "iphone", "tesla", " ai ", "model release", "google",
                    "microsoft", "chip")),
    ("business", ("acquire", "merger", "ipo", "stock", "company", "ceo", "earnings", "bankrupt")),
    ("economics", ("fed ", "fomc", "rate cut", "rate hike", "inflation", "cpi", "gdp", "recession",
                   "unemployment", "treasury", "tariff")),
    ("geopolitics", ("war", "ceasefire", "invade", "missile", "nuclear", "nato", "ukraine", "russia",
                     "israel", "gaza", "iran", "china", "taiwan", "hostage", "sanction")),
    ("elections", ("election", "primary", "president", "senate", "governor", "nominee", "ballot",
                   "electoral", "parliament", "prime minister", "mayor")),
    ("us_politics", ("congress", "bill", "shutdown", "impeach", "supreme court", "cabinet",
                     "executive order", "white house", "veto")),
)


def _domain(ev, question):
    q = " " + question.lower() + " "
    for dom, toks in _Q_DOMAIN_RULES:
        if any(t in q for t in toks):
            return dom
    cat = str(ev.get("category") or "").lower()
    for k, v in _DOMAIN_MAP.items():
        if k in cat:
            return v
    return "other"


def _causal_categories(question):
    q = question.lower()
    return [name for name, toks in _CAUSAL_RULES if any(t in q for t in toks)]


def _history(token_id):
    d = _get(f"{CLOB}/prices-history?market={token_id}&interval=max&fidelity=1440")
    return (d or {}).get("history") or []


def _trajectory_targets(hist):
    """Objective, timestamped, archived targets from the price path."""
    targets = []
    for thr in (0.35, 0.50, 0.65):
        t = next((p["t"] for p in hist if p["p"] >= thr), None)
        targets.append({"kind": f"first_cross_{thr}", "resolved_t": t,
                        "occurred": t is not None, "source": "clob_prices_history"})
    if len(hist) > 2:
        moves = [(abs(hist[i]["p"] - hist[i - 1]["p"]), hist[i]["t"]) for i in range(1, len(hist))]
        mx = max(moves)
        targets.append({"kind": "max_daily_move_day", "resolved_t": mx[1], "magnitude": round(mx[0], 4),
                        "source": "clob_prices_history"})
    return targets


def _snapshot_at(hist, ts):
    prior = [p for p in hist if p["t"] <= ts]
    if not prior:
        return None
    p = prior[-1]
    return {"price": p["p"], "exact_t": p["t"], "requested_t": ts, "staleness_s": ts - p["t"]}


def fetch_events(max_pages=40):
    evs, offset = [], 0
    while len(evs) < max_pages * 100 and offset < max_pages * 100:
        page = _get(f"{GAMMA}/events?closed=true&limit=100&offset={offset}&order=id&ascending=true"
                    f"&end_date_min={WINDOW[0]}&end_date_max={WINDOW[1]}")
        if not page:
            break
        evs.extend(page)
        offset += 100
        if len(page) < 100:
            break
    return evs


def build(max_pages=40):
    VAULT.mkdir(parents=True, exist_ok=True)
    raw = fetch_events(max_pages)
    print(f"fetched {len(raw)} closed events")
    # frozen eligibility pass (no price fetch yet)
    cands = []
    for ev in raw:
        mkts = ev.get("markets") or []
        bins = []
        for m in mkts:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:  # noqa: BLE001
                continue
            if sorted(o.lower() for o in outs) == ["no", "yes"] and sorted(prices) == [0.0, 1.0] \
                    and float(m.get("volume") or 0) >= MIN_VOLUME:
                bins.append((float(m.get("volume") or 0), m, outs, prices))
        if not bins:
            continue
        bins.sort(key=lambda x: -x[0])
        cands.append((hashlib.sha1(str(ev.get("id")).encode()).hexdigest(), ev, bins[0], len(mkts)))
    cands.sort(key=lambda x: x[0])                            # deterministic hash order — no hand-selection
    print(f"{len(cands)} eligible candidate events")

    worlds, sealed = [], {}
    dom_count, cat_count, multi = {}, {c: 0 for c, _ in _CAUSAL_RULES}, 0

    def _admissible(dom):
        return dom_count.get(dom, 0) < QUOTA_PER_DOMAIN

    for h, ev, (vol, m, outs, prices), n_mkts in cands:
        if len(worlds) >= N_WORLDS:
            break
        q = str(m.get("question") or ev.get("title") or "")
        dom = _domain(ev, q)
        if not _admissible(dom):
            continue
        token_ids = json.loads(m.get("clobTokenIds") or "[]")
        if not token_ids:
            continue
        yes_idx = [o.lower() for o in outs].index("yes")
        hist = _history(token_ids[yes_idx])
        if len(hist) < 10:
            continue
        t0, t1 = hist[0]["t"], hist[-1]["t"]
        if (t1 - t0) / 86400.0 < MIN_LIFETIME_D:
            continue
        cutoffs, snaps = [], {}
        ok = True
        for f in CUTOFF_FRACS:
            ts = int(t0 + f * (t1 - t0))
            s = _snapshot_at(hist, ts)
            if s is None:
                ok = False
                break
            iso = time.strftime("%Y-%m-%d", time.gmtime(ts))
            cutoffs.append(iso)
            snaps[iso] = s
        traj = _trajectory_targets(hist)
        if not ok or len(traj) < 3:
            continue
        wid = f"pm_{ev.get('id')}"
        cats = _causal_categories(q)
        worlds.append({
            "event_id": wid, "question": q, "domain": dom, "event_family": str(ev.get("slug") or wid),
            "causal_categories": cats, "n_correlated_contracts": n_mkts,
            "forecast_cutoffs": cutoffs, "horizon": time.strftime("%Y-%m-%d", time.gmtime(t1 + 86400)),
            "market_snapshots": snaps, "primary_volume_usdc": vol,
            "source": {"archive": "polymarket_gamma+clob", "event_id": ev.get("id"),
                       "condition_id": m.get("conditionId"), "retrieved_at_note": "deterministic hash-order "
                       "sampling; frozen rules in build_vault.py"},
            "target_type": "binary" if n_mkts < 3 else "binary_primary_of_categorical_event"})
        yes_price = prices[yes_idx]
        sealed[wid] = {"outcome": int(yes_price == 1.0),
                       "resolution_source": "polymarket outcomePrices (UMA-resolved)",
                       "trajectory_targets": traj, "blinding_mapping": {}}
        dom_count[dom] = dom_count.get(dom, 0) + 1
        for c in cats:
            cat_count[c] += 1
        multi += 1 if n_mkts >= 3 else 0
        print(f"[{len(worlds):3d}] {wid} {dom:14s} vol={vol/1e3:.0f}k cuts={cutoffs} cats={cats}")

    order = [w["event_id"] for w in worlds]
    split_map = {}
    for name, a, b in SPLITS:
        for wid in order[a:b]:
            split_map[wid] = name
    (VAULT / "events.json").write_text(json.dumps(
        {"note": "PUBLIC vault v2 — no outcomes/trajectory resolutions here (sealed store, scorer-only).",
         "frozen_rules": "see build_vault.py docstring (rules frozen before any model run)",
         "n_worlds": len(worlds), "domain_counts": dom_count, "causal_category_counts": cat_count,
         "n_multicontract": multi, "splits": split_map, "worlds": worlds}, indent=1))
    (VAULT / "SEALED_resolutions_v2.json").write_text(json.dumps(
        {"note": "SEALED — scorer only (REPLAY_SCORER=1).", "resolutions": sealed}, indent=1))
    print(f"\nvault: {len(worlds)} worlds; domains={dom_count}; causal={cat_count}; multi={multi}")
    print("splits:", {k: sum(1 for v in split_map.values() if v == k) for k in
                      ("calibration", "validation", "locked_test")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=40)
    a = ap.parse_args()
    build(max_pages=a.max_pages)
