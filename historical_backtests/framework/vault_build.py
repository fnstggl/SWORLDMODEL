"""Build ONE benchmark version's sealed question vault + isolated resolution vault.

Source: Polymarket gamma+CLOB public archives (independently timestamped opens, price paths, and
UMA-resolved outcomes). All selection rules are DETERMINISTIC and frozen in this file before any
case is examined; no LLM (frontier or period) participates in selection. This module runs
scorer-side (REPLAY_SCORER=1) because building the resolution vault requires outcomes; the
forecaster never imports it.

Eligibility (frozen):
 - question_open strictly after the model's effective temporal boundary (release), proven by the
   market's own createdAt/startDate timestamps;
 - resolved before VAULT_BUILD_FREEZE with a decisive outcome (outcomePrices 0/1);
 - binary yes/no, volume >= MIN_VOLUME, >= MIN_UNRESOLVED_DAYS of unresolved lifetime;
 - not excluded (sports / pure price thresholds / celebrity trivia / mechanical-calendar — see
   scales._EXCLUDE_TOKENS);
 - price at EVERY cutoff within [0.05, 0.95] (genuinely unresolved; no freebies);
 - event-cluster cap: <= 3 questions per gamma event slug;
 - scale quotas (scales.QUOTAS) filled greedily in chronological order, then remaining slots
   chronological.

Cutoffs (frozen rule): fractions (0.15, 0.35, 0.55, 0.75) of the UNRESOLVED lifetime
[question_open, realized_resolution), each >= 1 day after open and >= 1 day before resolution.
The question vault stores cutoffs + the ex-ante scheduled deadline (market endDate) — never the
realized resolution time or outcome (those live only in the isolated resolution vault).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.replay_v2.build_vault as V2B                      # verified gamma/CLOB helpers
from experiments.replay_v3.fit_survival_pack import _iso_ts, _yes_token, effective_resolution_fraction
from historical_backtests.framework import scales
from historical_backtests.framework.resolution_store import write_resolutions, seal_file
from historical_backtests.models.registry import get_model, _ts

ROOT = Path(__file__).resolve().parents[1]
CUTOFF_FRACS = (0.15, 0.35, 0.55, 0.75)
MIN_VOLUME = 20000.0
MIN_UNRESOLVED_DAYS = 14.0
UNDECIDED = (0.05, 0.95)
CLUSTER_CAP = 3
N_TOTAL, N_CAL, N_VAL, N_LOCK = 100, 40, 20, 40
SELECTION_SEED = 20260717                                            # frozen before forecasting

_DOMAINS = (("geopolitics", ("war", "ceasefire", "nato", "invasion", "nuclear", "sanction",
                             "hostage", "missile", "treaty", "iran", "russia", "ukraine", "china",
                             "taiwan", "israel", "gaza")),
            ("elections", ("election", "primary", "nominee", "candidate", "win the", "governor",
                           "mayor", "presiden")),
            ("legislation_regulation", ("congress", "senate", "bill", "law", "regulat", "fda",
                                        "ban", "act ", "tariff", "shutdown", "budget", "impeach")),
            ("courts", ("court", "judge", "ruling", "convict", "sentence", "lawsuit", "trial")),
            ("central_banks_macro", ("fed", "rate", "inflation", "recession", "gdp", "ecb")),
            ("company_strategy", ("ceo", "merger", "acquisition", "ipo", "launch", "company",
                                  "openai", "tesla", "spacex", "tiktok")),
            ("appointments_departures", ("resign", "fired", "step down", "appoint", "confirm",
                                         "nominat", "cabinet", "out as")),
            ("collective_adoption", ("countries", "recognize", "adopt", "membership", "users",
                                     "sales", "percent", "majority")))


def _domain(q: str) -> str:
    ql = q.lower()
    for name, toks in _DOMAINS:
        if any(t in ql for t in toks):
            return name
    return "other_social"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def fetch_pool(model: dict, *, freeze_iso: str, max_pages: int = 60) -> list:
    """Deterministic candidate pool: closed high-volume binary markets, volume-descending, whose
    market createdAt/startDate is strictly after the model release and endDate before the freeze."""
    rel_ts = _ts(model["effective_temporal_boundary"])
    freeze_ts = _ts(freeze_iso)
    pool, offset = [], 0
    while offset < max_pages * 100:
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false"
                        f"&end_date_min=2024-08-01&end_date_max={freeze_iso[:10]}") or []
        if not page:
            break
        offset += 100
        for m in page:
            q = str(m.get("question") or "").strip()
            outs = sorted(o.lower() for o in json.loads(m.get("outcomes") or "[]"))
            if outs != ["no", "yes"] or not q or not m.get("conditionId"):
                continue
            if float(m.get("volume") or 0) < MIN_VOLUME:
                continue
            open_ts = _iso_ts(m.get("startDate")) or _iso_ts(m.get("createdAt"))
            if not open_ts or open_ts <= rel_ts:
                continue                                     # question must OPEN after release
            prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            if sorted(prices) != [0.0, 1.0]:
                continue                                     # decisive UMA resolution only
            if scales.excluded(q):
                continue
            end_ts = _iso_ts(m.get("endDate"))
            if not end_ts or end_ts >= freeze_ts:
                continue
            pool.append({"m": m, "q": q, "open_ts": open_ts, "end_ts": end_ts})
    # dedupe by conditionId, deterministic order: volume already descending; stabilize by id
    seen, out = set(), []
    for c in pool:
        cid = c["m"]["conditionId"]
        if cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def materialize(c: dict) -> dict | None:
    """Price path → realized resolution ts (sticky/early-close proxy over the SCHEDULED window),
    cutoffs, per-cutoff market snapshots, undecided guard. None = ineligible."""
    m = c["m"]
    tok = _yes_token(m)
    hist = V2B._history(tok) if tok else []
    if len(hist) < 8:
        return None
    frac, ok, proxy = effective_resolution_fraction(
        hist, end_ts=c["end_ts"], closed_ts=_iso_ts(m.get("closedTime")))
    t0 = hist[0]["t"]
    open_ts = max(c["open_ts"], t0)
    yes_idx = [o.lower() for o in json.loads(m.get("outcomes") or "[]")].index("yes")
    outcome = int([float(x) for x in json.loads(m.get("outcomePrices") or "[]")][yes_idx] == 1.0)
    if frac is not None:
        resolution_ts = t0 + frac * (c["end_ts"] - t0)
    else:
        if outcome == 1:
            return None                                      # YES without a crossing: path too odd
        resolution_ts = c["end_ts"]                          # NO resolves at scheduled deadline
    unresolved_days = (resolution_ts - open_ts) / 86400.0
    if unresolved_days < MIN_UNRESOLVED_DAYS:
        return None
    cutoffs, snaps = [], {}
    for f in CUTOFF_FRACS:
        ts = open_ts + f * (resolution_ts - open_ts)
        if ts < open_ts + 86400.0 or ts > resolution_ts - 86400.0:
            return None
        s = V2B._snapshot_at(hist, int(ts))
        if s is None or not (UNDECIDED[0] <= s["price"] <= UNDECIDED[1]):
            return None                                      # already-decided at a cutoff → out
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))
        cutoffs.append(iso)
        snaps[iso] = {"market_price": s["price"], "price_exact_t": s["exact_t"],
                      "staleness_s": s["staleness_s"]}
    ev = m.get("events") or [{}]
    cluster = str((ev[0] or {}).get("slug") or m.get("slug") or m["conditionId"])[:60]
    return {"case": {"case_id": f"hb_{m['conditionId'][:16]}",
                     "raw_question": c["q"], "question_sha256": _sha(c["q"]),
                     "question_open": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(open_ts)),
                     "question_open_ts": open_ts,
                     "forecast_cutoffs": cutoffs, "market_snapshots": snaps,
                     "resolution_deadline": str(m.get("endDate")),
                     "resolution_deadline_ts": c["end_ts"],
                     "resolution_criterion": (str(m.get("description") or "")[:900]
                                              or f"resolves YES iff: {c['q']}"),
                     "domain": _domain(c["q"]),
                     "causal_scale": None, "secondary_scales": None,
                     "cluster_id": cluster, "split": None,
                     "source": {"archive": "polymarket_gamma+clob",
                                "condition_id": m["conditionId"],
                                "market_slug": m.get("slug"), "volume": float(m.get("volume") or 0)}},
            "resolution": {"actual_outcome": outcome,
                           "resolution_ts": resolution_ts,
                           "resolution_time_proxy": proxy,
                           "resolution_source": "polymarket outcomePrices (UMA) + CLOB price path",
                           "source_ts": c["end_ts"],
                           "source_hash": _sha(json.dumps(
                               {"cid": m["conditionId"],
                                "outcomePrices": m.get("outcomePrices")}, sort_keys=True))}}


def select(cands: list) -> list:
    """Quota-first greedy fill in chronological open order, cluster-capped, then chronological
    fill to N_TOTAL. Deterministic: sort key (open_ts, case_id)."""
    cands = sorted(cands, key=lambda x: (x["case"]["question_open_ts"], x["case"]["case_id"]))
    for x in cands:
        pri, sec = scales.classify_scale(x["case"]["raw_question"])
        x["case"]["causal_scale"], x["case"]["secondary_scales"] = pri, sec
    chosen, cluster_n, scale_n = [], {}, {s: 0 for s in scales.SCALES}

    def _take(x):
        cl = x["case"]["cluster_id"]
        if cluster_n.get(cl, 0) >= CLUSTER_CAP:
            return False
        chosen.append(x)
        cluster_n[cl] = cluster_n.get(cl, 0) + 1
        scale_n[x["case"]["causal_scale"]] += 1
        return True
    for scale, quota in scales.QUOTAS.items():               # pass 1: fill quotas
        for x in cands:
            if scale_n[scale] >= quota or len(chosen) >= N_TOTAL:
                break
            if x not in chosen and x["case"]["causal_scale"] == scale:
                _take(x)
    for x in cands:                                          # pass 2: chronological fill
        if len(chosen) >= N_TOTAL:
            break
        if x not in chosen:
            _take(x)
    chosen.sort(key=lambda x: (x["case"]["question_open_ts"], x["case"]["case_id"]))
    for i, x in enumerate(chosen):                           # chronological deterministic split
        x["case"]["split"] = ("calibration" if i < N_CAL
                              else "validation" if i < N_CAL + N_VAL else "rotating_locked")
    return chosen


def build(benchmark_id: str, registry_model_id: str, *, freeze_iso: str, max_pages: int = 60):
    model = get_model(registry_model_id)
    print(f"pool fetch (volume-desc, open>{model['effective_temporal_boundary']}) …", flush=True)
    pool = fetch_pool(model, freeze_iso=freeze_iso, max_pages=max_pages)
    print(f"pool: {len(pool)} candidates; materializing price paths …", flush=True)
    cands = []
    for i, c in enumerate(pool):
        try:
            out = materialize(c)
        except Exception as e:  # noqa: BLE001
            out = None
            print(f"  [{i}] error {type(e).__name__}: {e}", flush=True)
        if out:
            cands.append(out)
        if i % 50 == 49:
            print(f"  … {i + 1}/{len(pool)} → eligible {len(cands)}", flush=True)
        if len(cands) >= N_TOTAL * 4:                        # enough headroom for quota fill
            break
        time.sleep(0.05)
    chosen = select(cands)
    bdir = ROOT / "benchmark_versions" / benchmark_id
    bdir.mkdir(parents=True, exist_ok=True)
    qvault = {"benchmark_id": benchmark_id, "registry_model_id": registry_model_id,
              "vault_frozen_at": freeze_iso, "selection_seed": SELECTION_SEED,
              "selection_rules": "vault_build.py frozen rules (quota-first chronological, "
                                 f"cluster<= {CLUSTER_CAP}, undecided {UNDECIDED}, "
                                 f"cutoff fracs {CUTOFF_FRACS})",
              "n_cases": len(chosen), "cases": [x["case"] for x in chosen]}
    qpath = bdir / "question_vault.json"
    qpath.write_text(json.dumps(qvault, indent=1))
    qdigest = seal_file(qpath)
    rdigest = write_resolutions(benchmark_id, {x["case"]["case_id"]: x["resolution"]
                                               for x in chosen})
    comp = scales.composition_report([x["case"] for x in chosen])
    (bdir / "composition_report.json").write_text(json.dumps(comp, indent=1))
    print(json.dumps(comp, indent=1))
    print(f"question vault sealed sha={qdigest[:16]} ({len(chosen)} cases); "
          f"resolution vault sealed sha={rdigest[:16]} (isolated)")
    return qvault, comp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="openrouter_llama31_v1")
    ap.add_argument("--model", default="llama31_70b_instruct_post_release")
    ap.add_argument("--freeze", default=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ap.add_argument("--max-pages", type=int, default=60)
    a = ap.parse_args()
    build(a.benchmark, a.model, freeze_iso=a.freeze, max_pages=a.max_pages)
