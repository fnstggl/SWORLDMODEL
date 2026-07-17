"""Fit the INTENTION HAZARD-RATIO pack from a labeled statement→hazard-change corpus (item: effect
sizes must be measurements, not priors).

For each RESOLVED market in the governance-safe corpus (calibration split + wide corpus, sealed
splits excluded by condition_id and question — same rules as fit_survival_pack):
 1. archived news inside the market's window is fetched with the paired-date Google News RSS
    connector (as-of enforced: after:/before: both required);
 2. the LLM CLASSIFIES each dated public statement by a named actor into the universal stance
    taxonomy (committed_to_prevent … formally_committed) — classification only, no numbers;
 3. the statement's EFFECT is MEASURED from the archived price path: the market-implied resolution
    hazard in the post-statement window over the pre-statement window (constant-hazard-to-deadline
    inversion λ_t = −ln(1−p_t)/(T−t), medians over ±window);
 4. rows {commitment_level, hazard_ratio} feed event_time.fit_intention_hazard_ratios (partial
    pooling toward no-effect) and the result is written to intention_hr_pack.json — which REPLACES
    the documented priors wholesale at load time (event_time._hr_table).

Requires network + DEEPSEEK_API_KEY. The measurement functions below are PURE and unit-tested
offline (tests/test_wmv2_fitting_scripts.py); the fetch/classify loop is resumable
(intention_hr_rows.jsonl keyed by condition_id).
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VAULT = Path("experiments/replay_vault_v3")
ROWS = VAULT / "intention_hr_rows.jsonl"
PACK = VAULT / "intention_hr_pack.json"
WINDOW_S = 7 * 86400.0                                       # ± measurement window around a statement
MIN_PRE_POINTS = 3
MIN_POST_POINTS = 3
CORPUS_TARGET = 120                                          # statements, not markets
MODEL = "deepseek-chat"

_CLASSIFY_PROMPT = """Below are dated news headlines from the window of a resolved forecasting market.
Identify PUBLIC STATEMENTS by named actors that bear on the market's outcome, and classify each into
the universal stance taxonomy. CLASSIFY ONLY — never estimate numbers.

MARKET QUESTION: {q}
HEADLINES (with dates):
{items}

Return ONLY JSON:
{{"statements": [{{"actor": "<named person/institution>", "date": "<YYYY-MM-DD>",
  "quote": "<the statement, short>",
  "commitment_level": "committed_to_prevent|conditionally_opposed|weakly_opposed|neutral|inclined_toward|actively_pursuing|formally_committed",
  "reliability": "high|medium|low"}}]}}"""


# ---------------------------------------------------------------- pure measurement (unit-tested)
def implied_hazard(price: float, t: float, deadline_ts: float) -> float | None:
    """Market-implied constant resolution hazard from a YES price: p_t = 1 − exp(−λ(T−t)) ⇒
    λ = −ln(1−p)/(T−t). None outside a measurable regime (price ~0/1 or t at/after deadline)."""
    if not (0.001 <= price <= 0.985) or deadline_ts - t < 3600.0:
        return None
    return -math.log(1.0 - price) / (deadline_ts - t)


def statement_hazard_ratio(hist: list, statement_ts: float, deadline_ts: float,
                           *, window_s: float = WINDOW_S) -> float | None:
    """MEASURED effect of a dated statement: median implied hazard over (t, t+window] divided by the
    median over [t−window, t). None when either side lacks enough points or a measurable hazard —
    a row is only emitted when the measurement is real."""
    pre, post = [], []
    for p in (hist or []):
        lam = implied_hazard(float(p["p"]), float(p["t"]), deadline_ts)
        if lam is None:
            continue
        if statement_ts - window_s <= p["t"] < statement_ts:
            pre.append(lam)
        elif statement_ts < p["t"] <= statement_ts + window_s:
            post.append(lam)
    if len(pre) < MIN_PRE_POINTS or len(post) < MIN_POST_POINTS:
        return None
    pre.sort(); post.sort()
    med_pre, med_post = pre[len(pre) // 2], post[len(post) // 2]
    if med_pre <= 0:
        return None
    return med_post / med_pre


# ---------------------------------------------------------------- corpus loop (network + LLM)
def _llm():
    from swm.api.deepseek_backend import deepseek_chat_fn
    return deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=1200, temperature=0.1)


def _date_ts(s):
    import datetime as dt
    try:
        return dt.datetime.fromisoformat(str(s)[:10]).replace(tzinfo=dt.timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


def main():
    import experiments.replay_v2.build_vault as V2B
    from experiments.replay_v3.fit_survival_pack import (_iso_ts, _market_by_condition, _yes_token)
    from swm.world_model_v2.event_time import fit_intention_hazard_ratios
    from swm.world_model_v2.evidence_connectors import GoogleNewsRSSConnector
    from swm.engine.grounding import parse_json

    ev = json.loads((VAULT / "events.json").read_text())
    cal = [w for w in ev["worlds"] if ev["splits"].get(w["event_id"]) == "calibration"]
    sealed = [w for w in ev["worlds"] if ev["splits"].get(w["event_id"]) != "calibration"]
    excluded_cids = {(w.get("source") or {}).get("condition_id") for w in sealed}
    excluded_qs = {str(w.get("question", "")).strip().lower() for w in sealed}

    done = set()
    rows = []
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
            done.add(r["condition_id"])
    llm = _llm()
    news = GoogleNewsRSSConnector()

    def _process(cid, question):
        m = _market_by_condition(cid)
        if not m:
            return 0
        tok = _yes_token(m)
        hist = V2B._history(tok) if tok else []
        if len(hist) < 20:
            return 0
        t0, t_end = hist[0]["t"], max(_iso_ts(m.get("endDate")) or 0.0, hist[-1]["t"])
        import datetime as dt
        after = dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime("%Y-%m-%d")
        before = dt.datetime.fromtimestamp(t_end, dt.timezone.utc).strftime("%Y-%m-%d")
        items, trace = news.search_historical(question[:80], after_date=after, before_date=before,
                                              requirement_id=f"hr:{cid[:12]}", k=20)
        if not items:
            return 0
        lines = "\n".join(f"- [{i.feed_pubdate[:16]}] {i.title}" for i in items)
        raw = parse_json(llm(_CLASSIFY_PROMPT.format(q=question, items=lines[:2400]))) or {}
        n = 0
        for st in (raw.get("statements") or []):
            ts = _date_ts(st.get("date"))
            lvl = str(st.get("commitment_level", "")).strip().lower()
            if ts is None or not lvl or lvl == "neutral":
                continue
            hr = statement_hazard_ratio(hist, ts, t_end)
            if hr is None or not (0.05 <= hr <= 20.0):
                continue
            row = {"condition_id": cid, "question": question[:120],
                   "actor": str(st.get("actor", ""))[:60], "date": st.get("date"),
                   "quote": str(st.get("quote", ""))[:160],
                   "commitment_level": lvl, "reliability": st.get("reliability"),
                   "hazard_ratio": round(hr, 4)}
            rows.append(row)
            with ROWS.open("a") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
        return n

    n_stmt = sum(1 for r in rows)
    for w in cal:                                            # calibration split first
        cid = (w.get("source") or {}).get("condition_id")
        if not cid or cid in done:
            continue
        done.add(cid)
        n_stmt += _process(cid, w["question"])
        time.sleep(0.5)
        if n_stmt >= CORPUS_TARGET:
            break
    offset = 0
    while n_stmt < CORPUS_TARGET and offset < 3000:          # wide corpus, sealed splits excluded
        page = V2B._get(f"{V2B.GAMMA}/markets?closed=true&limit=100&offset={offset}"
                        f"&order=volumeNum&ascending=false"
                        f"&end_date_min=2025-01-01&end_date_max=2026-06-30") or []
        if not page:
            break
        offset += 100
        for m in page:
            cid, q = m.get("conditionId"), str(m.get("question") or "").strip()
            if (not cid or cid in done or cid in excluded_cids or not q
                    or q.lower() in excluded_qs):
                continue
            done.add(cid)
            n_stmt += _process(cid, q)
            time.sleep(0.5)
            if n_stmt >= CORPUS_TARGET:
                break

    pack = fit_intention_hazard_ratios(rows)
    pack["measurement"] = ("market-implied hazard ratio, median post/pre ±7d around each classified "
                           "dated statement; λ = −ln(1−p)/(T−t)")
    pack["governance"] = ("validation/locked benchmark worlds excluded by condition_id and question; "
                          "pack carries stance-class effect sizes only")
    pack["n_markets"] = len({r["condition_id"] for r in rows})
    PACK.write_text(json.dumps(pack, indent=1))
    print(f"wrote {PACK}: {pack['n_rows']} rows / {pack['n_markets']} markets → "
          f"{json.dumps(pack['hazard_ratios'], indent=1)}")


if __name__ == "__main__":
    main()
