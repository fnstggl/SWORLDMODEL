"""EXP-074: the portfolio backtest — map WHERE fidelity wins across many real domains, no-cheat.

Runs the event-backtest harness across six real domains at low vs high fidelity, to learn — from data, not
assertion — where the rich calibrated simulation beats the skeptic's baselines and where a simple model
already wins. This is the general social world model's honest scoreboard.

Domains (three downloaded this session — elections, adoption, referenda — plus opinion, macro):
  - gss_social / gss_spend  (population opinion, GSS)          — modelable evolving population
  - adoption                (OWID tech-adoption S-curves)       — diffusion mechanism vs persistence
  - referenda               (Swiss ballot measures 1848-2026)   — base-rate-dominated?
  - senate                  (MIT Senate returns 1976-2024)      — persistence-dominated?
  - fomc                    (FRED macro rate moves)             — momentum-dominated?

New datasets are parsed once into committed caches under experiments/results/exp074/ for reproducibility
(the raw downloads live in gitignored data/). Run: python -m experiments.exp074_portfolio_backtest
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from experiments.datasets_gss import load as load_gss
from experiments.exp073_event_backtest import ATTRS_FULL, _forecasts_for, _item_rows
from swm.eval.event_backtest import Question
from swm.eval.portfolio import Domain, run_portfolio, summarize
from swm.variables.calibrated_weights import CalibratedWeights, uninformative_prior

CACHE = "experiments/results/exp074"
RESULT = "experiments/results/exp074_portfolio_backtest.json"
GSS_SOCIAL = ["cappun", "gunlaw", "grass", "abany", "homosex", "premarsx", "fefam", "fepol", "letdie1"]
GSS_SPEND = ["natheal", "natenvir", "natfare", "nateduc", "natrace", "natcrime"]


def _clip(p, lo=1e-4, hi=1 - 1e-4):
    return lo if p < lo else (hi if p > hi else p)


# ============================ parse the 3 new datasets into committed caches ============================
def _parse_adoption():
    out = f"{CACHE}/adoption.json"
    if os.path.exists(out):
        return json.loads(Path(out).read_text())
    series = defaultdict(list)
    with open("data/owid_tech_adoption_us.csv", newline="") as f:
        for row in csv.reader(f):
            if row[1] == "Year" or len(row) < 3:
                continue
            try:
                series[row[0]].append((int(row[1]), float(row[2])))
            except ValueError:
                continue
    data = {k: sorted(v) for k, v in series.items() if len(v) >= 8}
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data))
    return data


def _parse_referenda():
    out = f"{CACHE}/referenda.json"
    if os.path.exists(out):
        return json.loads(Path(out).read_text())
    rows = []
    with open("data/swissvotes_dataset.csv", newline="", encoding="latin-1") as f:
        r = csv.reader(f, delimiter=";")
        header = next(r)
        idx = {name: header.index(name) for name in ("datum", "annahme", "volkja-proz", "rechtsform")
               if name in header}
        for row in r:
            try:
                datum = row[idx["datum"]]
                year = int(datum.split(".")[-1]) if "." in datum else int(datum[:4])
                acc = row[idx["annahme"]].strip()
                if acc not in ("0", "1"):
                    continue
                rows.append({"year": year, "date": datum, "accepted": int(acc),
                             "type": row[idx["rechtsform"]].strip()})
            except (ValueError, KeyError, IndexError):
                continue
    rows.sort(key=lambda x: x["year"])
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows))
    return rows


def _parse_senate():
    out = f"{CACHE}/senate.json"
    if os.path.exists(out):
        return json.loads(Path(out).read_text())
    agg = defaultdict(lambda: defaultdict(float))     # (year,state) -> {party: votes}
    with open("data/mit_senate_1976_2024.tab", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("stage") != "gen" or row.get("special") == "True":
                continue
            party = (row.get("party_simplified") or "").strip('"')
            try:
                v = float(row["candidatevotes"])
            except (ValueError, KeyError):
                continue
            agg[(int(row["year"]), row["state"].strip('"'))][party] += v
    races = []
    for (year, state), pv in agg.items():
        dem, rep = pv.get("DEMOCRAT", 0.0), pv.get("REPUBLICAN", 0.0)
        if dem + rep > 0:
            races.append({"year": year, "state": state, "dem_share": dem / (dem + rep)})
    races.sort(key=lambda x: (x["state"], x["year"]))
    Path(CACHE).mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(races))
    return races


# ============================ domain adapters ============================
def _gss_domain(rows_by_item, items, name):
    def build(fid):
        attrs = ["party", "age"] if fid == "few" else ATTRS_FULL
        fc, qs = _forecasts_for(rows_by_item, attrs, items)
        return qs, (lambda q: fc[q.qid])
    return Domain(name, build, ("few", "full"), kind="population")


def _adoption_domain(data):
    """As-of year Y (multiple origins), predict a technology's adoption at Y+10. few = persistence;
    full = logit-linear S-curve extrapolation (the diffusion mechanism)."""
    HORIZON = 10
    def _questions():
        qs, ctx = [], {}
        for tech, pts in data.items():
            yrs = [y for y, _ in pts]
            val = dict(pts)
            for i, (y, p) in enumerate(pts):
                t = y + HORIZON
                if t not in val:
                    continue
                hist = [(yy, val[yy]) for yy in yrs if yy <= y]
                if len(hist) < 4:
                    continue
                qid = f"{tech}@{y}->{t}"
                trend = hist[-1][1] + (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0] + 1e-9) * HORIZON
                qs.append(Question(qid, _clip(val[t] / 100.0),
                                   {"persistence": _clip(p / 100.0), "linear_trend": _clip(trend / 100.0)},
                                   asof=str(y), resolved=str(t)))
                ctx[qid] = (hist, y, t)
        return qs, ctx
    qs, ctx = _questions()

    def build(fid):
        if fid == "few":
            return qs, (lambda q: q.baselines["persistence"])
        def scurve(q):
            hist, y, t = ctx[q.qid]
            xs = [yy for yy, _ in hist]; ys = [_clip(v / 100.0) for _, v in hist]
            lz = [math.log(p / (1 - p)) for p in ys]
            mx = sum(xs) / len(xs); mz = sum(lz) / len(lz)
            den = sum((x - mx) ** 2 for x in xs) or 1.0
            b = sum((x - mx) * (z - mz) for x, z in zip(xs, lz)) / den
            a = mz - b * mx
            return _clip(1.0 / (1.0 + math.exp(-(a + b * t))))
        return qs, scurve
    return Domain("adoption", build, ("few", "full"), kind="diffusion")


def _referenda_domain(rows):
    """Predict a ballot measure's pass/fail. few = running base rate of passage; full = type-conditioned
    running rate. Rolling by date (train = all prior referenda)."""
    start = int(0.5 * len(rows))
    def build(fid):
        qs, fc = [], {}
        for i in range(start, len(rows)):
            prior = rows[:i]
            base = sum(r["accepted"] for r in prior) / len(prior)
            if fid == "few":
                p = base
            else:
                same = [r["accepted"] for r in prior if r["type"] == rows[i]["type"]]
                p = (sum(same) + base * 5) / (len(same) + 5) if same else base   # shrink to base rate
            qid = f"ref@{rows[i]['date']}#{i}"
            fc[qid] = _clip(p)
            qs.append(Question(qid, float(rows[i]["accepted"]),
                               {"base_rate": _clip(base), "persistence": float(rows[i - 1]["accepted"])},
                               asof=str(rows[i]["year"] - 1), resolved=str(rows[i]["year"])))
        return qs, (lambda q: fc[q.qid])
    return Domain("referenda", build, ("few", "full"), kind="referendum")


def _senate_domain(races):
    """Predict a state's Democratic two-party Senate share. few = persistence (state's last result);
    full = persistence + national uniform swing since that state's last race."""
    by_state = defaultdict(list)
    for r in races:
        by_state[r["state"]].append(r)
    for s in by_state:
        by_state[s].sort(key=lambda x: x["year"])
    year_mean = {}
    tmp = defaultdict(list)
    for r in races:
        tmp[r["year"]].append(r["dem_share"])
    for y, v in tmp.items():
        year_mean[y] = sum(v) / len(v)

    def build(fid):
        qs, fc = [], {}
        for state, seq in by_state.items():
            for k in range(1, len(seq)):
                prev, cur = seq[k - 1], seq[k]
                if cur["year"] < 1994:                     # a burn-in so baselines have history
                    continue
                swing = year_mean.get(cur["year"], 0.5) - year_mean.get(prev["year"], 0.5)
                p = prev["dem_share"] if fid == "few" else _clip(prev["dem_share"] + swing)
                qid = f"{state}@{cur['year']}"
                fc[qid] = _clip(p)
                qs.append(Question(qid, cur["dem_share"],
                                   {"persistence": prev["dem_share"], "base_rate": 0.5},
                                   asof=str(prev["year"]), resolved=str(cur["year"])))
        return qs, (lambda q: fc[q.qid])
    return Domain("senate", build, ("few", "full"), kind="election")


def _fomc_domain():
    data = json.loads(Path("experiments/results/exp071/fomc_macro.json").read_text())
    rows = []
    for i, d in enumerate(data):
        if d.get("move_fwd3") is None:
            continue
        rate3 = data[max(0, i - 3)]["rate"]
        rows.append({"month": d["month"],
                     "x": {"inflation": d["inflation"] / 10.0, "unemp": d["unemp"] / 10.0,
                           "rate": d["rate"] / 10.0, "recent_move": max(-1.0, min(1.0, d["rate"] - rate3))},
                     "y": 1 if d["move_fwd3"] > 0.1 else 0,
                     "mom": 1.0 if (d["rate"] - rate3) > 0.1 else (0.0 if (d["rate"] - rate3) < -0.1 else 0.5)})

    def build(fid):
        feats = ["inflation"] if fid == "few" else ["inflation", "unemp", "rate", "recent_move"]
        cut = int(0.6 * len(rows))
        tr, te = rows[:cut], rows[cut:]
        base = sum(r["y"] for r in tr) / len(tr)
        # CALIBRATED momentum baseline: train P(hike | recent-move bucket), not a hard 0/1 call (fair on log-loss)
        buck = defaultdict(list)
        for r in tr:
            buck[r["mom"]].append(r["y"])
        mom_rate = {k: (sum(v) + base * 3) / (len(v) + 3) for k, v in buck.items()}
        g = CalibratedWeights([uninformative_prior(f) for f in feats], temper_grid=(1.0, 4.0),
                              epochs=80).fit([[r["x"][f] for f in feats] for r in tr], [r["y"] for r in tr],
                                             tune=True)
        fc, qs = {}, []
        for r in te:
            qid = f"fomc@{r['month']}"
            fc[qid] = _clip(g.predict([r["x"][f] for f in feats]))
            qs.append(Question(qid, float(r["y"]),
                               {"momentum": _clip(mom_rate.get(r["mom"], base)), "base_rate": _clip(base)},
                               asof=r["month"], resolved=r["month"] + "-fwd3"))
        return qs, (lambda q: fc[q.qid])
    return Domain("fomc", build, ("few", "full"), kind="macro")


def run() -> dict:
    gss = load_gss()
    rbi = {it: _item_rows(gss, it) for it in GSS_SOCIAL + GSS_SPEND}
    domains = [
        _gss_domain(rbi, GSS_SOCIAL, "gss_social"),
        _gss_domain(rbi, GSS_SPEND, "gss_spend"),
        _adoption_domain(_parse_adoption()),
        _referenda_domain(_parse_referenda()),
        _senate_domain(_parse_senate()),
        _fomc_domain(),
    ]
    port = run_portfolio(domains)
    Path(RESULT).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(port, indent=1, default=str))
    print("EXP-074  portfolio backtest — WHERE does fidelity win? (6 real domains, no-cheat)\n")
    print(summarize(port))
    wins = [n for n, m in port["map"].items() if m.get("fidelity_helps") and m.get("beats_all_baselines")]
    print(f"\n  fidelity WINS (higher fidelity beats all baselines): {wins}")
    print(f"  wrote {RESULT}")
    return port


if __name__ == "__main__":
    run()
