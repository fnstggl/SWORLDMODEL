"""Real intervention-dataset loaders + dataset cards for the Phase 13 real benchmark (Parts 25, 30B).

Every loader returns a normalized list of DECISION ROWS — dicts with a `context` (features observable
before the action), an `action` (the treatment actually taken), a `reward` (the recorded outcome), a
`propensity` (P(action | context) under the assignment process — KNOWN for randomized/logged data,
None otherwise), and a `cluster` (the decision-environment resampling unit, Parts 32/36). Randomized
experiments carry their design propensity; quasi-experimental designs carry None (the benchmark uses
their identification design, not IPS). Each loader has a CARD dict documenting unit / treatment /
outcome / assignment / design / domain / source / license, and every loader records how many malformed
rows it dropped.

Raw files live under data/phase13_real/ (gitignored; fetch.py re-downloads by URL+sha16). NONE of the
outcomes are simulated: these are recorded results of real interventions and quasi-experiments.
"""
from __future__ import annotations

import csv
import hashlib
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "phase13_real")


def _p(name):
    return os.path.join(RAW, name)


def sha16(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _rows(path):
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


# ============================================================ RANDOMIZED (identified ground truth)
def load_thornton_hiv():
    """Thornton (2008) HIV-results RCT, Malawi: randomized cash incentive to learn one's HIV test
    result; outcome = came to learn the result. Randomized -> propensity is the treated share."""
    rows = [r for r in _rows(_p("thornton_hiv.csv"))]
    treat = [1 if _f(r["got"]) else 0 for r in rows]  # 'got' incentive assignment proxy; use 'any' amt
    out, dropped = [], 0
    p_treat = sum(1 for r in rows if _f(r.get("tinc"), 0) and _f(r["tinc"]) > 0) / max(1, len(rows))
    p_treat = min(0.95, max(0.05, p_treat))
    for i, r in enumerate(rows):
        a = 1 if (_f(r.get("tinc"), 0) or 0) > 0 else 0     # any incentive vs none (randomized amount)
        y = _f(r.get("got"))                                 # learned result (0/1)
        age = _f(r.get("age"))
        if y is None or age is None:
            dropped += 1
            continue
        out.append({"context": {"age": age, "hiv": _f(r.get("hiv2004"), 0) or 0,
                                 "dist": _f(r.get("distvct"), 0) or 0},
                    "action": a, "reward": y, "propensity": p_treat if a else 1 - p_treat,
                    "cluster": f"vill{r.get('villnum', i)}"})
    return out, dropped


def load_star():
    """Tennessee STAR class-size RCT (kindergarten): small class vs regular; outcome = standardized
    math+reading (normalized to [0,1] within the file). Randomized within school (cluster)."""
    rows = list(_rows(_p("star.csv")))
    ms = [_f(r["tmathssk"]) for r in rows if _f(r["tmathssk"]) is not None]
    rs = [_f(r["treadssk"]) for r in rows if _f(r["treadssk"]) is not None]
    lo, hi = min(ms + rs), max(ms + rs)
    out, dropped = [], 0
    p_small = sum(1 for r in rows if r.get("classk") == "small.class") / max(1, len(rows))
    for i, r in enumerate(rows):
        m, rd = _f(r.get("tmathssk")), _f(r.get("treadssk"))
        if m is None or rd is None or not r.get("classk"):
            dropped += 1
            continue
        a = 1 if r["classk"] == "small.class" else 0        # small vs (regular / regular+aide)
        y = ((m + rd) / 2 - lo) / (hi - lo)
        out.append({"context": {"free_lunch": 1.0 if r.get("freelunk") == "yes" else 0.0,
                                 "girl": 1.0 if r.get("sex") == "girl" else 0.0,
                                 "exp": _f(r.get("totexpk"), 0) or 0},
                    "action": a, "reward": y, "propensity": p_small if a else 1 - p_small,
                    "cluster": f"sch{r.get('schidkn', i)}"})
    return out, dropped


def load_jobs():
    """JOBS II job-search RCT (Vinokur et al.): randomized job-search seminar; outcome = employed
    (work1). Randomized -> known assignment share."""
    rows = list(_rows(_p("jobs.csv")))
    out, dropped = [], 0
    p = sum(1 for r in rows if _f(r["treat"]) == 1) / max(1, len(rows))
    for i, r in enumerate(rows):
        a = int(_f(r.get("treat"), 0) or 0)
        w = r.get("work1")                                   # 'psyemp' (employed) / 'psyump' (unemployed)
        y = 1.0 if w == "psyemp" else (0.0 if w == "psyump" else None)
        if y is None:
            dropped += 1
            continue
        out.append({"context": {"age": _f(r.get("age"), 0) or 0, "econ_hard": _f(r.get("econ_hard"), 0) or 0,
                                 "depress": _f(r.get("depress1"), 0) or 0,
                                 "nonwhite": _f(r.get("nonwhite"), 0) or 0,
                                 "educ": _f(r.get("educ"), 0) or 0},
                    "action": a, "reward": y, "propensity": p if a else 1 - p, "cluster": f"jobs{i % 30}"})
    return out, dropped


def load_nsw():
    """NSW (LaLonde/Dehejia-Wahba) randomized job-training experiment: treat vs control; outcome =
    employed in 1978 (re78 > 0). Randomized (the experimental sample)."""
    rows = list(_rows(_p("nsw_mixtape.csv")))
    out, dropped = [], 0
    p = sum(1 for r in rows if _f(r["treat"]) == 1) / max(1, len(rows))
    for i, r in enumerate(rows):
        a = int(_f(r.get("treat"), 0) or 0)
        re78 = _f(r.get("re78"))
        if re78 is None:
            dropped += 1
            continue
        out.append({"context": {"age": _f(r.get("age"), 0) or 0, "educ": _f(r.get("educ"), 0) or 0,
                                 "black": _f(r.get("black"), 0) or 0, "married": _f(r.get("marr"), 0) or 0,
                                 "re75": _f(r.get("re75"), 0) or 0},
                    "action": a, "reward": 1.0 if re78 > 0 else 0.0,
                    "propensity": p if a else 1 - p, "cluster": f"nsw{i % 20}"})
    return out, dropped


# ============================================================ NETWORK / peer-effects RCT
def load_social_insure():
    """Cai, de Janvry & Sadoulet (2015) weather-insurance RCT with a randomized two-stage
    INFORMATION-SESSION design: intensive vs simple session; outcome = insurance take-up. Peer/network
    effects operate through village adoption (pre_takeup_rate is the network-exposure feature)."""
    rows = list(_rows(_p("social_insure.csv")))
    out, dropped = [], 0
    p = sum(1 for r in rows if _f(r["intensive"]) == 1) / max(1, len(rows))
    for i, r in enumerate(rows):
        a = int(_f(r.get("intensive"), 0) or 0)
        y = _f(r.get("takeup_survey"))
        if y is None:
            dropped += 1
            continue
        out.append({"context": {"age": _f(r.get("age"), 0) or 0, "male": _f(r.get("male"), 0) or 0,
                                 "risk_averse": _f(r.get("risk_averse"), 0) or 0,
                                 "literacy": _f(r.get("literacy"), 0) or 0,
                                 "network_exposure": _f(r.get("pre_takeup_rate"), 0) or 0,
                                 "rice_area": _f(r.get("ricearea_2010"), 0) or 0},
                    "action": a, "reward": y, "propensity": p if a else 1 - p,
                    "cluster": f"vill_{r.get('village', i)}"})     # village = interference cluster
    return out, dropped


# ============================================================ MEDIA A/B (logged bandit, in-repo)
def load_upworthy(min_arms=2, max_tests=None):
    """Upworthy Research Archive: randomized headline A/B tests (real clicks). Each test is a K-armed
    LOGGED BANDIT — context = test, arms = headline packages, reward = CTR, propensity = 1/K (uniform
    randomization). Returns a list of TESTS (each a dict with arms), not flat rows."""
    import json
    path = os.path.join(RAW, "..", "..", "experiments", "results", "exp054_upworthy",
                        "upworthy_parsed.json")
    if not os.path.exists(path):
        return [], 0
    data = json.load(open(path))
    tests = data if isinstance(data, list) else data.get("tests", [])
    out, dropped = [], 0
    for t in tests:
        arms = t.get("arms", [])
        arms = [a for a in arms if _f(a.get("impressions"), 0) and a["impressions"] >= 1000]
        if len(arms) < min_arms:
            dropped += 1
            continue
        out.append({"test_id": t.get("test_id"), "arms": [
            {"headline": a.get("headline", "")[:120], "ctr": _f(a.get("ctr")) or
             (_f(a["clicks"]) / _f(a["impressions"]) if _f(a.get("impressions")) else 0.0),
             "impressions": int(_f(a.get("impressions"), 0) or 0)} for a in arms]})
        if max_tests and len(out) >= max_tests:
            break
    return out, dropped


# ============================================================ QUASI-EXPERIMENTAL
def load_kielmc_did():
    """Kiel & McClain (1995) garbage-incinerator natural experiment (DiD): homes near the incinerator
    (nearinc) before/after (y81) construction; outcome = real house price. Design = difference-in-
    differences (propensity None — identified by parallel trends, not weighting)."""
    rows = list(_rows(_p("kielmc.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        near, post, rprice = _f(r.get("nearinc")), _f(r.get("y81")), _f(r.get("rprice"))
        if None in (near, post, rprice):
            dropped += 1
            continue
        out.append({"context": {"near": near, "post": post, "rooms": _f(r.get("rooms"), 0) or 0,
                                 "area": _f(r.get("area"), 0) or 0, "age": _f(r.get("age"), 0) or 0},
                    "action": int(near * post), "reward": rprice, "propensity": None,
                    "did_group": near, "did_time": post, "cluster": f"kiel{int(near)}{int(post)}"})
    return out, dropped


def load_gov_transfers_rd():
    """Gov-transfers regression discontinuity: eligibility by a centered income cutoff (0); outcome =
    program participation/support. Design = sharp RD (running variable Income_Centered)."""
    rows = list(_rows(_p("gov_transfers.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        run, y = _f(r.get("Income_Centered")), _f(r.get("Support"))
        if run is None or y is None:
            dropped += 1
            continue
        out.append({"context": {"running": run, "age": _f(r.get("Age"), 0) or 0,
                                 "educ": _f(r.get("Education"), 0) or 0},
                    "action": 1 if run < 0 else 0, "reward": y, "propensity": None,
                    "running": run, "cluster": f"gt{i % 40}"})
    return out, dropped


def load_close_elections_rd():
    """Lee (2008)-style close-elections RD: Democrat win at the 50% vote-share threshold; outcome =
    next-period Democrat vote share. Design = sharp RD (running variable = demvoteshare-0.5)."""
    rows = list(_rows(_p("close_elections_lmb.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        share, y = _f(r.get("demvoteshare")), _f(r.get("score"))
        if share is None or y is None:
            dropped += 1
            continue
        run = share - 0.5
        out.append({"context": {"running": run, "year": _f(r.get("year"), 0) or 0},
                    "action": int(_f(r.get("democrat"), 0) or 0), "reward": y, "propensity": None,
                    "running": run, "cluster": f"st{r.get('state', i)}"})
    return out, dropped


def load_close_college_iv():
    """Card (1995) proximity-to-college IV: nearc4 (grew up near a 4-year college) instruments for
    education; outcome = log wage. Design = instrumental variables."""
    rows = list(_rows(_p("close_college.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        z, educ, y = _f(r.get("nearc4")), _f(r.get("educ")), _f(r.get("lwage"))
        if None in (z, educ, y):
            dropped += 1
            continue
        out.append({"context": {"black": _f(r.get("black"), 0) or 0, "smsa": _f(r.get("smsa"), 0) or 0,
                                 "south": _f(r.get("south"), 0) or 0, "exper": _f(r.get("exper"), 0) or 0},
                    "action": 1 if educ >= 13 else 0, "reward": y, "propensity": None,
                    "instrument": z, "treatment_cont": educ, "cluster": f"cc{i % 40}"})
    return out, dropped


def load_jtrain_did():
    """Job-training grants panel (Holzer et al. via Wooldridge) — DiD over firm-years 1987-89: grant
    receipt; outcome = scrap rate (lower is better -> reward = -log scrap). Panel supports sequential
    tasks too."""
    rows = list(_rows(_p("jtrain.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        grant, year, scrap = _f(r.get("grant")), _f(r.get("year")), _f(r.get("lscrap"))
        if None in (grant, year) or scrap is None:
            dropped += 1
            continue
        out.append({"context": {"year": year, "employ": _f(r.get("employ"), 0) or 0,
                                 "sales": _f(r.get("lsales"), 0) or 0},
                    "action": int(grant), "reward": -scrap, "propensity": None,
                    "firm": r.get("fcode", i), "year": year, "cluster": f"firm{r.get('fcode', i)}"})
    return out, dropped


def load_organ_donations_did():
    """Organ-donations DiD (Kessler & Roth via causaldata): California's active-choice policy change;
    outcome = donor registration rate. Design = DiD (treated state vs rest, before/after Q3-2011)."""
    rows = list(_rows(_p("organ_donations.csv")))
    out, dropped = [], 0
    for i, r in enumerate(rows):
        state, q, rate = r.get("State"), r.get("Quarter"), _f(r.get("Rate"))
        if not state or not q or rate is None:
            dropped += 1
            continue
        treated = 1 if state == "California" else 0
        post = 1 if q in ("Q32011", "Q42011", "Q12012") else 0
        out.append({"context": {"treated": treated, "post": post}, "action": int(treated * post),
                    "reward": rate, "propensity": None, "did_group": treated, "did_time": post,
                    "cluster": f"state_{state}"})
    return out, dropped


# ============================================================ SEQUENTIAL panels
def load_castle_panel():
    """Castle-doctrine DiD panel (Cheng & Hoekstra) — state-year panel 2000-2010; treatment = enacting
    a stand-your-ground law; outcome = log homicide rate. Real longitudinal treatment adoption ->
    sequential decision environments (one sequence per state over years)."""
    rows = list(_rows(_p("castle.csv")))
    by_state = {}
    dropped = 0
    for r in rows:
        sid, year = r.get("sid"), _f(r.get("year"))
        y = _f(r.get("l_homicide"))
        if y is None:
            y = _f(r.get("homicide"))
            y = (None if y is None else __import__("math").log(max(1e-6, y)))
        post = _f(r.get("post"))
        if sid is None or year is None or y is None:
            dropped += 1
            continue
        by_state.setdefault(sid, []).append({"year": year, "action": int(post or 0), "reward": -y,
                                             "cluster": f"cstate{sid}"})
    seqs = []
    for sid, steps in by_state.items():
        steps.sort(key=lambda s: s["year"])
        seq = {"steps": [{"context": {"year": s["year"], "t": i}, "action": s["action"],
                          "reward": s["reward"], "propensity": 0.5} for i, s in enumerate(steps)],
               "cluster": f"cstate{sid}"}
        seqs.append(seq)
    return seqs, dropped


# ============================================================ CMV persuasion (in-repo, observational)
def load_cmv(max_pairs=1200):
    """ChangeMyView persuasion (ConvoKit winning-args): matched pairs of arguments to the same OP;
    outcome = earned a delta. Observational matched-pair -> heterogeneous-effect / policy-targeting."""
    from swm.decision.outcome_import import import_convokit_cmv, to_samples
    path = os.path.join(RAW, "winning-args-corpus", "utterances.jsonl")
    if not os.path.exists(path):
        return [], 0
    labeled = import_convokit_cmv(path, max_pairs=max_pairs)
    return labeled, 0


# ============================================================ cards + registry
CARDS = {
    "thornton_hiv": {"unit": "individual", "treatment": "cash incentive to learn HIV result",
                     "outcome": "learned result (0/1)", "assignment": "randomized", "design": "rct",
                     "domain": "health", "propensity": "design (treated share)",
                     "source": "Thornton 2008 AER; causaldata", "license": "public research corpus"},
    "star": {"unit": "student", "treatment": "small class (vs regular)",
             "outcome": "normalized math+reading", "assignment": "randomized within school",
             "design": "rct", "domain": "education", "propensity": "design share",
             "source": "Tennessee STAR; Ecdat", "license": "public"},
    "jobs": {"unit": "individual", "treatment": "job-search seminar", "outcome": "employed",
             "assignment": "randomized", "design": "rct", "domain": "labor",
             "propensity": "design share", "source": "JOBS II (Vinokur); mediation pkg",
             "license": "public"},
    "nsw": {"unit": "individual", "treatment": "job training", "outcome": "employed 1978",
            "assignment": "randomized", "design": "rct", "domain": "labor",
            "propensity": "design share", "source": "NSW LaLonde/Dehejia-Wahba", "license": "public"},
    "social_insure": {"unit": "farmer", "treatment": "intensive info session",
                      "outcome": "insurance take-up", "assignment": "randomized two-stage",
                      "design": "rct_network", "domain": "development/agriculture",
                      "propensity": "design share", "interference": "village peer effects",
                      "source": "Cai-deJanvry-Sadoulet 2015 AEJ", "license": "public"},
    "upworthy": {"unit": "headline test", "treatment": "headline package (K-arm)",
                 "outcome": "click-through rate", "assignment": "randomized (uniform arms)",
                 "design": "logged_bandit", "domain": "media", "propensity": "1/K known",
                 "source": "Upworthy Research Archive (Matias 2021)", "license": "CC-BY 4.0"},
    "kielmc": {"unit": "house-sale", "treatment": "near incinerator × post-construction",
               "outcome": "real price", "assignment": "natural experiment", "design": "did",
               "domain": "housing/environment", "propensity": None,
               "source": "Kiel-McClain 1995; wooldridge", "license": "public"},
    "gov_transfers": {"unit": "household", "treatment": "transfer eligibility (income cutoff)",
                      "outcome": "participation/support", "assignment": "cutoff", "design": "rd",
                      "domain": "welfare/politics", "propensity": None,
                      "source": "causaldata gov_transfers", "license": "public"},
    "close_elections": {"unit": "district-year", "treatment": "Democrat win (50% threshold)",
                        "outcome": "next Dem vote share", "assignment": "close-race cutoff",
                        "design": "rd", "domain": "politics", "propensity": None,
                        "source": "Lee/LMB close_elections", "license": "public"},
    "close_college": {"unit": "individual", "treatment": "college (educ>=13)",
                      "outcome": "log wage", "assignment": "proximity instrument", "design": "iv",
                      "domain": "labor/education", "propensity": None,
                      "source": "Card 1995 proximity IV", "license": "public"},
    "jtrain": {"unit": "firm-year", "treatment": "training grant", "outcome": "-log scrap rate",
               "assignment": "grant panel", "design": "did", "domain": "labor",
               "propensity": None, "source": "Holzer/Wooldridge jtrain", "license": "public"},
    "organ_donations": {"unit": "state-quarter", "treatment": "active-choice policy",
                        "outcome": "donor registration rate", "assignment": "policy change",
                        "design": "did", "domain": "public-policy/health", "propensity": None,
                        "source": "Kessler-Roth; causaldata", "license": "public"},
    "castle": {"unit": "state-year", "treatment": "stand-your-ground enactment",
               "outcome": "-log homicide", "assignment": "staggered adoption", "design": "sequential_did",
               "domain": "public-policy/crime", "propensity": None,
               "source": "Cheng-Hoekstra; causaldata", "license": "public"},
    "cmv": {"unit": "argument", "treatment": "argument features", "outcome": "earned delta",
            "assignment": "observational matched-pair", "design": "matched_observational",
            "domain": "persuasion", "propensity": None,
            "source": "ConvoKit winning-args (Tan 2016)", "license": "public research corpus"},
}

FILES = {"thornton_hiv": "thornton_hiv.csv", "star": "star.csv", "jobs": "jobs.csv",
         "nsw": "nsw_mixtape.csv", "social_insure": "social_insure.csv",
         "kielmc": "kielmc.csv", "gov_transfers": "gov_transfers.csv",
         "close_elections": "close_elections_lmb.csv", "close_college": "close_college.csv",
         "jtrain": "jtrain.csv", "organ_donations": "organ_donations.csv", "castle": "castle.csv"}

LOADERS = {"thornton_hiv": load_thornton_hiv, "star": load_star, "jobs": load_jobs, "nsw": load_nsw,
           "social_insure": load_social_insure, "kielmc": load_kielmc_did,
           "gov_transfers": load_gov_transfers_rd, "close_elections": load_close_elections_rd,
           "close_college": load_close_college_iv, "jtrain": load_jtrain_did,
           "organ_donations": load_organ_donations_did}


def dataset_card(name):
    c = dict(CARDS.get(name, {}))
    f = FILES.get(name)
    if f:
        c["file"] = f
        c["sha16"] = sha16(_p(f))
    return c


if __name__ == "__main__":
    import json
    print("dataset row counts (real intervention data):")
    for name, loader in LOADERS.items():
        try:
            rows, dropped = loader()
            print(f"  {name:18} rows={len(rows):>6} dropped={dropped:<4} "
                  f"design={CARDS[name]['design']:<18} domain={CARDS[name]['domain']}")
        except Exception as e:
            print(f"  {name:18} ERROR {type(e).__name__}: {e}")
    up, upd = load_upworthy(max_tests=200)
    print(f"  upworthy(tests)    tests={len(up):>6} dropped={upd}")
    cmv, _ = load_cmv(max_pairs=400)
    print(f"  cmv(labeled)       msgs ={len(cmv):>6}")
    castle, cd = load_castle_panel()
    print(f"  castle(sequences)  seqs ={len(castle):>6} dropped={cd}")
