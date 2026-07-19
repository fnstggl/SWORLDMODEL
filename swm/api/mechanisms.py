"""Mechanism-specific simulators — the general-world-model way to specialize WITHOUT fragmenting into topics.

A naive system builds one simulator per SUBJECT (an election simulator, a sports simulator, a legislation
simulator). That does not generalize: the world has unbounded subjects. But it has a SMALL number of
GENERATIVE MECHANISMS — the abstract processes by which a yes/no outcome is actually produced. Elections,
referendums, shareholder votes and approval ratings are all the SAME mechanism (aggregate many units' choices,
threshold at a majority). Sports, court cases and races are all a CONTEST (two sides, one wins). A launch, a
death, a first-occurrence and a record-break are all an ARRIVAL (a rare event with a rate, before a deadline).

So we specialize on the seven mechanisms, not the subjects — each a tiny parametric Monte-Carlo simulator on
the SHARED substrate (base-rate anchor + honest parameter uncertainty + real elapsed horizon). A general router
(the LLM names the mechanism and supplies its grounded parameters; the *simulation* produces the probability)
maps ANY question to its mechanism. This covers essentially every binary social question with one honest engine
whose transition KERNEL matches how that class of outcome is generated:

  aggregation   vote/approval/referendum   threshold on a share      grounded by: polls / current share
  contest       sports/court/race/duel     head-to-head win          grounded by: ratings / odds
  diffusion     price/index/count/%         a number crosses a level  grounded by: current value + volatility
  arrival       launch/death/record/first  a rare event by a deadline grounded by: base rate over a horizon
  whipcount     legislation/treaty/merger  committed votes vs needed  grounded by: vote counts / positions
  escalation    war/bank-run/viral/unrest  a self-reinforcing tip     grounded by: GDELT social pressure
  persistence   incumbency/status-quo/ontime the default holds        grounded by: disruption hazard

Every simulator integrates its parameters' uncertainty (never asserts zero), regresses toward the base-rate
anchor when evidence is weak, and respects the real horizon — the same honesty guarantees as `latent_forecast`,
of which the `diffusion` and generic-`event` branches are two special cases.
"""
from __future__ import annotations

import math
import random

SECONDS_PER_YEAR = 365.25 * 86400


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _clip01(p):
    return min(0.999, max(0.001, p))


# --------------------------------------------------------------------------- the seven mechanism simulators

PROVENANCE_SD_FLOOR = {"grounded": 0.0, "quoted": 0.08, "invented": 0.15}
PROVENANCE_ANCHOR_W = {"grounded": 1.0, "quoted": 0.75, "invented": 0.55}


def _anchor(p, base_rate, provenance):
    """EXP-101 fix: shrink toward the outside-view anchor in log-odds, by parameter PROVENANCE. A kernel fed
    measured params keeps its full signal (w=1); one fed numbers the LLM merely read (quoted) or made up
    (invented) may not assert near-certainty — the uncertainty that matters is whether the params are even
    right, which no within-kernel sampling can see."""
    w = PROVENANCE_ANCHOR_W.get(provenance, 1.0)
    return _clip01(_sig(w * _logit(p) + (1 - w) * _logit(base_rate)))


def sim_aggregation(share, *, threshold=0.5, share_sd=None, direction=">", base_rate=0.5, n=4000, seed=0,
                    provenance="grounded"):
    """VOTE / APPROVAL / REFERENDUM: YES iff an aggregate SHARE beats a threshold (usually 0.5). The share is a
    latent Normal centered on the grounded current/poll share with poll-error sd (empirical ~0.03-0.06; wider
    when ungrounded). Integrates the poll error — a 52% lead with 4pt error is ~0.7, not 1.0. Ungrounded
    provenance widens the sd (the share ESTIMATE has error on top of poll error) and anchors to the base rate."""
    if share is None:
        return base_rate
    rng = random.Random(seed)
    sd = max(share_sd if share_sd is not None else 0.06, PROVENANCE_SD_FLOOR.get(provenance, 0.0))
    yes = 0
    for _ in range(n):
        s = rng.gauss(share, sd)
        yes += 1 if (s > threshold if direction == ">" else s < threshold) else 0
    return _anchor(yes / n, base_rate, provenance)


def sim_contest(*, win_prob=None, rating_diff=None, rating_sd=120.0, base_rate=0.5, n=4000, seed=0):
    """SPORTS / COURT / RACE: a head-to-head; YES = the tracked side wins. Either a grounded win_prob (from
    odds) or an Elo-style rating_diff (logistic, 400-scale) with rating uncertainty integrated. No data ⇒ base
    rate (often 0.5 for a fair contest)."""
    if win_prob is not None:
        return _clip01(win_prob)
    if rating_diff is None:
        return base_rate
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(n):
        d = rating_diff + rng.gauss(0, rating_sd)
        acc += 1.0 / (1.0 + 10 ** (-d / 400.0))
    return _clip01(acc / n)


def sim_arrival(*, base_rate=None, rate_per_year=None, horizon_years=1.0, ref_years=1.0, n=4000, seed=0):
    """ARRIVAL / HAZARD: a rare event with a Poisson rate, before the deadline. P(by H) = 1 - exp(-λ·H). λ is
    grounded either directly (rate_per_year) or backed out of a base rate observed over ref_years:
    λ = -ln(1-base_rate)/ref_years. Rate uncertainty (log-normal) is integrated. This is the honest form for
    'will X happen by <date>' — a launch, a first, a record, a death — where longer horizon ⇒ higher P."""
    hy = max(1e-4, horizon_years)
    if rate_per_year is None:
        if base_rate is None:
            return 0.5
        br = min(0.98, max(0.02, base_rate))
        rate_per_year = -math.log(1 - br) / max(1e-3, ref_years)
    if rate_per_year <= 0:
        return 0.02
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(n):
        lam = rate_per_year * math.exp(rng.gauss(0, 0.5))     # honest rate uncertainty (log-normal, ~±65%)
        acc += 1 - math.exp(-lam * hy)
    return _clip01(acc / n)


def sim_whipcount(*, committed_yes=0, committed_no=0, undecided=0, needed=None, total=None, lean=0.5,
                  n=4000, seed=0, base_rate=0.5, provenance="grounded"):
    """LEGISLATION / TREATY / MERGER / BOARD VOTE: YES iff committed-yes plus the undecided that break yes reach
    the needed threshold. Each undecided breaks YES with probability `lean` (grounded from party/whip signal);
    the count is Binomial, integrated. `needed` defaults to a bare majority of `total`.
    With GROUNDED counts the arithmetic gates are facts (have the votes ⇒ ~done). With quoted/invented counts
    the gates are conjecture: the counts themselves get sampling noise, `lean` gets estimate error, and the
    result anchors to the base rate — an invented whip count may not produce 0.02/0.98 (EXP-101)."""
    if needed is None:
        if total is None:
            return base_rate
        needed = total / 2.0 + 0.5
    rng = random.Random(seed)
    lean = min(1.0, max(0.0, lean))
    if provenance == "grounded":
        if committed_yes >= needed:
            return 0.98
        if committed_yes + undecided < needed:
            return 0.02
        yes = 0
        for _ in range(n):
            breaks = sum(1 for _ in range(int(undecided)) if rng.random() < lean)
            yes += 1 if committed_yes + breaks >= needed else 0
        return _clip01(yes / n)
    scale = max(2.0, 0.15 * float(total if total is not None else max(needed * 2, committed_yes + undecided)))
    yes = 0
    for _ in range(n):
        cy = max(0.0, rng.gauss(float(committed_yes), scale))
        und = max(0.0, rng.gauss(float(undecided), scale))
        ln = min(1.0, max(0.0, rng.gauss(lean, 0.15)))
        breaks = sum(1 for _ in range(int(round(und))) if rng.random() < ln)
        yes += 1 if cy + breaks >= needed else 0
    return _anchor(yes / n, base_rate, provenance)


def sim_escalation(*, base_rate=0.5, pressure=0.0, trend=0.0, reinforce=0.4, horizon_years=1.0,
                   push=1.0, n=4000, seed=0):
    """ESCALATION / CONTAGION / TIPPING: a self-reinforcing process (war escalation, bank run, viral spread,
    protest wave). Latent log-odds start at the base rate and are driven by measured PRESSURE (e.g. GDELT
    conflict/violence) and its TREND, amplified by a reinforcement gain and by the horizon (more time ⇒ more
    chance to tip), signed by `push` (+1 if pressure drives YES, -1 if it drives NO). Honest Gaussian
    uncertainty on the drift keeps it from the extremes."""
    hy = max(1e-4, horizon_years)
    sign = 1.0 if push >= 0 else -1.0
    drift = sign * (pressure + reinforce * trend) * (1 + math.tanh(hy)) * 1.4
    sd = 0.6 + 0.5 * abs(drift)
    rng = random.Random(seed)
    L0 = 0.75 * _logit(base_rate)
    acc = 0.0
    for _ in range(n):
        acc += _sig(L0 + rng.gauss(drift, sd))
    return _clip01(acc / n)


def sim_persistence(*, base_rate=0.85, disruption_hazard=None, horizon_years=1.0, ref_years=1.0,
                    happens=True, n=3000, seed=0):
    """PERSISTENCE / STATUS-QUO / ON-TIME: the default holds unless a rare disruption arrives (incumbent stays,
    scheduled thing happens on time, regime persists). P(status quo survives H) = exp(-h·H). `happens=True` ⇒
    YES means the default holds; `happens=False` ⇒ YES means the disruption occurs (complement)."""
    hy = max(1e-4, horizon_years)
    if disruption_hazard is None:
        br = min(0.98, max(0.02, base_rate))                  # base_rate = P(holds over ref horizon)
        disruption_hazard = -math.log(br) / max(1e-3, ref_years)
    survive = math.exp(-max(0.0, disruption_hazard) * hy)
    return _clip01(survive if happens else 1 - survive)


# --------------------------------------------------------------------------- router

MECHANISMS = {
    "aggregation": "a yes/no decided by whether a SHARE of many units (voters, shareholders, approvers) beats a "
                   "threshold — elections, referendums, approval ratings, board/shareholder votes",
    "contest": "a HEAD-TO-HEAD where one side wins — sports games, championships, court cases, races, duels, "
               "any 'will X beat Y' / 'will X win'",
    "diffusion": "a measurable NUMBER crossing a level — price, index, rate, %, count, temperature "
                 "(handled by the metric branch)",
    "arrival": "a rare EVENT happening BY a deadline with no strong internal dynamics — a launch, a release, a "
               "record broken, a first occurrence, a death, an announcement",
    "whipcount": "a formal MULTI-PARTY DECISION passing — legislation, treaties, mergers requiring approval, "
                 "confirmations, anything with counted committed votes vs a needed threshold",
    "escalation": "a SELF-REINFORCING social process tipping — war escalation/ceasefire, coups, bank runs, "
                  "viral spread, protest waves, unrest (use the measured social state)",
    "persistence": "the STATUS QUO holding or a scheduled thing happening ON TIME — an incumbent staying, a "
                   "regime persisting, a deal not collapsing, an event not being cancelled",
}


def route_prompt_block():
    """The mechanism menu injected into a compile prompt so the LLM names the generative mechanism."""
    return ("MECHANISM — which generative process produces this outcome? Pick exactly one:\n"
            + "\n".join(f'  - "{k}": {v}' for k, v in MECHANISMS.items()))


def simulate_mechanism(mechanism, params, *, base_rate=0.5, horizon_years=1.0, n=4000, seed=0):
    """Dispatch to the named mechanism simulator with its grounded params. Unknown/absent ⇒ None (caller falls
    back to the generic latent event/metric sim). Every simulator is anchored + honest by construction."""
    p = dict(params or {})
    p.setdefault("base_rate", base_rate)
    p.setdefault("n", n)
    p.setdefault("seed", seed)
    prov = p.get("provenance", "grounded")
    if mechanism in ("aggregation", "vote"):
        return sim_aggregation(p.pop("share", None), threshold=p.get("threshold", 0.5),
                               share_sd=p.get("share_sd"), direction=p.get("direction", ">"),
                               base_rate=p["base_rate"], n=p["n"], seed=p["seed"], provenance=prov)
    if mechanism in ("contest", "match"):
        return sim_contest(win_prob=p.get("win_prob"), rating_diff=p.get("rating_diff"),
                           rating_sd=p.get("rating_sd", 120.0), base_rate=p["base_rate"], n=p["n"], seed=p["seed"])
    if mechanism == "arrival":
        return sim_arrival(base_rate=p.get("base_rate"), rate_per_year=p.get("rate_per_year"),
                           horizon_years=horizon_years, ref_years=p.get("ref_years", 1.0),
                           n=p["n"], seed=p["seed"])
    if mechanism in ("whipcount", "legislation"):
        return sim_whipcount(committed_yes=p.get("committed_yes", 0), committed_no=p.get("committed_no", 0),
                             undecided=p.get("undecided", 0), needed=p.get("needed"), total=p.get("total"),
                             lean=p.get("lean", 0.5), base_rate=p["base_rate"], n=p["n"], seed=p["seed"],
                             provenance=prov)
    if mechanism in ("escalation", "contagion"):
        return sim_escalation(base_rate=p["base_rate"], pressure=p.get("pressure", 0.0),
                              trend=p.get("trend", 0.0), reinforce=p.get("reinforce", 0.4),
                              horizon_years=horizon_years, push=p.get("push", 1.0), n=p["n"], seed=p["seed"])
    if mechanism in ("persistence", "status_quo"):
        return sim_persistence(base_rate=p.get("base_rate", 0.85), disruption_hazard=p.get("disruption_hazard"),
                               horizon_years=horizon_years, ref_years=p.get("ref_years", 1.0),
                               happens=p.get("happens", True), n=p["n"], seed=p["seed"])
    return None


# --------------------------------------------------------------------------- front door: compile → route → sim

def build_mechanism_prompt(question, as_of_ts, horizon_years, social_block=None):
    import datetime as _dt
    date = _dt.datetime.utcfromtimestamp(int(as_of_ts)).strftime("%Y-%m-%d")
    days = int(horizon_years * 365.25)
    sb = (social_block + "\n") if social_block else ""
    return (
        f"You are a careful SUPERFORECASTER assembling the inputs to a SIMULATION — you do NOT state whether it "
        f"happens.\nTODAY IS {date}. It resolves in about {days} days. Use ONLY information available as of "
        f"today.\n\nQUESTION: {question}\n\n{sb}"
        f"{route_prompt_block()}\n\n"
        f"Give JSON: \"base_rate\" (outside-view reference-class rate; 0.5 if a fair coin), \"mechanism\" (one "
        f"label above), and the params for that mechanism:\n"
        f'  aggregation: "share" (current YES vote/approval share 0-1 — ONLY if a poll/count is stated in the '
        f'question text; OMIT it if you would be guessing), "threshold" (default 0.5), "share_sd" (poll error); \n'
        f'  contest: "win_prob" (0-1 if you know the odds) OR "rating_diff" (tracked side minus opponent, Elo '
        f"points);\n"
        f'  diffusion: "current_value","threshold","direction" (">"/"<"),"annual_vol_pct","grounded_conf";\n'
        f'  arrival: "rate_per_year" (expected occurrences/yr) OR rely on base_rate with "ref_years" (the '
        f"horizon that base rate refers to);\n"
        f'  whipcount: "committed_yes","undecided","needed" (or "total"),"lean" (prob an undecided breaks yes) '
        f'— ONLY counts stated in the question text; OMIT counts you would be inventing;\n'
        f'  escalation: "push" (+1 if rising conflict makes YES, -1 if it makes NO) — pressure is measured for '
        f"you;\n"
        f'  persistence: "happens" (true if YES = the status quo/scheduled thing holds) and base_rate = P(it '
        f"holds over ref_years).\n"
        f'Also give "evidence": "quoted" if your key numeric params appear in the question text above, else '
        f'"estimated". Estimates are welcome but are treated as uncertain — never fake a quote.\n'
        f"Return ONLY compact JSON.")


def mechanism_forecast(question, as_of_ts, resolve_ts, llm, *, n=4000, seed=0, metric_grounder=None,
                       social_grounder=None):
    """Compile the mechanism + grounded params (one LLM call), ground live where possible, and route to the
    matching simulator. Falls back to the generic latent event/metric sim when the mechanism is absent or its
    params are missing — so it is never worse than `latent_forecast`, only more specific where it can be."""
    from swm.api.latent_forecast import LatentSpec, parse_latent, simulate_latent
    from swm.api.retrieval_grounding import parse_json_lenient

    hy = max(1e-4, (float(resolve_ts) - float(as_of_ts)) / SECONDS_PER_YEAR)
    social = None
    if social_grounder is not None:
        try:
            social = social_grounder.ground_social(question, as_of_ts)
        except Exception:
            social = None
    r = parse_json_lenient(llm(build_mechanism_prompt(question, as_of_ts, hy,
                                                      social_block=social["block"] if social else None)))
    if not r:
        return None, None
    try:
        base_rate = min(0.98, max(0.02, float(r.get("base_rate", 0.5))))
    except Exception:
        base_rate = 0.5
    mech = str(r.get("mechanism", "")).lower().strip()

    # diffusion reuses the calibrated metric branch (with as-of price grounding) — one substrate, no duplication
    if mech == "diffusion":
        spec = parse_latent(_as_json(r, kind="metric"))
        if spec is not None:
            if metric_grounder is not None:
                try:
                    g = metric_grounder.ground_metric(question, spec.metric_name, as_of_ts)
                    if g and g.get("value") is not None:
                        spec.current_value, spec.grounded_conf = float(g["value"]), "high"
                        if g.get("annual_vol_pct"):
                            spec.annual_vol_pct = float(g["annual_vol_pct"])
                except Exception:
                    pass
            return simulate_latent(spec, hy, n=n, seed=seed), {"mechanism": "diffusion", "raw": r}

    params = dict(r)
    # provenance: LLM-supplied params are at best QUOTED from the as-of question text, never "grounded"
    # (grounded is reserved for measured feeds — polls/whip evidence pipelines, when they exist)
    params["provenance"] = "quoted" if str(r.get("evidence", "")).lower() == "quoted" else "invented"
    if mech in ("escalation", "contagion") and social is not None:      # inject MEASURED conflict pressure
        d = social["driver"]
        params["pressure"] = max(0.0, -d["goldstein"] / 5.0) + 6.0 * d["violence_rate"]
        params["trend"] = d["escalation_trend"]
        params.setdefault("push", 1.0)
        params["_country"] = social["country"]
    p = simulate_mechanism(mech, params, base_rate=base_rate, horizon_years=hy, n=n, seed=seed)
    if p is not None:
        return p, {"mechanism": mech, "raw": r, "_social": social["driver"] if social else None}

    # fallback: generic latent event sim (honest anchor + drivers), never worse than the base forecaster
    spec = parse_latent(_as_json(r, kind="event"))
    if spec is None:
        return base_rate, {"mechanism": "base_rate", "raw": r}
    return simulate_latent(spec, hy, n=n, seed=seed), {"mechanism": "fallback_event", "raw": r}


def _as_json(r, *, kind):
    import json
    return json.dumps({**r, "kind": kind})
