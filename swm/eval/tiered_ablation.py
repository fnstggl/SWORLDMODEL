"""PART B — the practical, tiered ablation harness that isolates the MARGINAL value of each layer.

The pilot's flaw (see docs/AUDIT_PART_A_WIRING.md): its "FULL" arm was a role-prompted observer-panel
ENSEMBLE, not a simulation, and it had no call-matched ensemble control — so a +0.0095 Brier edge over one
grounded call could not be attributed to anything. This harness fixes that with a tiered ladder, run at three
densities so evaluation never becomes so expensive it blocks iteration (final hard rule #19):

  TIER 1 — EVERY item (cheap, mandatory):
    B0 base_rate     — question-class / sample base rate (the free, well-calibrated skeptic)
    B1 grounded_1shot— ONE grounded call on the frozen dossier (the bar simulation must clear)
    B2 full          — current production routing + calibration + abstention (native output)

  TIER 2 — stratified ~20% sample (the arm that isolates SIMULATION from AVERAGING):
    B3 grounded_ens  — N grounded DIRECT forecasts (no roles, no interaction), pooled like the panel,
                       CALL-MATCHED to B2. B3−B1 = ensembling value; B4−B3 = lens/anchor value.

  TIER 3 — diagnostic ~5-10% sample / after architecture changes:
    B4 generic_panel — the current ObserverPanel (forecaster lenses, no stakeholders, no interaction)
    B5 indep_stake   — real cast stakeholders decide INDEPENDENTLY (SocietyRollout, 1 round, no signal)
    B6 interact_stake— same cast, INTERACTING over dated rounds (SocietyRollout, ≥2 rounds, public signal)
    B7 persistent    — persistent-state agents  [NOT YET BUILT — returns None w/ reason]
    B8 time_evolving — real-event-opportunity rollout  [NOT YET BUILT — returns None w/ reason]
    B9 parametric    — the parametric mechanism kernel (leak-free, ground=False)
    B10 experimental — graph / OASIS / TRIBE  [NOT YET BUILT — returns None w/ reason]

Every arm shares ONE frozen dossier (grounded once, same as-of). Every arm's spend (calls/tokens/cost/
latency) and the evidence/prompt/commit/model hashes are recorded, so a claim of simulation superiority can be
checked against the strongest *fair, compute-matched* alternative (hard rule #4). Marginal effects and paired
bootstrap CIs are computed in `report_marginals`.
"""
from __future__ import annotations

from swm.engine.calibrate import pool_distribution
from swm.engine.grounding import SceneGrounder, parse_json
from swm.eval.ablation import _SINGLE_PROMPT, score_arms
from swm.eval.instrument import CountingLLM, Meter, code_commit, evidence_hash, model_version

TIER1 = ("base_rate", "grounded_1shot", "full")
TIER2 = ("grounded_ens",)
TIER3 = ("generic_panel", "indep_stake", "interact_stake", "persistent", "time_evolving",
         "parametric", "experimental")
ALL_ARMS = TIER1 + TIER2 + TIER3

# the ladder of marginal effects (arm_hi, arm_lo, what the difference isolates)
MARGINALS = [
    ("grounded_1shot", "base_rate", "grounding value (grounded 1-shot − base rate)"),
    ("grounded_ens", "grounded_1shot", "ensembling value (call-matched ensemble − 1-shot)"),
    ("generic_panel", "grounded_ens", "forecaster-lens value (panel − plain ensemble)"),
    ("indep_stake", "generic_panel", "stakeholder-modeling value (stakeholders − forecaster lenses)"),
    ("interact_stake", "indep_stake", "interaction value (interacting − independent stakeholders)"),
    ("full", "grounded_1shot", "WHOLE-STACK value (production − grounded 1-shot)"),
]


def stratify_sample(items, key_fn, frac, seed=7):
    """Pick ~frac of items, spread across strata (question class/domain/horizon/outcome), so the Tier-2/3
    sample is representative, not clustered. Deterministic given seed. Returns a set of indices."""
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    strata = defaultdict(list)
    for i, it in enumerate(items):
        strata[key_fn(it)].append(i)
    picked = set()
    for _, idxs in strata.items():
        rng.shuffle(idxs)
        take = max(1, round(len(idxs) * frac)) if idxs else 0
        picked.update(idxs[:take])
    return picked


def _p_single(llm, question, today, dossier=None):
    ev = f"GROUNDED AS-OF EVIDENCE:\n{dossier.brief()}\n" if dossier is not None else \
        "(no evidence provided — reason from general knowledge only)\n"
    r = parse_json(llm(_SINGLE_PROMPT.format(q=question, today=today, evidence_block=ev)))
    if not r:
        return None
    try:
        return min(1.0, max(0.0, float(r["p"])))
    except (KeyError, TypeError, ValueError):
        return None


def _grounded_ensemble(llm, question, today, dossier, n):
    """B3 — N independent GROUNDED DIRECT forecasts (identical to B1's prompt), log-linear pooled the same way
    the panel pools. No forecaster lenses, no base-rate anchor: the pure 'does averaging N calls explain the
    panel's edge?' control. Call-matched to B2 via n."""
    ps = [p for p in (_p_single(llm, question, today, dossier) for _ in range(max(1, n))) if p is not None]
    if not ps:
        return None
    return pool_distribution([{"yes": p, "no": 1 - p} for p in ps])["yes"]


def _society_p(llm_cold, llm_hot, question, dossier, today, *, max_rounds, branches):
    """Run the REAL society simulation for a binary question and return P(yes). Used by B5 (independent,
    1 round) and B6 (interacting, ≥2 rounds). Casts the actual stakeholders from the dossier."""
    from swm.engine.casting import CastingDirector
    from swm.engine.society import SocietyRollout
    cast = CastingDirector(llm_cold).cast(question, dossier.brief(), today=today)
    cast.answer_space = {"type": "binary", "options": ["yes", "no"]}
    res = SocietyRollout(llm_hot, llm=llm_cold, branches=branches, max_rounds=max_rounds).run(
        question, cast, dossier, today=today)
    if not res.distribution:
        return None
    return res.distribution.get("yes")


def predict_arms(wm, question, *, as_of, class_rate, tier=1, search_fn=None, llm_raw=None,
                 llm_hot_raw=None):
    """Produce one metered prediction per arm (up to `tier`) on ONE question, sharing a single frozen dossier.
    Returns {arm: {"p": float|None, "spend": {...}, "note": str}} plus a "_meta" block (hashes/commit/model).
    `tier` ∈ {1,2,3}. `wm` is the production AgentWorldModel; `llm_raw`/`llm_hot_raw` are plain chat fns."""
    from swm.engine.front_door import parametric_binary_p
    from swm.eval.grade_agent_engine import p_yes
    today = as_of or ""

    # ---- ground ONCE; every evidence arm sees exactly this dossier ----
    ground_meter = Meter()
    grounder = SceneGrounder(CountingLLM(wm.llm, ground_meter), search_fn=search_fn, today=today)
    dossier = grounder.ground(question)
    ev_hash = evidence_hash(dossier.brief())
    out = {"_meta": {"evidence_hash": ev_hash, "commit": code_commit(), "model": model_version(),
                     "as_of": as_of, "grounding_spend": ground_meter.snapshot(),
                     "abstain": bool(dossier.abstain)}}

    def metered(arm, fn):
        m = Meter()
        p, note = None, ""
        try:
            p = fn(m)
        except Exception as e:                                  # an arm that dies abstains; never crashes the row
            note = f"error:{str(e)[:60]}"
        out[arm] = {"p": (None if p is None else float(p)), "spend": m.snapshot(), "note": note}

    # ---------- TIER 1 (every item) ----------
    out["base_rate"] = {"p": class_rate, "spend": Meter().snapshot(), "note": "class base rate"}
    metered("grounded_1shot",
            lambda m: (None if dossier.abstain else _p_single(CountingLLM(llm_raw, m), question, today, dossier)))

    def _full(m):
        # B2 uses production routing; it re-grounds internally via search_fn (same as-of) — that spend is the
        # full arm's real cost, so we meter the whole call by wrapping both llms for the duration.
        res = wm.simulate(question, as_of=as_of, binary=True, search_fn=search_fn)
        out["full_calls_reported"] = res.get("n_llm_calls")
        return None if res.get("abstain") else p_yes(res)
    metered("full", _full)
    # the production path isn't wrapped in the meter (it manages its own llms); record its REPORTED call count
    # so call-parity vs B3 is honest (tokens/cost for the production path are not instrumented here).
    if out.get("full_calls_reported"):
        out["full"]["spend"]["calls"] = int(out["full_calls_reported"])
    if tier <= 1:
        return out

    # ---------- TIER 2 (stratified ~20%) ----------
    # call-match B3 to B2's reported panel calls (fallback 10), clamped to a sane band
    n_match = out.get("full_calls_reported") or (wm.panel_reps * 5)
    n_match = max(4, min(16, int(n_match)))
    out["_meta"]["ensemble_n"] = n_match
    metered("grounded_ens",
            lambda m: (None if dossier.abstain else
                       _grounded_ensemble(CountingLLM(llm_raw, m), question, today, dossier, n_match)))
    if tier <= 2:
        return out

    # ---------- TIER 3 (diagnostic ~5-10%) ----------
    def _panel(m):
        from swm.engine.observer_panel import ObserverPanel
        pf = ObserverPanel(CountingLLM(wm.llm_hot, m), reps_per_lens=wm.panel_reps).forecast(
            question, dossier, today=today)
        return pf.p_event
    metered("generic_panel", lambda m: (None if dossier.abstain else _panel(m)))

    metered("indep_stake", lambda m: (None if dossier.abstain else _society_p(
        CountingLLM(wm.llm, m), CountingLLM(wm.llm_hot, m), question, dossier, today,
        max_rounds=1, branches=1)))
    metered("interact_stake", lambda m: (None if dossier.abstain else _society_p(
        CountingLLM(wm.llm, m), CountingLLM(wm.llm_hot, m), question, dossier, today,
        max_rounds=2, branches=2)))

    out["persistent"] = {"p": None, "spend": Meter().snapshot(),
                         "note": "NOT BUILT — persistent per-agent belief/memory state module (B7) does not "
                                 "exist yet; see AUDIT_PART_A §2"}
    out["time_evolving"] = {"p": None, "spend": Meter().snapshot(),
                            "note": "NOT BUILT — real-event-opportunity transitions (B8) do not exist yet"}
    metered("parametric", lambda m: parametric_binary_p(question, as_of, CountingLLM(wm.llm, m)))
    out["experimental"] = {"p": None, "spend": Meter().snapshot(),
                           "note": "NOT BUILT — graph/OASIS/TRIBE (B10); see Parts H/I"}
    return out


# --------------------------------------------------------------------- scoring / marginals
def _rows_for_scoring(runs):
    """Flatten predict_arms outputs [{arm:{p,...}}] into score_arms' row format [{arm:p, outcome:y}]."""
    rows = []
    for r in runs:
        row = {"outcome": r["outcome"]}
        for a in ALL_ARMS:
            row[a] = (r.get(a) or {}).get("p")
        rows.append(row)
    return rows


def _paired_brier_diff(runs, hi, lo, n_boot=2000, seed=12345):
    """Paired bootstrap + sign-flip permutation on (Brier_hi − Brier_lo) over items BOTH arms answered.
    Returns None if <5 paired items. Negative mean ⇒ hi is better (lower Brier)."""
    import random
    pairs = [((r[hi]["p"] - r["outcome"]) ** 2 - (r[lo]["p"] - r["outcome"]) ** 2)
             for r in runs if (r.get(hi) or {}).get("p") is not None
             and (r.get(lo) or {}).get("p") is not None]
    n = len(pairs)
    if n < 5:
        return {"hi": hi, "lo": lo, "n_pairs": n, "insufficient": True}
    mean = sum(pairs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(pairs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo_ci, hi_ci = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    # permutation: how often does a random sign-flip produce |mean| this large?
    obs = abs(mean)
    ge = 0
    for _ in range(n_boot):
        s = sum((p if rng.random() < 0.5 else -p) for p in pairs) / n
        if abs(s) >= obs:
            ge += 1
    p_perm = (ge + 1) / (n_boot + 1)
    return {"hi": hi, "lo": lo, "n_pairs": n, "mean_brier_diff": round(mean, 5),
            "ci95": [round(lo_ci, 5), round(hi_ci, 5)], "hi_better": mean < 0,
            "p_perm": round(p_perm, 4),
            "hi_wins_rows": round(sum(1 for d in pairs if d < 0) / n, 3)}


def report_marginals(runs) -> dict:
    """The full Part-B readout: per-arm scores + every marginal effect with paired bootstrap CI + permutation
    p. `runs` = list of predict_arms outputs each augmented with an 'outcome' key."""
    sb = score_arms(_rows_for_scoring(runs))
    marg = []
    for hi, lo, label in MARGINALS:
        d = _paired_brier_diff(runs, hi, lo)
        if d:
            d["isolates"] = label
            marg.append(d)
    spend = {}
    for a in ALL_ARMS:
        rs = [(r[a]["spend"]) for r in runs if a in r and r[a].get("p") is not None]
        if rs:
            spend[a] = {"mean_calls": round(sum(s["calls"] for s in rs) / len(rs), 1),
                        "mean_cost_usd": round(sum(s["cost_usd"] for s in rs) / len(rs), 5),
                        "mean_seconds": round(sum(s["seconds"] for s in rs) / len(rs), 1)}
    return {"n": len(runs), "arms": sb["arms"], "marginals": marg, "spend": spend}
