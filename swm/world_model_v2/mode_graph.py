"""Mode graph — the typed causal decomposition of a question into END-STATES, PATHWAYS, and
DECISION STRUCTURES. This is the layer that keeps the event-time architecture a WORLD model rather
than a generalized event-resolution template:

  * PATHWAYS is a registry of causal route types, actor-driven AND world-driven. A hurricane has no
    stance; inflation is not controlled by the most-opposed actor; adoption emerges from millions of
    weakly-coupled decisions. Stance logic is therefore ONE mechanism family, applied only where the
    pathway is actor-driven — world-driven pathways couple to the population / nonlinear / scheduled
    mechanisms instead (event_time._endogenous_consume), and stances shrink to near-irrelevance under
    the `aggregation` combination rule.
  * Each mode carries a DECISION STRUCTURE ({rule, approvers}) from which the stance-combination law
    is DERIVED. "Most-opposed binds" is the unanimity/veto case — correct for a treaty, wrong for a
    218-vote bill (majority), a voluntary resignation (unilateral), or a market (aggregation).
  * Stances are MODE-SCOPED (`stance(actor, mode)`): Russia can simultaneously pursue its own victory,
    be committed to preventing Ukraine's, and be conditionally open to a ceasefire. A stance may carry
    `target_mode`; per (actor, mode) the most specific stance wins.
  * CONTROL is graded (sole_authority … informal_influence), not a boolean: a president may want a
    bill but lack the votes; a legislature may pass one but lack implementation capacity.
  * `canonical_modes` makes the decomposition REPRODUCIBLE: K independent elicitation passes are
    reconciled with the compiler's structural hypotheses (id canonicalization, cluster, majority
    vote, averaged priors) — compile variance in the mode set becomes a measured consensus score
    instead of silent nondeterminism.

Everything here is question-general: no scenario branching, no benchmark keys. Effect-size tables
are NOT owned here — callers pass the fitted-or-prior hazard-ratio table in (event_time owns packs).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_Z80 = 1.2816

# ---------------------------------------------------------------- pathway registry (universal)
@dataclass(frozen=True)
class Pathway:
    pathway_id: str
    actor_driven: bool          # do intentional actors' stances/policies drive this route?
    shared_process: bool        # one shared process all principals act inside (talks, a procedure)
                                # vs separate per-actor sub-processes (each side's own campaign)
    default_rule: str           # decision rule when the mode declares none
    describe: str = ""


PATHWAYS = {p.pathway_id: p for p in (
    # ---- actor-driven routes ----
    Pathway("cooperative_agreement", True, True, "unanimity",
            "the principals must consent (treaty, deal, settlement)"),
    Pathway("unilateral_action", True, False, "unilateral",
            "one actor's own act (resign, launch, invade, veto-by-walking-away)"),
    Pathway("institutional_procedure", True, True, "majority",
            "a body's procedure decides (vote, ruling, approval)"),
    Pathway("operational_execution", True, True, "hierarchy",
            "an organization executes a plan (ship a product, complete a project)"),
    Pathway("competitive_interaction", True, False, "strongest_actor",
            "rivals contest an outcome (win a race/market/battle/election)"),
    # ---- world-driven routes (stances are NOT the central causal variable) ----
    Pathway("threshold_crossing", False, True, "aggregation",
            "a measured quantity crosses a level (inflation over 3%, cases over N)"),
    Pathway("diffusion_adoption", False, True, "aggregation",
            "spread through a population (adoption reaches 20%, virality)"),
    Pathway("market_aggregation", False, True, "aggregation",
            "many weakly-coupled decisions aggregate (price, demand, earnings)"),
    Pathway("physical_process", False, True, "none",
            "a natural/technical process runs (hurricane landfall, hardware failure)"),
    Pathway("stochastic_external", False, True, "none",
            "an exogenous shock arrives (disaster, death in office, black swan)"),
    Pathway("resource_depletion", False, True, "cumulative_pressure",
            "a stock accumulates or runs out (runway, ammunition, reserves)"),
    Pathway("cascade_failure", False, True, "weakest_link",
            "a chain/network fails at its weakest point (grid, supply chain, bank run)"),
    Pathway("scheduled_transition", False, True, "none",
            "a deterministic calendar transition (term expiry, contract end)"),
)}

DECISION_RULES = ("unanimity", "majority", "weighted_coalition", "hierarchy", "unilateral",
                  "weakest_link", "strongest_actor", "cumulative_pressure", "aggregation", "none")

#: legacy 3-pathway labels + unknown ids resolve conservatively
_LEGACY_PATHWAYS = {"any": None}


def pathway_of(pathway_id: str) -> Pathway:
    p = PATHWAYS.get(str(pathway_id or "").strip().lower())
    if p is not None:
        return p
    # unknown/unclassified: conservative actor-driven shared process, cumulative combination —
    # NEVER a silent 'any' bucket; callers see the real id echoed in provenance
    return Pathway(str(pathway_id or "unknown"), True, True, "cumulative_pressure", "unclassified")


# keyword fallback (used only when a mode carries no semantic pathway label — elicitation always
# labels; this covers bare compiler hypotheses)
_PATHWAY_KEYWORDS = (
    ("cooperative_agreement", ("ceasefire", "treaty", "agreement", "deal", "settlement", "negotiat",
                               "accord", "truce", "pact", "compromise")),
    ("institutional_procedure", ("vote", "bill", "ruling", "court", "confirm", "impeach", "approv",
                                 "legislat", "referendum", "certif", "ratif")),
    ("threshold_crossing", ("cross", "exceed", "threshold", "above", "below", "reach_", "inflation",
                            "rate_hit", "surpass")),
    ("diffusion_adoption", ("adopt", "diffus", "viral", "spread", "uptake", "penetration")),
    ("market_aggregation", ("price", "market_", "earnings", "revenue", "demand", "sales", "gdp")),
    ("physical_process", ("hurricane", "landfall", "earthquake", "outage", "eruption", "flood")),
    ("stochastic_external", ("death", "dies", "accident", "assassinat", "disaster")),
    ("scheduled_transition", ("expir", "term_end", "scheduled", "deadline_pass", "mandate_end")),
    ("operational_execution", ("launch", "ship", "deploy", "complete", "deliver", "rollout")),
    ("cascade_failure", ("collapse_of_grid", "bank_run", "contagion", "cascade", "default_wave")),
)


def mode_pathway(mode) -> str:
    """The causal pathway a mode is reached through. Prefer the mode's own semantic label (from
    elicitation); fall back to the requires_agreement flag, then keyword matching, for compiler
    hypotheses that carry neither. Default: unilateral_action (a bare end-state named after an actor's
    act is the modal compiler shape)."""
    if isinstance(mode, dict):
        pw = str(mode.get("pathway", "")).lower()
        if pw in PATHWAYS:
            return pw
        if "requires_agreement" in mode:
            return "cooperative_agreement" if mode["requires_agreement"] else "unilateral_action"
    mid = str(mode["id"] if isinstance(mode, dict) else mode).lower()
    for pw, keys in _PATHWAY_KEYWORDS:
        if any(k in mid for k in keys):
            return pw
    return "unilateral_action"


# ---------------------------------------------------------------- stance taxonomy (universal)
#: stance commitment levels — the LLM only ever CLASSIFIES into these; effect sizes live in the
#: caller-supplied hazard-ratio table (fitted pack or documented priors)
STANCE_LEVELS = ("committed_to_prevent", "conditionally_opposed", "weakly_opposed", "neutral",
                 "inclined_toward", "actively_pursuing", "formally_committed")
LEGACY_LEVELS = {"categorical_refusal": "committed_to_prevent",
                 "conditional_refusal": "conditionally_opposed",
                 "weak_opposition": "weakly_opposed",
                 "openness_to_agreement": "inclined_toward",
                 "formal_commitment_toward_agreement": "formally_committed"}
#: signed orientation weight of each level (used for POLICY conditioning, not for hazards)
STANCE_ORIENTATION = {"committed_to_prevent": -0.9, "conditionally_opposed": -0.55,
                      "weakly_opposed": -0.25, "neutral": 0.0, "inclined_toward": 0.35,
                      "actively_pursuing": 0.7, "formally_committed": 0.9}

#: reliability shrinks the LOG-effect toward 1.0 (an inferred leaning moves hazards less than a law)
RELIABILITY_SHRINK = {"high": 1.0, "medium": 0.6, "low": 0.3}
#: capability — can the actor practically act on this stance (means, position, resources)?
CAPABILITY_SHRINK = {"high": 1.0, "medium": 0.75, "low": 0.4}
#: GRADED control over the pathway — replaces the controls_pathway boolean. Log-effect multipliers.
CONTROL_WEIGHTS = {"sole_authority": 1.0, "veto": 1.0, "agenda_setting": 0.75,
                   "partial_implementation": 0.6, "coalition_member": 0.5,
                   "operational_capability": 0.5, "informal_influence": 0.3, "none": 0.25}
#: share of a stance's total log-effect kept on the DIRECT hazard channel when the behavioral
#: channel (stance→policy→actions→pathway process→hazard) is live for the mode — the rest of the
#: effect is expected to be realized through simulated behavior. Documented structural choice;
#: the sensitivity harness varies it.
ENDOGENOUS_STANCE_SPLIT = 0.6

#: pathway-process quantities — written by simulated actor actions (phase4_execution), institutions
#: and world-driven consumers; consumed by hazard rounds. THE endogenous half of the hazard clock.
PROGRESS_PREFIX = "pathway_progress:"
#: current process state → initial progress value (0.5 = neutral / no effect on hazards)
PROCESS_STATE_LEVELS = {"dormant": 0.15, "exploratory": 0.3, "active": 0.5,
                        "advanced": 0.7, "imminent": 0.85}


def progress_var(pathway_id: str) -> str:
    return f"{PROGRESS_PREFIX}{str(pathway_id).strip().lower()}"


def canon_level(level: str) -> str:
    lvl = str(level or "").strip().lower()
    return LEGACY_LEVELS.get(lvl, lvl)


def stance_control_weight(stance: dict) -> float:
    """Graded control weight of a stance; legacy controls_pathway booleans map veto/informal.
    An absent/None `control` falls through to the legacy boolean — only an explicit string "none"
    means the graded no-control level."""
    raw = stance.get("control")
    c = str(raw).strip().lower() if isinstance(raw, str) else ""
    if c in CONTROL_WEIGHTS:
        return CONTROL_WEIGHTS[c]
    legacy = stance.get("controls_pathway")
    if legacy is True:
        return CONTROL_WEIGHTS["veto"]
    if legacy is False:
        return CONTROL_WEIGHTS["informal_influence"]
    return CONTROL_WEIGHTS["informal_influence"]


def _stance_hr(stance: dict, hr_table: dict, *, control_scaled: bool = True):
    """One stance → shrunk (median, lo80, hi80) hazard-ratio interval, or None when the stance's
    level is unknown. Log-effect × reliability × capability × (graded control weight)."""
    tup = hr_table.get(canon_level(stance.get("commitment_level")))
    if not tup:
        return None
    s = RELIABILITY_SHRINK.get(str(stance.get("reliability", "medium")).lower(), 0.6)
    s *= CAPABILITY_SHRINK.get(str(stance.get("capability", "high")).lower(), 1.0)
    if control_scaled:
        s *= stance_control_weight(stance)
    med, lo, hi = tup
    return (math.exp(s * math.log(med)), math.exp(s * math.log(lo)), math.exp(s * math.log(hi)))


def _relevant_stances(stances: list, pathway: str, mode: dict = None) -> list:
    """Per actor, the MOST SPECIFIC relevant stance wins: target_mode == this mode  >  pathway match
    > 'any'. A stance targeting a DIFFERENT mode is irrelevant here (Russia's stance toward Ukrainian
    victory must not bind Russian-victory hazards)."""
    mode_id = str((mode or {}).get("id", "") or "")
    best = {}
    for st in (stances or []):
        if not isinstance(st, dict):
            continue
        actor = str(st.get("actor", ""))
        tm = str(st.get("target_mode", "") or "")
        pw = str(st.get("pathway", "any")).lower()
        if tm:
            if mode_id and tm == mode_id:
                rank = 2
            else:
                continue                              # scoped to another mode — irrelevant here
        elif pw == pathway:
            rank = 1
        elif pw == "any":
            rank = 0
        else:
            continue
        if actor not in best or rank > best[actor][0]:
            best[actor] = (rank, st)
    return [st for _, st in best.values()]


def _mode_decision_structure(mode: dict, pathway: str) -> dict:
    """The mode's decision structure; rule defaults derive from the pathway. approvers is a list of
    entity names whose consent/vote the structure requires (may be empty = every relevant actor)."""
    ds = dict((mode or {}).get("decision_structure") or {})
    rule = str(ds.get("rule", "")).lower()
    if rule not in DECISION_RULES:
        rule = pathway_of(pathway).default_rule
    return {"rule": rule, "approvers": [str(a) for a in (ds.get("approvers") or [])],
            "stages": [str(s) for s in (ds.get("stages") or [])]}


def combine_stances(stances: list, pathway: str, *, mode: dict = None, hr_table: dict) -> dict:
    """Combine grounded stances into ONE hazard-ratio distribution for a mode, under the mode's
    DECISION STRUCTURE — the combination law is derived from the structure, never hard-coded:

      unanimity / weakest_link   any required party can block → the most-opposed relevant stance
                                 binds (veto logic; correct for treaties, NOT for bills)
      majority / weighted_coalition
                                 no single member binds → weighted geometric mean of the approvers'
                                 (or all relevant) stance effects — the median legislator, not the
                                 most opposed one
      hierarchy / unilateral     the actor with the strongest CONTROL binds at full effect; everyone
                                 else is resistance with log-effect ×0.25
      strongest_actor            competitive contest → the largest-|log| stance dominates
      cumulative_pressure        stances add up (log-sum, each ×0.5) — no veto, no majority
      aggregation                population/market-scale outcome → stances shrink ×0.25 (a market is
                                 not commanded); world-driven state channels carry the causality
      none                       a physical/scheduled process — stances have NO effect (×1.0)

    Returns {median, lo80, hi80, binding_actor, binding_level, combination_rule, ...} — every binding
    choice auditable in provenance."""
    ds = _mode_decision_structure(mode, pathway)
    rule = ds["rule"]
    neutral = {"median": 1.0, "lo80": 0.8, "hi80": 1.25, "binding_actor": None,
               "binding_level": "no_grounded_stance", "binding_reliability": None,
               "binding_pathway": pathway, "combination_rule": rule, "n_stances_combined": 0}
    if rule == "none":
        return dict(neutral, binding_level="pathway_not_stance_driven")
    relevant = _relevant_stances(stances, pathway, mode)
    if ds["approvers"] and rule in ("unanimity", "majority", "weighted_coalition"):
        # a declared approver set narrows who can bind a consent/vote structure
        apr = {a.lower() for a in ds["approvers"]}
        narrowed = [st for st in relevant if str(st.get("actor", "")).lower() in apr]
        relevant = narrowed or relevant

    def _entry(st, tup):
        med, lo, hi = tup
        return {"median": round(med, 4), "lo80": round(lo, 4), "hi80": round(hi, 4),
                "binding_actor": st.get("actor"), "binding_level": st.get("commitment_level"),
                "binding_reliability": st.get("reliability"), "binding_pathway": pathway,
                "combination_rule": rule, "n_stances_combined": len(relevant)}

    if rule in ("unanimity", "weakest_link"):
        pool = [(st, _stance_hr(st, hr_table, control_scaled=False)) for st in relevant]
        pool = [(st, t) for st, t in pool if t]
        if not pool:
            return neutral
        st, t = min(pool, key=lambda x: x[1][0])
        return _entry(st, t)

    if rule in ("hierarchy", "unilateral"):
        pool = [(st, _stance_hr(st, hr_table)) for st in relevant]
        pool = [(st, t) for st, t in pool if t]
        if not pool:
            return neutral
        ctrl_st, ctrl_t = max(pool, key=lambda x: stance_control_weight(x[0]))
        logm, loglo, loghi = (math.log(x) for x in ctrl_t)
        for st, t in pool:                        # others: shrunk resistance/support
            if st is ctrl_st:
                continue
            logm += 0.25 * math.log(t[0])
            loglo += 0.25 * math.log(t[1])
            loghi += 0.25 * math.log(t[2])
        return _entry(ctrl_st, (math.exp(logm), math.exp(loglo), math.exp(loghi)))

    if rule in ("majority", "weighted_coalition"):
        pool = [(st, _stance_hr(st, hr_table)) for st in relevant]
        pool = [(st, t) for st, t in pool if t]
        if not pool:
            return neutral
        wts = [max(0.05, stance_control_weight(st)) for st, _ in pool]
        z = sum(wts)
        logm = sum(w * math.log(t[0]) for w, (_, t) in zip(wts, pool)) / z
        loglo = sum(w * math.log(t[1]) for w, (_, t) in zip(wts, pool)) / z
        loghi = sum(w * math.log(t[2]) for w, (_, t) in zip(wts, pool)) / z
        st = max(pool, key=lambda x: abs(math.log(x[1][0])))[0]     # largest contributor, for audit
        out = _entry(st, (math.exp(logm), math.exp(loglo), math.exp(loghi)))
        out["binding_actor"] = f"{rule}:{len(pool)} actors (largest: {st.get('actor')})"
        return out

    if rule == "strongest_actor":
        pool = [(st, _stance_hr(st, hr_table)) for st in relevant]
        pool = [(st, t) for st, t in pool if t]
        if not pool:
            return neutral
        st, t = max(pool, key=lambda x: abs(math.log(x[1][0])))
        return _entry(st, t)

    if rule == "aggregation":
        pool = [(st, _stance_hr(st, hr_table)) for st in relevant]
        pool = [(st, t) for st, t in pool if t]
        if not pool:
            return dict(neutral, binding_level="aggregation_no_stance_channel")
        z = len(pool)
        shrink = 0.25                              # a market/population is not commanded
        logm = shrink * sum(math.log(t[0]) for _, t in pool) / z
        loglo = shrink * sum(math.log(t[1]) for _, t in pool) / z
        loghi = shrink * sum(math.log(t[2]) for _, t in pool) / z
        st = pool[0][0]
        out = _entry(st, (math.exp(logm), math.exp(loglo), math.exp(loghi)))
        out["binding_actor"] = f"aggregation:{z} actors (shrunk ×{shrink})"
        return out

    # cumulative_pressure (and the conservative unknown default): stances add
    pool = [(st, _stance_hr(st, hr_table)) for st in relevant]
    pool = [(st, t) for st, t in pool if t]
    if not pool:
        return neutral
    logm = sum(0.5 * math.log(t[0]) for _, t in pool)
    loglo = sum(0.5 * math.log(t[1]) for _, t in pool)
    loghi = sum(0.5 * math.log(t[2]) for _, t in pool)
    logm = max(math.log(0.2), min(math.log(5.0), logm))
    loglo = max(math.log(0.1), min(math.log(5.0), loglo))
    loghi = max(math.log(0.2), min(math.log(8.0), loghi))
    st = max(pool, key=lambda x: abs(math.log(x[1][0])))[0]
    out = _entry(st, (math.exp(logm), math.exp(loglo), math.exp(loghi)))
    out["binding_actor"] = f"cumulative:{len(pool)} actors (largest: {st.get('actor')})"
    return out


# ---------------------------------------------------------------- policy conditioning (stance→behavior)
def pathway_orientation(stances: list, pathway: str) -> float:
    """The actor's net orientation toward the PATHWAY PROCESS advancing, in [-1, 1] — the quantity
    the Phase-4 policy consumes. Pursue-stances push positive. Prevent-stances push negative when
    UNTARGETED (opposing the process itself: refusing to negotiate) or when the pathway is one
    SHARED process (stalling any specific deal stalls the talks); a targeted prevent on a
    per-actor pathway (Russia preventing UKRAINE'S victory) says nothing about the actor's appetite
    for the process (their own campaign) and contributes 0 — that stance binds the TARGET MODE's
    hazard instead (combine_stances)."""
    shared = pathway_of(pathway).shared_process
    total = 0.0
    for st in (stances or []):
        if not isinstance(st, dict):
            continue
        pw = str(st.get("pathway", "any")).lower()
        if pw not in (pathway, "any"):
            continue
        w = STANCE_ORIENTATION.get(canon_level(st.get("commitment_level")), 0.0)
        if w < 0.0 and st.get("target_mode") and not shared:
            continue
        w *= RELIABILITY_SHRINK.get(str(st.get("reliability", "medium")).lower(), 0.6)
        w *= CAPABILITY_SHRINK.get(str(st.get("capability", "high")).lower(), 1.0)
        total += w
    return max(-1.0, min(1.0, total))


# ---------------------------------------------------------------- canonical mode decomposition
_MODES_PROMPT = """Through which mutually exclusive END-STATES can this question's outcome be REACHED?
List 2-6, each an end-state that SATISFIES the resolution criterion when it holds (not an intermediate
or escalation state). PASS {k} of {n} — reason independently this pass.
For each end-state give:
 * pathway — the causal ROUTE it is reached through, one of:
   cooperative_agreement (principals must consent), unilateral_action (one actor's own act),
   institutional_procedure (a body's vote/ruling), operational_execution (an organization executes),
   competitive_interaction (rivals contest it), threshold_crossing (a measured quantity crosses a
   level), diffusion_adoption (spread through a population), market_aggregation (many weakly-coupled
   decisions aggregate), physical_process (a natural/technical process), stochastic_external (an
   exogenous shock), resource_depletion (a stock runs out), cascade_failure (a chain fails),
   scheduled_transition (a deterministic calendar transition).
 * decision_structure — who/what decides it: rule one of unanimity|majority|weighted_coalition|
   hierarchy|unilateral|weakest_link|strongest_actor|cumulative_pressure|aggregation|none, and
   approvers = the named actors/bodies whose consent or vote that rule runs over (empty if none).
QUESTION: {q}
RESOLUTION CRITERION: {crit}
Return ONLY JSON:
{{"modes": [{{"id": "<snake_case>", "prior": <0..1 relative weight>,
   "pathway": "<one of the pathway ids above>",
   "decision_structure": {{"rule": "<rule>", "approvers": ["<name>", ...]}},
   "describe": "<one sentence>"}}]}}"""


def _canon_mode_id(mid: str) -> str:
    """Canonical mode identity: lowercase snake, time-index stripped (ceasefire_2026 → ceasefire —
    timing belongs to the simulation, not the mode identity)."""
    s = re.sub(r"[^a-z0-9]+", "_", str(mid).strip().lower()).strip("_")
    s = re.sub(r"_?(?:19|20)\d{2}$", "", s).strip("_")
    return s or str(mid)


def _tokens(mid: str) -> set:
    return {t for t in _canon_mode_id(mid).split("_") if len(t) > 2}


def _cluster_key(mid: str, clusters: dict) -> str:
    """Match a candidate id onto an existing cluster by canonical equality or token overlap ≥ 0.5."""
    cid = _canon_mode_id(mid)
    if cid in clusters:
        return cid
    toks = _tokens(cid)
    if toks:
        for key in clusters:
            kt = _tokens(key)
            if kt and len(toks & kt) / len(toks | kt) >= 0.5:
                return key
    return cid


def _elicit_modes_once(question, criterion, llm, k, n) -> list:
    from swm.engine.grounding import parse_json
    raw = parse_json(llm(_MODES_PROMPT.format(
        q=question, crit=(criterion or {}).get("resolves_yes_iff", "(as stated)"), k=k, n=n))) or {}
    out = []
    for m in (raw.get("modes") or []):
        if isinstance(m, dict) and m.get("id"):
            ent = {"id": str(m["id"])[:40], "prior": max(0.0, float(m.get("prior", 1.0) or 1.0))}
            pw = str(m.get("pathway", "")).lower()
            if pw in PATHWAYS:
                ent["pathway"] = pw
            elif "requires_agreement" in m:               # older elicitation shape
                ent["requires_agreement"] = bool(m["requires_agreement"])
            ds = m.get("decision_structure")
            if isinstance(ds, dict) and str(ds.get("rule", "")).lower() in DECISION_RULES:
                ent["decision_structure"] = {"rule": str(ds["rule"]).lower(),
                                             "approvers": [str(a)[:60] for a in
                                                           (ds.get("approvers") or [])][:12]}
            if m.get("describe"):
                ent["describe"] = str(m["describe"])[:160]
            out.append(ent)
    return out[:6]


def canonical_modes(*, question: str, criterion: dict, hypotheses: list, options: list,
                    llm=None, k_passes: int = 3) -> tuple:
    """The REPRODUCIBLE mode decomposition: reconcile the compiler's structural hypotheses, the
    contract's categorical options, and K independent elicitation passes into one canonical mode set
    with majority-vote support. Returns (modes, consensus_report). Fails toward the declared
    structure (never blocks): with no LLM, the compiler hypotheses/options pass through canonicalized."""
    sources = []                                   # each source: list of candidate mode dicts
    hyp = [{"id": str(h["id"])[:40], "prior": float(h.get("prior", 1.0) or 1.0),
            **({"pathway": h["pathway"]} if isinstance(h, dict) and h.get("pathway") in PATHWAYS else {}),
            **({"requires_agreement": h["requires_agreement"]}
               if isinstance(h, dict) and "requires_agreement" in h else {}),
            **({"decision_structure": h["decision_structure"]}
               if isinstance(h, dict) and isinstance(h.get("decision_structure"), dict) else {})}
           for h in (hypotheses or []) if isinstance(h, dict) and h.get("id")]
    if hyp:
        sources.append(("compiler_hypotheses", hyp))
    elif options and len(options) > 2:
        sources.append(("categorical_options", [{"id": str(o)[:40], "prior": 1.0}
                                                for o in options[:6]]))
    n_llm = 0
    if llm is not None:
        for k in range(1, max(1, int(k_passes)) + 1):
            try:
                cand = _elicit_modes_once(question, criterion, llm, k, k_passes)
            except Exception:  # noqa: BLE001 — elicitation must never block the forecast
                cand = []
            if cand:
                sources.append((f"elicitation_{k}", cand))
                n_llm += 1
    if not sources:
        return ([{"id": "resolution", "prior": 1.0}],
                {"n_sources": 0, "agreement": None, "note": "no structure and no llm"})
    clusters = {}                                  # canonical id -> {sources:set, priors:[], fields}
    for src_name, cands in sources:
        seen_in_src = set()
        for m in cands:
            key = _cluster_key(m["id"], clusters)
            c = clusters.setdefault(key, {"sources": set(), "priors": [], "pathways": [],
                                          "structures": [], "describe": None,
                                          "requires_agreement": None})
            if key in seen_in_src:
                c["priors"][-1] += m["prior"]      # time-indexed duplicates within one source: sum
                continue
            seen_in_src.add(key)
            c["sources"].add(src_name)
            c["priors"].append(m["prior"])
            if m.get("pathway"):
                c["pathways"].append(m["pathway"])
            if m.get("requires_agreement") is not None and c["requires_agreement"] is None:
                c["requires_agreement"] = m.get("requires_agreement")
            if isinstance(m.get("decision_structure"), dict):
                c["structures"].append(m["decision_structure"])
            if m.get("describe") and not c["describe"]:
                c["describe"] = m["describe"]
    n_sources = len(sources)
    need = max(1, (n_sources + 1) // 2)            # majority of independent sources
    modes, dropped = [], []
    for key, c in clusters.items():
        support = len(c["sources"])
        if support < need:
            dropped.append({"id": key, "support": f"{support}/{n_sources}"})
            continue
        ent = {"id": key, "prior": sum(c["priors"]) / len(c["priors"]), "support": support}
        if c["pathways"]:
            ent["pathway"] = max(set(c["pathways"]), key=c["pathways"].count)   # majority pathway
        elif c["requires_agreement"] is not None:
            ent["requires_agreement"] = c["requires_agreement"]
        if c["structures"]:
            rules = [s.get("rule") for s in c["structures"] if s.get("rule")]
            ent["decision_structure"] = {
                "rule": max(set(rules), key=rules.count) if rules else None,
                "approvers": max((s.get("approvers") or [] for s in c["structures"]), key=len)}
        if c["describe"]:
            ent["describe"] = c["describe"]
        modes.append(ent)
    if not modes:                                  # majority filter emptied everything: fail open
        best = max(clusters.items(), key=lambda kv: len(kv[1]["sources"]))
        modes = [{"id": best[0], "prior": 1.0, "support": len(best[1]["sources"])}]
    modes.sort(key=lambda m: -m["prior"])
    modes = modes[:6]
    agree = round(sum(m["support"] for m in modes) / (len(modes) * n_sources), 3)
    report = {"n_sources": n_sources, "n_elicitation_passes": n_llm,
              "sources": [s for s, _ in sources], "agreement": agree,
              "modes": [{"id": m["id"], "support": f"{m['support']}/{n_sources}",
                         "pathway": m.get("pathway")} for m in modes],
              "dropped_minority_candidates": dropped[:8]}
    return modes, report


# ---------------------------------------------------------------- pathway-process declaration
_PROCESS_PROMPT = """For each causal PATHWAY below, classify the CURRENT state of that process toward
the question's resolution, strictly as of {as_of}, from the evidence (not from what you know happened
later). dormant = no process underway; exploratory = feelers/preparation only; active = process
genuinely underway; advanced = well past midpoint; imminent = resolution via this route appears close.
QUESTION: {q}
RESOLUTION CRITERION: {crit}
PATHWAYS: {pathways}
EVIDENCE: {ev}
Return ONLY JSON: {{"process_states": [{{"pathway": "<id>",
  "state": "dormant|exploratory|active|advanced|imminent", "basis": "<short quote or fact>"}}]}}"""


def ground_process_states(question, criterion, pathways, *, as_of="", evidence_text="", llm=None) -> dict:
    """Classify each pathway's CURRENT process state from evidence (LLM classifies; the value mapping
    is the documented PROCESS_STATE_LEVELS table). {} on any failure — unknown stays neutral 0.5."""
    if llm is None or not pathways:
        return {}
    try:
        from swm.engine.grounding import parse_json
        raw = parse_json(llm(_PROCESS_PROMPT.format(
            q=question, crit=(criterion or {}).get("resolves_yes_iff", "(as stated)"),
            pathways=sorted(pathways), as_of=as_of or "(as stated)",
            ev=(evidence_text or "(none)")[:1400]))) or {}
        out = {}
        for row in (raw.get("process_states") or []):
            if not isinstance(row, dict):
                continue
            pw = str(row.get("pathway", "")).lower()
            state = str(row.get("state", "")).lower()
            if pw in {str(p).lower() for p in pathways} and state in PROCESS_STATE_LEVELS:
                out[pw] = {"state": state, "value": PROCESS_STATE_LEVELS[state],
                           "basis": str(row.get("basis", ""))[:160]}
        return out
    except Exception:  # noqa: BLE001 — grounding must never block the forecast
        return {}


def declare_pathway_processes(plan, modes: list, *, grounding: dict = None) -> dict:
    """Declare one `pathway_progress:<pathway>` quantity per pathway present in the mode set —
    initialized from the grounded process state (documented level map) or neutral 0.5. These are the
    state variables the simulated actors' ACTIONS (phase4_execution), institutional stage reviews and
    world-driven consumers move, and the hazard rounds consume — the endogenous clock. Idempotent."""
    grounding = grounding or {}
    pathways = sorted({mode_pathway(m) for m in (modes or [])})
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    added = {}
    for pw in pathways:
        var = progress_var(pw)
        if var in declared:
            continue
        g = grounding.get(pw) or {}
        val = float(g.get("value", 0.5))
        plan.quantities.append({"name": var, "qtype": "pathway_progress",
                                "value": round(val, 3), "sd": 0.15})
        added[pw] = {"var": var, "initial": round(val, 3),
                     "state": g.get("state", "unknown_neutral"), "basis": g.get("basis")}
    prior = set(getattr(plan, "_declared_pathways", None) or [])
    plan._declared_pathways = sorted(prior | set(pathways))
    return {"pathways": pathways, "declared": added}
