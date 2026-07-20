"""Mode graph — the typed causal decomposition of a question into END-STATES, PATHWAYS, and
DECISION STRUCTURES. This is the layer that keeps the event-time architecture a WORLD model rather
than a generalized event-resolution template:

  * PATHWAYS is a registry of causal route TYPES, actor-driven AND world-driven. A hurricane has no
    stance; inflation is not controlled by the most-opposed actor; adoption emerges from millions of
    weakly-coupled decisions. The registry is qualitative structure only — which kinds of routes
    exist and who decides on them — never effect sizes.
  * Each mode carries a DECISION STRUCTURE ({rule, approvers}) naming who actually decides it —
    a treaty needs the principals' consent; a 218-vote bill needs a majority; a resignation is
    unilateral. The structure routes WHICH institutional/actor mechanisms execute the mode; it is
    not a stance-combination formula.
  * Stances are MODE-SCOPED (`stance(actor, mode)`) qualitative records: Russia can simultaneously
    pursue its own victory, be committed to preventing Ukraine's, and be conditionally open to a
    ceasefire. A stance conditions the ACTOR'S OWN situated cognition (the behavior channel); it
    never becomes a hazard coefficient, an orientation weight, or a utility term (§NAP — those
    tables are buried in legacy_numeric_ablations).
  * `canonical_modes` makes the decomposition REPRODUCIBLE: K independent elicitation passes are
    reconciled with the compiler's structural hypotheses (id canonicalization, cluster, majority
    vote) — compile variance in the mode set becomes a measured consensus score instead of silent
    nondeterminism. Modes carry SUPPORT COUNTS (how many independent sources proposed them), never
    LLM-minted numeric priors.
  * `ground_process_states` classifies each pathway's CURRENT process state from evidence into a
    QUALITATIVE label with its evidential basis. There is no label→number map: a negotiation is
    not 45% complete. The typed record conditions actors and structure generation; causal
    consequences flow only through scenario-generated events and mechanisms.

Everything here is question-general: no scenario branching, no benchmark keys, no effect sizes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


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
#: stance commitment levels — the LLM only ever CLASSIFIES into these. The levels are a qualitative
#: vocabulary: they condition the actor's own cognition and the qualitative binding-commitment
#: check. They carry NO numeric orientation, shrink, control weight or hazard ratio (§NAP — the
#: historical tables are quarantined in legacy_numeric_ablations).
STANCE_LEVELS = ("committed_to_prevent", "conditionally_opposed", "weakly_opposed", "neutral",
                 "inclined_toward", "actively_pursuing", "formally_committed")
LEGACY_LEVELS = {"categorical_refusal": "committed_to_prevent",
                 "conditional_refusal": "conditionally_opposed",
                 "weak_opposition": "weakly_opposed",
                 "openness_to_agreement": "inclined_toward",
                 "formal_commitment_toward_agreement": "formally_committed"}

#: qualitative process-state vocabulary — an evidence-grounded CLASSIFICATION of whether a causal
#: process is underway. It maps to no number and no generic completion fraction.
PROCESS_STATE_VOCAB = ("dormant", "exploratory", "active", "advanced", "imminent")

#: prefix of the QUALITATIVE typed process-state quantities (string-valued; see
#: declare_typed_processes). The legacy numeric `pathway_progress:*` quantities no longer exist
#: on any production plan.
PROCESS_STATE_PREFIX = "process_state:"


def canon_level(level: str) -> str:
    lvl = str(level or "").strip().lower()
    return LEGACY_LEVELS.get(lvl, lvl)


def _mode_decision_structure(mode: dict, pathway: str) -> dict:
    """The mode's decision structure; rule defaults derive from the pathway. approvers is a list of
    entity names whose consent/vote the structure requires (may be empty = every relevant actor)."""
    ds = dict((mode or {}).get("decision_structure") or {})
    rule = str(ds.get("rule", "")).lower()
    if rule not in DECISION_RULES:
        rule = pathway_of(pathway).default_rule
    return {"rule": rule, "approvers": [str(a) for a in (ds.get("approvers") or [])],
            "stages": [str(s) for s in (ds.get("stages") or [])]}


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
Do NOT assign probabilities or numeric weights to the end-states — identify structure only.
QUESTION: {q}
RESOLUTION CRITERION: {crit}
Return ONLY JSON:
{{"modes": [{{"id": "<snake_case>",
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
            ent = {"id": str(m["id"])[:40]}
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
    with majority-vote support. Returns (modes, consensus_report). Modes carry SUPPORT counts
    (independent sources that proposed them) — no numeric priors exist anywhere in the mode set
    (§NAP: an LLM does not mint mode probabilities; relative likelihood of end-states is an OUTPUT
    of simulation, never an input). Fails toward the declared structure (never blocks): with no
    LLM, the compiler hypotheses/options pass through canonicalized."""
    sources = []                                   # each source: list of candidate mode dicts
    hyp = [{"id": str(h["id"])[:40],
            **({"pathway": h["pathway"]} if isinstance(h, dict) and h.get("pathway") in PATHWAYS else {}),
            **({"requires_agreement": h["requires_agreement"]}
               if isinstance(h, dict) and "requires_agreement" in h else {}),
            **({"decision_structure": h["decision_structure"]}
               if isinstance(h, dict) and isinstance(h.get("decision_structure"), dict) else {})}
           for h in (hypotheses or []) if isinstance(h, dict) and h.get("id")]
    if hyp:
        sources.append(("compiler_hypotheses", hyp))
    elif options and len(options) > 2:
        sources.append(("categorical_options", [{"id": str(o)[:40]} for o in options[:6]]))
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
        return ([{"id": "resolution", "support": 0}],
                {"n_sources": 0, "agreement": None, "note": "no structure and no llm"})
    clusters = {}                                  # canonical id -> {sources:set, fields}
    for src_name, cands in sources:
        seen_in_src = set()
        for m in cands:
            key = _cluster_key(m["id"], clusters)
            c = clusters.setdefault(key, {"sources": set(), "pathways": [],
                                          "structures": [], "describe": None, "names": set(),
                                          "requires_agreement": None})
            c["names"].add(_canon_mode_id(m["id"]))
            if key in seen_in_src:
                continue                           # time-indexed duplicates within one source
            seen_in_src.add(key)
            c["sources"].add(src_name)
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
        # the cluster's CANONICAL name is its shortest member id (the consensus name, reproducible
        # across compiles): 'peace_treaty' beats 'comprehensive_peace_treaty'
        cid = min(c["names"], key=lambda s: (len(s), s)) if c["names"] else key
        ent = {"id": cid, "support": support}
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
        modes = [{"id": best[0], "support": len(best[1]["sources"])}]
    modes.sort(key=lambda m: (-m["support"], m["id"]))
    modes = modes[:6]
    agree = round(sum(m["support"] for m in modes) / (len(modes) * n_sources), 3) \
        if n_sources else None
    report = {"n_sources": n_sources, "n_elicitation_passes": n_llm,
              "sources": [s for s, _ in sources], "agreement": agree,
              "modes": [{"id": m["id"], "support": f"{m['support']}/{n_sources}",
                         "pathway": m.get("pathway")} for m in modes],
              "dropped_minority_candidates": dropped[:8]}
    return modes, report


# ---------------------------------------------------------------- typed process-state grounding
_PROCESS_PROMPT = """For each causal PATHWAY below, classify the CURRENT state of that process toward
the question's resolution, strictly as of {as_of}, from the evidence (not from what you know happened
later). dormant = no process underway; exploratory = feelers/preparation only; active = process
genuinely underway; advanced = well past midpoint; imminent = resolution via this route appears close.
Also name the CONCRETE next step or stage the process is waiting on, if the evidence shows one.
QUESTION: {q}
RESOLUTION CRITERION: {crit}
PATHWAYS: {pathways}
EVIDENCE: {ev}
Return ONLY JSON: {{"process_states": [{{"pathway": "<id>",
  "state": "dormant|exploratory|active|advanced|imminent",
  "waiting_on": "<the concrete next step/stage, or null>",
  "basis": "<short quote or fact>"}}]}}"""


def ground_process_states(question, criterion, pathways, *, as_of="", evidence_text="", llm=None) -> dict:
    """Classify each pathway's CURRENT process state from evidence — a QUALITATIVE typed record
    {state, waiting_on, basis}. No value mapping exists: the record conditions actors and structure
    generation; it never enters a hazard, a progress bar, or any other number (§NAP). {} on any
    failure — an ungrounded process is simply ungrounded, never 'neutral 0.5'."""
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
            if pw in {str(p).lower() for p in pathways} and state in PROCESS_STATE_VOCAB:
                out[pw] = {"state": state,
                           "waiting_on": (str(row["waiting_on"])[:160]
                                          if row.get("waiting_on") else None),
                           "basis": str(row.get("basis", ""))[:160]}
        return out
    except Exception:  # noqa: BLE001 — grounding must never block the forecast
        return {}


def declare_typed_processes(plan, modes: list, *, grounding: dict = None) -> dict:
    """Declare one QUALITATIVE `process_state:<pathway>` record per pathway present in the mode
    set — a string-valued typed quantity carrying the grounded state label, plus the full record
    on `plan._process_records`. These are typed facts about what is actually true in the compiled
    world; they condition actor views and scenario structure generation. They are NOT numbers:
    nothing consumes them as intensity, progress, or probability (§NAP replaces the numeric
    `pathway_progress:*` declaration chain). An ungrounded pathway gets state `ungrounded` — an
    honest unknown, never a neutral midpoint. Idempotent."""
    grounding = grounding or {}
    pathways = sorted({mode_pathway(m) for m in (modes or [])})
    declared = {str(q.get("name")) for q in plan.quantities if isinstance(q, dict)}
    records = dict(getattr(plan, "_process_records", None) or {})
    added = {}
    for pw in pathways:
        var = f"{PROCESS_STATE_PREFIX}{pw}"
        g = grounding.get(pw) or {}
        rec = {"pathway": pw, "state": str(g.get("state") or "ungrounded"),
               "waiting_on": g.get("waiting_on"), "basis": g.get("basis")}
        records[pw] = rec
        if var in declared:
            continue
        plan.quantities.append({"name": var, "qtype": "process_state",
                                "value": rec["state"], "sd": None})
        added[pw] = rec
    plan._process_records = records
    prior = set(getattr(plan, "_declared_pathways", None) or [])
    plan._declared_pathways = sorted(prior | set(pathways))
    return {"pathways": pathways, "declared": added, "qualitative": True}
