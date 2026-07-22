"""Actor-state hypotheses (LLM-generated) and their weights (COUNTED, never LLM-rated).

The separation the whole accuracy fix rests on:

  * the LLM PROPOSES which private realities could exist and (later) simulates behavior inside
    a specified one — it emits `ActorStateHypothesis` objects with NO numbers;
  * the WEIGHTS come from `ActorStatePosteriorEngine`: a counted historical reference class per
    state (grounding.py) × hard-evidence elimination, combined by beta-binomial normalization
    into `ActorStatePosteriorRange`s with full `ActorStateWeightProvenance`.

Shared latent world conditions are weighted FIRST (counted), and actor states are conditional
on them, so correlated actors are never independently multiplied. Where the data cannot
identify the joint dependence, BOTH the independent and the comonotonic (shared-cause-locked)
structures are carried and the forecast's sensitivity across them is reported
(`dependence_sensitive`) — no arbitrary correlation is invented.

THE COMPLETENESS LAW (simulation-completion fix): private-state uncertainty is an INPUT to
simulation, never a reason to stop it. The represented states always carry the FULL branch
mass (their weights normalize to 1). What used to be "unknown-state mass" is now a small
BOUNDED per-actor residual r_a — the counted out-of-set frequency only, capped at
`MAX_ACTOR_RESIDUAL` — reported as an outcome-interval widening at finalize
(1 - prod(1-r_a)), never a world branch, never multiplied across actors as unknown worlds,
and never receiving the prior, the average action, or 50%."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key

STATES_VERSION = "lean_v2.states.v2"

#: per-actor residual cap — the genuinely-unrepresentable share may never exceed this (a
#: larger counted out-of-set share means the represented basis is wrong and the completeness
#: ladder must ADD states instead of widening the bound)
MAX_ACTOR_RESIDUAL = 0.2

#: numeric fields an actor-state hypothesis may NEVER contain (rejected + recorded)
_BANNED_STATE_KEYS = ("weight", "probability", "prob", "likelihood", "confidence_score",
                      "percent", "pct", "rate", "p", "odds", "score")


@dataclass
class ActorStateEvidenceLink:
    evidence_id: str
    relation: str                               # supports | contradicts | distinguishes
    hard: bool = False                          # hard evidence can ELIMINATE a state
    quote: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActorStateHypothesis:
    """One possible private reality for one actor. Purely qualitative — no probabilities."""
    actor_id: str
    state_id: str
    claim: str = ""
    beliefs: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    commitments: list = field(default_factory=list)
    pressures: str = ""
    relationships: dict = field(default_factory=dict)
    stances: list = field(default_factory=list)
    supporting_evidence_ids: list = field(default_factory=list)
    contradicting_evidence_ids: list = field(default_factory=list)
    historical_case_refs: list = field(default_factory=list)
    distinguishing_observations: list = field(default_factory=list)
    action_if_state: str = ""
    #: the canonical action this state tends toward (D2). A counted reference class may weight
    #: this state ONLY when the class's own action_option_id is compatible with this tendency —
    #: a "dissents-for-a-hike" class can never weight a state that tends to hold. Defaults to
    #: action_if_state when not separately typed.
    expected_action_tendency: str = ""
    reversal_capable: bool = False
    assumptions: list = field(default_factory=list)
    transition_triggers: list = field(default_factory=list)
    reference_class_key: str = ""               # which counted class weights this state
    aligned_condition: dict = field(default_factory=dict)  # {condition_id: condition_state}
    eliminated: bool = False
    elimination_reason: str = ""
    is_unknown: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def to_variant(self) -> dict:
        """Render into the engine's variant shape (state content the actor prompt renders).
        `action_if_state` is carried so a HARD-deadline forced vote can fall back to the
        grounded per-state action the completeness layer constructed — simulating the actor
        in that state, never inventing a vote."""
        return {"variant_id": self.state_id,
                "state": {"beliefs": list(self.beliefs), "goals": list(self.goals),
                          "stances": list(self.stances), "pressures": self.pressures,
                          "relationships": dict(self.relationships)},
                "action_if_state": self.action_if_state,
                "reversal_capable": self.reversal_capable,
                "is_unknown": self.is_unknown}


@dataclass
class ActorStatePosteriorRange:
    state_id: str
    mid: float                                  # normalized posterior weight (counted)
    lo: float
    hi: float
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def reject_numeric_state_weights(raw_state: dict) -> list:
    """Deterministically detect a probability/weight the LLM smuggled into a state hypothesis.
    Returns the list of rejected (path, value) — the caller records them and strips them."""
    rejected = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                kl = str(k).lower()
                if any(b == kl or kl.endswith("_" + b) for b in _BANNED_STATE_KEYS) \
                        and isinstance(v, (int, float)) and not isinstance(v, bool):
                    rejected.append({"path": f"{path}.{k}", "value": v})
                    o.pop(k, None)
                else:
                    walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, x in enumerate(o):
                walk(x, f"{path}[{i}]")
    walk(dict(raw_state))
    return rejected


# ------------------------------------------------------------------ hypothesis-set validation
def validate_hypothesis_set(actor_id: str, hyps: list, *, institution_rules: list,
                            hard_evidence_ids: set) -> dict:
    """Deterministic checks (§2): duplicates/paraphrases collapsed, hard-evidence-contradicted
    states eliminated, wording-only duplicates removed, coverage assessed. Always preserves an
    explicit unknown state. Returns {kept, eliminated, diagnostics}."""
    seen_sig: dict = {}
    kept, eliminated, notes = [], [], []
    for h in hyps:
        # paraphrase/duplicate collapse on a behavioral signature (beliefs+action tokens)
        sig = _behavioral_signature(h)
        if sig in seen_sig:
            eliminated.append({"state_id": h.state_id, "reason":
                               f"paraphrase/duplicate of {seen_sig[sig]}"})
            continue
        # hard-evidence contradiction eliminates the state
        hard_contra = [e for e in h.contradicting_evidence_ids if e in hard_evidence_ids]
        if hard_contra:
            h.eliminated = True
            h.elimination_reason = f"contradicted by hard evidence {hard_contra[:3]}"
            eliminated.append({"state_id": h.state_id, "reason": h.elimination_reason})
            continue
        # a state that changes no action cannot affect the decision — record, keep only if it
        # is the sole hypothesis (otherwise it is inert)
        seen_sig[sig] = h.state_id
        kept.append(h)
    reversal = any(h.reversal_capable for h in kept)
    coverage = "covered" if reversal and len(kept) >= 2 else "possibly_incomplete"
    if len(kept) < 2:
        notes.append("only one distinct state survived — decision space likely under-covered")
    if not reversal:
        notes.append("no reversal-capable state among survivors — an omitted reversal state "
                     "may exist (unknown-state search will probe this)")
    return {"kept": kept, "eliminated": eliminated, "coverage": coverage,
            "diagnostics": notes}


def _behavioral_signature(h: ActorStateHypothesis) -> str:
    toks = sorted(set(norm_key(h.action_if_state).split())
                  | {norm_key(b) for b in h.beliefs[:3]})
    return hashlib.sha256("\x00".join(t for t in toks if t).encode()).hexdigest()[:16]


def _reference_class_action_conflict(tbl: dict, state, feasible_options) -> str:
    """D2: return a non-empty reason string when the counted class's action is INCOMPATIBLE
    with the state's action tendency, else "". Compatibility is decided by TYPED canonical
    options, never by lexical prose overlap: the class's declared `action_option_id` and the
    state's `expected_action_tendency` (or `action_if_state`) are each normalized to a canonical
    option among the actor's feasible options; if both resolve and DIFFER, it is a conflict. If
    either side does not declare a typed action, no conflict is asserted (the match proceeds on
    the existing counted logic — this guard only ever REJECTS a proven direction inversion)."""
    class_action = (tbl.get("action_option_id") or "").strip()
    state_action = (getattr(state, "expected_action_tendency", "")
                    or getattr(state, "action_if_state", "")).strip()
    if not class_action or not state_action:
        return ""
    opts = list(feasible_options or [])
    if opts:
        from swm.world_model_v2.lean_v2.canonical_options import normalize_option
        c = normalize_option(class_action, opts)
        s = normalize_option(state_action, opts)
        if c is not None and s is not None and c.canonical_option_id != s.canonical_option_id:
            return (f"class action '{class_action}' -> {c.canonical_option_id} conflicts with "
                    f"state tendency '{state_action}' -> {s.canonical_option_id}")
        return ""
    # no feasible-option list: fall back to a direct normalized-string comparison
    if norm_key(class_action) and norm_key(state_action) \
            and norm_key(class_action) != norm_key(state_action):
        return (f"class action '{class_action}' conflicts with state tendency "
                f"'{state_action}' (no option set to reconcile)")
    return ""


# ------------------------------------------------------------------ the posterior engine
class ActorStatePosteriorEngine:
    """Turns counted reference classes + validated hypotheses into weighted states — the ONLY
    weight source. No qualitative label ever becomes a number here."""

    def __init__(self, grounding: dict):
        self.grounding = grounding or {}
        self.actor_classes = self.grounding.get("actor_state_reference_classes") or {}
        self.shared = self.grounding.get("shared_world_conditions") or {}
        self.provenance_log: list = []

    # -- shared world conditions (weighted FIRST, counted) --------------------------------
    def shared_condition_worlds(self) -> list:
        """[(condition_id, {state: weight}, provenance)] — counted rates over the condition's
        states. A condition with a usable counted YES rate splits into holds/does_not_hold by
        that rate; otherwise it is carried as a uniform-but-DISCLOSED sensitivity axis."""
        out = []
        for cid, sc in sorted(self.shared.items()):
            tbl = sc.get("table") or {}
            states = sc.get("states") or ["holds", "does_not_hold"]
            rate = (tbl.get("provenance") or {}).get("rate_mean")
            n = (tbl.get("provenance") or {}).get("denominator") or 0
            if rate is not None and n > 0 and len(states) == 2:
                weights = {states[0]: round(rate, 4), states[1]: round(1 - rate, 4)}
                prov = {"source": "counted_shared_condition", "n": n,
                        "interval": tbl.get("interval"), "cases": (tbl.get("provenance")
                                                                   or {}).get("numerator")}
            else:
                w = round(1.0 / len(states), 4)
                weights = {s: w for s in states}
                prov = {"source": "uniform_disclosed_no_count", "n": n,
                        "note": "no usable counted rate — carried as a disclosed sensitivity "
                                "axis, never an invented probability"}
            out.append((cid, weights, prov, sc.get("affects_actors") or []))
        return out

    # -- per-actor state weights (counted, conditional on a shared world) -----------------
    def weight_actor_states(self, actor_id: str, hyps: list, *,
                            shared_world: dict = None, feasible_options: list = None) -> tuple:
        """Returns ([ActorStatePosteriorRange...], bounded_residual, provenance). Weights are
        the normalized counted reference-class rates of the SURVIVING states and always sum to
        1 — the represented states carry the full branch mass. The second element is the
        BOUNDED omitted-state residual r_a (counted under-summing only, capped at
        MAX_ACTOR_RESIDUAL) used ONLY to widen the outcome interval at finalize — it is never
        branch mass, never a coverage penalty, never a label-derived number."""
        survivors = [h for h in hyps if not h.eliminated and not h.is_unknown]
        class_list = self.actor_classes.get(actor_id, [])
        matched, intervals, provs = {}, {}, {}
        # assign each counted class to its SINGLE best-matching state (argmax token overlap;
        # a tie prefers the reversal-capable state — a "dissents"/"minority" class describes the
        # minority state). One class → one state, so a class is never double-counted.
        claimed = set()
        for tbl in class_list:
            best, best_ov = None, 0
            qtoks = set(norm_key(tbl.get("quantity")).split())
            for h in survivors:
                if h.state_id in claimed:
                    continue
                htoks = (set(norm_key(h.claim).split())
                         | set(norm_key(h.action_if_state).split())) - {actor_id.lower()}
                ov = len(htoks & (qtoks - {actor_id.lower()}))
                if ov > best_ov or (ov == best_ov and ov > 0 and h.reversal_capable
                                    and (best is None or not best.reversal_capable)):
                    best, best_ov = h, ov
            if best is None or best_ov < 1:
                continue
            # D2: a counted class may weight a state ONLY when their action semantics AGREE.
            # If the class declares the option it counts (action_option_id) and the state
            # declares its expected action tendency, and those resolve to DIFFERENT canonical
            # options, REJECT the match — never assign a pro-hike rate to a hold state.
            conflict = _reference_class_action_conflict(tbl, best, feasible_options)
            if conflict:
                self.provenance_log.append({"actor_id": actor_id, "rejected_class": tbl.get("key"),
                                            "state": best.state_id, "reason": conflict})
                claimed.add(best.state_id)      # do not let another class silently re-bind it
                continue
            rate = (tbl.get("provenance") or {}).get("rate_mean")
            n = (tbl.get("provenance") or {}).get("denominator") or 0
            if rate is None or n <= 0:
                continue
            # the counted rate is the class's coverage signal (feeds the residual and seeds the
            # action baseline). Conditioning on the shared world is applied INSIDE the action
            # baseline via typed state<->condition alignment, not by an ad-hoc downweight here.
            matched[best.state_id] = rate
            intervals[best.state_id] = tbl.get("interval") or (0.0, 1.0)
            provs[best.state_id] = {"source": "counted_reference_class", "key": tbl.get("key"),
                                    "rate": rate, "n": n, "interval": intervals[best.state_id],
                                    "hierarchy_level": (tbl.get("provenance") or {})
                                    .get("hierarchy_level")}
            claimed.add(best.state_id)
        matched_sum = sum(matched.values())
        unmatched = [h.state_id for h in survivors if h.state_id not in matched]
        residual = max(0.0, 1.0 - matched_sum)      # for the bounded coverage residual only
        n_counted = len(matched)
        reversal = any(h.reversal_capable for h in survivors)
        # D8: allocate mass to ACTION TENDENCIES first, from a counted, hierarchically
        # partial-pooled baseline over the actor's feasible action classes — NEVER the number
        # of prose stories, NEVER residual/len(states). States that share a tendency split only
        # that tendency's total (trajectory/sensitivity), so story count cannot move the
        # forecast. The old equal split is gone; the residual below is a SEPARATE coverage bound.
        raw_weights = self._allocate_by_action_class(
            actor_id, survivors, matched, intervals, provs,
            shared_world=shared_world, feasible_options=feasible_options)
        # THE COMPLETENESS LAW: the represented states normalize to the FULL branch mass.
        # The residual is the COUNTED under-summing only (matched counted rates that leave
        # probability no represented state holds), capped — an interval-widening bound at
        # finalize, never branch mass and never a coverage penalty. Set thinness / missing
        # reversal states are handled by the completeness ladder (which ADDS states), not by
        # converting doubt into unanswerable world mass.
        if not survivors:
            # nothing to weight — the completeness invariant makes this unreachable for a
            # consequential actor; callers treat an empty row set as a hard readiness failure
            self.provenance_log.append({"actor_id": actor_id, "residual": MAX_ACTOR_RESIDUAL,
                                        "n_counted_states": 0, "empty_state_set": True})
            return [], MAX_ACTOR_RESIDUAL, {"residual": MAX_ACTOR_RESIDUAL,
                                            "empty_state_set": True, "n_counted_states": 0,
                                            "matched_sum": 0.0}
        if matched and matched_sum < 0.999 and not unmatched:
            residual_bound = round(min(MAX_ACTOR_RESIDUAL, residual), 4)
            residual_provenance = (f"counted classes sum to {matched_sum:.3f} with no "
                                   f"unmatched state to hold the remainder — bounded residual")
        elif n_counted == 0:
            # no counted class at all: the states carry the mass uniformly; the residual is
            # the bounded default (the ladder's decision-spanning basis drives this to 0)
            residual_bound = MAX_ACTOR_RESIDUAL
            residual_provenance = "no counted reference class — residual at the declared cap"
        else:
            residual_bound = 0.0
            residual_provenance = "counted classes + complement cover the represented basis"
        z = sum(raw_weights.values()) or 1.0
        rows = []
        for h in survivors:
            w = raw_weights.get(h.state_id, 0.0)
            share = w / z
            iv = intervals.get(h.state_id, (0.0, 1.0))
            rows.append(ActorStatePosteriorRange(
                state_id=h.state_id, mid=round(share, 4),
                lo=round(min(share, iv[0] / z), 4),
                hi=round(min(1.0, max(share, iv[1] / z)), 4),
                provenance=provs.get(h.state_id, {"source": "no_counted_class"})))
        self.provenance_log.append({"actor_id": actor_id, "residual": residual_bound,
                                    "matched_sum": round(matched_sum, 4),
                                    "n_counted_states": n_counted,
                                    "reversal_present": reversal})
        return rows, residual_bound, {"residual": residual_bound,
                                      "residual_provenance": residual_provenance,
                                      "n_counted_states": n_counted,
                                      "matched_sum": round(matched_sum, 4),
                                      "law": "represented states normalize to 1; residual is "
                                             "a bounded interval-widener, never branch mass"}

    # -- D8: action-tendency-first allocation ---------------------------------------------
    @staticmethod
    def _action_class_of(h: ActorStateHypothesis, feasible_options) -> str:
        """The canonical ACTION CLASS this state tends toward. Prefer the typed
        `expected_action_tendency`, else `action_if_state`; canonicalize to one of the actor's
        feasible options when a set is given so states that mean the same vote share a class.
        A state with no tendency is its own singleton class (weighted as a distinct action)."""
        tend = (getattr(h, "expected_action_tendency", "")
                or getattr(h, "action_if_state", "")).strip()
        if feasible_options and tend:
            from swm.world_model_v2.lean_v2.canonical_options import normalize_option
            c = normalize_option(tend, list(feasible_options))
            if c is not None:
                return c.canonical_option_id
        if tend:
            return norm_key(tend)
        return f"__state__{h.state_id}"

    def _allocate_by_action_class(self, actor_id, survivors, matched, intervals, provs, *,
                                  shared_world=None, feasible_options=None) -> dict:
        """Group survivors by action class, build a counted partial-pooled `ActorActionBaseline`
        over those classes (conditional on the shared world via typed alignment), give each class
        its baseline mass, and split a class's mass among its states (proportional to their
        counted rate, else equally within the tendency). Returns {state_id: weight} summing to 1.
        The within-class split never changes a class total — so story count cannot move the
        forecast. Provenance is recorded for states the counted matcher did not already claim."""
        from swm.world_model_v2.lean_v2.action_baseline import ActionCase, build_action_baseline
        cls_of = {h.state_id: self._action_class_of(h, feasible_options) for h in survivors}
        classes: list = []
        for h in survivors:
            if cls_of[h.state_id] not in classes:
                classes.append(cls_of[h.state_id])
        rep_state = {}                              # one representative state per action class
        for h in survivors:
            rep_state.setdefault(cls_of[h.state_id], h)
        # which counted class matched which state in the token matcher (for the untyped path)
        key_to_state = {provs[s].get("key"): s for s in matched if provs.get(s, {}).get("key")}
        # DIRECT counted evidence per action class — typed (`action_option_id`) FIRST so a
        # counted rate reaches its action class without depending on prose token overlap; else
        # the class the token matcher bound it to. This is the robust seed for the baseline.
        direct: dict = {}
        for tbl in self.actor_classes.get(actor_id, []):
            prov = tbl.get("provenance") or {}
            den = prov.get("denominator") or 0
            rate = prov.get("rate_mean")
            if den <= 0 or rate is None:
                continue
            # RAW counts (numerator / denominator), so the baseline's single Jeffreys prior
            # reproduces the counted beta-binomial rate rather than shrinking it a second time
            num = prov.get("numerator")
            num = round(float(rate) * den) if num is None else num
            num = max(0, min(int(den), int(num)))
            aoi = str(tbl.get("action_option_id") or "").strip()
            ac = None
            if aoi:
                ac = self._action_class_of(
                    ActorStateHypothesis(actor_id=actor_id, state_id="_",
                                         action_if_state=aoi), feasible_options)
            elif tbl.get("key") in key_to_state:
                ac = cls_of.get(key_to_state[tbl["key"]])
            if ac is None or ac not in classes:
                continue
            lvl = prov.get("hierarchy_level") or "broad_human_decision_class"
            ctx = dict(getattr(rep_state.get(ac), "aligned_condition", {}) or {})
            direct.setdefault(ac, []).append((int(num), int(den), lvl, ctx))
        cases: list = []
        for ac, evs in direct.items():
            for num, den, lvl, ctx in evs:
                cases.append(ActionCase(ac, lvl, ctx, weight=float(num)))   # raw positive count
        # binary complement: when exactly two action classes and only ONE carries direct counted
        # evidence, the other inherits the complement (denominator - numerator) — "the rest chose
        # the other option". With both (or neither) counted, each class stands on its own count.
        if len(classes) == 2:
            with_ev = [c for c in classes if direct.get(c)]
            without = [c for c in classes if not direct.get(c)]
            if len(with_ev) == 1 and len(without) == 1:
                for num, den, lvl, ctx in direct[with_ev[0]]:
                    cases.append(ActionCase(without[0], lvl, ctx, weight=float(den - num)))
        baseline = build_action_baseline(actor_id, "actor_decision", classes, cases,
                                         condition_state=shared_world or {})
        by_class: dict = {}
        for h in survivors:
            by_class.setdefault(cls_of[h.state_id], []).append(h)
        raw: dict = {}
        for ac, members in by_class.items():
            cmass = baseline.mass(ac)
            wr = {m.state_id: (matched.get(m.state_id) or 0.0) for m in members}
            tot = sum(wr.values())
            for m in members:
                within = (wr[m.state_id] / tot) if tot > 0 else (1.0 / len(members))
                raw[m.state_id] = cmass * within
                if m.state_id not in matched:      # the counted matcher already set matched provs
                    ci = baseline.interval(ac)
                    intervals[m.state_id] = (round(ci[0] * within, 4), round(ci[1] * within, 4))
                    provs[m.state_id] = {
                        "source": ("action_baseline_disclosed_uniform" if baseline.disclosed_uniform
                                   else "action_baseline_counted"),
                        "action_class": ac, "class_mass": round(cmass, 4),
                        "within_class_share": round(within, 4),
                        "levels_used": baseline.levels_used,
                        "note": "mass allocated to the action tendency (D8), never story count"}
        z = sum(raw.values()) or 1.0
        self._last_action_baselines = getattr(self, "_last_action_baselines", {})
        self._last_action_baselines[actor_id] = baseline.as_dict()
        return {k: v / z for k, v in raw.items()}

    def _match_class(self, h: ActorStateHypothesis, classes: dict) -> dict | None:
        if h.reference_class_key and h.reference_class_key in classes:
            return classes[h.reference_class_key]
        # deterministic best-effort match: the class whose quantity shares the most tokens
        # with the state's claim/action (no fuzzy scoring beyond token overlap)
        htoks = set(norm_key(h.claim).split()) | set(norm_key(h.action_if_state).split())
        best, best_ov = None, 0
        for tbl in classes.values():
            ov = len(htoks & set(norm_key(tbl.get("quantity")).split()))
            if ov > best_ov:
                best, best_ov = tbl, ov
        return best if best_ov >= 1 else None

    def manifest(self) -> dict:
        return {"version": STATES_VERSION, "provenance_log": self.provenance_log,
                "shared_conditions": sorted(self.shared.keys()),
                "actors_with_classes": sorted(self.actor_classes.keys())}


# ------------------------------------------------------------------ state generation call
_STATE_GEN_SCHEMA = """{"actors": [{"actor_id": "<id>", "states": [{
  "state_id": "<snake_case>", "claim": "<the private reality, qualitative>",
  "beliefs": [], "goals": [], "commitments": [], "pressures": "", "stances": [],
  "relationships": {}, "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
  "historical_case_refs": [], "distinguishing_observations": [],
  "action_if_state": "<what this actor would DO under this state>",
  "reversal_capable": false, "assumptions": [], "transition_triggers": [],
  "aligned_condition": {"<shared_condition_id>": "<condition_state this state goes with>"}}]}]}"""

_STATE_GEN_PROMPT = """Propose the genuinely DIFFERENT private realities each decisive actor could be in,
as of {as_of}. You describe WHICH realities are possible; you do NOT say how probable they are.

Question: {question}
Actors: {actors}
Shared world conditions in play: {conditions}
EVIDENCE:
{evidence}

Rules:
- For each actor, 1-3 states that would lead to MATERIALLY DIFFERENT actions. Ground each in
  evidence ids / historical case references where possible.
- Mark reversal_capable=true for any state that could flip the final answer.
- aligned_condition: if a state is more consistent with a particular shared-condition state,
  say which (this is how correlation enters — not a number).
- ABSOLUTELY NO probabilities, weights, percentages, likelihoods, or numeric scores anywhere.
  The weights are counted separately. A number here is an error.
- Start your reply with '{{'.

Reply ONLY with JSON:
{schema}"""


def generate_actor_states(*, question: str, as_of: str, evidence_text: str, actors: list,
                          shared_condition_ids: list, gateway, cache) -> tuple:
    """ONE state-generation call → per-actor `ActorStateHypothesis` lists. Numeric weights in
    the output are rejected + recorded. Returns (states_by_actor, numeric_rejections, meta);
    meta carries {from_cache, deps} so the completeness invariant can INVALIDATE a cached
    artifact that turned out empty/incomplete (cache correctness: an empty, unparseable or
    truncated result is NEVER cached, and a cached artifact proven inadequate is purged)."""
    from swm.engine.grounding import parse_json
    deps = {"question": norm(question, 300), "as_of": str(as_of)[:10],
            "actors": sorted(a["id"] for a in actors),
            "evidence_hash": hashlib.sha256(norm(evidence_text, 80000).encode())
            .hexdigest()[:20], "backend": gateway.backend_fingerprint, "v": STATES_VERSION}
    cached = cache.get("actor_state_generation", deps)
    text = cached
    meta = {"from_cache": cached is not None, "deps": deps}
    if text is None:
        prompt = _STATE_GEN_PROMPT.format(
            question=question, as_of=str(as_of)[:10],
            actors=", ".join(f"{a['id']} ({a.get('role', '')})" for a in actors),
            conditions=", ".join(shared_condition_ids) or "(none)",
            evidence=evidence_text[:2200], schema=_STATE_GEN_SCHEMA)
        text = gateway.call("state_generation", prompt)
    r = parse_json(text)
    if not isinstance(r, dict):
        return {}, [{"error": "state generation not a JSON object (never cached)"}], meta
    states_by_actor: dict = {}
    rejections = []
    valid_actor_ids = {a["id"] for a in actors}
    for a in r.get("actors") or []:
        aid = str(a.get("actor_id") or "")
        if aid not in valid_actor_ids:
            continue
        for s in a.get("states") or []:
            if not isinstance(s, dict):
                continue
            rej = reject_numeric_state_weights(s)
            if rej:
                rejections.append({"actor_id": aid, "state_id": s.get("state_id"),
                                   "rejected": rej})
            h = ActorStateHypothesis(
                actor_id=aid, state_id=str(s.get("state_id") or f"s{len(states_by_actor.get(aid, []))}"),
                claim=norm(s.get("claim"), 300),
                beliefs=[norm(b, 160) for b in (s.get("beliefs") or [])][:4],
                goals=[norm(g, 160) for g in (s.get("goals") or [])][:4],
                commitments=[norm(c, 160) for c in (s.get("commitments") or [])][:4],
                pressures=norm(s.get("pressures"), 200),
                relationships={norm_key(k): norm(v, 120)
                               for k, v in (s.get("relationships") or {}).items()},
                stances=[norm(st, 160) for st in (s.get("stances") or [])][:4],
                supporting_evidence_ids=[str(e) for e in
                                         (s.get("supporting_evidence_ids") or [])][:8],
                contradicting_evidence_ids=[str(e) for e in
                                            (s.get("contradicting_evidence_ids") or [])][:8],
                historical_case_refs=[norm(c, 120) for c in
                                      (s.get("historical_case_refs") or [])][:8],
                distinguishing_observations=[norm(o, 120) for o in
                                             (s.get("distinguishing_observations") or [])][:6],
                action_if_state=norm(s.get("action_if_state"), 200),
                reversal_capable=bool(s.get("reversal_capable")),
                assumptions=[norm(x, 160) for x in (s.get("assumptions") or [])][:6],
                transition_triggers=[norm(t, 160) for t in
                                     (s.get("transition_triggers") or [])][:6],
                aligned_condition={norm_key(k): norm(v, 80)
                                   for k, v in (s.get("aligned_condition") or {}).items()})
            states_by_actor.setdefault(aid, []).append(h)
    # cache correctness: only a NON-EMPTY parsed result may be cached — an empty artifact
    # would silently replay the failure on every future run
    if cached is None and any(states_by_actor.values()):
        cache.put("actor_state_generation", deps, text)
    return states_by_actor, rejections, meta
