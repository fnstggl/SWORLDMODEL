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
                            shared_world: dict = None) -> tuple:
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
            rate = (tbl.get("provenance") or {}).get("rate_mean")
            n = (tbl.get("provenance") or {}).get("denominator") or 0
            if rate is None or n <= 0:
                continue
            w = rate
            if shared_world and best.aligned_condition:
                for cid, want in best.aligned_condition.items():
                    have = (shared_world or {}).get(cid)
                    if have is not None and norm_key(have) != norm_key(want):
                        w = w * (1 - n / (n + 1.0))
            matched[best.state_id] = w
            intervals[best.state_id] = tbl.get("interval") or (0.0, 1.0)
            provs[best.state_id] = {"source": "counted_reference_class", "key": tbl.get("key"),
                                    "rate": rate, "n": n, "interval": intervals[best.state_id],
                                    "hierarchy_level": (tbl.get("provenance") or {})
                                    .get("hierarchy_level")}
            claimed.add(best.state_id)
        # unmatched states share the RESIDUAL of the counted rates — a grounded complement
        # (the counted classes cover the matched outcomes; what's left is the complementary
        # probability of the remaining modeled states), never a label-derived number
        matched_sum = sum(matched.values())
        unmatched = [h.state_id for h in survivors if h.state_id not in matched]
        residual = max(0.0, 1.0 - matched_sum)
        raw_weights = dict(matched)
        if unmatched:
            share = residual / len(unmatched)
            for sid in unmatched:
                raw_weights[sid] = share
                intervals[sid] = (0.0, min(1.0, residual))
                provs[sid] = {"source": "counted_complement",
                              "note": "complement of the counted matched rates (1 - "
                                      f"{matched_sum:.3f}) shared among {len(unmatched)} "
                                      "unmatched modeled state(s) — grounded, not a label"}
        n_counted = len(matched)
        reversal = any(h.reversal_capable for h in survivors)
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
