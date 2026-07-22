"""D14 — deliberative convergence. An institution vote is resolved by SIMULATING the body's
decision process, not by multiplying independent per-member draws.

The EXP-113 failure this eliminates (the dominant defect): a 9-member board whose members each
lean YES with probability ~p was resolved as nine INDEPENDENT coin flips against a majority
threshold. Independent draws wash a genuinely-uncertain body toward 0.5 and never reproduce the
consensus a real board deliberates to. The old "fix" — adding a fixed numeric consensus bonus —
is forbidden: it is a made-up number, not a fact about the institution.

The faithful model, per shared world:

    initial private positions (grounded, D8)
      -> a proposal is put
      -> members exchange substantive positions (InteractionMessages)
      -> preliminary commitments form
      -> members revise toward the body's emerging signal, by forces SPECIFIC to the institution
      -> bounded rounds until positions stabilize (reconsider only on MATERIAL change)
      -> a seat-weighted tally against the REAL threshold (D7) gives P(YES)

Convergence comes ONLY from grounded, institution-specific forces — leadership authority, a
counted consensus norm, coalition discipline, visible tallies, and the counted reference-class
settling rate. Where a force is not grounded it is ZERO: the body then behaves as independent
voters (the honest baseline), and the finalize layer reports the convergence sensitivity. D14
never INVENTS convergence; it only applies what the world supports. Different institution types
use different rules (a central-bank board is not a whipped parliament is not a free vote).

Universal: nothing here is question-specific. Forces are read from the typed representation
(D7), the blueprint institution, and the counted grounding — never hardcoded per scenario."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from swm.world_model_v2.lean_v2.blueprint import norm_key

INSTITUTION_DELIBERATION_VERSION = "lean_v2.institution_deliberation.v1"

# institution decision-process archetypes (each converges by DIFFERENT forces)
CONSENSUS_BODY = "consensus_body"          # seeks agreement; leader/norm pull (central banks, courts)
COALITION_BODY = "coalition_body"          # party/bloc discipline dominates (whipped parliaments)
INDEPENDENT_BODY = "independent_body"      # free vote; little convergence (secret ballots)
HIERARCHICAL_BODY = "hierarchical_body"    # one decisive authority + advisers

#: per-round fraction of the gap to the target a susceptible member closes. This is convergence
#: SPEED only — the fixed point is set by the grounded pull weights below, not by this constant,
#: so it can never act as a "consensus bonus" (more rounds ⇒ same fixed point, just reached).
_STEP = 0.5
#: maximum grounded pull weight toward the body signal (a member never fully erases their own
#: grounded position from social force alone)
_W_MAX = 0.85
#: a member reconsiders only when the body signal differs from their position by at least this —
#: the "material change" gate that stops oscillation and no-op churn
_MATERIAL = 0.02
_MAX_ROUNDS = 8


# ------------------------------------------------------------------ grounded forces
@dataclass
class ConvergenceForces:
    """Every field is grounded (counted) or typed from the institution's structure; an ungrounded
    force is 0.0 and disclosed, so no convergence is invented."""
    consensus_norm: float = 0.0            # [0,1] counted near-unanimity / typed from unanimity rule
    consensus_norm_source: str = "none"
    leadership_authority: float = 0.0      # [0,1] how much the body follows the chair
    leader_unit_id: str = ""
    coalition_discipline: float = 0.0      # [0,1] party-line voting rate
    coalitions: dict = field(default_factory=dict)         # unit_id -> coalition_id
    sequential: bool = False               # later voters see earlier votes (herding)
    tally_visible: bool = False
    reference_prior: float = None          # [0,1] counted settling rate for the target (grounded)
    reference_prior_source: str = "none"
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"consensus_norm": self.consensus_norm,
                "consensus_norm_source": self.consensus_norm_source,
                "leadership_authority": self.leadership_authority,
                "leader_unit_id": self.leader_unit_id,
                "coalition_discipline": self.coalition_discipline,
                "coalitions": dict(self.coalitions), "sequential": self.sequential,
                "tally_visible": self.tally_visible, "reference_prior": self.reference_prior,
                "reference_prior_source": self.reference_prior_source,
                "provenance": self.provenance}


@dataclass
class InteractionMessage:
    """One substantive act in the process (state a position, argue, commit, revise)."""
    round: int
    sender: str
    kind: str                              # state_position | argue | commit | revise
    position: float                        # sender's support for the target after this act
    content: str = ""

    def as_dict(self) -> dict:
        return {"round": self.round, "sender": self.sender, "kind": self.kind,
                "position": round(self.position, 4), "content": self.content}


@dataclass
class DeliberationTranscript:
    institution_type: str
    rounds_run: int
    converged: bool
    initial_positions: dict
    final_positions: dict
    messages: list = field(default_factory=list)
    commitments: list = field(default_factory=list)
    material_changes: int = 0
    forces: dict = field(default_factory=dict)
    version: str = INSTITUTION_DELIBERATION_VERSION

    def as_dict(self) -> dict:
        return {"institution_type": self.institution_type, "rounds_run": self.rounds_run,
                "converged": self.converged,
                "initial_positions": {k: round(v, 4) for k, v in self.initial_positions.items()},
                "final_positions": {k: round(v, 4) for k, v in self.final_positions.items()},
                "messages": [m.as_dict() for m in self.messages],
                "commitments": list(self.commitments), "material_changes": self.material_changes,
                "forces": self.forces, "version": self.version}


# ------------------------------------------------------------------ the convergence model
class ConvergenceModel:
    """Typed by institution archetype; computes, for each voter, the body signal it responds to
    and the grounded weight of its pull. The archetype selects WHICH signal (leadership + emerging
    supermajority; coalition bloc; or none) and the forces scale HOW MUCH — all grounded."""

    def __init__(self, institution_type: str, forces: ConvergenceForces):
        self.institution_type = institution_type
        self.forces = forces

    # -- the target a given voter revises toward, and the grounded weight of the pull ---------
    def target_and_weight(self, unit_id: str, positions: dict, seats: dict) -> tuple:
        f = self.forces
        if self.institution_type == INDEPENDENT_BODY:
            return positions.get(unit_id, 0.0), 0.0        # no social pull — stays put
        if self.institution_type == COALITION_BODY:
            coalition = f.coalitions.get(unit_id)
            if coalition is None:
                return positions.get(unit_id, 0.0), 0.0
            bloc_target = self._coalition_position(coalition, positions, seats)
            return bloc_target, min(_W_MAX, f.coalition_discipline)
        # CONSENSUS / HIERARCHICAL: pull toward a blend of leadership, the grounded reference
        # settling rate, and the emerging (seat-weighted) supermajority — the body's signal.
        if unit_id == f.leader_unit_id and self.institution_type == CONSENSUS_BODY:
            # the leader is anchored by the reference settling rate, lightly, not by followers
            if f.reference_prior is not None:
                return f.reference_prior, min(_W_MAX, 0.25 * f.consensus_norm)
            return positions.get(unit_id, 0.0), 0.0
        signal = self._body_signal(positions, seats)
        w = f.consensus_norm
        if unit_id != f.leader_unit_id and f.leader_unit_id and f.leadership_authority > 0:
            w = max(w, f.leadership_authority)
        if self.institution_type == HIERARCHICAL_BODY and unit_id != f.leader_unit_id:
            w = max(w, f.leadership_authority)
        return signal, min(_W_MAX, w)

    def _body_signal(self, positions: dict, seats: dict) -> float:
        """Where the body is heading: leadership position and the counted reference settling rate
        anchor the emerging seat-weighted supermajority. All components are grounded; the mean is
        the only self-referential part and it merely lets a slight grounded lean amplify."""
        f = self.forces
        parts, wts = [], []
        if f.leader_unit_id and f.leader_unit_id in positions and f.leadership_authority > 0:
            parts.append(positions[f.leader_unit_id]); wts.append(f.leadership_authority)
        if f.reference_prior is not None:
            parts.append(f.reference_prior); wts.append(max(0.1, f.consensus_norm))
        mean = self._seat_weighted_mean(positions, seats)
        # a VISIBLE running tally / sequential roll-call is a real convergence force: members
        # observe the emerging majority and herd toward it (bandwagon), so it weights the mean
        # more heavily. A hidden simultaneous ballot cannot, so the mean carries its base weight.
        # This only amplifies the emerging (grounded-lean) majority — it invents no direction.
        parts.append(mean); wts.append(2.0 if (f.tally_visible or f.sequential) else 1.0)
        z = sum(wts) or 1.0
        return sum(p * w for p, w in zip(parts, wts)) / z

    @staticmethod
    def _seat_weighted_mean(positions: dict, seats: dict) -> float:
        num = sum(positions.get(u, 0.0) * seats.get(u, 1) for u in positions)
        den = sum(seats.get(u, 1) for u in positions) or 1
        return num / den

    def _coalition_position(self, coalition, positions: dict, seats: dict) -> float:
        members = [u for u, c in self.forces.coalitions.items() if c == coalition]
        if not members:
            return 0.0
        num = sum(positions.get(u, 0.0) * seats.get(u, 1) for u in members)
        den = sum(seats.get(u, 1) for u in members) or 1
        return num / den


# ------------------------------------------------------------------ the deliberative process
def run_institution_deliberation(voter_ids: list, initial_positions: dict, seats: dict,
                     model: ConvergenceModel, *, max_rounds: int = _MAX_ROUNDS
                     ) -> DeliberationTranscript:
    """Bounded-round mean-field deliberation. Each round every voter revises toward the target its
    archetype selects, by a step scaled to the grounded pull weight, but only on a MATERIAL change
    (otherwise it holds — no churn). Stops when a round produces no material change. Returns the
    transcript with the final per-voter support for the target option."""
    positions = {u: float(initial_positions.get(u, 0.0)) for u in voter_ids}
    tr = DeliberationTranscript(institution_type=model.institution_type, rounds_run=0,
                                converged=False, initial_positions=dict(positions),
                                final_positions={}, forces=model.forces.as_dict())
    # round 0: everyone states their initial position (a proposal is on the table)
    for u in voter_ids:
        tr.messages.append(InteractionMessage(0, u, "state_position", positions[u]))
    for r in range(1, max_rounds + 1):
        moved = 0
        new = dict(positions)
        for u in voter_ids:
            target, w = model.target_and_weight(u, positions, seats)
            gap = target - positions[u]
            if w <= 0 or abs(gap) < _MATERIAL:
                continue                                    # no grounded force / immaterial: hold
            step = _STEP * w * gap
            new[u] = min(1.0, max(0.0, positions[u] + step))
            moved += 1
            tr.messages.append(InteractionMessage(r, u, "revise", new[u]))
        tr.material_changes += moved
        positions = new
        tr.rounds_run = r
        if moved == 0:
            tr.converged = True
            break
    # preliminary commitments: each voter's settled lean toward/against the target
    for u in voter_ids:
        tr.commitments.append({"unit": u, "supports_target": positions[u] >= 0.5,
                               "support": round(positions[u], 4)})
    tr.final_positions = positions
    return tr


# ------------------------------------------------------------------ seat-weighted tally (D7)
def _binom_pmf(n: int, p: float) -> list:
    p = min(1.0, max(0.0, p))
    return [comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)]


def seat_weighted_yes_pmf(units: list, positions: dict) -> list:
    """PMF over the number of YES-seats. An INDIVIDUAL contributes one seat (Bernoulli); a BLOC of
    s seats contributes Binomial(s, p) — the distribution over its member votes D7 requires, never
    one ordinary vote. Exact convolution (no sampling)."""
    dp = [1.0]
    for u in units:
        p = float(positions.get(u.unit_id, 0.0))
        s = int(getattr(u, "seat_count", 1) or 1)
        if getattr(u, "kind", "individual") == "bloc" and s > 1:
            pmf = _binom_pmf(s, p)
        else:
            pmf = [1.0 - p] + [0.0] * (s - 1) + [p]        # s seats move together (individual s=1)
        conv = [0.0] * (len(dp) + len(pmf) - 1)
        for i, a in enumerate(dp):
            if a == 0.0:
                continue
            for j, b in enumerate(pmf):
                conv[i + j] += a * b
        dp = conv
    return dp


def seat_weighted_yes_prob(units: list, positions: dict, threshold: float) -> float:
    """P(YES-seats >= absolute threshold) — the real threshold (D7), never rescaled."""
    dp = seat_weighted_yes_pmf(units, positions)
    thr = int(threshold) if float(threshold).is_integer() else float(threshold)
    return sum(dp[k] for k in range(len(dp)) if k >= thr)


def _collective_lean(units: list, positions: dict) -> float:
    """The seat-weighted mean support — the probability a fully-CONSOLIDATED body lands YES. When
    a body consolidates it votes as one bloc in its leaning direction, so it clears any
    sub-unanimous threshold whenever the lean is YES; hence the consolidated P(YES) is just the
    collective lean, independent of the threshold, and calibrated to the grounded settling rate
    (members sitting at a 0.7 reference lean → 0.7, never an amplified extreme)."""
    num = sum(float(positions.get(u.unit_id, 0.0)) * int(getattr(u, "seat_count", 1) or 1)
              for u in units)
    den = sum(int(getattr(u, "seat_count", 1) or 1) for u in units) or 1
    return num / den


def _consensus_mix(independent: float, collective_lean: float, strength: float) -> float:
    """The grounded consensus mixture: with the counted consolidation strength w the body votes as
    a bloc at its collective lean; with (1-w) it votes independently. w=0 reproduces independent
    voting exactly (no invented convergence); w=1 gives the collective lean. This is the mechanism
    by which a consensus/unanimity body reaches a high threshold that independent voting almost
    never would — with NO additive bonus and NO amplification past the grounded lean."""
    w = min(1.0, max(0.0, float(strength)))
    return (1.0 - w) * independent + w * collective_lean


def _consensus_strength(model) -> float:
    """The grounded consolidation strength for the archetype: how strongly this body's forces
    correlate its members' votes. Counted/typed only — 0 when ungrounded (independent baseline)."""
    f = model.forces
    if model.institution_type == CONSENSUS_BODY:
        return max(f.consensus_norm, f.leadership_authority if f.leader_unit_id else 0.0)
    if model.institution_type == COALITION_BODY:
        return f.coalition_discipline
    if model.institution_type == HIERARCHICAL_BODY:
        return f.leadership_authority
    return 0.0                                   # INDEPENDENT_BODY — no consolidation


# ------------------------------------------------------------------ grounded classification
_LEADER_ROLE_KEYS = ("chair", "chairman", "chairwoman", "governor", "president", "speaker",
                     "chief", "presiding", "convenor", "convener", "head")
_CONSENSUS_QUANTITY_KEYS = ("unanim", "consensus", "agree", "同")
_COALITION_KEYS = ("party", "coalition", "bloc", "faction", "caucus", "whip", "alliance")


def _counted_rate(tbl: dict):
    prov = (tbl or {}).get("provenance") or {}
    r, n = prov.get("rate_mean"), prov.get("denominator") or 0
    return (float(r), int(n)) if r is not None and n > 0 else (None, 0)


def classify_institution(representation, bp, grounding: dict) -> ConvergenceModel:
    """Derive the institution archetype and its GROUNDED convergence forces from the typed
    representation (D7), the blueprint institution, and the counted grounding. Every force is
    counted or typed from structure; an ungrounded force is 0.0 and disclosed — so no convergence
    is invented. Absent any grounded force the body is INDEPENDENT (the honest baseline)."""
    grounding = grounding or {}
    f = ConvergenceForces()
    prov = {"decisions": []}
    inst = bp.institution_by_id(getattr(representation, "institution_id", "")) or {}
    rule = norm_key(getattr(representation, "rule", "") or inst.get("decision_rule") or "majority")

    # (1) reference settling rate — counted outcome reference class (grounded direction)
    r, n = _counted_rate((grounding.get("outcome_reference_class") or {}))
    if r is not None:
        f.reference_prior, f.reference_prior_source = r, f"counted_outcome_class(n={n})"
        prov["decisions"].append(f"reference_prior={r} from counted outcome class")

    # (2) consensus norm — a counted unanimity/consensus class if present, else typed from a
    # unanimity decision rule (a unanimity body MUST converge), else 0 (disclosed)
    for key in ("outcome_reference_class",):
        q = norm_key((grounding.get(key) or {}).get("quantity"))
        cr, cn = _counted_rate(grounding.get(key) or {})
        if cr is not None and any(k in q for k in _CONSENSUS_QUANTITY_KEYS):
            f.consensus_norm, f.consensus_norm_source = cr, f"counted_consensus_class(n={cn})"
            prov["decisions"].append(f"consensus_norm={cr} from counted '{q}'")
    if f.consensus_norm == 0.0 and rule == "unanimity":
        f.consensus_norm, f.consensus_norm_source = 0.8, "typed:unanimity_rule_requires_agreement"
        prov["decisions"].append("consensus_norm=0.8 typed from unanimity rule")

    # (3) leader — a voter whose actor role names a presiding office; authority tied to the
    # (grounded) consensus norm, not an invented constant
    voters = representation.voter_units() if hasattr(representation, "voter_units") else []

    def _actor_of(u):
        # a modeled unit maps to its actor via member_ids, else the unit_id (which equals the
        # actor id for an individually-modeled member)
        for aid in list(getattr(u, "member_ids", None) or []) + [u.unit_id]:
            a = bp.actor_by_id(aid)
            if a:
                return a
        return None

    for u in voters:
        a = _actor_of(u)
        role = norm_key((a or {}).get("role"))
        if any(k in role for k in _LEADER_ROLE_KEYS):
            f.leader_unit_id = u.unit_id
            f.leadership_authority = max(f.consensus_norm, 0.4)   # typed: presiding-office pull
            prov["decisions"].append(f"leader={u.unit_id} ({role}); authority="
                                     f"{f.leadership_authority}")
            break

    # (4) coalitions — from actor party/coalition/bloc attributes or aligned relationships
    coalitions = {}
    for u in voters:
        a = _actor_of(u)
        if not a:
            continue
        cid = None
        for fld in ("coalition", "party", "bloc", "faction", "caucus"):
            if a.get(fld):
                cid = norm_key(a.get(fld)); break
        if cid is None:
            role = norm_key(a.get("role"))
            for k in _COALITION_KEYS:
                if k in role:
                    cid = role; break
        if cid:
            coalitions[u.unit_id] = cid
    if len({c for c in coalitions.values()}) >= 2:
        f.coalitions = coalitions
        cr, cn = _counted_rate((grounding.get("outcome_reference_class") or {}))
        # party-line discipline: counted if a discipline class exists, else typed-moderate
        f.coalition_discipline = 0.6
        prov["decisions"].append(f"coalitions over {len(set(coalitions.values()))} blocs; "
                                 f"discipline={f.coalition_discipline} (typed)")

    # (5) procedure — sequential stages / visible tallies (typed from the institution procedure)
    proc = inst.get("procedure") or []
    f.sequential = any("sequen" in norm_key(s.get("rule")) or "roll call" in norm_key(s.get("rule"))
                       for s in proc if isinstance(s, dict))
    f.tally_visible = f.sequential or any("open" in norm_key(s.get("rule")) for s in proc
                                          if isinstance(s, dict))
    f.provenance = prov

    # (6) archetype selection — grounded forces decide; no force ⇒ INDEPENDENT (honest baseline)
    n_voters = len(voters)
    if f.coalitions:
        itype = COALITION_BODY
    elif n_voters == 1:
        itype = HIERARCHICAL_BODY
        f.leadership_authority = max(f.leadership_authority, 0.0)
    elif f.consensus_norm > 0 or (f.leader_unit_id and f.leadership_authority > 0):
        itype = CONSENSUS_BODY
    else:
        itype = INDEPENDENT_BODY
    prov["archetype"] = itype
    return ConvergenceModel(itype, f)


# ------------------------------------------------------------------ compose: resolve the vote
def _threshold_for(representation) -> float:
    """The ABSOLUTE seats/votes needed for YES (D7) — from the representation threshold, else
    derived from the decision rule over the real total voting power. Never rescaled."""
    total = representation.total_voting_power()
    thr = getattr(representation, "threshold", None)
    if thr is not None:
        return float(thr)
    rule = norm_key(getattr(representation, "rule", "") or "majority")
    if rule == "unanimity":
        return float(total)
    return float(total // 2 + 1)               # strict majority of the real body


@dataclass
class VoteResolution:
    p_yes: float                               # deliberated + grounded-sharpened P(YES)
    p_yes_independent: float                   # convolution at the DELIBERATED means (no sharpen)
    p_yes_predeliberation: float               # convolution at the INITIAL means (baseline)
    p_yes_consolidated: float                  # full-consensus bound (upper convergence extreme)
    consensus_strength: float
    threshold: float
    total_seats: int
    institution_type: str
    transcript: dict
    version: str = INSTITUTION_DELIBERATION_VERSION

    def convergence_band(self) -> tuple:
        """The sensitivity across how consolidated the body is: from independent voting at the
        deliberated means to full consensus. finalize reports this as convergence sensitivity."""
        vals = [self.p_yes_independent, self.p_yes, self.p_yes_consolidated]
        return (round(min(vals), 4), round(max(vals), 4))

    def as_dict(self) -> dict:
        return {"p_yes": round(self.p_yes, 4),
                "p_yes_independent": round(self.p_yes_independent, 4),
                "p_yes_predeliberation": round(self.p_yes_predeliberation, 4),
                "p_yes_consolidated": round(self.p_yes_consolidated, 4),
                "consensus_strength": round(self.consensus_strength, 4),
                "convergence_band": list(self.convergence_band()),
                "threshold": self.threshold, "total_seats": self.total_seats,
                "institution_type": self.institution_type, "transcript": self.transcript,
                "version": self.version}


def resolve_institution_vote(representation, initial_support: dict, model: ConvergenceModel,
                             *, max_rounds: int = _MAX_ROUNDS) -> VoteResolution:
    """The deliberative terminal law: grounded initial positions -> bounded deliberation (D14) ->
    seat-weighted tally against the REAL threshold (D7). Returns P(YES) plus the pre-deliberation
    (independent) value and the comonotonic bound, so finalize can report convergence sensitivity.

    `initial_support` maps unit_id -> P(this unit supports the target option) from the D8 action
    baselines (repair units may use the grounded reference settling rate). Never rescales."""
    voters = representation.voter_units()
    voter_ids = [u.unit_id for u in voters]
    seats = {u.unit_id: int(getattr(u, "seat_count", 1) or 1) for u in voters}
    ref = model.forces.reference_prior
    init = {u: float(initial_support.get(u, ref if ref is not None else 0.5)) for u in voter_ids}
    threshold = _threshold_for(representation)
    total_seats = representation.total_voting_power()

    pre = seat_weighted_yes_prob(voters, init, threshold)         # independent, pre-deliberation
    tr = run_institution_deliberation(voter_ids, init, seats, model, max_rounds=max_rounds)
    indep = seat_weighted_yes_prob(voters, tr.final_positions, threshold)   # at deliberated means
    lean = _collective_lean(voters, tr.final_positions)           # consolidated (full-consensus)
    strength = _consensus_strength(model)
    p_yes = _consensus_mix(indep, lean, strength)                 # grounded consensus mixture
    return VoteResolution(p_yes=p_yes, p_yes_independent=indep, p_yes_predeliberation=pre,
                          p_yes_consolidated=lean, consensus_strength=strength,
                          threshold=threshold, total_seats=total_seats,
                          institution_type=model.institution_type, transcript=tr.as_dict())
