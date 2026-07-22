"""D8 — action-grounded weighting. The probability an actor takes an action is a COUNTED,
partial-pooled rate over ACTION CLASSES, never the number of prose stories generated for that
actor and never an automatic equal split.

The EXP-113 failure this eliminates: `weight_actor_states` fell back to `residual/len(states)` —
so three generated "hold" stories and one "raise" story made hold ≈ 0.75 purely because the LLM
wrote more hold stories. State/story COUNT determined probability. That is a text-generation
artifact, not a fact about the world.

The fix, in one sentence:

    allocate probability mass to ACTION TENDENCIES first — from a counted, hierarchically
    partial-pooled distribution over the actor's feasible action classes — and only THEN split
    a tendency's mass among the private states that share it (for trajectory/sensitivity), which
    never changes the tendency's total and so never lets story count move the forecast.

`ActorActionBaseline` is that distribution. It is built by hierarchical **Dirichlet-multinomial
partial pooling** over the specificity hierarchy (same person → same role → same institution →
similar decision type → similar process → broad human decision class, most specific last): a
level with many observations dominates; a sparse level is shrunk toward its broader parent by a
fixed concentration; with no observations at all the distribution is a DISCLOSED uniform over the
action classes (a sensitivity axis that widens the interval — never an invented point
probability). It can be built marginally or CONDITIONAL on a shared-world state (typed
state↔condition alignment), by counting only the cases consistent with that world.

Universal: nothing here is question-specific. It counts cases the grounding layer supplies; it
invents no rates."""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm_key
from swm.world_model_v2.lean_v2.grounding import HIERARCHY_LEVELS

ACTION_BASELINE_VERSION = "lean_v2.action_baseline.v1"

#: parent concentration (prior sample size) at each level: how many effective observations of
#: the broader parent it takes for a more-specific level to override it. Matches the grounding
#: layer's MIN_CASES_FOR_LEVEL so "enough specific cases" means the same thing everywhere.
POOLING_TAU = 4.0

#: per-class global prior pseudocount (Jeffreys), consistent with grounding._PRIOR_A — weak and
#: symmetric; real counts dominate as they accumulate.
JEFFREYS_PER_CLASS = 0.5

#: broad → specific (the hierarchy is stored most-specific-first)
_LEVELS_BROAD_TO_SPECIFIC = tuple(reversed(HIERARCHY_LEVELS))
_LEVEL_RANK = {lvl: i for i, lvl in enumerate(HIERARCHY_LEVELS)}  # 0 = most specific


@dataclass
class ActionCase:
    """One counted historical instance of an actor(-class) choosing among the SAME action
    classes. `action_class` is a canonical action id (the caller canonicalizes); `weight` lets a
    counted binary rate contribute fractional evidence (rate·n took the class, (1−rate)·n did
    not)."""
    action_class: str
    hierarchy_level: str = "broad_human_decision_class"
    condition_context: dict = field(default_factory=dict)   # {condition_id: condition_state}
    weight: float = 1.0
    basis_quote: str = ""
    date: str = ""

    def as_dict(self) -> dict:
        return {"action_class": self.action_class, "hierarchy_level": self.hierarchy_level,
                "condition_context": dict(self.condition_context), "weight": self.weight,
                "basis_quote": self.basis_quote, "date": self.date}


@dataclass
class ActorActionBaseline:
    """A grounded distribution over an actor's feasible ACTION CLASSES for one decision. The mass
    on each class is the count-based, partial-pooled rate — the thing D8 requires the forecast to
    use INSTEAD of story count."""
    actor_id: str
    decision_id: str
    action_classes: list                              # canonical class ids, order preserved
    class_mass: dict = field(default_factory=dict)    # {class_id: probability}, sums to 1
    class_interval: dict = field(default_factory=dict)  # {class_id: (lo, hi)}
    concentration: float = 0.0                        # effective Dirichlet sample size
    disclosed_uniform: bool = False                   # no counts → disclosed sensitivity axis
    n_total: float = 0.0                              # effective observations counted
    levels_used: list = field(default_factory=list)
    condition_state: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    version: str = ACTION_BASELINE_VERSION

    def mass(self, action_class: str) -> float:
        return float(self.class_mass.get(action_class, 0.0))

    def interval(self, action_class: str) -> tuple:
        return tuple(self.class_interval.get(action_class, (0.0, 1.0)))

    def top(self) -> str:
        return max(self.class_mass, key=self.class_mass.get) if self.class_mass else ""

    def entropy(self) -> float:
        from math import log
        return round(-sum(p * log(p) for p in self.class_mass.values() if p > 0), 4)

    def as_dict(self) -> dict:
        return {"actor_id": self.actor_id, "decision_id": self.decision_id,
                "action_classes": list(self.action_classes),
                "class_mass": {k: round(v, 4) for k, v in self.class_mass.items()},
                "class_interval": {k: [round(x, 4) for x in v]
                                   for k, v in self.class_interval.items()},
                "concentration": round(self.concentration, 3),
                "disclosed_uniform": self.disclosed_uniform, "n_total": round(self.n_total, 3),
                "levels_used": list(self.levels_used), "condition_state": dict(self.condition_state),
                "entropy": self.entropy(), "provenance": self.provenance, "version": self.version}


# ------------------------------------------------------------------ the pooling core
def partial_pool_categorical(action_classes: list, counts_by_level: dict, *,
                             tau: float = POOLING_TAU) -> tuple:
    """Hierarchical Dirichlet-multinomial partial pooling over the specificity levels.

    `counts_by_level` maps a hierarchy level → {class_id: count}. Processing broad → specific,
    each level's posterior mean is the observation shrunk toward its parent with strength `tau`:

        mean_level[k] = (tau · parent_mean[k] + n_level[k]) / (tau + N_level)

    A level with no observations passes the parent through unchanged; the initial parent is the
    symmetric Jeffreys prior (a disclosed uniform). Returns (mean: {class_id: p}, concentration,
    n_total, levels_used) — `concentration` is tau + Σ observed counts (the effective Dirichlet
    sample size, used for credible intervals)."""
    K = len(action_classes)
    if K == 0:
        return {}, 0.0, 0.0, []
    # initial parent = symmetric Jeffreys prior (disclosed uniform, weak)
    parent = {c: 1.0 / K for c in action_classes}
    n_total = 0.0
    levels_used = []
    for lvl in _LEVELS_BROAD_TO_SPECIFIC:
        raw = counts_by_level.get(lvl) or {}
        n = {c: float(raw.get(c, 0.0)) for c in action_classes}
        N = sum(n.values())
        if N <= 0:
            continue
        levels_used.append(lvl)
        n_total += N
        parent = {c: (tau * parent[c] + n[c]) / (tau + N) for c in action_classes}
    concentration = tau + n_total
    # normalize defensively (float drift)
    z = sum(parent.values()) or 1.0
    mean = {c: parent[c] / z for c in action_classes}
    return mean, concentration, n_total, levels_used


def _beta_interval(mean: float, concentration: float) -> tuple:
    """Per-class marginal credible band: the Dirichlet marginal for a class is Beta(a, A−a) with
    a = mean·concentration. Reuses the grounding layer's symmetric-band convention so a sparse
    baseline is visibly wide (matching _beta_binomial)."""
    a = max(1e-6, mean * concentration)
    b = max(1e-6, concentration - a)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    half = 1.645 * (var ** 0.5)
    return (round(max(0.0, mean - half), 4), round(min(1.0, mean + half), 4))


# ------------------------------------------------------------------ build
def _consistent_with_world(case: ActionCase, condition_state: dict) -> bool:
    """A case counts toward a conditional baseline when its declared condition context does not
    CONTRADICT the given world. A case with no context is world-agnostic (counts everywhere); a
    case whose context names a condition also named in `condition_state` must AGREE on it."""
    if not condition_state or not case.condition_context:
        return True
    for cid, want in case.condition_context.items():
        have = condition_state.get(norm_key(cid))
        if have is not None and norm_key(have) != norm_key(want):
            return False
    return True


def build_action_baseline(actor_id: str, decision_id: str, action_classes: list, cases: list,
                          *, condition_state: dict = None, tau: float = POOLING_TAU
                          ) -> ActorActionBaseline:
    """Build the counted, partial-pooled action distribution. `cases` are `ActionCase`s (a class
    label + hierarchy level + optional world context). With `condition_state`, only cases
    consistent with that world are counted (typed conditional baseline). With no usable case the
    baseline is a DISCLOSED uniform over the action classes — a sensitivity axis, never a point
    probability derived from story count."""
    classes = [str(c) for c in action_classes]
    condition_state = {norm_key(k): v for k, v in (condition_state or {}).items()}
    baseline = ActorActionBaseline(actor_id=actor_id, decision_id=decision_id,
                                   action_classes=classes, condition_state=condition_state)
    if not classes:
        baseline.provenance = {"note": "no feasible action classes"}
        return baseline
    # bucket cases by level, dropping those inconsistent with the conditional world
    counts_by_level: dict = {}
    kept, dropped = 0, 0
    for c in cases or []:
        if str(c.action_class) not in classes:
            dropped += 1
            continue
        if not _consistent_with_world(c, condition_state):
            dropped += 1
            continue
        lvl = c.hierarchy_level if c.hierarchy_level in _LEVEL_RANK else "broad_human_decision_class"
        counts_by_level.setdefault(lvl, {})
        counts_by_level[lvl][c.action_class] = counts_by_level[lvl].get(c.action_class, 0.0) \
            + max(0.0, float(c.weight))
        kept += 1
    mean, concentration, n_total, levels_used = partial_pool_categorical(
        classes, counts_by_level, tau=tau)
    baseline.class_mass = {c: round(mean.get(c, 0.0), 6) for c in classes}
    baseline.concentration = concentration
    baseline.n_total = n_total
    baseline.levels_used = levels_used
    if n_total <= 0:
        # no counted evidence — DISCLOSED uniform over the action classes (max entropy), wide
        # intervals; this is a sensitivity axis the finalize layer widens on, NOT a probability
        # asserted from the number of stories.
        baseline.disclosed_uniform = True
        baseline.class_interval = {c: (0.0, 1.0) for c in classes}
        baseline.provenance = {"source": "disclosed_uniform_no_count",
                               "note": "no counted action cases — uniform over action classes as "
                                       "a disclosed sensitivity axis, never story-count derived",
                               "cases_kept": kept, "cases_dropped": dropped,
                               "n_classes": len(classes)}
        return baseline
    baseline.class_interval = {c: _beta_interval(baseline.class_mass[c], concentration)
                               for c in classes}
    baseline.provenance = {"source": "counted_partial_pooled",
                           "levels_used": levels_used, "tau": tau,
                           "n_total": round(n_total, 3), "concentration": round(concentration, 3),
                           "cases_kept": kept, "cases_dropped": dropped,
                           "conditional_on": dict(condition_state) or None,
                           "law": "Dirichlet-multinomial hierarchical shrinkage; specificity "
                                  "dominates with count, sparse levels shrink to parent"}
    return baseline


# ------------------------------------------------------------------ deriving cases from grounding
def cases_from_counted_class(tbl: dict, action_class: str, *, complement_class: str = None,
                             condition_context: dict = None) -> list:
    """Turn one counted binary reference class (grounding `ReferenceClassTable.as_dict()`) into
    ActionCases: rate·n instances took `action_class`, (1−rate)·n took `complement_class` (when a
    single complementary class is known). The cases inherit the table's hierarchy level so they
    pool correctly. This is how the existing counted actor classes feed the action baseline
    without any new LLM call."""
    prov = tbl.get("provenance") or {}
    rate = prov.get("rate_mean")
    n = prov.get("denominator") or 0
    lvl = prov.get("hierarchy_level") or "broad_human_decision_class"
    if rate is None or n <= 0:
        return []
    ctx = dict(condition_context or {})
    out = [ActionCase(action_class=action_class, hierarchy_level=lvl, condition_context=ctx,
                      weight=round(float(rate) * n, 4), basis_quote=tbl.get("quantity", ""))]
    if complement_class is not None:
        out.append(ActionCase(action_class=complement_class, hierarchy_level=lvl,
                              condition_context=ctx, weight=round((1.0 - float(rate)) * n, 4),
                              basis_quote=f"complement of {tbl.get('quantity', '')}"))
    return out
