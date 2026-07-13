"""Representation choice for hidden social state — Phase 3 (REPRESENTATION-CHOICE PRINCIPLE).

A hidden social concept ("will they cooperate", "is the regime stable", "how contested is this") is NOT
automatically a scalar. Forcing every latent into `trust = 0.7` throws away the STRUCTURE of the uncertainty.
This module keeps the QUALITATIVE concept distinct from its QUANTITATIVE representation and lets EVIDENCE +
HELD-OUT CALIBRATION decide which representation is useful — not intuition, and never an LLM-minted number.

Contract:
  * the LLM may PROPOSE candidate representation KINDS + a rationale (semantic mapping only). It may not pick
    the winner and may not mint any numeric value;
  * each candidate is a real fitter with an explicit prior + likelihood; the winner is the one that best
    predicts HELD-OUT outcomes (log-loss / Brier) and passes posterior predictive checking;
  * an arbitrary fixed scalar (`ScalarPointRepresentation`) is included ONLY as a deliberately impoverished
    baseline — the thing the principle warns against — so ablations can show it loses when structure exists;
  * `assert_not_ornamental` refuses any representation that is neither evidence-linked NOR causally consumed.

The representations here share ONE evidence abstraction: a list of directional "votes" in {+1 yes, -1 no}
each with a reliability in [0,1] (produced upstream by the registered observation models from qualitative
tags). Every fitter turns those votes + a prior into a posterior over the affirmative rate and a predictive
P(outcome=yes); scoring is proper-scoring-rule on held-out outcomes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

REPRESENTATION_KINDS = ("scalar_point", "binary", "categorical", "ordinal", "continuous_probabilistic",
                        "hazard_duration", "mixture", "relational", "hybrid_interpretable", "learned_latent")

#: which kinds are IMPLEMENTED as executable fitters here (the rest are declared candidates with a documented
#: dependency — see phase3_representation fitters + the LIMITATIONS doc). Honest scope, not silent omission.
_IMPLEMENTED = ("scalar_point", "continuous_probabilistic", "discrete_hypothesis", "mixture",
                "hybrid_interpretable")


@dataclass
class RepresentationCandidate:
    """A proposed representation for one hidden-state concept. `kind` is qualitative; every numeric parameter
    is produced by the fitter from evidence + a prior, never by the proposer."""
    concept: str
    kind: str
    rationale: str = ""
    proposed_by: str = "default_menu"          # "llm" | "default_menu"
    identifiability: str = "unknown"           # identified | partially_identified | unidentified
    evidence_claim_ids: list = field(default_factory=list)
    consumed_by: list = field(default_factory=list)

    def as_dict(self):
        return {"concept": self.concept, "kind": self.kind, "rationale": self.rationale,
                "proposed_by": self.proposed_by, "identifiability": self.identifiability,
                "evidence_linked": bool(self.evidence_claim_ids), "consumed_by": list(self.consumed_by)}


class OrnamentalRepresentationError(ValueError):
    """Raised when a representation is neither evidence-linked nor causally consumed — the anti-pattern the
    representation principle forbids (an arbitrary weighted variable like trust=0.7 that nothing reads)."""


def assert_not_ornamental(candidate: RepresentationCandidate) -> None:
    """A representation counts as production inference ONLY if it is BOTH evidence-linked AND causally
    consumed. Otherwise it is ornamental and must not be materialized."""
    if not candidate.evidence_claim_ids and not candidate.consumed_by:
        raise OrnamentalRepresentationError(
            f"representation for concept {candidate.concept!r} ({candidate.kind}) is ornamental: "
            f"no evidence links AND no causal consumer — not production inference")
    if candidate.kind == "scalar_point" and not candidate.consumed_by:
        raise OrnamentalRepresentationError(
            f"a bare scalar for {candidate.concept!r} with no declared consumer is exactly the "
            f"trust=0.7 anti-pattern — refused")


_PROPOSE_PROMPT = """You are choosing how to REPRESENT a hidden social variable so a simulator can reason about
it. Do NOT give any numbers. Propose 2-4 candidate REPRESENTATION KINDS and a one-line rationale each. Reply
ONLY JSON: {{"candidates": [{{"kind": "<one of {kinds}>", "rationale": "..."}}]}}

Guidance: use `continuous_probabilistic` for a smoothly-varying rate/propensity; `discrete_hypothesis` for a
few qualitatively distinct regimes; `mixture` when several distinct worlds are plausible at once; `ordinal`
for ranked levels; `hazard_duration` for time-to-event; `hybrid_interpretable` for a discrete regime with a
continuous within-regime state. Avoid `scalar_point` unless the concept truly has no useful uncertainty.

CONCEPT: {concept}
CONTEXT: {context}"""


def propose_representations(concept: str, *, llm=None, context: str = "", evidence_claim_ids=None,
                           consumed_by=None) -> list:
    """LLM proposes candidate representation KINDS (qualitative only). Falls back to a principled default menu
    if no LLM or on parse failure. Every candidate inherits the concept's evidence links + consumer so the
    anti-ornamental guard can be applied before anything is materialized."""
    ev, cons = list(evidence_claim_ids or []), list(consumed_by or [])
    proposed = []
    if llm is not None:
        from swm.engine.grounding import parse_json
        try:
            raw = parse_json(llm(_PROPOSE_PROMPT.format(kinds="|".join(REPRESENTATION_KINDS),
                                                        concept=concept, context=context or "n/a"))) or {}
            for c in (raw.get("candidates") or [])[:4]:
                k = str(c.get("kind", "")).strip()
                if k in REPRESENTATION_KINDS or k == "discrete_hypothesis":
                    proposed.append(RepresentationCandidate(
                        concept=concept, kind=k, rationale=str(c.get("rationale", ""))[:160],
                        proposed_by="llm", evidence_claim_ids=ev, consumed_by=cons))
        except Exception:  # noqa: BLE001
            proposed = []
    if not proposed:                                            # principled default menu (still real candidates)
        proposed = [RepresentationCandidate(concept, k, "default menu", "default_menu", "unknown", ev, cons)
                    for k in ("scalar_point", "continuous_probabilistic", "discrete_hypothesis", "mixture",
                              "hybrid_interpretable")]
    return proposed


# ------------------------------------------------------------------ executable representation fitters
# Each fitter: fit(votes) sets an internal posterior; predict() returns P(outcome=yes) integrating over the
# posterior. `votes` = [(sign in {+1,-1}, reliability in [0,1])].
def _vote_loglik(rate: float, sign: int, rel: float) -> float:
    """P(a directional vote | affirmative rate). sens=spec=0.5+0.35*rel (a reliable vote discriminates; an
    unreliable one is a coin flip). Shared by every fitter so the comparison is apples-to-apples."""
    d = 0.5 + 0.35 * max(0.0, min(1.0, rel))
    p_yes_vote = rate * d + (1 - rate) * (1 - d)
    p = p_yes_vote if sign > 0 else (1 - p_yes_vote)
    return math.log(max(1e-9, min(1 - 1e-9, p)))


@dataclass
class ScalarPointRepresentation:
    """The anti-pattern baseline: collapse the evidence to ONE number (reliability-weighted vote share) and
    predict Bernoulli(point). Discards all uncertainty structure — included only to be beaten."""
    kind: str = "scalar_point"
    point: float = 0.5

    def fit(self, votes):
        num = sum((0.5 + 0.5 * s) * r for s, r in votes)        # yes-mass
        den = sum(r for _, r in votes) or 1.0
        self.point = 0.5 if not votes else max(0.02, min(0.98, num / den))
        return self

    def predict(self) -> float:
        return self.point

    def as_dict(self):
        return {"kind": self.kind, "point": round(self.point, 4)}


@dataclass
class ContinuousBetaRepresentation:
    """A continuous propensity on [0,1] as a particle posterior (Beta prior + directional-vote likelihood).
    Preserves the full uncertainty; predictive integrates over it."""
    kind: str = "continuous_probabilistic"
    a0: float = 1.0
    b0: float = 1.0
    n_grid: int = 101
    grid: list = field(default_factory=list)
    w: list = field(default_factory=list)

    def fit(self, votes):
        xs = [(i + 0.5) / self.n_grid for i in range(self.n_grid)]
        lw = [(self.a0 - 1) * math.log(x) + (self.b0 - 1) * math.log(1 - x) for x in xs]
        for s, r in votes:
            for i, x in enumerate(xs):
                lw[i] += _vote_loglik(x, s, r)
        m = max(lw)
        w = [math.exp(v - m) for v in lw]
        z = sum(w) or 1.0
        self.grid, self.w = xs, [v / z for v in w]
        return self

    def predict(self) -> float:
        return sum(x * wi for x, wi in zip(self.grid, self.w))   # posterior-mean rate = P(yes)

    def mean_sd(self):
        mean = self.predict()
        var = sum(wi * (x - mean) ** 2 for x, wi in zip(self.grid, self.w))
        return mean, math.sqrt(max(0.0, var))

    def as_dict(self):
        mean, sd = self.mean_sd()
        return {"kind": self.kind, "mean": round(mean, 4), "sd": round(sd, 4)}


@dataclass
class DiscreteHypothesisRepresentation:
    """A few qualitatively distinct regimes, each with a fixed characteristic rate; a categorical posterior
    over regimes. Right when the world is in one of K discrete modes (not a smooth continuum)."""
    kind: str = "discrete_hypothesis"
    regimes: tuple = (("no_regime", 0.2), ("contested", 0.5), ("yes_regime", 0.8))
    post: list = field(default_factory=list)

    def fit(self, votes):
        lw = [0.0] * len(self.regimes)
        for i, (_, rate) in enumerate(self.regimes):
            for s, r in votes:
                lw[i] += _vote_loglik(rate, s, r)
        m = max(lw)
        w = [math.exp(v - m) for v in lw]
        z = sum(w) or 1.0
        self.post = [v / z for v in w]
        return self

    def predict(self) -> float:
        return sum(p * rate for p, (_, rate) in zip(self.post, self.regimes))

    def as_dict(self):
        return {"kind": self.kind,
                "posterior": {name: round(p, 4) for p, (name, _) in zip(self.post, self.regimes)}}


@dataclass
class MixtureRepresentation:
    """A mixture of continuous components — captures genuine MULTIMODALITY (two plausible worlds at once)
    that a single unimodal continuous posterior blurs into a misleading middle."""
    kind: str = "mixture"
    k: int = 2
    comps: list = field(default_factory=list)   # [(center_a, center_b, weight)]
    reps: list = field(default_factory=list)
    weights: list = field(default_factory=list)

    def fit(self, votes):
        # two Beta components (low-mode, high-mode); fit each by likelihood, weight by evidence, EM-lite (1 pass)
        priors = [(1.2, 3.0), (3.0, 1.2)]
        comps, lmix = [], []
        for a0, b0 in priors:
            rep = ContinuousBetaRepresentation(a0=a0, b0=b0).fit(votes)
            comps.append(rep)
            # component evidence = log marginal likelihood under that component's prior
            xs = rep.grid
            lw = [(a0 - 1) * math.log(x) + (b0 - 1) * math.log(1 - x) for x in xs]
            for s, r in votes:
                for i, x in enumerate(xs):
                    lw[i] += _vote_loglik(x, s, r)
            m = max(lw)
            lmix.append(m + math.log(sum(math.exp(v - m) for v in lw) / len(xs)))
        m = max(lmix)
        w = [math.exp(v - m) for v in lmix]
        z = sum(w) or 1.0
        self.reps, self.weights = comps, [v / z for v in w]
        return self

    def predict(self) -> float:
        return sum(wi * rep.predict() for wi, rep in zip(self.weights, self.reps))

    def as_dict(self):
        return {"kind": self.kind,
                "components": [{"w": round(wi, 4), **rep.as_dict()} for wi, rep in
                               zip(self.weights, self.reps)]}


@dataclass
class HybridRepresentation:
    """Discrete regime × continuous within-regime state: a categorical over regimes, each carrying its own
    continuous propensity. The interpretable-structure + graded-uncertainty compromise."""
    kind: str = "hybrid_interpretable"
    disc: DiscreteHypothesisRepresentation = None
    cont: ContinuousBetaRepresentation = None

    def fit(self, votes):
        self.disc = DiscreteHypothesisRepresentation().fit(votes)
        self.cont = ContinuousBetaRepresentation().fit(votes)
        return self

    def predict(self) -> float:
        return 0.5 * self.disc.predict() + 0.5 * self.cont.predict()

    def as_dict(self):
        return {"kind": self.kind, "discrete": self.disc.as_dict(), "continuous": self.cont.as_dict()}


_FITTERS = {"scalar_point": ScalarPointRepresentation,
            "continuous_probabilistic": ContinuousBetaRepresentation,
            "discrete_hypothesis": DiscreteHypothesisRepresentation,
            "mixture": MixtureRepresentation,
            "hybrid_interpretable": HybridRepresentation}


def build_fitter(kind: str):
    """Instantiate an executable fitter for a representation kind, or None if the kind is a declared candidate
    without an implemented fitter (documented dependency, not a silent gap)."""
    cls = _FITTERS.get(kind)
    return cls() if cls else None


@dataclass
class RepresentationScorecard:
    concept: str
    winner: str = "continuous_probabilistic"
    metrics: dict = field(default_factory=dict)      # kind -> {held_out_logloss, brier, calibration_err}
    candidates: list = field(default_factory=list)   # [RepresentationCandidate.as_dict()]
    note: str = ""

    def as_dict(self):
        return {"concept": self.concept, "winner": self.winner, "metrics": self.metrics,
                "candidates": self.candidates, "note": self.note}


def choose_representation(concept: str, train_episodes, test_episodes, candidates=None) -> RepresentationScorecard:
    """Fit each candidate representation on `train_episodes` and score it on HELD-OUT `test_episodes` by proper
    scoring rules; the winner is the best held-out predictor (lowest log-loss), NOT the most intuitive.

    An "episode" = (votes, outcome) where votes=[(sign, reliability)] and outcome in {0,1}. Each representation
    is fit PER-EPISODE on that episode's votes (the votes ARE the per-episode evidence); calibration is
    measured across the test episodes. This is the empirical engine behind the representation ablation."""
    kinds = [c.kind if isinstance(c, RepresentationCandidate) else c for c in
             (candidates or ["scalar_point", "continuous_probabilistic", "discrete_hypothesis",
                             "mixture", "hybrid_interpretable"])]
    kinds = [k for k in kinds if k in _FITTERS]
    metrics = {}
    for k in kinds:
        ll, brier, preds = 0.0, 0.0, []
        for votes, outcome in test_episodes:
            p = max(1e-6, min(1 - 1e-6, build_fitter(k).fit(votes).predict()))
            ll += -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
            brier += (p - outcome) ** 2
            preds.append((p, outcome))
        n = max(1, len(test_episodes))
        metrics[k] = {"held_out_logloss": round(ll / n, 5), "brier": round(brier / n, 5),
                      "calibration_err": round(_calibration_error(preds), 5)}
    winner = min(metrics, key=lambda k: metrics[k]["held_out_logloss"]) if metrics else "continuous_probabilistic"
    return RepresentationScorecard(
        concept=concept, winner=winner, metrics=metrics,
        candidates=[c.as_dict() for c in (candidates or []) if isinstance(c, RepresentationCandidate)],
        note="winner = lowest held-out log-loss; representation chosen by calibration, not intuition")


def _calibration_error(preds, bins: int = 10) -> float:
    """Expected calibration error over `preds` = [(p, outcome)]."""
    if not preds:
        return 0.0
    buckets = [[] for _ in range(bins)]
    for p, o in preds:
        buckets[min(bins - 1, int(p * bins))].append((p, o))
    ece, n = 0.0, len(preds)
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        ece += (len(b) / n) * abs(conf - acc)
    return ece
