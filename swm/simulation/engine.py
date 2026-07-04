"""SimulationEngine — the actual multi-step, multi-actor simulator (Phase 2/4/6).

This is the architecture the directive demands, and it satisfies the strict definition of a
simulation: it **explicitly represents stakeholder segments**, **simulates their reactions over
multiple timesteps**, **updates the world state after each reaction** (score, exposure, social
proof, front-page flag, novelty/fatigue), and **derives the outcome probability from the
distribution of simulated trajectories** — NOT a classifier over the initial features.

HN score-formation model (one trajectory):
  t0 submit -> t1 /new exposure: each segment upvotes ∝ affinity·features · author_rep · attention
     -> update score, social proof, novelty
     -> stochastic FRONT-PAGE transition (logistic in early score): if it crosses, exposure jumps
        ~20x and casual front-page browsers flood in; if not, it decays in /new
  t2..t_n front-page cascade: social proof raises upvote propensity (bandwagon), novelty decays
  final score = accumulated upvotes across steps  →  many trajectories → P(score ≥ band)

Entity state enters as `author_rep` (the author's as-of reputation), so a proven author is more
likely to get the early velocity that crosses the front-page threshold — the mechanism by which
repeat-entity state is *supposed* to help. Context (topic salience, domain reputation) modulates
early exposure. A thin Platt `readout` calibrates the trajectory-derived P(hit); the readout only
reads the simulated outcome, it does not replace the simulation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from swm.simulation.actors import SegmentAgent, default_hn_segments
from swm.simulation.policies import PolicyParams, frontpage_prob, sample_poisson, social_proof
from swm.simulation.reactions import HeuristicReactionModel, Reaction
from swm.simulation.trajectory_state import TrajectoryState
from swm.transition.transition_head import BAND_EDGES

_LOGIT = lambda p: math.log(max(1e-6, p) / max(1e-6, 1 - p))  # noqa: E731
_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))  # noqa: E731


@dataclass
class HNSimulationEngine:
    segments: list = field(default_factory=default_hn_segments)
    reaction_model: object = field(default_factory=HeuristicReactionModel)
    params: PolicyParams = field(default_factory=PolicyParams)
    thresholds: tuple = tuple(BAND_EDGES)
    readout: tuple | None = None            # (a, b) Platt on logit(p_hit_raw); None => raw
    default_author_rep: float = 0.0

    # ---------------- one trajectory ----------------
    def _trajectory(self, feats: dict, author_rep: float, ctx: dict, rng: random.Random,
                    record: bool = False):
        p = self.params
        segs = [s.copy() for s in self.segments]
        st = TrajectoryState(segments=segs, novelty=1.0,
                             exposure_pool=p.new_page_exposure * ctx.get("exposure_mult", 1.0),
                             context=ctx)
        score = 0.0
        for step in range(p.n_steps):
            st.timestep = step + 1
            st.social_proof = social_proof(score, p)
            step_up = 0.0
            for seg in segs:
                exposed = st.exposure_pool * seg.weight * seg.attention
                if exposed <= 0:
                    continue
                pr = self.reaction_model.upvote_propensity(seg, feats, st, author_rep)
                up = sample_poisson(exposed * pr, rng)
                step_up += up
                if record and up > 0:
                    st.reactions.append(Reaction(seg.segment_id, feats.get("_id", ""), "upvote",
                                                 float(up), float(step), uncertainty=0.0))
            score += step_up
            st.accumulated_score = score
            st.peak_velocity = max(st.peak_velocity, step_up)
            st.accumulated_outcomes.append((step + 1, "react", step_up))
            # front-page transition depends on EARLY velocity: only attempted in the first
            # `fp_window` steps. A post that hasn't crossed by then dies in /new (exposure collapses).
            if not st.on_front_page:
                if step < p.fp_window and rng.random() < frontpage_prob(score, p):
                    st.on_front_page = True
                    st.exposure_pool *= p.frontpage_multiplier
                    for seg in segs:
                        if seg.segment_id == "casual_frontpage":
                            seg.attention *= 2.0
                else:
                    st.exposure_pool *= (0.5 if step < p.fp_window else 0.08)  # sinks in /new
            else:
                st.exposure_pool *= p.frontpage_decay
            st.novelty *= p.novelty_decay
        return score, st

    # ---------------- many trajectories -> distribution ----------------
    def simulate(self, feats: dict, *, author_rep: float | None = None, ctx: dict | None = None,
                 n_samples: int = 200, seed: int = 0, record_steps: bool = False) -> dict:
        author_rep = self.default_author_rep if author_rep is None else author_rep
        ctx = ctx or {}
        rng = random.Random(seed)
        scores = []
        step_traces = []
        for i in range(n_samples):
            s, st = self._trajectory(feats, author_rep, ctx, rng, record=record_steps and i < 5)
            scores.append(s)
            if record_steps and i < 5:
                step_traces.append([snap for snap in [st.snapshot()]])
        n = len(scores)
        thr = {t: sum(1 for s in scores if s >= t) / n for t in self.thresholds}   # from trajectories
        edges = list(self.thresholds)
        bands = [sum(1 for s in scores if s < edges[0]) / n]
        for i in range(len(edges) - 1):
            bands.append(sum(1 for s in scores if edges[i] <= s < edges[i + 1]) / n)
        bands.append(thr[edges[-1]])
        p_hit_raw = thr.get(40, thr[edges[min(1, len(edges) - 1)]])
        return {
            "thresholds": thr, "band_probs": bands, "p_hit_raw": p_hit_raw,
            "sim_mean_score": sum(scores) / n, "sim_p_frontpage": sum(1 for s in scores if s >= edges[0]) / n,
            "n_samples": n, "sample_scores": scores if record_steps else None,
            "step_traces": step_traces or None,
        }

    def predict(self, feats: dict, *, author_rep: float | None = None, ctx: dict | None = None,
                n_samples: int = 200, seed: int = 0) -> dict:
        sim = self.simulate(feats, author_rep=author_rep, ctx=ctx, n_samples=n_samples, seed=seed)
        p_hit = sim["p_hit_raw"]
        if self.readout is not None:
            a, b = self.readout
            p_hit = _SIG(a * _LOGIT(p_hit) + b)
        sim["p_hit"] = p_hit
        return sim

    # ---------------- calibration readout (reads only the simulated outcome) ----------------
    def fit_readout(self, raws: list[float], y: list[int]) -> "HNSimulationEngine":
        from swm.transition.readout import LogisticReadout
        if len(set(y)) == 2:
            m = LogisticReadout(epochs=250).fit([[_LOGIT(min(1 - 1e-6, max(1e-6, r)))] for r in raws], y)
            # extract scalar (a,b): predict on logit axis
            # LogisticReadout standardizes; reconstruct a,b by 2-point fit
            import statistics
            xs = [_LOGIT(min(1 - 1e-6, max(1e-6, r))) for r in raws]
            lo, hi = min(xs), max(xs)
            if hi - lo < 1e-6:
                self.readout = (1.0, 0.0)
            else:
                p_lo = _LOGIT(min(1 - 1e-6, max(1e-6, m.predict_proba([lo]))))
                p_hi = _LOGIT(min(1 - 1e-6, max(1e-6, m.predict_proba([hi]))))
                a = (p_hi - p_lo) / (hi - lo)
                b = p_lo - a * lo
                self.readout = (a, b)
        return self
