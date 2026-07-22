"""LAYER 3 — Monte Carlo evaluation of a finalist email under the recipient's HIDDEN STATE.

Layers 1–2 optimize against a fixed recipient. But we don't know the recipient exactly: their base
responsiveness is a posterior, their inferred traits carry confidence, and on any given day their mood,
attention, and the send timing vary. The honest P(reply) is therefore not a point — it's the FRACTION of
simulated recipient-trajectories that reply, integrated over that hidden state.

This is where "we can simulate thousands" is the entire point: for each finalist we draw the recipient's
latent state N times and score the reply under each draw. It reuses the same idea as the HN
`/v1/simulate` endpoint (P(hit) = fraction of trajectories that cross) — repurposed from a front-page
cascade to a reply. The output is a distribution with an interval, never a bare number, and it stays
`unvalidated` (the elasticities are priors, not backtested) — the number is a claim to check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.decision.strategy_scorer import StrategyScorer, _sigmoid, _logit


@dataclass
class MCResult:
    p_reply: float                       # fraction of simulated trajectories that reply
    p_mean: float                        # mean per-draw P(reply)
    interval80: tuple                    # 10th–90th percentile of per-draw P(reply)
    n_samples: int
    grade: str = "unvalidated"
    extrapolation: dict = field(default_factory=dict)   # message-feature support diagnostic

    def summary(self) -> dict:
        note = ("fraction of simulated recipient-trajectories that reply, integrated over the "
                "recipient's hidden state (base-rate posterior, trait uncertainty, mood, attention, "
                "timing). Trust the RANKING first.")
        if self.grade in (None, "unvalidated", "F"):
            note += (" UNVALIDATED: elasticities are world-knowledge priors, not backtested — treat "
                     "the level as a claim to check.")
        else:
            note += (f" Elasticities are graded {self.grade} on held-out REAL persuasion outcomes; "
                     "transporting that calibration to cold email is an assumption.")
        if self.extrapolation.get("out_of_support"):
            note += (" WARNING: this message maxes %d/%d levers — it sits in a low-density corner of "
                     "message-space the fitted data barely covers, so the linear-logit model "
                     "EXTRAPOLATES and the absolute level is over-confident (the ranking is robust; "
                     "the number is not)." % (self.extrapolation.get("n_extreme", 0),
                                              self.extrapolation.get("n_levers", 0)))
        return {"report_type": "simulation",
                "p_reply_fraction": round(self.p_reply, 4),
                "p_reply_mean": round(self.p_mean, 4),
                "interval80": [round(self.interval80[0], 4), round(self.interval80[1], 4)],
                "n_samples": self.n_samples, "calibration_grade": self.grade,
                "extrapolation": self.extrapolation, "note": note}


def mc_evaluate(recipient_vars: dict, base_mean: float, strategy: dict, *,
                base_n_effective: float = 6.0, confidences: dict | None = None,
                n_samples: int = 2000, seed: int = 0, weights: dict | None = None,
                grade: str = "unvalidated", levers: list | None = None) -> MCResult:
    """Integrate P(reply) over the recipient's hidden state for a fixed message `strategy`.

    base_mean / base_n_effective — the recipient's responsiveness posterior (Beta(mean·n, (1-mean)·n)).
    confidences — per-variable confidence in [0,1]; a low-confidence trait is jittered more.
    """
    import random
    rng = random.Random(seed)
    confidences = confidences or {}
    per_draw = []
    replied = 0
    a0 = max(0.05, base_mean * base_n_effective)
    b0 = max(0.05, (1.0 - base_mean) * base_n_effective)

    for _ in range(n_samples):
        # 1) base responsiveness ~ its Beta posterior
        base = rng.betavariate(a0, b0)
        # 2) recipient traits jittered by (1 − confidence): unsure traits vary more
        rvars = {}
        for k, v in recipient_vars.items():
            c = confidences.get(k, 0.5)
            sigma = 0.18 * (1.0 - c)
            rvars[k] = min(1.0, max(-1.0, v + rng.gauss(0.0, sigma)))
        # 3) transient state: mood (logit shift), attention (scales effort sensitivity), timing
        mood = rng.gauss(0.0, 0.25)
        attention = min(1.0, max(0.05, rng.gauss(rvars.get("attention_availability", 0.6), 0.15)))
        rvars["attention_availability"] = attention
        timing = rng.choice([0.0, 0.0, 0.0, -0.15, 0.1])   # most sends land on an ordinary day
        scorer = StrategyScorer(recipient=rvars, base_responsiveness=base, n_weight_samples=1,
                                seed=rng.randint(0, 1_000_000), weights=weights, levers=levers or [])
        p = _sigmoid(_logit(scorer.mean(strategy)) + mood + timing)
        per_draw.append(p)
        if rng.random() < p:
            replied += 1

    s = sorted(per_draw)
    lo = s[int(0.1 * len(s))]
    hi = s[min(len(s) - 1, int(0.9 * len(s)))]
    # message-feature support diagnostic: a fully-optimized message maxes many levers at once — a
    # low-density corner of message-space the fitted data barely covers, where a linear-logit model
    # extrapolates to over-confident certainty. Flag it (the OPE analog is positivity/overlap).
    from swm.decision.strategy_scorer import MESSAGE_VARS
    lever_vals = [strategy.get(v) for v in MESSAGE_VARS if isinstance(strategy.get(v), (int, float))]
    n_extreme = sum(1 for v in lever_vals if v >= 0.9)
    extrap = {"n_levers": len(lever_vals), "n_extreme": n_extreme,
              "extreme_fraction": round(n_extreme / max(1, len(lever_vals)), 3),
              "out_of_support": (len(lever_vals) >= 6 and n_extreme / max(1, len(lever_vals)) >= 0.6)}
    return MCResult(p_reply=replied / n_samples, p_mean=sum(per_draw) / len(per_draw),
                    interval80=(lo, hi), n_samples=n_samples, grade=grade, extrapolation=extrap)


@dataclass
class FunnelMCResult:
    """Valenced funnel evaluation: positive / negative reply probabilities as distributions over the
    recipient's hidden state AND the funnel weight priors, plus the mean stage trace (the WHY —
    which gate is limiting). The OBJECTIVE is p_positive − λ·p_negative; 'any reply' is not success."""
    p_positive: float
    p_negative: float
    objective: float                     # mean of (p_pos − λ·p_neg)
    interval80: tuple                    # 10th–90th pct of per-draw objective
    stage_trace: dict
    n_samples: int
    grade: str = "structural_prior_unvalidated"

    def summary(self) -> dict:
        return {"report_type": "funnel_simulation",
                "p_positive_mean": round(self.p_positive, 4),
                "p_negative_mean": round(self.p_negative, 4),
                "objective_mean": round(self.objective, 4),
                "interval80": [round(self.interval80[0], 4), round(self.interval80[1], 4)],
                "stage_trace": self.stage_trace, "n_samples": self.n_samples,
                "calibration_grade": self.grade,
                "note": "conjunctive response-funnel model (open × understand × believe × relevant "
                        "× worth × easy), VALENCED: objective = P(positive) − 0.25·P(negative). "
                        "STRUCTURAL PRIOR — no labeled cold-email corpus backs these magnitudes; "
                        "trust the ranking and the stage diagnosis, treat absolute levels as claims."}


def mc_evaluate_funnel(recipient_vars: dict, base_mean: float, strategy: dict, *,
                       base_n_effective: float = 6.0, confidences: dict | None = None,
                       n_samples: int = 400, seed: int = 0, levers: list | None = None):
    """Integrate the funnel objective over the recipient's hidden state: per draw, sample base
    responsiveness (Beta posterior) + trait jitter, then one funnel weight draw. Same hidden-state
    discipline as mc_evaluate; the response model is the conjunctive funnel."""
    import random
    from swm.decision.response_funnel import NEGATIVE_REPLY_WEIGHT, FunnelScorer
    rng = random.Random(seed)
    confidences = confidences or {}
    a0 = max(0.05, base_mean * base_n_effective)
    b0 = max(0.05, (1.0 - base_mean) * base_n_effective)
    pp, pn, obj = [], [], []
    stage_acc: dict = {}
    for i in range(n_samples):
        base = rng.betavariate(a0, b0)
        rvars = {}
        for k, v in recipient_vars.items():
            c = confidences.get(k, 0.5)
            rvars[k] = min(1.0, max(-1.0, v + rng.gauss(0.0, 0.18 * (1.0 - c))))
        sc = FunnelScorer(recipient=rvars, base_responsiveness=base, n_weight_samples=1,
                          seed=rng.randint(0, 1_000_000), levers=levers or [])
        d = sc.score_dist(strategy)
        pp.append(d.mean)
        pn.append(d.mean_neg)
        obj.append(d.mean - NEGATIVE_REPLY_WEIGHT * d.mean_neg)
        for k, v in d.stage_trace.items():
            stage_acc[k] = stage_acc.get(k, 0.0) + v
    s = sorted(obj)
    lo = s[int(0.1 * len(s))]
    hi = s[min(len(s) - 1, int(0.9 * len(s)))]
    return FunnelMCResult(p_positive=sum(pp) / len(pp), p_negative=sum(pn) / len(pn),
                          objective=sum(obj) / len(obj), interval80=(lo, hi),
                          stage_trace={k: round(v / n_samples, 4) for k, v in stage_acc.items()},
                          n_samples=n_samples)
