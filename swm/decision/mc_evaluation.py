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

    def summary(self) -> dict:
        return {"report_type": "simulation",
                "p_reply_fraction": round(self.p_reply, 4),
                "p_reply_mean": round(self.p_mean, 4),
                "interval80": [round(self.interval80[0], 4), round(self.interval80[1], 4)],
                "n_samples": self.n_samples, "calibration_grade": self.grade,
                "note": "fraction of simulated recipient-trajectories that reply, integrated over the "
                        "recipient's hidden state (base-rate posterior, trait uncertainty, mood, "
                        "attention, timing). UNVALIDATED: elasticities are world-knowledge priors, not "
                        "backtested — treat the level as a claim to check; trust the ranking first."}


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
    return MCResult(p_reply=replied / n_samples, p_mean=sum(per_draw) / len(per_draw),
                    interval80=(lo, hi), n_samples=n_samples, grade=grade)
