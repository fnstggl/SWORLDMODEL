"""Regime router — decide, per question, whether to run the rich simulation or defer to a baseline.

EXP-074 measured that fidelity wins in some regimes (modelable evolving populations/diffusions, weak simple
baselines) and not others (strong persistence/momentum/market baselines). This turns that map into a
DECISION: given a few features of a question, predict whether the high-fidelity calibrated simulation will
beat the skeptic's baseline — and route accordingly (rich_sim / blend / baseline). Never lose to a simple
model by over-simulating; never under-model where fidelity pays.

The router is a calibrated logistic (BayesianLogistic) over interpretable features, SEEDED from the portfolio
domains and improved as more domains are harvested — it uses the project's own calibration engine on itself.
Features (all interpretable, all available before simulating):
  - mechanism KIND (population / diffusion / election / macro / referendum / institution / market) — one-hot;
  - BASELINE STRENGTH — how much better the best simple baseline is than a coin flip (strong ⇒ lean baseline);
  - IRREDUCIBLE fraction — from the structural variance decomposition (high ⇒ nothing beats the base rate).
A world-knowledge PRIOR (strong baseline / high irreducibility push toward 'baseline') keeps it sensible
with few training domains.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.variables.bayes_logistic import BayesianLogistic

KINDS = ("population", "diffusion", "election", "macro", "referendum", "institution", "market")


def _feat(kind, baseline_strength, irreducible_frac):
    oh = [1.0 if kind == k else 0.0 for k in KINDS]
    return oh + [float(baseline_strength), float(irreducible_frac)]


# prior means on the coefficients: strong baseline and high irreducibility LOWER the odds fidelity wins;
# population/diffusion RAISE them (world knowledge, so the router is sensible before much training data).
def _prior_w():
    w = [0.0] * (len(KINDS) + 2)
    for i, k in enumerate(KINDS):
        w[i] = {"population": 1.2, "diffusion": 1.4, "institution": 0.3, "election": -0.6,
                "macro": -0.8, "referendum": -1.0, "market": -1.6}.get(k, 0.0)
    w[len(KINDS)] = -3.0        # baseline_strength: strong simple baseline -> fidelity less likely to win
    w[len(KINDS) + 1] = -2.0    # irreducible_frac: mostly-noise -> defer to base rate
    return w


@dataclass
class RegimeRouter:
    model: BayesianLogistic = None
    prior_scale: float = 1.0

    def fit(self, examples):
        """`examples`: list of {kind, baseline_strength, irreducible_frac, y} where y=1 if the rich sim beat
        the baselines. Coefficients shrink toward the world-knowledge prior (works with few examples)."""
        X = [_feat(e["kind"], e["baseline_strength"], e.get("irreducible_frac", 0.5)) for e in examples]
        y = [int(e["y"]) for e in examples]
        w0 = [self.prior_scale * v for v in _prior_w()]
        self.model = BayesianLogistic(l2=2.0, w0=w0).fit(X, y) if X else None
        return self

    def _p(self, kind, baseline_strength, irreducible_frac):
        x = _feat(kind, baseline_strength, irreducible_frac)
        if self.model is not None:
            return self.model.predict_proba(x)
        # unfit: pure world-knowledge prior
        z = sum(w * xi for w, xi in zip([self.prior_scale * v for v in _prior_w()], x))
        return 1.0 / (1.0 + 2.718281828 ** (-z))

    def route(self, kind, *, baseline_strength=0.0, irreducible_frac=0.5, hi=0.6, lo=0.4) -> dict:
        """Return the routing decision + calibrated P(rich sim beats the baseline)."""
        p = self._p(kind, baseline_strength, irreducible_frac)
        decision = "rich_sim" if p >= hi else ("baseline" if p <= lo else "blend")
        reason = ("modelable population/diffusion, weak baseline" if decision == "rich_sim"
                  else "strong simple baseline / high irreducibility" if decision == "baseline"
                  else "mixed — blend the sim with the baseline")
        return {"decision": decision, "p_fidelity_wins": round(p, 4), "kind": kind, "reason": reason}


def _prim(card):
    sv = card.get("skill_vs", {})
    return sv.get("persistence", sv.get("momentum", sv.get("base_rate")))


def examples_from_portfolio(portfolio) -> list:
    """Build router training rows from an EXP-074 portfolio result: each domain -> a labeled example."""
    ex = []
    for name, dom in portfolio.get("domains", {}).items():
        fids = [v for v in dom.get("by_fidelity", {}).values() if "skill_vs" in v]
        if not fids:
            continue
        lo, hi = (fids[0] if len(fids) > 1 else None), fids[-1]
        # label: did high fidelity beat the PRIMARY simple baseline (persistence/momentum) by a real margin,
        # and did fidelity help over low-fidelity? (beating every baseline incl. a trend is too strict.)
        prim = _prim(hi)
        helped = (lo is None) or (_prim(hi) is not None and _prim(lo) is not None and _prim(hi) > _prim(lo))
        y = 1 if (prim is not None and prim > 0.05 and helped) else 0
        bl = hi.get("baseline_loss", {})
        base = bl.get("base_rate")
        pers = bl.get("persistence", bl.get("momentum"))
        strength = max(0.0, 1 - pers / base) if (base and pers) else 0.0
        ex.append({"kind": dom.get("kind", ""), "baseline_strength": round(strength, 3),
                   "irreducible_frac": 0.5, "y": y, "domain": name})
    return ex
