"""Learned-prior registry — the calibration flywheel across datasets and domains.

The vision needs every variable, for any question, to arrive with a calibrated weight-with-uncertainty. No
single dataset covers every variable, but MANY datasets each pin down SOME elasticities. This registry is
where those accumulate: each backtest/fit contributes per-variable elasticity estimates (a mean + a CI +
an evidence count) keyed by a SEMANTIC key `(variable, outcome-class)`, and estimates from different domains
are COMBINED Bayesianly (precision-weighted) — so the more data we calibrate on, the tighter and more
transferable the priors become. The compiler then consults the registry so an emitted variable gets a
data-calibrated weight where we have evidence, and falls back to an LLM/literature prior where we don't.

Precision-weighted combination (each source i has mean μᵢ, sd σᵢ): posterior precision = Σ 1/σᵢ², posterior
mean = (Σ μᵢ/σᵢ²) / (Σ 1/σᵢ²). More/《tighter evidence ⇒ a stronger prior; independent domains reinforce.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from swm.variables.calibrated_weights import WeightPrior

DEFAULT_PATH = "swm/variables/learned_priors.json"


def semantic_key(variable: str, outcome_class: str) -> str:
    """Canonicalize (variable, outcome-class) to a stable key. Case/punctuation-normalized but LEVEL-PRESERVING
    — `party=republican` and `party=democrat` are distinct keys (opposite elasticities). Cross-phrasing/domain
    transfer (mood ↔ affect) is the future embedding-keyed registry (see ARCHITECTURE_PEAK.md item 4); the
    string key here transfers only exact-meaning variables across datasets."""
    def norm(s):
        s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
        return re.sub(r"\s+", " ", s).strip()
    return f"{norm(variable)}|{norm(outcome_class)}"


@dataclass
class PriorRecord:
    mean: float
    sd: float
    n: int = 0                     # total evidence (datapoints) behind this estimate
    source: str = "fit"

    def precision(self):
        return 1.0 / max(1e-6, self.sd) ** 2

    def as_dict(self):
        return {"mean": round(self.mean, 5), "sd": round(self.sd, 5), "n": self.n, "source": self.source}


@dataclass
class PriorRegistry:
    records: dict = field(default_factory=dict)      # semantic_key -> PriorRecord

    # ---- persistence ----
    @classmethod
    def load(cls, path=DEFAULT_PATH):
        try:
            with open(path) as f:
                raw = json.load(f)
            return cls({k: PriorRecord(**v) for k, v in raw.items()})
        except (FileNotFoundError, ValueError):
            return cls({})

    def save(self, path=DEFAULT_PATH):
        with open(path, "w") as f:
            json.dump({k: r.as_dict() for k, r in sorted(self.records.items())}, f, indent=1)
        return self

    # ---- read ----
    def get(self, variable, outcome_class, *, min_n=1):
        r = self.records.get(semantic_key(variable, outcome_class))
        if r is None or r.n < min_n:
            return None
        return WeightPrior(name=variable, mean=r.mean, sd=r.sd, source=f"registry({r.source},n={r.n})")

    def prior_for(self, variable, outcome_class, *, fallback: WeightPrior = None):
        """The best available prior for a variable: the learned registry record if present, else the fallback
        (an LLM/literature prior), else an uninformative wide prior."""
        got = self.get(variable, outcome_class)
        if got is not None:
            return got
        if fallback is not None:
            return fallback
        return WeightPrior(name=variable, mean=0.0, sd=3.0, source="uninformative")

    # ---- write (Bayesian combine) ----
    def update(self, variable, outcome_class, mean, sd, *, n=1, source="fit"):
        key = semantic_key(variable, outcome_class)
        new = PriorRecord(mean, max(1e-3, sd), n, source)
        old = self.records.get(key)
        if old is None:
            self.records[key] = new
            return self
        p_old, p_new = old.precision(), new.precision()          # precision-weighted (Bayesian) combination
        prec = p_old + p_new
        self.records[key] = PriorRecord(mean=(old.mean * p_old + new.mean * p_new) / prec,
                                        sd=(1.0 / prec) ** 0.5, n=old.n + n, source=f"{old.source}+{source}")
        return self

    def register_from_fit(self, model, outcome_class, *, source="fit"):
        """Ingest a fitted CalibratedWeights: store each variable's fitted weight + posterior SD as evidence
        (weighted by the fit's data). This is how a backtest feeds the flywheel."""
        rep = model.weight_report()
        n = getattr(model, "n_train", 0) or 1
        for r in rep:
            self.update(r["name"], outcome_class, r["weight"], r["sd"], n=n, source=source)
        return self
