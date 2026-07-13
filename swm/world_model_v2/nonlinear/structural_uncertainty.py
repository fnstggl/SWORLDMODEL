"""Structural-form uncertainty — Phase 7, Part 9.

When several structural forms remain plausible after validation (linear vs saturating; threshold vs smooth
dose-response; simple vs complex contagion; one regime vs two; hazard vs point process), the honest thing is
NOT to average their incompatible coefficients into one number. It is to keep the competing forms, weight them
by held-out evidence, propagate each as its own branch/particle, and REPORT the disagreement.

`FormPosterior` holds the evidence-weighted set of forms for one mechanism. Weights come from held-out
performance (a softmax over −½·n_eff·Brier, i.e. an approximate model-evidence / information-criterion
weighting), never from an LLM prior. `structural_disagreement` quantifies how much the answer depends on which
form is right — the quantity a downstream decision needs to see before trusting a nonlinear claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class FormCandidate:
    form_id: str
    params: dict
    held_out_score: float                # lower is better (Brier / logloss on the SAME validation split)
    n_eff: float = 100.0                 # effective validation sample (scales evidence sharpness)
    weight: float = 0.0                  # filled by FormPosterior.normalize
    note: str = ""


@dataclass
class FormPosterior:
    """Evidence-weighted competing structural forms for ONE mechanism/causal-process."""
    mechanism_family: str
    candidates: list = field(default_factory=list)     # [FormCandidate]
    selection_metric: str = "brier"
    selection_split: str = ""

    def normalize(self, *, temperature: float = 1.0):
        """Softmax weights over −½·n_eff·score (approx. model evidence). Ties → near-uniform (preserved)."""
        if not self.candidates:
            return self
        scores = [-0.5 * c.n_eff * c.held_out_score / max(1e-9, temperature) for c in self.candidates]
        m = max(scores)
        exps = [math.exp(min(30.0, s - m)) for s in scores]
        z = sum(exps) or 1.0
        for c, e in zip(self.candidates, exps):
            c.weight = e / z
        return self

    def selected(self) -> FormCandidate:
        return min(self.candidates, key=lambda c: c.held_out_score)

    def is_ambiguous(self, *, margin: float = 0.15) -> bool:
        """True if ≥2 forms retain non-trivial weight — keep both, do not collapse (Part 9)."""
        ws = sorted((c.weight for c in self.candidates), reverse=True)
        return len(ws) >= 2 and ws[1] >= margin

    def mix(self, evaluate) -> float:
        """Weighted mixture E_form[f(x)] for a callable `evaluate(candidate) -> value`. Used only when the
        forms are compatible enough to mix (same output meaning); otherwise branch instead of mixing."""
        return sum(c.weight * evaluate(c) for c in self.candidates)

    def structural_disagreement(self, evaluate) -> dict:
        """How much the answer depends on WHICH form — the spread of per-form predictions under the weights."""
        vals = [(evaluate(c), c.weight) for c in self.candidates]
        z = sum(w for _, w in vals) or 1.0
        mean = sum(v * w for v, w in vals) / z
        var = sum(w * (v - mean) ** 2 for v, w in vals) / z
        return {"mixture_mean": round(mean, 6), "disagreement_sd": round(math.sqrt(max(0, var)), 6),
                "per_form": {c.form_id: round(evaluate(c), 6) for c in self.candidates},
                "weights": {c.form_id: round(c.weight, 4) for c in self.candidates},
                "ambiguous": self.is_ambiguous()}

    def as_dict(self):
        return {"mechanism_family": self.mechanism_family, "selection_metric": self.selection_metric,
                "selection_split": self.selection_split, "selected_form": self.selected().form_id,
                "ambiguous": self.is_ambiguous(),
                "candidates": [{"form_id": c.form_id, "held_out_score": round(c.held_out_score, 6),
                                "weight": round(c.weight, 4), "n_eff": c.n_eff, "note": c.note}
                               for c in sorted(self.candidates, key=lambda c: c.held_out_score)]}
