"""TRIBE v2 adapter — DISABLED BY DEFAULT, QUARANTINED, NON-COMMERCIAL. See docs/AUDIT_PART_I_TRIBE.md.

TRIBE v2 (facebook/tribev2) is an fMRI ENCODING model: stimulus (video+audio+text) → predicted BOLD. It does
NOT predict behavior. Two hard blockers stand before any product use, both documented in the audit:
  1. LICENSE: CC-BY-NC-4.0 (non-commercial) + Llama-3.2 / V-JEPA2 / Wav2Vec-BERT component terms.
  2. UNPROVEN BRIDGE: no BOLD→behavior mapping exists; it must be LEARNED and must beat the ordinary
     Llama-3.2-3B embedding (TRIBE's own text encoder) on held-out behavior before it can be claimed to add
     value (docs/AUDIT_PART_I_TRIBE.md §I6/I7 — the Upworthy representation ablation, T3 is the bar).

This module therefore does NOT run TRIBE. It defines the FEATURE CONTRACT the audit specifies and refuses to
execute unless the caller *explicitly* opts in AND a real backend is present (a GPU box with the weights).
Nothing in `swm/engine/*` may import this file (pinned by test_experimental_is_quarantined). It exists so the
representation-ablation experiment can be built and reviewed off-line, with the plain-embedding control, and
never so an unvalidated non-commercial neural layer can enter a forecast.

Usage (research only, on a GPU box with the weights and an accepted non-commercial license):
    ad = TribeAdapter(enabled=True, backend=my_tribe_backend)   # backend implements the 5 calls below
    feats = ad.features_for("Silence, engineered.")             # returns TribeFeatures (T3..T6)
"""
from __future__ import annotations

from dataclasses import dataclass, field

LICENSE = "CC-BY-NC-4.0 (NON-COMMERCIAL) + Llama-3.2 / V-JEPA2 / Wav2Vec-BERT component licenses"

# Functional networks/ROIs the paper RECOVERS (not cherry-picked by name) — the T6 low-dim summary schema.
# Kept as a documented list so the ablation reports a fixed, pre-registered feature set.
ROI_SUMMARIES = (
    "language_network", "broca_45", "sts_superior_temporal_sulcus",   # language / syntax / speech
    "tpj_temporoparietal_junction", "mtg_middle_temporal_gyrus",      # semantic / emotional / social cognition
    "mpfc_default_mode",                                              # self-referential / valuation-adjacent
    "auditory_cortex", "ventral_visual_stream_ffa_ppa_eba_vwfa",     # sensory (mostly dropout-masked for text)
)


@dataclass
class TribeFeatures:
    stimulus: str
    llama_embedding: list = None          # T3 — ordinary Llama-3.2-3B text embedding (THE CONTROL to beat)
    tribe_latent: list = None             # T4 — TRIBE transformer latent
    cortical_pred: list = None            # T5 — predicted BOLD, ~20,484 cortical vertices
    roi_summary: dict = field(default_factory=dict)   # T6 — {roi: scalar} over ROI_SUMMARIES
    repro_variance: float = None          # variance across repeated inference (I3 stability gate)
    note: str = ""


class TribeUnavailable(RuntimeError):
    """Raised when TRIBE features are requested without an explicit opt-in AND a real backend."""


@dataclass
class TribeAdapter:
    """Feature contract for TRIBE v2. Disabled unless `enabled=True` AND a `backend` is supplied. The backend
    is any object exposing `llama_embedding(text)`, `tribe_latent(text)`, `cortical_pred(text)` — none of which
    are runnable in this environment (no GPU, weights are proxy-limited), by design."""
    enabled: bool = False
    backend: object = None                # the real TRIBE runner (GPU + weights); None here on purpose
    commercial_use: bool = False          # MUST stay False — the license forbids commercial use

    def __post_init__(self):
        if self.commercial_use:
            raise TribeUnavailable(
                f"TRIBE is {LICENSE}: commercial_use=True is not permitted. Keep it False (research only).")

    def available(self) -> bool:
        return bool(self.enabled and self.backend is not None)

    def _require(self):
        if not self.available():
            raise TribeUnavailable(
                "TRIBE adapter is disabled/unbacked (research-only, non-commercial, GPU-required). This is "
                "intentional — see docs/AUDIT_PART_I_TRIBE.md. Provide enabled=True and a real backend on a "
                "GPU box with an accepted non-commercial license to run the Upworthy representation ablation.")

    def features_for(self, stimulus: str, *, reps: int = 1) -> TribeFeatures:
        """Return the T3..T6 feature bundle for one stimulus (cache one pass per unique stimulus upstream).
        Refuses unless explicitly enabled with a backend."""
        self._require()
        emb = self.backend.llama_embedding(stimulus)
        lat = self.backend.tribe_latent(stimulus)
        cort = self.backend.cortical_pred(stimulus)
        roi = self.backend.roi_summary(cort) if hasattr(self.backend, "roi_summary") else {}
        return TribeFeatures(stimulus=stimulus, llama_embedding=emb, tribe_latent=lat,
                             cortical_pred=cort, roi_summary=roi,
                             note="research-only; validate T4/T5/T6 > T3 on held-out behavior before any use")

    def contrast(self, a: str, b: str) -> dict:
        """Candidate-to-candidate neural contrast (paired worlds) — the shape the ablation consumes. Refuses
        unless enabled."""
        self._require()
        fa, fb = self.features_for(a), self.features_for(b)
        return {"a": a, "b": b,
                "roi_delta": {k: (fa.roi_summary.get(k, 0.0) - fb.roi_summary.get(k, 0.0))
                              for k in ROI_SUMMARIES}}
