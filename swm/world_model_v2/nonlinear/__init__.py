"""Phase 7 — production nonlinear & context-dependent mechanisms.

This package is the Phase-7 subsystem. It is ADDITIVE to Phases 1/2/3/6 (and safe against the parallel
Phase-9/10 work): it introduces no change to the shared compiler / materialize / rollout / WorldState /
StateDelta schemas, plugging in only through the sanctioned seams — a registered `TransitionOperator`
(operators.py, Mode A), the entity `latent_state` extension door (history.py), the event `payload`
(the nonlinear_spec), and an integrity-hashed sidecar registry (registry_ext.py).

Layers:
  forms.py                 typed structural-form registry (evaluable, pure-Python runtime)
  context.py / history.py  typed context-conditioning + event-history/memory schemas (leakage-guarded)
  posterior.py             Phase-3 per-particle propagation — E[f(X)] ≠ f(E[X]) (the missing helper)
  pooling.py               hierarchical partial pooling (empirical Bayes)
  structural_uncertainty.py   evidence-weighted competing forms + disagreement
  applicability.py         nonlinear applicability + transport/extrapolation gates
  composition.py           nonlinear composition + stability guards
  safety.py                numerical stability + append-only failure records
  operators.py             the execution plane: nonlinear TransitionOperators (StateDelta + future events)
  registry_ext.py          additive Phase-6 sidecar registry (nonlinear extensions)
  fit.py / compare.py      OFFLINE fitting + model comparison (numpy/scipy/sklearn optional) → serialized packs
"""
from __future__ import annotations

from swm.world_model_v2.nonlinear import forms, context, history, posterior, pooling, safety
from swm.world_model_v2.nonlinear import structural_uncertainty, applicability, composition, registry_ext

# importing operators registers the Phase-7 transition operators + event types (import side effect)
from swm.world_model_v2.nonlinear import operators  # noqa: F401

__all__ = ["forms", "context", "history", "posterior", "pooling", "safety", "structural_uncertainty",
           "applicability", "composition", "registry_ext", "operators"]
