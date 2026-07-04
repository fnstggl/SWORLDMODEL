"""Trajectory sampling / multi-step rollout (audit C.8, spec sections 3 & 7).

The output of a world model is NOT one prophecy — it is a distribution of plausible futures. Given
an initial state and an action plan, sample N trajectories: at each step predict the outcome
distribution, sample an outcome, evolve the state via the transition, continue. Aggregate into
per-step outcome intervals.

Honesty gate lives here too: a rollout is labeled by whether the (domain, horizon) has a backtest.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.state.state import Action, WorldState
from swm.state.transition import TransitionModel


@dataclass
class Rollout:
    steps: int
    n_samples: int
    per_step: list[dict]          # [{t, outcome_mean, interval, band_probs_mean}, ...]
    trajectories: list[list[float]]  # sampled magnitudes per trajectory
    report_type: str = "simulation"
    calibration_grade: str = "unvalidated"
    warning: str = ""


def rollout(model: TransitionModel, initial: WorldState, action_plan: list[Action],
            *, n_samples: int = 200, seed: int = 0,
            validated_grade: str | None = None, validated_horizon: int = 0,
            domain: str = "") -> Rollout:
    """Sample n_samples independent futures through the action plan; aggregate per step."""
    horizon = len(action_plan)
    trajs: list[list[float]] = []
    step_mags: list[list[float]] = [[] for _ in range(horizon)]
    step_bands: list[list[list[float]]] = [[] for _ in range(horizon)]
    for s in range(n_samples):
        rng = random.Random(seed * 100003 + s)
        state = initial
        mags = []
        for j, action in enumerate(action_plan):
            pred = model.predict_outcome(state, action)
            state, ev = model.step(state, action, rng=rng)
            mags.append(ev.magnitude)
            step_mags[j].append(ev.magnitude)
            step_bands[j].append(pred["band_probs"])
        trajs.append(mags)

    per_step = []
    for j in range(horizon):
        m = sorted(step_mags[j])
        n = len(m)
        band_mean = [sum(b[k] for b in step_bands[j]) / n for k in range(len(step_bands[j][0]))]
        per_step.append({
            "t": j + 1,
            "outcome_median": m[n // 2],
            "interval80": [m[int(0.1 * n)], m[min(n - 1, int(0.9 * n))]],
            "band_probs_mean": [round(x, 4) for x in band_mean],
        })

    # honesty gate
    if validated_grade and horizon <= validated_horizon:
        grade, warn = validated_grade, ""
    else:
        grade = "unvalidated"
        warn = (f"Qualitative simulation only; not calibrated on this domain/horizon "
                f"(validated<= {validated_horizon} steps"
                f"{' on '+domain if domain else ''}, requested {horizon}).")
    return Rollout(steps=horizon, n_samples=n_samples, per_step=per_step, trajectories=trajs,
                   calibration_grade=grade, warning=warn)
