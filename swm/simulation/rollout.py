"""Multi-step rollout + the free-running calibration-by-horizon eval (audit gap: Q5).

The audit's sharpest correct criticism: the repo sampled trajectories but never *evaluated* a
free-running rollout — one that advances state on its OWN sampled outcomes and measures how
calibration degrades with horizon. This module does exactly that, so the distinctive world-model
claim ("roll the state forward and predict N steps out") is testable, not asserted.

Two functions:
- `simulate(...)`: free-running rollout of an AggregateTransition over an action plan — sample an
  outcome, transition the population state, continue → a distribution of futures per step.
- `calibration_by_horizon(...)`: the honest multi-step eval. For held-out entity sequences it
  compares, per horizon h:
    * TEACHER-FORCED: state advanced with the ACTUAL outcomes (an upper bound — you knew the truth),
    * FREE-RUNNING:   state advanced with the model's own SAMPLED outcomes (the real world-model
      regime — you don't know the truth).
  If free-running calibration/log-loss does not degrade much faster than teacher-forced, multi-step
  simulation is trustworthy; if it blows up, it isn't — and we report which.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from swm.eval.metrics import expected_calibration_error, log_loss
from swm.state.population import PopulationState
from swm.state.state import Action
from swm.transition.aggregate_transition import AggregateTransition
from swm.transition.transition_head import band_of, rand_band, sample_in_band


@dataclass
class Rollout:
    steps: int
    n_samples: int
    per_step: list[dict]
    report_type: str = "simulation"
    calibration_grade: str = "unvalidated"
    warning: str = ""


def simulate(transition: AggregateTransition, initial: PopulationState, action_plan: list[Action],
             *, n_samples: int = 200, seed: int = 0,
             validated_grade: str | None = None, validated_horizon: int = 0,
             domain: str = "") -> Rollout:
    """Free-running rollout: at each step predict, sample an outcome, transition, continue."""
    import copy
    horizon = len(action_plan)
    step_mags: list[list[float]] = [[] for _ in range(horizon)]
    step_bands: list[list[list[float]]] = [[] for _ in range(horizon)]
    for s in range(n_samples):
        rng = random.Random(seed * 100003 + s)
        pop = copy.deepcopy(initial)
        tr = copy.deepcopy(transition)
        for j, action in enumerate(action_plan):
            pred = tr.predict(pop, action)
            band = rand_band(pred["band_probs"], rng)
            mag = sample_in_band(band, rng)
            tr.transition(pop, action, mag)
            step_mags[j].append(mag)
            step_bands[j].append(pred["band_probs"])
    per_step = []
    for j in range(horizon):
        m = sorted(step_mags[j])
        n = len(m)
        nb = len(step_bands[j][0])
        band_mean = [sum(b[k] for b in step_bands[j]) / n for k in range(nb)]
        per_step.append({"t": j + 1, "outcome_median": m[n // 2],
                         "interval80": [m[int(0.1 * n)], m[min(n - 1, int(0.9 * n))]],
                         "band_probs_mean": [round(x, 4) for x in band_mean]})
    if validated_grade and horizon <= validated_horizon:
        grade, warn = validated_grade, ""
    else:
        grade = "unvalidated"
        warn = (f"Qualitative simulation; not calibrated at horizon {horizon}"
                f"{' on ' + domain if domain else ''} (validated<= {validated_horizon}).")
    return Rollout(horizon, n_samples, per_step, calibration_grade=grade, warning=warn)


def calibration_by_horizon(build_transition, sequences: list[list[tuple[Action, float]]], *,
                           target_threshold: int = 40, n_samples: int = 40, seed: int = 0,
                           initial_pop: PopulationState | None = None) -> dict:
    """Compare teacher-forced vs free-running calibration per horizon.

    build_transition() -> a FRESH AggregateTransition fitted on the TRAINING slice (so each sequence
    starts from the same trained head but an independent state). `sequences` MUST be HELD-OUT
    per-entity (Action, magnitude) lists (e.g. test-slice authors) — the caller owns that split.
    `initial_pop` is the WARM population state after training; pass it so eval-time state features
    match the trained regime (a cold start mismatches the train feature distribution). Returns
    per-horizon log-loss + ECE for teacher-forced (state advanced by ACTUAL outcomes, an upper
    bound) vs free-running (state advanced by the model's own SAMPLED outcomes, the real regime).
    """
    import copy
    thr = target_threshold
    tf: dict[int, list[tuple[int, float]]] = {}   # teacher-forced: horizon -> (y, p)
    fr: dict[int, list[tuple[int, float]]] = {}   # free-running

    def _start_pop(seq):
        if initial_pop is not None:
            p = copy.deepcopy(initial_pop)
            p.timestamp = seq[0][0].timing.get("ts", p.timestamp)
            return p
        return PopulationState(timestamp=seq[0][0].timing.get("ts", 0.0))

    for seq in sequences:
        if len(seq) < 2:
            continue
        base_tr = build_transition()
        pop = _start_pop(seq)
        tr = copy.deepcopy(base_tr)
        for h, (action, mag) in enumerate(seq):
            p = tr.predict(pop, action)["thresholds"].get(thr, 0.0)
            tf.setdefault(h, []).append((1 if mag >= thr else 0, min(1 - 1e-6, max(1e-6, p))))
            tr.transition(pop, action, mag)                      # ACTUAL outcome
        rng = random.Random(seed)
        pop = _start_pop(seq)
        tr = copy.deepcopy(base_tr)
        for h, (action, mag) in enumerate(seq):
            p = tr.predict(pop, action)["thresholds"].get(thr, 0.0)
            fr.setdefault(h, []).append((1 if mag >= thr else 0, min(1 - 1e-6, max(1e-6, p))))
            band = rand_band(tr.predict(pop, action)["band_probs"], rng)
            tr.transition(pop, action, sample_in_band(band, rng))  # SAMPLED outcome
    def summarize(d):
        rows = []
        for h in sorted(d):
            ys = [y for y, _ in d[h]]; ps = [p for _, p in d[h]]
            if len(ys) >= 5:
                rows.append({"horizon": h + 1, "n": len(ys),
                             "log_loss": round(log_loss(ys, ps), 4),
                             "ece": round(expected_calibration_error(ys, ps), 4),
                             "realized_rate": round(sum(ys) / len(ys), 3)})
        return rows
    return {"teacher_forced": summarize(tf), "free_running": summarize(fr)}
