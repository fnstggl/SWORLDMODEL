"""Phase 3 recovery tests: hierarchical shrinkage beats no-pooling AND full-pooling on synthetic truth;
evidence-conditioned latents beat priors; the filtered rollout actually moves branch weights and improves
terminal prediction; structural-hypothesis posteriors concentrate on the generating mechanism."""
import math
import random

import swm.world_model_v2.actor_cognition  # noqa: F401 — registers entity extensions
from swm.world_model_v2.contracts import OutcomeContract
from swm.world_model_v2.events import Event, EventQueue, register_event_type
from swm.world_model_v2.inference_layer import (HypothesisSet, StructuralHypothesis, TimedObservation,
                                                hierarchical_rates, latent_from_rate_evidence,
                                                run_filtered, shrunk_mean, shrunk_rate)
from swm.world_model_v2.init_state import InitialStateModel, LatentVariableRecord
from swm.world_model_v2.observation import GaussianMeasurement, register_observation_model
from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
from swm.world_model_v2.transitions import StateDelta, TransitionOperator, TransitionProposal

T0 = 1.0e9
DAY = 86400.0


# ------------------------------------------------------------------ shrinkage recovery
def test_hierarchical_shrinkage_beats_no_pooling_and_full_pooling():
    """Persons with true rates around a population mean; few observations each. Partial pooling must
    recover truth better (MSE) than raw per-person rates AND better than one pooled rate."""
    rng = random.Random(5)
    true = {f"p{i}": min(0.95, max(0.05, rng.gauss(0.3, 0.1))) for i in range(40)}
    obs = {p: (sum(1 for _ in range(8) if rng.random() < t), 8) for p, t in true.items()}
    post = hierarchical_rates(obs)
    pooled = sum(k for k, _ in obs.values()) / sum(n for _, n in obs.values())
    mse_shrunk = sum((post[p].mean() - true[p]) ** 2 for p in true) / len(true)
    mse_raw = sum((obs[p][0] / obs[p][1] - true[p]) ** 2 for p in true) / len(true)
    mse_pool = sum((pooled - true[p]) ** 2 for p in true) / len(true)
    assert mse_shrunk < mse_raw
    assert mse_shrunk < mse_pool
    # uncertainty honesty: posterior sd shrinks with more data
    big = shrunk_rate(30, 100, prior_mean=0.3, prior_strength=10)
    small = shrunk_rate(3, 10, prior_mean=0.3, prior_strength=10)
    assert big.sd() < small.sd()


def test_shrunk_mean_moves_toward_data_with_n():
    p0 = shrunk_mean([], prior_mu=0.5, prior_sd=0.2, obs_sd=0.1)
    p2 = shrunk_mean([0.9, 0.85], prior_mu=0.5, prior_sd=0.2, obs_sd=0.1)
    p20 = shrunk_mean([0.9] * 20, prior_mu=0.5, prior_sd=0.2, obs_sd=0.1)
    assert p0.mu == 0.5 and p0.method == "prior_only"
    assert 0.5 < p2.mu < p20.mu < 0.9 + 1e-9
    assert p20.sd < p2.sd < p0.sd


def test_latent_from_rate_evidence_carries_provenance_and_uncertainty():
    pool = {"prior_mean": 0.2, "prior_strength": 10.0}
    rec = latent_from_rate_evidence("alice.responsiveness", 6, 10, pool=pool,
                                    evidence_ids=["ev1", "ev2"], lo=0.5, hi=1.8)
    assert rec.method == "dataset" and rec.evidence == ["ev1", "ev2"]
    assert rec.candidates["sd"] > 0                          # never a point
    assert 0.5 <= rec.candidates["mean"] <= 1.8


# ------------------------------------------------------------------ filtered rollout recovery
def _drift_world(drift_latent):
    """A quantity 'level' drifts each day by the entity's hidden drift rate; observations measure level."""
    w = WorldState(world_id="f", branch_id="root", clock=SimulationClock(now=T0, as_of=T0))
    e = Entity(identity="sys")
    e.set("preferences", F(drift_latent, dist={"mean": 0.5, "sd": 0.25, "lo": 0.0, "hi": 1.0},
                           status="sampled"), key="drift")
    w.entities["sys"] = e
    from swm.world_model_v2.quantities import Quantity, register_quantity_type
    register_quantity_type("level", units="unit")
    w.quantities["level"] = Quantity(name="level", qtype="level", value=0.0, timestamp=T0)
    return w


register_event_type("drift_tick", scheduling="scheduled", validated=True)


class DriftOperator(TransitionOperator):
    name = "drift"

    def applicable(self, world, event):
        # dedicated event type — NOT background_tick, which the engine auto-injects (would double-count)
        return event.etype == "drift_tick"

    def propose(self, world, event, rng):
        return TransitionProposal(operator=self.name,
                                  action={"days": float(event.payload.get("elapsed_days", 1.0))})

    def apply(self, world, proposal):
        drift = float(world.entity("sys").value("preferences", key="drift") or 0.5)
        q = world.quantities["level"]
        before = q.value
        q.value = before + drift * proposal.action["days"]
        d = StateDelta(at=world.clock.now, event_type="drift", operator=self.name)
        return d.change("quantities[level]", before, q.value)


def _queue_builder(horizon_days=10):
    def build(world):
        q = EventQueue(horizon_ts=T0 + horizon_days * DAY)
        for d in range(1, horizon_days + 1):
            q.schedule(Event(ts=T0 + d * DAY, etype="drift_tick",
                             payload={"elapsed_days": 1.0}))
        return q
    return build


def test_filtered_rollout_updates_weights_and_recovers_hidden_drift():
    """Ground truth drift=0.8; noisy level observations at days 2 and 4. The filtered posterior over the
    hidden drift must move toward truth, weights must actually change, and the terminal level prediction
    must beat the unfiltered prior rollout."""
    register_observation_model("level", GaussianMeasurement(sd=0.4, delay_days=0.0, p_missing=0.0))
    truth_drift = 0.8
    # truth: level(t) = 0.8t → obs at day2 ≈1.6, day4≈3.2 (noise sd 0.4 in the model)
    observations = [TimedObservation("o1", "level", 1.62, T0 + 2 * DAY),
                    TimedObservation("o2", "level", 3.18, T0 + 4 * DAY)]
    base = _drift_world(0.5)
    init = InitialStateModel(base_world=base, latents=[LatentVariableRecord(
        path="sys.preferences[drift]", candidates={"mean": 0.5, "sd": 0.25, "lo": 0.0, "hi": 1.0})])
    contract = OutcomeContract(family="continuous", options=[], resolution_rule="terminal level",
                               readout=lambda w: w.quantities["level"].value,
                               horizon_ts=T0 + 10 * DAY).validate()
    res, branches = run_filtered(init, _queue_builder(10), [DriftOperator()], contract,
                                 observations, n_particles=40, seed=3)
    # weights genuinely moved: either they are non-uniform now, or an ESS-triggered resample fired
    weights = [b.weight for b in branches]
    resampled = any(e.get("event") == "resample" for e in res["assimilation"]["log"])
    assert resampled or max(weights) > 1.5 / len(branches)
    # posterior drift expectation from weighted branches
    num = sum(b.weight * float(b.world.entity("sys").value("preferences", key="drift"))
              for b in branches)
    den = sum(b.weight for b in branches)
    drift_post = num / den
    assert abs(drift_post - truth_drift) < 0.15              # recovered (prior mean was 0.5)
    # terminal prediction: filtered median must beat the prior-only rollout median
    from swm.world_model_v2.rollout import WorldModelV2Run
    prior_run = WorldModelV2Run(initial=init, queue_builder=_queue_builder(10),
                                operators=[DriftOperator()], contract=contract, n_particles=40)
    prior_res, _ = prior_run.run(seed=3)
    truth_terminal = truth_drift * 10
    assert abs(res["quantiles"]["p50"] - truth_terminal) < abs(prior_res["quantiles"]["p50"]
                                                               - truth_terminal)
    assert res["assimilation"]["n_observations"] == 2


def test_structural_hypothesis_posterior_concentrates_on_generating_mechanism():
    """Two structural hypotheses: fast-drift world vs slow-drift world (patched state). Observations
    generated from fast drift → the structural posterior must concentrate on 'fast'."""
    register_observation_model("level", GaussianMeasurement(sd=0.4, delay_days=0.0, p_missing=0.0))

    def patch(drift):
        def f(world):
            world.entity("sys").set("preferences",
                                    F(drift, status="sampled", method="hypothesis_patch"), key="drift")
        return f

    hyps = HypothesisSet([
        StructuralHypothesis("fast", 0.5, describe="drift≈0.8", world_patch=patch(0.8)),
        StructuralHypothesis("slow", 0.5, describe="drift≈0.2", world_patch=patch(0.2)),
    ])
    base = _drift_world(0.5)
    init = InitialStateModel(base_world=base, latents=[])
    contract = OutcomeContract(family="continuous", options=[], resolution_rule="level",
                               readout=lambda w: w.quantities["level"].value,
                               horizon_ts=T0 + 10 * DAY).validate()
    obs = [TimedObservation("o1", "level", 1.6, T0 + 2 * DAY),
           TimedObservation("o2", "level", 3.2, T0 + 4 * DAY)]
    res, _ = run_filtered(init, _queue_builder(10), [DriftOperator()], contract, obs,
                          n_particles=30, seed=11, hypotheses=hyps)
    assert res["structural_posterior"]["fast"] > 0.85
    assert res["structural_prior"] == {"fast": 0.5, "slow": 0.5}
