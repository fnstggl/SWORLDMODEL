# SWORLDMODEL ⇄ behaviour-model integration contract

This defines the **eventual** interface by which the fine-tuned behaviour model plugs into
the SWORLDMODEL runtime as an actor policy. It is a contract only — **the behaviour model
is NOT wired into the production simulator in this task**. Do not import `swm` from
`machine_learning` or vice-versa until this contract is deliberately implemented behind a
feature flag.

## Where it sits

```
General model (e.g. DeepSeek): researches + constructs the scenario
        │
SWORLDMODEL runtime: enforces time, information access, authority, feasibility,
        │            ordering, physical / institutional constraints
        ▼
Behaviour model (this subsystem's fine-tuned adapter):
        │   predicts what an actor or population does next
        ▼
Calibration + evaluation layer: corrects probabilities, timing, action
                                frequencies, population distributions
```

The behaviour model **proposes**; the runtime **disposes**. The runtime remains the
authority on feasibility and constraints — a predicted action that violates a constraint is
rejected/re-sampled by the runtime, never forced through.

## Interface

```python
def predict_actor_behavior(
    actor_local_state: ActorLocalState,        # pre-cutoff private + profile state the runtime holds
    current_observation: Observation,          # what the actor observes now
    available_actions: list[Action] | None,    # the feasible action set (None = unknown/open)
    simulation_time: SimTime,                   # the runtime's clock at the decision point
    sampling_config: SamplingConfig,            # temperature, n_samples, task, calibration on/off
) -> ActorBehaviorDistribution: ...
```

### `ActorBehaviorDistribution`

```python
@dataclass
class ActorBehaviorDistribution:
    task_type: str                     # e.g. PREDICT_NEXT_ACTION / PREDICT_NEXT_MESSAGE / PREDICT_TIME_TO_ACTION
    # discrete tasks:
    action_probs: dict[str, float] | None      # action/choice -> probability (sums to 1)
    # generative tasks:
    message_samples: list[str] | None          # sampled next messages
    # timing tasks:
    time_to_action: TimingDistribution | None  # {p_act, mean_seconds, quantiles, censored}
    # response/nonresponse:
    p_response: float | None
    # metadata:
    calibrated: bool                   # whether the calibration layer adjusted the raw model output
    raw_model_id: str                  # base + adapter revision that produced this
    n_samples: int
    warnings: list[str]                # e.g. "available_actions unknown; open-vocabulary output"
```

## Adapter mapping (how the model actually produces this)

1. The runtime's `(actor_local_state, current_observation, available_actions, simulation_time)`
   is mapped into a **canonical behaviour-event context** (the same shape converters emit):
   `context.actor_profile`, `context.private_state_before`, `context.known_history`,
   `context.current_observation`, `context.available_actions`, `cutoff.cutoff_time =
   simulation_time`.
2. `examples/formatters/sft.format_record` renders the leakage-safe prompt (everything up to
   `TARGET:`).
3. The fine-tuned adapter generates `sampling_config.n_samples` completions
   (`evaluation/model_eval.generate_predictions`).
4. Completions are parsed into the task's target shape and aggregated into
   `action_probs` / `message_samples` / `time_to_action` / `p_response`.
5. The **calibration layer** (temperature scaling / isotonic / frequency-matching, fit on a
   held-out split) adjusts probabilities, timing, and action frequencies before returning.

## Guarantees the behaviour model must uphold

- It never sees post-cutoff information (guaranteed by the canonical schema + formatter).
- It emits explicit `NO_ACTION` / `NO_RESPONSE` / `WAIT` and censored timing rather than
  silently dropping inactivity.
- When `available_actions is None`, it flags open-vocabulary output in `warnings` and does
  not fabricate a closed option set.
- It returns `raw_model_id` (base + adapter revision) so every prediction is reproducible.

## Non-goals for this task

- No production wiring, no feature flag flipped, no `swm` import.
- No calibration layer *fit* yet (the interface + hooks exist; fitting needs a trained
  adapter + a held-out split).
- The comparison baselines (`prompted DeepSeek`, `current SWORLDMODEL actor policy`) are
  interface stubs in `evaluation/model_eval.py` — they raise with guidance rather than
  calling a paid API or the production runtime.
