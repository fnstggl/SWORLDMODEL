# EXP-030 — Event-conditioned belief DYNAMICS: the temporal transition operator (the missing half)

Every other result in this repo is **cross-sectional** — predict an outcome from a state, now. A *world*
model also needs **dynamics**: how a belief STATE evolves over TIME when an EVENT hits, `P(s_{t+1} | s_t,
event)`. This is the single biggest architectural gap we identified (prompted by Yu et al. 2026,
"Building Social World Models with LLMs"), and the regime we kept losing (markets, EXP-017). This builds
it and validates it no-cheat on real event-driven belief trajectories.

## Setup (no-cheat)
- **SWM-Bench / Kalshi** (Yu et al. 2026): 760 test belief transitions, each a market-price trajectory
  (a proxy for collective belief), ~12 candidate news events timestamped strictly before the target, and
  the next belief value. Chronological split (train before Nov 2025, test after); news before the target
  → no leakage.
- **Operator** (`swm/transition/belief_dynamics.py`): a general event-conditioned transition — state
  features (level, momentum, volatility) + an event channel. The event channel is our thesis applied to
  events: a 12-agent swarm read each transition's question + trajectory + news and inferred the **signed
  directional impact** the news implies for the belief (event *semantics*, never the outcome). The null
  branch is persistence (Δ=0), the efficient-market martingale every event effect is measured against.

## Result — the operator beats persistence (the honest bar)
| tier | MAE ↓ | 3-way DA ↑ | DA on real moves ↑ | corr(Δ̂,Δ) ↑ |
|---|---|---|---|---|
| persistence (martingale null) | 0.0603 | 0.379 | 0.00 | 0.00 |
| state + cheap keyword features (no LLM) | 0.0703 | 0.380 | 0.337 | 0.116 |
| learned state + LLM impact | 0.0811 | 0.345 | **0.475** | 0.181 |
| **LLM event-impact channel (Δ = scale·impact)** | **0.0594** | **0.426** | 0.123 | 0.159 |

**The LLM event-impact channel beats persistence on the full set on BOTH magnitude (MAE 0.0594 < 0.0603)
and 3-way directional accuracy (0.426 vs 0.379, +4.7 pts).** It is a genuine, non-regressive win.

### Where the dynamics really pay off — the event-driven subset
On the 52 transitions where the LLM judged a real event present (|impact| ≥ 0.15 — the paper's
"attributed subset"):

| tier | MAE ↓ | 3-way DA ↑ | DA on real moves ↑ |
|---|---|---|---|
| persistence | 0.0896 | 0.250 | 0.00 |
| **LLM event-impact channel** | **0.0826** | **0.635** | **0.846** |

**When an event is actually present, the transition engine calls the belief's direction correctly 85% of
the time (vs the martingale's 0%), and beats persistence on MAE too.** This is exactly the regime a world
model exists for — anticipating how collective belief moves in response to events.

## Why it works — and the honest mechanism finding
- **Cheap state/time-series features can't beat persistence** (MAE 0.070 > 0.060) — reproducing the
  paper's result that time-series models are near-chance on news-driven direction.
- **Adding learned state features to the LLM impact HURTS** (MAE 0.081): the model over-moves on quiet
  periods. The clean winner is the **pure event-impact channel**, which is *naturally gated* — when no
  event is present the LLM impact is ~0, so the prediction collapses to persistence. (An explicit
  `gate_by_impact` mode reproduces this for the learned model.) The signal is the *event understanding*,
  not the trajectory shape.
- Coverage: the LLM judged a directional event present in 54% of transitions; the other 46% correctly
  fall back to persistence.

## Honest limits
- The full-set MAE win is small (0.0594 vs 0.0603, ~1.5%); the decisive wins are **directional accuracy**
  and the **event-driven subset** — which is precisely where dynamics matter and where the paper's SWM
  also concentrates its gains.
- Validated on Kalshi (760 transitions); Polymarket (noisier, more algorithmic/endogenous moves) is the
  harder next test, where the paper too trails on magnitude.
- The event channel is a one-shot LLM impact per transition, not the paper's full posterior-guided
  attribution training; adding hindsight-attribution pseudo-labeling is the natural next lift.
- This is aggregate belief dynamics; the same operator form (`P(next belief | belief, event)`) applies to
  an individual updating a belief — wiring it to the per-person VariableMap is future work.

## Reproduce
Download SWM-Bench (fetch note in `experiments/datasets_swm.py`) → the committed event-impact signals
live at `experiments/results/exp030_swm/swm_impact.json` → `python -m experiments.exp030_belief_dynamics`.
`python -m pytest tests/test_belief_dynamics.py` covers state features, event gating, and the operator.
