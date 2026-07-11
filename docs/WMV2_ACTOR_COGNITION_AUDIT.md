# WMv2 actor-cognition audit — the scalar bottleneck, proven and replaced

## PART 1 — what ACTUALLY executed in the prior "max-capacity" arm (E10)

**Finding, stated plainly: the suspicion is CORRECT.** In the previous round's E10, exact message semantics
entered the simulation through **one scalar** — `reply_propensity ∈ [0,1]` — applied as a bounded multiplier
on a fitted reply hazard. It was not an individualized actor model. Per the acceptance rule, that arm is
**renamed `V2_SCALAR_CONTENT` and demoted to baseline C0**; it must never again be described as
"maximum-capacity V2."

### Exact call graph (code as of the E0–E10 run; refs into `swm/world_model_v2/reference/enron.py`)

```
experiments/wmv2_enron_maxcap.py::eval_split
└── v2_predict(ex, fm, content_fn=chat, …)                      enron.py:203
    ├── p_base = fm.base_p(ex)                                  enron.py:217   [FITTED metadata rate]
    ├── content_multiplier(ex, chat_fn, meter)                  enron.py:181   [THE ONLY CONTENT PATH]
    │   ├── prompt = _CONTENT_PROMPT.format(...)                enron.py:165
    │   ├── txt = chat_fn(prompt)                               → DeepSeek deepseek-chat, ONE call
    │   └── return (reply_propensity, why)                      ← the ENTIRE semantic output
    ├── p_base = min(0.97, max(0.01, p_base*(0.45+1.75*prop)))  enron.py:224   [THE CONVERSION FUNCTION]
    ├── InitialStateModel(base_world, latents=[attention])      enron.py:232-235
    ├── ReplyDecision (operator "reply_decision_fitted")        enron.py:239   [reads p_base; NEVER the text]
    │   └── H = hazard[b]·(p_base/global)·(0.4+0.85·att);  p = 1-(1-H)^(1/n_opp)
    └── terminal readout: p_by[b] = fraction of particles replied ≤ b
```

### The eleven required answers

1. **EntityState fields instantiated:** exactly TWO — `recipient.attention` (F(0.7, assumed) → latent) and
   `recipient.current_action` (None). No goals, beliefs, commitments, obligations, resources, workload
   field, memory, or planned actions were on the entity.
2. **Fields populated from corpus data:** NONE on the entity. Corpus metadata entered only as
   `ex.feats` → `fm.base_p(ex)` (pair/recipient rates, workload tercile, hour/weekday multipliers) —
   outside the typed state.
3. **Fields sampled as latent state:** ONE — `recipient.attention ~ N(0.7, 0.25) clipped [0.1, 1]`
   (prior-backed), per particle.
4. **Fields visible to the recipient (the observable view):** the `_CONTENT_PROMPT` contents only —
   recipient id, sender id, `pair_n`, `pair_rate`, `inbox_7d`, `subject[:200]`, `body[:1500]`.
5. **Fields passed to DeepSeek:** exactly the item-4 list. Nothing else. (Leak-free — but also
   cognition-free.)
6. **Exact DeepSeek input schema:** one prose prompt (`_CONTENT_PROMPT`) with those 7 slots.
7. **Exact DeepSeek output schema:** `{"reply_propensity": <0..1>, "why": "<6 words>"}`.
8. **Every state update caused by the LLM:** **NONE.** The LLM touched no entity field, no belief, no
   relationship, no quantity. Its scalar modified a local Python variable (`p_base`) that parameterized the
   fitted decision operator's hazard. `why` was logged and discarded.
9. **Typed actions available:** TWO — `reply` | `wait` (inside `ReplyDecision.propose`). No reply_later,
   no delegate, no clarify.
10. **The conversion function:** `p_base ← clamp(p_base × (0.45 + 1.75·prop), 0.01, 0.97)`, then
    `H = min(0.95, hazard[b] · (p_base/global_rate) · (0.4 + 0.85·attention))`,
    `p_reply = 1 − (1−H)^(1/n_opp)`.
11. **Was `reply_propensity` the only semantic signal entering the simulator?** **YES.** Grep-provable:
    `subject`/`body` appear inside `content_multiplier`'s prompt construction and nowhere else in
    `v2_predict`'s flow; the only value returned into the world build is `prop` (and the logged `why`).

Representative captured trace (forensic id 1, `wmv2_enron_forensic.json`): DeepSeek read the exact
regulatory-letter body and returned `{"reply_propensity": 0.85, "why": "Relevant to Enron's regulatory
interests."}` → the entire reading collapsed to `p_base 0.087 → 0.176` → the same fitted binary
reply/wait hazard as every other arm. Nothing about WHO the recipient is, what they're obligated to do,
whether the task is theirs, or what actions they could take ever existed in the world.

**Verdict on the architecture question:** yes — this is a plausible bottleneck for why the simulation adds
nothing over the fitted model. A one-scalar channel cannot carry intent/obligation/effort/ownership
structure, and a two-action policy cannot express deferral, delegation, or clarification. Whether removing
the bottleneck actually improves held-out prediction is an empirical question — answered by the C-ladder
below, not assumed.

---

## PART 2–4 — the structured replacement (universal, not email-specific)

New universal module: **`swm/world_model_v2/actor_cognition.py`** — everything lives in the shared V2
ontology; the Enron file contributes only a parameter pack (observable context builder, workload
normalization, fitted dispersion). The same machinery is exercised on a NEGOTIATION scenario in the test
suite (`test_universality_negotiation_domain_uses_identical_machinery`) with zero messaging code.

### The boundary (every actor decision, every domain)

```
typed WorldState
→ actor-observable view (pair history, load, local time, thread marker — never labels/future)
→ structured INTERPRETATION (ONE LLM call → 12 typed clamped dims + intent + why; abstains on parse failure)
   intent ∈ {request_action, request_information, directive, share_information, social, transactional, other}
   urgency, obligation, task_ownership, effort_required, relevance_to_goals, risk_of_inaction,
   benefit_of_action, relationship_salience, needs_clarification, needs_delegation, thread_continuity
→ FITTED calibration layer (train-only logistic: features + metadata-anchor logit → engagement mass;
   with w=0 it reduces EXACTLY to the metadata model — semantics must earn weight from data)
→ TYPED ACTION DISTRIBUTION (messaging pack: reply_now / reply_later / ask_clarification / delegate / ignore;
   the fitted mass is never re-decided by the split — interpretation shapes HOW it splits)
→ simulator validation → StateDelta → shared-world consequences → terminal-state readout
```

The LLM **never mints a probability** in this architecture (the previous round proved raw LLM scalars are
worse-calibrated than bag-of-words). It reads meaning; measurement maps meaning to behavior.

### Hidden actor state (PART 3) — per particle, coherent, labeled

| latent | prior | provenance |
|---|---|---|
| attention | N(0.7 − 0.2·workload, 0.25) ∈ [0.05,1] | prior_backed, workload-coupled |
| responsiveness | N(1, **fitted** σ from per-recipient rate dispersion) ∈ [0.5,1.8] | dataset (fit_hetero_sd) |
| obligation_sensitivity | N(1, 0.2) ∈ [0.5,1.5] | broad prior |
| workload_pressure | inbox_7d / 2·tercile₂ | observed |
| relationship strength | familiarity×reciprocity from pair history | inferred from data |

Correlated draws via the existing `CorrelationRule` machinery (responsiveness↔attention +0.3, declared).
No unsupported precise constants: dispersion is measured, bounds are wide, every record is labeled.

### Temporal processes (PART 4)

- `attention_t ~ transition(attention_{t−1}, workload, time_of_day, noise)` — bounded mean-reversion
  (τ=0.5d) toward a workload/daytime target + diffusion; PERSISTED to the entity at each observation.
- `relationship_t ~ transition(relationship_{t−1}, engaged)` — bounded shift (|Δ|≤0.05), recorded as a
  StateDelta at terminal actions.
- Interruptions (C6): `distraction` hazard (rate 1+2·workload/day) shocks attention −0.15, recovery via
  mean reversion.
- Deferral: `reply_later` completes at the first observation after a per-particle sampled delay
  (lognormal, median ~0.8d, clamped [0.1,5]d); past-horizon completions honestly count as no reply.

### The ladder (PART 5) — each mechanism enters one rung at a time

C0 scalar policy (prev E10) · C1 interpretation+fitted layer (closed form) · C2 +typed actions in the event
world · C3 +hidden actor state · C4 +dynamic attention · C5 +relationship state · C6 maximum structured
actor model. All C1–C6 share ONE identical interpretation call per example (memoized) so rung differences
isolate mechanism. Baselines on identical rows: E0/E1/E2/E3/E4.

### Acceptance rule (PART 6) — adopted

An arm may be called "maximum-capacity V2" only if it has: structured actor state ✓, structured message
interpretation ✓, typed action distribution ✓, persistent hidden state ✓, dynamic state transitions ✓,
terminal-state readout ✓. C6 qualifies. C0 (scalar) is preserved as a baseline and may not carry the label.

## Results

See `experiments/results/wmv2_enron_actor_ladder.json` (+ forensic
`wmv2_enron_actor_forensic.json`) and the verdict section appended to `docs/WMV2_EVALUATION_REPORT.md`
after the run completes.
