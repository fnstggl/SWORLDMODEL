# WMv2 Enron MAX-CAPACITY benchmark — forensic audit + verdict

**Reference World A, second round: the first genuine maximum-capacity run.** The prior "full V2" arm
(I7/I8) is renamed **`V2_METADATA_TEMPORAL`** (arm **E5** here) because it never interpreted message
content — it tested only metadata + the temporal hazard. This round adds a content-conditioned recipient
policy through the typed boundary (exact email text → DeepSeek → typed action distribution) and ablates
every experimental mechanism independently against the full max-capacity arm.

- Run artifacts: `experiments/results/wmv2_enron_maxcap.json` (metrics + paired CIs),
  `experiments/results/wmv2_enron_forensic.json` (20 per-prediction traces with raw LLM I/O).
- Runner: `experiments/wmv2_enron_maxcap.py`. Content mechanism: `swm/world_model_v2/reference/enron.py`
  (`content_multiplier`, `_CONTENT_PROMPT`, `v2_predict(..., content_fn=…)`).
- Model: **DeepSeek `deepseek-chat` (V3)**. 1,460 API calls, ~619k tokens (estimated chars/4),
  **~$0.42**, runtime ~36 min, particles=24, limit=60k messages.
- Data: 42,268 train / 11,206 time-forward test / 2,884 person-disjoint test; reply rate 3.4%; leak-free
  (features from strictly-prior messages; reply labels reconstructed from *later* messages, never shown to
  any model); fits on TRAIN only; Platt calibration fit on a train-only validation slice.

## The ladder (every experimental mechanism independently ablated)

| arm | what it is | content? | LLM? | sim mechanisms |
|---|---|---|---|---|
| **E0** | class base rate | no | no | — |
| **E1** | fitted metadata model (relationship>recipient>global × workload/hour/weekday × hazard) | no | no | — |
| **E2** | non-LLM hashed bag-of-words text baseline (fit on train) | text, no LLM | no | — |
| **E3** | grounded one-shot LLM, raw propensity → hazard | **exact msg** | yes | — |
| **E4** | call-matched direct LLM ensemble (3 reads pooled) | **exact msg** | yes×3 | — |
| **E5** | **V2_METADATA_TEMPORAL** (prior "full V2"): particle + event rollout, NO content | no | no | latent+event+relationship |
| **E6** | content policy only (latent OFF, event OFF) | **exact msg** | yes | — |
| **E7** | E10 − event rollout | **exact msg** | yes | latent+relationship |
| **E8** | E10 − latent | **exact msg** | yes | event+relationship |
| **E9** | E10 − relationship | **exact msg** | yes | latent+event |
| **E10** | **MAX-CAPACITY**: content + latent + event + relationship, all ON | **exact msg** | yes | all |

E6–E9 are strict leave-one-out variants of E10, so `E10 − Ex` isolates exactly one mechanism's marginal
contribution. E3/E4 are the *undisciplined* LLM baselines (propensity used directly), E10 is the same LLM
read wrapped as a bounded multiplier on the fitted base rate.

---

## PART 1 — the explicit questions, answered

- **Was DeepSeek called?** **Yes.** 1,460 `deepseek-chat` calls this run; every request is metered
  (calls/tokens/cost) and 20 are logged verbatim (prompt + raw response) in `wmv2_enron_forensic.json`.
- **Was any LLM called inside the V2 decision policy?** **Yes — in E3/E4/E6–E10.** The content policy
  (`content_multiplier`) issues one LLM read per example that becomes a bounded multiplier on the fitted
  hazard base rate inside `v2_predict`. **In E0/E1/E2/E5 no LLM is called** — E5 (`V2_METADATA_TEMPORAL`,
  the arm previously mislabeled "full V2") runs the particle/event world with a purely fitted decision
  operator (`reply_decision_fitted`). This is the correction the rename encodes.
- **Was the policy uniform / fitted / hand-coded / model-based?**
  - E1: **fitted** statistical (measured rates, no free constants).
  - E5: **fitted** decision operator inside a hand-built event world (no LLM).
  - E3/E4: **model-based** (LLM), used raw.
  - E6–E10: **hybrid** — fitted base rate × model-based (LLM) content multiplier, bounded to ≤~2.2× swing.
- **Which claimed V2 components did NOT execute in the max-capacity arm (E10)?** **None were disabled.**
  Content, latent attention (particle-sampled), event-driven rollout, and relationship persistence all ran.
  See the Full-Capacity omission log (PART 5) for the fields that remain *legitimately* absent (no
  defensible parameter source or would create leakage), which is a different thing from a silent disable.

### Per-prediction trace: how one prediction is actually computed

Every held-out prediction follows the same path, provable from the trace file:

1. **Typed WorldState materialized** at `sent_ts` — `recipient` entity with `attention` (latent) and
   `current_action=None`; clock `now=as_of=sent_ts`.
2. **Actor-observable view only** enters the policy: sender, prior-history count + reply rate with this
   sender, current inbox load, subject, body. No future messages, no labels. (`_CONTENT_PROMPT`.)
3. **One DeepSeek read** → `{"reply_propensity", "why"}` → bounded multiplier `0.45 + 1.75·prop` on the
   fitted base rate `p_base` (so prop 0.5 → ×1.0; content *redistributes*, cannot manufacture certainty).
4. **Particle rollout**: 24 particles, each samples `attention` from its prior, runs an event queue of
   `reply_check` opportunities under a per-opportunity hazard solving `1−(1−h)^n = H_bucket` (the
   "30 days ≠ 30 identical guesses" integration fix), emitting a terminal reply/no-reply + delay.
5. **Terminal readout**: `p_by[bucket] = fraction of particles that replied by that bucket`. Deltas logged
   per particle (operator `reply_decision_fitted`, `p_dist`, reason codes). Clock advances to horizon.

### Three worked examples (full traces in `wmv2_enron_forensic.json`)

**id 7 — the one reply in the forensic set (metadata wins).** Established correspondent (12 prior emails,
27% reply rate; recipient 16% overall), busy inbox (81/wk). Exact body: *"hey....at the high point of
direct access, how much of the load was off the system? Dot"* — a direct question.
DeepSeek returned `{"reply_propensity": 0.3, "why": "Low priority, not urgent, busy inbox"}` — it
*under*-rated a direct question. Fitted base `p_base = 0.514` → after content `0.501`.
`E1@7d = 0.506` (best), `E3 raw LLM = 0.295` (too low), `E10 = 0.458`. **Observed: replied in 0.9 h.**
Here the relationship metadata already knew this was a likely reply; content dragged it slightly the
wrong way. 24 particles, 2,217 deltas, `selected_action=reply`.

**id 1 — content over-confident, sim damps it (sim/metadata win).** Thin history (2 prior, 17%), empty
inbox. Body is an FYI about a regulatory letter. DeepSeek returned
`{"reply_propensity": 0.85, "why": "Relevant to Enron's regulatory interests."}` — high. Content pushed
`p_base 0.087 → 0.176`, but the event rollout with a low base produced 1 reply in 24 particles →
`E10@7d = 0.042`. Raw LLM `E3 = 0.837` (wildly overconfident). **Observed: no reply.** E1 (0.086) and E10
(0.042) correct; the *raw* LLM catastrophically wrong.

**id 0 — content and metadata agree down (both correct).** Heavy but low-yield pair (233 prior, 6%). A
forwarded news blast. DeepSeek: `{"reply_propensity": 0.05, "why": "Low reply rate, not directly
actionable."}`. `p_base 0.118 → 0.074`; `E1 = 0.117`, `E10 = 0.042`. **Observed: no reply.** Correct.

**Pattern across the 20:** DeepSeek's high-propensity reads (ids 1, 15, 17, 18, 19 at 0.85–0.9) mostly did
**not** result in replies; the single reply (id 7) came from a low content read but a high *metadata* base.
On this small forensic slice the raw content signal is noisy and slightly anti-correlated; the fitted base
rate carries the prediction. (The aggregate n=120 numbers below are the statistically meaningful test.)

---

## PARTS 3–4 — results (raw + calibrated), apples-to-apples

All arms on the **same** LLM subsample rows (n=120 time-forward, n=60 person-disjoint), so Brier is
directly comparable. Paired bootstrap CIs (1,000 resamples) are the decision statistic.

### Time-forward (n=120)

| arm | Brier@7d | logloss | AUROC | note |
|---|---|---|---|---|
| E0 base rate | 0.0866 | 0.342 | 0.500 | |
| **E1 fitted metadata** | **0.0577** | 0.218 | 0.824 | **the bar** |
| E2 text BoW | 0.0855 | 0.326 | 0.769 | worse than metadata |
| E3 raw LLM | 0.1638 | 0.488 | 0.758 | badly miscalibrated |
| E4 LLM ensemble | 0.1424 | 0.435 | 0.806 | still miscalibrated |
| E5 V2_METADATA_TEMPORAL | 0.0584 | 0.288 | 0.852 | ≈ E1 |
| E6 content only | 0.0663 | 0.566 | 0.744 | single-shot, poor horizon |
| E7 E10−event | 0.0669 | 0.566 | 0.746 | |
| E8 E10−latent | 0.0618 | 0.290 | 0.859 | |
| E9 E10−relationship | 0.0704 | 0.309 | 0.859 | |
| **E10 MAX-CAPACITY** | **0.0622** | 0.292 | 0.857 | **matches, does not beat E1** |

Calibrated (Platt fit on train-val, applied to test): E1 0.098→**0.0986** (already calibrated, no change);
E5 slope 0.60 → **0.101** (shrinks overconfidence, still worse than E1).

**Paired CIs @7d (n=120):**

| comparison | Δ Brier (E_a − E_b) | CI95 | verdict |
|---|---|---|---|
| **E10 vs E1** (max-cap beats the bar?) | **+0.00444** | [−0.00486, +0.01649] | **NS — does NOT beat** |
| **E10 vs E5** (content effect) | **+0.00376** | [−0.00286, +0.01295] | **NS — content adds nothing** |
| E10 vs E3 (sim-wrap beats raw LLM?) | −0.10161 | [−0.150, −0.061] | **sim-wrap crushes raw LLM** |
| E3 vs E2 (raw LLM beats text embed?) | +0.07829 | [+0.020, +0.139] | **raw LLM WORSE than BoW** |
| E10 vs E7 (event rollout) | −0.00477 | [−0.020, +0.012] | NS |
| E10 vs E8 (latent) | +0.00038 | [−0.002, +0.003] | NS |
| E10 vs E9 (relationship) | −0.00820 | [−0.020, +0.002] | NS |

### Person-disjoint (n=60, base ~2%)

Everything collapses toward the 2% base rate. E1 0.0164, E5 0.0182, E10 0.0165, E3 **0.269** (LLM
overpredicts replies from strangers catastrophically). Paired: E10 vs E1 +0.0001 [−0.0015, +0.0012] **NS**;
E10 vs E5 −0.00177 [−0.0056, −0.00003] (E10 marginally beats the *noisier* metadata-temporal arm, not the
bar); E10 vs E3 −0.253; relationship effect **exactly 0.0** (held-out persons have no prior relationship,
so the field is empty — the mechanism correctly contributes nothing).

---

## PART 7 — verdict

| question | answer | evidence |
|---|---|---|
| **Does exact message content add predictive value?** | **No** (on Brier) | E10 vs E5 +0.0038 [−0.003, +0.013] NS; E6 content-only (0.066) is *worse* than E1 (0.058) |
| **Does the LLM policy beat text embeddings?** | **No** | E3 vs E2 +0.078 [+0.020, +0.139] — raw LLM significantly WORSE than hashed BoW; AUROC comparable (0.76 vs 0.77) |
| **Does latent state add value?** | **No** | E10 vs E8 +0.0004 [−0.002, +0.003] NS |
| **Does event-driven rollout add value?** | **No on Brier** | E10 vs E7 −0.005 [−0.020, +0.012] NS — but it *does* fix the content arm's horizon calibration (logloss 0.566→0.29) |
| **Does relationship persistence add value?** | **No** | E10 vs E9 −0.008 [−0.020, +0.002] NS time-forward; exactly 0 person-disjoint |
| **Does max-capacity V2 beat the strongest direct baseline?** | **No** | E10 vs E1 +0.0044 [−0.005, +0.016] NS; point estimate slightly worse |
| **Does calibration improve V2?** | **Marginally** | Platt shrinks E5's overconfidence (slope 0.60) but does not reach E1; E1 is already calibrated |
| **Does the gain justify the cost?** | **No** | there is no significant gain; ~$0.42 + 36 min + 1,460 API calls vs ~0 marginal cost for the fitted model |

### The one real positive (methodological, not predictive)

The undisciplined LLM read **loses**: E3 Brier 0.164, worse than a bag-of-words baseline (0.086) and far
worse than fitted metadata (0.058), because the model systematically over-predicts replies (id 1: 0.85 for
a message that got none; person-disjoint E3 Brier 0.269). Wrapping that identical read as a **bounded
multiplier on the fitted base rate inside the typed world** converts it to parity (E10 0.062 ≈ E1 0.058,
Δ NS) — a **−0.10 Brier** rescue (E10 vs E3). **On this task the world-model boundary's value is calibration
discipline, not predictive lift.** Structured simulation once again *reproduces* a fitted statistical model
faithfully; it does not beat it.

### Honest limitation surfaced by the traces

E10's probabilities quantize to k/24 (the particle floor ≈ 0.042); for a low-rate task the event arm cannot
express probabilities between 0 and 4.2%, and many negatives read exactly 0.042 or 0.0. This handicaps the
event arm's Brier vs the closed-form E1 and is part of why simulation does not win here. More particles
narrow the floor at linear compute cost; that is a research lever, not evidence of lift.

---

## PART 5 — Full-Capacity omission log

**Principle adopted:** *the maximum-capacity arm uses every causally-relevant, defensibly-parameterized
field, mechanism and policy available; experimental status is not grounds to disable a feature in
experimental evaluation. Unsupported precision, leakage, and logically-invalid mechanisms remain
prohibited.* E10 disabled **nothing** it could defensibly run. Fields left out, with cause:

| omitted feature | reason | evidence missing | what would be required |
|---|---|---|---|
| full thread text beyond the immediate body | body truncated to 1,500 chars for cost/context | quoted-history parsing is lossy in Enron | robust thread reconstruction + longer context budget |
| recipient's *other* concurrent obligations (calendar, deadlines) | not in the maildir | no calendar/task ground truth | external calendar data joined to the mailbox |
| sender authority / org hierarchy | no reliable org chart in the public dump | title/manager edges unlabeled | an Enron org graph with confidence |
| explicit thread-level urgency labels | would be a supervised label, risks leakage | none available leak-free | independent, as-of urgency annotation |
| per-recipient learned response style (fine-tune) | no defensible per-person parameter source at n≈few | insufficient per-person history | a hierarchical model with shrinkage, future work |

None of these were *silently* disabled; each lacks a defensible parameter source or would introduce
leakage, which the principle explicitly still forbids.

---

## Status

**Architecture-validated + first-content-benchmark NO-EVIDENCE-OF-LIFT.** With message content now tested
(the one signal the prior round could not see), the defining claim — that structured simulation adds
held-out predictive value over a fitted statistical model — remains **undemonstrated on Enron reply
prediction**. All null and negative findings above are preserved. Next: run the same full-capacity
protocol on the remaining portfolio (Upworthy, ForecastBench V2-supported subset, BehaviorBench
interaction/persistence, OmniBehavior repair, Higgs/SEISMIC diffusion) per `WMV2_BENCHMARK_MAP.md`.
Do not merge PR #75.
