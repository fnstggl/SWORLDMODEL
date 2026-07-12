# Behavioral foundation-model audit — source-grounded, with a hard honesty constraint

*Every candidate below was checked against primary sources (arXiv / GitHub / HuggingFace / dataset cards) by
three parallel research passes. **All eight candidates are REAL** — none fabricated. This environment has **no
GPU** (`nvidia-smi` absent), so Parts D/E/G pilots are **built and scripted, not executed** — per the hard
rule, nothing here claims an experiment ran. The decisive question — does a behavior-trained model predict
real human choices better than grounded DeepSeek on held-out outcomes — requires a rented 16-24 GB GPU; the
scripts to run it are in `experiments/behavior_pilot/`.*

## The one-line verdicts

| # | Candidate | Exists? | What it really is | Status |
|---|---|---|---|---|
| 1 | **OSim-8B / OdysSim** | ✅ | Conversational human/user simulator, base Qwen3-8B, **MIT weights, ungated, commercial-OK** | **P0 — first to test** |
| 2 | **Minitaur-8B** (= small Centaur) | ✅ | Forced-choice cognitive model (Psych-101), base Llama-3.1-8B, log-likelihoods | **P1 (P0 for choice tasks)** |
| 3 | **Centaur-70B** | ✅ | 70B sibling of Minitaur; needs 80 GB / 4-bit LoRA | later — do NOT start here |
| 4 | **Be.FM / Be.FM-1.5** | ✅ | Behavioral-economics model (games, Big-Five), Llama-3.1; **all weights GATED**, commercial unclear | P1 (pending access) |
| 5 | **BehaviorBench** | ✅ | Benchmark (public dataset `befm/BehaviorBench`) | benchmark only |
| 6 | **LCBM / Content Behavior Corpus** | ✅ | Content→behavior blueprint + dataset; **NO public weights** | training blueprint (+P1 data) |
| 7 | **OmniSapiens 2.0 / Human Behavior Atlas** | ✅ | Behavior *understanding* (affect/intent), Qwen2.5-Omni-7B; **dataset CC-BY-NC (non-commercial)** | P1 research / reject commercial |
| 8 | **OmniBehavior** | ✅ | Longitudinal realism benchmark (Kuaishou); **data/code NOT yet released** | benchmark only — inaccessible now |
| — | TRIBE v2 | ✅ | fMRI encoder — comparison point only (see `AUDIT_PART_I_TRIBE.md`) | research-only |

## The connection to what we just measured (read this first)

EXP-098 just showed that our **DeepSeek stakeholder simulation is *worse* than one grounded call on
deliberation forecasting** (indep_stake Brier 0.135 vs grounded_1shot 0.088; interaction made it worse still).
So **swapping DeepSeek stakeholders for OSim stakeholders on deliberation would be polishing a component that
isn't adding value in the first place** — the wrong place to spend the first GPU budget.

But OSim/OmniBehavior target a **different** class: individual & longitudinal *behavior realism* (reply,
engagement, abandonment) — exactly the product's individual-prediction / diffusion use cases. And OmniBehavior
independently documents the failure mode that matters most: **general LLMs are hyper-active and positivity-
biased — they predict 40–60% action rates where real users act ~10% of the time**, and they homogenize
personas. *That* is a behavior a specialist model might fix, and it is measurable. **So the honest pilot is
OSim-vs-DeepSeek on individual-response / engagement realism, NOT on deliberation forecasting.**

## PART A — the audit (facts only, sources inline)

**OSim-8B / OdysSim** — arXiv [2606.14199](https://arxiv.org/abs/2606.14199) (CMU-LTI, Sap et al.; June 2026,
post-cutoff → fetch-verified only). Weights [`cmu-lti/osim-8b`](https://huggingface.co/cmu-lti/osim-8b), base
**Qwen3-8B**, **MIT**, ungated; code [`sunnweiwei/OdysSim`](https://github.com/sunnweiwei/OdysSim) Apache-2.0.
Generates human conversational turns conditioned on the other party; **conversational, not forced-choice**;
system prompt carries social grounding → **accepts free-form stakeholder dossiers**. Trained on the OdysSim
corpus (62 datasets, ~21.4M interactions). 23-task "Soul-Index" (reports 64.6 avg, competitive with frontier
— *authors' claim, uncorroborated*). ⚠️ **No documented calibrated-probability interface**; **no built-in
longitudinal/multi-agent state**; WEIRD/English bias (paper: "a behavioral baseline, not a representative
human sample"). 8B BF16 ≈16 GB (fits 24 GB; ~8 GB 4-bit); **no official quant, no hosted endpoint**; also 4B/
0.6B variants. Upstream corpus mixes non-commercial sources — the **released MIT 8B weights are the safe path;
do not redistribute the corpus** without per-source review.

**Minitaur-8B** = [`marcelbinz/Llama-3.1-Minitaur-8B`](https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B)
— the 8B sibling of **Centaur** (Binz & Schulz, *Nature* 2025,
[s41586-025-09215-4](https://www.nature.com/articles/s41586-025-09215-4); arXiv
[2410.20268](https://arxiv.org/abs/2410.20268)). Base **Llama-3.1-8B**, trained on **Psych-101**
(10.68M choices, 60k participants, 160 experiments). Input = natural-language experiment transcript, human
choices marked; output = **token log-likelihoods over choices** (native choice-probability interface —
better-suited to calibration than OSim). **Forced-choice, individual-level.** Weights: **Llama-3.1 Community
License** (non-OSI, <700M-MAU clause); Psych-101 **Apache-2.0**, but the **test split is CC-BY-ND & gated**.
Runs on a free Colab GPU (~16 GB BF16, community 4-bit MLX quants exist). Paper: "generalizes less robustly
OOD than the 70B."

**Centaur-70B** — [`marcelbinz/Llama-3.1-Centaur-70B`](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B),
ungated, needs **80 GB** (or single-80 GB via 4-bit LoRA adapter). Same recipe/data as Minitaur. **Do not
start here** (budget rule).

**Be.FM / Be.FM-1.5** — arXiv [2505.23058](https://arxiv.org/abs/2505.23058) (MobLab + Stanford/Jackson +
UMich/Mei). Base Llama-3.1; predicts economic-game choices, Big-Five, demographics; **strong distributional
alignment**. HF org [`befm`](https://huggingface.co/befm): `Be.FM-8B/70B`, `BeFM1.5-4B/70B` — **all gated**
(Google Form + Be.FM Terms of Use; **commercial use not clearly granted**). 8B/4B fit 24 GB *if access is
approved*. No calibrated-probability head advertised.

**BehaviorBench** — arXiv [2606.24162](https://arxiv.org/abs/2606.24162); public dataset `befm/BehaviorBench`
(ungated). 4 capabilities (prediction/simulation, strategy, trait inference, knowledge). Finding: general
models win individual+knowledge; behavioral-FT (Be.FM-1.5) leads distributional alignment. **Use as an
evaluation set.**

**LCBM / Content Behavior Corpus** — arXiv [2309.00359](https://arxiv.org/abs/2309.00359) (ICLR 2024; Adobe +
IIIT-Delhi + SUNY). Base **Vicuna-13B**, **no checkpoint released** (HF org: "None public yet"). Dataset
[`content-behavior-corpus`](https://huggingface.co/datasets/behavior-in-the-wild/content-behavior-corpus):
24,595 videos, **MIT annotations** (underlying YouTube videos keep creators' licenses). Behavior signals are
**population-aggregate** (views, likes-%, scene replay curve) — **not** the rich per-event schema we
hypothesized (no clicks/purchases/replies in the released portion; those live in the unreleased email/Twitter
data). **A method + dataset, not a model.**

**OmniSapiens 2.0 / Human Behavior Atlas** — HBA arXiv [2510.04899](https://arxiv.org/abs/2510.04899) (ICLR
2026) + OmniSapiens (ICML 2026); [`MIT-MI/human_behavior_atlas`](https://github.com/MIT-MI/human_behavior_atlas);
weights [`HumanBehaviorAtlas/OmniSapiens2.0`](https://huggingface.co/HumanBehaviorAtlas/OmniSapiens2.0), base
**Qwen2.5-Omni-7B** (~9B, ungated, vLLM/SGLang, no quant). Reasons over multimodal behavioral signals
(emotion/sentiment/humor/sarcasm/intent) — **understanding, not action generation, not longitudinal**.
Dataset **CC-BY-NC 4.0 (non-commercial)** + upstream EULAs (DAIC-WOZ, MOSEI…). **Reject for commercial use.**

**OmniBehavior** — arXiv [2604.08362](https://arxiv.org/abs/2604.08362) (CAS + Kuaishou). 200 users × 3 months,
22 actions, 5 scenarios, traces 50→100k+ actions. Diagnoses **hyper-activity bias** (LLMs 40–60% vs ~10%
real), **positivity bias**, **persona homogenization** (intra/inter ratio 0.7–0.87 vs 0.29 real),
**over-engagement**. Best model Claude-Opus-4.5 = **44.55** (hard, unsolved). **Data/code NOT yet released**
("after data auditing"). The right realism benchmark — but not downloadable yet.

## PART A(2) — License & access matrix

| Model | Code | Weights | Dataset | Gated? | Commercial? | Fits 16-24 GB? |
|---|---|---|---|---|---|---|
| OSim-8B | Apache-2.0 | **MIT** | mixed (62 src) | no | **yes (weights)** | yes (16 GB) |
| Minitaur-8B | — | Llama-3.1 | Psych-101 Apache-2.0 (test CC-BY-ND) | no | yes* (<700M MAU) | yes (16 GB) |
| Centaur-70B | repro | Llama-3.1 | Psych-101 | no | yes* | no (80 GB) |
| Be.FM (all) | — | Llama-3.1 | proprietary | **YES (form)** | **unclear** | 4B/8B yes *if approved* |
| BehaviorBench | — | — | public | no | eval-only | n/a (dataset) |
| LCBM | LLaVA (MIT) | **none** | CBC MIT (videos vary) | no | data yes; no weights | n/a (no weights) |
| OmniSapiens 2.0 | MIT | unspecified | **CC-BY-NC** | no | **NO** | yes (9B) |
| OmniBehavior | pending | — | **unreleased** | n/a | n/a | n/a |

`*` Llama community license is not OSI-open; fine under the <700M-MAU clause + acceptable-use policy.

## PART A(3) — Hardware & cost matrix (rented GPU, self-host)

| Model | Params | VRAM (BF16) | VRAM (4-bit) | Min GPU | Quant available |
|---|---|---|---|---|---|
| OSim-8B | 8B | ~16 GB | ~8 GB | 1×24 GB (L4/A10/3090) | community only (bitsandbytes at load) |
| Minitaur-8B | 8B | ~16 GB | ~6-8 GB | free Colab T4/L4 | **yes (MLX 4-bit)** |
| Be.FM-8B / 1.5-4B | 8B / 4B | ~16 / ~9 GB | ~8 / ~4 GB | 1×24 / 16 GB | no official |
| OmniSapiens-7B | ~9B | ~18 GB | — | 1×24 GB (A10/L4) | no |
| Centaur-70B | 70B | ~140 GB | ~40 GB | 1×80 GB (4-bit LoRA) | Unsloth path |

**First GPU spend:** one 24 GB GPU (L4/A10G ≈ $0.5–1.0/hr, or 3090/4090). A 30–50 item OSim vs DeepSeek pilot
with ~10 samples/item ≈ a few GPU-hours + a few $ of DeepSeek API. **Cap the pilot: ≤50 items, ≤$20, stop if
the first 10–15 items show a severe incompatibility.** No 70B in the first pass.

## PART C — the shared adapter (BUILT)

[`swm/experimental/behavior_models.py`](../swm/experimental/behavior_models.py) — one interface
(`BehaviorRequest` → `BehaviorResponse`) over interchangeable backends. **`DeepSeekBehaviorBackend` is the one
runnable backend** (inject a chat fn; offline-testable). OSim/Minitaur/Be.FM/OmniSapiens are **stub backends
that REFUSE unless a real GPU runner is injected** — they never fabricate a decision. Disabled by default,
**quarantined** in `swm/experimental/` (a test pins that the production engine never imports it). This is the
apparatus for a fair same-inputs A/B once a GPU is available.

## Labeled evaluations — BUILT + DeepSeek baselines RUN here (OSim column added on the pod)

A realism probe can't show OSim *predicts* better (hard rule). So there are now three labeled harnesses on
real human outcomes; the DeepSeek arm was run in-sandbox, the OSim arm runs on the GPU pod with the same
prompts. **These committed baselines are the numbers OSim must beat.**

**1. Economic-game distributional alignment** — BehaviorBench `moblab/game_behavior` (real human choices).
Metric = Wasserstein-1 between the model's sampled choices and the human distribution (lower = more human).
`swm/eval/behavior_eval.py`, `experiments/behavior_pilot/behaviorbench_eval.py`.

| game | human mean | DeepSeek mean | W1_norm (↓) |
|---|---|---|---|
| bomb (risk) | 33.0 | **77.7** | 0.586 |
| dictator (fairness) | 36.5 | 40.0 | 0.092 |
| ultimatum_responder | 37.9 | 28.8 | 0.199 |
| **mean** | | | **0.293** |

The **bomb game is a smoking gun**: DeepSeek "opens 78 boxes" where real humans open 33 — far more
risk-seeking than people. *This* is precisely the human-misalignment a behavior-trained model should fix; it's
the cleanest place for OSim to show measurable value.

**2. Headline-click ranking** — Upworthy randomized A/B (real clicks; CC-BY). Metric = precision@1 (picked the
empirical CTR winner) + pairwise accuracy. `swm/eval/response_datasets.py`,
`experiments/behavior_pilot/upworthy_eval.py`.

| arm | precision@1 (random 0.34) | pairwise acc | n tests |
|---|---|---|---|
| DeepSeek | **0.56** | 0.652 | 50 |

DeepSeek already beats random (0.56 vs 0.34) at picking the winning headline — OSim must clear 0.56 to earn
the engagement claim.

**3. Individual reply + delay** — Enron (leak-free thread reconstruction, time-forward split);
`load_enron_reply_delay` + `time_forward_split` in `swm/eval/response_datasets.py`. Loader built + tested;
needs the ~1.7 GB Enron maildir (download on the pod) → metrics: reply-occurrence Brier/log-loss + response-
delay MAE. This is OSim's core claimed strength (#1).

**On the pod, add the OSim arm** (`OSIM_ENDPOINT` set): re-run all three; OSim only earns "keep" if it lowers
W1_norm (esp. on bomb), raises Upworthy precision@1 above 0.56, and/or beats DeepSeek on Enron reply — on
these untouched labels.

## OSim-8B pilot — MATCHED RUN (A40) — decisively MIXED, class-specific

Both arms, identical items (5 games × 20 items × 6 reps), one process → apples-to-apples. **OSim is more
human-aligned on economic-game distributions overall (mean W1_norm 0.191 vs DeepSeek 0.228), but loses the
product-relevant tasks.**

| game | human | DeepSeek W1↓ | OSim W1↓ | winner |
|---|---|---|---|---|
| bomb (risk) | 33 | 0.543 (opens 86!) | **0.186** (opens 46) | OSim, huge |
| ultimatum_proposer | 45 | 0.166 (offers 32) | **0.116** (offers 48) | OSim |
| guessing | 37 | 0.130 | **0.119** | OSim slight |
| dictator | 32 | **0.101** | 0.119 | DeepSeek slight |
| ultimatum_responder | 44 | **0.202** | 0.415 (accepts 2!) | DeepSeek, huge |
| **mean** | | **0.228** | **0.191** | OSim (−16%) |

Upworthy headline clicks (matched, n=100): DeepSeek precision@1 **0.45** vs OSim **0.35** (random 0.337) —
**OSim at chance; DeepSeek beats random.**

**Verdict:** (1) OSim genuinely captures human *economic/strategic* behavior distributions better than DeepSeek
— big wins on risk (bomb) and bargaining offers (proposer), wins 3/5 games. (2) OSim has a **catastrophic
over-compliance failure** as an ultimatum *responder* (min-accept ≈2 vs human 44) — a direct red flag for the
"will X accept/reply?" product question. (3) OSim is **at chance on headline-click ranking** — no value on the
engagement product task. **NARROW KEEP** (economic/strategic population simulation only); **reject** for
engagement ranking; **wary** for reply prediction. Neither task is the core deliberation-forecasting product
(EXP-098: simulation doesn't help there). Bootstrap CIs added to `behavior_eval` to confirm significance on
the next run; the individual reply+delay test (Enron) remains the unrun, most product-relevant experiment.

## OSim-8B pilot — FIRST RUN (A40, vLLM 0.24) — MIXED, not yet matched (superseded by the matched run above)

OSim-8B served on an A40 (needs `VLLM_USE_FLASHINFER_SAMPLER=0`) and ran all three evals. **Result leans
negative for adopting OSim as a general stakeholder model, with one real bright spot.** ⚠️ This first run was
**unmatched** — the OSim arm used `--limit 12 --reps 6` while the committed DeepSeek baseline used
`--limit 8 --reps 5`, so the item subsets differ (visible in differing human means). Treat as directional,
not decisive; the matched re-run (both arms, same flags, DeepSeek key on the pod) is the next step.

| signal | OSim | DeepSeek | read |
|---|---|---|---|
| bomb (risk) W1↓ | **0.274** (opens ~50) | 0.586 (opens ~78) | OSim clearly more human (~33) — its one clear win |
| dictator W1↓ | 0.127 (gives 43) | 0.092 | OSim over-gives — worse |
| ultimatum responder W1↓ | 0.375 (accepts ~3) | 0.199 | OSim accepts anything — worse |
| Upworthy precision@1 | 0.417 | 0.56 | OSim near random (0.34); DeepSeek much better |
| realism action-rate | 26% | 18.8% | both human-ish; OSim slightly higher |
| realism persona spread | 0.06 | 0.13 | **OSim homogenizes personas more** (OmniBehavior's failure) |

**Verdict so far:** OSim helps *only* on risk realism; it is equal-or-worse on fairness/bargaining, headline
clicks, and persona diversity. This argues against a general OSim swap and — if OSim is used at all — only for
the narrow risk/engagement-realism sub-task. Robustness is low (n tiny, no CIs, arms unmatched); the matched,
scaled re-run + per-game bootstrap CIs must land before any keep/reject is final. Per the hard rule, no
improvement is claimed from these unmatched, small-n numbers.

## PART D/E/G — the pilot (SCRIPTED, not run — no GPU here)

`experiments/behavior_pilot/` contains the runnable plan (see its README):
- **`run_osim_server.md`** — exact rented-GPU/Colab commands to serve `cmu-lti/osim-8b` via vLLM and expose an
  OpenAI-compatible endpoint the adapter's OSim runner calls.
- **`pilot.py`** — the harness: same dossiers/scenarios/stimuli/as-of/sample-count through arms **A** grounded
  DeepSeek, **B** DeepSeek stakeholder agents, **C** OSim agents, **D** mixed; **E** Minitaur on forced-choice
  items. Caches every output; caps items+spend; scores Brier/logloss/choice-accuracy/calibration + the
  **realism metrics from OmniBehavior**: action-rate vs real (hyper-activity), inactivity rate, persona
  heterogeneity (intra/inter distribution ratio), positivity bias. It **hard-stops with a clear message if no
  GPU/endpoint is present** — so it cannot pretend to have run.

## PART F — LCBM training-data blueprint for OUR flywheel

LCBM's real lesson: **verbalize behavior inline and train jointly with content** — but its released corpus is
video views/likes only. Our **forward ledger + flywheel already collect a richer per-event stream** than CBC
ships. The proprietary schema to log for a future content→behavior model (one JSON row per exposure):

```
sender/creator · recipient/audience segment · exact content (text/message) · channel · context ·
relationship history · exposure (shown/seen) · timestamp · action taken · click · reply · purchase ·
share · belief/sentiment change · delay-to-action
```

Most of these (content, channel, context, action, delay, reply) are already in the individual-response and
diffusion paths; **exposure, purchase, and sentiment-change are the missing columns**. **Do NOT train a model
yet** — the audit shows no license-clean labeled corpus is large enough, and EXP-098 shows the marginal value
of extra simulation machinery on our current tasks is unproven. Grow the ledger first; the schema above makes
it LCBM-ready without committing GPU now.

## PART H — Keep / reject

| Candidate | Decision | Why |
|---|---|---|
| **OSim-8B** | **PILOT (P0)** — do not promote yet | Only candidate that is commercial-OK, ungated, conversational, dossier-grounded, fits 24 GB. Must show **held-out lift on individual/engagement realism** (not deliberation) + no calibration/over-engagement regression before any promotion. |
| **Minitaur-8B** | **PILOT (P1)** for forced-choice | Native choice log-likelihoods (good for calibration); Llama license OK. Test on choice-shaped items only; weaker OOD. |
| **OmniBehavior** | **adopt as benchmark when released** | The correct realism yardstick (hyper-activity/positivity/homogenization). Watch for the data drop. |
| **BehaviorBench** | **use as eval set** | Public; distributional-alignment tasks. |
| Be.FM / Be.FM-1.5 | **hold** | Gated + commercial-unclear. Request access; do not depend on it. |
| Centaur-70B | **defer** | 80 GB; start with its 8B Minitaur sibling. |
| LCBM | **blueprint only** | No weights; population-aggregate; use the schema, not the model. |
| OmniSapiens/HBA | **reject for production** | CC-BY-NC + upstream EULAs; understanding not simulation. |
| TRIBE v2 | **reject for production** | Non-commercial; unlearned BOLD→behavior bridge (`AUDIT_PART_I_TRIBE.md`). |

**Promotion gate (all five required):** clean commercial license · measured held-out lift · acceptable
latency/cost · no calibration degradation · no increase in over-engagement / confident-wrong tails. **Today,
zero candidates clear it** — because none has been measured on held-out outcomes yet, and the one experiment
we do have (EXP-098) argues against adding simulation machinery on deliberation. The next real step is the
OSim pilot on the **individual-response / engagement** class, on a rented 24 GB GPU.

## Negative results preserved

- Swapping stakeholder backends will not help **deliberation** forecasting — EXP-098 shows the stakeholder
  layer itself is net-negative there. Any OSim value must be sought on individual/engagement realism.
- No license-clean, downloadable dataset today supports **cold-outreach reply**, **causal best-action with
  message text at scale**, or **event-timed opinion change** (see `DATASET_REGISTRY.md`). OmniBehavior would
  help but is unreleased.
