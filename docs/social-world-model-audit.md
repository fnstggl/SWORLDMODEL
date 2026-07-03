# Social World Model — Deep Research & Build Audit

**Status:** Research draft. Written to be brutally realistic, not promotional.
**Framing rule for the whole document:** we are building *calibrated, backtested, partially-observed probabilistic prediction of social responses* — not "prophecy," not "map every variable," not "simulate the world." Every claim below is tagged as **[Established]**, **[Speculative]**, or **[Original research required]**.

---

## A. Executive Summary

**The one-sentence thesis.** A "social world model" is a probabilistic, partially-observed dynamical model of a social system that (1) *infers* latent states of people/populations (beliefs, preferences, attention, relationships) from observed behavior, (2) *predicts the distribution* of social responses to candidate actions, and (3) supports *counterfactual comparison* of those actions — and its only honest success metric is **calibrated accuracy on held-out, real, time-forward outcomes** plus **decision lift over the customer's current method.**

**What is real today [Established].**
- World-model RL (Ha & Schmidhuber; Dreamer line) proves you can learn a compressed latent state + a learned transition model and plan inside it — but in *games and control*, not human society.
- LLM "generative agents" (Park et al.) produce *believable* social behavior and are excellent for **qualitative prototyping and idea generation**.
- Recommender systems, CTR models, and uplift modeling already predict specific human responses (clicks, replies, conversions) with strong, backtested accuracy — they are the **baselines you must beat**, not blue sky.

**What is oversold [Speculative → currently unsupported].**
- "Silicon sampling" / synthetic respondents replacing real surveys and polls at population fidelity. Multiple 2023–2024 studies show LLM personas **flatten within-group variance, caricature subgroups, and are unstable to prompt wording.** Believable ≠ calibrated. This is the central scientific gap.
- Anything framed as "simulate the future" of elections, markets, or virality as a general engine. There is no evidence any system does this in a calibrated, generalizing way. Point-in-time hits (one election) are not evidence of a generalizing model.

**The strategic conclusion.** Do **not** start by trying to build a general social simulator. Start with a **narrow, supervised, backtestable wedge where the customer already has the outcome labels and already pays to guess them:** **outbound-message response prediction & optimization (email reply / marketing copy performance), beginning with a customer's own historical sends.** It has (a) clean binary/continuous outcomes with timestamps, (b) a trivially honest backtest (temporal holdout), (c) an incumbent baseline to beat (their A/B tests and their gut), (d) a private-data flywheel that compounds into a moat, and (e) a paying buyer. Generality, if it ever comes, is earned by stacking calibrated wedges — not asserted up front.

**The single biggest risk.** That "believable" gets mistaken for "accurate." The entire company must be organized around an **evaluation harness that can embarrass the product** — proper scoring rules, temporal splits, leakage audits, and lift-over-baseline — because the failure mode of this whole category is confident, well-narrated, uncalibrated nonsense.

**Two findings from the research that reshape the plan.**
1. **There is no public dataset that is simultaneously (a) real, (b) message-content-readable, and (c) carries an observed human-response outcome with timestamps.** The large outcome datasets (Criteo, Avazu CTR) are *content-blind* (features hashed for anonymization); the content-readable ones (Enron, Avocado) have *no open/click/reply outcome labels* (reply is only *derivable* from headers) and are *contamination traps*. The one clean, content-bearing A/B corpus — the Upworthy Research Archive (32,487 tests, public since 2021) — is almost certainly in every post-2021 LLM's training data. **Consequence:** the core thesis "an LLM reads your message and predicts the human response" cannot be validated end-to-end on public data. You must validate on a **customer's private outcome logs** or **your own instrumented, post-model-cutoff tests.** This is inconvenient for a demo but *excellent* for a moat — the only clean data is data you or your customer own.
2. **On the cleanest live forecasting benchmarks (ForecastBench, Metaculus AIB), skilled human forecasters still beat frontier LLMs at p < 0.001.** Temper every expectation accordingly; the honest posture is "calibrated decision support," not "superhuman oracle."

**Competitive reality (verified, public sources — see K.3).** The academic pioneers are already commercializing this with serious capital: **Simile** — founded by the actual Generative-Agents authors (Joon Sung Park, Michael Bernstein, Percy Liang) — raised a **~$100M Series A led by Index Ventures** (angels incl. Fei-Fei Li, Andrej Karpathy). **Aaru** raised a Series A (~$88M total, Redpoint-led) at a "$1B headline" (multi-tier) valuation on <$10M ARR, simulating populations for elections/consumer/policy questions. A solo founder **cannot** win by out-simulating Stanford or out-hyping Aaru. The only open lane is the one both under-emphasize: **rigorous, public, calibrated, contamination-controlled backtesting on a narrow paid wedge.** Be the credible one.

---

## B. Literature Map

> Citations are given with enough detail to locate the primary source. Where the competitor/literature research agents refine these, the reconciled version is in the companion brief files. Limitations are stated honestly.

### B.1 World models (the RL lineage) — **[Established, but out-of-domain]**
- **Ha & Schmidhuber, "World Models" (arXiv:1803.10122, 2018)**; peer-reviewed NeurIPS 2018 version **"Recurrent World Models Facilitate Policy Evolution" (arXiv:1809.01999).** Core ideas we borrow: a **V**ision model (VAE) compresses observations to latent `z`; an **M**emory model (MDN-RNN) learns the transition `p(z_{t+1} | z_t, a_t, h_t)` with a *mixture density* output (i.e., **probabilistic, multimodal futures**, not a point prediction); a tiny **C**ontroller plans/acts. An agent can be trained *entirely inside the learned "dream"* and transfer back — but only after a temperature/uncertainty hack to stop the policy exploiting model flaws (a lesson for us: policies exploit model error). **Takeaway:** latent state + learned stochastic dynamics + planning-in-model is a proven pattern. **Limitation:** CarRacing/VizDoom only, near-fully-observed, stationary dynamics, dense reward, cheap resettable simulation. Human society is none of those.
- **Hafner et al., Dreamer line: "Dream to Control" (ICLR 2020, arXiv:1912.01603); DreamerV2 "Mastering Atari with Discrete World Models" (ICLR 2021, arXiv:2010.02193); DreamerV3 "Mastering Diverse Domains through World Models" (arXiv:2301.04104, 2023; Nature 2025).** Recurrent State-Space Model (RSSM) with *separated deterministic + stochastic* latent; trains a policy by imagined rollouts; V3 uses one fixed hyperparameter config across 150+ tasks. **Takeaway:** the RSSM factoring (deterministic memory + stochastic latent) is the template for a social latent state. **Limitation:** control/games; reward is given, environment is simulable and resettable — social systems are not.
- **Survey to cite:** **"Understanding World or Predicting Future? A Comprehensive Survey of World Models" (arXiv:2411.14499, 2024; ACM Computing Surveys 2025).** Taxonomy of "understanding" (internal-representation) vs. "future-prediction" world models. The **generative/video branch** (DeepMind **Genie**, ICML 2024 best paper, arXiv:2402.15391; OpenAI Sora-as-"world simulator") is **contested** — generative video ≠ a model with calibrated dynamics (documented physics failures; LeCun's critique). **LeCun's JEPA** ("A Path Towards Autonomous Machine Intelligence," 2022; V-JEPA 2024, V-JEPA 2 arXiv:2506.09985) is an aspirational agenda, not a finished result. None of these are social.

### B.2 Generative agents & LLM social simulation — **[Established for *believability*; unproven for *calibration*]**
- **Park, O'Brien, Cai, Morris, Liang, Bernstein, "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023; arXiv:2304.03442).** 25 LLM agents in a sandbox with memory stream + retrieval + reflection + planning; produced emergent believable behavior (a party spreads by word of mouth). **Evaluated on believability, not predictive accuracy of real people.** This is the canonical "insight/prototyping" tool, not a prediction engine.
- **Park, Popowski, Cai, Morris, Liang, Bernstein, "Social Simulacra" (UIST 2022; arXiv:2208.04024).** Generate a *populated* prototype of an online community to surface anti-social failure modes before launch. Explicit purpose: **design-time what-if**, cheaply, for social computing systems. Directly relevant to "prototype a community/feature." Honest about being a design aid.
- **Park et al., "Generative Agent Simulations of 1,000 People" (arXiv:2411.10109, 2024).** Build agents from ~2-hour voice interviews with **1,052** real participants; the agents reproduce participants' GSS answers at **~85% of the rate the participants reproduce *themselves* two weeks later** (a *relative* self-consistency benchmark, not absolute accuracy). **The most serious calibration attempt to date** and the template for "persona-from-rich-interview." **This paper is now a company: Simile (see K.3).** **Limitation:** expensive interviews; reproduces *stated survey* responses, not real-world behavior over time; US-only; behavioral-experiment replication is weaker; privacy/deepfake concerns.
- **Aher, Arriaga, Kalai, "Turing Experiments" (ICML 2023)** and **Horton, "LLMs as Simulated Economic Agents" (2023).** LLMs qualitatively *reproduce classic effects* (Milgram, ultimatum, framing). **Limitation:** textbook directional effects ≠ quantitative population prediction.
- **DeepMind Concordia (Vezhnevets et al., 2023)** — generative-agent sim library with a game-master. Good engineering reference.
- **Large LLM-ABM platforms: OASIS (1M agents, arXiv:2411.11581), AgentSociety (arXiv:2502.08691), S3 (arXiv:2307.14984), GenSim (NAACL 2025), Project Sid (arXiv:2411.00114); SOTOPIA (ICLR 2024, arXiv:2310.11667) for social-intelligence *evaluation*.** Useful engineering prior art, but "reproduces real phenomena" almost always means *qualitative* pattern-matching, rarely calibrated point-prediction against held-out humans, and typically without control/non-LLM baselines or significance tests. **Validation is the open problem** — see **Larooij & Törnberg, "Validation is the central challenge for generative social simulation" (2025)**, which warns that LLMs may *reproduce* macro-patterns already in training data rather than *generate* them (a leakage threat to the whole enterprise).

### B.2.5 "Social world model" as a *term* — **[Nascent, 2025–2026; not an established field]**
The exact phrase is new and internally inconsistent — do not imply a settled subfield. Three incompatible meanings coexist:
- **The LLM-representation framing (the anchor):** **Zhou, Liu, Yerukola, Kim, Sap (CMU), "Social World Models" (arXiv:2509.00559, 2025).** Makes hidden intentions/beliefs/evolving social state *explicit* via **S3AP**, a structured per-timestep social-state formalism (agents' states, actions, perspectives, environment); reports gains on FANToM and SOTOPIA. **Cite this as the originating framing** — and our C.0 definition aligns with it.
- **The MARL/mechanism-design framing:** "Social World Model-Augmented Mechanism Design Policy Learning" (arXiv:2510.19270, 2025) — SWM = a layer inferring agents' latent traits from trajectories to design policy.
- **The Dreamer+ToM framing:** latent teammate-modeling that factors an RSSM latent into environment vs. partner components with a ToM head (2026 workshop work). Closely adjacent: **MuMA-ToM (AAAI-25 oral, arXiv:2408.12574)**, multimodal multi-agent ToM.
**Takeaway:** we adopt the Zhou et al. sense (explicit, structured, partially-observed social state) but instantiate it *pragmatically* around a backtestable readout rather than a full mental-state ontology.

### B.3 Silicon sampling / synthetic respondents — **[Contested; the crux of the whole category]**
- **Argyle et al., "Out of One, Many: Using Language Models to Simulate Human Samples" (Political Analysis 31(3):337–351, 2023; arXiv:2209.06899).** Introduced "algorithmic fidelity"; GPT-3 conditioned on demographics reproduces *some* correlational structure of ANES subgroups. The optimistic anchor — **but it predicted pre-2021 ANES distributions the model had effectively memorized, so it is partly a recall test, not a forecast.**
- **Santurkar et al., "Whose Opinions Do Language Models Reflect?" (OpinionQA, ICML 2023; arXiv:2303.17548).** LLM opinion distributions **skew toward higher-income/liberal/educated groups and misrepresent others**; steering helps only partially.
- **Durmus et al. (Anthropic), "GlobalOpinionQA" (arXiv:2306.16388, 2023).** Defaults resemble US/W-European opinion; country-prompting yields shallow stereotyped portrayals.
- **Bisbee et al., "Synthetic Replacements for Human Survey Data? The Perils of LLMs" (Political Analysis 32(4):401–416, 2024).** Means track ANES, **but variance is far too low, regression coefficients differ, and outputs are unstable across model versions and prompt phrasing.** The cleanest "means-OK, inference-broken" demonstration.
- **Dominguez-Olmedo, Hardt, Mendler-Dünner, "Questioning the Survey Responses of LLMs" (NeurIPS 2024; arXiv:2306.07951).** Apparent alignment is substantially an **option-order/labeling artifact**; LLM "responses" lack the entropy of real populations.
- **Cheng, Piccardi, Yang, "CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations" (EMNLP 2023; arXiv:2310.11501)** and **Wang, Morgenstern, Dickerson, "LLMs that replace human participants can harmfully misportray and flatten identity groups" (arXiv:2402.01908, 2024).** Simulations of political/marginalized groups are **flattened caricatures** reproducing *out-group stereotypes*. Reinforced by **"Take caution in using LLMs as human surrogates" (PNAS 2025).**
- **SubPOP (Suh et al., ACL 2025; arXiv:2502.16761)** is the current SOTA at predicting *response distributions* (fine-tuning cuts the human gap up to ~46%) — but its GSS/Pew sources are themselves public and now leaked.
- **General finding across the critiques:** LLMs **collapse within-subgroup heterogeneity**, **caricature/stereotype**, are **sensitive to prompt trivia**, and **drift across model versions.** *Implication:* synthetic respondents are usable for **hypothesis generation and coverage of the response space**, **not** as a calibrated substitute for real data on the metric that matters. **Any wedge that depends on population fidelity is high-risk** — and this is precisely the terrain Aaru and Simile are betting on (K.3).

### B.4 Theory of Mind in LLMs — **[Contested capability]**
- Pro: **Kosinski, "Theory of Mind May Have Spontaneously Emerged in LLMs" (2023)** — claims high false-belief performance. Con: **Ullman, "LLMs Fail on Trivial Alterations to ToM Tasks" (arXiv:2302.08399, 2023)**; **Shapira et al., "Clever Hans or Neural Theory of Mind?" (2023)**; **Sap et al., "Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs" (EMNLP 2022).** Benchmarks: **SocialIQA (Sap et al., 2019), ToMi, BigToM (Gandhi et al., 2023), FANToM (Kim et al., 2023).** **Takeaway:** LLM ToM is brittle and benchmark-sensitive; do **not** build a load-bearing hidden-mental-state module that assumes robust ToM. Treat inferred mental state as a weak, uncertainty-tagged prior.

### B.5 Agent-based modeling (the pre-LLM discipline) — **[Established methodology, hard to calibrate]**
- **Schelling (1971)** segregation; **Axelrod (1997)**; **Epstein & Axtell, "Growing Artificial Societies" (1996)**; Epstein's "generative social science." Diffusion/contagion mechanics we will actually reuse: **independent cascade / linear threshold models, SEIR-style spread, and Hawkes self-exciting point processes** for engagement/virality timing. **Takeaway:** ABM gives *mechanism and emergence* but is notoriously hard to calibrate to a specific real outcome; the LLM era swaps hand-coded rules for learned/LLM behavior but inherits the calibration problem.

### B.6 Social reasoning + planning that actually worked — **[Established, narrow]**
- **Meta, "CICERO" (Science, 2022): human-level Diplomacy** by combining an LLM dialogue model with explicit planning and modeling of *other players' intentions*. Proof that **modeling others' latent intentions + planning beats pure language modeling** in a genuinely social, adversarial game. A key architectural inspiration (separate the belief/intent model from the language surface).

### B.7 Backtestable social-prediction anchors — **[Established data + the contamination problem]**
- **Upworthy Research Archive (Matias, Munger, Aubin Le Quéré, Ebersole, *Nature Scientific Data* 8:195, 2021; DOI 10.1038/s41597-021-00934-7):** **32,487 headline A/B tests, 150,817 fielded packages, 538M impressions, 8.18M clicks** (Jan 2013–Apr 2015), free CC BY 4.0 on OSF (osf.io/jd64p). The rare clean, content-bearing, public **message-performance backtest** — but **public since 2021 → in every post-2021 LLM's training data.** (Exclude the ~22% of tests flagged with a June-2013–Jan-2014 randomization/cache `problem`.)
- **Ad CTR benchmarks: Criteo (4.37B rows) and Avazu (Kaggle).** Enormous labeled click data — the baseline turf of CTR models — **but features are hashed/anonymized, so there is no readable creative** (see D: the content-vs-outcome gap).
- **Forecasting benchmarks that are contamination-clean by construction: ForecastBench (arXiv:2409.19839, 2024) and Metaculus AI Benchmark.** Both score LLMs on events *unresolved at submission time*. **Sobering headline result: skilled human forecasters still beat the top LLMs (p < 0.001).** The methodology template for our own prospective harness (E).
- **Halawi et al., "Approaching Human-Level Forecasting with Language Models" (arXiv:2402.18563, 2024)** — the reference for *strict temporal holdout* (test only on post-cutoff questions).
- **Persuasion: ConvoKit corpora with real outcome labels** — Winning Arguments / r/ChangeMyView deltas (Tan et al., WWW 2016), IQ2 debates (pre/post audience votes), Persuasion-for-Good; plus **Anthropic's persuasion dataset (2024, explicit pre/post Likert delta).** Better-formed backtests than surveys — but all contamination-exposed (D, F).
- **ANES, GSS, World Values Survey, Pew** for opinion-distribution replication *research* (not a product), and only on **post-cutoff waves** (GSS 2024, ANES 2024, WVS-8) to avoid recall.

> The companion research briefs (competitors, recent literature, datasets, leakage/benchmarks) reconcile and extend these citations with URLs and dates; key URLs are consolidated in **Appendix M**.

---

## C. Architecture Proposal

**Design commitments (non-negotiable):**
1. **POMDP by construction.** Latent social state `s` is never fully observed; we only see partial, noisy `o`. No module may assume perfect information.
2. **Probabilistic everywhere.** Every prediction is a distribution with calibrated uncertainty, never a point "prophecy."
3. **Composition of a learned statistical core + LLM components + mechanistic dynamics** — not "an LLM prompt." LLMs are *feature extractors and behavior priors*, held accountable by a statistical head trained on real outcomes.
4. **The evaluator is a first-class, adversarial component**, not an afterthought.

### C.0 The formal object
A social world model is a POMDP-style tuple over a population `i = 1..N`:
- **Latent state** `s_t = { s_t^i }` per entity, plus a **shared context state** `c_t` (topics, norms, salient events).
- **Observations** `o_t` — partial, typed events (messages, clicks, posts, purchases, poll answers).
- **Actions / interventions** `a_t` — a candidate thing *we* do (send message X to segment S; launch product P; publish policy Q).
- **Transition** `p(s_{t+1}, o_{t+1} | s_t, c_t, a_t)`.
- **Inference** (filtering) `p(s_t | o_{1:t})`.
- **Readout** `p(y | s_t, a_t)` where `y` is the *business-observable outcome* we backtest (reply / click / conversion / vote share / share count).

We almost never need the full `s`. **We need `p(y | context, action)` to be calibrated.** That reframing is what keeps the project honest and shippable.

### C.1 Observation ingestion
- **Typed event schema** (not "all data"): `Event = {actor_id, timestamp, channel, type ∈ {message,click,open,reply,post,react,purchase,poll_response,...}, content_ref, target_ids[], features{}}`.
- Content is embedded once (text/image → vector) and stored with a content hash for dedup and **leakage tracking** (see E).
- Ingestion is **append-only and timestamped** so every training/eval split can be reconstructed as "what was knowable at time T." This is the single most important engineering discipline for backtesting.

### C.2 Entity / persona modeling
- **Persona = (structured traits) ⊕ (learned latent).** Structured: declared/observed fields (segment, tenure, past behavior aggregates, demographics *if legitimately available*). Latent: `θ_i ∈ R^d`, a learned embedding fit to the entity's behavior history (like a user embedding in a recommender, but shared across tasks).
- Two regimes: **known entities** (your CRM contacts — rich history → strong `θ_i`) and **cold/synthetic entities** (persona sampled from a segment distribution → wide uncertainty, explicitly flagged).
- **We do not claim to model a real named individual's inner life.** We model a distribution over responses conditioned on what we legitimately observe.

### C.3 Hidden-state inference (filtering)
- `p(s_t^i | o_{1:t}^i)` via **amortized inference** (an encoder RNN/transformer over the entity's event history → posterior over `s`), optionally a **particle filter** for the shared context `c_t` when multimodality matters.
- Output is a posterior, not a value. Downstream consumers must receive uncertainty.
- **[Original research required]** how much explicit latent structure (beliefs/attention/relationship) beats a black-box history encoder for the specific readout `y`. Start black-box; add structure only where it demonstrably lifts calibrated accuracy.

### C.4 Memory
- **Episodic:** the append-only event log + vector retrieval (the Park "memory stream," but for real entities).
- **Semantic:** periodically-refreshed summaries / running sufficient statistics per entity (recency, frequency, monetary, topic affinities, response-rate priors) — cheap, robust, and often the strongest features.
- **Parametric:** the fitted `θ_i` and the model weights.
- Memory writes are timestamped; retrieval at eval time is **restricted to pre-T events** by the harness.

### C.5 Social graph / context representation
- **Typed, weighted multigraph** `G = (V, E)`, `edge = {src, dst, type ∈ {follows, reports_to, friends, same_household,...}, weight, first_seen}`.
- Used for **diffusion/contagion** readouts (who exposes whom) via independent-cascade / Hawkes influence, and for **GNN features** on entities.
- For the first wedge (1:1 outbound), the graph is often trivial/absent — **do not build it until a wedge needs it.**

### C.6 Event representation
- Events (things that *happen* in the world: a news story, a competitor launch, a season) are first-class `c_t` drivers: `WorldEvent = {timestamp, type, embedding, salience, decay}`. They shift the shared context and let the model condition responses on "what else is going on." Start with a minimal, hand-curated event feed per wedge; expand later.

### C.7 Action / intervention representation
- **The most under-appreciated module.** An action must be *encodable* so the transition can condition on it and so we can compare novel actions we've never seen.
- `Action = {type, content (embedded), audience_selector, channel, timing, dosage}`. Example (email): `{type: send, content: subject+body embedding + structured features (length, CTA, personalization tokens, tone scores), audience: segment filter, channel: email, timing: dow/hour}`.
- **Treatment vs. control** is explicit: a comparison is `{a_treat, a_control}` on a matched audience. This is what makes `/compare-actions` and uplift honest.

### C.8 Transition model (the core)
Three composable mechanisms; use the simplest that backtests well:
1. **Direct discriminative readout** `p(y | context, action)` — a supervised model (gradient-boosted trees or a transformer head) over engineered + embedded features. **This is the workhorse and the first thing to build.** It is a "one-step world model": given state-summary + action, predict the outcome distribution. Boring, strong, backtestable.
2. **Mechanistic dynamics** for multi-step spread/timing: Hawkes / cascade / SEIR for *how a response propagates over time and network*. Used when the outcome is temporal/networked (virality, adoption curves).
3. **LLM rollout** (generative-agent simulation) for *insight and coverage of qualitative responses* (objections, sentiments, failure modes) and as a **feature generator** for (1) — never as the sole predictor of a backtested metric until it demonstrably beats (1).
- **[Original research required]:** where does LLM-agent rollout add *calibrated* lift over the discriminative head? Treat this as the core scientific question of the company, tested continuously.

### C.9 Uncertainty model
- **Aleatoric** (irreducible response randomness) from the readout's predictive distribution (e.g., Beta/Dirichlet head, or quantile/CRPS output).
- **Epistemic** (model ignorance) from **deep ensembles** and/or **conformal prediction** for distribution-free coverage guarantees.
- **Segment-level calibration** enforced and monitored (reliability diagrams per segment). A prediction ships with an interval and a calibration grade.

### C.10 Evaluator / backtesting harness (first-class)
- **Temporal, entity-disjoint splits.** Train on `< T`, predict for `[T, T+Δ)`, score against realized outcomes.
- **Proper scoring rules:** log loss / Brier (binary), CRPS (distributions), calibration error (ECE), plus **decision metrics** (uplift@k, expected value of acting on the model vs. baseline).
- **Baselines are mandatory and versioned:** always-predict-base-rate, logistic/GBM on simple features, the customer's own historical A/B win rate, and (for message tasks) a strong LLM-zero-shot baseline. **A wedge is only real if it beats all of these on a proper scoring rule.**
- **Leakage audit** (see E) runs as a gate in CI.
- **The harness must be able to make the product look bad.** If it can't, it's theater.

### C.11 Interface / API
- Stateless prediction endpoints + stateful "world" objects (a fitted model over a customer's population), versioned. Full spec in **Section J.**

### C.12 What we deliberately do NOT build early
No knowledge graph of "all variables." No global social graph. No general "simulate any scenario" box. No claim of individual mind-reading. These are where credibility goes to die.

---

## D. Data Acquisition Plan

**Principle:** the best data is data where **you (or your customer) caused the action and observed the outcome, with timestamps, and the model has not seen the outcome.** That is why *customer-owned funnel data* beats scraped public data for both accuracy and moat.

> **The decisive finding (verified across the dataset research): there is no public dataset that is simultaneously (a) real, (b) message-content-readable, and (c) carries an observed human-response outcome (open/click/reply) with timestamps.**
> - Datasets with **observed outcomes at scale** — **Criteo (4.37B rows), Avazu, iPinYou, KDD'12** — are **content-anonymized** (32-bit hashed features; even KDD'12's ad titles are hashed token IDs). There is *no creative for an LLM to read.*
> - Datasets with **readable content** — **Enron (~500K emails), Avocado (LDC2015T03, gated/paid)** — have **no open/click/reply labels**; reply is only *derivable* from headers, and Enron is a severe **contamination trap** (ubiquitously mirrored, 2000–2002 events the model already knows the ending of).
> - The one clean, content-bearing **A/B** corpus, **Upworthy**, is public since 2021 → **memorizable.** And on it, published LLM methods **barely beat random** at picking the winning headline (Ye et al., LOLA, *Marketing Science* 2024; arXiv:2406.02611) — engagement prediction is genuinely hard *even with* possible contamination.
>
> **Consequence:** the core thesis — *LLM reads a message, predicts the human response* — **cannot be validated end-to-end on public data.** You must use (1) a **customer's private outcome logs**, or (2) **your own instrumented, post-model-cutoff sends/tests**. Inconvenient for a demo; this *is* the moat (D + K.4).

| Data type | Realistic access (solo/tiny team) | Use | Risk |
|---|---|---|---|
| **Customer's own email/CRM/outbound history** | High — provided under contract by design partner | **Primary training + backtest** for the first wedge | PII/consent; must handle securely |
| **Customer's A/B test logs** | High — they already run them | Uplift ground truth, lift proof | Small N per test |
| **Upworthy Research Archive** (32.5K headline A/B tests + clicks; CC BY 4.0) | High — public (OSF) | Message-perf method dev & public credibility result | **Contaminated (public since 2021)**; exclude flagged ~22%; domain = news headlines |
| **Criteo (4.37B) / Avazu CTR** | High — public (Kaggle) | CTR baseline harness | **Content-blind (hashed)**; not "social"; Criteo negatives subsampled |
| **Hacker News** (BigQuery + Firebase API) | High — free, legally clean | Virality/engagement backtest (content + score + time) | Contamination (older posts); snapshot score-at-post via Firebase |
| **Reddit dumps** (Academic Torrents, 3.28TB) | Medium — torrent | Engagement/persuasion research | **Legal/ToS risk (May-2024 policy + DMCA); get counsel before commercial use**; contamination |
| **Twitter/X datasets** | **Low — avoid** | — | Academic API closed; redistribution violates ToS; mostly dehydrated |
| **ANES / GSS / WVS / Pew** | High (Pew: acct + 6-mo embargo) | Survey-distribution replication *research* | Opinion ≠ behavior; use **post-cutoff waves only** (GSS'24, ANES'24, WVS-8) |
| **Persuasion corpora (ConvoKit)** — CMV deltas, IQ2 pre/post votes, Persuasion-for-Good, Anthropic persuasion set | High — `pip install convokit` / HF | Persuasion backtest (real outcome exists) | Contaminated (old); ethics (see Zurich CMV scandal, L.7) |
| **Kickstarter (Web Robots, monthly)** | High — free | Demand-proxy backtest; **post-cutoff snapshots = leakage-safe** | Fuzzy license; funding ≠ social response |
| **Product Hunt / app-store** | Low/Medium — gated API/scrape | Demand research | No clean licensed corpus; build-it-yourself |
| **Synthetic (LLM-generated personas/responses)** | High — you generate it | **Augmentation, coverage, cold-start priors** | **Never** as eval ground truth for fidelity; self-confirmation |
| **News/event timelines** | High — public feeds | `c_t` world-event conditioning | Salience/decay must be learned |

**Sequencing.**
1. **Sign 2–3 design partners** who hand over historical outbound + outcomes. This is the whole game for the first wedge.
2. **Stand up public benchmarks (Upworthy, Criteo)** as an *always-on internal harness* so you can develop without a customer and prove method quality publicly.
3. **Use synthetic data only** for cold-start and coverage, always evaluated against real holdouts.
4. **Longitudinal data** (the dream for true dynamics) is expensive and slow; you *earn* it by running the product and logging outcomes — the flywheel, not a day-one purchase.

**Compliance is a feature, not a footnote:** consent, PII minimization, per-customer data isolation, deletion, and "no training on one customer's data to serve a competitor" are prerequisites to selling to any serious buyer.

---

## E. Evaluation / Backtesting Plan

This section is the product's spine. If it's weak, nothing else matters.

**E.1 Splitting.** Always **time-forward**: train/fit on events with `timestamp < T`, predict outcomes realized in `[T, T+Δ)`. For entity generalization, also hold out **unseen entities**. Report both "seen-entity, future-time" and "new-entity, future-time" — they answer different questions.

**E.2 Metrics.**
- Binary outcomes (reply/click/convert): **log loss, Brier, AUC-PR (for rare events), ECE/reliability diagram.**
- Distributional outcomes (share counts, vote share, response distributions): **CRPS, coverage of predictive intervals.**
- **Decision metrics (what the buyer cares about):** *uplift@k* (if you act on the top-k the model recommends, how much more outcome do you get than the baseline policy?), and *expected decision value*.
- **Calibration is a first-class metric, not a nice-to-have.** A model that's 70% accurate and calibrated beats a 75% model that's overconfident, for decision-making.

**E.3 Baselines (must beat all).** base-rate; RFM/logistic; GBM-on-features; customer's historical A/B win-rate / current targeting policy; strong-LLM-zero-shot. Publish the ladder.

**E.4 Leakage controls (the thing that kills credibility).**
- **Temporal leakage:** no feature may use post-T information. Enforced by the append-only log + a split-time cutoff applied at feature build.
- **Training-data contamination:** a pretrained LLM's weights may already "know" a historical outcome. **Standard n-gram/string decontamination provably fails** — paraphrase/translation bypasses it (Yang et al., "Rephrased Samples," arXiv:2311.04850). Mitigate by (a) preferring *recent, private, post-cutoff* outcomes the model can't have seen; (b) **contamination probes** — the "Time Travel in LLMs" completion test (Golchin & Surdeanu, arXiv:2308.08493): can the model reproduce the item / recover the answer with the input redacted?; (c) timestamp every benchmark item vs. the frozen model cutoff. Caveat: a naive pre/post-cutoff performance gap is itself noisy (models have temporal blind spots), so post-cutoff is *necessary but not sufficient* — you also need outcomes genuinely *unknown at prediction time.*
- **The gold standard is a *prospective* harness, not a retrospective one.** Copy ForecastBench/Metaculus mechanics: **register predictions before outcomes are known, resolve later, score with proper rules.** This is the only posture that fully survives the contamination critique — and it is exactly what customer live-holdout A/Bs give you for free.
- **Audience leakage:** no target-derived feature (e.g., "this person later unsubscribed") in the input.
- **Duplication leakage:** content-hash dedup across train/test.
- The leakage audit is a **CI gate**; a failing probe blocks release.

**E.5 Honesty protocol.** Every shipped claim states: task, split, metric, baselines beaten, N, and calibration grade. No "up to X%" marketing numbers without the split that produced them. **We separate two report types explicitly:** *Insight/Simulation* (qualitative, believable, not calibrated — e.g., "here are likely objections") vs. *Prediction* (calibrated, backtested, with intervals). Customers must never confuse the two, and the UI enforces the distinction.

---

## F. Best Initial Wedge Recommendation

### F.0 Wedge comparison

| Wedge | Backtestable? | Data access (tiny team) | Baseline to beat | Sell-ability | Verdict |
|---|---|---|---|---|---|
| **Email reply prediction (B2B outbound)** | **Excellent** (clean reply label + timestamp) | **High** (customer's own sends) | Their gut + A/B + rep intuition | **High** (revenue-linked) | **★ Start here** |
| Marketing message perf (open/click) | Excellent (Upworthy public + customer) | High | CTR models, A/B | High | ★ Strong second / same wedge |
| Sales objection simulation | Weak (no clean outcome label) | Medium | None (it's insight) | Medium | Insight add-on, not a prediction wedge |
| Consumer survey replacement | **Poor** (fidelity science shaky; contamination; means-OK/variance-broken) | Medium (post-cutoff waves only) | Real surveys (cheap enough) | High demand, high **risk** | Avoid as first wedge |
| Persuasion / opinion change | **Better-formed backtest** (real deltas/votes exist) but contaminated + ethics | Medium (ConvoKit/Anthropic set) | Polls, human RCTs | Medium; ethics-loaded | Research track, not first wedge |
| Political persuasion simulation | Poor + ethical/reputational minefield | Low/legal risk | Polls | Toxic risk | **Avoid** |
| Social virality prediction | Medium (noisy, heavy-tailed, leakage); **even SOTA barely beats random on Upworthy** | Medium (HN clean; Reddit legal) | Base-rate is brutal | Medium | Later, harder |
| Product launch demand | Poor (few shots, confounded) | Low | Sales forecasts | High value, low tractability | Later |

### F.1 Recommendation: **Outbound-message response prediction & optimization, starting with B2B email reply.**

**Why this wins:**
- **Outcome is clean and fast:** did they reply / positively reply, within a window? Binary/short-latency label with a timestamp. Perfect for time-forward backtest.
- **The customer already has the labeled history** (thousands of past sends with reply outcomes) → you can backtest on *their* data on day one, no cold start, no leakage from public pretraining.
- **There is a real incumbent to beat** (their A/B tests, their SDRs' intuition, generic "best time to send" folklore) → lift is measurable and dollar-denominated (replies → pipeline → revenue).
- **Tight loop → flywheel:** every send you score and every outcome you log improves the model *and* accrues to a proprietary dataset (data moat).
- **It is honestly a one-step social world model:** infer the recipient's latent responsiveness state from history, encode the candidate message as an action, predict the response distribution, compare candidate messages (`/compare-actions`), and explain drivers. You are building the real architecture — just on the tractable slice.

**Scope discipline:** predict `P(positive reply | recipient history, message, timing, context)` and *rank/optimize candidate messages & send-times*. Do **not** promise to write the perfect email or predict downstream revenue in v1.

### F.2 Wedge spec sheets

**Wedge A — B2B email reply prediction & message optimization (PRIMARY)**
- **Input schema:** `{recipient: {id, segment, past_events[], firmographics?}, message: {subject, body, links, personalization_tokens, tone/length features, embedding}, send_time: {dow, hour, tz}, context: {recent_touches, world_events?}}`
- **Output schema:** `{p_positive_reply: {mean, interval}, p_any_reply, p_unsub, calibration_grade, top_drivers[], recommended_variant}`
- **Baselines:** base-rate; RFM+logistic; GBM on features; LLM-zero-shot "will they reply?"; customer's current A/B pick rate.
- **Metric:** log loss + ECE for the probability; **uplift@k** for the optimization use ("acting on model's top variant beats current pick by X% replies"); CRPS if predicting reply-rate for a batch.
- **Dataset:** customer's historical sends+outcomes (primary); Enron/Avocado for structure/pretraining features (no outcome leakage); Upworthy for copy-performance priors.
- **Leakage avoidance:** temporal split at send-time; exclude post-send features; content-hash dedup; contamination probe on any LLM feature.
- **Prove lift:** offline temporal backtest beating all baselines on log loss + ECE; then a **live A/B (holdout)**: model-picked variant vs. customer's normal process → replies lift with confidence interval.
- **Sell it:** "We backtested on *your* last 12 months and beat your A/B pick rate by X% replies at equal send volume, calibrated. Here's the reliability diagram. 2-week paid pilot on your data."

**Wedge B — Marketing copy / subject-line performance (SECOND, same engine)**
- **Input:** creative variants + audience segment + channel. **Output:** predicted CTR/open distribution + ranked variants + intervals.
- **Dataset:** **Upworthy Research Archive** for *method development and a public credibility result* — but treat it as **contaminated (public since 2021)** and pair the real test with a **freshly-collected, post-model-cutoff corpus** (your own instrumented sends, or monthly Kickstarter/PH snapshots) so the headline claim is leakage-safe. + customer campaign logs. **Metric:** CRPS/log loss vs. realized rates, uplift@k, ECE. **Leakage:** temporal split; exclude Upworthy's flagged ~22% (2013–14 randomization bug); run a Time-Travel contamination probe before publishing any Upworthy number.
- **Sell it:** "Rank your creative before you spend; we prove it on a public A/B archive and on your history."

Both wedges share one architecture (C.7 action encoder + C.8 discriminative readout + C.9 uncertainty + C.10 harness). That shared core is the company.

---

## G. 30 / 90 / 365-Day Roadmap

### G.1 First 30-day prototype ("does the method beat baselines at all?")
- **Data:** Upworthy archive (public) + one design partner's historical email export (if signable) — otherwise Enron + Upworthy to start.
- **Build:** ingestion → append-only event store; action encoder for messages (embeddings + hand features); GBM discriminative readout; ensemble/conformal uncertainty; **the backtest harness with temporal split, log loss/Brier/ECE, and the baseline ladder + leakage gate.**
- **Deliverable:** a report that says, on a clean temporal split, "our readout beats base-rate/logistic/LLM-zero-shot on log loss and is calibrated (ECE = …)" — **or honestly reports it does not.** This report *is* the 30-day milestone. No UI needed yet.

### G.2 First 90-day research prototype ("calibrated lift, on a real customer, with a UI")
- **Data:** 2–3 design partners' outbound history under contract.
- **Build:** per-customer "world" object (fitted model + entity embeddings); `/predict`, `/compare-actions`, `/explain`, `/backtest` endpoints; a **dashboard** (not a chatbot) showing predicted reply prob + interval + drivers + calibration; per-segment reliability monitoring.
- **Science:** test whether LLM-agent rollout or explicit latent-state adds calibrated lift over the GBM readout (the core research question). Keep whatever wins on the harness.
- **Deliverable:** on each partner's temporal backtest, beat their A/B pick rate on uplift@k with calibrated intervals; run at least one **live holdout A/B** to show real lift. Ship the honesty protocol (Insight vs. Prediction separation) in the UI.

### G.3 12-month roadmap ("a real product + the first defensible moats")
- **Q1:** nail Wedge A on 3 customers; public Upworthy result for credibility (Wedge B method).
- **Q2:** productize (self-serve ingestion, versioned per-customer models, `/simulate` for multi-step send-sequence outcomes via Hawkes timing); expand to marketing-creative (Wedge B) for the same buyers.
- **Q3:** add **light social-graph / diffusion** only where a customer's outcome is networked (referrals, community); begin the **evaluation-benchmark moat** (a public, rigorously-split social-prediction leaderboard you author and win).
- **Q4:** multi-wedge platform (email + marketing + one adjacent), with the **data flywheel** (aggregated, privacy-preserving cross-customer priors that lift cold-start) as the emerging data moat.
- **Explicitly NOT in 12 months:** general "simulate any social scenario," election/market prediction as a product, population-fidelity survey replacement. Those are research bets, tagged **[Original research required]**, pursued (if at all) only after the wedge funds them.

---

## H. Open Research Problems (honestly enumerated)

1. **When does generative-agent rollout add *calibrated* lift over a discriminative readout?** [Original research required] — the central question. Possibly rarely; possibly only for novel actions with no historical analog.
2. **Population fidelity of synthetic respondents** — can prompting/fine-tuning fix variance-flattening and subgroup caricature enough to be calibrated, not just believable? [Contested, likely partial]
3. **Robust hidden-state inference for real people from sparse behavior** — how much latent structure helps vs. a black-box history encoder. [Original research required]
4. **Non-stationarity / distribution shift** — social dynamics drift (novelty decay, seasonality, world events); how to keep calibration under shift. [Hard, established as hard]
5. **Counterfactual validity for novel actions** — predicting responses to actions unlike anything in the training log (the actual value proposition of a "world model") without a proper experiment. [Original research required]
6. **Contamination-proof evaluation** of LLM components on historical social outcomes. [Methodologically open]
7. **Individual vs. aggregate** — where individual-level prediction is impossible but aggregate is feasible, and how to communicate that boundary. [Design + stats]
8. **Ethics/consent-preserving modeling** of persuasion without enabling manipulation. [Policy + technical]

---

## I. Specific Build Plan — Repo / Module Structure

Monorepo, Python core (ML) + a thin service layer. Structure mirrors the architecture so modules map 1:1 to Section C.

```
sworldmodel/
├── README.md
├── docs/
│   └── social-world-model-audit.md        # this document
├── swm/                                    # the library (importable core)
│   ├── ingestion/                          # C.1  typed event schema, append-only store
│   │   ├── schema.py                       #      Event, Action, WorldEvent dataclasses
│   │   └── store.py                        #      timestamped append-only log + as-of reads
│   ├── entities/                           # C.2  persona = structured traits ⊕ latent θ_i
│   │   └── embeddings.py
│   ├── inference/                          # C.3  amortized filtering p(s_t | o_1:t)
│   │   └── filter.py                       #      encoder posterior + optional particle filter
│   ├── memory/                             # C.4  episodic (retrieval) + semantic (RFM/summaries)
│   │   └── memory.py
│   ├── graph/                              # C.5  typed weighted multigraph + diffusion (built later)
│   │   └── diffusion.py                    #      independent-cascade / Hawkes
│   ├── actions/                            # C.7  action/intervention encoder (subject+body → features)
│   │   └── encoder.py
│   ├── transition/                         # C.8  the core
│   │   ├── readout.py                      #      discriminative p(y|context,action)  <-- workhorse
│   │   ├── mechanistic.py                  #      Hawkes/cascade multi-step
│   │   └── llm_rollout.py                  #      generative-agent sim (insight + features)
│   ├── uncertainty/                        # C.9  ensembles + conformal + calibration heads
│   │   └── calibration.py
│   ├── eval/                               # C.10 FIRST-CLASS: harness, baselines, leakage gate
│   │   ├── harness.py                      #      temporal/entity splits
│   │   ├── metrics.py                      #      log loss, Brier, CRPS, ECE, uplift@k
│   │   ├── baselines.py                    #      base-rate, logistic, GBM, LLM-zero-shot
│   │   └── leakage.py                      #      temporal/contamination/dup probes (CI gate)
│   └── worlds/                             #      per-customer fitted "world" object, versioned
│       └── world.py
├── api/                                    # J    FastAPI service
│   └── app.py                              #      /predict /simulate /compare-actions
│                                           #      /infer-hidden-state /backtest /explain
├── benchmarks/                             #      Upworthy + Criteo harness (public credibility)
│   └── upworthy/
├── experiments/                            #      dated notebooks/scripts; every result reproducible
├── tests/
│   └── test_leakage_gate.py                #      leakage gate must pass in CI
└── pyproject.toml
```

**Build order (maps to the 30-day plan):** `ingestion.schema` → `ingestion.store` → `actions.encoder` → `transition.readout` → `uncertainty.calibration` → `eval.*` (harness+metrics+baselines+leakage) → `benchmarks/upworthy`. Everything else is deferred until a wedge demands it. **The `eval/` package is written before, not after, the model.**

---

## J. API Spec

Two object types: **stateless prediction calls** and a **`world`** (a fitted model over a customer's population, versioned & immutable once published). All outputs are **distributions with intervals + a calibration grade**, never bare point values. All endpoints echo the `as_of` time to make leakage impossible to hide.

```
POST /v1/predict
  # p(outcome | context, single action). The one-step readout.
  req:  { world_id, entity_ref | entity_features, action, as_of }
  res:  { outcome: {name, p_mean, p_interval:[lo,hi]}, calibration_grade,
          drivers:[{feature, contribution}], model_version }

POST /v1/compare-actions
  # rank candidate actions on the same audience; honest treatment/control.
  req:  { world_id, entity_ref|segment, actions:[a1,a2,...], as_of }
  res:  { ranked:[{action_id, p_mean, p_interval, prob_best}], expected_uplift_vs_first }

POST /v1/simulate
  # multi-step rollout (sequence of actions, or diffusion over time/graph).
  req:  { world_id, initial_context, action_plan:[...], horizon, n_samples }
  res:  { trajectories_summary:{outcome_over_time:[{t, p_mean, interval}]},
          note: "SIMULATION — for insight; calibration grade per step" }

POST /v1/infer-hidden-state
  # posterior over latent state from observed history (explicitly a distribution).
  req:  { world_id, entity_ref, observations, as_of }
  res:  { latent_summary:{responsiveness:{mean,interval}, topic_affinities:{...}},
          uncertainty:"high|med|low", caveat:"inferred, not observed" }

POST /v1/backtest
  # run the harness on a temporal split; returns metrics vs. baseline ladder.
  req:  { world_id, task, split:{train_before, test_window}, baselines:[...] }
  res:  { metrics:{log_loss, brier, ece, crps?, uplift_at_k},
          baselines:{...}, beats_all_baselines: bool, leakage_gate:"pass|fail", n }

POST /v1/explain
  # drivers + counterfactual (“change subject → predicted +Δ, with interval”).
  req:  { world_id, prediction_id }
  res:  { drivers:[...], counterfactuals:[{change, delta_mean, delta_interval}],
          report_type:"prediction|insight" }
```

**Contract rules:** `/simulate` and `/infer-hidden-state` responses are tagged **insight** (believable, not guaranteed calibrated); `/predict`, `/compare-actions`, `/backtest` are tagged **prediction** (calibrated, with the split that produced the grade). The tag is machine-readable so the UI can't blur the line.

---

## K. Product Positioning

**Consumer/UI (Section 10 answer): a decision dashboard, not a chatbot.** The temptation is "ChatGPT for the future"; the *credible* v1 is a **prediction + comparison dashboard**: enter/choose an action → see calibrated outcome distribution, ranked variants, drivers, and a reliability diagram. A **scenario tree / timeline** view is right for the later `/simulate` multi-step feature, clearly labeled *insight*. A chat interface is a thin convenience layer on top, never the source of truth. The UI's job is to **make calibration and uncertainty legible** — that's the differentiator vs. a confident-sounding LLM.

**Enterprise product (Section 11 answer):** a **per-customer "world" fitted on their data**, delivered as (a) an API into their sales/marketing stack (score/rank sends & creatives pre-send), (b) a dashboard for operators, (c) a backtest/lift report for the economic buyer, and (d) governance (data isolation, calibration SLAs, audit logs). Priced on outcome lift (replies/conversions gained), which the backtest + live holdout make defensible. Land in RevOps/growth (they already A/B test and feel the pain), expand to broader "response modeling."

**Positioning line (honest):** *"Calibrated prediction of how people will respond to your messages, proven on your own history and on public A/B archives — so you decide before you send."* Not "simulate the future."

### K.3 Competitive landscape (verified from public sources; no private knowledge claimed)

**The pattern that matters:** the strongest players make the *biggest, least-backtested* claims. That is both the threat (they raise the capital and set the narrative) and the opening (few publish rigorous, contamination-controlled, calibrated backtests). Sources consolidated in Appendix M.

- **Simile** — **the direct threat, because it is the actual science.** Founded by the **Generative Agents authors — Joon Sung Park, Michael Bernstein, Percy Liang** (+ Lainie Yallen), i.e., the "1,000 People" paper commercialized. Raised a **~$100M Series A led by Index Ventures** (Bain Capital Ventures, Hanabi; angels **Fei-Fei Li, Andrej Karpathy**), ~early 2026. Approach: **AI voice-interview real people → build interview-grounded agents → simulate populations** for purchasing, earnings-call Q&A, policy. Reported clients **CVS Health, Gallup**; claims ~80% accuracy predicting earnings-call analyst questions (company-reported). **Implication:** do **not** try to out-simulate Stanford on generative-agent fidelity or interview-grounded personas — they will win that. Compete on the axis they under-emphasize publicly: *calibrated, backtested, decision-linked prediction on a specific paid workflow.*
- **Aaru** — **the hype anchor.** Founded March 2024 (Cam Fink, Ned Koh, John Kessler). Thousands of demographic/personality-seeded agents that "browse the internet" and update; predicts elections, consumer, pricing, policy. Product **Lumen**. **Series A led by Redpoint, ~$88M total raised, "$1B headline valuation" but multi-tier (blended < $1B), on <$10M ARR.** Clients reportedly EY, Accenture, IPG, McDonald's, Boston Beer, A24, Bayer. **All accuracy claims are company-sourced and none independently backtested**; the flagship "NY primary within 371 votes" claim later drifted to "~2,000 votes." Notably, a **cofounder's own pitch is "do not trust our model,"** and they concede they can't simulate complex individuals (Trump/Powell "too much variance"). **Implication:** their soft underbelly is exactly the thing we build the company around — independent, published calibration.
- **Synthetic-respondent / audience-sim field:** **Artificial Societies** (YC W25, $5.35M, networks of 300–5,000 personas; self-reported ~80% social-media-response accuracy), **Synthetic Users**, **Subconscious.ai** (discrete-choice "digital twins"), **Evidenza** (synthetic B2B audiences + "synthetic CMOs"; profitable), **Fairgen** (statistically *boosts a real sample* — the most methodologically honest), **Ask Rally, Lakmoos**. **Cautionary signal: Roundtable** (Princeton-PhD founder) **pivoted *out* of synthetic respondents into survey-fraud detection** — a credentialed team exiting the pure-synthetic thesis.
- **Adjacent (different mechanism, not direct competitors):** prediction markets/forecasting — **Metaculus, Polymarket, Kalshi, Good Judgment.** They aggregate human/crowd belief on *existing* resolvable questions; they cannot price *pre-launch counterfactuals* ("what if we send this?"). That counterfactual gap is the synthetic approach's real reason to exist — and where its validation is weakest.

### K.4 Moats — technical, data, evaluation (research Q17–19)

- **Technical moat [weak on its own].** Architecture is largely publishable/replicable; frontier labs and Simile out-resource any solo team on model quality. The defensible technical asset is not the model but the **plumbing that makes calibrated, leakage-free, per-customer prediction reliable and cheap** — the action encoder, the amortized state inference, and especially the harness. Real but modest.
- **Data moat [the strongest available].** Because *no public data* pairs readable content with observed outcomes (D), **proprietary content→outcome logs are genuinely scarce.** Every send you score and every outcome you log builds a corpus no one else has, per customer and (with privacy-preserving aggregation) across customers for cold-start priors. This compounds. **This is the moat to prioritize.**
- **Evaluation moat [the differentiator].** Own the credibility layer: author and win a **public, contamination-controlled, prospective social-prediction benchmark** with proper scoring rules. In a category defined by unverifiable claims, being *the reference for what "accurate" means* is a durable, narrative-defining position competitors have left open. Pair it with a reputation for calibrated honesty (publishing when you *lose* to baselines).

---

## L. Brutal Critique — Why This Probably Fails, and How to De-risk

**L.1 The category's original sin: believable ≠ accurate.** Generative agents and synthetic respondents are *compelling* and *wrong* in ways that are hard to notice, and the literature (Santurkar, Bisbee, Ullman) documents exactly this. **De-risk:** the evaluator is the company; never ship an uncalibrated number; separate Insight from Prediction in the product itself; win a *public* backtest (Upworthy) so the credibility isn't self-reported.

**L.2 You might not beat boring baselines.** RFM + GBM + "best send time" folklore are strong. If your fancy world model can't beat logistic regression on the customer's data, there is no product. **De-risk:** make "beat the baseline ladder on a temporal split" the 30-day go/no-go; kill or pivot the wedge if it fails. This is a *feature* of the plan, not a bug.

**L.3 Leakage will fake your early wins.** Pretrained-LLM contamination, temporal leakage, and dedup failures produce spectacular fake accuracy that collapses in production. **De-risk:** leakage gate in CI; prefer recent private outcomes; contamination probes; live holdout A/B before any lift claim.

**L.4 Cold start & data access.** Without a customer's history you have no clean labels; scraped social data is contaminated and ToS-encumbered. **De-risk:** design-partner-first GTM; public benchmarks (Upworthy/Criteo) to develop method without a customer; synthetic only for cold-start priors, never for fidelity claims.

**L.5 Non-stationarity.** Even a good model decays as behavior and context drift. **De-risk:** continuous re-fit, drift/calibration monitoring, per-segment reliability, and honest "confidence decays after N days" messaging.

**L.6 The generality trap.** The vision ("simulate anything social") is where credibility and money evaporate. **De-risk:** contractually resist scope creep; generality is *earned* by stacking calibrated wedges, each with its own backtest, or not attempted.

**L.7 Ethics / reputation.** Persuasion modeling shades into manipulation; population simulation shades into surveillance/dark patterns. The **University of Zurich covert r/ChangeMyView experiment (2024–25)** — bots persuading real users without consent — was retracted and the researchers warned; it is the cautionary boundary for any live social experimentation. **De-risk:** avoid political persuasion as a wedge; consent-based data only; no covert field experiments on people who haven't opted in; publish an acceptable-use boundary; make the product about *your own* outbound to people who can opt out.

**L.8 Competitive & narrative risk.** The pioneers themselves are commercializing with heavy capital: **Simile (~$100M, the actual Generative-Agents authors)** and **Aaru (~$88M, $1B headline)** — see K.3. You cannot out-fund or out-simulate them. **De-risk:** don't compete on narrative or general simulation; compete on the one thing they under-publish — **rigorous, public, contamination-controlled, calibrated backtests on a specific paid workflow**, and a data moat of proprietary content→outcome logs. Being the *credible, narrow, backtested* one is the wedge within the wedge. If they eventually publish rigorous calibration and move down-market into your workflow, that is the real existential threat — monitor it and stay ahead on the eval moat.

**Bottom line.** This can become a real company *only* if it is organized around evaluation and starts absurdly narrow. The most likely failure is building an impressive simulator that no one can trust because it was never made to embarrass itself. Build the embarrassment machine first; earn generality later; never sell prophecy.

---

## Claims to avoid (research Q20) — because they sound fake or are scientifically unsupported

Do **not** say, imply, or let marketing say:
1. **"We simulate the future" / "digital twin of society" / "we predict what people will do."** Deterministic-prophecy framing. Say: *calibrated probability distributions over specific outcomes, with intervals.*
2. **"Our synthetic respondents replace surveys/polls at population fidelity."** The literature (Bisbee, Santurkar, Dominguez-Olmedo, Wang, PNAS 2025) says LLMs match means but flatten variance, caricature subgroups, and are prompt/version-unstable. At most: *directional insight and hypothesis generation; validate against real data.*
3. **"We model any individual's mind / know what a named person thinks."** Brittle ToM (Ullman, Shapira); also a privacy/ethics landmine. Say: *a distribution over responses conditioned on legitimately observed behavior.*
4. **Point accuracy numbers without a split** ("94% accurate"). Always attach: task, temporal split, metric, baselines beaten, N, calibration grade, contamination check.
5. **"Backtested" on data the model was trained on.** That's a recall test. Reserve "backtested" for temporal/holdout or prospective evaluation with a contamination probe.
6. **"Superhuman prediction of human behavior."** On the cleanest live benchmarks, LLMs still lose to skilled humans. Don't claim what the field can't yet do.
7. **Causal/persuasion-lift claims from observational text data.** Requires an experiment (uplift/holdout), not a correlation on scraped content.
8. **Emergence/"the agents developed a real culture" as evidence of accuracy.** Believability ≠ calibration; emergence demos lack controls and significance tests.

---

## Appendix M — Consolidated sources (verified 2026-07-03; no private knowledge claimed)

*Company facts are from public reporting and company sites; accuracy claims by companies are flagged as company-sourced and not independently verified.*

**Foundational / literature.** Ha & Schmidhuber, World Models arXiv:1803.10122; NeurIPS arXiv:1809.01999 · Dreamer arXiv:1912.01603 / DreamerV2 arXiv:2010.02193 / DreamerV3 arXiv:2301.04104 (Nature 2025) · World-models survey arXiv:2411.14499 (ACM CSUR 2025) · Genie arXiv:2402.15391 · V-JEPA 2 arXiv:2506.09985 · Generative Agents arXiv:2304.03442 (UIST 2023) · Social Simulacra (UIST 2022, 10.1145/3526113.3545616) · 1,000 People arXiv:2411.10109 · OASIS arXiv:2411.11581 · AgentSociety arXiv:2502.08691 · SOTOPIA arXiv:2310.11667 · Concordia (DeepMind 2023) · "Social World Models" arXiv:2509.00559 (S3AP) · MuMA-ToM arXiv:2408.12574 (AAAI-25) · CICERO (Meta, Science 2022).

**ToM.** Kosinski arXiv:2302.02083 (→ PNAS 2024); Ullman arXiv:2302.08399; Shapira arXiv:2305.14763 (EACL 2024); Sap arXiv:2210.13312 (EMNLP 2022); Strachan, Nature Human Behaviour 2024; benchmarks FANToM arXiv:2310.15421, BigToM arXiv:2306.15448, SocialIQa arXiv:1904.09728, ToMBench arXiv:2402.15052, OpenToM arXiv:2402.06044.

**Silicon sampling & critiques.** Argyle, Political Analysis 2023 (arXiv:2209.06899); Santurkar OpinionQA arXiv:2303.17548; Durmus GlobalOpinionQA arXiv:2306.16388; Bisbee, Political Analysis 2024; Dominguez-Olmedo NeurIPS 2024 (arXiv:2306.07951); CoMPosT arXiv:2310.11501; Wang/Morgenstern/Dickerson arXiv:2402.01908; PNAS 2025 (10.1073/pnas.2501660122); SubPOP arXiv:2502.16761; Larooij & Törnberg (2025); Sharma sycophancy arXiv:2310.13548; Wang LLM-judge bias arXiv:2305.17926; Hofmann, Nature 2024 (arXiv:2403.00742).

**Contamination / forecasting benchmarks.** Time Travel arXiv:2308.08493; Rephrased Samples arXiv:2311.04850; Halawi et al. arXiv:2402.18563; ForecastBench arXiv:2409.19839 (forecastbench.org); Metaculus AIB (metaculus.com/aib).

**Datasets.** Upworthy Research Archive — Matias et al., *Nature Scientific Data* 8:195 (2021), 10.1038/s41597-021-00934-7, osf.io/jd64p · Criteo 1TB (ailab.criteo.com) / Avazu (Kaggle) · Enron (cs.cmu.edu/~enron) · Avocado LDC2015T03 · Hacker News (BigQuery `bigquery-public-data.hacker_news`; github.com/HackerNews/API) · Reddit Academic Torrents (academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4; Reddit May-2024 content policy — TechCrunch 2024-05-09) · ConvoKit persuasion corpora (convokit.cornell.edu); Winning Arguments arXiv:1602.01103 · Anthropic persuasion (anthropic.com/research/measuring-model-persuasiveness; HF Anthropic/persuasion) · LOLA/Upworthy LLM result arXiv:2406.02611 · Kickstarter (webrobots.io/kickstarter-datasets) · Zurich CMV scandal (Retraction Watch 2025-04-29) · Surveys: electionstudies.org (ANES), gss.norc.org (GSS), worldvaluessurvey.org (WVS), pewresearch.org/datasets, CES (Harvard Dataverse 10.7910/DVN/II2DB6).

**Competitors (public reporting).** Simile — $100M Series A led by Index Ventures (Bain Capital Ventures, Hanabi; angels Fei-Fei Li, Andrej Karpathy); founders Joon Sung Park, Michael Bernstein, Percy Liang, Lainie Yallen; clients CVS, Gallup (Bloomberg 2026-02-12; TechFundingNews; simile.ai) · Aaru — Series A led by Redpoint, ~$88M total, "$1B headline" multi-tier valuation, <$10M ARR; founders Cam Fink, Ned Koh, John Kessler; product Lumen (TechCrunch 2025-12-05; Semafor 2024-09-20; Fortune 2026-06-17; aaru.com) · Artificial Societies — YC W25, $5.35M (societies.io; Unite.AI 2025-08) · Synthetic Users, Subconscious.ai, Evidenza, Fairgen (TechCrunch 2024-05-09), Roundtable (pivoted to fraud detection) · Prediction markets: Metaculus, Polymarket, Kalshi, Good Judgment.

*Verification caveats: a few 2026 arXiv IDs surfaced via search metadata given the environment date; confirm exact author lists/venues before formal citation. All company valuation/accuracy figures are as publicly reported and, where company-sourced, not independently verified.*
