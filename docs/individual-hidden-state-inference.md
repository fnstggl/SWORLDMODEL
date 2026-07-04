# Individual hidden-state inference — the core problem

**Status:** design note for the hardest and most valuable part of the system (the audit's open
problem H.3). First-principles, deliberately anti-hype. The claims here are instrumental: latent
variables are judged **only** by whether they improve *calibrated, backtested* prediction of an
observable outcome — never as claims about a person's inner truth (which has no ground truth).

---

## 1. The problem, stated precisely

The outcome of one message to one person is:

```
y  =  f_i( message_features , context )  +  ε
        └── person i's RESPONSE FUNCTION ──┘     └ irreducible noise ┘
```

The whole game is that `f_i` differs per person, and the parameters that make it differ are
**latent** — not in any database. You observe a person only through the low-dimensional projection
of their past behavior; you must invert that to a posterior over the parameters of `f_i`, using
almost no per-person data.

### The regime shift (the key insight)
- **Aggregate regime** — "what's the reply rate for this segment?" Individual latents integrate
  out; you estimate a *mean*. Law of large numbers is your friend. Recommenders and silicon-sampling
  live here. Variance is your enemy and it collapses for free.
- **Individual regime** — "will *this* email land with *this* person?" The posterior over person `i`'s
  latents dominates, and its **width** is the entire question. You are not estimating a mean; you are
  doing per-person Bayesian inference from sparse evidence, and the honest output is often "this could
  go either way."

Formally, `Var(y) = Var_s(E[y|s]) + E_s[Var(y|s)]`. Aggregate work cares about the first term
shrinking across people. Individual work cares about making the posterior `p(s_i | evidence)` tight
enough that `E[y | s_i]` moves off the segment mean *by more than the noise floor*. If it can't, the
individual thesis fails for that task — and that is a measurable, early go/no-go (§7).

### Reframe #1: model the response function, not the person
"Gets mad easily" ≈ high **reactance / neuroticism** — but that label predicts nothing until it
*interacts with a message feature* (pushiness, presumption, fluff). So the object we infer is the
per-person **response function's parameters**, and the informative evidence is past
`(message_feature → reaction)` pairs, for this person or people like them. This is why the action
encoder (audit C.7) and the latent model are one system, not two.

### Reframe #2: demographics are priors, not causes
"Young, lower-SES" does **not** cause the reply. It shifts the *prior distribution* over the causal
latents (traits, stance, reading register). Treating demographics as causal and deterministic is
exactly what produces the caricature/variance-flattening failure the literature documents (audit
B.3). We use demographics only as a **weak, wide prior** that behavioral evidence overwrites.

---

## 2. The representation (concrete — no "map all variables")

A small, **factored** latent state per person, each factor a short interpretable vector with an
explicit posterior (mean + uncertainty). Not a giant opaque blob; not an unbounded ontology.

| Factor | Example dims | Timescale | Cheapest legitimate evidence |
|---|---|---|---|
| **Stable dispositions** `θ_trait` | reactivity, skepticism, need-for-detail, warmth, risk-aversion | months–years | their own past writing; operator description |
| **Response-style** `θ_style` | short-vs-detailed, formal-vs-casual, reading level, humor tolerance, fluff-intolerance | slow-drift | past `(msg→reply)` pairs; their reply length/register |
| **Sender-stance** `θ_stance` | trust, warmth, funnel-stage, prior-annoyance | per-relationship | thread history, prior opens/ignores |
| **Current state** `s_state` | busyness, mood, attention, salient recent event | hours–days | reply latency, time-of-day activity, recent events |
| **Demographic prior** `π_demo` | age, region, role, SES band | static | declared/CRM — used ONLY as a prior over the above |

Two design rules: (1) every factor ships a **posterior, not a value** — the width is load-bearing;
(2) factors are only allowed to exist if ablating them **hurts calibrated backtest accuracy**. A
latent that doesn't earn its keep on held-out outcomes gets deleted. This is what keeps the
representation honest and finite.

---

## 3. Two engines

### Engine A — Elicitation (human → machine): extracting tacit knowledge
The operator often *knows* the recipient in ways they can't articulate (Polanyi: "we know more than
we can tell"). You cannot hand them a form — that's the "map all variables" trap and it fails because
tacit knowledge is inarticulable and self-report is unreliable. So invert it: **the machine adapts to
the human**, not the reverse.

1. **Passive first.** Ingest what's legitimately available — the person's own past replies (richest
   signal by far: reading level, register, reactivity, stance, all at once), the thread, CRM notes.
   Form a *draft* posterior before asking anything.
2. **Correct-a-guess, not fill-a-form.** Show the inferred profile back as a falsifiable hypothesis:
   *"Looks like they prefer short, direct, low-fluff messages and are a bit skeptical — right?"*
   Humans are far better at **correcting** a guess than **generating** from a blank page
   (recognition ≫ recall). This is how you externalize knowledge the operator "doesn't fully know
   they have." Each correction is a labeled update to the posterior.
3. **Free-text → structured extraction.** Let them talk/write naturally about the person; an LLM maps
   the unstructured description to structured latent estimates **with uncertainty**. This is the one
   thing LLMs are genuinely reliable at here — turning tacit prose into features — as opposed to
   *being* the predictor.
4. **Forced-choice micro-probes** where uncertainty is high and consequential. Humans give unreliable
   absolute judgments but reliable comparative ones: *"Which opener fits them better, A or B?"*
   (discrete-choice elicitation).
5. **One high-value question, chosen by the machine.** Compute **value of information**: which single
   unknown, if resolved, would most change the prediction? Ask *only that*. Not a questionnaire — one
   surgical question. This is the elegant form of "the human transmits context easily": the machine
   spends the human's attention only where it moves the outcome.

### Engine B — Inference (machine infers latents from behavioral traces)
1. **Behavior is the projection of latent state.** Learn an amortized encoder
   `p(s_i | o_{1:t})` (audit C.3) that inverts observed behavior to a posterior. Start black-box; add
   the §2 factor structure only where it lifts calibrated accuracy.
2. **Their own language is the strongest cheap latent probe.** Embeddings of the person's prior
   replies carry register, reading level, emotional reactivity, values, and current stance
   simultaneously — and you have legitimacy because they wrote it to you. Prioritize this over
   demographic guessing (where both accuracy and ethics fail).
3. **Response-to-probe is gold.** How they reacted to *past variations* you sent is the most direct
   evidence about `f_i`. Five sends where only the short/direct one got a reply is strong evidence
   about `θ_style`. Sparse, but high-signal.
4. **Hierarchical partial pooling — the load-bearing technical idea.** Do **not** estimate each person
   from scratch (impossible with ~0 data) or force everyone to the segment mean (kills the individual
   signal). Put a hierarchical prior:
   `person_i ~ Normal(segment_s, Σ_within)`, `segment_s ~ Normal(population, Σ_between)`.
   - No data on a person → predict the segment mean, **wide** posterior (honest cold start).
   - A little data → **shrink** from segment toward the individual, in proportion to evidence.
   - You model the person's **deviation from their segment**, with calibrated uncertainty — which is
     exactly the fix for both cold-start *and* variance-flattening. This is the principled bridge
     between the aggregate and individual regimes, and probably the most important single choice in
     the whole system.

---

## 4. Uncertainty is the product, not a caveat

In the individual regime the **width** of `p(s_i | evidence)` is the deliverable. The system should
say *"this could go either way — here's what would narrow it"* vs *"we're fairly confident this
lands."* And it should close the loop: when the posterior is wide and the stakes are high, trigger
Engine A's value-of-information question. Wide-but-honest beats narrow-but-wrong for a decision-maker.

Distinguish the two uncertainties explicitly:
- **Epistemic** (we don't know this person yet) — *reducible* by more evidence / one good question.
  This is what VOI targets.
- **Aleatoric** (their kid was sick that morning) — *irreducible*. No latent inference beats it.

---

## 5. Honest ceilings (do not overclaim)

- **No ground truth for the latents.** You can never verify "he gets mad easily" directly — only the
  downstream outcome. So latents are **instrumental**; validate the *prediction*, not the psychology.
- **Identifiability is limited.** Many latent configurations produce the same behavior — "didn't reply
  because busy" vs "because hostile" are often indistinguishable from the trace. The posterior must
  stay **multimodal** and honest, not collapse to one confident story.
- **The individual noise floor is high.** A single email/text outcome is close to a coin flip for many
  people; aleatoric variance dominates. The achievable goal is to **shift the probability
  meaningfully and be calibrated about how much** — not to "know the answer." Anyone claiming
  per-person certainty is lying or leaking.
- **Ethics/consent.** Inferring psychological traits to target individuals is exactly where this gets
  dangerous. Prefer signals the person volunteered to you (their replies); operate on your own opt-in
  outbound; never covertly profile. (See audit L.7.)

---

## 6. What's established vs. genuinely novel here

- **Established:** amortized/Bayesian latent-variable inference; hierarchical partial pooling;
  discrete-choice elicitation; LLM structured extraction from text; VOI / active learning. Each is
  textbook. None is the risk.
- **The novel bet (our actual research):** that a **factored, LLM-populated, hierarchically-pooled
  per-person response-function posterior**, fed by *(their own text + operator correct-a-guess + one
  VOI question)*, **beats the segment mean on calibrated individual-outcome prediction, at realistic
  data sparsity** — and that the elicitation loop extracts enough tacit operator knowledge to matter.
  That compound claim is unproven and is the thing to prove or kill.

---

## 7. How to test it so you don't fool yourself

The one experiment that adjudicates the whole thesis:

> On held-out, time-forward outcomes, does `p(y | per-person posterior, message)` beat
> `p(y | segment mean, message)` on **log loss and calibration (ECE)** — and does the gap **grow**
> as you add per-person evidence (their text, a correction, a VOI answer)?

- If per-person **doesn't** beat segment → individual modeling is dead weight for that task; ship the
  segment model and stop. (Cheap, early, honest.)
- If it beats segment and the gap grows with evidence → the elicitation/inference loop is real, and
  each of the three evidence sources can be ablated to price its marginal lift (what's worth asking
  the operator for?).
- Always report the **noise-floor baseline** (best possible calibrated model given aleatoric limits,
  estimated from repeat sends) so "we're not better" is distinguishable from "no one could be."

This keeps the most seductive part of the product — "we understand the individual" — permanently
accountable to a number that can say *no*.
