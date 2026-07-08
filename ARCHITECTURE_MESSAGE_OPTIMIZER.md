# Architecture — the message optimizer (find the *optimal* message action, not the best of a few)

## The problem with "LLM writes N drafts → rank → pick best"

That loop makes the **LLM the search operator** and the world model a **filter on the LLM's taste**. Three
structural failures follow:

1. **The search space is "things the LLM would write"** — a tiny, style-biased blob (chatty, hedgy,
   cover-letter-shaped). The global optimum almost certainly isn't inside it. You find the best of a bad basin.
2. **It optimizes in text space** — enormous, discrete, un-smooth — so you can only afford a handful of
   samples. A Monte-Carlo engine used to evaluate ~4 points is backwards.
3. **Generator and objective are the same model**, so its blind spots are invisible to itself.

The fix inverts it: **the world model becomes the objective function of a search; the LLM is demoted from
author to a constrained move-proposer. The email is *constructed* by the search, not written.**

## The key first-principles move: optimize in the space the model scores

The world model doesn't score words — it scores a **`VariableMap`** (`personalization`, `ask_directness`,
`pushiness`, `length_fit`, `clarity`, plus recipient-conditioned content-stance variables:
`credential_signaling`, `contrarian_pitch`, `secret_density`). That's a ~10-dimensional decision space, not
a trillion tokens. So we optimize **there**, text-free, and only decode to words at the end. No human draft
anchors the search → it finds the actual argmax, not a local optimum near someone's guess.

## The three layers

### L1 — Strategy optimization (`swm/decision/message_optimizer.py`)
Gradient-free **coordinate ascent + random restarts** over the message-controllable variables, maximizing
the objective for a **fixed recipient**. The space is small and the scorer is cheap, so we evaluate
**thousands** of strategy vectors (a human compares four). Output is a **`StrategySpec`** — the optimal
*strategy*, not an email — with per-variable driver attributions, e.g. *"credential_signaling → 0 because
this recipient's `status_orientation` is high."*

**The objective** (`swm/decision/strategy_scorer.py`) is `P(reply | recipient, strategy)` as an
**elasticity-prior logistic with recipient-conditioned interaction terms**. The interactions are the crux:
a plain linear readout cannot express *"credentials help a status-seeker but hurt a prestige-skeptic"* —
that's `credential_signaling × status_orientation`, and it's what makes the **same objective** recommend
opposite messages for different recipients. Each elasticity is a `WeightPrior` (mean + CI), so the scorer
returns a **distribution** of P(reply), not a point.

### L2 — Compositional construction (`swm/decision/compositional_search.py`)
"Build it sentence by sentence inside the simulation." An email is a sequence of **moves** (slots):
`opener → hook → thesis/secret → ask → close`. **Beam search**: at each slot a proposer offers a few short
candidate sentences, the world model scores every **partial** assembly (`encode_text_to_strategy` → scorer),
and the top beams extend. The proposer (offline **sentence bank**, or a real LLM `propose_fn`) only ever
emits a **local move** — there is no "write me an email" call — which is precisely why the chatty-AI-email
failure mode can't occur. The email is **selected into existence by P(reply)**.

### L3 — Monte-Carlo evaluation (`swm/decision/mc_evaluation.py`)
The honest `P(reply)` isn't a point — the recipient's base rate is a posterior, their traits carry
confidence, and mood/attention/timing vary. For each finalist we **draw the recipient's hidden state N
times** and score the reply under each draw; `P(reply)` is the **fraction of trajectories that reply**, with
an interval. Same machinery as the HN `/v1/simulate` "fraction that cross," repurposed to replies. This is
where *"we can simulate thousands"* is the entire point.

### L4 — Semantic critic gate (`swm/decision/semantic_critic.py`)
L1–L3 optimize *strategy*, but the scorer reads only lexical **variables**, so it is blind to two things
that make an email read as AI slop, and both live *below* the variable resolution:

- **Incoherence / embellishment** — a sentence can be dense with the right markers and still be nonsense.
  *"most of the AI stack rents margin it should own, and inference is where that flips"* scores high on
  `secret_density`+`contrarian_pitch` yet doesn't parse: vague referent ("that flips"), an abstract-noun
  metaphor ("rents margin it should own"), a tryhard opener.
- **Annoyingness** — *"One line back and I'll leave you alone"* scores as a low-friction ask but reads as a
  manipulative tic.

Meaning and tone are an LLM's strength and the variable readout's blind spot, so the critic is a separate
**adversarial** evaluator over the actual text, scoring two axes — **coherence** (concrete, parseable
claim?) and **naturalness** (not annoying/embellishing?). It runs in two places:

1. **Inside the beam search** — the *cheap lexical* critic penalizes any partial assembly containing a
   flagged line, so the email is never *built* toward slop.
2. **As the final gate + repair loop** (`polish_email`) — the *full* critic (LLM `judge_fn` if available,
   else lexical) flags each line; every flagged move is swapped for the cleanest on-strategy alternative
   from the proposer, then re-critiqued, up to N rounds. If no clean realization exists, the least-slop
   version is returned **with its flags surfaced** — honest, not hidden.

The lexical fallback uses a curated slop/annoyance lexicon plus structural incoherence heuristics (vague
referents, abstract-noun metaphors, buzzword density, missing concrete anchor), and every signal saturates.
A production `judge_fn` (an LLM sentence judge) replaces it for far higher fidelity.

## The guardrail (why it's a real optimum, not a Goodhart exploit)

Naively maximizing a flawed proxy finds the proxy's exploits (twelve question marks because
`ask_directness` pays). So:

- **Pessimistic lower bound** — the objective is the **q20 percentile** of P(reply) across the elasticity
  ensemble (`ScoreDist.lower_bound`), not the mean. A high-variance cheat can't win.
- **Ensemble** — every elasticity is sampled from its prior CI; agreement across draws is required.
- **Validity manifold** — every encoder signal **saturates** (`_sat`): repeating a trick stops paying, and
  `ask_directness` is capped and penalized past ~2 questions. Candidates are well-formed sentences.

## The LLM's job shrinks to bias-free roles (`swm/decision/llm_moves.py`)

The LLM never authors the email or picks the winner — the world model + beam search do. It fills three
constrained seams (all pluggable; a live `chat_fn(prompt)->text` such as
`swm/api/deepseek_backend.default_chat_fn` drives them, and everything degrades to the offline bank +
lexical critic when no key is set):

1. **Move proposer** — `spec_to_instructions` translates the L1 strategy vector into plain writing rules
   ("do not mention schools/awards/press", "lead with the non-consensus claim", "one short sentence, no
   urgency"); the LLM then writes K candidate sentences for **one slot**, given those rules + the recipient
   evidence + the sender's real facts (it may not invent beyond them). Called **per beam with the beam's
   prefix**, so each move continues the draft coherently and never repeats an earlier line. Slot roles are
   disjoint so the moves compose into one email, not four paraphrases.
2. **Sentence judge** — the LLM critic behind the L4 gate (coherent?/annoying? per line).
3. **Targeted rewriter** — when the critic flags a line, its *reason* is fed back to the LLM to rewrite
   *that line* plainer. This is the generator-level fix: resampling fresh candidates just returns the same
   slop register, so the repair must instruct the writer, not re-filter it. Repairs are ranked by the same
   (LLM) critic that flags them, so the ranker isn't blind to the issue the gate found.

A real lesson fell out of building this: **selection-based repair cannot fix a generator whose whole
candidate distribution is biased** (every sample tryhard/repetitive). You need generator-level correction —
sharper instructions plus a critic→writer feedback rewrite. That is why the register instruction ("being
contrarian is about the claim, not the tone; write plainly") and the targeted rewriter both exist.

**The optimizer authors; the LLM writes the parts it's told to and judges its own lines.**

## Honesty

Everything is **`unvalidated`**. The elasticities are **coarse world-knowledge priors**, not fitted to
reply outcomes — the analog of `llm_prior.prior_from_llm`, cached for reproducibility. **Trust the ranking
and the lever directions first; treat the absolute `P(reply)` as a claim to check.** `ELASTICITY_SCALE`
keeps the level sane but is explicitly not calibrated. Import labeled reply outcomes and fit the
elasticities (via `swm/variables/calibrated_weights.py`) to earn a real grade.

## Pipeline & map to the repo

`swm/decision/message_pipeline.py` orchestrates `recipient (World persona + public-figure web evidence) →
L1 → L2 → L3`, and runs naive drafts through the **same** evaluator so the lift is measured, not asserted.

- `swm/decision/strategy_scorer.py` — the objective (elasticity priors + interactions + ensemble).
- `swm/decision/message_optimizer.py` — L1 strategy search.
- `swm/decision/compositional_search.py` — L2 beam search + text encoder + sentence bank.
- `swm/decision/mc_evaluation.py` — L3 recipient-hidden-state Monte Carlo.
- `swm/decision/semantic_critic.py` — L4 coherence/naturalness/redundancy critic (beam penalty + gate + repair).
- `swm/decision/llm_moves.py` — live-LLM seams: move proposer, sentence judge, targeted rewriter.
- `swm/decision/message_pipeline.py` — end-to-end orchestration (offline bank or live LLM via `chat_fn`).
- Builds on: `swm/variables/schema.py` (message content-stance variables), `calibrated_weights.WeightPrior`,
  `swm/entities/public_figure.py` (recipient inference), `swm/decision/best_action.py` (best-arm racing —
  the natural home for racing L3 finalists next).

## Result on the Thiel case (`experiments/exp084_message_optimizer.py`)

L1 recommends: personalization/clarity/ask/length/contrarian/secret **high**, pushiness and
`credential_signaling` → **0** (the sign-flip fires on `credential_signaling × status_orientation`). L2
assembles a tight, contrarian, personalized, low-friction email — and **chooses to drop** the credential
opener, the awards hook, and the pushy ask, because the scorer rejects them. L3 evaluates it at ~0.55
reply-fraction with a wide `[0.21, 0.84]` interval, versus ~0.13 for a credential cover-letter and ~0.11 for
a pushy follow-up on the same evaluator. Ranking and lift are the trustworthy signal; the level is
`unvalidated`.
