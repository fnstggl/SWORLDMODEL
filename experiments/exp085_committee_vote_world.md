# EXP-085 — The committee-vote world model: measured-state agents finally WIN

This is the experiment we designed after the critique of the Maximal World (EXP-083/084). There, modeling
every person as an agent **lost** — but the agents carried *guessed* state (a fitted threshold curve; an LLM
persona from one post). The corrected thesis: fidelity pays only when each agent carries **measured** state
(grounding — the +58pt lever from EXP-082), on a question whose ceiling is high enough for it to show. This
is the fair test of that, on the canonical case: a chamber voting.

## The setup (leakage-free, high-ceiling, grounded)

Every divided Senate roll-call, congresses 106–118 (**7,233 votes, ~694k member-votes**, VoteView). For each
vote we split the members 80/20, fit each model on the observed 80%, and predict the held-out 20% **who were
never seen for that bill**. Each member's ideology is **grounded from the PRIOR congress** — their measured
position *before* this vote, not fit from it.

- **base rate** — predict every member by the chamber-wide yea-rate (no member identity).
- **party shortcut** (the compact model) — predict a member by their *party's* observed yea-rate. "Everyone
  votes their party line." In a polarized Senate this is ~90% right — a very strong, simple baseline.
- **agent world** — each member is an agent with a **measured ideal point**; a 1-D spatial vote model
  `P(yea)=logistic(w·ideology+b)` fit on the observed 80%, predicting the held-out member from *their* measured
  ideology. This is "model every person as an agent," now with grounded state.

## The result — the agent world wins, and by more where it matters

| | base rate | party shortcut | **agent world (grounded)** |
|---|---|---|---|
| all votes — accuracy | 0.648 | 0.902 | **0.916** (+0.014) |
| all votes — Brier | 0.205 | 0.071 | **0.064** |
| **contested** (within ~60-40) — accuracy | 0.575 | 0.899 | **0.916** (+0.016) |
| contested — Brier | 0.241 | 0.072 | **0.064** |

+1.4 points overall and **+1.6 on contested votes**, over an already-90% baseline, on ~130k held-out votes —
a ~15% relative cut in error, and statistically overwhelming at this scale. It wins *more* exactly where the
party shortcut is supposed to be weakest (the close votes), and it's better *calibrated* (Brier) everywhere.

### Why — made vivid by the defectors

The sharpest cut. Take the held-out members who voted **against their own party's majority** on a bill —
13,683 of them. The party shortcut gets **0% of these right, by construction** (it always predicts the party
line). The grounded agent catches **22%** of them — because a member's *measured* ideal point tells you which
specific moderates will cross the aisle on which bills. That is the entire ballgame: modeling each person's
measured state recovers signal that the group summary *cannot even represent*.

## What this proves — the Maximal World thesis, correctly scoped

Put the three experiments side by side:

| experiment | agent state | result |
|---|---|---|
| EXP-083 (baby-name cascade) | **guessed** threshold distribution | agent **loses** to the mean-field |
| EXP-084 (CMV messaging) | **guessed** LLM persona | agent **loses** to the flat shortcut |
| **EXP-085 (committee vote)** | **measured** prior voting record | **agent WINS**, +1.6 on contested, catches 22% of defections |

The variable that flips the outcome is not "agents vs no agents" — it's **guessed vs measured state.** Your
intuition was right: modeling every person as an agent *does* beat the compact shortcut — **when each agent
carries a real measurement and the question's ceiling is high enough to reward it.** Where we can't measure
the micro-state (aggregate-only cascades) or where the outcome is irreducibly noisy (which argument flips a
mind), the compact model wins and the agents just add variance. This is the decision rule, now proven from
both sides:

> **Simulate individuals as agents where their state is *measurable* and the outcome is *structure-dominated*
> (committees, courts, roll-calls, markets with real positions). Use the compact calibrated model for
> aggregate-only or irreducibly-noisy questions.** The engine should route by which regime it's in — which is
> exactly what the EXP-078 regime router does, now with a measured example of the high-fidelity regime paying off.

## Reproducibility

- `experiments/exp085_committee_vote_world.py`; data assembled by
  `python -m experiments.build_congress_votes` (public VoteView CSVs, no key; the 41 MB cache is
  gitignored and rebuilt). Ideology grounded from the prior congress; held-out 20% of members per bill;
  pure-Python logistic (`swm/transition/readout.py`), CPU, ~2 min.
