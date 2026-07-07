# EXP-069 — deep per-person inference: the interview-gap lever, measured

**The question.** SOTA individual simulation (Park et al., *Generative Agent Simulations of 1,000 People*)
reaches ~85% normalized accuracy by conditioning an agent on a **2-hour interview** per person. We can't
interview everyone; our scalable analog is **deep multi-pass inference over a person's writing history**
(`swm/variables/deep_inference.py`). It was designed but only shallow-tested. This measures it on real data
and answers: *do we still need it?*

**Data.** 160 CMV authors, **8–25 documents each** (real writing histories), with per-document persona
signals (an agent swarm's output); 125 of those documents carry a real persuasion outcome. Leakage-free by
construction: a held-out action is predicted only from documents strictly *before* it.

---

## 1. Does depth help? — **yes, monotonically**

Predicting a held-out document's persona facets from the persona inferred from the person's *other prior*
documents:

| predictor | MAE |
|---|---|
| population baseline (everyone the same) | 0.0934 |
| deep persona (all prior docs) | 0.0907 |
| **confidence-blend (deep, shrunk toward population by 1−confidence)** | **0.0856** |

**The depth curve** — MAE as we condition on more of the person's history:

| prior docs | ≤1 | ≤2 | ≤4 | ≤8 | ≤16 |
|---|---|---|---|---|---|
| MAE | 0.0956 | 0.0931 | 0.0900 | 0.0843 | 0.0835 |

**Monotone: more history → a better model of the person** (−13% error from depth 1→16). This is the
interview-gap curve, on the scalable substitute: a person has a stable, learnable persona, and the deeper
you read their history the better you predict their behavior. The **confidence-blend is the best predictor**
(beats the population baseline by ~8%), exactly because the depth+consistency confidence math lets deep,
consistent traits move the answer while thin/noisy ones stay near the population — so it is now the default
`persona_to_vars` output and wired into the Level-1 response model.

## 2. Does it pay off downstream? — **outcome-dependent, and honestly weak here**

Predicting whether a challenger's argument *persuades* (the 125 labeled docs) from the challenger's persona:

| predictor | log-loss |
|---|---|
| base rate | 0.657 |
| shallow (1-doc persona) | 0.752 |
| deep (full history) | 0.739 |

Deep beats shallow (+0.013) but **neither beats the base rate** — the arguer's stable persona does *not*
predict whether a *specific* argument lands, because that outcome is driven by the argument and the *other*
person (the OP), not the arguer's disposition (and N=125 is small). This is the honest boundary: deep
persona improves the model of **who the person is**, which pays off when the outcome depends on *their*
disposition — not on an idiosyncratic interaction.

---

## Do we still need deep per-person inference? — **yes, for the right thing**

- **Yes as the accuracy lever for modeling a person.** Depth measurably and monotonically improves the
  person-model (Part 1), and the confidence-blend beats the population baseline — this is the scalable
  analog of the interview, and it works. It is now integrated: `DeepPersonaStore.vars_asof(entity, now,
  prior=…)` yields the confidence-blended persona traits, which feed straight into the Level-1 response
  model's receptivity (verified in tests: an open/humble history yields higher receptivity than a
  combative/certain one).
- **But it is not a universal accuracy multiplier.** It helps exactly when the target depends on the
  person's stable disposition (how *this* person will respond, what will move *them*), and not on noisy
  interaction outcomes where the other party and the specifics dominate (Part 2). So we pull this lever for
  the individual/single-agent mechanism, and we don't expect it to rescue outcomes that aren't
  disposition-driven.

**Magnitude, honestly:** the improvement over the population baseline is real but modest (~8% on facet MAE,
−13% from depth) — because a single document is a noisy realization of a trait, and we top out around what
the person's own consistency allows, mirroring why SOTA's ceiling is ~85% (people aren't perfectly
self-consistent either). Deep inference moves us toward that ceiling; it does not exceed it.

## Tests — `tests/test_deep_inference.py` (7, all pass)

Depth-scaled confidence monotonicity, `persona_to_vars` confidence-blend, and `vars_asof` leakage-free +
feeding the Level-1 response model. Full suite: **264 passed**.
