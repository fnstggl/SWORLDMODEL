# EXP-069 — the general action layer + interventional scoreboard, re-earned on real data

The action layer (PR #34) reproduced best-message *shape* in unit tests. This experiment re-earns the
**scored** result on the real datasets, end-to-end through the new generic machinery — and runs the new
interventional scoreboard on genuinely randomized `do(x)` ground truth.

## A. CMV best-message through the NEW generic action layer

Fit the `StructuredResponseModel` (person × message) on the ChangeMyView temporal-train split (identical
setup to EXP-060). For each held-out OP that received several arguments with **mixed** outcomes (person
fixed, message varies — a natural experiment), select the best argument with the generic
`swm/decision/best_action.py` racing loop (each message a typed `do`-operator), and score with the new
`swm/eval/policy_regret.py` module.

| metric | value |
|---|---|
| mixed-outcome OPs | 23 |
| **NEW-layer precision@1** | **0.7391** |
| old `best_message` path precision@1 | 0.7391 |
| random-pick rate | 0.5181 |
| **causal lift over random** | **+0.221** |
| selection parity with old path | **True** |
| CATE-sign accuracy (pairwise) | 0.7692 |
| policy regret vs oracle | 0.2609 |

The generic action layer, fed the same validated per-person model, **exactly reproduces** EXP-060's
+22-point causal lift (0.739 vs 0.518) and picks the identical argument the old `best_message` chose on
every OP — so the new, more capable layer (typed interventions, best-arm racing, navigable output,
confidence + honest-tie) is a strict generalization, not a regression. CATE-sign 0.77 confirms it ranks the
causally-better argument well above chance.

## B. Upworthy interventional scoreboard on real randomized A/B

On the real Upworthy randomized headline archive (the observed CTR gap between arms *is* the causal effect),
a lexical headline→CTR policy picks an arm; the new `policy_regret` module scores it.

| metric | value |
|---|---|
| held-out experiments | 1476 |
| model policy CTR | 0.01679 |
| random / observed CTR | 0.01638 |
| oracle (best-arm) CTR | 0.0207 |
| fraction of achievable uplift captured | 0.096 |
| CATE-sign accuracy | 0.493 (chance 0.5) |

The lexical policy captures ~10% of achievable uplift and ranks arms at chance — reproducing EXP-054's
finding that the interventional task is **semantic, not lexical** (the semantic selector, EXP-056, captured
36.5% but needs an LLM). The point here is that the new interventional scoreboard (policy value / regret +
CATE-sign) runs correctly on genuine randomized `do(x)` ground truth — the honest KPI going forward, not
log-loss.

## Bottom line

The action layer's headline claim (+22pt causal best-message lift) is now re-earned on real data through the
generic layer, with proven selection parity to the validated old path; and the interventional scoreboard is
demonstrated on real randomized-intervention data. Run: `python -m experiments.exp069_action_layer_validation`.
