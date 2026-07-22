# SWORLDMODEL behaviour-ML — Final readiness report

**Ready for GPU training:** YES (smoke test passed; 8 datasets normalized+validated).

## Dataset status summary

- total datasets: **23**
- normalized + validated (train-usable): **8**
- normalized eval-only: **3**
- blocked / infrastructure / license-blocked: **5**
- status breakdown: `{'NORMALIZED_AND_VALIDATED': 8, 'ACCESS_BLOCKED': 3, 'CONVERTER_READY_STORAGE_BLOCKED': 7, 'INFRASTRUCTURE_ONLY': 1, 'NORMALIZED_EVAL_ONLY': 3, 'LICENSE_BLOCKED': 1}`
- exact example count (validated train-usable, as normalized here): **450,661**
- estimated tokens (validated train-usable): **~144,211,520**

> Note: several large datasets are CONVERTER_READY_STORAGE_BLOCKED — their converters are
> implemented + fixture/sample-tested, but full normalization is deferred until run on a
> large volume (see blockers.md + next_commands.md for the exact resume commands).

## Per-dataset

| dataset | final status | role | examples | critical_ok | train-eligible (if approved) |
|---|---|---|---:|---|---|
| acl_online_shopping | ACCESS_BLOCKED | ACCESS_BLOCKED | 0 | None | False |
| darpa_socialsim | ACCESS_BLOCKED | ACCESS_BLOCKED | 0 | None | False |
| mirobench | ACCESS_BLOCKED | ACCESS_BLOCKED | 0 | None | False |
| kuairand | CONVERTER_READY_STORAGE_BLOCKED | TRAIN_CANDIDATE | 0 | None | True |
| omnibehavior | CONVERTER_READY_STORAGE_BLOCKED | TRAIN_CANDIDATE | 0 | None | True |
| opera | CONVERTER_READY_STORAGE_BLOCKED | TRAIN_CANDIDATE | 0 | None | True |
| simbench | CONVERTER_READY_STORAGE_BLOCKED | CROSS_DATASET_EVAL_ONLY | 0 | None | False |
| some | CONVERTER_READY_STORAGE_BLOCKED | CROSS_DATASET_EVAL_ONLY | 0 | None | False |
| surge | CONVERTER_READY_STORAGE_BLOCKED | TRAIN_CANDIDATE | 0 | None | True |
| upworthy | CONVERTER_READY_STORAGE_BLOCKED | TRAIN_CANDIDATE | 0 | None | True |
| agentsociety | INFRASTRUCTURE_ONLY | INFRASTRUCTURE_ONLY | 0 | None | False |
| behaviorbench | LICENSE_BLOCKED | LICENSE_RESTRICTED_EVAL_ONLY | 0 | None | False |
| abcd | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 141,653 | True | True |
| casino | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 15,327 | True | True |
| dealornodeal | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 48,830 | True | True |
| debate | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 7,555 | True | True |
| open_bandit | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 120,000 | True | True |
| persuasionforgood | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 52,464 | True | True |
| psych101 | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 10,200 | True | True |
| werewolf | NORMALIZED_AND_VALIDATED | TRAIN_CANDIDATE | 54,632 | True | True |
| craigslistbargain | NORMALIZED_EVAL_ONLY | CROSS_DATASET_EVAL_ONLY | 106,922 | True | False |
| criteo_uplift | NORMALIZED_EVAL_ONLY | CROSS_DATASET_EVAL_ONLY | 4,004 | True | False |
| socsci210 | NORMALIZED_EVAL_ONLY | LICENSE_RESTRICTED_EVAL_ONLY | 315 | True | False |

## Training views (preview — before human approval, records are gated)

| view | datasets | manifest records | est. tokens |
|---|---|---:|---:|
| actor_choice_v1 | psych101|open_bandit|persuasionforgood|dealornodeal|casino|a | 101,226 | 32,392,320 |
| social_interaction_v1 | debate|persuasionforgood|dealornodeal|casino|abcd | 130,645 | 41,806,400 |
| long_horizon_behavior_v1 | debate|psych101|persuasionforgood|dealornodeal|casino|abcd | 35,181 | 11,257,920 |
| population_response_v1 |  | 0 | 0 |
| causal_intervention_v1 | open_bandit | 48,115 | 15,396,800 |
| unified_behavior_multitask_v1 | debate|psych101|open_bandit|persuasionforgood|dealornodeal|c | 315,167 | 100,853,440 |
| cross_dataset_evaluation_v1 | socsci210|criteo_uplift|craigslistbargain | 108,241 | 34,637,120 |

## Recommended first adapters

1. `8b_actor_choice` (actor_choice_v1) — the densest, cleanest signal (choices/actions).
2. `8b_social_interaction` (social_interaction_v1) — negotiation/persuasion/deduction messages.
3. `8b_unified_multitask` — once the specialized adapters validate, train the unified mixture.

## Recommended unified model mixture

`unified_behavior_multitask_v1` with temperature 0.6 + max-dataset-dominance 0.35 so no single
large dataset dominates; rare tasks floored at 5%. Non-commercial datasets (OmniBehavior, DND,
Criteo) stay out of any commercial view.

## Remaining human-review requirements

- Review each dataset's audit report (`reports/audit/<id>.md`) + human-review sample.
- Approve datasets in `registry/training_approvals.yaml` (nothing approved by default).
- Confirm licensing for commercial use if that is intended (several sets are non-commercial).

## Exact commands

- Smoke test: `python -m machine_learning.cli smoke run`
- First 8B run: `python -m machine_learning.cli train run 8b_actor_choice --launch`
- See `next_commands.md` for the full sequence.

## Unresolved risks

- Large datasets (OmniBehavior 6.4GB, OPeRA 8.6GB, SoMe 50GB+, SocSci210 1.45GB, Psych-101 859MB,
  KuaiRand 194MB-46GB) are NORMALIZED ONLY ON SAMPLES here — full runs need a large SWM_DATA_ROOT.
- SocSci210 has NO declared license and its `response` may be human or persona-simulated — eval-only.
- CraigslistBargain license is unstated (held out as eval-only).
- DEBATE / MiroBench / DARPA SocialSim / ACL-shopping data are not publicly released (blocked).
- The 8B base model default (Qwen2.5-7B) is ~7.6B, not exactly 8B; swap in Llama-3.1-8B if desired.

