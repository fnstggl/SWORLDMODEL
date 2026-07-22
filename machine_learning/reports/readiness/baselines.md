# Baseline evaluation (non-learned reference numbers)

These are the floors the fine-tuned model must beat, computed on the real
in-domain / cross-dataset test splits. Generation tasks (messages, trajectories)
report only a trivial reference — meaningful eval requires the model.

| dataset | task | n_test | baseline | key metric |
|---|---|---:|---|---|
| psych101 | PREDICT_NEXT_CHOICE | 950 | majority-class(C) | accuracy=0.0526 |
| socsci210 | PREDICT_INTERVENTION_EFFECT | 7 | zero-effect | predict no effect; effect MAE requires model estimate vs realized arm |
| socsci210 | PREDICT_POPULATION_RESPONSE | 8 | mean-rate | no scalar rate in aggregate_metrics |
| open_bandit | PREDICT_NEXT_ACTION | 1992 | majority-class(recommend_item) | accuracy=1.0 |
| open_bandit | PREDICT_POLICY_VALUE | 1992 | mean-reward(0.0055) | reward_mae=0.01293 |
| criteo_uplift | PREDICT_INTERVENTION_EFFECT | 2000 | zero-effect | predict no effect; effect MAE requires model estimate vs realized arm |
| criteo_uplift | PREDICT_POPULATION_RESPONSE | 4 | mean-rate | no scalar rate in aggregate_metrics |
| persuasionforgood | PREDICT_FINAL_OUTCOME | 72 | majority-class(None) | accuracy=1.0 |
| persuasionforgood | PREDICT_NEXT_MESSAGE | 1468 | most-frequent-message | token_f1=0.0062 |
| persuasionforgood | PREDICT_RESPONSE_OR_NONRESPONSE | 745 | base-rate(0.976) | accuracy=0.9705 |
| casino | PREDICT_FINAL_OUTCOME | 100 | majority-class(None) | accuracy=1.0 |
| casino | PREDICT_NEXT_ACTION | 220 | majority-class(Submit-Deal) | accuracy=0.4955 |
| casino | PREDICT_NEXT_MESSAGE | 1157 | most-frequent-message | token_f1=0.0063 |
