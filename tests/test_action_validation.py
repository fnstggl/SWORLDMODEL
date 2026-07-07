"""Guard: the general action layer reproduces the validated +22pt best-message lift on REAL CMV data, and
its selection matches the old best_message path exactly. Skips if the committed CMV inference cache is absent."""
import os

import pytest

CMV = "experiments/results/exp021_cmv/cmv_inferences.json"


@pytest.mark.skipif(not os.path.exists(CMV), reason="CMV inference cache not present")
def test_new_action_layer_reproduces_cmv_best_message_lift():
    from experiments.exp069_action_layer_validation import run_cmv
    out = run_cmv()
    assert out["n_ops"] >= 15
    # the NEW generic best_action layer, fed the same validated model, matches the old best_message selection
    assert out["selection_parity_with_old_path"] is True
    assert out["new_layer_precision@1"] == out["old_path_precision@1"]
    # and re-earns the causal lift over a random pick (EXP-060: 0.739 vs 0.518)
    assert out["new_layer_precision@1"] - out["random_pick_rate"] > 0.1
    assert out["cate_sign_accuracy"] > 0.5                 # ranks the causally-better argument above chance
