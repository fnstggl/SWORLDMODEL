"""Task taxonomy, task families, and the special behavioural tokens.

These constants are the single source of truth mirrored by
``registry/task_taxonomy.yaml`` (a test asserts they agree). Converters, example
builders, the sampler, and the evaluation harness all import from here so the taxonomy
can never drift between code and config.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# The 16 supported prediction tasks.
# --------------------------------------------------------------------------------------
TASK_TYPES = [
    "PREDICT_NEXT_CHOICE",
    "PREDICT_NEXT_ACTION",
    "PREDICT_NEXT_MESSAGE",
    "PREDICT_NEXT_SPEAKER",
    "PREDICT_BELIEF_CHANGE",
    "PREDICT_PRIVATE_STATE_UPDATE",
    "PREDICT_TIME_TO_ACTION",
    "PREDICT_RESPONSE_OR_NONRESPONSE",
    "PREDICT_TRAJECTORY_CONTINUATION",
    "PREDICT_FINAL_OUTCOME",
    "PREDICT_POPULATION_RESPONSE",
    "PREDICT_POPULATION_TIME_SERIES",
    "PREDICT_DISCUSSION_TREE",
    "PREDICT_INTERVENTION_EFFECT",
    "RANK_CANDIDATE_ACTIONS",
    "PREDICT_POLICY_VALUE",
]
TASK_TYPE_SET = frozenset(TASK_TYPES)

# --------------------------------------------------------------------------------------
# Task families — used for cross-dataset evaluation design + balanced sampling.
# Every task family MUST have at least one held-out cross-dataset test.
# --------------------------------------------------------------------------------------
TASK_FAMILIES = {
    "individual_choice": [
        "PREDICT_NEXT_CHOICE", "PREDICT_NEXT_ACTION", "PREDICT_RESPONSE_OR_NONRESPONSE",
    ],
    "social_conversation": [
        "PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_SPEAKER", "PREDICT_BELIEF_CHANGE",
        "PREDICT_PRIVATE_STATE_UPDATE", "PREDICT_DISCUSSION_TREE",
    ],
    "long_horizon": [
        "PREDICT_TIME_TO_ACTION", "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME",
    ],
    "negotiation": [
        # negotiation reuses conversation/action/outcome task types; grouped for eval design
        "PREDICT_NEXT_MESSAGE", "PREDICT_NEXT_ACTION", "PREDICT_FINAL_OUTCOME",
    ],
    "population_response": [
        "PREDICT_POPULATION_RESPONSE", "PREDICT_POPULATION_TIME_SERIES",
    ],
    "intervention_effect": [
        "PREDICT_INTERVENTION_EFFECT", "RANK_CANDIDATE_ACTIONS", "PREDICT_POLICY_VALUE",
    ],
}


def family_of(task_type: str) -> str | None:
    """Return the primary family for a task type (first family that lists it)."""
    for fam, tasks in TASK_FAMILIES.items():
        if task_type in tasks:
            return fam
    return None


# --------------------------------------------------------------------------------------
# Special behavioural tokens. Inactivity, waiting, censoring, and unknown action spaces
# are FIRST-CLASS outcomes — never silently dropped because they are inconvenient.
# --------------------------------------------------------------------------------------
NO_ACTION = "<NO_ACTION>"          # actor deliberately took no action this step
NO_RESPONSE = "<NO_RESPONSE>"      # actor did not respond at all
WAIT = "<WAIT>"                    # actor waited / deferred
UNKNOWN_ACTION_SPACE = "<UNKNOWN_ACTION_SPACE>"  # available_actions is genuinely unknown
MISSING_TIMESTAMP = "<MISSING_TIMESTAMP>"
EPISODE_TERMINATION = "<EPISODE_END>"

SPECIAL_TOKENS = [
    NO_ACTION, NO_RESPONSE, WAIT, UNKNOWN_ACTION_SPACE, MISSING_TIMESTAMP, EPISODE_TERMINATION,
]

# --------------------------------------------------------------------------------------
# The required key inside `payload.target` for each task (used by validation to catch
# empty/malformed/truncated targets fast, before full JSON-schema validation).
# --------------------------------------------------------------------------------------
TARGET_PRIMARY_KEY = {
    "PREDICT_NEXT_CHOICE": "choice",
    "PREDICT_NEXT_ACTION": "action_type",
    "PREDICT_NEXT_MESSAGE": "message_text",
    "PREDICT_NEXT_SPEAKER": "speaker_id",
    "PREDICT_BELIEF_CHANGE": "belief_after",
    "PREDICT_PRIVATE_STATE_UPDATE": "private_state_after",
    "PREDICT_TIME_TO_ACTION": "acted",
    "PREDICT_RESPONSE_OR_NONRESPONSE": "responded",
    "PREDICT_TRAJECTORY_CONTINUATION": "continuation",
    "PREDICT_FINAL_OUTCOME": "outcome",
    "PREDICT_POPULATION_RESPONSE": "aggregate_metrics",
    "PREDICT_POPULATION_TIME_SERIES": "time_series",
    "PREDICT_DISCUSSION_TREE": "tree",
    "PREDICT_INTERVENTION_EFFECT": "treated_outcome",
    "RANK_CANDIDATE_ACTIONS": "chosen_id",
    "PREDICT_POLICY_VALUE": "reward",
}
