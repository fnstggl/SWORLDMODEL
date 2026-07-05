"""Public prediction API for the social world model.

One entry point for any individual-regime prediction:

    sim = Simulator(platform="email").fit(stream)
    result = sim.simulate(entity_id, action, context, user_context=..., llm_inference=...)

`result` is a `Prediction` carrying a calibrated probability, an honest confidence, the regime the
query fell into, an abstain flag when the query is outside the validated envelope, the model's
calibration badge, and an auditable variable/provenance explanation.
"""
from swm.api.simulate import Prediction, Simulator

__all__ = ["Simulator", "Prediction"]
