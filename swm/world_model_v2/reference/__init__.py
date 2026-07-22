"""Reference worlds — empirical validation environments, NOT engines.

Each module here is a CONFIGURATION: dataset → examples (leak-free), mechanisms FITTED from the train split
and registered with honest calibration status, a world built from universal primitives (entities, latents,
event queue, operators), and the baseline/ablation arm ladder. The architecture lives in swm/world_model_v2/;
nothing here may define its own state system or bypass the shared runtime.
"""
