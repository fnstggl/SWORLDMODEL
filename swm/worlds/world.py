"""Per-customer fitted "world" object, versioned + immutable (audit C.11, J).

Bundles the fitted readout + entity embeddings + calibration for one customer's population. Once
published a version is immutable; predictions echo world_id + model_version + as_of.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
