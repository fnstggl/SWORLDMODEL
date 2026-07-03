"""FastAPI service (audit J).

Endpoints: /predict /compare-actions /simulate /infer-hidden-state /backtest /explain.
/simulate and /infer-hidden-state are tagged "insight"; the rest "prediction". Every response
echoes as_of so leakage is impossible to hide.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
