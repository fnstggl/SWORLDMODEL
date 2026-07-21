"""Suite-wide fixtures.

SWM_ALLOW_NUMERIC_BASELINE=1 is the §19 allowance: the numerical actor system may serve as an
explicit TEST COMPARISON. Offline suites (no LLM backend) run rollouts against the numeric
baseline arm under this marker. Production never sets it — the strict-integrity enforcement
tests (tests/test_core_arch_invariants.py) explicitly UNSET it and prove the default route
cannot reach a numeric actor policy, a template personality, or the generic outcome prior.
"""
import os

os.environ.setdefault("SWM_ALLOW_NUMERIC_BASELINE", "1")
# §28 allowance: the generic outcome prior remains runnable as an explicit baseline/diagnostic
# in offline suites; production leaves this unset and refuses the broad-prior terminal draw.
os.environ.setdefault("SWM_ALLOW_GENERIC_PRIOR", "1")
