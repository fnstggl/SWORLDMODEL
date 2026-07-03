"""Upworthy Research Archive harness — public credibility result (audit B.7, D).

Headline A/B -> predicted CTR distribution; score with CRPS/log-loss/ECE vs realized rates.
TREAT AS CONTAMINATED (public since 2021): exclude the flagged ~22% (2013-14 randomization bug)
and pair the headline claim with a fresh post-cutoff corpus. Data: osf.io/jd64p (CC BY 4.0).

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
