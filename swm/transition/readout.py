"""Discriminative readout p(y | context, action) — THE WORKHORSE (audit C.8).

A one-step social world model: state-summary + encoded action -> outcome DISTRIBUTION. Start with
gradient-boosted trees over engineered + embedded features. This is the first thing to build and
the baseline everything fancier must beat on a proper scoring rule.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
