"""Action/intervention encoder (audit C.7).

Encodes a candidate action so the transition model can condition on it — INCLUDING novel actions
with no historical analog. For email: subject+body embedding + structured features (length, CTA,
personalization tokens, tone), audience selector, channel, timing. Enables /compare-actions.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
