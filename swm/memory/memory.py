"""Episodic (retrieval) + semantic (RFM/summaries) memory (audit C.4).

Episodic = the append-only log + vector retrieval. Semantic = cheap running sufficient statistics
(recency/frequency/monetary, topic affinities, response-rate priors) — often the strongest features.
Retrieval at eval time is restricted to pre-T events by the harness.

Stub — see docs/social-world-model-audit.md for the design. Not yet implemented."""

#: build-order and design are in docs/social-world-model-audit.md
IMPLEMENTED = False  # flip to True as this module lands; see the audit for its spec
