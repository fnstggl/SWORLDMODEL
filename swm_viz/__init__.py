"""swm_viz — a read-only visualizer for the real World Model V2 (Lean V2) simulation.

Nothing in this package modifies ``swm/``. It transparently observes a real
``execution_profile="lean_v2"`` run (every LLM call + the compiled world + all state/decision
provenance), folds it into a replayable ``recording.json``, and serves a localhost frontend
that plays the social simulation back step by step.
"""
