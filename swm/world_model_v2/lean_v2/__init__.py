"""Lean V2 — the first-principles consumer execution path for World Model V2.

Reached ONLY through `unified_runtime.simulate_world(..., execution_profile="lean_v2")`.
`full_fidelity` (research grade) and `lean_adaptive` (Lean V1, the current default) are untouched.

The irreducible consumer simulation:

    1. parse the question and decisive evidence;
    2. compile ONE coherent causal world (the ConsumerWorldBlueprint — one structured call,
       deterministic validators, at most one targeted repair call);
    3. PROVE the world can produce a terminal answer (three-valued answerability preflight,
       before any actor spends a call);
    4. keep only actors/mechanisms that can change that answer (terminal-causal backward slice;
       retain when uncertain; dynamic promotion preserved);
    5. represent the genuinely different plausible private states (grounded weight RANGES,
       never LLM-invented precise probabilities);
    6. call the LLM once per genuinely DISTINCT human decision context (decision-equivalence
       cache + single-flight; one bounded extra deliberation call when materially justified);
    7. execute mechanical rules and consequences from precompiled parameterized templates
       (novel actions still compile; interpretation/persuasion stay actor-simulated);
    8. merge equivalent weighted world states EXACTLY, preserving total probability mass;
    9. add computation only if the forecast remains materially unstable (pilot-gated,
       localized challenger; no pointless replicates of unresolved runs);
    10. return probability + grounding + uncertainty + limitations under the
        forecast-recovery contract.

Every external call flows through ONE gateway (budget, tier routing, one bounded retry,
stage ledger). Every stage checkpoints. Nothing relaunches itself."""

LEAN_V2_VERSION = "lean-v2-1.0"
PROMPT_VERSION = "lean_v2.prompts.v1"
SCHEMA_VERSION = "lean_v2.schema.v1"
COMPILER_VERSION = "lean_v2.compiler.v1"
