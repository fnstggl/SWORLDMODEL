"""Shared behavior-model adapter — DISABLED BY DEFAULT, QUARANTINED. See docs/AUDIT_BEHAVIOR_MODELS.md.

One interface over interchangeable stakeholder-agent backends, so the SAME dossier/scenario/stimulus can be
run through DeepSeek (the current production reasoner), or through a behavior-TRAINED model (OSim / Minitaur /
Be.FM / OmniSapiens) once one is verified, licensed, and available on a GPU. The point is a fair, same-inputs
A/B: does a behavior-trained model predict real human choices better than a general LLM stakeholder, on
untouched held-out outcomes? (docs/AUDIT_BEHAVIOR_MODELS.md Part D.)

Design mirrors the TRIBE adapter (`tribe_adapter.py`): the ONLY runnable backend here is DeepSeek (an injected
chat fn — testable offline). Every behavior-trained backend is a STUB that refuses unless given a real runner
(a GPU box with the weights and an accepted license) — this environment has no GPU, so they must not silently
fabricate. Nothing in `swm/engine/*` imports this file (pinned by test_experimental_is_quarantined).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BehaviorRequest:
    """The full, identical input every backend receives — so no two are compared on different evidence."""
    dossier: str                          # exact person OR segment identity/incentives/beliefs (grounded)
    scenario: str                         # the situation
    stimulus: str = ""                    # the exact message/content the person reacts to (if any)
    relationship: str = ""                # relationship state / prior rapport
    goals: str = ""                       # goals & incentives
    history: list = field(default_factory=list)     # prior interaction history (turns/events)
    allowed_actions: list = field(default_factory=list)   # the forced-choice option set (empty ⇒ free/binary)
    world_state: str = ""                 # current public/world state
    elapsed: str = ""                     # elapsed time since last interaction / as-of horizon


@dataclass
class BehaviorResponse:
    action: str = None                    # chosen action (one of allowed_actions, or "respond"/"no_response")
    rationale: str = ""                   # textual reason where the backend provides one
    p: float = None                       # optional probability / calibrated confidence of the action
    logprob: float = None                 # optional log-prob (forced-choice models)
    abstain: bool = False
    abstain_reason: str = ""
    backend: str = ""                     # model metadata
    latency_s: float = None
    tokens: int = None
    cost_usd: float = None


class BackendUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------- backends
class BehaviorBackend:
    name = "abstract"

    def decide(self, req: BehaviorRequest) -> BehaviorResponse:      # pragma: no cover - interface
        raise NotImplementedError


_DECIDE_PROMPT = """You ARE the following person/segment, reasoning as them (not as an analyst).
WHO YOU ARE: {dossier}
YOUR GOALS/INCENTIVES: {goals}
RELATIONSHIP CONTEXT: {relationship}
WHAT HAS HAPPENED BEFORE: {history}
CURRENT WORLD STATE: {world_state}
TIME CONTEXT: {elapsed}

THE SITUATION: {scenario}
{stimulus_block}
Choose ONE action, as this person, right now. {actions_block}
Return ONLY JSON: {{"action": "<one option>", "p": <0..1 your probability of taking it>,
"why": "<one sentence, this person's real reason>"}}"""


@dataclass
class DeepSeekBehaviorBackend(BehaviorBackend):
    """The current production reasoner as a behavior backend. Runnable (inject a chat fn; testable offline)."""
    name: str = "deepseek"
    llm: object = None                    # callable(prompt)->text; if None, built from default_chat_fn
    temperature: float = 0.7

    def _llm(self):
        if self.llm is not None:
            return self.llm
        from swm.api.deepseek_backend import default_chat_fn
        self.llm = default_chat_fn(system="You inhabit one specific person. Reply ONLY compact JSON.",
                                   max_tokens=250, temperature=self.temperature)
        if self.llm is None:
            raise BackendUnavailable("no DeepSeek/HF backend configured")
        return self.llm

    def decide(self, req: BehaviorRequest) -> BehaviorResponse:
        from swm.engine.grounding import parse_json
        actions = req.allowed_actions or ["respond", "no_response"]
        prompt = _DECIDE_PROMPT.format(
            dossier=req.dossier, goals=req.goals or "(unspecified)", relationship=req.relationship or "(none)",
            history=" | ".join(str(h) for h in req.history[:8]) or "(none)",
            world_state=req.world_state or "(as grounded)", elapsed=req.elapsed or "(now)",
            scenario=req.scenario,
            stimulus_block=(f"THE EXACT MESSAGE/CONTENT:\n---\n{req.stimulus[:2000]}\n---\n" if req.stimulus else ""),
            actions_block=f"OPTIONS: {actions}")
        t0 = time.time()
        raw = self._llm()(prompt)
        r = parse_json(raw) or {}
        act = r.get("action")
        if act not in actions:
            return BehaviorResponse(abstain=True, abstain_reason=f"unparseable/out-of-set action: {act!r}",
                                    backend=self.name, latency_s=round(time.time() - t0, 2))
        p = None
        try:
            p = min(1.0, max(0.0, float(r["p"])))
        except (KeyError, TypeError, ValueError):
            pass
        return BehaviorResponse(action=act, rationale=str(r.get("why", ""))[:200], p=p, backend=self.name,
                                latency_s=round(time.time() - t0, 2),
                                tokens=(len(prompt) + len(raw or "")) // 4)


@dataclass
class _StubBehaviorBackend(BehaviorBackend):
    """A behavior-TRAINED backend we have not verified/licensed/hosted. Refuses unless a real runner is
    injected AND commercial terms are cleared. Never fabricates a decision."""
    name: str = "stub"
    runner: object = None                 # the real GPU-hosted model callable; None here on purpose
    commercial_ok: bool = False

    def decide(self, req: BehaviorRequest) -> BehaviorResponse:
        if self.runner is None:
            raise BackendUnavailable(
                f"{self.name} backend has no runner. This environment has no GPU and the weights/license are "
                f"unverified — see docs/AUDIT_BEHAVIOR_MODELS.md. Inject a real runner on a GPU box to pilot.")
        out = self.runner(req)            # the runner returns a BehaviorResponse-compatible dict
        return BehaviorResponse(action=out.get("action"), rationale=out.get("rationale", ""),
                                p=out.get("p"), logprob=out.get("logprob"), backend=self.name,
                                latency_s=out.get("latency_s"), tokens=out.get("tokens"))


def osim_backend(runner=None):        return _StubBehaviorBackend(name="osim", runner=runner)
def minitaur_backend(runner=None):    return _StubBehaviorBackend(name="minitaur", runner=runner)
def befm_backend(runner=None):        return _StubBehaviorBackend(name="be.fm", runner=runner)
def omnisapiens_backend(runner=None): return _StubBehaviorBackend(name="omnisapiens", runner=runner)


# --------------------------------------------------------------------- adapter
@dataclass
class BehaviorModelAdapter:
    """Disabled by default. Holds one or more named backends and runs a request through the selected one,
    metering latency/tokens/cost and turning any BackendUnavailable into an honest abstention."""
    enabled: bool = False
    backends: dict = field(default_factory=dict)     # {name: BehaviorBackend}

    def available(self) -> bool:
        return bool(self.enabled and self.backends)

    def decide(self, backend_name: str, req: BehaviorRequest) -> BehaviorResponse:
        if not self.enabled:
            raise BackendUnavailable("BehaviorModelAdapter is disabled (research-only). Set enabled=True.")
        b = self.backends.get(backend_name)
        if b is None:
            raise BackendUnavailable(f"no backend {backend_name!r}; have {sorted(self.backends)}")
        try:
            return b.decide(req)
        except BackendUnavailable as e:
            return BehaviorResponse(abstain=True, abstain_reason=str(e), backend=backend_name)
