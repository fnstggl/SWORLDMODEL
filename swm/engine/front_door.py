"""AgentWorldModel — one call, ONE mechanism: grounded agents interacting.

The mechanism zoo is dead. There is no menu of bracket/committee/electorate/logistic for an LLM to pick
from (and pick wrong); there is one substrate — ground the scene, cast the real actors, let LLM-reasoned
personas interact over real dated rounds — and the only thing that varies per question is the CASTING
(who the agents are, what the answer space is, what the interaction structure is). Routing:

  collective_choice / population_share → SocietyRollout   (distribution over the NAMED options)
  individual_reaction                  → IndividualSimulator (this person × this exact message, N runs)
  artifact_optimization                → ArtifactOptimizer (real generated texts, ranked by the audience)

Sequence per call: ground (abstain loudly if the deciding facts can't be established — and short-circuit
with provenance if the evidence shows the world already answered) → cast → simulate → calibrate
(grade-or-abstain; a fitted shrink where the class earned one) → a native-typed Forecast.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from swm.engine.actions import ArtifactOptimizer
from swm.engine.calibrate import GradeRegistry, shrink_distribution
from swm.engine.casting import CastingDirector
from swm.engine.grounding import SceneGrounder
from swm.engine.individual import IndividualSimulator
from swm.engine.outcome import Forecast, build_headline
from swm.engine.society import SocietyRollout

_ARTIFACT_WORDS = ("headline", "subject line", "tagline", "copy", "slogan", "ad ", "landing page",
                   "best message", "best email")


@dataclass
class AgentWorldModel:
    llm: object                            # cold backend (grounding, casting, variants)
    llm_hot: object = None                 # hot backend (persona decisions; temperature > 0)
    branches: int = 3
    max_rounds: int = 2
    k_artifacts: int = 5
    registry: GradeRegistry = field(default_factory=GradeRegistry)
    search_fn: object = None               # override retrieval (fixtures in tests)
    today: str = ""

    def __post_init__(self):
        self.llm_hot = self.llm_hot or self.llm

    # ---------------- the one entry point ----------------
    def simulate(self, question: str, *, message: str = None, recipient: str = None,
                 channel: str = "email") -> dict:
        today = self.today or _time.strftime("%Y-%m-%d")

        # explicit individual ask (recipient + exact artifact) skips scene-casting: the scene IS the person
        if recipient and message:
            return self._individual(question, recipient, message, channel, today)

        dossier = SceneGrounder(self.llm, search_fn=self.search_fn, today=today).ground(question)
        if dossier.abstain:
            f = Forecast(question=question, mechanism="abstained", abstain=True,
                         abstain_reason=dossier.abstain_reason, grounding=dossier.as_report(),
                         calibration={"grade": "n/a", "note": "no forecast was emitted"})
            f.headline = build_headline(f)
            return f.as_dict()

        if dossier.resolved:                              # the world already answered — report it, cited
            f = Forecast(question=question, mechanism="resolved_by_evidence",
                         distribution={str(dossier.resolved["answer"]): 1.0},
                         grounding=dossier.as_report(),
                         calibration={"grade": "resolved", "abstain_confident": False,
                                      "note": "outcome already decided per the cited evidence"},
                         headline=(f"ALREADY RESOLVED: {dossier.resolved['answer']} "
                                   f"[{dossier.resolved.get('source', '?')}] — "
                                   f"\"{str(dossier.resolved.get('evidence', ''))[:140]}\""))
            return f.as_dict()

        cast = CastingDirector(self.llm).cast(question, dossier.brief(), today=today)
        if cast.process == "individual_reaction":
            person = recipient or (cast.actors[0].name if cast.actors else "")
            return self._individual(question, person, message or question, channel, today,
                                    dossier=dossier)
        if cast.process == "artifact_optimization" or (message is None and any(
                w in question.lower() for w in _ARTIFACT_WORDS)):
            return self._artifacts(question, cast, dossier, today)
        return self._society(question, cast, dossier, today)

    # ---------------- modes ----------------
    def _society(self, question, cast, dossier, today):
        res = SocietyRollout(self.llm_hot, llm=self.llm, branches=self.branches,
                             max_rounds=self.max_rounds).run(question, cast, dossier, today=today)
        cal = self.registry.calibration_for(f"society:{cast.process}")
        dist = shrink_distribution(res.distribution, cal.get("shrink", 1.0))
        f = Forecast(question=question, mechanism=f"grounded_agents:{cast.process}",
                     answer_space=cast.answer_space,
                     distribution={o: round(p, 4) for o, p in dist.items()},
                     interval=res.interval, calibration=cal, grounding=dossier.as_report(),
                     audit=res.audit, rounds=res.rounds, n_personas=res.n_personas,
                     n_llm_calls=res.n_calls, detail={"cast": cast.as_dict(),
                                                      "branch_distributions": res.branch_distributions})
        if not f.distribution:
            f.abstain, f.abstain_reason = True, "no persona decisions parsed — simulation produced nothing"
        f.headline = build_headline(f)
        return f.as_dict()

    def _individual(self, question, person, message, channel, today, dossier=None):
        if not person:
            f = Forecast(question=question, mechanism="abstained", abstain=True,
                         abstain_reason="individual_reaction with no identifiable person — "
                                        "pass recipient=<name>")
            f.headline = build_headline(f)
            return f.as_dict()
        res = IndividualSimulator(self.llm_hot, llm=self.llm).simulate(person, message, channel=channel)
        cal = self.registry.calibration_for("individual:response")
        f = Forecast(question=question, mechanism="grounded_agents:individual_reaction",
                     answer_space={"type": "binary", "options": ["responds", "does_not_respond"]},
                     calibration=cal, abstain=res.abstain, abstain_reason=res.abstain_reason,
                     grounding={**res.grounding, "detail": [], "abstain": res.abstain},
                     n_personas=len(res.per_state), n_llm_calls=res.n_runs,
                     detail={"per_state": res.per_state, "reasons": res.reasons,
                             "person": person})
        if not res.abstain:
            p = shrink_distribution({"responds": res.p_response,
                                     "does_not_respond": 1 - res.p_response},
                                    cal.get("shrink", 1.0))["responds"]
            f.distribution = {"responds": round(p, 4), "does_not_respond": round(1 - p, 4)}
            f.interval = {"responds": res.interval_80}
            f.headline = (f"{person} responds to THIS message: {p:.0%} "
                          f"(80% sampling interval {res.interval_80[0]:.0%}–{res.interval_80[1]:.0%}, "
                          f"{res.n_runs} grounded runs)")
            if cal.get("abstain_confident"):
                f.headline += f"  [{cal.get('grade')} — hypothesis, not a calibrated forecast]"
        else:
            f.headline = build_headline(f)
        return f.as_dict()

    def _artifacts(self, question, cast, dossier, today):
        res = ArtifactOptimizer(self.llm_hot, llm=self.llm).run(
            question, cast, dossier, k=self.k_artifacts, today=today)
        cal = self.registry.calibration_for("artifact:engagement")
        f = Forecast(question=question, mechanism="grounded_agents:artifact_optimization",
                     answer_space={"type": "artifacts",
                                   "options": [a["text"] for a in res.get("ranked", [])]},
                     ranked_artifacts=res.get("ranked", []), calibration=cal,
                     grounding=dossier.as_report(),
                     n_personas=res.get("n_personas", 0), n_llm_calls=res.get("n_calls", 0),
                     detail={"cast": cast.as_dict(), "note": res.get("note", "")})
        if not f.ranked_artifacts:
            f.abstain, f.abstain_reason = True, res.get("error", "no artifacts survived evaluation")
        f.headline = build_headline(f)
        return f.as_dict()


def agent_world_model(*, branches=3, max_rounds=2, k_artifacts=5, today="") -> AgentWorldModel:
    """The recommended front door. DeepSeek backends: cold (t=0.2) for grounding/casting, hot (t=0.9)
    for persona decisions — the N runs must actually differ. Raises if no LLM key is configured: this
    engine does not degrade to heuristics, it refuses (constitution rule 2)."""
    from swm.api.deepseek_backend import default_chat_fn
    cold = default_chat_fn(system="You are a precise assistant inside a forecasting engine. "
                                  "Reply with ONLY compact JSON.", max_tokens=1600, temperature=0.2)
    hot = default_chat_fn(system="You inhabit one specific person inside a grounded social simulation. "
                                 "Reason as them, not as an analyst. Reply with ONLY compact JSON.",
                          max_tokens=500, temperature=0.9)
    if cold is None:
        raise RuntimeError("no LLM backend configured (DEEPSEEK_API_KEY / HF_TOKEN) — the agent engine "
                           "refuses to run ungrounded heuristics in place of reasoning")
    return AgentWorldModel(llm=cold, llm_hot=hot, branches=branches, max_rounds=max_rounds,
                           k_artifacts=k_artifacts, today=today)
