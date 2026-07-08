"""VariableInferenceEngine — maps EVERY behavioral variable for (entity, action, context).

This is the engine the thesis demands: for any person + message + setting, make the best possible
estimate of every variable acting on their response — the KNOWN ones from data/user context, the
INFERRED ones from context clues (LLM), always with provenance + confidence, always as-of.

Sources, in trust order:
  1. USER CONTEXT  — provided known variables (highest trust).
  2. DATA          — as-of history statistics (base responsiveness, relationship, recency,
                     reciprocity). High confidence, scaled by how much evidence exists.
  3. LLM INFERENCE — dispositional / relational / incentive / state variables inferred from the
                     message + as-of history summary + platform. Provenance "llm", per-variable
                     confidence from the inferer. (A precomputed inference dict — e.g. from an agent
                     swarm — can be passed, so this runs without an API key.)
  4. PLATFORM      — rule-based norms of the channel (email vs GitHub vs HN vs SMS).
  5. MESSAGE-FIT   — cheap lexical/structural heuristics on the action text (always available).
  6. PRIORS        — population defaults for anything still unset.

Leakage guarantee: the engine only ever sees history strictly before the action and the action's own
content — never the outcome. The LLM infers variables, never the response.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from swm.variables.variable_map import VariableMap

_PUSHY = re.compile(r"\b(urgent|asap|immediately|act now|last chance|final notice|don'?t miss|"
                    r"limited time|just following up|circling back|per my last|quick question)\b", re.I)
_PERSONAL = re.compile(r"\b(you|your|you're|thanks for your|i saw your|i read your|congrat)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hey|hello|dear)\b", re.I)

# platform response norms + formality + visibility (rule-based, "known")
_PLATFORM = {
    "email":  {"platform_response_norm": 0.30, "formality": 0.7, "visibility": 0.1},
    "github": {"platform_response_norm": 0.15, "formality": 0.4, "visibility": 0.8},
    "hn":     {"platform_response_norm": 0.10, "formality": 0.3, "visibility": 0.9},
    "sms":    {"platform_response_norm": 0.55, "formality": 0.2, "visibility": 0.05},
    "slack":  {"platform_response_norm": 0.45, "formality": 0.3, "visibility": 0.4},
    "cmv":    {"platform_response_norm": 0.63, "formality": 0.4, "visibility": 0.9},  # r/changemyview delta base
    "reddit": {"platform_response_norm": 0.20, "formality": 0.3, "visibility": 0.8},
    "generic": {"platform_response_norm": 0.30, "formality": 0.5, "visibility": 0.4},
}


@dataclass
class VariableInferenceEngine:
    platform: str = "generic"
    llm_infer_fn: object = None       # callable(entity_id, action, context, history) -> inference dict

    def infer(self, entity_id: str, action, context=None, *, history=None, user_context=None,
              llm_inference: dict | None = None, web_inference: dict | None = None) -> VariableMap:
        vm = VariableMap(entity_id=entity_id)
        self._from_history(vm, history)
        self._from_platform(vm, (getattr(action, "channel", None) or self.platform))
        self._from_message(vm, action)
        inf = llm_inference
        if inf is None and self.llm_infer_fn is not None:
            try:
                inf = self.llm_infer_fn(entity_id, action, context, history)
            except Exception:
                inf = None
        if inf:
            self._apply_llm(vm, inf)
        # web-sourced evidence about a PUBLIC FIGURE (what they've publicly done / responded to /
        # said they value). Applied AFTER the llm prior so grounded external signal overrides it, but
        # BEFORE user context so a fact you tell us still wins. Bias-to-infer: when we're not told a
        # variable, this is how we fill it for someone we've never messaged.
        if web_inference:
            self._apply_web(vm, web_inference)
        vm.merge_user_context(user_context)
        return vm.fill_priors()

    # ---- data (as-of history) -> known variables ----
    def _from_history(self, vm, history):
        if not history:
            return
        h = history if isinstance(history, dict) else _summarize(history)
        n = h.get("n_prior", 0)
        if n <= 0:
            return
        conf = 1.0 - math.exp(-n / 5.0)               # more history -> higher confidence
        if "response_rate" in h and h["response_rate"] is not None:
            vm.set("base_responsiveness", h["response_rate"], provenance="data", confidence=conf,
                   evidence=f"{n} prior interactions, rate {h['response_rate']:.2f}")
        vm.set("relationship_strength", min(1.0, math.log1p(n) / 3.0), provenance="data",
               confidence=conf, evidence=f"{n} prior interactions")
        if h.get("recency_days") is not None:
            rec = math.exp(-h["recency_days"] / 30.0)   # 1=just now, ->0 as it ages
            vm.set("recency_of_contact", rec, provenance="data", confidence=conf,
                   evidence=f"last contact {h['recency_days']:.0f}d ago")
        if h.get("reciprocity") is not None:
            vm.set("reciprocity_debt", h["reciprocity"], provenance="data", confidence=conf * 0.8,
                   evidence="derived from who-owes-whom in the thread")
        if h.get("status") is not None:
            vm.set("status", h["status"], provenance="data", confidence=conf, evidence="role in setting")

    # ---- platform -> known norms ----
    def _from_platform(self, vm, platform):
        p = _PLATFORM.get(platform, _PLATFORM["generic"])
        for k, v in p.items():
            vm.set(k, v, provenance="data", confidence=0.7, evidence=f"{platform} norm")

    # ---- message-fit heuristics (always available) ----
    def _from_message(self, vm, action):
        text = ""
        if hasattr(action, "meta"):
            text = action.meta.get("title", "") or action.meta.get("text", "")
        text = text or getattr(action, "text", "") or ""
        cf = getattr(action, "content_features", {}) or {}
        n_words = max(1, len(text.split()))
        vm.set("pushiness", min(1.0, len(_PUSHY.findall(text)) / 2.0), provenance="heuristic",
               confidence=0.5, evidence="lexical pushiness markers")
        vm.set("personalization", min(1.0, len(_PERSONAL.findall(text)) / 4.0), provenance="heuristic",
               confidence=0.4, evidence="second-person / personalization markers")
        vm.set("ask_directness", 1.0 if "?" in text else float(cf.get("ask_directness", 0.4)),
               provenance="heuristic", confidence=0.4, evidence="explicit question / ask")
        # length fit: moderate length is best; very long or empty is worse
        lf = math.exp(-((math.log1p(n_words) - math.log(40)) ** 2) / 2.0)
        vm.set("length_fit", lf, provenance="heuristic", confidence=0.35, evidence=f"{n_words} words")
        if cf:
            for k in ("clarity", "effort_cost", "personalization"):
                if k in cf:
                    vm.set(k, cf[k], provenance="heuristic", confidence=0.45, evidence="content feature")

    # ---- LLM-inferred variables ----
    def _apply_llm(self, vm, inference: dict):
        """inference: {var_name: value} or {var_name: {"value":.., "confidence":.., "evidence":..}}."""
        for name, payload in inference.items():
            if isinstance(payload, dict):
                vm.set(name, payload.get("value", 0.5), provenance="llm",
                       confidence=payload.get("confidence", 0.5), evidence=payload.get("evidence", "llm"))
            else:
                vm.set(name, payload, provenance="llm", confidence=0.5, evidence="llm")

    # ---- web-sourced variables (public-figure evidence) ----
    def _apply_web(self, vm, inference: dict):
        """Same payload shape as _apply_llm, but provenance='web' (cited external evidence about the
        entity's observed public behavior). Default confidence is a touch higher than a bare llm prior
        because it is grounded in retrieved signal, not pure guesswork."""
        for name, payload in inference.items():
            if isinstance(payload, dict):
                vm.set(name, payload.get("value", 0.5), provenance="web",
                       confidence=payload.get("confidence", 0.55),
                       evidence=payload.get("evidence", "web evidence"))
            else:
                vm.set(name, payload, provenance="web", confidence=0.55, evidence="web evidence")


def _summarize(events: list) -> dict:
    """(ts, outcome) pairs -> as-of history stats."""
    if not events:
        return {"n_prior": 0}
    outs = [o for _, o in events]
    last = max(ts for ts, _ in events)
    return {"n_prior": len(outs), "response_rate": sum(outs) / len(outs), "last_ts": last}
