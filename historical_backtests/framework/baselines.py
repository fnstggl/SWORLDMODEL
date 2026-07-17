"""Fair baseline arms — SAME pinned historical model, SAME frozen evidence capsule text.

Evidence parity is the whole point: the scientific comparison is `complete WMv2 simulation vs the
identical model forecasting directly from identical evidence`. Costs/calls recorded per arm by
the backend's audit ledger (separate audit files per arm).
"""
from __future__ import annotations

import re


def _extract_prob(text: str):
    m = re.findall(r"\"?probability\"?\s*[:=]\s*(0?\.\d+|1\.0|0|1)", str(text))
    if m:
        try:
            return max(0.001, min(0.999, float(m[-1])))
        except ValueError:
            pass
    m = re.findall(r"\b(0?\.\d+)\b", str(text))
    if m:
        try:
            return max(0.001, min(0.999, float(m[-1])))
        except ValueError:
            pass
    return None


def _q(question, cutoff, ev):
    return (f"Today is {cutoff[:10]}. Using ONLY the archived evidence below (all published "
            f"before today), forecast this question.\nQUESTION: {question}\n\nEVIDENCE:\n{ev}\n")


def direct(llm, question, cutoff, ev) -> dict:
    t = llm(_q(question, cutoff, ev) +
            'Respond ONLY JSON: {"probability": <0..1 that the answer is YES>, '
            '"rationale": "<one sentence>"}')
    return {"arm": "direct_same_model", "p": _extract_prob(t), "n_calls": 1}


def ensemble(llm, question, cutoff, ev, k: int = 5) -> dict:
    ps = []
    for i in range(k):
        t = llm(_q(question, cutoff, ev) +
                f"(independent draw {i + 1}/{k}) Respond ONLY JSON: "
                '{"probability": <0..1>}')
        p = _extract_prob(t)
        if p is not None:
            ps.append(p)
    return {"arm": "call_matched_ensemble", "p": (sum(ps) / len(ps) if ps else None),
            "n_calls": k, "spread": (max(ps) - min(ps)) if len(ps) > 1 else None}


_PERSONAS = ("a superforecaster who reasons from base rates and reference classes",
             "a domain insider who reasons from institutions and named actors",
             "a skeptical auditor who stress-tests the strongest claim in the evidence")


def observer_panel(llm, question, cutoff, ev) -> dict:
    ps = []
    for persona in _PERSONAS:
        t = llm(f"You are {persona}.\n" + _q(question, cutoff, ev) +
                'Respond ONLY JSON: {"probability": <0..1>}')
        p = _extract_prob(t)
        if p is not None:
            ps.append(p)
    return {"arm": "observer_panel", "p": (sum(ps) / len(ps) if ps else None),
            "n_calls": len(_PERSONAS)}


def analogical(llm, question, cutoff, ev) -> dict:
    t = llm(_q(question, cutoff, ev) +
            "First list 3 RESOLVED historical analogs (before today) with their outcomes, then "
            "derive the base rate they imply, then adjust for this case's evidence.\n"
            'Respond ONLY JSON: {"analogs": ["..."], "base_rate": <0..1>, '
            '"probability": <0..1>}')
    return {"arm": "analogical", "p": _extract_prob(t), "n_calls": 1}


def run_all(llm, question, cutoff, ev_text, market_price) -> list:
    out = []
    for fn in (direct, ensemble, observer_panel, analogical):
        try:
            out.append(fn(llm, question, cutoff, ev_text))
        except Exception as e:  # noqa: BLE001 — a failed arm is recorded, never invented
            out.append({"arm": fn.__name__, "p": None, "error": f"{type(e).__name__}: {e}"[:160]})
    out.append({"arm": "constant_half", "p": 0.5, "n_calls": 0})
    out.append({"arm": "market_price_at_cutoff", "p": market_price, "n_calls": 0})
    return out
