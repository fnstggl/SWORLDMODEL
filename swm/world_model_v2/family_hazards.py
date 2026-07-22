"""Learned event-family hazards — fitted base rates replacing ignorance-shaped Beta draws (fidelity #3).

The resolved-market archive is a training set. Families are FROZEN keyword rules over question text
(never benchmark IDs); rates are fit ONLY on calibration-split worlds (leakage-safe: validation/locked
outcomes never enter), with partial pooling toward the global rate (small-n families shrink hard).
The pack is a versioned artifact; the aggregate-outcome and structural-prior mechanisms use the family
rate as their BASE rate (labeled `fitted_family_prior`) instead of a broad Beta draw. When no pack or no
family matches, the broad prior remains (labeled, honest).
"""
from __future__ import annotations

import json
from pathlib import Path

PACK = Path("experiments/replay_vault_v3/family_hazard_pack.json")

#: FROZEN family rules (ordered; first match wins) — question-text semantics, not IDs.
FAMILY_RULES = (
    ("personnel_out_by_date", ("out as", "resign", "fired", "step down", "removed", "ousted", "out by")),
    ("meeting_or_deal_by_date", ("meeting by", "meet by", "deal by", "agreement by", "ceasefire",
                                 "peace deal", "sign", "summit", "talks by", "agrees to")),
    ("approval_or_passage", ("pass", "approve", "confirm", "ratif", "enact", "bill", "nominee")),
    ("sports_match", ("win on 20", "fc win", "vs.", "beat", "match", "game on")),
    ("price_threshold", ("above", "below", "reach $", "hit $", "price", "all-time high", "market cap")),
    ("announcement_or_visit", ("announce", "visit", "launch", "release", "unveil", "declare")),
)


def classify_family(question: str) -> str:
    q = " " + str(question).lower() + " "
    for fam, toks in FAMILY_RULES:
        if any(t in q for t in toks):
            return fam
    return "generic"


def fit_pack(worlds_with_outcomes: list, *, pool_strength: float = 8.0) -> dict:
    """worlds_with_outcomes: [{question, outcome}] — CALIBRATION split only (caller enforces).
    Partial pooling: rate = (yes + k*global) / (n + k)."""
    global_yes = sum(w["outcome"] for w in worlds_with_outcomes)
    global_n = max(1, len(worlds_with_outcomes))
    g = global_yes / global_n
    fams = {}
    for w in worlds_with_outcomes:
        fams.setdefault(classify_family(w["question"]), []).append(int(w["outcome"]))
    rates = {}
    for fam, ys in fams.items():
        rates[fam] = {"n": len(ys), "raw_rate": round(sum(ys) / len(ys), 4),
                      "pooled_rate": round((sum(ys) + pool_strength * g) / (len(ys) + pool_strength), 4)}
    return {"version": "family-hazards-1.0", "fit_on": "calibration split only",
            "global_rate": round(g, 4), "n_worlds": global_n, "pool_strength": pool_strength,
            "families": rates}


def family_base_rate(question: str) -> tuple:
    """(pooled_rate|None, family, provenance) from the fitted pack, if present and the family was fit."""
    if not PACK.exists():
        return None, classify_family(question), "no_pack"
    pack = json.loads(PACK.read_text())
    fam = classify_family(question)
    ent = (pack.get("families") or {}).get(fam)
    if ent is None:
        return pack.get("global_rate"), fam, "global_pooled"
    return ent["pooled_rate"], fam, f"fitted_family_prior(n={ent['n']})"
