"""Active leakage probes — per-row contamination detection (never a single global safety claim).

Each probe RECORDS model behavior in the forecaster process; the SCORER (separate process, sealed store)
turns the records into the per-row leakage classification:

  known_contaminated          — the name-only probe stated the correct resolution with confidence
  contamination_susceptible   — the recognition probe identified the blinded event
  uncertain_leakage           — probes inconclusive (parse failures)
  low_leakage_risk            — blinded, unrecognized, name-only inconclusive
(clean_pre_cutoff_model is unavailable: no pre-cutoff checkpoint exists for this backend — recorded.)
"""
from __future__ import annotations

from swm.replay.blinding import apply_mapping

_NAME_ONLY = """Question: {q}
Do you know, as a matter of historical fact, how this resolved? Answer ONLY JSON:
{{"known": true|false, "resolution": "<what happened, one sentence, or null>", "confidence": <0..1>}}"""

_NO_EVIDENCE = """You are given ONLY this question, no evidence. Estimate the probability of YES.
Question: {q}
Return ONLY JSON: {{"p_yes": <0..1>, "basis": "<one sentence>"}}"""

_RECOGNITION = """The following is a pseudonymized description of a possibly-real event. If you can identify
the REAL event it describes, name it. Question: {q}
Return ONLY JSON: {{"identified": true|false, "real_event": "<name or null>", "confidence": <0..1>}}"""


def run_probes(llm, *, real_question: str, blinded_question: str, mapping: dict) -> dict:
    """Run all probes; returns raw records (no outcome is known or used here)."""
    from swm.engine.grounding import parse_json
    out = {}
    try:
        out["name_only"] = parse_json(llm(_NAME_ONLY.format(q=real_question))) or {"parse_failed": True}
    except Exception as e:  # noqa: BLE001
        out["name_only"] = {"error": str(e)[:100]}
    try:
        out["no_evidence_blinded"] = parse_json(llm(_NO_EVIDENCE.format(q=blinded_question))) or \
            {"parse_failed": True}
    except Exception as e:  # noqa: BLE001
        out["no_evidence_blinded"] = {"error": str(e)[:100]}
    try:
        rec = parse_json(llm(_RECOGNITION.format(q=blinded_question))) or {"parse_failed": True}
        # in-process identification check: does the named "real event" contain any REAL entity name from
        # the mapping? (mapping is forecaster-side; it contains no outcome)
        named = str(rec.get("real_event") or "")
        rec["names_real_entity"] = any(n.lower() in named.lower() for n in mapping if len(n) >= 4)
        out["recognition"] = rec
    except Exception as e:  # noqa: BLE001
        out["recognition"] = {"error": str(e)[:100]}
    return out


def classify_row(probes: dict, *, arm: str, name_only_correct: bool | None) -> str:
    """The scorer's per-row leakage classification. `name_only_correct` is computed by the SCORER against
    the sealed resolution; None = could not be evaluated."""
    if arm == "cutoff_prompted_unblinded":
        return "contamination_not_excluded"
    no = probes.get("name_only", {})
    if name_only_correct and float(no.get("confidence", 0) or 0) >= 0.6:
        return "known_contaminated"
    rec = probes.get("recognition", {})
    if rec.get("identified") and (rec.get("names_real_entity") or
                                  float(rec.get("confidence", 0) or 0) >= 0.7):
        return "contamination_susceptible"
    if any(v.get("parse_failed") or v.get("error") for v in probes.values() if isinstance(v, dict)):
        return "uncertain_leakage"
    return "low_leakage_risk"
