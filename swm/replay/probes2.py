"""Replay v2 per-row leakage audit (Part 17) — SIX probes per clean row, full prompts+outputs recorded.

Classification (scorer-side, sealed outcomes):
  clean_pre_cutoff_model / clean_blinded / low_leakage_risk / uncertain_leakage /
  contamination_susceptible / known_contaminated.
Only the first three enter headline results. A failed row stays in the contamination census.
"""
from __future__ import annotations

from swm.replay.blinding import apply_mapping

P_NAME_ONLY = """Question: {q}
Do you know, as a matter of historical fact, how this resolved? Answer ONLY JSON:
{{"known": true|false, "resolution": "<one sentence or null>", "confidence": <0..1>}}"""

P_NO_EVIDENCE = """You are given ONLY this question, no evidence. Estimate the probability of YES.
Question: {q}
Return ONLY JSON: {{"p_yes": <0..1>, "basis": "<one sentence>"}}"""

P_RECOGNITION = """The following is a pseudonymized description of a possibly-real event. If you can identify
the REAL event it describes, name it. Question: {q}
Return ONLY JSON: {{"identified": true|false, "real_event": "<name or null>", "confidence": <0..1>}}"""

P_IDENTITY_PERM = """Estimate the probability of YES for this question. Question: {q}
Return ONLY JSON: {{"p_yes": <0..1>}}"""

P_COUNTERFACTUAL = """Evidence (assume it is accurate and complete):
{ev}
Question: {q}
Return ONLY JSON: {{"p_yes": <0..1>, "driver": "<which evidence item drove your answer>"}}"""

P_TEMPORAL_FACT = """Today is {cutoff}. Answer from the perspective of that date ONLY.
{q_fact}
Return ONLY JSON: {{"answer": "<short>", "confidence": <0..1>}}"""


def _ask(llm, prompt):
    from swm.engine.grounding import parse_json
    try:
        return {"prompt": prompt[:500], "output": parse_json(llm(prompt)) or {"parse_failed": True}}
    except Exception as e:  # noqa: BLE001
        return {"prompt": prompt[:500], "output": {"error": str(e)[:100]}}


def run_probes_v2(llm, *, real_question: str, blinded_question: str, mapping: dict,
                  cutoff: str, evidence_text: str = "") -> dict:
    """All six probes. Records prompts+outputs; interpretation is the scorer's job."""
    out = {}
    out["name_only"] = _ask(llm, P_NAME_ONLY.format(q=real_question))
    out["no_evidence_blinded"] = _ask(llm, P_NO_EVIDENCE.format(q=blinded_question))
    rec = _ask(llm, P_RECOGNITION.format(q=blinded_question))
    named = str((rec["output"] or {}).get("real_event") or "")
    rec["output"]["names_real_entity"] = any(n.lower() in named.lower() for n in mapping if len(n) >= 4)
    out["recognition"] = rec
    # identity permutation: swap two pseudonyms — a prediction that moves a lot on a pure relabel is
    # keying on surface identity, not causal structure
    pseudos = sorted(set(mapping.values()))
    if len(pseudos) >= 2:
        permuted = blinded_question.replace(pseudos[0], "§TMP§").replace(pseudos[1], pseudos[0]) \
                                   .replace("§TMP§", pseudos[1])
        out["identity_permutation"] = {"base": _ask(llm, P_IDENTITY_PERM.format(q=blinded_question)),
                                       "permuted": _ask(llm, P_IDENTITY_PERM.format(q=permuted))}
    else:
        out["identity_permutation"] = {"skipped": "fewer than 2 pseudonyms"}
    # counterfactual evidence: flip the directional language of the evidence — the prediction must respond
    if evidence_text:
        flipped = _flip(evidence_text)
        out["counterfactual_evidence"] = {
            "base": _ask(llm, P_COUNTERFACTUAL.format(ev=evidence_text[:1800], q=blinded_question)),
            "flipped": _ask(llm, P_COUNTERFACTUAL.format(ev=flipped[:1800], q=blinded_question))}
    else:
        out["counterfactual_evidence"] = {"skipped": "no evidence text"}
    # temporal fact: an answer that differs before/after the cutoff
    out["temporal_fact"] = _ask(llm, P_TEMPORAL_FACT.format(
        cutoff=cutoff, q_fact=f"As of {cutoff}, has the following already been RESOLVED (occurred or "
                              f"definitively failed)? {blinded_question}"))
    return out


_FLIPS = (("leads", "trails"), ("ahead", "behind"), ("rising", "falling"), ("gains", "losses"),
          ("supports", "opposes"), ("approved", "rejected"), ("likely", "unlikely"),
          ("increase", "decrease"), ("strong", "weak"), ("won", "lost"), ("up", "down"),
          ("surge", "slump"), ("majority", "minority"), ("agreed", "refused"))


def _flip(text: str) -> str:
    out = text
    for a, b in _FLIPS:
        out = out.replace(f" {a} ", " §X§ ").replace(f" {b} ", f" {a} ").replace(" §X§ ", f" {b} ")
    return out


def classify_row_v2(probes: dict, *, arm: str, name_only_correct: bool | None) -> str:
    """Scorer-side classification (Part 17)."""
    if arm == "cutoff_prompted_unblinded":
        return "contamination_not_excluded_diagnostic"
    no = (probes.get("name_only") or {}).get("output") or {}
    if name_only_correct and float(no.get("confidence", 0) or 0) >= 0.6:
        return "known_contaminated"
    rec = (probes.get("recognition") or {}).get("output") or {}
    if rec.get("identified") and (rec.get("names_real_entity") or
                                  float(rec.get("confidence", 0) or 0) >= 0.7):
        return "contamination_susceptible"
    # counterfactual: prediction that ignores flipped evidence is using memorized outcome
    cf = probes.get("counterfactual_evidence") or {}
    if "base" in cf:
        pb = ((cf["base"].get("output") or {}).get("p_yes"))
        pf = ((cf["flipped"].get("output") or {}).get("p_yes"))
        if isinstance(pb, (int, float)) and isinstance(pf, (int, float)) and abs(pb - pf) < 0.02 \
                and (pb <= 0.2 or pb >= 0.8):
            return "contamination_susceptible"
    parse_issues = any(
        isinstance(v, dict) and isinstance(v.get("output"), dict) and
        (v["output"].get("parse_failed") or v["output"].get("error"))
        for v in probes.values())
    if parse_issues:
        return "uncertain_leakage"
    if arm == "pre_cutoff_checkpoint":
        return "clean_pre_cutoff_model"
    return "clean_blinded"
