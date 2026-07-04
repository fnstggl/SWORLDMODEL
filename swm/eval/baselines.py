"""Baselines + the noise floor (audit E.3; exp001).

The baseline ladder itself lives in swm/eval/harness.py (L0–L4) so rungs share as-of feature
construction. This module holds the aleatoric noise floor: the best Brier score ANY calibrated
model could achieve, estimated from repeat structure — so "we're not better" is distinguishable
from "no one could be".
"""
from __future__ import annotations

from swm.ingestion.store import Send


def noise_floor_brier(sends: list[Send], min_sends_per_recipient: int = 10) -> dict:
    """For recipients with >= min_sends sends, the irreducible Brier bound is E[p_i(1-p_i)]
    where p_i is each recipient's true reply propensity (estimated by their empirical rate).
    This is what a perfect oracle knowing each person's propensity would still score."""
    per: dict[str, list[int]] = {}
    for s in sends:
        per.setdefault(s.recipient_id, []).append(1 if s.replied else 0)
    eligible = {k: v for k, v in per.items() if len(v) >= min_sends_per_recipient}
    if not eligible:
        return {"noise_floor_brier": None, "n_recipients_used": 0,
                "note": f"no recipient has >= {min_sends_per_recipient} sends"}
    total_n = sum(len(v) for v in eligible.values())
    floor = sum(
        len(v) * ((sum(v) / len(v)) * (1 - sum(v) / len(v))) for v in eligible.values()
    ) / total_n
    return {"noise_floor_brier": round(floor, 4), "n_recipients_used": len(eligible),
            "n_sends_used": total_n}


IMPLEMENTED = True
