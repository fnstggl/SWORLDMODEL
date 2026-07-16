"""Fit the family SURVIVAL pack (event-time architecture, component 4) — CALIBRATION SPLIT ONLY.

For each of the 40 calibration worlds in the frozen v3 vault, refetch the archived CLOB price path
(public, timestamped) and compute the lifetime fraction at which the YES price first crossed 0.9 —
the labeled effective-resolution proxy. `swm.world_model_v2.event_time.fit_survival_pack` turns these
first-passage fractions into discrete per-bucket hazards with partial pooling toward the global curve.

Governance: uses ONLY the calibration split; no validation/locked outcome, price, or path is touched.
The proxy is a price-path statistic, not the sealed resolution label.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.replay_v2.build_vault as V2B                      # verified archive fetch helpers
from swm.world_model_v2.event_time import SURV_PACK, fit_survival_pack

VAULT = Path("experiments/replay_vault_v3")
THRESHOLD = 0.9


def _yes_token(condition_id):
    ms = V2B._get(f"{V2B.GAMMA}/markets?condition_ids={condition_id}&closed=true") or []
    for m in ms:
        outs = json.loads(m.get("outcomes") or "[]")
        toks = json.loads(m.get("clobTokenIds") or "[]")
        low = [str(o).lower() for o in outs]
        if "yes" in low and len(toks) == len(outs):
            return toks[low.index("yes")]
    return None


def main():
    ev = json.loads((VAULT / "events.json").read_text())
    cal = [w for w in ev["worlds"] if ev["splits"].get(w["event_id"]) == "calibration"]
    print(f"calibration worlds: {len(cal)} (splits are frozen chronological — see build_vault3)")
    rows, skipped = [], 0
    for w in cal:
        cid = (w.get("source") or {}).get("condition_id")
        tok = _yes_token(cid) if cid else None
        hist = V2B._history(tok) if tok else []
        if len(hist) < 8:
            skipped += 1
            print(f"  SKIP {w['event_id']}: path unavailable/thin")
            continue
        t0, t1 = hist[0]["t"], hist[-1]["t"]
        cross = next((p["t"] for p in hist if p["p"] >= THRESHOLD), None)
        frac = (cross - t0) / max(1.0, (t1 - t0)) if cross is not None else None
        rows.append({"question": w["question"], "lifetime_fraction_resolved": frac})
        print(f"  {w['event_id']}: frac={None if frac is None else round(frac, 3)}")
        time.sleep(0.15)
    pack = fit_survival_pack(rows)
    pack["proxy"] = f"first CLOB YES price >= {THRESHOLD} crossing, as lifetime fraction; None = censored"
    pack["n_worlds_used"] = len(rows)
    pack["n_skipped"] = skipped
    SURV_PACK.write_text(json.dumps(pack, indent=1))
    fams = ", ".join("{}(n={})".format(k, v["n"]) for k, v in pack["families"].items())
    print("\nwrote {}: global={} families={}".format(SURV_PACK, pack["global_hazards"], fams))


if __name__ == "__main__":
    main()
