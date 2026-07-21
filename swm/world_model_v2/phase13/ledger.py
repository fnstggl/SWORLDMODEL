"""Phase 13 prospective decision ledger (Part 30C) — freeze before, resolve after, fabricate never.

Append-only JSONL. `freeze(problem, result)` locks the decision context, admissible actions,
recommendation, predicted utility/effect and uncertainty BEFORE any real outcome exists, with an
artifact hash. `record_choice` and `record_outcome` append follow-up rows referencing the frozen row —
a frozen row is NEVER edited. The ledger may begin (and stay) empty of outcomes; outcomes only enter
when reality supplies them."""
from __future__ import annotations

import hashlib
import json
import os
import time as _time


class DecisionLedger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _append(self, row: dict) -> dict:
        row["ledger_ts"] = _time.time()
        payload = json.dumps(row, sort_keys=True, default=str)
        row["row_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def freeze(self, problem, result) -> dict:
        top = None
        for e in result.evaluated:
            if e["action_id"] == result.recommended:
                top = e
                break
        row = {
            "kind": "frozen_decision",
            "decision_id": problem.decision_id,
            "contract_hash": problem.contract_hash(),
            "as_of": problem.as_of, "horizon": problem.horizon,
            "context": problem.context[:400],
            "admissible_actions": [e["action_id"] for e in result.evaluated],
            "recommendation": result.recommended,
            "recommendation_kind": result.recommendation_kind,
            "predicted_utility": (top or {}).get("expected_utility"),
            "predicted_effect_vs_reference": ((top or {}).get("paired_vs_reference") or {}).get("paired_mean"),
            "uncertainty": {"q10": (top or {}).get("q10"), "q90": (top or {}).get("q90"),
                            "cvar": (top or {}).get("cvar")},
            "causal_claim": result.causal_claim,
            "runtime_fingerprint": result.runtime_fingerprint,
            "seed": result.seed,
            "chosen_real_action": None, "realized_outcome": None,
        }
        return self._append(row)

    def record_choice(self, decision_id: str, frozen_hash: str, chosen_action: str) -> dict:
        return self._append({"kind": "real_choice", "decision_id": decision_id,
                             "frozen_row_hash": frozen_hash, "chosen_real_action": chosen_action})

    def record_outcome(self, decision_id: str, frozen_hash: str, realized_outcome) -> dict:
        return self._append({"kind": "realized_outcome", "decision_id": decision_id,
                             "frozen_row_hash": frozen_hash, "realized_outcome": realized_outcome})

    def rows(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
