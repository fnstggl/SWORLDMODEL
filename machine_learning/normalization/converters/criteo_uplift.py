"""Criteo Uplift Modeling Dataset (v2.1) — a randomized ad-exposure trial.

Real source (verified from a streamed HF sample of criteo/criteo-uplift):
  columns = f0..f11 (12 anonymized dense features, double),
            treatment (0/1, ad-treatment arm; ~85% treated),
            exposure  (0/1, whether the ad was actually shown given treatment),
            visit     (0/1, site visit outcome),
            conversion(0/1, conversion outcome).
Assignment is RANDOMIZED (a real RCT); there is NO per-row propensity column.

Emits:
  PREDICT_INTERVENTION_EFFECT — per unit, potential-outcomes framing. ONLY the realized
      arm is observed: its outcome is filled, the counterfactual arm's outcome is left
      null (observed=False), NEVER imputed. The per-unit effect is not identifiable.
  PREDICT_POPULATION_RESPONSE — a block of rows aggregated per treatment arm into one
      population example: aggregate_metrics = {rate, n, successes} (conversion rate).
  PREDICT_POLICY_VALUE — reward = conversion of a logged (features, treatment-action)
      pair under the randomized assignment (propensity unknown -> null).

Honesty: the counterfactual outcome is never fabricated. There is no logged per-row
propensity, so PREDICT_POLICY_VALUE.propensity is null (assignment is randomized, so the
marginal treatment rate is a population quantity, not a per-row logged probability).
License is CC-BY-NC-SA-4.0 (non-commercial) -> reserved as a CROSS_DATASET_EVAL_ONLY
held-out intervention-transfer test (training_suitability = eval_only).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter

#: Rows aggregated into one population-response example (per treatment arm).
BLOCK_SIZE = 500
_FEATURE_COLS = [f"f{i}" for i in range(12)]


def _load_rows(raw_dir: Path) -> list[dict]:
    """Load Criteo rows from streamed parquet shards or a JSON fixture (list of dicts)."""
    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)
                if ".cache" not in f]
    if not pq_files:
        pq_files = [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True)
                    if ".cache" not in f]
    if pq_files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for f in sorted(pq_files):
            rows.extend(pq.read_table(f).to_pylist())
        return rows
    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"no Criteo parquet/json found under {raw_dir}")


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class Converter(BaseConverter):
    DATASET_ID = "criteo_uplift"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "criteo_uplift"
    DOC = {
        "dataset_id": "criteo_uplift",
        "original_fields": [
            {"name": "f0..f11", "meaning": "12 anonymized dense features (double)", "example": 12.616},
            {"name": "treatment", "meaning": "randomized ad-treatment arm (1=treated, 0=control)", "example": 1},
            {"name": "exposure", "meaning": "whether the ad was actually shown (given treatment)", "example": 0},
            {"name": "visit", "meaning": "site-visit outcome (0/1)", "example": 0},
            {"name": "conversion", "meaning": "conversion outcome (0/1)", "example": 0},
        ],
        "canonical_mapping": [
            {"source_field": "f0..f11", "canonical_path": "payload.input.population_or_actor_features / logged_context"},
            {"source_field": "treatment", "canonical_path": "causal_metadata.treatment_arm / payload.input.assignment_mechanism.realized_arm"},
            {"source_field": "conversion", "canonical_path": "payload.target.reward / treated_or_control_outcome.conversion / aggregate_metrics.successes"},
            {"source_field": "visit", "canonical_path": "treated_or_control_outcome.visit / aggregate_metrics.visit_successes"},
            {"source_field": "exposure", "canonical_path": "treated_or_control_outcome.exposure"},
        ],
        "tasks_produced": ["PREDICT_INTERVENTION_EFFECT", "PREDICT_POPULATION_RESPONSE", "PREDICT_POLICY_VALUE"],
        "unavailable_fields": [
            "counterfactual outcome (only the realized arm is observed; never imputed)",
            "per-row logging propensity (assignment is randomized; no logged probability)",
            "timestamps / user identity (rows are independent anonymized impressions)",
        ],
        "chronology_rules": "Features f0..f11 are pre-treatment covariates and the assignment arm is known before the outcome; only the outcome (conversion/visit) is the label and is confined to payload.target. Population/policy aggregates put the outcome only in the target.",
        "split_key": "time_period (blocks of rows; group_id=criteo_uplift-tp-<block>)",
        "leakage_risks": [
            "the realized outcome must stay in payload.target; features/assignment are the only inputs",
            "per-unit intervention effect is NOT identifiable — the counterfactual arm is null, not a prediction target to be imputed",
        ],
        "known_limitations": [
            "streamed sample from offset 0 is entirely treatment=1 (dataset is ~85% treated); control rows appear later in the full ~14M-row file",
            "features are anonymized doubles with no semantic meaning",
            "aggregation block boundaries are a normalization choice (BLOCK_SIZE=500), not a source field",
        ],
        "license_implications": "CC-BY-NC-SA-4.0: NON-COMMERCIAL. Reserved as a held-out cross-dataset intervention-transfer evaluation only (not training).",
        "training_suitability": "eval_only",
        "assumptions": [
            "column names f0..f11, treatment, exposure, visit, conversion (verified from HF stream)",
            "treatment=1 is the treated arm, treatment=0 the control/holdout arm",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        rows = _load_rows(raw_dir)
        if not rows:
            raise FileNotFoundError(f"Criteo source produced 0 rows under {raw_dir}")
        # per-row INTERVENTION_EFFECT + POLICY_VALUE
        for i, row in enumerate(rows):
            block = i // BLOCK_SIZE
            yield from self._per_row(i, block, row)
        # per-(block, arm) POPULATION_RESPONSE
        yield from self._population(rows)

    # ---- per-row records ---------------------------------------------------------------
    def _per_row(self, i: int, block: int, row: dict) -> Iterator[dict]:
        feats = {c: row.get(c) for c in _FEATURE_COLS if c in row}
        t = _i(row.get("treatment"))
        realized = "treatment" if t == 1 else "control"
        episode_id = f"criteo_uplift-row-{i}"
        group_id = f"criteo_uplift-tp-{block}"
        loc = {"files": ["stream_shard.parquet"], "indices": [i], "ids": [episode_id]}
        causal = {"is_experimental": True, "randomized": True, "assignment_mechanism": "rct",
                  "treatment_arm": realized, "control_arm": "control", "propensity": None,
                  "unit_of_assignment": "impression"}
        ctx = {"actor_profile": {}, "current_observation": {"kind": "state", "meta": {"features": feats}},
               "world_state": {"trial": "criteo_uplift_v2.1", "time_period_block": block},
               "available_actions": None, "language": ""}

        # INTERVENTION_EFFECT — only the realized arm observed; counterfactual is null.
        treated_outcome = self._arm_outcome(t == 1, "treatment", row)
        control_outcome = self._arm_outcome(t == 0, "control", row)
        yield self.make(
            task_type="PREDICT_INTERVENTION_EFFECT",
            payload={
                "input": {
                    "population_or_actor_features": feats,
                    "treatment": {"arm": "treatment", "description": "ad-treatment (may be exposed to advertising)"},
                    "control": {"arm": "control", "description": "control/holdout (no ad-treatment)"},
                    "assignment_mechanism": {"type": "rct", "randomized": True, "unit": "impression",
                                             "realized_arm": realized},
                },
                "target": {
                    "treated_outcome": treated_outcome,
                    "control_outcome": control_outcome,
                    "estimated_effect": {"identified": False,
                                         "note": "per-unit effect not identifiable from a single realized arm"},
                },
            },
            episode_id=episode_id, sequence_index=0, group_id=group_id, actor_role="population",
            population_id=group_id, context=ctx, causal_metadata=causal, raw_locator=loc,
            transformation_steps=["read row", "fill realized arm's outcome", "leave counterfactual arm null"],
            data_quality={"missing_fields": ["counterfactual_outcome"], "chronology_verified": True,
                          "target_verified": True, "license_verified": True, "confidence": "high",
                          "warnings": ["only the realized arm is observed; counterfactual is null, not imputed"]})

        # POLICY_VALUE — reward = conversion under randomized assignment; no logged propensity.
        yield self.make(
            task_type="PREDICT_POLICY_VALUE",
            payload={"input": {"logged_context": feats,
                               "action": {"treatment": t, "exposure": _i(row.get("exposure"))},
                               "propensity": None, "logging_policy": "randomized_trial"},
                     "target": {"reward": _i(row.get("conversion")), "value": None}},
            episode_id=episode_id, sequence_index=0, group_id=group_id, actor_role="population",
            population_id=group_id, context=ctx, causal_metadata=causal, raw_locator=loc,
            transformation_steps=["read row", "action = treatment assignment", "reward = conversion"],
            data_quality={"missing_fields": ["propensity"], "chronology_verified": True,
                          "target_verified": True, "license_verified": True, "confidence": "high",
                          "warnings": ["assignment randomized; no per-row logged propensity (marginal rate is a population quantity)"]})

    @staticmethod
    def _arm_outcome(observed: bool, arm: str, row: dict) -> dict:
        if observed:
            return {"observed": True, "arm": arm, "conversion": _i(row.get("conversion")),
                    "visit": _i(row.get("visit")), "exposure": _i(row.get("exposure"))}
        return {"observed": False, "arm": arm, "conversion": None, "visit": None, "exposure": None}

    # ---- population aggregate -----------------------------------------------------------
    def _population(self, rows: list[dict]) -> Iterator[dict]:
        n_blocks = (len(rows) + BLOCK_SIZE - 1) // BLOCK_SIZE
        for block in range(n_blocks):
            chunk = rows[block * BLOCK_SIZE:(block + 1) * BLOCK_SIZE]
            arms: dict[int, list[dict]] = {}
            for row in chunk:
                arms.setdefault(_i(row.get("treatment")), []).append(row)
            for t, arm_rows in sorted(arms.items(), key=lambda kv: (kv[0] is None, kv[0])):
                arm = "treatment" if t == 1 else "control"
                n = len(arm_rows)
                conv = sum(_i(r.get("conversion")) or 0 for r in arm_rows)
                vis = sum(_i(r.get("visit")) or 0 for r in arm_rows)
                group_id = f"criteo_uplift-tp-{block}"
                pop_id = f"{group_id}-{arm}"
                episode_id = f"criteo_uplift-pop-tp{block}-{arm}"
                causal = {"is_experimental": True, "randomized": True, "assignment_mechanism": "rct",
                          "treatment_arm": arm, "control_arm": "control", "propensity": None,
                          "unit_of_assignment": "impression"}
                yield self.make(
                    task_type="PREDICT_POPULATION_RESPONSE",
                    payload={"input": {"population_features": {"n_units": n, "time_period_block": block},
                                       "intervention": {"arm": arm, "treatment": t,
                                                        "description": f"randomized {arm} arm over a block of impressions"}},
                             "target": {"aggregate_metrics": {
                                 "rate": (conv / n) if n else None, "n": n, "successes": conv,
                                 "conversion_rate": (conv / n) if n else None,
                                 "visit_rate": (vis / n) if n else None, "visit_successes": vis}}},
                    episode_id=episode_id, group_id=group_id, actor_role="population",
                    population_id=pop_id, causal_metadata=causal,
                    context={"world_state": {"trial": "criteo_uplift_v2.1", "time_period_block": block, "arm": arm},
                             "available_actions": None, "language": ""},
                    raw_locator={"files": ["stream_shard.parquet"],
                                 "indices": [block * BLOCK_SIZE, block * BLOCK_SIZE + n],
                                 "ids": [episode_id]},
                    transformation_steps=[f"aggregate block {block}", f"group by treatment arm={arm}",
                                          "conversion rate = successes/n"],
                    data_quality={"missing_fields": [], "chronology_verified": True, "target_verified": True,
                                  "license_verified": True, "confidence": "high", "inferred_fields": ["aggregation_block"],
                                  "warnings": ["aggregation block is a normalization choice, not a source time field"]})
