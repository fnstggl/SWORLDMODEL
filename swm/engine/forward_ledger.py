"""PART C — the forward-locked, append-only, VERSIONED multi-arm ledger.

The flywheel (`flywheel.py`) logs the single PRODUCTION forecast for calibration. This ledger is different and
complementary: for a forward question it LOCKS every ablation arm (B0/B1/B2 always; B3 on a stratified sample;
B4-B10 on a diagnostic sample) together with the full provenance needed to compare them fairly LATER, when the
world resolves — the contamination-proof version of the Part-B experiment.

Two invariants the program demands:
  * APPEND-ONLY. Locks and resolutions are new lines; nothing is edited in place. `load()` folds to the last
    state per (qid, lock_version).
  * VERSIONED. A row's `lock_version` is a hash of everything that would make a re-run non-comparable — code
    commit, model version, evidence hash, and the engine/prompt config. Change ANY of them and a re-lock
    writes a NEW version row; the old forecast is never overwritten (hard rule #: "never overwrite a prior
    forecast after changing prompts/evidence/model/calibration/routing/code — create a new version instead").

After resolution, `score()` returns per-arm Brier/log-loss/direction on the resolved rows, the paired marginal
ladder (reusing the Part-B stats), and the per-class BEST-ARCHITECTURE ranking (accuracy at its cost tier).
`refit_eligible()` returns only rows a calibrator may fit on (resolved, not flagged `reported`) so evaluation
rows are never used to tune the calibration that is then reported on them.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LEDGER = "data/forward_ledger.jsonl"


def _qid(question, as_of):
    return hashlib.sha1(f"{question}|{as_of}".encode()).hexdigest()[:16]


def _version(commit, model, evidence_hash, config) -> str:
    payload = json.dumps({"commit": commit, "model": model, "ev": evidence_hash, "cfg": config},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


@dataclass
class LedgerRow:
    qid: str
    lock_version: str                      # hash(commit, model, evidence_hash, config) — a new value = new row
    ts: float
    question: str
    outcome_space: dict = field(default_factory=dict)   # native answer space
    resolution_criteria: str = ""
    question_class: str = ""
    domain: str = ""
    horizon_days: float = None
    as_of: str = ""
    resolve_by: str = ""                   # expected resolution date
    evidence_hash: str = ""
    prompt_hashes: dict = field(default_factory=dict)
    commit: str = ""
    model: str = ""
    model_params: dict = field(default_factory=dict)
    selected_architecture: str = ""        # what production routing chose for B2
    router_explanation: str = ""
    n_agents: int = None
    segment_weights: dict = field(default_factory=dict)
    n_rounds: int = None
    interaction_structure: str = ""
    arms: dict = field(default_factory=dict)   # {arm: {"p":..,"spend":{calls,tokens,cost,seconds},"note":..}}
    abstain: bool = False
    abstain_reason: str = ""
    status: str = "locked"                 # locked | resolved | unresolvable
    outcome: float = None
    resolved_ts: float = None
    resolution_source: str = ""
    reported: bool = False                 # True once used in a reported eval → excluded from calibration refit
    v: int = 2                             # ledger schema version


class ForwardLedger:
    def __init__(self, path: str = DEFAULT_LEDGER):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- lock ----------------
    def lock_from_prediction(self, pred: dict, *, question, question_class, domain, outcome_space=None,
                             resolution_criteria="", horizon_days=None, resolve_by="", config=None,
                             selected_architecture="", router_explanation="", n_agents=None,
                             segment_weights=None, n_rounds=None, interaction_structure="",
                             prompt_hashes=None, model_params=None, ts=None) -> str:
        """Lock a `predict_arms` output (tiered_ablation) as a forward row. Idempotent per (qid, lock_version):
        re-locking the SAME config is a no-op; a changed commit/model/evidence/config writes a NEW version."""
        meta = pred.get("_meta", {})
        commit, model = meta.get("commit", ""), meta.get("model", "")
        ev_hash = meta.get("evidence_hash", "")
        cfg = dict(config or {})
        ver = _version(commit, model, ev_hash, cfg)
        qid = _qid(question, pred.get("_meta", {}).get("as_of", ""))
        arms = {a: v for a, v in pred.items() if not a.startswith("_")
                and isinstance(v, dict) and "p" in v}
        row = LedgerRow(
            qid=qid, lock_version=ver, ts=(ts if ts is not None else _time.time()),
            question=str(question)[:400], outcome_space=outcome_space or {}, resolution_criteria=resolution_criteria,
            question_class=question_class, domain=domain, horizon_days=horizon_days,
            as_of=meta.get("as_of", ""), resolve_by=resolve_by, evidence_hash=ev_hash,
            prompt_hashes=prompt_hashes or {}, commit=commit, model=model, model_params=model_params or {},
            selected_architecture=selected_architecture, router_explanation=router_explanation,
            n_agents=n_agents, segment_weights=segment_weights or {}, n_rounds=n_rounds,
            interaction_structure=interaction_structure, arms=arms, abstain=bool(meta.get("abstain")))
        key = (row.qid, row.lock_version)
        if key in {(r.qid, r.lock_version) for r in self.load()}:     # idempotent per version
            return row.lock_version
        self._append(row)
        return row.lock_version

    def _append(self, row: LedgerRow):
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")

    def load(self) -> list:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            try:
                rows.append(LedgerRow(**json.loads(line)))
            except (ValueError, TypeError):
                continue
        by_key = {}
        for r in rows:                                                # last state per (qid, version) wins
            by_key[(r.qid, r.lock_version)] = r
        return list(by_key.values())

    # ---------------- resolve ----------------
    def resolve(self, qid: str, outcome: float, *, source="manual", lock_version=None) -> int:
        """Append a resolution for every locked version of this qid (or one version). Append-only."""
        n = 0
        for r in self.load():
            if r.qid == qid and r.status == "locked" and (lock_version is None or r.lock_version == lock_version):
                r.status, r.outcome, r.resolved_ts, r.resolution_source = \
                    "resolved", float(outcome), _time.time(), source
                self._append(r)
                n += 1
        return n

    def open_rows(self):
        return [r for r in self.load() if r.status == "locked"]

    def refit_eligible(self):
        """Rows a calibrator MAY fit on: resolved and not flagged as reported-eval rows."""
        return [r for r in self.load() if r.status == "resolved" and not r.reported]

    # ---------------- score ----------------
    def score(self, *, min_n=8) -> dict:
        """Per-arm scores on resolved rows + paired marginal ladder + per-class best-architecture ranking."""
        from swm.eval.tiered_ablation import ALL_ARMS, report_marginals
        resolved = [r for r in self.load() if r.status == "resolved" and r.outcome is not None]
        if not resolved:
            return {"n_resolved": 0}
        runs = []
        for r in resolved:
            run = {"outcome": r.outcome, "question_class": r.question_class}
            for a in ALL_ARMS:
                run[a] = r.arms.get(a, {"p": None})
            runs.append(run)
        rep = report_marginals(runs)
        # per-class best architecture (lowest Brier among arms with >= min_n resolved in that class)
        classes = {}
        for qc in {r.question_class for r in resolved}:
            sub = [run for run in runs if run["question_class"] == qc]
            arm_brier = {}
            for a in ALL_ARMS:
                pairs = [((run[a] or {}).get("p"), run["outcome"]) for run in sub]
                pairs = [(p, y) for p, y in pairs if p is not None]
                if len(pairs) >= min_n:
                    arm_brier[a] = round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4)
            if arm_brier:
                best = min(arm_brier, key=arm_brier.get)
                classes[qc] = {"n": len(sub), "arm_brier": arm_brier, "best_arm": best}
        return {"n_resolved": len(resolved), "report": rep, "per_class_best_architecture": classes}
