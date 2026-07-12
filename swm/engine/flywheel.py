"""The outcome-data flywheel — the moat. Every forecast is logged; every resolution re-calibrates.

MiroFish's community said it plainly about the whole category: "no accuracy moat — competitive advantage
requires a calibrated simulation layer trained on PROPRIETARY OUTCOME DATA." We already built the calibrated
layer (grade-or-abstain, per-domain temperatures, leak-free backtests). This module closes the loop that
turns it into a compounding asset:

  1. LOG      — every non-abstained forecast the engine emits is appended to a durable JSONL stream with
                everything needed to score it later: question, class, domain-kind, p, as_of, resolve_by,
                the engine config that produced it, and grounding provenance.
  2. RESOLVE  — when a record's resolve_by passes, `auto_resolve` retrieves the CURRENT news and asks the
                LLM whether the question has resolved and which way (cited); `record_outcome` takes manual
                resolutions. Resolution uses live retrieval on purpose — the outcome is in the present.
  3. REFIT    — `refit` pools the resolved stream per question-class, refits the out-of-sample temperature
                and PER-DOMAIN temperatures, and writes them into the GradeRegistry the live engine reads.
                Every resolved question makes the next forecast better-calibrated. That compounding stream
                is the thing agent-scale cannot substitute for.

The log is append-only and versioned per line (schema v1); records are hashed ids so re-logging the same
forecast is idempotent.
"""
from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LOG = "data/forecast_log.jsonl"


def _rid(question, as_of, ts):
    return hashlib.sha1(f"{question}|{as_of}|{int(ts)}".encode()).hexdigest()[:16]


@dataclass
class ForecastRecord:
    rid: str
    ts: float                              # when the forecast was made (unix)
    question: str
    question_class: str                    # e.g. society:event / deliberation:no_market / individual:response
    domain: str                            # router kind: deliberation | contest | announcement | ...
    mechanism: str
    p: float = None                        # P(yes) for a binary; top-option p otherwise
    distribution: dict = field(default_factory=dict)
    as_of: str = ""
    resolve_by: str = ""                   # when the outcome should be checkable (may be "")
    engine_config: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)    # coverage, standing_directional, n_passages
    status: str = "open"                   # open | resolved | unresolvable
    outcome: float = None                  # 1.0 / 0.0 for binary; for options: 1.0 if top option won
    resolved_ts: float = None
    resolution_source: str = ""
    v: int = 1


class FlywheelLog:
    """Append-only JSONL forecast log + resolution + refit. One instance per deployment."""

    def __init__(self, path: str = DEFAULT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. LOG ----------------
    def log(self, *, question, question_class, domain, mechanism, p=None, distribution=None,
            as_of="", resolve_by="", engine_config=None, grounding=None, ts=None) -> str:
        ts = ts if ts is not None else _time.time()
        rec = ForecastRecord(rid=_rid(question, as_of, ts), ts=ts, question=str(question)[:400],
                             question_class=question_class, domain=domain, mechanism=mechanism,
                             p=(round(float(p), 4) if p is not None else None),
                             distribution={k: round(float(v), 4) for k, v in (distribution or {}).items()},
                             as_of=as_of, resolve_by=resolve_by or "",
                             engine_config=engine_config or {}, grounding=grounding or {})
        if any(r.rid == rec.rid for r in self.load()):    # idempotent
            return rec.rid
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec.rid

    def load(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(ForecastRecord(**json.loads(line)))
            except (ValueError, TypeError):
                continue
        # last state per rid wins (resolutions are appended as updated records)
        by_id = {}
        for r in out:
            by_id[r.rid] = r
        return list(by_id.values())

    def open_records(self, *, due_only=True, now=None, no_date_grace_days=30) -> list:
        now = now if now is not None else _time.time()
        recs = [r for r in self.load() if r.status == "open"]
        if not due_only:
            return recs
        due = []
        for r in recs:
            if r.resolve_by:
                try:
                    if _time.mktime(_time.strptime(r.resolve_by, "%Y-%m-%d")) <= now:
                        due.append(r)
                except ValueError:
                    continue
            elif now - r.ts >= no_date_grace_days * 86400:   # no known date → re-check periodically after grace
                due.append(r)
        return due

    # ---------------- 2. RESOLVE ----------------
    def record_outcome(self, rid: str, outcome: float, *, source="manual") -> bool:
        recs = {r.rid: r for r in self.load()}
        r = recs.get(rid)
        if r is None:
            return False
        r.status, r.outcome, r.resolved_ts, r.resolution_source = "resolved", float(outcome), _time.time(), source
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(r)) + "\n")
        return True

    def auto_resolve(self, llm, *, search_fn=None, limit=25, now=None) -> dict:
        """Check every due open record against CURRENT news: retrieval + an LLM resolution read (cited).
        Only records a verdict when the evidence is decisive; otherwise the record stays open."""
        from swm.engine.grounding import parse_json
        from swm.engine.retrieval import multi_search
        search = search_fn or multi_search
        checked = resolved = 0
        for r in self.open_records(now=now)[:limit]:
            checked += 1
            passages = search([r.question, f"{r.question} result outcome"], 6)
            if len(passages) < 2:
                continue
            ptxt = "\n".join(p.cite() for p in passages[:16])
            v = parse_json(llm(
                f"QUESTION (a forecast we made, resolve_by {r.resolve_by}): {r.question}\n"
                f"CURRENT EVIDENCE:\n{ptxt}\n\n"
                f"Has this question's outcome been decided? Use ONLY the evidence; be conservative.\n"
                f'Return ONLY JSON: {{"resolved": <true|false>, "outcome": "yes"|"no"|null, '
                f'"evidence": "<cite>"}}')) or {}
            if v.get("resolved") and v.get("outcome") in ("yes", "no"):
                self.record_outcome(r.rid, 1.0 if v["outcome"] == "yes" else 0.0,
                                    source=f"auto:{str(v.get('evidence', ''))[:120]}")
                resolved += 1
        return {"checked": checked, "resolved": resolved}

    # ---------------- 3. REFIT ----------------
    @staticmethod
    def _horizon_bucket(r) -> str:
        """Forecast horizon at emit time: resolve_by − as_of (fallback: resolution delay). Calibration must be
        horizon-aware (vision gap 7): an election at 180 days and a reply within 24h are different classes of
        uncertainty even in the same domain."""
        for a, b in ((r.resolve_by, r.as_of),):
            try:
                days = (_time.mktime(_time.strptime(a, "%Y-%m-%d")) -
                        _time.mktime(_time.strptime(b[:10], "%Y-%m-%d"))) / 86400.0
                break
            except (ValueError, TypeError):
                days = None
        if days is None and r.resolved_ts:
            days = (r.resolved_ts - r.ts) / 86400.0
        if days is None:
            return "h?"
        return "h<7d" if days < 7 else ("h7-45d" if days < 45 else "h>45d")

    def refit(self, registry=None, *, min_n=12, holdout_frac=0.3) -> dict:
        """Refit per-class temperature + per-(domain, horizon) temperatures from the RESOLVED stream and write
        them into the live GradeRegistry. STRICT TEMPORAL DISCIPLINE (vision gap 7): the temperature is fit on
        the EARLIER records only; the report's held-out metrics come from the LATER untouched records — never
        fit-and-report on the same rows. (The recorded T for production still uses all data — production wants
        the best estimate — but the honest performance number is the temporal holdout's.)"""
        from swm.engine.calibrate import GradeRegistry, crossfit_temperature, fit_temperature
        registry = registry or GradeRegistry()
        resolved = [r for r in self.load() if r.status == "resolved" and r.p is not None]
        report = {"n_resolved": len(resolved), "classes": {}}
        by_class = {}
        for r in resolved:
            by_class.setdefault(r.question_class, []).append(r)
        for qc, rs in by_class.items():
            if len(rs) < min_n:
                report["classes"][qc] = {"n": len(rs), "skipped": f"n<{min_n}"}
                continue
            rs = sorted(rs, key=lambda r: r.ts)             # time order — earlier trains, later evaluates
            cut = max(1, int(len(rs) * (1 - holdout_frac)))
            train, hold = rs[:cut], rs[cut:]
            hold_metrics = None
            if len(hold) >= 4:
                T_tr = fit_temperature([r.p for r in train], [r.outcome for r in train])
                from swm.engine.calibrate import apply_temperature
                hp = [apply_temperature(r.p, T_tr) for r in hold]
                hy = [r.outcome for r in hold]
                hold_metrics = {"n_holdout": len(hold), "T_train": T_tr,
                                "holdout_brier_raw": round(sum((r.p - y) ** 2 for r, y in
                                                               zip(hold, hy)) / len(hold), 4),
                                "holdout_brier_cal": round(sum((p - y) ** 2 for p, y in
                                                               zip(hp, hy)) / len(hold), 4)}
            preds = [r.p for r in rs]
            ys = [r.outcome for r in rs]
            cal = crossfit_temperature(preds, ys)
            class_rate = sum(ys) / len(ys)
            skill = 1.0 - (sum((p - y) ** 2 for p, y in zip(preds, ys)) /
                           max(1e-9, sum((class_rate - y) ** 2 for y in ys)))
            entry = registry.record(qc, backtest_report={"skill_vs": {"class_rate": round(skill, 4)},
                                                         "n": len(rs),
                                                         "rmse": (sum((p - y) ** 2 for p, y in zip(preds, ys))
                                                                  / len(rs)) ** 0.5},
                                    preds=preds, outcomes=ys, temperature=cal["temperature"])
            domain_T = {}
            for dom in {r.domain for r in rs}:
                sub = [(r.p, r.outcome) for r in rs if r.domain == dom]
                if len(sub) >= 8:
                    domain_T[dom] = fit_temperature([p for p, _ in sub], [y for _, y in sub])
            # gap 7: horizon-bucketed temperatures too — (domain, horizon) compound keys in the same registry
            for key in {f"{r.domain}|{self._horizon_bucket(r)}" for r in rs}:
                sub = [(r.p, r.outcome) for r in rs if f"{r.domain}|{self._horizon_bucket(r)}" == key]
                if len(sub) >= 8:
                    domain_T[key] = fit_temperature([p for p, _ in sub], [y for _, y in sub])
            if domain_T:
                registry.record_domain_temperatures(qc, domain_T)
            report["classes"][qc] = {"n": len(rs), "grade": entry["grade"],
                                     "temperature": cal["temperature"], "per_domain": domain_T,
                                     "temporal_holdout": hold_metrics,
                                     "brier_skill_vs_class_rate": round(skill, 4)}
        return report
