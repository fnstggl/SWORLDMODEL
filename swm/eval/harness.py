"""Backtesting harness: the exp001 ablation ladder on a temporal split (audit C.10, E.1).

Everything here is as-of-correct by construction: personas and pooled rates for a send at time t
are computed from store.history_asof(recipient, t) — future information is unreachable, not
merely unused.

run_ladder() is the go/no-go experiment (experiments/exp001_person_vs_segment.md):
  L0 base rate -> L1 +message features -> L2 +segment rate -> L3 +person posterior (pooled)
  -> L4 +their-text style factors & interactions
Rungs L5/L6 (operator corrections, VOI answers) run live, not in the retrospective harness.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from swm.actions.encoder import FEATURE_NAMES, encode_message, feature_vector
from swm.entities.persona import build_persona, segment_priors
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss, uplift_at_k
from swm.ingestion.store import EventStore, Send
from swm.transition.readout import LogisticReadout

_LOGIT = lambda p: math.log(max(1e-9, p) / max(1e-9, 1 - p))  # noqa: E731


@dataclass
class RungResult:
    name: str
    log_loss: float
    brier: float
    ece: float
    uplift_at_20: float
    n_test: int


def _persona_feature_rows(
    store: EventStore, sends: list[Send], seg_rate: float, rung: str
) -> tuple[list[list[float]], list[int]]:
    """Build as-of feature matrix for a rung. rung in {'L1','L2','L3','L4'}."""
    X, y = [], []
    for s in sends:
        persona = None
        if rung in ("L3", "L4"):
            hist = store.history_asof(s.recipient_id, s.timestamp)
            persona = build_persona(s.recipient_id, hist, segment_reply_rate=seg_rate)
        f = encode_message(
            s.content, send_ts=s.timestamp, channel=s.channel,
            persona=persona if rung == "L4" else None,  # interactions only at L4
        )
        row = feature_vector(f)
        # appended scalar features beyond the encoder's, by rung:
        row.append(_LOGIT(seg_rate) if rung in ("L2", "L3", "L4") else 0.0)
        row.append(_LOGIT(persona.responsiveness.mean) if persona is not None else 0.0)
        X.append(row)
        y.append(1 if s.replied else 0)
    return X, y


EXTRA_NAMES = ["segment_rate_logit", "person_rate_logit"]


def run_ladder(store: EventStore, *, split_quantile: float = 0.8, seed: int = 0) -> dict:
    """Run L0–L4 on a temporal split. Returns a dict with per-rung metrics + the verdict."""
    sends = store.labeled_sends()
    if len(sends) < 30:
        return {"error": f"only {len(sends)} labeled sends; need >= 30 for a meaningful split"}
    sends.sort(key=lambda s: s.timestamp)
    cut = sends[int(split_quantile * len(sends))].timestamp
    train = [s for s in sends if s.timestamp < cut]
    test = [s for s in sends if s.timestamp >= cut]
    if not train or not test or len(set(s.replied for s in train)) < 2:
        return {"error": "degenerate split (empty side or single-class training data)"}

    # Segment prior from TRAINING window only.
    per_recipient: dict[str, list[int]] = {}
    for s in train:
        per_recipient.setdefault(s.recipient_id, []).append(1 if s.replied else 0)
    seg_rate = segment_priors([(sum(v), len(v)) for v in per_recipient.values()])

    results: list[RungResult] = []

    # L0: base rate
    p0 = (sum(s.replied for s in train) + 1) / (len(train) + 2)
    preds0 = [p0] * len(test)
    ytest = [1 if s.replied else 0 for s in test]
    results.append(_score("L0_base_rate", ytest, preds0))

    # L1, L2, L3, L4: logistic with increasing evidence
    for rung in ("L1", "L2", "L3", "L4"):
        Xtr, ytr = _persona_feature_rows(store, train, seg_rate, rung)
        Xte, _ = _persona_feature_rows(store, test, seg_rate, rung)
        model = LogisticReadout(seed=seed).fit(Xtr, ytr)
        preds = [model.predict_proba(x) for x in Xte]
        results.append(_score(f"{rung}", ytest, preds))

    # bootstrap: does L3 beat L2 on log loss?
    l2 = next(r for r in results if r.name == "L2")
    l3 = next(r for r in results if r.name == "L3")
    p_boot = _bootstrap_pvalue(store, train, test, seg_rate, seed=seed)

    verdict = "GO" if (l3.log_loss < l2.log_loss and l3.ece <= l2.ece + 0.02 and p_boot < 0.05) \
        else "NO-GO (ship segment model; revisit with more per-person data)"
    return {
        "split_time": cut,
        "n_train": len(train),
        "n_test": len(test),
        "test_base_rate": sum(ytest) / len(ytest),
        "segment_rate_train": seg_rate,
        "rungs": [r.__dict__ for r in results],
        "bootstrap_p_L3_beats_L2": p_boot,
        "verdict": verdict,
        "seen_recipient_fraction": sum(
            1 for s in test if s.recipient_id in per_recipient) / len(test),
    }


def _score(name: str, y: list[int], p: list[float]) -> RungResult:
    return RungResult(
        name=name,
        log_loss=round(log_loss(y, p), 4),
        brier=round(brier_score(y, p), 4),
        ece=round(expected_calibration_error(y, p), 4),
        uplift_at_20=round(uplift_at_k(y, p, k=0.2), 4),
        n_test=len(y),
    )


def _bootstrap_pvalue(
    store: EventStore, train: list[Send], test: list[Send], seg_rate: float,
    *, n_boot: int = 200, seed: int = 0,
) -> float:
    """P(L3 does NOT beat L2) by resampling the TEST set (models fixed). Cheap and honest
    about test-set sampling noise, which dominates at v1 sizes."""
    X2tr, ytr = _persona_feature_rows(store, train, seg_rate, "L2")
    X3tr, _ = _persona_feature_rows(store, train, seg_rate, "L3")
    m2 = LogisticReadout(seed=seed).fit(X2tr, ytr)
    m3 = LogisticReadout(seed=seed).fit(X3tr, ytr)
    X2te, yte = _persona_feature_rows(store, test, seg_rate, "L2")
    X3te, _ = _persona_feature_rows(store, test, seg_rate, "L3")
    p2 = [m2.predict_proba(x) for x in X2te]
    p3 = [m3.predict_proba(x) for x in X3te]
    rng = random.Random(seed)
    n = len(yte)
    worse = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ll2 = log_loss([yte[i] for i in idx], [p2[i] for i in idx])
        ll3 = log_loss([yte[i] for i in idx], [p3[i] for i in idx])
        if ll3 >= ll2:
            worse += 1
    return worse / n_boot


IMPLEMENTED = True
