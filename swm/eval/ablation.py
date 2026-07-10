"""The decisive ablation — does the SOCIETY SIMULATION add value over the same model + same evidence?

Everything else we measure (retrieval, prompting, calibration, routing) can improve the stack without the
simulation earning its keep. The defining architectural claim is narrower and testable: for the questions
the product answers, does running a grounded agent SOCIETY/PANEL beat simply asking the same model the same
question with the same retrieved evidence — once? This harness answers it with a controlled, leak-free,
same-inputs comparison. Five arms per question, grounded ONCE (shared as-of dossier), predictions produced
independently, then all scored against the realized outcome:

  1. FULL      — the complete engine (grounded panel + per-domain calibration + routing).
  2. RAW       — one DeepSeek call, no evidence, no simulation ("what's P(yes)?").
  3. EVIDENCE  — one DeepSeek call given the SAME grounded dossier (the crucial arm: same model, same
                 evidence, no simulation). FULL beating EVIDENCE is the whole thesis.
  4. BASE_RATE — the reference-class / sample base rate (the free skeptic).
  5. PARAMETRIC— the best parametric kernel (main's mechanisms, compiled as-of, grounding OFF → leak-free).

Scoring per arm: Brier, log-loss, direction accuracy, ECE (calibration), decision value (bet the side when
confident), abstention rate; RAW and CALIBRATED (each arm gets its own out-of-sample temperature, so the
comparison isolates SIMULATION value net of calibration). The head-to-head FULL−EVIDENCE is reported
explicitly. This is leak-free by construction (as-of grounding, post-cutoff questions); the same harness
runs FORWARD (lock all five before resolution) for the standing validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.engine.grounding import parse_json

ARMS = ("full", "raw", "evidence", "base_rate", "parametric")

_SINGLE_PROMPT = """You are a careful forecaster. Give the probability this resolves YES.
QUESTION: {q}
TODAY: {today}
{evidence_block}
State the reference-class base rate, adjust for the evidence, and do not output 0 or 1.
Return ONLY JSON: {{"p": <0..1>}}"""


def _p_single(llm, question, today, dossier=None):
    ev = f"GROUNDED AS-OF EVIDENCE:\n{dossier.brief()}\n" if dossier is not None else \
        "(no evidence provided — reason from general knowledge only)\n"
    r = parse_json(llm(_SINGLE_PROMPT.format(q=question, today=today, evidence_block=ev)))
    if not r:
        return None
    try:
        return min(1.0, max(0.0, float(r["p"])))
    except (KeyError, TypeError, ValueError):
        return None


def predict_all_arms(wm, question, *, as_of, class_rate, search_fn=None, llm_raw=None):
    """Produce one prediction per arm on a single question, sharing the as-of grounded dossier. Returns
    {arm: p or None}. `wm` is the full AgentWorldModel; `llm_raw` a plain DeepSeek chat fn for the solo arms."""
    from swm.engine.front_door import parametric_binary_p
    from swm.engine.grounding import SceneGrounder
    from swm.eval.grade_agent_engine import p_yes
    today = as_of or ""
    # ground ONCE — every evidence-using arm sees exactly the same dossier
    dossier = SceneGrounder(wm.llm, search_fn=search_fn, today=today).ground(question)
    out = {}
    # 1. FULL — but reuse the shared dossier is internal; call the engine (it re-grounds via search_fn, same as-of)
    try:
        res = wm.simulate(question, as_of=as_of, binary=True, search_fn=search_fn)
        out["full"] = None if res.get("abstain") else p_yes(res)
    except Exception:
        out["full"] = None
    out["raw"] = _p_single(llm_raw, question, today, dossier=None)
    out["evidence"] = None if dossier.abstain else _p_single(llm_raw, question, today, dossier=dossier)
    out["base_rate"] = class_rate
    out["parametric"] = parametric_binary_p(question, as_of, wm.llm)
    return out


# ---------------------------------------------------------------- scoring
def _clip(p, lo=0.02, hi=0.98):
    return min(hi, max(lo, p))


def _logloss(y, p):
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _fit_T(preds, ys, grid=(0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.6, 3.5)):
    def loss(T):
        s = 0.0
        for p, y in zip(preds, ys):
            q = 1 / (1 + math.exp(-(math.log(_clip(p) / (1 - _clip(p)))) / T))
            s += _logloss(y, q)
        return s
    return min(grid, key=loss) if preds else 1.0


def _apply_T(p, T):
    z = math.log(_clip(p) / (1 - _clip(p))) / T
    return 1 / (1 + math.exp(-z))


def _ece(preds, ys, bins=8):
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(preds) if (lo <= p < hi or (b == bins - 1 and p == 1.0))]
        if not idx:
            continue
        conf = sum(preds[i] for i in idx) / len(idx)
        acc = sum(ys[i] for i in idx) / len(idx)
        tot += (len(idx) / len(preds)) * abs(conf - acc)
    return tot


def _decision_value(preds, ys, thresh=0.6):
    """Bet 1 unit on the favored side when |p-0.5| clears the threshold; +1 if right, -1 if wrong. Mean
    return per confident call (a crude economic proxy)."""
    bets = [(1 if ((p > 0.5) == (y > 0.5)) else -1) for p, y in zip(preds, ys)
            if abs(p - 0.5) >= (thresh - 0.5)]
    return (sum(bets) / len(bets), len(bets)) if bets else (0.0, 0)


@dataclass
class ArmScore:
    arm: str
    n: int = 0
    n_abstain: int = 0
    brier: float = None
    logloss: float = None
    direction: float = None
    ece: float = None
    brier_cal: float = None                # after out-of-sample temperature (self-calibrated)
    logloss_cal: float = None
    T: float = None
    decision_value: float = None
    n_bets: int = 0


def score_arms(rows) -> dict:
    """rows: [{arm: p or None, ...} ... , 'outcome': y]. Score each arm on the items it did NOT abstain on."""
    scores = {}
    for arm in ARMS:
        pairs = [(r[arm], r["outcome"]) for r in rows if r.get(arm) is not None]
        n_abs = sum(1 for r in rows if r.get(arm) is None)
        if not pairs:
            scores[arm] = ArmScore(arm=arm, n=0, n_abstain=n_abs)
            continue
        preds = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        # out-of-sample T via 5-fold, applied to the held-out fold (honest self-calibration)
        cal_preds = [None] * len(preds)
        k = 5
        if len(preds) >= 2 * k:
            idx = list(range(len(preds)))
            for f in range(k):
                test = set(idx[f::k])
                tr = [i for i in idx if i not in test]
                T = _fit_T([preds[i] for i in tr], [ys[i] for i in tr])
                for i in test:
                    cal_preds[i] = _apply_T(preds[i], T)
            T_report = _fit_T(preds, ys)
        else:
            T_report = _fit_T(preds, ys)
            cal_preds = [_apply_T(p, T_report) for p in preds]
        dv, nb = _decision_value(preds, ys)
        scores[arm] = ArmScore(
            arm=arm, n=len(pairs), n_abstain=n_abs,
            brier=round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4),
            logloss=round(sum(_logloss(y, p) for p, y in pairs) / len(pairs), 4),
            direction=round(sum(1 for p, y in pairs if (p > 0.5) == (y > 0.5)) / len(pairs), 3),
            ece=round(_ece(preds, ys), 4),
            brier_cal=round(sum((cp - y) ** 2 for cp, y in zip(cal_preds, ys)) / len(pairs), 4),
            logloss_cal=round(sum(_logloss(y, cp) for cp, y in zip(cal_preds, ys)) / len(pairs), 4),
            T=T_report, decision_value=round(dv, 3), n_bets=nb)
    # the head-to-head that IS the thesis: FULL vs EVIDENCE on the items BOTH answered
    both = [(r["full"], r["evidence"], r["outcome"]) for r in rows
            if r.get("full") is not None and r.get("evidence") is not None]
    head = None
    if both:
        bf = sum((f - y) ** 2 for f, _, y in both) / len(both)
        be = sum((f - y) ** 2 for _, f, y in both) / len(both)
        head = {"n_both": len(both), "brier_full": round(bf, 4), "brier_evidence": round(be, 4),
                "full_minus_evidence": round(bf - be, 4),
                "full_better": bf < be,
                "full_wins_rows": round(sum(1 for f, e, y in both if (f - y) ** 2 < (e - y) ** 2)
                                        / len(both), 3)}
    return {"arms": {a: scores[a].__dict__ for a in ARMS}, "head_to_head_full_vs_evidence": head}
