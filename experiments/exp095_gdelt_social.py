"""EXP-095: does GDELT as-of social-state grounding improve the forecaster on the country/social slice?

For every clean question that NAMES a country (the slice where a country's measured event-stream state could
matter), run the latent-state simulation two ways — WITHOUT and WITH the GDELT social grounder (which injects
the measured as-of conflict/violence/protest trajectory and adds a grounded escalation driver) — and compare
AUC / calibrated log-loss / skill-vs-crowd. Report overall AND on the genuine CONFLICT/POLITICS subset (where
the social state is on-topic) vs the off-topic remainder (sports/other) — the routing signal for when to fire it.

Run: HF_TOKEN=.. DEEPSEEK_API_KEY=.. python -m experiments.exp095_gdelt_social [max_items]
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.api.asof_market import CryptoAsofGrounder
from swm.api.gdelt_social import GdeltSocialGrounder, detect_country
from swm.api.inner_crowd import _logit, _sig
from swm.api.latent_forecast import latent_forecast
from swm.api.resilient_llm import resilient_chat_fn
from swm.eval.forecasting_corpus import load_corpus
from swm.eval.metrics import log_loss

PRED = "experiments/results/exp095_social_predictions.json"
RESULT = "experiments/results/exp095_gdelt_social.json"
_CONFLICT = re.compile(r"war|invad|ceasefire|attack|strike|missile|troop|military|coup|sanction|nuclear|"
                       r"hostage|border|conflict|protest|election|president|minister|regime|treaty|"
                       r"annex|occupy|referendum|assassin|overthrow|unrest|riot", re.I)


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    return round(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)), 4) if pos and neg else None


def _temp(ps, ys):
    return min([x / 20 for x in range(2, 31)], key=lambda l: log_loss(ys, [_sig(l * _logit(p)) for p in ps]))


def _score(name, ps, ys, crowd):
    half = len(ps) // 2
    lam = _temp(ps[:half], ys[:half])
    cal = [_sig(lam * _logit(p)) for p in ps]
    base = sum(ys[:half]) / max(1, half)
    ll = log_loss(ys[half:], cal[half:])
    llc = log_loss(ys[half:], crowd[half:])
    llb = log_loss(ys[half:], [base] * len(ys[half:]))
    return {"name": name, "n": len(ps), "auc": _auc(ps, ys), "ll_cal": round(ll, 4),
            "skill_vs_crowd": round(1 - ll / llc, 4), "skill_vs_base": round(1 - ll / llb, 4)}


def run(max_items=200, n=3000) -> dict:
    corpus = [it for it in load_corpus() if it.cutoff_clean and detect_country(it.question)[0]][:max_items]
    llm = resilient_chat_fn(max_tokens=700)
    asof = CryptoAsofGrounder()
    social = GdeltSocialGrounder(window_days=21)

    done = json.loads(Path(PRED).read_text()) if Path(PRED).exists() else {}
    pr = {q: dict(v) for q, v in done.items()}
    tasks = [(it, cond) for it in corpus for cond in ("plain", "grounded")
             if not (it.qid in pr and cond in pr[it.qid])]

    def _one(it, cond):
        sg = social if cond == "grounded" else None
        p, _ = latent_forecast(it.question, it.as_of, it.resolve_ts, llm, n=n,
                               metric_grounder=asof, social_grounder=sg)
        return it.qid, cond, p

    ct = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        for fut in as_completed([ex.submit(_one, it, c) for it, c in tasks]):
            qid, cond, p = fut.result()
            pr.setdefault(qid, {})[cond] = p
            ct += 1
            if ct % 40 == 0:
                Path(PRED).write_text(json.dumps(pr))
                print(f"    {ct}/{len(tasks)} (cache={llm.calls.get('cache', 0)})")
    Path(PRED).write_text(json.dumps(pr))

    qids = [it.qid for it in corpus if it.qid in pr and pr[it.qid].get("plain") is not None
            and pr[it.qid].get("grounded") is not None]
    by = {it.qid: it for it in corpus}
    ys = [by[q].outcome for q in qids]
    crowd = [by[q].crowd_prob for q in qids]
    conflict = [bool(_CONFLICT.search(by[q].question)) for q in qids]

    def slice_eval(mask, label):
        idx = [i for i, m in enumerate(mask) if m]
        if len(idx) < 8:
            return None
        sub_ys = [ys[i] for i in idx]
        sub_cr = [crowd[i] for i in idx]
        out = {"slice": label, "n": len(idx)}
        for cond in ("plain", "grounded"):
            ps = [pr[qids[i]][cond] for i in idx]
            e = _score(cond, ps, sub_ys, sub_cr)
            out[cond] = {"auc": e["auc"], "ll_cal": e["ll_cal"], "skill_vs_crowd": e["skill_vs_crowd"]}
        out["crowd_auc"] = _auc(sub_cr, sub_ys)
        return out

    n_moved = sum(1 for q in qids if abs(pr[q]["grounded"] - pr[q]["plain"]) > 0.02)
    slices = [s for s in [slice_eval([True] * len(qids), "ALL_country"),
                          slice_eval(conflict, "CONFLICT/POLITICS"),
                          slice_eval([not c for c in conflict], "off-topic(sports/other)")] if s]
    res = {"n": len(qids), "n_conflict": sum(conflict), "n_grounding_moved_forecast": n_moved, "slices": slices}
    Path(RESULT).write_text(json.dumps(res, indent=1))

    print(f"\nEXP-095  GDELT social grounding on {len(qids)} country-naming questions "
          f"({sum(conflict)} conflict/politics; grounding moved {n_moved})")
    for s in slices:
        print(f"\n  [{s['slice']}]  n={s['n']}  crowd_auc={s['crowd_auc']}")
        for cond in ("plain", "grounded"):
            e = s[cond]
            print(f"    {cond:9s} AUC={e['auc']}  ll_cal={e['ll_cal']}  skill_vs_crowd={e['skill_vs_crowd']}")
    print(f"\n  wrote {RESULT}")
    return res


if __name__ == "__main__":
    run(max_items=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
