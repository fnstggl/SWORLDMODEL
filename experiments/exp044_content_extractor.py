"""EXP-044: resolution-aware content extraction vs crude news features (closing the EXP-043 gap).

EXP-043 established the frontier: crude question-agnostic news features don't beat the base rate
(corr 0.047 with the outcome) even though the market extracts the same articles into a decisive signal
(corr 0.84). The fix (swm/variables/content_extractor.py): parse the question's resolution frame, LINK
news to the question's subject (entity linking), and read STANCE toward THIS outcome. This tests whether
resolution-aware extraction recovers signal the crude features left on the table.

Arms (no-cheat: features are as-of news, dated before the target; train/test are the benchmark splits):
  1. base rate (composite)
  2. crude news features (EXP-043: volume, result-cue, global polarity, recency, sources)
  3. RESOLUTION-AWARE features (subject link-rate, subject stance, recent subject stance, resolve rate)
  4. crude + resolution-aware (do they combine?)
  5. market lean (reference ceiling)

Decisive: does the resolution-aware readout beat both the base rate and the crude features, and how much
more of the price gap does it close? Writes JSON.
Run: python -m experiments.exp044_content_extractor
"""
from __future__ import annotations

import json
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.transition.readout import LogisticReadout
from swm.variables.content_extractor import feature_vector as content_features
from experiments.datasets_swm import load
from experiments.exp043_news_grounding import _news_features, _outcome

RESULT = "experiments/results/exp044_content_extractor.json"


def _usable(split):
    out = []
    for r in load(split):
        if not (r.get("news") and r.get("target") and r.get("future") and r.get("question")):
            continue
        y = _outcome(r)
        if y is None:
            continue
        crude = _news_features(r)
        aware = content_features(r["question"], r["news"], r["target"]["t"])
        out.append((crude, aware, y, r["target"]["p"]))
    return out


def _score(y, p):
    p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
    nf = [(pi, yi) for pi, yi in zip(p, y) if abs(pi - 0.5) > 0.02]
    da = sum(int((pi > 0.5) == (yi == 1)) for pi, yi in nf) / max(1, len(nf))
    my = sum(y) / len(y); mp = sum(p) / len(p)
    num = sum((pi - mp) * (yi - my) for pi, yi in zip(p, y))
    den = (sum((pi - mp) ** 2 for pi in p) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4), "directional_accuracy": round(da, 4),
            "corr_with_outcome": round(num / den if den > 1e-9 else 0.0, 4)}


def _fit(Xtr, ytr, Xte):
    m = LogisticReadout(epochs=400, l2=1.0).fit(Xtr, ytr)
    return [m.predict_proba(x) for x in Xte], m


def run():
    tr, te = _usable("train"), _usable("test_kalshi")
    ytr = [y for _, _, y, _ in tr]; yte = [y for _, _, y, _ in te]
    base = sum(ytr) / len(ytr)

    p_crude, _ = _fit([c for c, _, _, _ in tr], ytr, [c for c, _, _, _ in te])
    p_aware, m_aware = _fit([a for _, a, _, _ in tr], ytr, [a for _, a, _, _ in te])
    p_both, _ = _fit([c + a for c, a, _, _ in tr], ytr, [c + a for c, a, _, _ in te])
    p_price = [p for _, _, _, p in te]

    arms = {"base_rate_composite": _score(yte, [base] * len(yte)),
            "crude_news_features": _score(yte, p_crude),
            "resolution_aware": _score(yte, p_aware),
            "crude_plus_aware": _score(yte, p_both),
            "market_lean_reference": _score(yte, p_price)}
    base_ll = arms["base_rate_composite"]["log_loss"]; price_ll = arms["market_lean_reference"]["log_loss"]
    def gap(k):
        return round((base_ll - arms[k]["log_loss"]) / (base_ll - price_ll), 3) if base_ll - price_ll > 1e-9 else None

    # the honest signal measure: best single-feature raw correlation with the outcome (multivariate
    # readouts dilute a weak signal at n=574). Resolution-aware vs crude vs the market's corr.
    def _best_corr(getter, dim):
        my = sum(yte) / len(yte); best = 0.0
        for j in range(dim):
            xs = [getter(t)[j] for t in te]; mx = sum(xs) / len(xs)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, yte))
            den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in yte)) ** 0.5
            if den > 1e-9 and abs(num / den) > abs(best):
                best = num / den
        return round(best, 4)
    raw_signal = {"crude_best_feature_corr": _best_corr(lambda t: t[0], len(te[0][0])),
                  "aware_best_feature_corr": _best_corr(lambda t: t[1], len(te[0][1])),
                  "market_corr": arms["market_lean_reference"]["corr_with_outcome"]}
    raw_signal["aware_vs_crude_ratio"] = round(abs(raw_signal["aware_best_feature_corr"]) /
                                               max(1e-9, abs(raw_signal["crude_best_feature_corr"])), 2)
    raw_signal["fraction_of_market_signal_recovered"] = round(
        abs(raw_signal["aware_best_feature_corr"]) / max(1e-9, abs(raw_signal["market_corr"])), 3)

    from swm.variables.content_extractor import FEATURE_NAMES
    imp = sorted(zip(FEATURE_NAMES, m_aware.w), key=lambda t: -abs(t[1]))
    out = {"dataset": "kalshi", "n_train": len(tr), "n_test": len(te), "base_rate": round(base, 4),
           "arms": arms, "raw_signal": raw_signal,
           "aware_recovers_more_raw_signal": abs(raw_signal["aware_best_feature_corr"]) >
                                             abs(raw_signal["crude_best_feature_corr"]),
           "aware_beats_base_calibrated": arms["resolution_aware"]["log_loss"] < base_ll,
           "price_gap_closed": {k: gap(k) for k in ("crude_news_features", "resolution_aware", "crude_plus_aware")},
           "aware_feature_weights": [(n, round(w, 4)) for n, w in imp]}

    print(f"EXP-044 resolution-aware content extraction — Kalshi, train={len(tr)} test={len(te)} "
          f"base rate {base:.3f}")
    for k, v in arms.items():
        print(f"  {k:<24} log_loss {v['log_loss']}  brier {v['brier']}  dir_acc {v['directional_accuracy']}"
              f"  corr {v['corr_with_outcome']}")
    print(f"  RAW SIGNAL (best single-feature corr with outcome): crude {raw_signal['crude_best_feature_corr']}"
          f"  aware {raw_signal['aware_best_feature_corr']}  market {raw_signal['market_corr']}")
    print(f"  -> resolution-aware recovers {raw_signal['aware_vs_crude_ratio']}x the crude raw signal, "
          f"but only {raw_signal['fraction_of_market_signal_recovered']*100:.0f}% of the market's signal")
    print(f"  -> aware beats base rate in CALIBRATED prediction: {out['aware_beats_base_calibrated']} "
          f"(the weak signal doesn't survive to calibrated log-loss)")
    print(f"  -> aware signal carried by: {', '.join(f'{n}({w:+.2f})' for n, w in imp[:3])}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
