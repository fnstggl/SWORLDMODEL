"""Scheduled predict -> calibrate -> score loop, calling the real Claude API (audit J, K).

This is the productionized version of the manual EXP-002/003 rounds: it does what the
human-in-the-loop did by hand, on a cron. Wire it to a scheduler (Railway cron, GitHub
Actions, or the claude-code-remote send_later trigger) to accrue a live, contamination-free
scorecard that compounds n over time (the answer to the "n is small" bottleneck).

    python -m experiments.auto_loop --start 2026-05-20 --end 2026-05-27 --n 100

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile). Model: claude-opus-4-8.
Predictions are calibrated by the per-threshold Platt layer in data/calibration.json
(fit offline from past rounds; see fit_calibration()). Appends one row per run to
data/scorecard.jsonl so accuracy-over-time is trackable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.hn_harness2 import THRESHOLDS, fetch_window
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss

MODEL = "claude-opus-4-8"
CAL_PATH = "data/calibration.json"

_SCHEMA = {
    "type": "object",
    "properties": {"predictions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "p_ge_10": {"type": "number"}, "p_ge_40": {"type": "number"},
            "p_ge_100": {"type": "number"}, "p_ge_300": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["id", "p_ge_10", "p_ge_40", "p_ge_100", "p_ge_300"],
        "additionalProperties": False,
    }}},
    "required": ["predictions"], "additionalProperties": False,
}

_SYSTEM = (
    "You are the transition model of a social world model, predicting how the Hacker News "
    "community will respond to a submission. For each story output the probability its score "
    "reaches >=10, >=40, >=100, >=300 points (monotonic). Reason from: the author's past "
    "track record (a single past hit proves capability — weight max_past, not just median), "
    "topic resonance (HN loves deep-technical explainers, reverse-engineering, build-your-own, "
    "nuclear/energy debates, big-tech-layoff and platform-drama stories; is lukewarm on generic "
    "Show HN self-promo and mainstream media), and timing. Most random submissions never get "
    "traction: the base rate for >=10 is ~0.13. Be calibrated, not optimistic."
)


def _platt(p: float, ab: tuple[float, float]) -> float:
    a, b = ab
    z = math.log(min(1 - 1e-6, max(1e-6, p)) / (1 - min(1 - 1e-6, max(1e-6, p))))
    return 1 / (1 + math.exp(-(a * z + b)))


def predict_via_api(inputs: list[dict], batch: int = 12) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    out: dict[int, dict] = {}
    for i in range(0, len(inputs), batch):
        chunk = inputs[i:i + batch]
        payload = [{k: x.get(k) for k in ("id", "title", "author", "domain", "is_text_post",
                    "hour_utc", "weekday", "author_n_past", "author_median_past",
                    "author_max_past")} for x in chunk]
        resp = client.messages.create(
            model=MODEL, max_tokens=4000, system=_SYSTEM,
            messages=[{"role": "user", "content":
                       "Predict for these stories:\n" + json.dumps(payload, indent=1)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        for p in json.loads(text)["predictions"]:
            out[p["id"]] = p
        print(f"  predicted {min(i+batch, len(inputs))}/{len(inputs)}")
    return out


def fit_calibration(round_files: list[tuple[str, str]]) -> None:
    """Fit per-threshold Platt (a,b) on past (predictions, outcomes) rounds -> data/calibration.json.
    Only stores a correction where it improves held-out ECE; else identity (1,0)."""
    def logit(p): p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))
    def sig(z): return 1 / (1 + math.exp(-z))
    cal = {}
    for thr in THRESHOLDS:
        X, Y = [], []
        for pp, op in round_files:
            P = {p["id"]: p for p in json.loads(Path(pp).read_text())}
            O = {o["id"]: o for o in json.loads(Path(op).read_text())}
            for i in P:
                if i in O and f"p_ge_{thr}" in P[i]:
                    X.append(logit(P[i][f"p_ge_{thr}"])); Y.append(1 if O[i]["score"] >= thr else 0)
        a, b = 1.0, 0.0
        for _ in range(4000):
            ga = sum((sig(a * x + b) - y) * x for x, y in zip(X, Y)) / len(X)
            gb = sum(sig(a * x + b) - y for x, y in zip(X, Y)) / len(X)
            a -= 0.02 * ga; b -= 0.02 * gb
        raw_ece = expected_calibration_error(Y, [sig(x) for x in X])
        cal_ece = expected_calibration_error(Y, [sig(a * x + b) for x in X])
        cal[str(thr)] = [a, b] if cal_ece < raw_ece else [1.0, 0.0]
    Path(CAL_PATH).write_text(json.dumps(cal, indent=1))
    print("wrote", CAL_PATH, cal)


def run(start: str, end: str, n: int, seed: int = 0) -> None:
    prefix = f"data/auto_{start}"
    fetch_window(start, end, n, prefix, seed)
    inputs = json.loads(Path(f"{prefix}_inputs.json").read_text())
    outcomes = {o["id"]: o for o in json.loads(Path(f"{prefix}_outcomes.json").read_text())}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set; fetched data is ready but skipping live prediction.")
        return
    raw = predict_via_api(inputs)
    Path(f"{prefix}_predictions.json").write_text(json.dumps(list(raw.values()), indent=1))
    cal = json.loads(Path(CAL_PATH).read_text()) if Path(CAL_PATH).exists() else {}
    ids = [x["id"] for x in inputs if x["id"] in raw]
    row = {"window": f"{start}/{end}", "n": len(ids)}
    for thr in THRESHOLDS:
        y = [1 if outcomes[i]["score"] >= thr else 0 for i in ids]
        ab = tuple(cal.get(str(thr), [1.0, 0.0]))
        p = [_platt(raw[i][f"p_ge_{thr}"], ab) for i in ids]
        base = sum(y) / len(y)
        row[f"ge{thr}"] = {"logloss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
                           "ece": round(expected_calibration_error(y, p), 4),
                           "base_logloss": round(log_loss(y, [base] * len(y)), 4)}
    with open("data/scorecard.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("appended to data/scorecard.jsonl:", json.dumps(row, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.start, a.end, a.n, a.seed)


if __name__ == "__main__":
    main()
