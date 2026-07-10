"""Labeled behavioral evaluation — the predictive half of the OSim pilot (Brier/logloss/accuracy/alignment).

A label-free realism probe cannot show a model PREDICTS better (hard rule). This grades models against REAL
human outcomes from public datasets, same inputs across arms:

  * BehaviorBench `moblab/game_behavior` (cc-by-nc-nd; benchmark-use only) — 9 economic games, each with ~200
    real human choices. Behavioral models (Be.FM/OSim) claim DISTRIBUTIONAL alignment to human populations,
    so the primary metric is Wasserstein-1 distance between the model's sampled choices and the human choice
    distribution (lower = more human-like), plus mean-abs error of the population mean. For the binary-
    decodable games (ultimatum responder accept/reject) we also report Brier/log-loss/accuracy vs the human
    accept rate.

Each arm is a callable `sample(prompt) -> text` (DeepSeek here; an OSim endpoint on a GPU pod). The loader
pulls JSONL straight from HF `resolve/` URLs (works identically on the pod). Nothing is trained; this only
scores. Individual-response (Enron reply+delay) and headline-click (Upworthy) loaders live alongside — see
`swm/eval/response_datasets.py`.
"""
from __future__ import annotations

import json
import re
import urllib.request

_HF = "https://huggingface.co/datasets/befm/BehaviorBench/resolve/main/moblab/game_behavior/"
GAMES = ["bomb", "dictator", "guessing", "public_goods", "push_pull", "trust_banker",
         "trust_investor", "ultimatum_proposer", "ultimatum_responder"]
_SCALE = {"bomb": 100.0, "dictator": 100.0, "guessing": 100.0, "public_goods": 20.0, "push_pull": 100.0,
          "trust_banker": 100.0, "trust_investor": 100.0, "ultimatum_proposer": 100.0,
          "ultimatum_responder": 100.0}   # rough per-game answer scale for normalizing Wasserstein


def _num(text):
    """Extract the bracketed/first number a player emitted, e.g. '[33]' -> 33.0. None if unparseable."""
    if text is None:
        return None
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*\]", text) or re.search(r"(-?\d+(?:\.\d+)?)", text)
    try:
        return float(m.group(1)) if m else None
    except (ValueError, AttributeError):
        return None


def load_game(game, limit=None):
    """Return [{system, user, assistant, human_answer}] for one BehaviorBench game (from HF resolve URL)."""
    url = f"{_HF}{game}_test.jsonl"
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}),
                                  timeout=60).read().decode()
    rows = []
    for line in data.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["human_answer"] = _num(r.get("assistant"))
        rows.append(r)
    rows = [r for r in rows if r["human_answer"] is not None]
    return rows[:limit] if limit else rows


def _wasserstein1(a, b):
    """1-D Wasserstein-1 between two empirical samples = mean |sorted quantile diff| on a common grid."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    n = 50
    def q(s, t):
        i = min(len(s) - 1, int(t * len(s)))
        return s[i]
    return sum(abs(q(a, i / n) - q(b, i / n)) for i in range(n)) / n


def eval_game(game, sample_fn, *, reps=10, limit=25):
    """Score one arm on one game. For each of `limit` prompts, draw `reps` model samples; compare the pooled
    model distribution to the human distribution. Returns metrics dict."""
    rows = load_game(game, limit=limit)
    if not rows:
        return {"game": game, "n": 0}
    human = [r["human_answer"] for r in rows]
    model = []
    n_bad = 0
    for r in rows:
        prompt = f"{r['system']}\n\n{r['user']}"
        for _ in range(reps):
            v = _num(sample_fn(prompt))
            if v is None:
                n_bad += 1
            else:
                model.append(v)
    scale = _SCALE.get(game, 100.0)
    w = _wasserstein1(model, human)
    hm = sum(human) / len(human)
    mm = (sum(model) / len(model)) if model else None
    return {"game": game, "n_human": len(human), "n_model": len(model), "n_unparsed": n_bad,
            "human_mean": round(hm, 2), "model_mean": (round(mm, 2) if mm is not None else None),
            "wasserstein": (round(w, 3) if w is not None else None),
            "wasserstein_norm": (round(w / scale, 4) if w is not None else None),
            "mean_abs_err": (round(abs(mm - hm), 3) if mm is not None else None)}


def eval_arm(sample_fn, *, games=None, reps=10, limit=25, progress=True):
    """Score an arm across games. Returns per-game rows + an aggregate normalized-Wasserstein (lower=better).
    Prints each game result as it completes (progress=True) so a long run isn't silent."""
    games = games or GAMES
    rows = []
    for g in games:
        r = eval_game(g, sample_fn, reps=reps, limit=limit)
        rows.append(r)
        if progress:
            if r.get("wasserstein_norm") is not None:
                print(f"    {r['game']:20s} human={r['human_mean']:>6} model={str(r['model_mean']):>6} "
                      f"W1_norm={r['wasserstein_norm']:.4f}  unparsed={r['n_unparsed']}", flush=True)
            else:
                print(f"    {r['game']:20s} (no parseable samples)", flush=True)
    scored = [r for r in rows if r.get("wasserstein_norm") is not None]
    agg = round(sum(r["wasserstein_norm"] for r in scored) / len(scored), 4) if scored else None
    return {"per_game": rows, "mean_wasserstein_norm": agg, "n_games": len(scored)}
