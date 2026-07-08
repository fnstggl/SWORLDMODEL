"""SWM-Bench (Yu et al. 2026) — event-driven belief-trajectory data for the transition operator.

Each record is a belief transition: a history window of a prediction-market price (a proxy for
collective belief on a proposition), a set of candidate NEWS events timestamped before the target, a
posterior ATTRIBUTION (which event a hindsight LLM judged drove the shift), and the TARGET next belief
value. This is the P(s_{t+1} | s_t, event) regime — the temporal, event-conditioned belief DYNAMICS
that a general social world model needs and that our cross-sectional models never covered.

Source: HF ulab-ai/swm-bench (Qwen3.5-397B-attributed-data). Chronological split (train before
Nov 1 2025, test after); news drawn strictly before the target, so no leakage.
Download once:
  HF_TOKEN=... curl -L -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/datasets/ulab-ai/swm-bench/resolve/main/Qwen3.5-397B-attributed-data/<file> \
    -o data/swm_<file>
"""
from __future__ import annotations

import json
from pathlib import Path

FILES = {
    "train": "data/swm_train_with_nonzero_attribution.jsonl",
    "test_kalshi": "data/swm_test_kalshi.jsonl",
    "test_polymarket": "data/swm_test_polymarket.jsonl",
}


def load(split: str):
    path = Path(FILES[split])
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def belief_change(rec) -> float:
    """Δ = target belief − last observed belief (the quantity the world model predicts)."""
    if not rec.get("history") or not rec.get("target"):
        return 0.0
    return float(rec["target"]["p"]) - float(rec["history"][-1]["p"])


def attributed_events(rec, thresh: float = 0.5):
    """Candidate news the hindsight attributor judged causal (score ≥ thresh). Hindsight — TRAIN only."""
    out = []
    for a in rec.get("attributions", []):
        if a.get("score", 0.0) >= thresh:
            i = a.get("news_idx")
            if isinstance(i, int) and 0 <= i < len(rec.get("news", [])):
                out.append((i, a["score"], rec["news"][i]))
    return out


if __name__ == "__main__":
    for split in ("train", "test_kalshi", "test_polymarket"):
        recs = load(split)
        if not recs:
            print(f"{split}: (not downloaded)"); continue
        import statistics
        ch = [belief_change(r) for r in recs if r.get("history") and r.get("target")]
        nnews = statistics.mean(len(r.get("news", [])) for r in recs)
        print(f"{split}: {len(recs)} transitions | mean |Δ| {statistics.mean(abs(c) for c in ch):.4f} "
              f"| avg #news {nnews:.1f}")
