"""Posterior-guided event attribution — the SWM paper's core training recipe (Yu et al. 2026).

Nobody labels "which event caused this belief shift." The paper's insight: attribution is easy in
HINDSIGHT. A posterior attributor that sees the realized outcome can reliably say which candidate news
drove the shift (SWM-Bench ships these hindsight labels, from a large LLM). We then train a FORWARD
attributor — which must act before the outcome — to match those hindsight labels. Foresight learns from
hindsight (pseudo-labeling), so a cheap learned model recovers attribution without an LLM at inference.

`ForwardAttributor`: per-news causal scorer P_η(news is the driver | pre-shift features), trained by
logistic regression against the posterior labels. Pre-shift features only (salience to the question,
resolution-word cues, specificity) — no outcome leakage in the FEATURES; the LABELS are hindsight, which
is exactly the allowed supervision. Its transition-level "event strength" (how likely a strong causal
event is present) becomes a learned gate for the belief-transition operator.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from swm.transition.readout import LogisticReadout

_RESULT = re.compile(r"\b(win|wins|won|winner|loses|lost|defeat|victory|elected|concede|conceded|"
                     r"exit poll|results?|announced|confirms?|confirmed|resigns?|resigned|"
                     r"projected|declared|clinch|secures?|beats?|record|surges?|plunges?|crash)\b", re.I)
_STOP = set("the a an of to in on for and or is will be by at as with from this that not it its "
            "what when who how does do next".split())


def _tok(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def news_features(news: dict, qtok: set) -> list:
    """Pre-shift features of one candidate news item (no outcome information)."""
    title = news.get("title", "") or ""
    desc = news.get("description", "") or ""
    ntok = _tok(title + " " + desc)
    sal = len(qtok & ntok) / len(qtok) if qtok else 0.0
    txt = title + " " + desc
    return [sal,
            min(1.0, len(_RESULT.findall(txt)) / 2.0),          # resolution/result cues
            1.0 if re.search(r"\d", title) else 0.0,             # has a number (score, %, count)
            min(1.0, len(title) / 120.0)]                        # specificity proxy


FEAT_NAMES = ["salience", "result_cue", "has_number", "title_len"]


@dataclass
class ForwardAttributor:
    """P_η: scores each candidate news's probability of being the causal driver, trained on hindsight."""
    model: LogisticReadout = None                    # type: ignore
    thresh: float = 0.5                              # posterior score >= thresh => causal (positive label)

    def fit(self, records, epochs=300):
        X, y = [], []
        for r in records:
            qtok = _tok(r.get("question", "") + " " + r.get("description", ""))
            attr = {a["news_idx"]: a.get("score", 0.0) for a in r.get("attributions", [])}
            for i, n in enumerate(r.get("news", []) or []):
                X.append(news_features(n, qtok))
                y.append(1 if attr.get(i, 0.0) >= self.thresh else 0)
        if len(set(y)) < 2:
            return self
        self.model = LogisticReadout(epochs=epochs, l2=1.0).fit(X, y)
        return self

    def score_news(self, rec) -> list:
        """P(causal) for each candidate news in the transition."""
        if self.model is None:
            return [0.0] * len(rec.get("news", []) or [])
        qtok = _tok(rec.get("question", "") + " " + rec.get("description", ""))
        return [self.model.predict_proba(news_features(n, qtok)) for n in (rec.get("news", []) or [])]

    def event_strength(self, rec) -> float:
        """Transition-level P(a strong causal event is present) — the learned gate. Max over candidates."""
        s = self.score_news(rec)
        return max(s) if s else 0.0

    def attributed_salience(self, rec) -> float:
        """Attribution-weighted salience of the news set (attention-pooled by P_η)."""
        s = self.score_news(rec)
        if not s or sum(s) == 0:
            return 0.0
        qtok = _tok(rec.get("question", "") + " " + rec.get("description", ""))
        sal = [news_features(n, qtok)[0] for n in rec["news"]]
        return sum(w * v for w, v in zip(s, sal)) / sum(s)
