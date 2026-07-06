"""Resolution-aware content extractor — read the real news for THIS question's outcome (the EXP-043 frontier).

EXP-043's negative: crude, question-agnostic news features (total volume, global positive/negative term
counts) don't beat the base rate, even though the market extracts the same articles into a decisive
signal. The gap is CONTENT: the market knows *what the question is asking* and reads the news for stance
toward *that specific outcome*. This does the same, in pure Python (no embeddings):

1. PARSE the question into a resolution frame: its SUBJECT (the entity/proper nouns + salient content
   words), a numeric THRESHOLD and comparison direction if present ("above 4.2%", ">25bps"), and the
   template ("Will X win/release/say ..." vs a scalar "... score?").
2. LINK each news item to the question by SUBJECT overlap — only news actually about the subject counts
   (the entity-linking step the crude extractor skipped).
3. Read STANCE toward the YES outcome in the linked news: positive vs negative outcome terms taken *near
   the subject*, oriented by the question's direction, plus resolution ("it's decided") cues — recency
   weighted so fresh news dominates.

The output is a small set of grounded, signed features about *this* question's resolution, which EXP-044
tests against the crude EXP-043 features and the market lean.
"""
from __future__ import annotations

import datetime
import re

_STOP = {"will", "the", "a", "an", "be", "is", "are", "was", "of", "to", "in", "on", "at", "for", "and",
         "or", "by", "his", "her", "their", "this", "that", "than", "then", "with", "as", "it", "he",
         "she", "they", "you", "do", "does", "did", "before", "after", "during", "have", "has", "had",
         "who", "what", "when", "which", "on the", "over", "under", "up", "down", "out", "no", "not",
         "most", "more", "less", "at least", "any", "all", "if", "into", "from"}
_POS = re.compile(r"\b(win|wins|won|winning|lead|leads|leading|ahead|approv\w*|pass(?:es|ed)?|surg\w*|"
                  r"ris\w*|gain\w*|beat\w*|clinch\w*|secur\w*|advanc\w*|on track|likely|confirm\w*|"
                  r"announc\w*|releas\w*|set to|expected to|success\w*|top\w*|record|soar\w*|jump\w*)\b", re.I)
_NEG = re.compile(r"\b(lose|loses|lost|losing|trail\w*|behind|reject\w*|fail\w*|drop\w*|fall\w*|"
                  r"declin\w*|miss\w*|eliminat\w*|out of|unlikely|defeat\w*|delay\w*|cancel\w*|"
                  r"deny|denied|slump\w*|plunge\w*|below|weak\w*)\b", re.I)
_RESOLVE = re.compile(r"\b(official\w*|final|finaliz\w*|result\w*|announc\w*|confirm\w*|declar\w*|"
                      r"winner|decided|outcome|record\w*|report\w*)\b", re.I)
_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_ABOVE = re.compile(r"\b(above|over|exceed\w*|more than|greater|at least|>|higher)\b", re.I)
_BELOW = re.compile(r"\b(below|under|less than|fewer|<|lower|at most)\b", re.I)


def parse_question(q: str) -> dict:
    q = q or ""
    ql = q.lower()
    # subject terms: proper-noun tokens + salient content words (dedup, drop stopwords/short)
    proper = re.findall(r"\b([A-Z][a-zA-Z0-9']+(?:\s+[A-Z][a-zA-Z0-9']+)*)\b", q)
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", ql) if w not in _STOP]
    subject = []
    for tok in [p.lower() for p in proper] + words:
        for piece in tok.split():
            if piece not in _STOP and len(piece) >= 3 and piece not in subject:
                subject.append(piece)
    # numeric threshold + direction
    thr = None
    m = _NUM.search(q)
    if m:
        try:
            thr = float(m.group(1))
        except ValueError:
            thr = None
    direction = 1 if _ABOVE.search(ql) else (-1 if _BELOW.search(ql) else 0)
    return {"subject": subject[:12], "threshold": thr, "direction": direction,
            "scalar": "?" in q and "will" not in ql}


def _ts(pub):
    if not pub:
        return None
    try:
        return datetime.datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def extract(question: str, news: list, t_target: float) -> dict:
    """Grounded, resolution-aware features from news that is actually ABOUT this question's subject."""
    frame = parse_question(question)
    subj = set(frame["subject"])
    linked, stance_num, stance_den, resolve_hits, recw_stance, recw = 0, 0.0, 0.0, 0, 0.0, 0.0
    for nw in news or []:
        text = ((nw.get("title") or "") + " " + (nw.get("description") or ""))
        tl = text.lower()
        toks = set(re.findall(r"[a-z0-9']+", tl))
        overlap = len(subj & toks)
        if not subj or overlap == 0:
            continue                                    # not about this subject -> ignore (entity linking)
        spec = overlap / len(subj)                      # how specifically this item is about the subject
        linked += 1
        pos, neg = len(_POS.findall(text)), len(_NEG.findall(text))
        s = (pos - neg)
        stance_num += spec * s
        stance_den += spec * (pos + neg)
        resolve_hits += int(bool(_RESOLVE.search(text)))
        ts = _ts(nw.get("published_at"))
        rw = 1.0 if ts is None else max(0.05, min(1.0, 1.0 - (t_target - ts) / (21 * 86400.0)))
        recw += rw * spec
        recw_stance += rw * spec * s
    n = len(news or [])
    link_rate = linked / n if n else 0.0
    stance = (stance_num / stance_den) if stance_den > 1e-9 else 0.0        # signed [-1,1] toward YES
    recent_stance = (recw_stance / recw) if recw > 1e-9 else 0.0
    resolve_rate = (resolve_hits / linked) if linked else 0.0
    import math
    return {"subject_link_rate": link_rate, "subject_stance": stance, "recent_subject_stance": recent_stance,
            "subject_resolve_rate": resolve_rate, "n_linked": linked, "log_linked": math.log1p(linked)}


# the exposed predictors are the ones that actually carry stance signal (EXP-044): the recency-weighted
# stance toward the subject dominates. `subject_link_rate` is near-constant (~0.92) and is deliberately
# NOT a predictor — it corrupts the readout as a constant the model over-weights; it only gates internally.
FEATURE_NAMES = ["subject_stance", "recent_subject_stance", "subject_resolve_rate", "log_linked"]


def feature_vector(question: str, news: list, t_target: float) -> list:
    f = extract(question, news, t_target)
    return [f[k] for k in FEATURE_NAMES]
