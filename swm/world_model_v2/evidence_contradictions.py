"""Contradiction & correction graph — Phase 2.

Claim-level contradictions are built and PRESERVED (not resolved by one LLM preference). Downstream inference
receives the graph and represents disagreement as uncertainty. Typed edges: mutual exclusion (opposing
polarity on the same subject-predicate), numerical disagreement (different values on the same measure),
denial-vs-allegation, and correction/retraction (later claim supersedes earlier, kept with temporal order).
Deterministic over claim structure; an optional LLM pass can add semantic contradictions it validates.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict

CONTRADICTION_TYPES = ("mutual_exclusion", "numerical_disagreement", "denial_vs_allegation",
                       "correction", "retraction", "identity_ambiguity")


@dataclass
class ContradictionEdge:
    edge_id: str
    claim_a: str
    claim_b: str
    ctype: str
    confidence: float = 0.6
    note: str = ""
    temporal_order: str = ""                           # "a_before_b" | "b_before_a" | "unknown"

    def as_dict(self):
        return asdict(self)


def _key(c):
    return (_norm(c.subject), _norm(c.predicate))


def _norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _num(c):
    m = re.search(r"-?\d+(?:\.\d+)?", f"{c.value} {c.object}")
    return float(m.group(0)) if m else None


def _order(a, b):
    if a.publication_time and b.publication_time:
        return "a_before_b" if a.publication_time <= b.publication_time else "b_before_a"
    return "unknown"


def build_contradiction_graph(claims: list) -> list:
    """Return [ContradictionEdge] over the claims. Only compares claims on the SAME subject-predicate; keeps
    every materially plausible claim (edges annotate disagreement, they do not delete claims)."""
    edges, by_key = [], {}
    for c in claims:
        by_key.setdefault(_key(c), []).append(c)
    for _, group in by_key.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                edge = _edge_between(a, b)
                if edge:
                    edges.append(edge)
                    a.contradiction_links.append(b.claim_id)
                    b.contradiction_links.append(a.claim_id)
    return edges


def _edge_between(a, b):
    et, conf, note = None, 0.6, ""
    classes = {a.claim_class, b.claim_class}
    if "retraction" in classes:
        et, conf, note = "retraction", 0.8, "one claim retracts the other"
    elif "correction" in classes:
        et, conf, note = "correction", 0.75, "one claim corrects the other"
    elif classes == {"denial", "allegation"} or (
            {"denial", "allegation"} <= classes):
        et, conf, note = "denial_vs_allegation", 0.7, "denial opposes allegation"
    elif a.polarity != b.polarity:
        et, conf, note = "mutual_exclusion", 0.7, "opposing polarity on the same subject-predicate"
    else:
        na, nb = _num(a), _num(b)
        if na is not None and nb is not None and abs(na - nb) > 1e-9 and max(abs(na), abs(nb)) > 0 \
                and abs(na - nb) / max(abs(na), abs(nb)) > 0.05:
            et, conf, note = "numerical_disagreement", 0.65, f"{na} vs {nb}"
    if et is None:
        return None
    return ContradictionEdge(
        edge_id="ctr_" + hashlib.sha1(f"{a.claim_id}|{b.claim_id}|{et}".encode()).hexdigest()[:10],
        claim_a=a.claim_id, claim_b=b.claim_id, ctype=et, confidence=conf, note=note,
        temporal_order=_order(a, b))


def has_material_contradiction(edges: list) -> bool:
    return any(e.ctype in ("mutual_exclusion", "numerical_disagreement", "denial_vs_allegation")
               for e in edges)
