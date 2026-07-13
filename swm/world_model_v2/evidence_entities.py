"""Entity resolution — Phase 2.

Resolve entity mentions across sources into candidate entities, PRESERVING ambiguity. Incorrect merging is a
high-risk failure (it contaminates every downstream claim), so the resolver never forces a single match when
support is split — it keeps ranked candidates. Deterministic core (normalized-alias + contextual scoring);
the LLM may propose alias links but may not silently finalize an ambiguous identity.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict

_TITLES = ("mr", "mrs", "ms", "dr", "sen", "senator", "rep", "representative", "gov", "governor",
           "president", "ceo", "judge", "justice", "the")
_SUFFIX = ("inc", "inc.", "corp", "corp.", "co", "co.", "llc", "ltd", "plc", "company", "group",
           "association", "union", "party")


def normalize_mention(m: str) -> str:
    toks = [t for t in re.split(r"[\s.,]+", m.strip().lower()) if t and t not in _TITLES]
    while toks and toks[-1] in _SUFFIX:
        toks.pop()
    return " ".join(toks)


@dataclass
class EntityCandidate:
    entity_id: str
    canonical: str
    support: float
    evidence_mentions: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class EntityResolution:
    mention: str
    normalized: str
    candidates: list = field(default_factory=list)     # ranked [EntityCandidate]
    resolved: bool = False                             # True iff a single dominant candidate
    method: str = "normalized_alias"
    valid_time: str = ""

    @property
    def top(self):
        return self.candidates[0] if self.candidates else None

    def as_dict(self):
        d = asdict(self)
        d["candidates"] = [c.as_dict() for c in self.candidates]
        return d


class EntityResolver:
    """Cluster mentions into candidate entities by normalized form + alias links, keeping ambiguity when
    support is split. `dominance` is the margin the top candidate needs over the second to be `resolved`."""

    def __init__(self, *, dominance: float = 0.6, aliases: dict | None = None):
        self.dominance = dominance
        self.aliases = {normalize_mention(k): normalize_mention(v) for k, v in (aliases or {}).items()}

    def resolve(self, mentions: list) -> list:
        """Return one EntityResolution per DISTINCT mention string, with ranked candidates. Mentions sharing
        a normalized form (or an alias link) map to the same candidate entity id."""
        norm_counts: dict[str, int] = {}
        norm_examples: dict[str, list] = {}
        for m in mentions:
            n = self.aliases.get(normalize_mention(m), normalize_mention(m))
            if not n:
                continue
            norm_counts[n] = norm_counts.get(n, 0) + 1
            norm_examples.setdefault(n, []).append(m)
        total = sum(norm_counts.values()) or 1
        out, seen = [], {}
        for m in mentions:
            if m in seen:
                out.append(seen[m]); continue
            n = self.aliases.get(normalize_mention(m), normalize_mention(m))
            # candidates = the normalized cluster the mention falls in, plus any near-alias clusters
            cands = self._candidates_for(n, norm_counts, norm_examples, total)
            resolved = bool(cands) and (len(cands) == 1 or
                                        cands[0].support >= self.dominance * sum(c.support for c in cands))
            er = EntityResolution(mention=m, normalized=n, candidates=cands, resolved=resolved)
            seen[m] = er
            out.append(er)
        return out

    def _candidates_for(self, n, counts, examples, total) -> list:
        cands = []
        for cn, cnt in counts.items():
            score = cnt / total
            if cn == n:
                score += 0.5                                # exact normalized match dominates
            elif cn in n or n in cn:
                score += 0.15                               # substring (e.g. "biden" vs "joe biden")
            elif _token_overlap(cn, n) >= 0.5:
                score += 0.05
            else:
                continue
            cands.append(EntityCandidate(entity_id=_eid(cn), canonical=cn, support=round(score, 4),
                                         evidence_mentions=examples.get(cn, [])[:5]))
        cands.sort(key=lambda c: -c.support)
        return cands[:4]


def _token_overlap(a, b):
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb))


def _eid(canonical):
    return "ent_" + hashlib.sha1(canonical.encode()).hexdigest()[:10]
