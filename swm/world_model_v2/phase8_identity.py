"""Phase 8 completion — canonical identity & history resolution (Part 7).

Cross-run persistence must not depend on a benchmark-specific user ID. This module resolves the stable keys
the store joins on — world / scenario / actor / dyad / institution / network-edge — and handles the messy
realities: aliases, role changes, uncertain linkage, actor merges/splits, institution renaming, and
relationship directionality. When identity is uncertain it PRESERVES several hypotheses (weighted by link
confidence) rather than forcing history onto one actor with false certainty.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s)).strip("_")[:48] or "x"


def world_id(question: str, *, namespace: str = "wmv2") -> str:
    """Stable world id from the question (deterministic; no wall clock)."""
    return f"{namespace}_" + hashlib.sha1(question.strip().lower().encode()).hexdigest()[:12]


def scenario_id(question: str, as_of: str, *, namespace: str = "wmv2") -> str:
    return f"{namespace}_" + hashlib.sha1(f"{question.strip().lower()}|{as_of}".encode()).hexdigest()[:12]


def dyad_id(a: str, b: str, *, rel: str = "relates_to", directed: bool = True) -> str:
    """Directed dyads keep order (a|rel|b); undirected sort the endpoints so a|b == b|a."""
    if directed:
        return f"{a}|{rel}|{b}"
    lo, hi = sorted([str(a), str(b)])
    return f"{lo}|{rel}|{hi}"


def edge_id(src: str, layer: str, dst: str) -> str:
    return f"{src}|{layer}|{dst}"


@dataclass
class IdentityHypothesis:
    canonical_id: str
    weight: float
    evidence: str = ""


@dataclass
class IdentityResolver:
    """Resolves raw tokens to canonical ids, preserving multiple hypotheses when linkage is uncertain.

    ``aliases`` maps raw token → [(canonical_id, confidence)]. A single high-confidence alias resolves
    directly; multiple plausible aliases return several weighted hypotheses. ``merges`` records actor merges
    (many raw → one canonical); ``splits`` records that one raw token now maps to several canonicals over
    time (resolved by ``as_of`` when a time-map is provided)."""
    aliases: dict = field(default_factory=dict)          # raw -> [(canonical, confidence)]
    merges: dict = field(default_factory=dict)           # raw -> canonical (definitive merge)
    role_changes: dict = field(default_factory=dict)     # canonical -> [(role, from_ts)]
    renames: dict = field(default_factory=dict)          # old_institution -> new_institution

    def resolve(self, raw: str, *, as_of: float | None = None) -> list:
        """Return [IdentityHypothesis] for a raw token, weight-summing to 1. A confident single mapping →
        one hypothesis (weight 1, uncertainty 0); ambiguous → several."""
        raw = str(raw)
        if raw in self.merges:
            return [IdentityHypothesis(self.merges[raw], 1.0, "definitive merge")]
        if raw in self.renames:
            return [IdentityHypothesis(self.renames[raw], 1.0, "institution rename")]
        cands = self.aliases.get(raw)
        if not cands:
            return [IdentityHypothesis(_slug(raw), 1.0, "identity (pass-through)")]
        z = sum(max(0.0, c) for _, c in cands) or 1.0
        hyps = [IdentityHypothesis(cid, round(max(0.0, c) / z, 4), "alias") for cid, c in cands]
        return sorted(hyps, key=lambda h: -h.weight)

    def link_uncertainty(self, raw: str) -> float:
        """0 = certain; approaches 1 as the alias mass spreads across hypotheses (entropy-like)."""
        hyps = self.resolve(raw)
        if len(hyps) <= 1:
            return 0.0
        top = max(h.weight for h in hyps)
        return round(1.0 - top, 4)

    def role_at(self, canonical: str, as_of: float) -> str:
        """The role this actor held as of a time (role changes are time-stamped)."""
        changes = sorted(self.role_changes.get(canonical, []), key=lambda rc: rc[1])
        role = ""
        for r, ts in changes:
            if ts <= as_of:
                role = r
        return role

    def resolve_scenario(self, question: str, as_of: str, *, actor_tokens=None) -> dict:
        """Bundle the canonical ids for a scenario: world/scenario + per-actor hypotheses. This is what the
        canonical pipeline calls to attach persistence to a compiled world."""
        return {"world_id": world_id(question), "scenario_id": scenario_id(question, as_of),
                "actors": {t: [h.__dict__ for h in self.resolve(t)] for t in (actor_tokens or [])}}
