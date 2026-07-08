"""Episodic memory + generative reflection — the retrieval substrate (audit C.4).

The single-individual regime was carrying a person as a *global average*: a `VariableMap` of stable
traits (who they are) + a transient `state` (how they are now). That throws away the most predictive
thing about a person for "will they respond to THIS?" — **how they reacted to similar situations before,
recently.** This module adds that recall layer, modeled on the Generative Agents (Park et al. 2023)
memory architecture, but held to *calibrated prediction*, not believability:

  1. **Memory stream** — an append-only, timestamped log of a person's episodes (things they did/received):
     each carries natural-language `text`, an `importance` (poignancy), a lazily-computed `embedding`, and
     optional structured `meta` (e.g. {"responded": True, "topic": "pricing"}).
  2. **Retrieval** — to score a new situation, pull the top-k episodes by the Generative-Agents weighting
     **recency × importance × relevance**: recency is an exponential time-decay toward `as_of`, importance
     is the stored poignancy, relevance is embedding cosine to the query. Each component is min-max
     normalized across the candidate set, then weighted-summed (the paper's α=β=γ=1 default is the default
     here, tunable).
  3. **Reflection** — periodically synthesize the raw episodes into higher-level abstractions ("responds to
     concise, respectful pricing messages; ignores repeated cold contact"), stored back as *reflection*
     episodes with their own embedding and boosted importance — so they participate in future retrieval and
     compound. Pluggable `reflect_fn` (LLM in production); a transparent behavioral-pattern fallback offline.

**Leakage guarantee (the non-negotiable):** retrieval is `timestamp < as_of` (strict — the episode being
predicted is never visible to itself), `as_of` is REQUIRED, and `assert_no_leak` is a post-condition the
tests exercise — the same contract as `swm/retrieval/asof_store.py`, specialized to per-person episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.memory.embeddings import TextEmbedder, tokenize

IMPLEMENTED = True


class LeakageError(ValueError):
    """Raised when a retrieval would let a person's future reach a prediction about their present."""


@dataclass
class Episode:
    """One timestamped memory of a person: something they did, received, or (for reflections) an insight."""
    entity_id: str
    timestamp: float                    # REQUIRED; same time axis used by the as_of gate
    text: str = ""
    importance: float = 0.5             # poignancy 0..1 (LLM-rated or heuristic); weights retrieval
    kind: str = "observation"           # "observation" | "reflection"
    meta: dict = field(default_factory=dict)     # structured payload, e.g. {"responded": bool, "topic": ...}
    embedding: list = field(default=None, repr=False)
    source_ids: tuple = ()              # for reflections: episode_ids of the episodes they summarize
    episode_id: str = ""

    def ensure_embedding(self, embedder: TextEmbedder) -> "Episode":
        if self.embedding is None:
            self.embedding = embedder(self.text)
        return self


def _minmax(vals: list[float]) -> list[float]:
    """Min-max normalize to [0,1]; a flat set maps to all-1 (component is uninformative, not zero)."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


# default retrieval weights (Generative Agents used α_recency = α_importance = α_relevance = 1)
DEFAULT_WEIGHTS = {"recency": 1.0, "importance": 1.0, "relevance": 1.0}


@dataclass
class MemoryStream:
    """One person's episodic memory + retrieval + reflection. `half_life` is in the timestamp's own units
    (e.g. days if timestamps are day indices) — the age at which the recency weight halves."""
    entity_id: str = ""
    embedder: TextEmbedder = field(default_factory=TextEmbedder)
    half_life: float = 30.0
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    episodes: list = field(default_factory=list)
    _importance_since_reflect: float = 0.0

    # ---- writes ------------------------------------------------------------------------------------
    def add(self, episode: Episode) -> Episode:
        if episode.timestamp is None:
            raise LeakageError(f"episode for {episode.entity_id!r} has no timestamp; refused "
                               "(un-gateable → could leak the future)")
        episode.entity_id = episode.entity_id or self.entity_id
        if not episode.episode_id:
            episode.episode_id = f"{episode.entity_id}:{len(self.episodes)}"
        episode.ensure_embedding(self.embedder)
        self.episodes.append(episode)
        if episode.kind == "observation":
            self._importance_since_reflect += episode.importance
        return episode

    def remember(self, text: str, *, ts: float, importance: float = 0.5, kind: str = "observation",
                 meta: dict = None) -> Episode:
        return self.add(Episode(entity_id=self.entity_id, timestamp=ts, text=text,
                                importance=max(0.0, min(1.0, importance)), kind=kind, meta=meta or {}))

    def record_contact(self, *, ts: float, text: str, responded: bool, topic: str = "",
                       importance: float = None, extra: dict = None) -> Episode:
        """Convenience for the individual regime: log a contact and its outcome as an episode. Importance
        defaults higher for the salient (rarer) event of actually responding."""
        meta = {"responded": bool(responded), "topic": topic, **(extra or {})}
        imp = importance if importance is not None else (0.75 if responded else 0.45)
        return self.remember(text, ts=ts, importance=imp, kind="observation", meta=meta)

    # ---- retrieval (the guarantee lives here) ------------------------------------------------------
    def _candidates(self, as_of: float, include_reflections: bool = True) -> list[Episode]:
        if as_of is None:
            raise LeakageError("retrieve requires an explicit as_of; without it a query could leak the "
                               "person's future into a prediction about their present")
        return [e for e in self.episodes
                if e.timestamp < as_of and (include_reflections or e.kind == "observation")]

    def _recency(self, ts: float, as_of: float) -> float:
        age = max(0.0, as_of - ts)
        return 0.5 ** (age / self.half_life) if self.half_life and self.half_life > 0 else 1.0

    def retrieve(self, query: str, *, as_of: float, k: int = 8, weights: dict = None,
                 include_reflections: bool = True) -> list[dict]:
        """Top-k episodes for `query` as of `as_of`, by recency × importance × relevance (each min-max
        normalized across candidates, then weighted-summed). Returns dicts with the episode + component
        scores so downstream code can reuse relevance/recency without recomputing."""
        cands = self._candidates(as_of, include_reflections)
        if not cands:
            return []
        w = weights or self.weights
        qv = self.embedder(query)
        rec = [self._recency(e.timestamp, as_of) for e in cands]
        imp = [e.importance for e in cands]
        rel = [max(0.0, self.embedder.similarity(qv, e.embedding)) for e in cands]
        nrec, nimp, nrel = _minmax(rec), _minmax(imp), _minmax(rel)
        scored = []
        for i, e in enumerate(cands):
            score = (w.get("recency", 1.0) * nrec[i] + w.get("importance", 1.0) * nimp[i]
                     + w.get("relevance", 1.0) * nrel[i])
            scored.append({"episode": e, "score": score, "recency": rec[i],
                           "importance": imp[i], "relevance": rel[i]})
        scored.sort(key=lambda r: -r["score"])
        top = scored[:k]
        self.assert_no_leak(as_of, [r["episode"] for r in top])
        return top

    def assert_no_leak(self, as_of: float, returned: list[Episode]) -> None:
        bad = [e.episode_id for e in returned if e.timestamp >= as_of]
        if bad:
            raise LeakageError(f"LEAK: episodes at/after as_of={as_of} returned: {bad}")

    # ---- reflection (mint new abstractions, feed them back) ----------------------------------------
    def personal_response_rate(self, as_of: float) -> float | None:
        """The person's own base response rate over their past contacts — the reference the situation-
        conditioned memory signal deviates from (isolates 'this kind of message' from 'this person')."""
        outs = [bool(e.meta.get("responded")) for e in self._candidates(as_of, include_reflections=False)
                if "responded" in e.meta]
        return (sum(1 for o in outs if o) / len(outs)) if outs else None

    def reflect(self, *, as_of: float, reflect_fn: object = None, max_source: int = 30,
                min_source: int = 4) -> list[Episode]:
        """Synthesize recent observations into higher-level reflection episodes and add them back so they
        participate in future retrieval. `reflect_fn(list[Episode]) -> list[{text, importance, meta?}]`
        (LLM in production); the fallback derives transparent behavioral patterns."""
        obs = self._candidates(as_of, include_reflections=False)
        if len(obs) < min_source:
            return []
        recent = sorted(obs, key=lambda e: e.timestamp)[-max_source:]
        if reflect_fn is not None:
            try:
                raw = reflect_fn(recent) or []
            except Exception:
                raw = []
        else:
            raw = _behavioral_reflections(recent)
        made = []
        src_ids = tuple(e.episode_id for e in recent)
        for r in raw:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            ep = Episode(entity_id=self.entity_id, timestamp=as_of, text=text,
                         importance=max(0.0, min(1.0, float(r.get("importance", 0.8)))),
                         kind="reflection", meta=dict(r.get("meta", {})), source_ids=src_ids)
            self.add(ep)
            made.append(ep)
        self._importance_since_reflect = 0.0
        return made

    def maybe_reflect(self, *, as_of: float, threshold: float = 6.0, reflect_fn: object = None) -> list[Episode]:
        """Reflect only once accumulated observation-importance since the last reflection crosses a
        threshold — the Generative-Agents trigger, so reflection is periodic, not every step."""
        if self._importance_since_reflect < threshold:
            return []
        return self.reflect(as_of=as_of, reflect_fn=reflect_fn)

    def __len__(self) -> int:
        return len(self.episodes)


def _top_topics(episodes: list[Episode], n: int = 3) -> list[str]:
    _STOP = {"the", "a", "an", "to", "of", "and", "or", "is", "are", "you", "your", "i", "we", "it",
             "this", "that", "for", "on", "in", "with", "about", "re", "hi", "hey", "hello", "thanks"}
    counts: dict[str, int] = {}
    for e in episodes:
        for t in set(tokenize(e.text)):
            if t not in _STOP and len(t) > 2:
                counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def _behavioral_reflections(episodes: list[Episode]) -> list[dict]:
    """Transparent fallback: derive a couple of behavioral abstractions from the recent contact record.
    Not meant to be clever — meant to be honest, embeddable, and to close the reflection→retrieval loop."""
    contacts = [e for e in episodes if "responded" in e.meta]
    out: list[dict] = []
    if len(contacts) >= 4:
        resp = [e for e in contacts if e.meta.get("responded")]
        rate = len(resp) / len(contacts)
        # per-topic response tendency: reflect where the person's topic behavior departs from their average
        by_topic: dict[str, list[bool]] = {}
        for e in contacts:
            tp = (e.meta.get("topic") or "").strip().lower()
            if tp:
                by_topic.setdefault(tp, []).append(bool(e.meta.get("responded")))
        for tp, outs in by_topic.items():
            if len(outs) >= 2:
                tr = sum(1 for o in outs if o) / len(outs)
                if tr - rate > 0.25:
                    out.append({"text": f"Tends to respond to {tp} messages.",
                                "importance": 0.85, "meta": {"topic": tp, "reflected_rate": round(tr, 2)}})
                elif rate - tr > 0.25:
                    out.append({"text": f"Tends to ignore {tp} messages.",
                                "importance": 0.8, "meta": {"topic": tp, "reflected_rate": round(tr, 2)}})
        if not out:
            engaged = _top_topics(resp, 3) if resp else _top_topics(contacts, 3)
            if engaged:
                out.append({"text": f"Engages most with topics: {', '.join(engaged)}.",
                            "importance": 0.75, "meta": {"topics": engaged}})
    return out


@dataclass
class EpisodicStore:
    """Multi-entity episodic memory — one `MemoryStream` per person, sharing an embedder + defaults. This
    is the object a single retrieval-augmented `response_fn` reads for many people (like `DeepPersonaStore`
    for personas), and the shared infra both regimes reuse."""
    embedder: TextEmbedder = field(default_factory=TextEmbedder)
    half_life: float = 30.0
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    _streams: dict = field(default_factory=dict)

    def stream(self, entity_id: str) -> MemoryStream:
        s = self._streams.get(entity_id)
        if s is None:
            s = MemoryStream(entity_id=entity_id, embedder=self.embedder, half_life=self.half_life,
                             weights=dict(self.weights))
            self._streams[entity_id] = s
        return s

    def remember(self, entity_id: str, text: str, *, ts: float, importance: float = 0.5,
                 kind: str = "observation", meta: dict = None) -> Episode:
        return self.stream(entity_id).remember(text, ts=ts, importance=importance, kind=kind, meta=meta)

    def record_contact(self, entity_id: str, *, ts: float, text: str, responded: bool, topic: str = "",
                       importance: float = None, extra: dict = None) -> Episode:
        return self.stream(entity_id).record_contact(ts=ts, text=text, responded=responded, topic=topic,
                                                     importance=importance, extra=extra)

    def retrieve(self, entity_id: str, query: str, *, as_of: float, k: int = 8,
                 weights: dict = None, include_reflections: bool = True) -> list[dict]:
        if entity_id not in self._streams:
            return []
        return self._streams[entity_id].retrieve(query, as_of=as_of, k=k, weights=weights,
                                                 include_reflections=include_reflections)

    def reflect(self, entity_id: str, *, as_of: float, reflect_fn: object = None) -> list:
        return self.stream(entity_id).reflect(as_of=as_of, reflect_fn=reflect_fn)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._streams
