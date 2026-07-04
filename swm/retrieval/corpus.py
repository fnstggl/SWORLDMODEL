"""Leakage-safe timestamped retrieval corpus (EXP-008, audit section 6).

The core discipline of a backtestable world model: retrieval may only surface information that
existed BEFORE the moment of the forecast. A document that a forecaster could not have read at
as_of must be unreachable — even if the caller hands it to the retriever by mistake.

This module makes that guarantee structural, not a convention:

  * every `Document` MUST carry a numeric timestamp (construction fails otherwise);
  * `as_of(t)` returns only documents strictly before t;
  * `reference_class(...)` and every public retrieval path route through `as_of`, then run a final
    `_assert_no_future` guard that raises `LeakageError` if any returned doc is >= t. Defense in
    depth: a future document cannot escape even if the similarity code is buggy.

A retrieval method is REJECTED (raises) the instant it would return a post-as_of item. There is no
"soft" mode. This is what lets EXP-008 claim that retrieved context cannot pull post-resolution
information.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class LeakageError(RuntimeError):
    """Raised when a retrieval would surface a document at or after the as_of horizon."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    timestamp: float          # when this document became knowable (unix seconds). REQUIRED.
    text: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None or not isinstance(self.timestamp, (int, float)):
            raise ValueError(f"Document {self.doc_id!r} needs a numeric timestamp (got "
                             f"{self.timestamp!r}); an untimestamped doc cannot be leakage-gated.")


def _tokens(s: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2}


def _assert_no_future(docs: list[Document], t: float) -> list[Document]:
    for d in docs:
        if d.timestamp >= t:
            raise LeakageError(f"document {d.doc_id!r} at ts={d.timestamp} is not < as_of={t}: "
                               "retrieval attempted to surface post-as_of information")
    return docs


class TimestampedCorpus:
    """An append-only set of timestamped documents with a hard as_of gate."""

    def __init__(self, docs: list[Document] | None = None):
        self._docs: list[Document] = []
        for d in docs or []:
            self.add(d)

    def add(self, doc: Document) -> None:
        if not isinstance(doc, Document):
            raise TypeError("corpus stores Document instances only")
        self._docs.append(doc)

    def __len__(self) -> int:
        return len(self._docs)

    def as_of(self, t: float) -> list[Document]:
        """THE gate. Only documents strictly before t are reachable. Final guard re-checks."""
        return _assert_no_future([d for d in self._docs if d.timestamp < t], t)

    def reference_class(self, query: str, t: float, k: int = 20,
                        min_sim: float = 0.0) -> list[Document]:
        """The k most similar documents by token overlap, drawn ONLY from before t. Every result is
        guaranteed < t (a resolved-before-as_of reference class); post-as_of docs are unreachable."""
        q = _tokens(query)
        visible = self.as_of(t)

        def sim(d: Document) -> float:
            dt = _tokens(d.text)
            return len(q & dt) / max(1, len(q | dt))

        ranked = sorted(((sim(d), d) for d in visible), key=lambda p: p[0], reverse=True)
        out = [d for s, d in ranked if s >= min_sim][:k]
        return _assert_no_future(out, t)
