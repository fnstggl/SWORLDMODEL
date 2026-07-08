"""Retrieval layer — fill each agent's variables from accessible knowledge (the missing input).

The generative loop (EXP-057) instantiates agents and has them reason, but its `context` was a stub. This
is the real input: for a question, gather the relevant evidence from the outside world so the LLM can
identify the deciding agents and map each one's known + inferred variables from it. Pluggable, like every
other LLM-touching part:

  - `web_search_retriever(search_fn)` — PRODUCTION: `search_fn(query) -> [{title, snippet, date?, source?}]`
    (a web-search backend). Used to pull live context for an arbitrary question.
  - `asof_retriever(news_by_question)` — REPRODUCIBLE / LEAKAGE-FREE eval: serve the committed AS-OF news
    for a question (dated strictly before its resolution), so a backtest cannot see the future.

`retrieve(question)` returns a `RetrievedContext` (a bounded, timestamped evidence bundle + a flat string
for prompts). Leakage discipline is the caller's: for a leakage-free skill number, use `asof_retriever`
(or a `search_fn` that filters to a date) so the evidence pre-dates the resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedContext:
    question: str
    snippets: list = field(default_factory=list)     # [{title, snippet, date, source}]
    as_of: str = ""                                   # the evidence cutoff (for leakage auditing)

    def to_prompt(self, max_items: int = 10) -> str:
        lines = []
        for s in self.snippets[:max_items]:
            d = f"({s.get('date', '')[:10]}) " if s.get("date") else ""
            t = (s.get("title") or "").strip()
            sn = (s.get("snippet") or s.get("description") or "").strip()
            lines.append(f"- {d}{t}" + (f" — {sn}" if sn else ""))
        return "\n".join(lines)

    def __len__(self):
        return len(self.snippets)


@dataclass
class Retriever:
    fetch_fn: object                  # callable(question) -> list[dict]  (the pluggable backend)
    max_items: int = 12

    def retrieve(self, question: str, as_of: str = "") -> RetrievedContext:
        items = self.fetch_fn(question) or []
        return RetrievedContext(question=question, snippets=list(items)[:self.max_items], as_of=as_of)


def web_search_retriever(search_fn, results: int = 12) -> Retriever:
    """Production retriever over a web-search backend. `search_fn(query) -> [{title, snippet, url, date}]`."""
    def fetch(question):
        try:
            return search_fn(question)[:results]
        except Exception:
            return []
    return Retriever(fetch_fn=fetch, max_items=results)


def asof_retriever(news_by_question: dict) -> Retriever:
    """Leakage-free eval retriever: serve committed AS-OF news for a question (already pre-resolution).
    `news_by_question`: {question -> [{title, description, published_at, source}]}."""
    def fetch(question):
        news = news_by_question.get(question, [])
        return [{"title": n.get("title"), "snippet": n.get("description"),
                 "date": n.get("published_at", ""), "source": n.get("source")} for n in news]
    return Retriever(fetch_fn=fetch)
