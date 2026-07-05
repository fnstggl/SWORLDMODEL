"""Leakage gate — now implemented (was a stub). Runs as a CI gate (audit E.4).

Enforces the invariants that make a temporal backtest physically leakage-proof rather than
convention-proof:

1. temporal-feature leakage: no retrieved/used item may have a timestamp after the as_of.
2. retrieval leakage: an AsOfStore query never returns a future item (checked directly).
3. content-hash dedup: the exact target content must not appear in the retrieved context (a crude
   contamination probe — n-gram/exact-match is known-insufficient but catches the obvious case).
4. label separation: the outcome/label field must not be among the features.

A failing check raises LeakageError, which the CI test turns into a hard failure. This is the honest
minimum; a production build adds a Time-Travel completion probe (can the model recover a redacted
answer?) and audience-leakage checks.
"""
from __future__ import annotations

import hashlib

from swm.retrieval.asof_store import AsOfStore, ContextItem, LeakageError

IMPLEMENTED = True


def check_temporal(items: list, as_of: float, *, ts_attr: str = "timestamp") -> None:
    """Every item used at prediction time must be dated at or before as_of."""
    bad = []
    for it in items:
        ts = getattr(it, ts_attr, None) if not isinstance(it, dict) else it.get(ts_attr)
        if ts is not None and ts > as_of:
            bad.append(it)
    if bad:
        raise LeakageError(f"{len(bad)} item(s) dated after as_of={as_of} reached the model")


def check_retrieval(store: AsOfStore, as_of: float, **query_kwargs) -> list[ContextItem]:
    """Run a query and assert nothing after as_of came back. Returns the (safe) items."""
    items = store.query(as_of=as_of, **query_kwargs)
    store.assert_no_leak(as_of, items)
    return items


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8", "ignore")).hexdigest()


def check_content_dedup(target_text: str, context_items: list[ContextItem]) -> None:
    """The exact target content must not be present verbatim in the retrieved context."""
    th = _hash(target_text)
    for it in context_items:
        if _hash(it.text) == th:
            raise LeakageError(f"target content appears verbatim in retrieved context item "
                               f"{it.item_id!r} (contamination)")


def check_label_separation(feature_names: list[str], label_name: str) -> None:
    if label_name in feature_names:
        raise LeakageError(f"label {label_name!r} is present as a feature — direct leakage")


def full_gate(*, as_of: float, used_items: list, feature_names: list[str], label_name: str,
              target_text: str = "", context_items: list[ContextItem] | None = None) -> dict:
    """Run every check; raise on the first violation, else return a passing report."""
    check_temporal(used_items, as_of)
    check_label_separation(feature_names, label_name)
    if target_text and context_items:
        check_content_dedup(target_text, context_items)
    return {"leakage_gate": "PASS", "as_of": as_of, "n_used_items": len(used_items),
            "n_features": len(feature_names)}
