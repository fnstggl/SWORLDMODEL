"""Stable pseudonymous identifiers.

Original participant identifiers are replaced with deterministic pseudonyms so identity is
preserved *for leakage isolation* without exposing raw identifiers. The mapping is
one-way (sha1) and stable across runs. Free-text content (messages, actions) is NEVER
pseudonymized — that content IS the behaviour we want to model; only *identifiers* are.
"""
from __future__ import annotations

import hashlib


def pseudonym(dataset_id: str, kind: str, raw_id) -> str:
    """Return a stable pseudonym '<dataset_id>-<kind>-<sha1_12>' for a raw identifier.

    ``raw_id`` may be any hashable/str-able value. None/empty returns a stable
    '<dataset_id>-<kind>-none' so absent ids are still distinguishable but not fabricated.
    """
    if raw_id is None or (isinstance(raw_id, str) and raw_id == ""):
        return f"{dataset_id}-{kind}-none"
    h = hashlib.sha1(f"{dataset_id}|{kind}|{raw_id}".encode("utf-8")).hexdigest()[:12]
    return f"{dataset_id}-{kind}-{h}"
