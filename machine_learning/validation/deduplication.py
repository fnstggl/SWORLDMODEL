"""Exact + near-duplicate detection, within and across datasets.

* **exact**: identical canonical content (``content_hash``) — collapsed at normalization,
  re-checked here across datasets (the same record appearing in two datasets is a
  cross-source dup that would corrupt a cross-dataset eval).
* **near**: a normalized-text signature over the rendered SFT prompt+target (lowercased,
  whitespace-collapsed, punctuation-stripped, then a shingle MinHash band). Collisions are
  reported as candidate near-dups with a Jaccard estimate — a high near-dup rate signals a
  templated/degenerate converter.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..config import normalized_dir
from ..examples.formatters.sft import format_record
from ..normalization.common.parquet_io import iter_records

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm_text(s: str) -> str:
    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash_band(shingles: set[str], n: int = 4) -> str:
    if not shingles:
        return "empty"
    mins = sorted(hashlib.md5(s.encode()).hexdigest()[:8] for s in shingles)[:n]
    return "|".join(mins)


@dataclass
class DedupReport:
    dataset_id: str
    n_records: int = 0
    n_exact_dups: int = 0
    n_near_dup_candidates: int = 0
    exact_examples: list = field(default_factory=list)
    near_examples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "n_records": self.n_records,
                "n_exact_dups": self.n_exact_dups, "n_near_dup_candidates": self.n_near_dup_candidates,
                "exact_rate": round(self.n_exact_dups / max(self.n_records, 1), 4),
                "near_dup_rate": round(self.n_near_dup_candidates / max(self.n_records, 1), 4),
                "exact_examples": self.exact_examples[:20], "near_examples": self.near_examples[:20]}


def check_dataset(dataset_id: str, *, limit: int | None = None, near: bool = True) -> DedupReport:
    rep = DedupReport(dataset_id=dataset_id)
    seen_hash: dict[str, str] = {}
    bands: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for i, r in enumerate(iter_records(normalized_dir(dataset_id))):
        if limit and i >= limit:
            break
        rep.n_records += 1
        h = r.get("provenance", {}).get("content_hash")
        if h and h in seen_hash:
            rep.n_exact_dups += 1
            if len(rep.exact_examples) < 20:
                rep.exact_examples.append({"record_id": r["record_id"], "dup_of": seen_hash[h]})
        elif h:
            seen_hash[h] = r["record_id"]
        if near:
            fx = format_record(r)
            sh = _shingles(_norm_text(fx.text))
            band = _minhash_band(sh)
            for other_id, other_sh in bands[band]:
                jac = _jaccard(sh, other_sh)
                if jac >= 0.9:
                    rep.n_near_dup_candidates += 1
                    if len(rep.near_examples) < 20:
                        rep.near_examples.append({"record_id": r["record_id"], "near": other_id,
                                                 "jaccard": round(jac, 3)})
                    break
            else:
                if len(bands[band]) < 50:
                    bands[band].append((r["record_id"], sh))
    return rep


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def cross_dataset_exact(dataset_ids: list[str], *, limit_per: int | None = 20000) -> dict:
    """Detect the SAME canonical content appearing in more than one dataset."""
    owner: dict[str, str] = {}
    collisions: list[dict] = []
    for did in dataset_ids:
        for i, r in enumerate(iter_records(normalized_dir(did))):
            if limit_per and i >= limit_per:
                break
            h = r.get("provenance", {}).get("content_hash")
            if not h:
                continue
            if h in owner and owner[h] != did:
                collisions.append({"content_hash": h[:16], "datasets": [owner[h], did]})
            else:
                owner.setdefault(h, did)
    return {"n_collisions": len(collisions), "examples": collisions[:50]}
