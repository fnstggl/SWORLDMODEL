"""Evidence-leakage auditor — Phase 2. Produces a per-question leakage report over an EvidenceBundle.

Checks (each produces item-level flags + a bundle-level summary):
  1. resolution-term scanning        — outcome words/phrases appearing in evidence text;
  2. future-date detection           — explicit dates in the text strictly after as_of;
  3. retrospective-language detection— phrasing that only exists after an outcome is known;
  4. duplicate/syndication collapse  — near-duplicate items (shingle Jaccard) counted once;
  5. timestamp-basis audit           — share of items whose timestamps are only claimed (RSS) vs
     verified server-side (wiki revisions, snapshots); nonzero as-of slack is flagged;
  6. snapshot coverage               — items with no immutable snapshot reference.

The auditor NEVER edits the bundle; it recommends quarantine. Sensitivity policy: any hard flag
(future date, resolution term) → recommend exclude + rerun; soft flags lower the evidence-quality grade.
Google `before:`/RSS dates are treated as discovery hints, not proof — that is exactly what check 5 grades.
"""
from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass, field

from swm.world_model_v2.state import rfc3339

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_PATTERNS = (
    re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", re.I),
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS}),?\s+(\d{{4}})", re.I),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
)

RETROSPECTIVE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\bwould (?:go on to|later|eventually|ultimately)\b",
    r"\bin (?:the aftermath|hindsight|retrospect)\b",
    r"\blooking back\b",
    r"\bwhat we now know\b",
    r"\b(?:it|this) (?:later|eventually|ultimately) (?:became|proved|turned out)\b",
    r"\bat the time,?\s+(?:few|no one|nobody)\b",
    r"\banniversary of\b",
    r"\bhas since\b",
    r"\bturned out to be\b",
))


@dataclass
class LeakageReport:
    question_id: str
    as_of: str
    item_flags: dict = field(default_factory=dict)     # item_id -> [flags]
    duplicates: list = field(default_factory=list)     # [[item_id, item_id], ...]
    hard_leaks: list = field(default_factory=list)     # item ids recommended for exclusion
    summary: dict = field(default_factory=dict)

    def clean(self) -> bool:
        return not self.hard_leaks

    def as_dict(self):
        return {"question_id": self.question_id, "as_of": self.as_of,
                "item_flags": self.item_flags, "duplicates": self.duplicates,
                "hard_leaks": self.hard_leaks, "summary": self.summary}


def _extract_dates(text: str):
    import calendar
    months = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
    out = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            g = m.groups()
            try:
                if len(g) == 3 and g[0].lower() in months:                 # Month D, Y
                    y, mo, d = int(g[2]), months[g[0].lower()], int(g[1])
                elif len(g) == 3 and g[1] and g[1].lower() in months:      # D Month Y
                    y, mo, d = int(g[2]), months[g[1].lower()], int(g[0])
                else:                                                      # ISO
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                    out.append(calendar.timegm((y, mo, min(d, 28), 12, 0, 0, 0, 0, 0)))
            except (ValueError, KeyError):
                continue
    return out


def _shingles(text: str, k: int = 5) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}


def audit_bundle(bundle, *, resolution_terms=(), now: float | None = None) -> LeakageReport:
    """Run all checks over an EvidenceBundle. `resolution_terms`: phrases whose presence in evidence
    text means the outcome leaked (e.g. the resolved option text, 'declared the winner'). Term scanning
    is literal + case-insensitive; callers supply outcome-specific phrasings."""
    rep = LeakageReport(question_id=bundle.question_id, as_of=rfc3339(bundle.as_of))
    terms = [t.lower() for t in resolution_terms if len(t) >= 4]
    sh = {}
    n_claimed_only, n_verified, n_no_snapshot = 0, 0, 0
    for it in bundle.items:
        flags = list(it.leakage_flags)
        text = f"{it.title} {it.text}"
        low = text.lower()
        for t in terms:
            if t in low:
                flags.append(f"resolution_term:{t[:40]}")
        future = [d for d in _extract_dates(text) if d > bundle.as_of + 86400.0]
        if future:
            flags.append(f"future_date_in_text:{rfc3339(max(future))}")
        for pat in RETROSPECTIVE_PATTERNS:
            if pat.search(text):
                flags.append(f"retrospective_language:{pat.pattern[:30]}")
                break
        if it.published_at is not None and not it.published_verified:
            n_claimed_only += 1
            flags.append("timestamp_claimed_not_verified")
        elif it.published_verified:
            n_verified += 1
        if not it.snapshot_ref:
            n_no_snapshot += 1
        sh[it.item_id] = _shingles(text)
        if flags:
            rep.item_flags[it.item_id] = flags
        if any(f.startswith(("resolution_term", "future_date_in_text")) for f in flags):
            rep.hard_leaks.append(it.item_id)
    # duplicate/syndication collapse (soft flag; duplicates overweight one voice)
    ids = list(sh)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = sh[ids[i]], sh[ids[j]]
            if a and b:
                jac = len(a & b) / max(1, len(a | b))
                if jac > 0.6:
                    rep.duplicates.append([ids[i], ids[j]])
                    rep.item_flags.setdefault(ids[j], []).append(f"near_duplicate_of:{ids[i]}")
    n = max(1, len(bundle.items))
    rep.summary = {
        "n_items": len(bundle.items), "n_quarantined_at_gate": len(bundle.quarantine),
        "hard_leaks": len(rep.hard_leaks), "n_duplicate_pairs": len(rep.duplicates),
        "share_verified_timestamps": round(n_verified / n, 3),
        "share_claimed_only_timestamps": round(n_claimed_only / n, 3),
        "share_missing_snapshot": round(n_no_snapshot / n, 3),
        "nonzero_slack": bundle.slack_s > 0,
        "evidence_quality_grade": _grade(n_verified / n, len(rep.hard_leaks), bundle.slack_s),
        "audited_at": rfc3339(now if now is not None else _time.time()),
        "recommendation": ("EXCLUDE hard-leak items and re-audit" if rep.hard_leaks else
                           ("usable — verify-timestamp share low, prefer snapshot sources"
                            if n_verified / n < 0.5 else "usable")),
    }
    return rep


def _grade(verified_share: float, hard_leaks: int, slack: float) -> str:
    if hard_leaks:
        return "F (leaked)"
    if slack > 0:
        return "C (nonzero as-of slack)"
    if verified_share >= 0.7:
        return "A (mostly server-verified timestamps)"
    if verified_share >= 0.3:
        return "B (mixed timestamp basis)"
    return "C (claimed timestamps only)"
