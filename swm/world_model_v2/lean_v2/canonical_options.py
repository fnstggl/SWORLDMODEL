"""Canonical action / vote option identity — one typed option system, universal across every
institution and decision (D1).

The BoJ failure: a valid cast `"vote:Raise to 1.0%"` was dropped because the terminal matcher
compared raw provider strings and never stripped the `vote:` menu prefix. The general fix is a
typed canonical option with a safe normalizer used *everywhere* a provider option string meets a
mechanical comparison (vote recording, terminal evaluation, obligation menus).

Safety contract:
  * a leading `vote:` / `Vote:` / `vote ` menu prefix, surrounding punctuation, casing and
    whitespace are normalized away before matching;
  * an exact (normalized) or registered-alias match wins;
  * a containment match is accepted ONLY when exactly one option matches — if two options could
    both match, normalization FAILS (returns None) rather than silently guessing;
  * an unknown option returns None so the caller can fail validation + trigger one targeted
    repair. It is NEVER silently mapped to a different option.

No question-specific option lists live here; the option set is always built from the blueprint's
own `record_vote` params or action ids at runtime."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm_key

#: menu-prefix tokens the provider may echo back in front of the actual option
_PREFIXES = ("vote:", "vote ", "vote for:", "vote for ", "cast_vote:", "cast vote:",
             "option:", "choice:", "select:")


def strip_menu_prefix(raw: str) -> str:
    """Remove a leading `vote:` / `cast_vote:` style menu prefix (case-insensitive)."""
    s = str(raw or "").strip()
    low = s.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            return s[len(p):].strip()
    return s


def _canon(raw: str) -> str:
    """Aggressive normalization for comparison: strip menu prefix, lowercase, collapse to
    alphanumerics + single spaces. Keeps digits and % semantics (via norm_key which retains
    alphanumerics)."""
    return norm_key(strip_menu_prefix(raw))


@dataclass
class CanonicalOption:
    canonical_option_id: str
    display_label: str
    aliases: set = field(default_factory=set)          # normalized alias strings
    institution_id: str = ""
    terminal_semantics: str = ""                        # free tag, e.g. "yes_target" / "no"

    def matches(self, normalized: str) -> bool:
        return normalized == norm_key(self.canonical_option_id) \
            or normalized == _canon(self.display_label) \
            or normalized in self.aliases

    def as_dict(self) -> dict:
        return {"canonical_option_id": self.canonical_option_id,
                "display_label": self.display_label, "aliases": sorted(self.aliases),
                "institution_id": self.institution_id,
                "terminal_semantics": self.terminal_semantics}


def _make_id(label: str) -> str:
    """Deterministic snake_case canonical id from a display label."""
    s = re.sub(r"[^a-z0-9]+", "_", strip_menu_prefix(label).lower()).strip("_")
    return s or "option"


def build_option_set(option_labels, *, institution_id: str = "",
                     extra_aliases: dict = None) -> list:
    """Build `CanonicalOption`s from the blueprint's own option strings. `extra_aliases` maps a
    display_label to additional alias strings explicitly registered in the action schema."""
    extra_aliases = extra_aliases or {}
    out, seen = [], set()
    for label in option_labels or []:
        label = str(label)
        cid = _make_id(label)
        base = cid
        i = 1
        while cid in seen:                              # keep ids unique
            i += 1
            cid = f"{base}_{i}"
        seen.add(cid)
        aliases = {_canon(label), norm_key(cid)}
        for a in extra_aliases.get(label, []):
            aliases.add(_canon(a))
        aliases.discard("")
        out.append(CanonicalOption(canonical_option_id=cid, display_label=label,
                                   aliases=aliases, institution_id=institution_id))
    return out


def normalize_option(raw, options) -> "CanonicalOption | None":
    """Return the `CanonicalOption` the raw provider string denotes, or None if it cannot be
    safely resolved. `options` is a list[CanonicalOption] or a list[str] (built on the fly).

    Resolution order: exact/alias match → unique containment → None (never a guess)."""
    if not options:
        return None
    if options and isinstance(options[0], str):
        options = build_option_set(options)
    n = _canon(raw)
    if not n:
        return None
    # 1) exact or alias
    exact = [o for o in options if o.matches(n)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None                                     # genuinely ambiguous — never guess
    # 2) unique containment (the cast paraphrases an option, e.g. "raise the rate to 1.0 percent"
    #    contains "raise to 1 0"); accept ONLY when exactly one option is involved
    def toks(s):
        return set(s.split())
    nt = toks(n)
    contain = []
    for o in options:
        ot = toks(_canon(o.display_label))
        if not ot:
            continue
        if ot <= nt or nt <= ot or (ot & nt and len(ot & nt) >= max(1, len(ot) - 1)):
            contain.append(o)
    if len(contain) == 1:
        return contain[0]
    return None                                         # ambiguous or unknown → caller repairs


def normalize_to_label(raw, option_labels) -> "str | None":
    """Convenience: return the matching original option *string* (display_label) or None."""
    opt = normalize_option(raw, list(option_labels or []))
    return opt.display_label if opt is not None else None
