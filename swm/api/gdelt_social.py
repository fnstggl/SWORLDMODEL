"""GDELT social-state grounding for the forecaster — the vision-aligned lever for the geopolitical/social slice.

For a metric question we ground the current *number* (as-of price). For a SOCIAL question — a war, ceasefire,
coup, protest wave, sanction, election-crisis — the analogous grounding is the current *state of that part of
the social world*: how conflictual/cooperative the country's event stream is right now, how much violence and
protest it is generating, and which way those are trending. GDELT's bulk CAMEO event stream measures exactly
this, leakage-free, as of the question's date (`swm.retrieval.gdelt_index.asof_social_state`).

This grounder turns that measured state into two things the latent simulation consumes:
  1. a STRUCTURED CONTEXT BLOCK injected into the compile prompt (like as-of news, but quantified and
     trend-bearing) — so the LLM's base rate and drivers are anchored on the real trajectory, not a memory; and
  2. a directly-derived GROUNDED DRIVER whose SIGN follows the question's polarity (escalation questions move
     with rising conflict; de-escalation/peace questions move against it) and whose STRENGTH follows how far the
     measured trend departs from calm — a real datum entering the sim, not an LLM guess.

Country detection is a deterministic name→FIPS lookup (the same pattern as `detect_product` for crypto); the
polarity of the question (does rising conflict push YES or NO?) is inferred by the LLM in the compile step,
which is why the driver is offered as measured MAGNITUDE + measured trend and the model orients it.
"""
from __future__ import annotations

import re

from swm.retrieval.gdelt_index import asof_social_state

# Country name / demonym / hotspot → GDELT ActionGeo FIPS 10-4 code (column 51, the dominant key in the index).
# Comprehensive across the geopolitically-active world; a bare lookup, not inference — the LLM handles polarity.
_FIPS = {
    "united states": "US", "u.s.": "US", "usa": "US", "america": "US", "american": "US",
    "russia": "RS", "russian": "RS", "moscow": "RS", "kremlin": "RS",
    "ukraine": "UP", "ukrainian": "UP", "kyiv": "UP", "kiev": "UP",
    "china": "CH", "chinese": "CH", "beijing": "CH",
    "taiwan": "TW", "taiwanese": "TW", "taipei": "TW",
    "iran": "IR", "iranian": "IR", "tehran": "IR",
    "israel": "IS", "israeli": "IS", "jerusalem": "IS", "idf": "IS",
    "palestine": "GZ", "palestinian": "GZ", "gaza": "GZ", "hamas": "GZ", "west bank": "WE",
    "north korea": "KN", "pyongyang": "KN", "dprk": "KN",
    "south korea": "KS", "seoul": "KS",
    "india": "IN", "indian": "IN", "delhi": "IN",
    "pakistan": "PK", "pakistani": "PK", "islamabad": "PK",
    "afghanistan": "AF", "afghan": "AF", "taliban": "AF", "kabul": "AF",
    "syria": "SY", "syrian": "SY", "damascus": "SY",
    "iraq": "IZ", "iraqi": "IZ", "baghdad": "IZ",
    "yemen": "YM", "yemeni": "YM", "houthi": "YM", "houthis": "YM",
    "lebanon": "LE", "lebanese": "LE", "hezbollah": "LE", "beirut": "LE",
    "saudi arabia": "SA", "saudi": "SA", "riyadh": "SA",
    "turkey": "TU", "turkish": "TU", "ankara": "TU", "erdogan": "TU",
    "venezuela": "VE", "venezuelan": "VE", "caracas": "VE", "maduro": "VE",
    "sudan": "SU", "sudanese": "SU", "khartoum": "SU",
    "ethiopia": "ET", "ethiopian": "ET", "tigray": "ET",
    "nigeria": "NI", "nigerian": "NI",
    "united kingdom": "UK", "u.k.": "UK", "britain": "UK", "british": "UK", "england": "UK", "london": "UK",
    "france": "FR", "french": "FR", "paris": "FR",
    "germany": "GM", "german": "GM", "berlin": "GM",
    "japan": "JA", "japanese": "JA", "tokyo": "JA",
    "mexico": "MX", "mexican": "MX",
    "brazil": "BR", "brazilian": "BR",
    "egypt": "EG", "egyptian": "EG", "cairo": "EG",
    "myanmar": "BM", "burma": "BM", "burmese": "BM",
    "somalia": "SO", "somali": "SO",
    "libya": "LY", "libyan": "LY",
    "mali": "ML", "malian": "ML",
    "niger": "NG",
    "congo": "CG", "drc": "CG",
    "armenia": "AM", "armenian": "AM",
    "azerbaijan": "AJ", "azerbaijani": "AJ", "nagorno": "AJ", "karabakh": "AJ",
    "georgia country": "GG", "tbilisi": "GG",
    "belarus": "BO", "belarusian": "BO", "minsk": "BO",
    "poland": "PL", "polish": "PL", "warsaw": "PL",
    "hungary": "HU", "hungarian": "HU",
    "serbia": "RI", "serbian": "RI", "kosovo": "KV",
    "haiti": "HA", "haitian": "HA",
    "colombia": "CO", "colombian": "CO",
    "philippines": "RP", "filipino": "RP",
    "thailand": "TH", "thai": "TH",
    "cuba": "CU", "cuban": "CU",
}
# multi-word hotspots must be scanned before single tokens so "west bank" wins over "bank"
_KEYS = sorted(_FIPS, key=lambda k: -len(k))


def detect_country(text):
    """The first country/demonym/hotspot named in the text → its FIPS code (longest match wins). None if absent.
    Matches on word boundaries so trailing punctuation ("Ukraine?", "Israel.") still resolves."""
    t = (text or "").lower()
    for name in _KEYS:
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", t):
            return _FIPS[name], name
    return None, None


def _lvl(x, lo, hi):
    return "low" if x < lo else ("high" if x > hi else "moderate")


class GdeltSocialGrounder:
    """Grounds a social/geopolitical question in the measured as-of state of its country's event stream."""

    name = "gdelt_social"

    def __init__(self, *, window_days=30):
        self.window_days = window_days

    def ground_social(self, question, as_of):
        """Measured as-of social state for the question's country → {country, state, block, driver} or None.

        `block` is the quantified context injected into the compile prompt; `driver` is a measured escalation
        magnitude + trend (unsigned) the simulation orients by question polarity."""
        code, name = detect_country(question)
        if code is None or as_of is None:
            return None
        st = asof_social_state(code, float(as_of), window_days=self.window_days)
        if st is None:
            return None
        # Human-readable, trend-bearing context. Goldstein: negative = net conflictual (−10..+10).
        tone_dir = "worsening" if st["tone_trend"] < -0.3 else ("improving" if st["tone_trend"] > 0.3 else "flat")
        conf_dir = ("rising" if st["conflict_trend"] > 0.15 else
                    ("easing" if st["conflict_trend"] < -0.15 else "steady"))
        prot_dir = ("rising" if st["protest_trend"] > 0.002 else
                    ("falling" if st["protest_trend"] < -0.002 else "steady"))
        block = (
            f"MEASURED SOCIAL STATE of {name.title()} as of {st['as_of']} (GDELT event stream, "
            f"{st['n_events']} events, leakage-free — this is the real current trajectory, use it to set the "
            f"base rate and drivers):\n"
            f"  - conflict/cooperation (Goldstein −10..+10, negative=conflictual): {st['goldstein']} "
            f"({_lvl(-st['goldstein'], 0.5, 2.5)} conflict), trend {conf_dir}\n"
            f"  - media tone (negative=hostile/crisis): {st['tone']}, trend {tone_dir}\n"
            f"  - violence event rate: {st['violence_rate']} ({_lvl(st['violence_rate'], 0.02, 0.06)}), "
            f"protest rate: {st['protest_rate']} ({_lvl(st['protest_rate'], 0.02, 0.05)}, {prot_dir}), "
            f"diplomacy rate: {st['diplomacy_rate']}\n")
        # Unsigned measured "escalation pressure": how far conflict/violence depart from calm, plus trend.
        base_mag = max(0.0, -st["goldstein"] / 5.0) + 6.0 * st["violence_rate"] + 4.0 * st["protest_rate"]
        trend_mag = 0.6 * max(0.0, st["conflict_trend"]) + 120.0 * max(0.0, st["protest_trend"])
        driver = {"escalation_magnitude": round(min(1.0, base_mag), 3),
                  "escalation_trend": round(min(1.0, trend_mag), 3),
                  "goldstein": st["goldstein"], "violence_rate": st["violence_rate"]}
        return {"country": code, "name": name, "state": st, "block": block, "driver": driver}
