"""Structural-state grounding — the SLOW latent state the event flow rides on.

GDELT measures what is happening THIS WEEK. But the same shock resolves differently depending on the slow,
institutional/economic STATE of the society it lands in: a protest wave in a consolidated democracy with strong
rule of law fizzles; the identical wave in a fragile, low-rule-of-law state tips into a coup. That slow state —
regime type, institutional strength, growth, unemployment, inflation — is the world model's true LATENT STATE;
GDELT is the TRANSITION/flow on top of it. The world-model coupling is `outcome ≈ f(structural_state × shock)`:
fragility AMPLIFIES a shock, robustness DAMPENS it.

Two keyless live sources realize the two structural sub-layers:
  - VDemGrounder      institutional / regime state — V-Dem indices via Our World in Data's filtered CSV
                      (electoral democracy, liberal democracy, rule of law, political corruption), country-year.
  - WorldBankGrounder economic state — World Bank WDI API (GDP growth, unemployment, inflation, GDP/capita).
StructuralGrounder fuses them into an as-of state + a natural-language block for the compile prompt + a single
FRAGILITY scalar in [0,1] (weak institutions + weak rule of law + economic stress) that the forecaster uses to
scale the GDELT escalation driver — the structural×event transition.

As-of / leakage-free: every value is the most recent one dated at or before the question's year; these series
are annual and lag, so a value dated ≤ the as-of year was genuinely known then.
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

from swm.api.gdelt_social import detect_country

CACHE = Path("data/structural_cache")

# The same geopolitically-active set as the GDELT grounder, keyed by the SAME lowercased names its detector
# returns → ISO-3166 alpha-3 (used by both World Bank and OWID/V-Dem).
_NAME_ISO3 = {
    "united states": "USA", "u.s.": "USA", "usa": "USA", "america": "USA", "american": "USA",
    "russia": "RUS", "russian": "RUS", "moscow": "RUS", "kremlin": "RUS",
    "ukraine": "UKR", "ukrainian": "UKR", "kyiv": "UKR", "kiev": "UKR",
    "china": "CHN", "chinese": "CHN", "beijing": "CHN", "taiwan": "TWN", "taiwanese": "TWN", "taipei": "TWN",
    "iran": "IRN", "iranian": "IRN", "tehran": "IRN", "israel": "ISR", "israeli": "ISR", "jerusalem": "ISR",
    "idf": "ISR", "palestine": "PSE", "palestinian": "PSE", "gaza": "PSE", "hamas": "PSE", "west bank": "PSE",
    "north korea": "PRK", "pyongyang": "PRK", "dprk": "PRK", "south korea": "KOR", "seoul": "KOR",
    "india": "IND", "indian": "IND", "delhi": "IND", "pakistan": "PAK", "pakistani": "PAK", "islamabad": "PAK",
    "afghanistan": "AFG", "afghan": "AFG", "taliban": "AFG", "kabul": "AFG",
    "syria": "SYR", "syrian": "SYR", "damascus": "SYR", "iraq": "IRQ", "iraqi": "IRQ", "baghdad": "IRQ",
    "yemen": "YEM", "yemeni": "YEM", "houthi": "YEM", "houthis": "YEM",
    "lebanon": "LBN", "lebanese": "LBN", "hezbollah": "LBN", "beirut": "LBN",
    "saudi arabia": "SAU", "saudi": "SAU", "riyadh": "SAU",
    "turkey": "TUR", "turkish": "TUR", "ankara": "TUR", "erdogan": "TUR",
    "venezuela": "VEN", "venezuelan": "VEN", "caracas": "VEN", "maduro": "VEN",
    "sudan": "SDN", "sudanese": "SDN", "khartoum": "SDN", "ethiopia": "ETH", "ethiopian": "ETH", "tigray": "ETH",
    "nigeria": "NGA", "nigerian": "NGA",
    "united kingdom": "GBR", "u.k.": "GBR", "britain": "GBR", "british": "GBR", "england": "GBR", "london": "GBR",
    "france": "FRA", "french": "FRA", "paris": "FRA", "germany": "DEU", "german": "DEU", "berlin": "DEU",
    "japan": "JPN", "japanese": "JPN", "tokyo": "JPN", "mexico": "MEX", "mexican": "MEX",
    "brazil": "BRA", "brazilian": "BRA", "egypt": "EGY", "egyptian": "EGY", "cairo": "EGY",
    "myanmar": "MMR", "burma": "MMR", "burmese": "MMR", "somalia": "SOM", "somali": "SOM",
    "libya": "LBY", "libyan": "LBY", "mali": "MLI", "malian": "MLI", "niger": "NER",
    "congo": "COD", "drc": "COD", "armenia": "ARM", "armenian": "ARM",
    "azerbaijan": "AZE", "azerbaijani": "AZE", "nagorno": "AZE", "karabakh": "AZE",
    "belarus": "BLR", "belarusian": "BLR", "minsk": "BLR", "poland": "POL", "polish": "POL", "warsaw": "POL",
    "hungary": "HUN", "hungarian": "HUN", "serbia": "SRB", "serbian": "SRB", "kosovo": "XKX",
    "haiti": "HTI", "haitian": "HTI", "colombia": "COL", "colombian": "COL",
    "philippines": "PHL", "filipino": "PHL", "thailand": "THA", "thai": "THA", "cuba": "CUB", "cuban": "CUB",
}


def detect_iso3(text):
    """Country ISO-3 for the question (reuses the GDELT name detector, then maps its name → ISO-3). None if absent."""
    _, name = detect_country(text)
    if name is None:
        return None, None
    return _NAME_ISO3.get(name), name


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "swm-structural/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


class VDemGrounder:
    """Institutional / regime latent state — V-Dem indices via Our World in Data's filtered CSV (keyless).
    Each index is downloaded once (whole country-year panel) and cached, then read as-of."""

    name = "vdem_owid"
    # OWID grapher slug → short field name; all are V-Dem 0..1 indices (higher = stronger institution)
    SERIES = {"electoral-democracy-index": "electoral_democracy",
              "liberal-democracy-index": "liberal_democracy",
              "rule-of-law-index": "rule_of_law",
              "political-corruption-index": "corruption"}     # NOTE: higher corruption index = MORE corrupt

    def _panel(self, slug):
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = CACHE / f"vdem_{slug}.json"
        if cp.exists():
            return json.loads(cp.read_text())
        try:
            txt = _get(f"https://ourworldindata.org/grapher/{slug}.csv?csvType=filtered&"
                       f"useColumnShortNames=true").decode("utf-8", "ignore")
        except Exception:
            return {}
        panel = {}
        for ln in txt.strip().split("\n")[1:]:
            c = ln.split(",")
            if len(c) < 4 or not c[1]:                        # entity, code, year, value
                continue
            try:
                panel.setdefault(c[1], {})[int(c[2])] = float(c[3])
            except (ValueError, IndexError):
                continue
        cp.write_text(json.dumps(panel))
        return panel

    def ground(self, iso3, year):
        """The most recent V-Dem indices for `iso3` dated at or before `year` (leakage-free). {} if none."""
        out = {}
        for slug, field in self.SERIES.items():
            panel = self._panel(slug).get(iso3, {})
            past = [(int(y), v) for y, v in panel.items() if int(y) <= year]
            if past:
                out[field] = round(max(past)[1], 3)
                out[f"{field}_year"] = max(past)[0]
        return out


class WorldBankGrounder:
    """Economic latent state — World Bank WDI (keyless). Most-recent-at-or-before-year values."""

    name = "worldbank_wdi"
    SERIES = {"NY.GDP.MKTP.KD.ZG": "gdp_growth_pct", "SL.UEM.TOTL.ZS": "unemployment_pct",
              "FP.CPI.TOTL.ZG": "inflation_pct", "NY.GDP.PCAP.CD": "gdp_per_capita"}

    def ground(self, iso3, year):
        CACHE.mkdir(parents=True, exist_ok=True)
        out = {}
        for ind, field in self.SERIES.items():
            cp = CACHE / f"wb_{iso3}_{ind}_{year}.json"
            if cp.exists():
                v = json.loads(cp.read_text())
            else:
                try:
                    d = json.loads(_get(f"https://api.worldbank.org/v2/country/{iso3}/indicator/{ind}"
                                        f"?date={year - 8}:{year}&format=json&per_page=40"))
                    rows = [(int(r["date"]), r["value"]) for r in (d[1] or []) if r["value"] is not None] \
                        if len(d) > 1 and d[1] else []
                    v = {"year": max(rows)[0], "value": round(max(rows)[1], 3)} if rows else None
                except Exception:
                    v = None
                cp.write_text(json.dumps(v))
            if v:
                out[field] = v["value"]
                out[f"{field}_year"] = v["year"]
        return out


class StructuralGrounder:
    """Fuse the institutional (V-Dem) and economic (World Bank) latent state → as-of state + prompt block +
    a FRAGILITY scalar in [0,1] the forecaster uses to scale event shocks (the structural×event coupling)."""

    name = "structural_state"

    def __init__(self):
        self.vdem = VDemGrounder()
        self.wb = WorldBankGrounder()

    def ground_structural(self, question, as_of):
        import datetime as _dt
        iso3, name = detect_iso3(question)
        if iso3 is None or as_of is None:
            return None
        year = _dt.datetime.utcfromtimestamp(int(as_of)).year
        vd = self.vdem.ground(iso3, year)
        ec = self.wb.ground(iso3, year)
        if not vd and not ec:
            return None
        fragility = self._fragility(vd, ec)
        block = self._block(name, vd, ec, fragility)
        return {"iso3": iso3, "name": name, "vdem": vd, "econ": ec, "fragility": fragility, "block": block}

    @staticmethod
    def _fragility(vd, ec):
        """0 = robust (strong institutions, calm economy), 1 = fragile. Weak rule of law / weak democracy /
        high corruption / high inflation / high unemployment all raise it. Missing pieces fall back to neutral."""
        parts = []
        if "rule_of_law" in vd:
            parts.append(1 - vd["rule_of_law"])               # V-Dem indices are 0..1, higher=stronger
        if "electoral_democracy" in vd:
            parts.append(1 - vd["electoral_democracy"])
        if "corruption" in vd:
            parts.append(vd["corruption"])                    # higher index = more corrupt = more fragile
        if "inflation_pct" in ec:
            parts.append(min(1.0, max(0.0, (ec["inflation_pct"] - 3) / 30)))   # >33% infl saturates
        if "unemployment_pct" in ec:
            parts.append(min(1.0, max(0.0, (ec["unemployment_pct"] - 4) / 20)))
        return round(sum(parts) / len(parts), 3) if parts else 0.5

    @staticmethod
    def _block(name, vd, ec, fragility):
        lines = [f"STRUCTURAL STATE of {name.title()} (slow-moving institutional + economic latent state, "
                 f"leakage-free, most-recent-known):"]
        if vd:
            lines.append(f"  - institutions (V-Dem 0-1, higher=stronger): electoral-democracy "
                         f"{vd.get('electoral_democracy','?')}, rule-of-law {vd.get('rule_of_law','?')}, "
                         f"corruption-index {vd.get('corruption','?')} (higher=more corrupt)")
        if ec:
            lines.append(f"  - economy: GDP growth {ec.get('gdp_growth_pct','?')}%, unemployment "
                         f"{ec.get('unemployment_pct','?')}%, inflation {ec.get('inflation_pct','?')}%")
        lines.append(f"  - overall structural FRAGILITY {fragility} (0=robust/stable, 1=fragile) — a fragile "
                     f"state amplifies shocks (unrest tips over), a robust state absorbs them.")
        return "\n".join(lines) + "\n"
