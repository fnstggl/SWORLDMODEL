"""Build the replay vault from the audited resolved-event list (one-time; idempotent).

Events are real, objectively resolved public matters; each gets TWO forecast cutoffs (a far and a near
snapshot, both strictly before resolution). Outcomes + resolution notes go ONLY into
SEALED_resolutions.json — the public events.json carries no outcome. Correlated questions share an
event_family so the scorer clusters them (e.g. trump_2024/harris_2024 are ONE election).
"""
from __future__ import annotations
import json
import time as _time
from pathlib import Path

VAULT = Path("experiments/replay_vault")

# (event_id, question, near_cutoff, horizon, domain, family, outcome_yes(SEALED), note(SEALED))
_ROWS = [
    ("us2024_trump", "Will Donald Trump win the 2024 US presidential election?", "2024-10-20", "2024-11-06",
     "elections", "us2024_presidential", 1, "Trump won; called 2024-11-06."),
    ("us2024_harris", "Will Kamala Harris win the 2024 US presidential election?", "2024-10-20", "2024-11-06",
     "elections", "us2024_presidential", 0, "Harris lost the 2024 election."),
    ("biden_nominee", "Will Joe Biden be the Democratic nominee for the 2024 US presidential election?",
     "2024-07-05", "2024-08-22", "elections", "us2024_nomination", 0, "Biden withdrew 2024-07-21."),
    ("uk_labour", "Will the Labour Party win the 2024 United Kingdom general election?", "2024-06-25",
     "2024-07-04", "elections", "uk2024_ge", 1, "Labour won a majority 2024-07-04."),
    ("shutdown_oct24", "Will there be a US federal government shutdown on October 1, 2024?", "2024-09-20",
     "2024-10-01", "politics", "us_shutdown_2024", 0, "CR signed 2024-09-26; no shutdown."),
    ("fed_sep24", "Will the US Federal Reserve cut interest rates at its September 2024 meeting?",
     "2024-09-10", "2024-09-19", "econ", "fomc_2024h2", 1, "FOMC cut 50bp 2024-09-18."),
    ("fed_jan25", "Will the US Federal Reserve cut interest rates at its January 2025 meeting?",
     "2025-01-20", "2025-01-30", "econ", "fomc_2025h1", 0, "FOMC held steady 2025-01-29."),
    ("recession_24", "Will the United States enter a recession in 2024?", "2024-07-01", "2024-12-31",
     "macro", "us_macro_2024", 0, "No NBER recession in 2024."),
    ("nvda_split", "Will Nvidia announce a stock split in 2024?", "2024-05-01", "2024-06-30",
     "finance", "nvda_2024", 1, "10-for-1 split announced 2024-05-22."),
    ("gpt5_2024", "Will OpenAI release a model called GPT-5 in 2024?", "2024-08-01", "2024-12-31",
     "tech", "openai_2024", 0, "No GPT-5 in 2024."),
    ("apple_intel", "Will Apple release its Apple Intelligence features in 2024?", "2024-08-01", "2024-12-31",
     "tech", "apple_2024", 1, "Launched 2024-10."),
    ("gaza_cf24", "Will Israel and Hamas agree to a ceasefire by the end of 2024?", "2024-10-01", "2024-12-31",
     "geopolitics", "gaza_ceasefire", 0, "No ceasefire in 2024; deal Jan 2025."),
    ("gaza_cf25", "Will an Israel-Hamas ceasefire take effect in January 2025?", "2025-01-10", "2025-01-31",
     "geopolitics", "gaza_ceasefire", 1, "Took effect 2025-01-19."),
    ("assad_fall", "Will Bashar al-Assad's government fall in Syria in 2024?", "2024-11-25", "2024-12-31",
     "geopolitics", "syria_2024", 1, "Government fell 2024-12-08."),
    ("ru_ua_cf24", "Will Russia and Ukraine agree to a ceasefire in 2024?", "2024-06-01", "2024-12-31",
     "geopolitics", "ru_ua_2024", 0, "No ceasefire in 2024."),
    ("india_t20", "Will India win the 2024 ICC Men's T20 Cricket World Cup?", "2024-06-20", "2024-06-29",
     "sports", "t20_2024", 1, "India won the final 2024-06-29."),
    ("starship_catch", "Will SpaceX catch a Starship booster with the launch tower in 2024?", "2024-10-01",
     "2024-12-31", "science", "spacex_2024", 1, "Caught on Flight 5, 2024-10-13."),
    ("btc_100k", "Will Bitcoin exceed one hundred thousand US dollars by the end of 2024?", "2024-11-15",
     "2024-12-31", "finance", "btc_2024", 1, "Passed $100k 2024-12-04."),
]

_FAR_DAYS = 21.0


def _far(near: str) -> str:
    t = _time.mktime(_time.strptime(near, "%Y-%m-%d")) - _FAR_DAYS * 86400.0
    return _time.strftime("%Y-%m-%d", _time.gmtime(t))


def main():
    VAULT.mkdir(parents=True, exist_ok=True)
    events, sealed = [], {}
    for eid, q, near, horizon, domain, family, outcome, note in _ROWS:
        events.append({"event_id": eid, "question": q,
                       "forecast_cutoffs": [_far(near), near], "horizon": horizon,
                       "domain": domain, "event_family": family, "entities": [],
                       "outcome_contract": "resolves YES iff the described event occurred by the horizon, "
                                           "per public record"})
        sealed[eid] = {"outcome": outcome, "resolution_note": note, "blinding_mappings": {}}
    (VAULT / "events.json").write_text(json.dumps(
        {"note": "PUBLIC vault — no outcomes here. Outcomes live in SEALED_resolutions.json (scorer only).",
         "events": events}, indent=1))
    (VAULT / "SEALED_resolutions.json").write_text(json.dumps(
        {"note": "SEALED — scorer only (REPLAY_SCORER=1). The forecaster must never read this file.",
         "resolutions": sealed}, indent=1))
    print(f"vault built: {len(events)} events, {len({e['event_family'] for e in events})} families")


if __name__ == "__main__":
    main()
