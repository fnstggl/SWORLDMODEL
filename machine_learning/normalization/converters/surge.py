"""SURGE (synlp/SURGE) — event-centric social-media sentiment TIME SERIES + interaction
structure. Written against the DOCUMENTED repo format (NOT downloaded; a moderate git repo).

Documented layout (github.com/synlp/SURGE, data/):
  data/events/<event>_<granularity>/
    ├── comment_count.csv              Discussion Intensity  c_t = |P_t|  (volume per time bin)
    ├── comment_count_normalized.csv   train-split-normalized variant
    ├── sentiment_polarity.csv         Sentiment Polarity    s_bar_t      (per time bin)
    ├── sentiment_polarity_normalized.csv
    ├── normalization.json             train-split-only normalization statistics
    └── text_view.jsonl                per-bin top-3 main + earliest-2 replies (POST-IDs only)
  data/events/<event>/
    ├── edges.jsonl                    reply / repost edges (post-IDs, ISO times)
    └── post_id_lookup.jsonl           post_id -> (platform, url) for external hydration
  RAW POST TEXT IS INTENTIONALLY EXCLUDED (only post-IDs are released).

Exact CSV column headers and JSONL field names are NOT published in the repo docs, so this
converter parses them ROBUSTLY (header aliases) and DOC.assumptions flags that they must be
reconciled against the real repo when it is acquired.

Emits (split_unit = event; group_id = event so all granularities of an event hold out together):
  PREDICT_POPULATION_TIME_SERIES  — given an observed prefix of an event's (volume, sentiment)
                                    series, the post-cutoff series is the target.
  PREDICT_POPULATION_RESPONSE     — aggregate volume/sentiment response over the event window.
  PREDICT_TRAJECTORY_CONTINUATION — interaction-structure growth: given a prefix of the reply/
                                    repost edge stream (ordered by time), the remaining edges.

No individual-person tasks are produced: SURGE ships only aggregate series + anonymized post-ID
edges (no text, no persona, no per-user labels). Post-IDs are pseudonymized (deterministically,
preserving parent/child structure) so the interaction graph cannot be re-hydrated to accounts.

License: MIT (code) + CC-BY-4.0 (author-created data). training_suitability="train" (attribution).
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before

#: Known granularity suffixes (real repo uses three temporal granularities); reconcile on acquire.
_GRAN_TOKENS = {"hour", "hourly", "6hour", "6h", "1h", "day", "daily", "1d", "week", "weekly", "1w",
                "minute", "min", "12h", "3h", "2h"}

_TIME_ALIASES = ["timestamp", "time", "datetime", "date", "iso_time", "created_at"]
_BIN_ALIASES = ["bin", "bucket", "index", "t", "step", "bin_index", "window"]
_VOLUME_ALIASES = ["comment_count", "count", "volume", "c_t", "discussion_intensity", "n_posts",
                   "num_posts", "value", "y"]
_SENTIMENT_ALIASES = ["sentiment_polarity", "sentiment", "polarity", "s_bar_t", "s_t", "mean_sentiment",
                      "value", "y"]

_PARENT_ALIASES = ["parent", "parent_id", "src", "source", "from", "in_reply_to", "reply_to", "target_id"]
_CHILD_ALIASES = ["child", "child_id", "dst", "target", "to", "post_id", "id", "source_id"]
_EDGE_TIME_ALIASES = ["time", "timestamp", "created_at", "t", "iso_time", "date", "datetime"]
_EDGE_TYPE_ALIASES = ["type", "edge_type", "relation", "kind", "interaction_type"]


def _first_alias(fieldnames, aliases, fallback=None):
    low = {f.lower(): f for f in (fieldnames or [])}
    for a in aliases:
        if a in low:
            return low[a]
    return fallback


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_series_csv(path: str) -> list[dict]:
    """Read one time-series CSV -> ordered [{bin, t, value}] with robust header detection."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        time_key = _first_alias(fields, _TIME_ALIASES)
        bin_key = _first_alias(fields, _BIN_ALIASES)
        # value column: first field that is not the time/bin column
        val_key = None
        for f in fields:
            if f not in (time_key, bin_key):
                val_key = f
                break
        rows = []
        for i, r in enumerate(reader):
            b = r.get(bin_key) if bin_key else i
            rows.append({"bin": b if b is not None else i,
                         "t": r.get(time_key) if time_key else (r.get(bin_key) if bin_key else i),
                         "value": _num(r.get(val_key)) if val_key else None})
    return rows


def _join_series(volume: list[dict], sentiment: list[dict]) -> list[dict]:
    """Join volume + sentiment series by bin (falling back to row position)."""
    sent_by_bin = {str(s["bin"]): s["value"] for s in sentiment}
    n = max(len(volume), len(sentiment))
    out = []
    for i in range(n):
        v = volume[i] if i < len(volume) else None
        s = sentiment[i] if i < len(sentiment) else None
        b = (v or s or {}).get("bin", i)
        t = (v or s or {}).get("t", b)
        sent_val = sent_by_bin.get(str(b)) if v is not None else (s["value"] if s else None)
        out.append({"bin": b, "t": t,
                    "volume": v["value"] if v else None,
                    "sentiment": sent_val})
    return out


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _split_event_gran(dir_basename: str) -> tuple[str, str]:
    if "_" in dir_basename:
        stem, suffix = dir_basename.rsplit("_", 1)
        if suffix.lower() in _GRAN_TOKENS:
            return stem, suffix
    return dir_basename, ""


class Converter(BaseConverter):
    DATASET_ID = "surge"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "surge"
    DOC = {
        "dataset_id": "surge",
        "original_fields": [
            {"name": "comment_count.csv", "meaning": "Discussion Intensity c_t=|P_t| (post volume) per time bin"},
            {"name": "sentiment_polarity.csv", "meaning": "mean Sentiment Polarity s_bar_t per time bin"},
            {"name": "*_normalized.csv", "meaning": "train-split-normalized variants of the two series"},
            {"name": "normalization.json", "meaning": "train-split-only normalization statistics"},
            {"name": "edges.jsonl", "meaning": "reply/repost interaction edges (parent/child post-IDs + ISO time)"},
            {"name": "post_id_lookup.jsonl", "meaning": "post_id -> (platform, url) for external hydration"},
            {"name": "text_view.jsonl", "meaning": "per-bin representative post-IDs (raw text NOT included)"},
        ],
        "canonical_mapping": [
            {"source_field": "<event>_<granularity> dir", "canonical_path": "episode_id / topic_id / population_id (event)"},
            {"source_field": "comment_count.csv[value]", "canonical_path": "time_series[].volume / aggregate_metrics.total_volume"},
            {"source_field": "sentiment_polarity.csv[value]", "canonical_path": "time_series[].sentiment / aggregate_metrics.mean_sentiment"},
            {"source_field": "csv[bin|timestamp]", "canonical_path": "time_series[].t (observed_prefix vs target split at cutoff)"},
            {"source_field": "edges.jsonl[parent,child,time,type]", "canonical_path": "known_history[] edge events / target.continuation"},
            {"source_field": "post_id (parent/child)", "canonical_path": "pseudonymized (participant); structure preserved"},
        ],
        "tasks_produced": ["PREDICT_POPULATION_TIME_SERIES", "PREDICT_POPULATION_RESPONSE",
                           "PREDICT_TRAJECTORY_CONTINUATION"],
        "unavailable_fields": [
            "raw post text (intentionally excluded by the dataset; only post-IDs released)",
            "per-user persona / individual labels (aggregate + graph only)",
            "verified private beliefs/goals",
        ],
        "chronology_rules": "POPULATION_TIME_SERIES exposes the observed prefix series[:cutoff] and hides "
                            "series[cutoff:] as the target. TRAJECTORY_CONTINUATION orders edges by ISO time "
                            "and hides edges after the cutoff. POPULATION_RESPONSE exposes only the event "
                            "descriptor (not the series values) and predicts the aggregate.",
        "split_key": "event (group_id = pseudonymized event; every granularity of an event held out together)",
        "leakage_risks": [
            "different temporal granularities of the SAME event must not straddle splits -> tied via group_id=event",
            "post-IDs could be hydrated to real accounts -> pseudonymized deterministically (parent/child structure kept)",
        ],
        "known_limitations": [
            "normalized CSV variants exist; the non-normalized series is used as the target (normalized noted)",
            "text_view.jsonl / post_id_lookup.jsonl are references only (no text) and are not expanded",
            "an event with only a series (no edges.jsonl) yields no TRAJECTORY_CONTINUATION (skipped)",
        ],
        "license_implications": "MIT (code) + CC-BY-4.0 (data): training + commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": [
            "EXACT CSV column headers and JSONL field names are NOT published in the repo docs; this converter "
            "detects them via header aliases (time/bin/value; parent/child/time/type) and these MUST BE "
            "RECONCILED against the real repo when acquired",
            "directory basename is '<event>_<granularity>'; the granularity suffix is stripped only when it "
            "matches a known granularity token, else the whole basename is the event",
            "edges.jsonl for an event lives in a sibling data/events/<event>/ dir (or the same series dir)",
        ],
    }

    # ---- discovery ---------------------------------------------------------------------
    def _discover(self, raw_dir: Path):
        series_dirs: dict[str, dict] = {}
        for csvpath in glob.glob(str(raw_dir / "**" / "*.csv"), recursive=True):
            if ".cache" in csvpath:
                continue
            name = Path(csvpath).name.lower()
            parent = str(Path(csvpath).parent)
            if name == "comment_count.csv":
                series_dirs.setdefault(parent, {})["volume"] = csvpath
            elif name == "sentiment_polarity.csv":
                series_dirs.setdefault(parent, {})["sentiment"] = csvpath
        edges_by_base: dict[str, str] = {}
        edges_by_dir: dict[str, str] = {}
        for ep in glob.glob(str(raw_dir / "**" / "edges.jsonl"), recursive=True):
            if ".cache" in ep:
                continue
            edges_by_dir[str(Path(ep).parent)] = ep
            edges_by_base[Path(ep).parent.name] = ep
        return series_dirs, edges_by_base, edges_by_dir

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        series_dirs, edges_by_base, edges_by_dir = self._discover(raw_dir)
        if not series_dirs and not edges_by_dir:
            raise FileNotFoundError(f"no SURGE series CSV / edges JSONL found under {raw_dir}")

        seen_edge_dirs: set[str] = set()
        for sdir, files in sorted(series_dirs.items()):
            base = Path(sdir).name
            event, gran = _split_event_gran(base)
            edge_path = edges_by_dir.get(sdir) or edges_by_base.get(event) or edges_by_base.get(base)
            if edge_path:
                seen_edge_dirs.add(str(Path(edge_path).parent))
            yield from self._one_series(sdir, event, gran, files, edge_path)

        # events that ship ONLY an edge graph (no series) still get TRAJECTORY_CONTINUATION
        for edir, epath in sorted(edges_by_dir.items()):
            if edir in seen_edge_dirs:
                continue
            event, gran = _split_event_gran(Path(edir).name)
            rec = self._trajectory(event, gran, epath, files_note=["edges.jsonl"])
            if rec is not None:
                yield rec

    # ---- per-event series --------------------------------------------------------------
    def _one_series(self, sdir, event, gran, files, edge_path) -> Iterator[dict]:
        volume = _read_series_csv(files["volume"]) if "volume" in files else []
        sentiment = _read_series_csv(files["sentiment"]) if "sentiment" in files else []
        series = _join_series(volume, sentiment)
        n = len(series)
        event_pseud = self.pseudonym("group", event)
        gran_tag = gran or "series"
        episode_id = f"surge-{event}-{gran_tag}"
        rel = [str(Path(files.get("volume", files.get("sentiment", ""))).name)]
        loc = {"files": [f"data/events/{event}_{gran}"], "indices": list(range(n)), "ids": [episode_id]}
        missing = []
        if not volume:
            missing.append("comment_count.csv")
        if not sentiment:
            missing.append("sentiment_polarity.csv")

        world = {"event": event, "granularity": gran, "n_bins": n, "platform": "social_media"}

        def dq(extra_missing=None, warnings=None, confidence="high"):
            return {"missing_fields": missing + (extra_missing or []),
                    "warnings": warnings or [], "confidence": confidence,
                    "chronology_verified": True, "target_verified": True,
                    "possible_leakage": False, "license_verified": True}

        # ---------------------------------- POPULATION_TIME_SERIES --------------------
        if n >= 2:
            cutoff = max(1, n // 2)
            prefix = series[:cutoff]
            target = series[cutoff:]
            payload = {
                "input": {"population_features": {"event": event, "granularity": gran, "n_bins": n},
                          "event_context": world,
                          "observed_prefix": prefix},
                "target": {"time_series": target},
            }
            yield self.make(
                task_type="PREDICT_POPULATION_TIME_SERIES", payload=payload, episode_id=episode_id,
                cutoff_sequence_index=cutoff, cutoff_time=(prefix[-1].get("t") if prefix else None),
                actor_role="population", population_id=event_pseud, group_id=event_pseud,
                topic_id=event, participant_ids=[], persistent_identity_available=False,
                context={"world_state": world, "current_observation": {"observed_prefix_len": cutoff},
                         "available_actions": None, "language": ""},
                raw_locator=loc,
                transformation_steps=["read comment_count.csv + sentiment_polarity.csv",
                                      "join by bin", f"cutoff after {cutoff} of {n} bins",
                                      "post-cutoff series -> target"],
                data_quality=dq(warnings=None if n > 2 else ["short series (n<=2)"]))

        # ---------------------------------- POPULATION_RESPONSE -----------------------
        vols = [b["volume"] for b in series if isinstance(b["volume"], (int, float))]
        sents = [b["sentiment"] for b in series if isinstance(b["sentiment"], (int, float))]
        total_volume = sum(vols) if vols else None
        mean_sentiment = (sum(sents) / len(sents)) if sents else None
        peak_volume = max(vols) if vols else None
        peak_bin = None
        if vols:
            peak_bin = max(range(len(series)),
                           key=lambda i: (series[i]["volume"] if isinstance(series[i]["volume"], (int, float)) else float("-inf")))
        pos = sum(1 for s in sents if s > 0)
        neg = sum(1 for s in sents if s < 0)
        neu = sum(1 for s in sents if s == 0)
        dist = None
        if sents:
            tot = len(sents)
            dist = {"positive": pos / tot, "negative": neg / tot, "neutral": neu / tot}
        payload = {
            "input": {"population_features": {"event": event, "granularity": gran, "n_bins": n},
                      "intervention": {"event": event},
                      "historical_context": {"note": "aggregate over the full event window"}},
            "target": {"aggregate_metrics": {"total_volume": total_volume, "mean_sentiment": mean_sentiment,
                                             "peak_volume": peak_volume, "peak_bin": peak_bin, "n_bins": n},
                       "response_distribution": dist},
        }
        yield self.make(
            task_type="PREDICT_POPULATION_RESPONSE", payload=payload, episode_id=episode_id,
            actor_role="population", population_id=event_pseud, group_id=event_pseud,
            topic_id=event, participant_ids=[], persistent_identity_available=False,
            context={"world_state": world, "available_actions": None, "language": ""},
            raw_locator=loc,
            transformation_steps=["read the two series", "aggregate volume + sentiment over the event window"],
            data_quality=dq(extra_missing=(["sentiment_values"] if not sents else []) + (["volume_values"] if not vols else [])))

        # ---------------------------------- TRAJECTORY_CONTINUATION -------------------
        if edge_path:
            rec = self._trajectory(event, gran, edge_path, series_episode=episode_id, loc_files=[f"data/events/{event}"])
            if rec is not None:
                yield rec

    # ---- interaction-structure growth --------------------------------------------------
    def _trajectory(self, event, gran, edge_path, series_episode=None, loc_files=None, files_note=None):
        edges = _read_jsonl(edge_path)
        if not edges:
            return None
        fields = list(edges[0].keys())
        p_key = _first_alias(fields, _PARENT_ALIASES)
        c_key = _first_alias(fields, _CHILD_ALIASES)
        t_key = _first_alias(fields, _EDGE_TIME_ALIASES)
        ty_key = _first_alias(fields, _EDGE_TYPE_ALIASES)

        def edge_time(e):
            return str(e.get(t_key)) if t_key else ""

        ordered = sorted(range(len(edges)), key=lambda i: (edge_time(edges[i]), i))
        event_pseud = self.pseudonym("group", event)
        episode_id = series_episode or f"surge-{event}-{gran or 'graph'}"

        evs = []
        for idx, oi in enumerate(ordered):
            e = edges[oi]
            parent = e.get(p_key) if p_key else None
            child = e.get(c_key) if c_key else None
            actor = self.pseudonym("participant", child) if child is not None else self.pseudonym("participant", f"edge{oi}")
            evs.append(history_event(
                idx, actor, "interaction", t=e.get(t_key) if t_key else None,
                action_type=(e.get(ty_key) if ty_key else "reply_or_repost"),
                action_content={"parent": self.pseudonym("participant", parent) if parent is not None else None,
                                "child": self.pseudonym("participant", child) if child is not None else None},
                meta={"edge_index": oi}))
        m = len(evs)
        if m < 2:
            return None
        cutoff = max(1, m // 2)
        hist = history_before(evs, cutoff)
        cont = evs[cutoff:]
        loc = {"files": (loc_files or ["data/events"]), "indices": [o for o in ordered], "ids": [episode_id]}
        payload = {"input": {"history": hist, "horizon": len(cont)},
                   "target": {"continuation": cont}}
        return self.make(
            task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
            sequence_index=cutoff, cutoff_sequence_index=cutoff,
            cutoff_time=(hist[-1].get("t") if hist else None),
            actor_role="population", population_id=event_pseud, group_id=event_pseud,
            topic_id=event, participant_ids=[], persistent_identity_available=False,
            context={"world_state": {"event": event, "granularity": gran, "n_edges": m,
                                     "structure": "reply/repost graph"},
                     "known_history": hist, "available_actions": None, "language": ""},
            raw_locator=loc,
            transformation_steps=["read edges.jsonl", "order edges by ISO time", "pseudonymize post-IDs",
                                  f"cutoff after {cutoff} of {m} edges", "remaining edges -> continuation"],
            data_quality={"missing_fields": [], "warnings": ["post-IDs pseudonymized; parent/child structure preserved"],
                          "confidence": "high", "chronology_verified": True, "target_verified": True,
                          "possible_leakage": False, "license_verified": True,
                          "inferred_fields": ["edge ordering (by ISO time)"]})
