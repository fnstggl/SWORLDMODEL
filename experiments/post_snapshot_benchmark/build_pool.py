"""Build the frozen post-snapshot representative event pool.

This process is the only component allowed to inspect source settlement fields.
It emits resolution-free public artifacts into the repository and writes three
split-specific resolution stores to a scorer-only root outside the forecast
mount.  Nothing in the selection algorithm uses which side won.

The source event is the clustering unit: all of its correlated contracts stay
in one world.  A second normalized family key collapses repeated intraday
threshold series (for example, hourly Bitcoin ranges) so they cannot inflate
the independent-world count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
RELEASE_FLOOR = "2026-04-24T00:00:00Z"
CONSTRUCTION_CEILING = "2026-07-14T23:59:59Z"
MIN_OPEN_SECONDS = 6 * 3600
MIN_VOLUME_USDC = 250.0
N_MIN_POOL = 300
N_SELECTED = 100
CUTOFF_FRACTIONS = (0.20, 0.45, 0.70, 0.90)
SELECTION_SEED = "wmv2-post-snapshot-representative-v1"

ROOT = Path("experiments/results/post_snapshot_benchmark")
SOURCE_CACHE = Path("/private/tmp/wmv2-post-snapshot-source-cache-keyset-volume-v1")


def _parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch_json(url: str, *, retries: int = 8):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wmv2-post-snapshot/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - bounded network retry
            last = exc
            retry_after = None
            if isinstance(exc, HTTPError):
                retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(20, 2 ** attempt)
            except (TypeError, ValueError):
                delay = min(20, 2 ** attempt)
            time.sleep(delay)
    detail = f"HTTP {last.code}" if isinstance(last, HTTPError) else type(last).__name__
    raise RuntimeError(f"public archive request failed after {retries} attempts: {detail}")


def fetch_source_events(max_pages: int = 60) -> list[dict]:
    """Fetch newest closed source events with the provider's keyset cursor."""
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    events = []
    after_cursor = None
    for page in range(max_pages):
        params = {
            "closed": "true", "limit": 500,
            "order": "volume", "ascending": "false",
            "end_date_min": RELEASE_FLOOR, "end_date_max": CONSTRUCTION_CEILING,
        }
        if after_cursor:
            params["after_cursor"] = after_cursor
        query = urllib.parse.urlencode(params)
        cache_path = SOURCE_CACHE / f"events_page_{page:03d}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
        else:
            payload = _fetch_json(f"{GAMMA}/events/keyset?{query}") or {}
            cache_path.write_text(json.dumps(payload))
            cache_path.chmod(0o600)
        rows = payload.get("events") or []
        if not rows:
            break
        events.extend(rows)
        print(f"source page {page + 1}: {len(rows)} rows ({len(events)} cumulative)", flush=True)
        time.sleep(0.75)
        after_cursor = payload.get("next_cursor")
        if not after_cursor:
            break
    return events


_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?\b", re.I)
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|et|utc|gmt)\b", re.I)
_NUMBER = re.compile(r"(?<![a-z])[$€£]?\d[\d,.]*(?:%|[kmb])?(?![a-z])", re.I)


def family_key(title: str, domain: str) -> str:
    """Normalize recurring threshold/time variants into one independence family."""
    text = str(title or "").lower().replace("ethereum", "eth").replace("bitcoin", "btc")
    dates = []

    def preserve_date(match):
        token = f" datetoken{chr(97 + len(dates) % 26)} "
        dates.append((token.strip(), re.sub(r"\s+", " ", match.group(0).lower())))
        return token

    text = _DATE.sub(preserve_date, text)
    text = _TIME.sub(" <time> ", text)
    text = _NUMBER.sub(" <n> ", text)
    text = re.sub(r"\b(?:today|tomorrow|tonight|this week|next week)\b", " <date> ", text)
    text = re.sub(r"[_?—–:/()\[\]-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for token, date in dates:
        text = text.replace(token, date)
    return f"{domain}:{text}"


_DOMAIN_RULES = (
    ("crypto", ("bitcoin", "btc", "ethereum", "eth ", "solana", "crypto", "xrp", "doge")),
    ("sports", ("nba", "nfl", "nhl", "mlb", "uefa", "fifa", "ufc", " vs ", "match", "game ",
                "tournament", "grand prix", "wimbledon", "world cup", "series", "playoff")),
    ("politics", ("election", "president", "prime minister", "senate", "congress", "parliament",
                  "governor", "mayor", "vote", "bill", "court", "cabinet", "referendum")),
    ("geopolitics", ("ceasefire", "war", "attack", "missile", "nato", "ukraine", "russia", "israel",
                     "gaza", "iran", "china", "taiwan", "sanction", "hostage", "peace deal")),
    ("economics", ("fed ", "fomc", "interest rate", "inflation", "cpi", "gdp", "unemployment",
                   "tariff", "recession", "central bank")),
    ("technology", ("openai", "deepseek", "anthropic", "google", "apple", "microsoft", "tesla",
                    "model", "iphone", "ai ", "chip", "spacex", "launch")),
    ("business", ("stock", "ipo", "merger", "acquire", "earnings", "ceo", "company", "bankrupt")),
    ("culture", ("movie", "film", "box office", "album", "billboard", "spotify", "netflix", "oscar",
                 "grammy", "youtube", "tiktok")),
    ("weather_science", ("temperature", "rain", "hurricane", "earthquake", "nasa", "fda", "trial")),
)


def classify_domain(title: str, event: dict) -> str:
    text = f" {title.lower()} "
    for domain, tokens in _DOMAIN_RULES:
        if any(token in text for token in tokens):
            return domain
    tags = " ".join(str(t.get("label") or "") for t in (event.get("tags") or [])).lower()
    for domain, tokens in _DOMAIN_RULES:
        if any(token in tags for token in tokens):
            return domain
    return "other"


def _binary_market(event: dict):
    """Highest-volume unambiguous binary market; winner identity never affects eligibility."""
    candidates = []
    for market in event.get("markets") or []:
        try:
            outcomes = json.loads(market.get("outcomes") or "[]")
            prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
            volume = float(market.get("volume") or market.get("volumeNum") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if sorted(str(o).lower() for o in outcomes) != ["no", "yes"]:
            continue
        if sorted(prices) != [0.0, 1.0] or volume < MIN_VOLUME_USDC:
            continue
        if str(market.get("umaResolutionStatus") or "resolved").lower() not in ("resolved", ""):
            continue
        candidates.append((volume, str(market.get("id")), market, outcomes, prices))
    return max(candidates, default=None, key=lambda item: (item[0], item[1]))


def eligible_candidates(source_events: list[dict]) -> tuple[list[dict], dict]:
    floor = _parse_ts(RELEASE_FLOOR) or 0
    ceiling = _parse_ts(CONSTRUCTION_CEILING) or math.inf
    public, resolutions = [], {}
    for event in source_events:
        selected = _binary_market(event)
        if selected is None:
            continue
        volume, _, market, outcomes, prices = selected
        opened = _parse_ts(event.get("createdAt") or event.get("creationDate") or market.get("createdAt"))
        resolved = _parse_ts(event.get("closedTime") or market.get("closedTime") or
                             market.get("endDate") or event.get("endDate"))
        if opened is None or resolved is None or opened < floor or resolved > ceiling:
            continue
        if resolved - opened < MIN_OPEN_SECONDS:
            continue
        question = str(market.get("question") or event.get("title") or "").strip()
        if not question:
            continue
        domain = classify_domain(question, event)
        event_id = f"pm_{event.get('id')}"
        yes_index = [str(o).lower() for o in outcomes].index("yes")
        token_ids = json.loads(market.get("clobTokenIds") or "[]")
        canonical_source = {
            "event_id": str(event.get("id")), "market_id": str(market.get("id")),
            "condition_id": market.get("conditionId"), "created_at": _iso(opened),
            "resolution_time": _iso(resolved), "question": question,
            "description": str(market.get("description") or event.get("description") or ""),
            "resolution_rule": str(market.get("resolutionSource") or event.get("resolutionSource") or ""),
        }
        source_hash = hashlib.sha256(
            json.dumps(canonical_source, sort_keys=True).encode()).hexdigest()
        public.append({
            "event_id": event_id, "source": "polymarket_gamma", "source_event_id": str(event.get("id")),
            "source_market_id": str(market.get("id")), "event_world_cluster": event_id,
            "source_condition_id": market.get("conditionId"),
            "independence_family": family_key(str(event.get("title") or question), domain),
            "question": question, "description": canonical_source["description"],
            "resolution_rule": canonical_source["resolution_rule"], "domain": domain,
            "question_open_time": _iso(opened), "resolution_time": _iso(resolved),
            "duration_s": int(resolved - opened), "n_correlated_contracts": len(event.get("markets") or []),
            "primary_volume_usdc": round(volume, 6), "source_record_sha256": source_hash,
            "yes_token_id": (str(token_ids[yes_index]) if len(token_ids) == 2 else None),
        })
        resolutions[event_id] = {
            "event_id": event_id, "outcome": int(prices[yes_index] == 1.0),
            "resolution_time": _iso(resolved), "source": "polymarket_settlement",
            "source_event_id": str(event.get("id")), "source_market_id": str(market.get("id")),
        }
    return public, resolutions


def collapse_independence_families(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Keep one deterministic event per normalized recurring family."""
    groups = {}
    for row in candidates:
        groups.setdefault(row["independence_family"], []).append(row)
    kept, mapping = [], {}
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: (
            hashlib.sha256(f"{SELECTION_SEED}:{r['event_id']}".encode()).hexdigest(), r["event_id"]))
        winner = ordered[0]
        kept.append(winner)
        mapping[key] = {"representative_event_id": winner["event_id"],
                        "source_event_ids": [r["event_id"] for r in sorted(rows, key=lambda x: x["event_id"])],
                        "n_source_events": len(rows)}
    return sorted(kept, key=lambda r: (r["question_open_time"], r["event_id"])), mapping


def select_representative(candidates: list[dict], n: int = N_SELECTED) -> list[dict]:
    """Deterministic domain-capped systematic sample independent of outcomes/performance."""
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} independent eligible worlds; need {n}")
    domain_counts = Counter(row["domain"] for row in candidates)
    # Preserve the source distribution without allowing its sports majority to
    # dominate the benchmark.  The cap is frozen and is never relaxed.
    caps = {domain: min(30, max(5, math.ceil(n * count / len(candidates)) + 8))
            for domain, count in domain_counts.items()}
    ranked = sorted(candidates, key=lambda row: (
        hashlib.sha256(f"{SELECTION_SEED}:{row['event_id']}".encode()).hexdigest(), row["event_id"]))
    chosen, used = [], Counter()
    for row in ranked:
        if used[row["domain"]] >= caps[row["domain"]]:
            continue
        chosen.append(dict(row))
        used[row["domain"]] += 1
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError(
            f"domain-capped selector produced {len(chosen)} worlds, expected {n}; "
            f"pool/caps={dict(domain_counts)}/{caps}")
    chosen.sort(key=lambda row: (row["question_open_time"], row["event_id"]))
    for index, row in enumerate(chosen):
        row["selection_index_chronological"] = index
        row["split"] = "calibration" if index < 40 else ("validation" if index < 60 else "locked_test")
        start, end = _parse_ts(row["question_open_time"]), _parse_ts(row["resolution_time"])
        row["forecast_cutoffs"] = [_iso(start + fraction * (end - start))
                                   for fraction in CUTOFF_FRACTIONS]
    return chosen


def _market_history(token_id: str | None) -> list[dict]:
    if not token_id:
        return []
    query = urllib.parse.urlencode({"market": token_id, "interval": "max", "fidelity": 60})
    payload = _fetch_json(f"{CLOB}/prices-history?{query}") or {}
    return sorted((payload.get("history") or []), key=lambda point: point.get("t", 0))


def attach_market_snapshots(selected: list[dict]) -> dict:
    snapshots = {}
    for row in selected:
        history = _market_history(row.pop("yes_token_id", None))
        event = {}
        for cutoff in row["forecast_cutoffs"]:
            ts = _parse_ts(cutoff) or 0
            prior = [point for point in history if float(point.get("t", math.inf)) <= ts]
            if prior:
                point = prior[-1]
                event[cutoff] = {"bid": None, "ask": None, "midpoint": float(point["p"]),
                                 "last_trade": float(point["p"]), "volume": None, "spread": None,
                                 "source_timestamp": _iso(float(point["t"])),
                                 "requested_cutoff": cutoff,
                                 "staleness_s": round(ts - float(point["t"]), 3),
                                 "source": "polymarket_clob_prices_history"}
            else:
                event[cutoff] = {"unavailable": True, "reason": "no archived tick at or before cutoff"}
        snapshots[row["event_id"]] = event
    return snapshots


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _seal_split_stores(scorer_root: Path, selected: list[dict], resolutions: dict) -> dict:
    scorer_root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("calibration", "validation", "locked_test"):
        payload = {"schema_version": 1, "split": split,
                   "resolutions": {row["event_id"]: resolutions[row["event_id"]]
                                   for row in selected if row["split"] == split}}
        path = scorer_root / f"{split}_resolutions.json"
        _write_json(path, payload)
        path.chmod(0o600)
        hashes[split] = {"path_outside_forecast_mount": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "n_worlds": len(payload["resolutions"])}
    return hashes


def build(*, max_pages: int, scorer_root: Path) -> dict:
    source = fetch_source_events(max_pages=max_pages)
    candidates, resolutions = eligible_candidates(source)
    independent, clustering = collapse_independence_families(candidates)
    if len(independent) < N_MIN_POOL:
        raise RuntimeError(
            f"only {len(independent)} independent candidates after clustering; need at least {N_MIN_POOL}")
    selected = select_representative(independent)
    market = attach_market_snapshots(selected)
    selected_ids = {row["event_id"] for row in selected}
    # Keep the public candidate pool outcome-free and credential-free.
    public_pool = [{k: v for k, v in row.items() if k != "yes_token_id"} for row in independent]
    split_map = {row["event_id"]: row["split"] for row in selected}
    sealed = _seal_split_stores(scorer_root, selected, resolutions)
    frozen_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rules = {
        "model_release_floor": RELEASE_FLOOR, "construction_ceiling": CONSTRUCTION_CEILING,
        "minimum_open_seconds": MIN_OPEN_SECONDS, "minimum_volume_usdc": MIN_VOLUME_USDC,
        "minimum_independent_pool": N_MIN_POOL, "selected_worlds": N_SELECTED,
        "cutoff_fractions": CUTOFF_FRACTIONS, "selection_seed": SELECTION_SEED,
        "domain_cap_rule": "min(30, max(5, ceil(pool_share*100)+8)); never relaxed",
        "selection_uses_outcome_identity": False,
        "clustering": "source event plus normalized recurring-threshold family",
        "splits": "chronological earliest 40 calibration, next 20 validation, latest 40 locked_test",
    }
    _write_json(ROOT / "eligible_candidate_pool.json", {
        "schema_version": 1, "frozen_at": frozen_at, "rules": rules,
        "n_source_events_examined": len(source), "n_binary_eligible_before_family_collapse": len(candidates),
        "n_independent_eligible_worlds": len(public_pool), "worlds": public_pool,
    })
    _write_json(ROOT / "event_world_clustering_map.json", {
        "schema_version": 1, "frozen_at": frozen_at, "families": clustering,
    })
    _write_json(ROOT / "frozen_selection_manifest.json", {
        "schema_version": 1, "frozen_at": frozen_at, "rules": rules,
        "selected_event_ids": [row["event_id"] for row in selected],
        "selection_sha256": hashlib.sha256(
            json.dumps([row["event_id"] for row in selected]).encode()).hexdigest(),
        "domain_counts": Counter(row["domain"] for row in selected), "worlds": selected,
    })
    _write_json(ROOT / "representative_vault.json", {
        "schema_version": 1, "frozen_at": frozen_at, "n_worlds": len(selected),
        "n_required_forecasts": len(selected) * len(CUTOFF_FRACTIONS),
        "outcomes_present": False, "worlds": selected,
    })
    _write_json(ROOT / "temporal_split_manifest.json", {
        "schema_version": 1, "frozen_at": frozen_at,
        "counts": Counter(split_map.values()), "splits": split_map,
        "world_integrity": all(row["split"] == split_map[row["event_id"]] for row in selected),
    })
    _write_json(ROOT / "market_snapshots.json", {
        "schema_version": 1, "frozen_at": frozen_at,
        "rule": "last official archive tick at or before exact cutoff; missing remains missing",
        "snapshots": market,
    })
    report = {
        "schema_version": 1, "frozen_at": frozen_at, "source_events_examined": len(source),
        "binary_eligible_before_clustering": len(candidates), "independent_pool": len(public_pool),
        "selected_worlds": len(selected), "required_primary_forecasts": len(selected) * 4,
        "selected_domain_counts": Counter(row["domain"] for row in selected),
        "split_counts": Counter(split_map.values()), "sealed_resolution_stores": sealed,
        "selected_resolutions_read_by_forecaster": False,
        "selection_contains_exactly_100": len(selected_ids) == N_SELECTED,
    }
    _write_json(ROOT / "pool_build_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("--scorer-root", type=Path, required=True,
                        help="absolute directory outside the forecast mount")
    args = parser.parse_args()
    report = build(max_pages=args.max_pages, scorer_root=args.scorer_root.resolve())
    # The report exposes only counts and store hashes, never outcomes.
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
