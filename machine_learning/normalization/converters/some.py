"""SoMe (LivXue/Social-Media-Agents-Benchmark) — REAL crawled social behavior.

The HF repo mixes genuinely-observed human behavior with agent-benchmark scaffolding.
This converter normalizes ONLY the real human-behavior records and DROPS (with a counted,
logged tally — never silent) the scaffolding/reference layers.

KEEP (genuine human behavior):
  database/user_data/<uid>/<uid>.json = {"user": <profile>, "weibo": [<post>, ...]}
    profile: id, name, followers_count, friends_count, total_rcl_count, verified_reason,
             description, location, gender
    post:    id, text, created_time ("YYYY-MM-DD HH:MM"), source, region_name,
             reposts_count, comments_count, attitudes_count, pic_num, pictures, video
    -> a real, time-ordered post sequence for one platform user WITH crowd engagement.
  database/raw_data/*_sensitive.jsonl = flat multi-platform posts (content, user_id,
    like_count/comment_count/repost_count/view_count, create_time/post_publish_time, ...)
    -> genuine posts, grouped by user_id into the same sequence structure.

DROP + COUNT (not human behavior sequences):
  * report records (report_id/link/content, incl. the streamed `train` parquet): real
    crawled news/fact-check reference text, but NO actor / engagement / sequence -> cannot
    support any behavior task. Counted as dropped['report'].
  * raw_data posts flagged is_noise==1 / is_delete==1 -> dropped['noise'|'deleted'].
  * posts with empty text/content -> dropped['empty']; users with no usable post -> dropped['no_behavior'].

Emits:
  PREDICT_NEXT_ACTION       — for post k in a user's chronological sequence, predict the post
                              action from posts 0..k-1 (prior engagement excluded for
                              chronology safety). target.action_type="post".
  PREDICT_POPULATION_RESPONSE — for each post, the crowd's aggregate engagement
                              (reposts/comments/attitudes) as target.aggregate_metrics;
                              input.population_features = post text + author features.

Honesty notes:
  * The streamed sample (`train` split) is the REPORTS corpus only, so on that sample the
    converter emits 0 behavior records and drops every row (counted) — a genuine limitation
    documented here and reported by the acquisition. Real behavior requires the user_data /
    raw_data trees (storage-blocked, ~50GB, platform-ToS).
  * Prior posts' final engagement counts are measured after they were authored, so they are
    EXCLUDED from NEXT_ACTION history (only text/time/region/source retained) to avoid leakage.
  * EVAL-ONLY: crawled Weibo/32-platform content under platform ToS; no redistribution.
"""
from __future__ import annotations

import glob
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before

log = logging.getLogger(__name__)


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _is_report(rec: dict) -> bool:
    if "report_id" in rec:
        return True
    return ("content" in rec and "link" in rec and "weibo" not in rec
            and "user" not in rec and "user_id" not in rec)


def _normalize_userdata_post(w: dict) -> dict:
    """A user_data 'weibo' item -> common post dict."""
    return {
        "id": w.get("id"),
        "text": (w.get("text") or "").strip(),
        "created_time": w.get("created_time"),
        "source": w.get("source"),
        "region": w.get("region_name"),
        "reposts": _int(w.get("reposts_count")),
        "comments": _int(w.get("comments_count")),
        "attitudes": _int(w.get("attitudes_count")),
        "pic_num": _int(w.get("pic_num")) or 0,
        "has_video": bool(w.get("video")),
    }


def _normalize_rawdata_post(p: dict) -> dict:
    """A raw_data jsonl post -> common post dict."""
    return {
        "id": p.get("post_id") or p.get("unique_id"),
        "text": (p.get("content") or p.get("title") or "").strip(),
        "created_time": p.get("create_time") or p.get("post_publish_time") or p.get("push_time"),
        "source": p.get("platform_name") or p.get("platform"),
        "region": p.get("public_location") or p.get("poi_location"),
        "reposts": _int(p.get("repost_count")),
        "comments": _int(p.get("comment_count")),
        "attitudes": _int(p.get("like_count")),
        "views": _int(p.get("view_count")),
        "pic_num": len(p.get("images") or []),
        "has_video": bool(p.get("play_url") or p.get("video_online_url")),
    }


class Converter(BaseConverter):
    DATASET_ID = "some"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "some"
    DOC = {
        "dataset_id": "some",
        "original_fields": [
            {"name": "user", "meaning": "platform-user profile (id, followers/friends counts, gender, location, verified_reason, description)"},
            {"name": "weibo[]", "meaning": "the user's posts: id, text, created_time, source, region_name, reposts/comments/attitudes_count, pic_num, video"},
            {"name": "raw_data post", "meaning": "flat multi-platform post: content, user_id, like/comment/repost/view_count, create_time, is_noise, is_delete"},
            {"name": "report (report_id, link, content)", "meaning": "crawled news/fact-check reference text (DROPPED: no actor/engagement/sequence)"},
        ],
        "canonical_mapping": [
            {"source_field": "weibo[k].text", "canonical_path": "payload.target.action_content.text (NEXT_ACTION) / payload.input.population_features.post_text (POPULATION_RESPONSE)"},
            {"source_field": "weibo[k].{reposts,comments,attitudes}_count", "canonical_path": "payload.target.aggregate_metrics (POPULATION_RESPONSE)"},
            {"source_field": "weibo[0..k-1] (text/time/region only)", "canonical_path": "payload.input.history / context.known_history (engagement excluded for chronology)"},
            {"source_field": "user.id", "canonical_path": "decision_unit.actor_id/population_id + episode (pseudonymized); split_unit=platform_user"},
            {"source_field": "user.{followers,friends}_count, gender, location", "canonical_path": "context.actor_profile / population_features.author"},
            {"source_field": "report_id/link/content", "canonical_path": "DROPPED (counted in dropped['report'])"},
        ],
        "tasks_produced": ["PREDICT_NEXT_ACTION", "PREDICT_POPULATION_RESPONSE"],
        "unavailable_fields": [
            "reply/interaction targets (only aggregate engagement counts, not per-responder edges) -> PREDICT_TRAJECTORY_CONTINUATION not produced from posts alone",
            "precise post timestamps below minute granularity",
            "stable cross-platform identity (per-source user ids)",
        ],
        "chronology_rules": "Posts are sorted by created_time; for the decision at post k only posts 0..k-1 appear in history, and their engagement counts (measured post-hoc) are excluded. A post's own engagement is the POPULATION_RESPONSE target, never an input.",
        "split_key": "platform_user (pseudonymized user id) -> episode + population_id; a user's records stay together",
        "leakage_risks": [
            "a post's crowd engagement is a downstream outcome; it is kept out of NEXT_ACTION inputs and prior-post history",
            "the same real event may be posted by many users; splitting by platform_user isolates authors",
        ],
        "known_limitations": [
            "the streamed `train` split is the REPORTS corpus (report_id/link/content) with no actor/engagement/sequence -> all dropped-and-counted; behavior tasks require the user_data/raw_data trees (storage-blocked)",
            "engagement counts are crawl-time snapshots, not final",
            "raw_data posts flagged is_noise/is_delete are dropped and counted",
        ],
        "license_implications": "Crawled Weibo + 32-platform content under platform ToS (the Apache repo header does not clear it). Evaluation-only, cross-dataset transfer test; no redistribution.",
        "training_suitability": "eval_only",
        "assumptions": [
            "user_data files are {user, weibo[]}; created_time strings 'YYYY-MM-DD HH:MM' sort chronologically",
            "report rows carry report_id or (content+link) without a user/weibo -> scaffolding/reference, dropped",
        ],
    }

    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        dropped: Counter = Counter()
        users: list[dict] = []                     # [{'user':..., 'weibo':[...]}]
        raw_by_uid: dict = defaultdict(list)       # uid -> [common_post]

        # ---- 1. parquet shards (streamed sample = reports) --------------------------------
        for f in sorted(glob.glob(str(raw_dir / "**" / "stream_shard_*.parquet"), recursive=True)):
            if ".cache" in f:
                continue
            import pyarrow.parquet as pq
            for row in pq.read_table(f).to_pylist():
                if _is_report(row):
                    dropped["report"] += 1
                else:
                    dropped["unrecognized_parquet_row"] += 1

        # ---- 2. json files: user_data dicts, raw_data lists, or fixture lists --------------
        for f in sorted(glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)):
            if ".cache" in f or "dataset_info" in f:
                continue
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and "weibo" in data:
                users.append(data)
            elif isinstance(data, list):
                for el in data:
                    if not isinstance(el, dict):
                        continue
                    if "weibo" in el and "user" in el:
                        users.append(el)
                    elif _is_report(el):
                        dropped["report"] += 1
                    elif el.get("user_id") is not None and ("content" in el or "title" in el):
                        self._ingest_raw(el, raw_by_uid, dropped)
                    else:
                        dropped["unrecognized_record"] += 1

        # ---- 3. raw_data jsonl (flat multi-platform posts) --------------------------------
        for f in sorted(glob.glob(str(raw_dir / "**" / "raw_data" / "*.jsonl"), recursive=True)):
            if ".cache" in f:
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        p = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(p, dict):
                        self._ingest_raw(p, raw_by_uid, dropped)

        emitted_any = False

        # ---- emit user_data behavior sequences --------------------------------------------
        for u in users:
            profile = u.get("user") or {}
            posts = [_normalize_userdata_post(w) for w in (u.get("weibo") or [])]
            posts = [p for p in posts if p["text"]]
            if not posts:
                dropped["no_behavior"] += 1
                continue
            uid = str(profile.get("id") if profile.get("id") is not None else id(u))
            for rec in self._sequence_records(uid, profile, posts, source="weibo"):
                emitted_any = True
                yield rec

        # ---- emit raw_data behavior sequences (grouped by user) ---------------------------
        for uid, posts in raw_by_uid.items():
            posts = [p for p in posts if p["text"]]
            if not posts:
                dropped["no_behavior"] += 1
                continue
            profile = {"id": uid}
            for rec in self._sequence_records(str(uid), profile, posts, source="rawdata"):
                emitted_any = True
                yield rec

        self.dropped = dict(dropped)
        if dropped:
            log.warning("[some] dropped %d non-behavior/scaffolding records: %s",
                        sum(dropped.values()), self.dropped)
        if not emitted_any:
            log.warning("[some] no genuine human-behavior records in %s "
                        "(sample is reference/scaffolding only); dropped=%s", raw_dir, self.dropped)

    def _ingest_raw(self, p: dict, raw_by_uid: dict, dropped: Counter) -> None:
        if _int(p.get("is_delete")) == 1:
            dropped["deleted"] += 1
            return
        if _int(p.get("is_noise")) == 1:
            dropped["noise"] += 1
            return
        post = _normalize_rawdata_post(p)
        if not post["text"]:
            dropped["empty"] += 1
            return
        uid = p.get("user_id")
        if uid is None:
            dropped["no_user"] += 1
            return
        raw_by_uid[str(uid)].append(post)

    def _author_features(self, profile: dict) -> dict:
        return {
            "followers_count": _int(profile.get("followers_count")),
            "friends_count": _int(profile.get("friends_count")),
            "total_rcl_count": _int(profile.get("total_rcl_count")),
            "gender": profile.get("gender"),
            "location": profile.get("location"),
            "verified_reason": profile.get("verified_reason"),
            "description": profile.get("description"),
        }

    def _sequence_records(self, uid: str, profile: dict, posts: list[dict],
                          source: str) -> Iterator[dict]:
        posts = sorted(posts, key=lambda p: (p.get("created_time") or ""))
        author = self.pseudonym("actor", uid)
        author_feats = self._author_features(profile)
        episode_id = f"some-{source}-user-{author}"

        # leakage-safe history events: text/time/region only (NO engagement)
        events = [
            history_event(k, author, "action", action_type="post", text=p["text"],
                          action_content={"created_time": p["created_time"], "region": p["region"],
                                          "source": p["source"]},
                          meta={"source": source})
            for k, p in enumerate(posts)
        ]

        for k, p in enumerate(posts):
            hist = history_before(events, k)
            loc = {"files": [f"user_data/{uid}.json" if source == "weibo" else f"raw_data/{uid}"],
                   "indices": [k], "ids": [str(p.get("id"))]}

            # ---- NEXT_ACTION -------------------------------------------------------------
            ctx = {
                "actor_profile": author_feats,
                "known_history": hist,
                "current_observation": {"kind": "state", "meta": {"platform": source, "region": p["region"]}},
                "world_state": {"platform": source},
                "available_actions": None,
                "language": "zh",
            }
            action_content = {"text": p["text"], "created_time": p["created_time"],
                              "region": p["region"], "source": p["source"],
                              "pic_num": p["pic_num"], "has_video": p["has_video"]}
            yield self.make(
                task_type="PREDICT_NEXT_ACTION",
                payload={"input": {"history": hist,
                                   "observation": ctx["current_observation"],
                                   "available_actions": None},
                         "target": {"action_type": "post", "acted": True,
                                    "action_content": action_content}},
                episode_id=episode_id,
                sequence_index=k,
                cutoff_sequence_index=k,
                participant_ids=[author],
                actor_id=author,
                actor_role="social_media_user",
                population_id=author,
                experiment_id=source,
                context=ctx,
                source_language="zh",
                raw_locator=loc,
                transformation_steps=[
                    "load real post sequence",
                    "sort by created_time",
                    f"cutoff before post {k}; prior-post engagement excluded",
                    "target = the post the user authored",
                ],
                data_quality={
                    "missing_fields": ["precise_timestamp", "cross_platform_identity"],
                    "warnings": ["prior-post engagement withheld from history for chronology safety"],
                    "confidence": "high",
                    "chronology_verified": True,
                    "target_verified": True,
                    "possible_leakage": False,
                    "license_verified": True,
                },
            )

            # ---- POPULATION_RESPONSE (crowd engagement to this post) ---------------------
            reposts, comments, attitudes = p.get("reposts"), p.get("comments"), p.get("attitudes")
            total = sum(x for x in (reposts, comments, attitudes) if isinstance(x, int))
            metrics = {"reposts_count": reposts, "comments_count": comments,
                       "attitudes_count": attitudes, "total_engagement": total,
                       "author_followers_count": author_feats["followers_count"]}
            if p.get("views") is not None:
                metrics["view_count"] = p["views"]
            missing_eng = [f for f, v in (("reposts_count", reposts), ("comments_count", comments),
                                          ("attitudes_count", attitudes)) if v is None]
            yield self.make(
                task_type="PREDICT_POPULATION_RESPONSE",
                payload={"input": {"population_features": {"post_text": p["text"],
                                                           "author": author_feats,
                                                           "created_time": p["created_time"],
                                                           "region": p["region"],
                                                           "source": p["source"],
                                                           "platform": source},
                                   "historical_context": {"n_prior_posts": k}},
                         "target": {"aggregate_metrics": metrics}},
                episode_id=episode_id,
                sequence_index=k,
                cutoff_sequence_index=k,
                participant_ids=[author],
                actor_id=author,
                actor_role="social_media_user",
                population_id=author,
                experiment_id=source,
                context={"actor_profile": author_feats,
                         "current_observation": {"kind": "post", "text": p["text"]},
                         "world_state": {"platform": source}, "available_actions": None,
                         "language": "zh"},
                source_language="zh",
                raw_locator=loc,
                transformation_steps=[
                    "load real post",
                    "target = crowd engagement (reposts/comments/attitudes) for this post",
                ],
                data_quality={
                    "missing_fields": missing_eng,
                    "warnings": ["engagement is a crawl-time snapshot, not final"],
                    "confidence": "high",
                    "chronology_verified": True,
                    "target_verified": True,
                    "possible_leakage": False,
                    "license_verified": True,
                },
            )
