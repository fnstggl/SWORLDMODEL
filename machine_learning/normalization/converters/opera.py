"""OPeRA (NEU-HAI/OPeRA) — real Amazon online-shopping sessions captured via a browser
plugin + questionnaire: Observation, Persona, Rationale, Action (+ screenshot refs).

Real source (verified from the HF parquet configs OPeRA_filtered/{action,user,session}):
  action config (one row per action within a session)::
      session_id     : "<user_uuid>_<start_iso>_<end_iso>"   (user id is the prefix)
      action_id      : uuid
      timestamp      : ISO string e.g. "2025-04-03T01:06:31.719Z"
      action_type    : "click" | "input" | "scroll" | "terminate" (open set)
      click_type     : "product_link" | "product_option" | "purchase" | "review" | "search"
                       | "cart_side_bar" | ... | null
      semantic_id    : semantic path of the targeted element (the concrete action target)
      mouse_position, element_meta, window_size : JSON strings (element/viewport detail)
      url            : masked Amazon URL of the page (the OBSERVED page)
      page_meta      : JSON string — structured page observation (search_term/cart_items/...)
      simplified_html: the full simplified DOM (very large; kept as a length reference only)
      rationale      : participant's SELF-REPORTED reason for the action (often "", short intent
                       when present e.g. "look more" / "buy it")
      products       : JSON list of products in view (asin/title/price/options)
      input_text     : text typed for "input" actions (e.g. a search query)
      image          : opaque screenshot filename reference e.g. "004067.jpg" (NOT pixels)
  user config: {user_id, survey (JSON persona), raw_survey, interview_transcript,
                interview_transcript_processed}
  session config: {session_id, user_id, action_count}  (derivable; not required)

Emits (one session == one episode; actor_id = pseudonymized user id, stable across the
user's sessions; split by platform_user so users AND their sessions are held out together):
  PREDICT_NEXT_ACTION             — Observation(+Persona+Rationale) -> the shopping action.
  PREDICT_TIME_TO_ACTION          — inter-action seconds within the session (sessions are
                                    complete/terminate explicitly, so no right-censoring).
  PREDICT_TRAJECTORY_CONTINUATION — from a mid-session cutoff, the remaining (capped) actions.
  PREDICT_FINAL_OUTCOME           — session outcome {purchased, action_count, ended_with_terminate},
                                    cut off BEFORE the purchase/terminate action.

Honesty:
  * rationale is placed in context.private_state_before per OPeRA's O+P+R->A design, BUT it is a
    SELF-REPORT — every record with a non-empty rationale carries the warning "participant rationale
    is a self-report, not an infallible/complete private state" and confidence is NOT "high".
  * screenshots are kept ONLY as opaque references in meta.image_ref — image content is never
    fabricated. The full DOM is kept as a length reference, not embedded.
  * FINAL_OUTCOME.purchased is DERIVED from the presence of a purchase-type click (marked as an
    inferred_field with medium confidence), not a fabricated ground-truth label.

License: CC-BY-4.0 (training + commercial use with attribution).
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..base import Converter as BaseConverter
from ..common.dialogue import history_event, history_before

CAP = 200  # documented per-session bound on history/continuation length
_SELF_REPORT_WARNING = ("participant rationale is a self-report, not an infallible/complete "
                        "private state")


def _jparse(s):
    """Parse a JSON string field; return the parsed value, or the raw value unchanged."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def _parse_iso(s):
    if not isinstance(s, str) or not s:
        return None
    t = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s.rstrip("Z"), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _user_of(session_id: str) -> str:
    """user_id is the session_id prefix before the first '_<iso-timestamp>'."""
    return (session_id or "").split("_")[0] if session_id else ""


def _load_opera(raw_dir: Path) -> tuple[list[dict], dict]:
    """Return (action_rows, users_by_id). Loads from the real multi-config parquet layout OR
    JSON fixtures. Action rows are classified by the 'rationale'/'action_id' signature; user
    rows by 'survey'; the images config and session config are ignored."""
    actions: list[dict] = []
    users: dict[str, dict] = {}

    def _add_user(r: dict):
        uid = r.get("user_id")
        if uid:
            users[str(uid)] = r

    pq_files = [f for f in glob.glob(str(raw_dir / "**" / "*.parquet"), recursive=True)
                if ".cache" not in f and "/images/" not in f.replace("\\", "/")]
    if pq_files:
        import pyarrow.parquet as pq
        for f in sorted(pq_files):
            cols = set(pq.ParquetFile(f).schema.names)
            if "rationale" in cols and "action_id" in cols:
                actions.extend(pq.read_table(f).to_pylist())
            elif "survey" in cols or "interview_transcript" in cols:
                for r in pq.read_table(f).to_pylist():
                    _add_user(r)
        return actions, users

    json_files = [f for f in glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
                  if ".cache" not in f and "dataset_info" not in f]
    if not json_files:
        raise FileNotFoundError(f"no OPeRA parquet/json found under {raw_dir}")
    for f in sorted(json_files):
        data = json.loads(Path(f).read_text())
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            if not isinstance(r, dict):
                continue
            if "action_id" in r or ("action_type" in r and "session_id" in r and "survey" not in r):
                actions.append(r)
            elif "survey" in r or "interview_transcript" in r:
                _add_user(r)
    return actions, users


class Converter(BaseConverter):
    DATASET_ID = "opera"
    VERSION = "1.0.0"
    FIXTURE_SUBDIR = "opera"
    DOC = {
        "dataset_id": "opera",
        "original_fields": [
            {"name": "session_id", "meaning": "<user_uuid>_<start_iso>_<end_iso>; user id is the prefix"},
            {"name": "action_id", "meaning": "unique action id within the session"},
            {"name": "timestamp", "meaning": "ISO instant of the action, e.g. 2025-04-03T01:06:31.719Z"},
            {"name": "action_type", "meaning": "click | input | scroll | terminate (open set)"},
            {"name": "click_type", "meaning": "product_link/product_option/purchase/review/search/... | null"},
            {"name": "semantic_id", "meaning": "semantic path of the targeted element (concrete action target)"},
            {"name": "url", "meaning": "masked Amazon URL of the observed page"},
            {"name": "page_meta", "meaning": "structured page observation (search_term/cart_items/...) as JSON"},
            {"name": "simplified_html", "meaning": "full simplified DOM (large; kept as a length reference only)"},
            {"name": "rationale", "meaning": "participant self-reported reason for the action (often empty)"},
            {"name": "products", "meaning": "JSON list of in-view products (asin/title/price/options)"},
            {"name": "input_text", "meaning": "typed text for input actions (e.g. a search query)"},
            {"name": "image", "meaning": "opaque screenshot filename reference (NOT pixels)"},
            {"name": "user.survey", "meaning": "persona: JSON demographic + shopping profile"},
            {"name": "user.interview_transcript", "meaning": "post-hoc interview (often empty)"},
        ],
        "canonical_mapping": [
            {"source_field": "user_id (from session_id)", "canonical_path": "decision_unit.actor_id (pseudonymized, persistent)"},
            {"source_field": "session_id", "canonical_path": "episode.session_id (pseudonymized) / episode_id"},
            {"source_field": "url,page_meta,window_size,simplified_html", "canonical_path": "context.current_observation"},
            {"source_field": "action_type", "canonical_path": "payload.target.action_type"},
            {"source_field": "click_type,semantic_id,input_text,element_meta,products", "canonical_path": "payload.target.action_content"},
            {"source_field": "rationale", "canonical_path": "context.private_state_before.self_reported_rationale (self-report; warned)"},
            {"source_field": "survey", "canonical_path": "context.actor_profile.survey"},
            {"source_field": "image", "canonical_path": "meta.image_ref (opaque; content never fabricated)"},
            {"source_field": "timestamp", "canonical_path": "history_event.t / cutoff_time; diffs -> TIME_TO_ACTION"},
            {"source_field": "click_type=='purchase'", "canonical_path": "payload.target.outcome.purchased (derived)"},
        ],
        "tasks_produced": ["PREDICT_NEXT_ACTION", "PREDICT_TIME_TO_ACTION",
                           "PREDICT_TRAJECTORY_CONTINUATION", "PREDICT_FINAL_OUTCOME"],
        "unavailable_fields": [
            "verified private state (rationale is a self-report; often empty)",
            "screenshot pixel content (kept as an opaque filename reference only)",
            "explicit enumerated available-action set (defined by the live DOM; not enumerated -> None)",
            "right-censored time-to-action (sessions terminate explicitly; no censoring)",
        ],
        "chronology_rules": "Actions are ordered by timestamp within a session. For the decision at step k, "
                            "the observation is the page at step k (which the participant sees BEFORE acting) plus "
                            "history 0..k-1; the action at k is ONLY in payload.target. FINAL_OUTCOME cuts off "
                            "before the first purchase/terminate action.",
        "split_key": "platform_user (actor_id = pseudonymized user id; a user's sessions are held out together)",
        "leakage_risks": [
            "rationale is the participant's concurrent stated intent; per the OPeRA O+P+R->A design it is placed in "
            "private_state_before as a conditioning input, flagged as a self-report and never treated as ground truth "
            "(confidence != high whenever it is non-empty)",
            "the step-k page observation precedes the action and is safe; the concrete action target (semantic_id) is "
            "target-side only",
        ],
        "known_limitations": [
            "self-reported rationale is frequently empty and, when present, terse; a self-report, not a full private state",
            "the full simplified DOM is stored as a length reference (present + char count), not embedded, to bound size",
            "FINAL_OUTCOME.purchased is derived from the presence of a purchase-type click (proxy, medium confidence)",
            f"history/continuation bounded to CAP={CAP} actions per session (documented)",
        ],
        "license_implications": "CC-BY-4.0: training + commercial use permitted with attribution.",
        "training_suitability": "train",
        "assumptions": [
            "user id is the session_id prefix before the first ISO timestamp",
            "click_type=='purchase' marks a purchase action; last action_type=='terminate' marks a normal session end",
            "timestamps are ISO-8601 with a trailing Z",
        ],
    }

    # ---- helpers -----------------------------------------------------------------------
    def _persona(self, user_row: dict | None) -> dict:
        if not user_row:
            return {}
        prof = {"survey": _jparse(user_row.get("survey")) or {}}
        it = user_row.get("interview_transcript")
        if it:
            prof["interview_transcript"] = it
        return prof

    def _observation(self, a: dict) -> dict:
        html = a.get("simplified_html") or ""
        return {"url": a.get("url"), "page_meta": _jparse(a.get("page_meta")),
                "window_size": _jparse(a.get("window_size")),
                "products": _jparse(a.get("products")),
                "dom_html_ref": {"present": bool(html), "chars": len(html)},
                "screenshot_ref": a.get("image"), "kind": "web_page"}

    def _action_content(self, a: dict) -> dict:
        return {"click_type": a.get("click_type"), "semantic_id": a.get("semantic_id"),
                "input_text": a.get("input_text"), "element_meta": _jparse(a.get("element_meta")),
                "products": _jparse(a.get("products")), "url": a.get("url"),
                "image_ref": a.get("image")}

    def _hist_event(self, a: dict, index: int, actor: str) -> dict:
        return history_event(
            index, actor, "action", t=a.get("timestamp"), action_type=a.get("action_type"),
            action_content={"click_type": a.get("click_type"), "semantic_id": a.get("semantic_id"),
                            "input_text": a.get("input_text")},
            meta={"url": a.get("url"), "image_ref": a.get("image")})

    # ---- driver ------------------------------------------------------------------------
    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        actions, users = _load_opera(raw_dir)
        # group actions by session, preserving original index for the raw locator
        sessions: dict = {}
        for gi, a in enumerate(actions):
            sessions.setdefault(a.get("session_id"), []).append((gi, a))
        for session_id in sessions:
            yield from self._one_session(session_id, sessions[session_id], users)

    def _one_session(self, session_id, indexed_actions, users) -> Iterator[dict]:
        # order by timestamp (stable on original order)
        indexed_actions = sorted(indexed_actions, key=lambda p: (_parse_iso(p[1].get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc), p[0]))
        acts = [a for _, a in indexed_actions]
        orig = [gi for gi, _ in indexed_actions]
        n = len(acts)
        if n == 0:
            return

        user_id = _user_of(session_id)
        actor = self.pseudonym("actor", user_id)
        session_pseud = self.pseudonym("group", session_id)
        episode_id = f"opera-{session_pseud}"
        persona = self._persona(users.get(str(user_id)))
        reprs = [self._hist_event(a, k, actor) for k, a in enumerate(acts)]

        def loc(idxs):
            return {"files": ["OPeRA/action"], "indices": list(idxs), "ids": [episode_id]}

        # ------------------------------------------------ NEXT_ACTION ------------------
        for k in range(min(n, CAP)):
            a = acts[k]
            hist = history_before(reprs, k)
            obs = self._observation(a)
            rationale = (a.get("rationale") or "").strip()
            private = {"self_reported_rationale": rationale} if rationale else {}
            ctx = {"actor_profile": persona, "private_state_before": private,
                   "known_history": hist, "current_observation": obs,
                   "world_state": {"site": "amazon.com"}, "available_actions": None, "language": "en"}
            payload = {
                "input": {"history": hist, "observation": obs, "available_actions": None},
                "target": {"action_type": a.get("action_type") or "<UNKNOWN>", "acted": True,
                           "action_content": self._action_content(a)},
            }
            missing = [] if persona else ["actor_profile"]
            warnings = []
            confidence = "high"
            if rationale:
                warnings.append(_SELF_REPORT_WARNING)
                warnings.append("private_state_before.self_reported_rationale is the participant's concurrent "
                                "stated intent (OPeRA O+P+R->A); may partially reveal action intent")
                confidence = "medium"
            else:
                missing.append("rationale")
            yield self.make(
                task_type="PREDICT_NEXT_ACTION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, session_id=session_pseud,
                cutoff_time=a.get("timestamp"), actor_id=actor, actor_role="shopper",
                participant_ids=[actor], group_id=actor, persistent_identity_available=True,
                context=ctx, raw_locator=loc([orig[k]]),
                transformation_steps=["load OPeRA actions+persona", "group+order by session/timestamp",
                                      f"cutoff before action {k}", "page -> observation; rationale -> private_state"],
                data_quality={"missing_fields": missing, "warnings": warnings, "confidence": confidence,
                              "chronology_verified": True, "target_verified": True,
                              "possible_leakage": False, "license_verified": True,
                              "inferred_fields": ["sequence_index"]})

        # ------------------------------------------------ TIME_TO_ACTION ---------------
        for k in range(1, min(n, CAP + 1)):
            prev_t, cur_t = _parse_iso(acts[k - 1].get("timestamp")), _parse_iso(acts[k].get("timestamp"))
            if prev_t is None or cur_t is None:
                continue
            delta = (cur_t - prev_t).total_seconds()
            if delta < 0:
                continue
            hist = history_before(reprs, k)
            payload = {
                "input": {"history": hist, "current_time": acts[k - 1].get("timestamp")},
                "target": {"acted": True, "time_to_action_seconds": delta,
                           "censoring": {"censored": False, "observation_window_seconds": None,
                                         "reason": "next action observed within the (complete) session"}},
            }
            yield self.make(
                task_type="PREDICT_TIME_TO_ACTION", payload=payload, episode_id=episode_id,
                sequence_index=k, cutoff_sequence_index=k, session_id=session_pseud,
                cutoff_time=acts[k - 1].get("timestamp"), actor_id=actor, actor_role="shopper",
                participant_ids=[actor], group_id=actor, persistent_identity_available=True,
                context={"actor_profile": persona, "known_history": hist,
                         "current_observation": self._observation(acts[k - 1]),
                         "world_state": {"site": "amazon.com"}, "available_actions": None, "language": "en"},
                raw_locator=loc([orig[k - 1], orig[k]]),
                transformation_steps=["order session by timestamp", f"gap ts[{k}]-ts[{k-1}] seconds"],
                data_quality={"missing_fields": [], "confidence": "high",
                              "chronology_verified": True, "target_verified": True,
                              "possible_leakage": False, "license_verified": True,
                              "inferred_fields": ["sequence_index"]})

        # ------------------------------------------------ TRAJECTORY_CONTINUATION ------
        if n >= 2:
            cutoff = max(1, n // 2)
            hist = reprs[:cutoff]
            cont = reprs[cutoff:cutoff + CAP]
            reaches_end = cutoff + CAP >= n
            payload = {"input": {"history": hist, "horizon": len(cont)},
                       "target": {"continuation": cont}}
            yield self.make(
                task_type="PREDICT_TRAJECTORY_CONTINUATION", payload=payload, episode_id=episode_id,
                sequence_index=cutoff, cutoff_sequence_index=cutoff, session_id=session_pseud,
                cutoff_time=acts[cutoff - 1].get("timestamp"), actor_id=actor, actor_role="shopper",
                participant_ids=[actor], group_id=actor, persistent_identity_available=True,
                context={"actor_profile": persona, "known_history": hist,
                         "current_observation": self._observation(acts[cutoff - 1]),
                         "world_state": {"site": "amazon.com"}, "available_actions": None, "language": "en"},
                raw_locator=loc(orig),
                transformation_steps=["order session by timestamp", f"cutoff after {cutoff} actions",
                                      "remaining actions as continuation"],
                data_quality={"missing_fields": [], "confidence": "high",
                              "warnings": ([] if reaches_end else [f"continuation capped to {CAP} actions"]),
                              "chronology_verified": True, "target_verified": True,
                              "possible_leakage": False, "license_verified": True,
                              "inferred_fields": ["sequence_index"]})

        # ------------------------------------------------ FINAL_OUTCOME ----------------
        purchase_idx = next((i for i, a in enumerate(acts) if a.get("click_type") == "purchase"), None)
        terminate_idx = next((i for i, a in enumerate(acts) if a.get("action_type") == "terminate"), None)
        purchased = purchase_idx is not None
        if purchase_idx is not None:
            cut = purchase_idx
        elif terminate_idx is not None:
            cut = terminate_idx
        else:
            cut = n
        hist = reprs[:cut][:CAP]
        outcome = {"purchased": purchased, "action_count": n,
                   "ended_with_terminate": (acts[-1].get("action_type") == "terminate")}
        payload = {"input": {"history": hist, "state": {"site": "amazon.com"}},
                   "target": {"outcome": outcome, "outcome_type": "shopping_session_result"}}
        yield self.make(
            task_type="PREDICT_FINAL_OUTCOME", payload=payload, episode_id=episode_id,
            sequence_index=cut, cutoff_sequence_index=cut, session_id=session_pseud,
            cutoff_time=(acts[cut - 1].get("timestamp") if cut else None),
            actor_id=actor, actor_role="shopper", participant_ids=[actor], group_id=actor,
            persistent_identity_available=True,
            context={"actor_profile": persona, "known_history": hist,
                     "world_state": {"site": "amazon.com"}, "available_actions": None, "language": "en"},
            raw_locator=loc(orig),
            transformation_steps=["order session by timestamp", "cutoff before purchase/terminate",
                                  "derive session outcome (purchased/action_count/terminated)"],
            data_quality={"missing_fields": [], "confidence": "medium",
                          "inferred_fields": ["purchased (derived from a purchase-type click)"],
                          "warnings": ["'purchased' is derived from the presence of a purchase-type click, "
                                       "a proxy for a completed purchase"],
                          "chronology_verified": True, "target_verified": True,
                          "possible_leakage": False, "license_verified": True})
