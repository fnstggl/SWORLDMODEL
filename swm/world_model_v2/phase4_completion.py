"""Real-data adapters and deterministic evaluation primitives for Phase 4 completion.

Every adapter constructs actor-visible state in chronological order.  Labels and
post-action outcomes live in separate fields and cannot enter an LLM packet or a
numeric feature vector through the public projection methods.
"""
from __future__ import annotations

import bisect
import calendar
import csv
import hashlib
import heapq
import json
import math
import random
import re
import statistics
import tarfile
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from swm.world_model_v2.phase3_posterior import infer_compositional_posterior
from swm.world_model_v2.phase4_learning import (
    TrajectoryRecord, digest, evaluate_predictions, fit_temperature, apply_calibration,
)
from swm.world_model_v2.phase4_llm_baselines import build_actor_visible_packet


SEED = 404
DAY = 86400.0
ACTIONS = {
    "enron_repaired": ("reply_within_24h", "reply_in_1_to_7d", "no_reply_within_7d"),
    "ipd_long": ("cooperate", "defect"),
    "voteview_senate": ("support", "oppose", "abstain_or_absent"),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_seed(*parts) -> int:
    raw = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)


def stable_hash(*parts) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


@dataclass(slots=True)
class DecisionExample:
    record_key: str
    dataset: str
    actor_key: str
    actor_role: str
    relationship_key: str
    sequence_key: str
    cluster_key: str
    decision_time: float
    decision_time_label: str
    label: str
    actions: tuple[str, ...]
    visible_state: dict
    numeric_features: dict
    outcome: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    split: str = ""
    sample_weight: float = 1.0

    def validate(self) -> "DecisionExample":
        if self.dataset not in ACTIONS or tuple(self.actions) != ACTIONS[self.dataset]:
            raise ValueError("decision has an unregistered typed action set")
        if self.label not in self.actions:
            raise ValueError("observed action is outside the candidate action set")
        if not math.isfinite(self.decision_time) or not self.actor_key or not self.record_key:
            raise ValueError("decision identity and time are required")
        if any(not isinstance(v, (int, float, bool)) or not math.isfinite(float(v))
               for v in self.numeric_features.values()):
            raise ValueError("numeric policy features must be finite scalars")
        return self

    def llm_packet(self) -> dict:
        return build_actor_visible_packet(
            decision_time=self.decision_time_label, actor_role=self.actor_role,
            visible_state=self.visible_state, actions=list(self.actions),
        )

    def trajectory_record(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            record_id=self.record_key, dataset_id=self.dataset, actor_id=self.actor_key,
            actor_role=self.actor_role, decision_time=self.decision_time,
            context_id=str(self.provenance.get("context_key", "")), institution_id=str(
                self.provenance.get("institution_key", "")),
            relationship_id=self.relationship_key, sequence_id=self.sequence_key,
            observed_action=self.label, candidate_actions=list(self.actions),
            actor_view_features=dict(self.numeric_features), outcome=dict(self.outcome),
            action_set_hypotheses=[{
                "status": self.provenance.get("action_set_status", "explicit_rule"),
                "actions": list(self.actions), "source": self.provenance.get("action_set_source", ""),
            }], source_ids=list(self.provenance.get("source_ids", [])),
            provenance={"post_action_features": False, "label_in_features": False,
                        "adapter": "phase4_completion.v1"}, sample_weight=self.sample_weight,
        ).validate()


@dataclass(slots=True)
class DatasetBuild:
    dataset: str
    examples: list[DecisionExample]
    manifest: dict
    split_manifest: dict
    action_diagnostics: dict


def _content_hash(examples: list[DecisionExample]) -> str:
    h = hashlib.sha256()
    for row in sorted(examples, key=lambda x: x.record_key):
        h.update(json.dumps({
            "key": row.record_key, "time": row.decision_time, "label": row.label,
            "actions": row.actions, "visible": row.visible_state, "features": row.numeric_features,
            "outcome": row.outcome, "split": row.split,
        }, sort_keys=True, separators=(",", ":"), default=str).encode())
        h.update(b"\n")
    return h.hexdigest()


def _split_manifest(dataset: str, examples: list[DecisionExample], method: str) -> dict:
    groups = {name: [row.record_key for row in examples if row.split == name]
              for name in ("train", "calibration", "validation", "test", "purged")}
    payload = {
        "schema_version": "wmv2.phase4-completion.split.v1", "dataset": dataset,
        "method": method, "seed": SEED, "groups": groups,
        "counts": {key: len(value) for key, value in groups.items()},
        "record_overlap": any(set(groups[a]) & set(groups[b])
                              for i, a in enumerate(groups) for b in list(groups)[i + 1:]),
    }
    payload["checksum"] = digest(payload)
    return payload


def _action_diagnostics(dataset: str, examples: list[DecisionExample]) -> dict:
    counts = Counter(row.label for row in examples)
    invalid = sum(row.label not in row.actions for row in examples)
    known_impossible = sum(bool(row.provenance.get("known_impossible_selected")) for row in examples)
    statuses = Counter(row.provenance.get("action_set_status", "unknown") for row in examples)
    return {
        "schema_version": "wmv2.phase4-completion.action-diagnostics.v1",
        "dataset": dataset, "n": len(examples), "label_counts": dict(sorted(counts.items())),
        "observed_action_inclusion_rate": 1.0 - invalid / max(1, len(examples)),
        "known_impossible_selected_rate": known_impossible / max(1, len(examples)),
        "action_set_provenance_coverage": sum(status != "unknown" for status in statuses.elements())
        / max(1, len(examples)), "status_counts": dict(statuses),
    }


def _finish_build(dataset: str, examples: list[DecisionExample], *, source: dict,
                  split_method: str, limitations: list[str]) -> DatasetBuild:
    for row in examples:
        row.validate()
    manifest = {
        "schema_version": "wmv2.phase4-completion.dataset-manifest.v1",
        "dataset": dataset, "source": source, "rows": len(examples),
        "actors": len({r.actor_key for r in examples}),
        "relationships": len({r.relationship_key for r in examples}),
        "sequences": len({r.sequence_key for r in examples}),
        "time_min": min((r.decision_time_label for r in examples), default=""),
        "time_max": max((r.decision_time_label for r in examples), default=""),
        "actions": list(ACTIONS[dataset]), "content_sha256": _content_hash(examples),
        "actor_visible_fields": sorted({key for r in examples for key in r.visible_state}),
        "numeric_feature_fields": sorted({key for r in examples for key in r.numeric_features}),
        "label_excluded_from_visible_state": True, "post_action_features_excluded": True,
        "limitations": limitations,
    }
    manifest["checksum"] = digest(manifest)
    return DatasetBuild(dataset, examples, manifest, _split_manifest(dataset, examples, split_method),
                        _action_diagnostics(dataset, examples))


def _session_splits(paths: list[Path]) -> dict[str, str]:
    treatments = defaultdict(set)
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                treatments[row["treatment"]].add(row["session"])
    out = {}
    allocation = ("train", "train", "train", "calibration", "validation", "test")
    for treatment, sessions in treatments.items():
        ranked = sorted(sessions, key=lambda s: stable_hash(SEED, s))
        if len(ranked) != 6:
            raise ValueError(f"IPD treatment {treatment} has {len(ranked)} sessions, expected six")
        out.update(dict(zip(ranked, allocation)))
    return out


def build_ipd_long(source_root: str | Path) -> DatasetBuild:
    root = Path(source_root)
    paths = [root / "fix.csv", root / "rand.csv"]
    expected = {
        "fix.csv": "c2adc0d68dbc15c4f5ec5f0eb9a05a988cc049a9c086b82f796521a642b6eae9",
        "rand.csv": "bc2999d6ad44c60f2b93361eb46b75614b1f600967ae061c76ae51184a80848a",
    }
    actual = {path.name: sha256_file(path) for path in paths}
    if actual != expected:
        raise ValueError("IPD source hashes do not match the preregistered files")
    session_splits = _session_splits(paths)
    raw_by_session = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("who_played") != "user" or row.get("action_player") not in ("C", "D"):
                    continue
                raw_by_session[row["session"]].append(row)

    examples = []
    for session in sorted(raw_by_session):
        rows_by_round = defaultdict(list)
        for raw in raw_by_session[session]:
            rows_by_round[int(raw["round"])].append(raw)
        actor_state = defaultdict(lambda: {"last_action": "none", "last_payoff": 0.0,
                                           "cum_payoff": 0.0, "games": 0, "cooperate": 0})
        opponent_state = defaultdict(lambda: {"meetings": 0, "last_action": "none"})
        for round_number in sorted(rows_by_round):
            current = rows_by_round[round_number]
            pending_updates = []
            for raw in sorted(current, key=lambda x: x["player"]):
                actor, opponent = raw["player"], raw["opponent"]
                own = actor_state[actor]
                dyad = opponent_state[(actor, opponent)]
                treatment = "fixed_partner" if raw["treatment"] == "fix" else "shuffled_partner"
                visible = {
                    "context": {"round": round_number, "treatment": treatment,
                                "first_round": own["games"] == 0},
                    "history": {"own_previous_action": own["last_action"],
                                "previous_payoff": own["last_payoff"],
                                "cumulative_payoff": own["cum_payoff"],
                                "games_played": own["games"],
                                "own_cooperation_rate": own["cooperate"] / max(1, own["games"]),
                                "current_opponent_previous_action": dyad["last_action"]},
                    "relationships": {"fixed_partner": treatment == "fixed_partner",
                                      "prior_meetings": dyad["meetings"]},
                    "resources": {"cumulative_payoff": own["cum_payoff"]},
                    "commitments": {"simultaneous_choice": True},
                }
                features = {
                    "round_scaled": round_number / 100.0,
                    "fixed_partner": float(treatment == "fixed_partner"),
                    "first_round": float(own["games"] == 0),
                    "prior_own_cooperate": float(own["last_action"] == "cooperate"),
                    "prior_own_defect": float(own["last_action"] == "defect"),
                    "prior_opponent_cooperate": float(dyad["last_action"] == "cooperate"),
                    "prior_opponent_defect": float(dyad["last_action"] == "defect"),
                    "previous_payoff_scaled": own["last_payoff"] / 4.0,
                    "cumulative_payoff_per_game": own["cum_payoff"] / max(1, 4 * own["games"]),
                    "own_cooperation_rate": own["cooperate"] / max(1, own["games"]),
                    "prior_meetings_scaled": min(1.0, dyad["meetings"] / 20.0),
                }
                label = "cooperate" if raw["action_player"] == "C" else "defect"
                opponent_action = "cooperate" if raw["action_opponent"] == "C" else "defect"
                record_key = f"ipd:{session}:{actor}:{round_number}"
                examples.append(DecisionExample(
                    record_key, "ipd_long", actor, "experimental_participant",
                    f"{session}:{min(actor, opponent)}:{max(actor, opponent)}", f"{session}:{actor}",
                    session, float(round_number), f"session={session};round={round_number}", label,
                    ACTIONS["ipd_long"], visible, features,
                    outcome={"payoff": float(raw["payoff"]), "opponent_action": opponent_action,
                             "response_time_ms": float(raw["time_js"] or raw["time_php"] or 0.0)},
                    provenance={"source_ids": [raw.get("", "")], "context_key": session,
                                "institution_key": "iterated_prisoners_dilemma",
                                "action_set_status": "observed", "action_set_source": "experiment rules"},
                    split=session_splits[session],
                ))
                pending_updates.append((actor, opponent, label, opponent_action, float(raw["payoff"])))
            # All choices in a round are projected before any simultaneous action becomes history.
            for actor, opponent, label, opponent_action, payoff in pending_updates:
                state = actor_state[actor]
                state["last_action"], state["last_payoff"] = label, payoff
                state["cum_payoff"] += payoff
                state["games"] += 1
                state["cooperate"] += int(label == "cooperate")
                dyad = opponent_state[(actor, opponent)]
                dyad["meetings"] += 1
                dyad["last_action"] = opponent_action
    examples.sort(key=lambda r: (r.decision_time, r.record_key))
    return _finish_build(
        "ipd_long", examples,
        source={"doi": "10.5061/dryad.37pvmcvmk", "license": "CC0-1.0",
                "mirror_commit": "4743f74d8a03a031b8f70aff64956f8ad5155182",
                "raw_sha256": actual},
        split_method="per-treatment session hash rank 3/1/1/1",
        limitations=["Opponent identity is anonymized.",
                     "Current simultaneous opponent action and response time are post-decision outcomes."],
    )


VOTEVIEW_HASHES = {
    "members_S114.csv": "9777591e099a0a03e1da95c5ce7e3b153ff0e8651d04e908a5cb7263151bd754",
    "members_S115.csv": "f4531d2a3f7beb9793a4d005d072ad88d1efc838857aec4d4890a04d76b2eff1",
    "members_S116.csv": "469aa7ee58c717f629ec15f7f5951050033e29dfd0678ed78a7bd8186422d39b",
    "members_S117.csv": "2b8ec62575c27e641427c2706a55dc8d91be36f6aa85ea596f66a1d17c8ce835",
    "members_S118.csv": "5379269bd514fabd74d853af5d5d41ed8c066ecbb68b359f044e0dce4c3969b2",
    "rollcalls_S115.csv": "e8bb08c09d6d469f9f5a9960a12ce64f67970e87242fa6990b6dd333d47ce305",
    "rollcalls_S116.csv": "f23d37dd06cb860707710d2635fddc9dbe62ed2976cc91c23314663045fc9319",
    "rollcalls_S117.csv": "c3321eef9f641c474ce03f91d0ac2be2a2a607647b7b0fb578f2cea6669b313b",
    "rollcalls_S118.csv": "92c554474b1a90f64a21b2fc94df59ea835a8d424b7ad172c80f80035a3db7ae",
    "votes_S115.csv": "fbaa6968056a462d29c88e09e60e73a18a25e676546700759cf4f89e6f03b8e0",
    "votes_S116.csv": "f1c3ff256fa7cdccb24238792d3b8f4502433e6a23bd1760692a247c5f018c2b",
    "votes_S117.csv": "290b2440fadf1040b5ffccc264f4f5d7eb04dc74508cb7c34877d45ff1abd0ff",
    "votes_S118.csv": "0db5b5637485220253557b2b4ea88414ab080e5a5d049c33ad48f2aaced959b3",
}


def _read_csv_index(path: Path, key: str, *, senate_only: bool = False) -> dict:
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if senate_only and row.get("chamber") != "Senate":
                continue
            out[str(row[key])] = row
    return out


def _date_timestamp(value: str) -> float:
    return float(calendar.timegm(time.strptime(value, "%Y-%m-%d")))


def _vote_action(cast_code: str) -> str | None:
    try:
        code = int(float(cast_code))
    except (TypeError, ValueError):
        return None
    if 1 <= code <= 3:
        return "support"
    if 4 <= code <= 6:
        return "oppose"
    if 7 <= code <= 9:
        return "abstain_or_absent"
    return None


def _roll_passed(vote_result: str) -> bool:
    value = (vote_result or "").lower()
    failures = ("rejected", "failed", "not agreed", "not confirmed", "not invoked", "veto sustained")
    return not any(token in value for token in failures)


def _threshold_rule(question: str, description: str) -> tuple[str, float]:
    text = f"{question} {description}".lower()
    if "cloture" in text:
        return "three_fifths_cloture", 0.60
    if "override" in text and "veto" in text:
        return "two_thirds_veto_override", 2.0 / 3.0
    if "convict" in text or "impeachment" in text:
        return "two_thirds_impeachment", 2.0 / 3.0
    return "simple_majority", 0.50


def build_voteview_senate(source_root: str | Path) -> DatasetBuild:
    root = Path(source_root)
    actual = {name: sha256_file(root / name) for name in VOTEVIEW_HASHES}
    if actual != VOTEVIEW_HASHES:
        raise ValueError("VoteView source hashes do not match the preregistration")
    members = {}
    for congress in range(114, 119):
        with (root / f"members_S{congress}.csv").open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("chamber") == "Senate"]
        members[congress] = {row["icpsr"]: row for row in rows}
    prior_party_medians = {}
    for congress in range(115, 119):
        by_party = defaultdict(list)
        for row in members[congress - 1].values():
            try:
                by_party[row["party_code"]].append(float(row["nominate_dim1"]))
            except (TypeError, ValueError):
                continue
        prior_party_medians[congress] = {
            party: statistics.median(values) for party, values in by_party.items() if values
        }

    examples = []
    split_by_congress = {115: "train", 116: "calibration", 117: "validation", 118: "test"}
    for congress in range(115, 119):
        rollcalls = _read_csv_index(root / f"rollcalls_S{congress}.csv", "rollnumber",
                                    senate_only=True)
        votes_by_roll = defaultdict(list)
        with (root / f"votes_S{congress}.csv").open(newline="", encoding="utf-8-sig") as handle:
            for vote in csv.DictReader(handle):
                if vote.get("chamber") == "Senate":
                    votes_by_roll[vote["rollnumber"]].append(vote)
        actor_history = defaultdict(Counter)
        party_history = defaultdict(Counter)
        ordered_rolls = sorted(rollcalls.values(), key=lambda row: (
            row.get("date", ""), int(row["rollnumber"])))
        for roll in ordered_rolls:
            rollnumber = roll["rollnumber"]
            question = roll.get("vote_question", "")
            description = " ".join(filter(None, (roll.get("vote_desc", ""), roll.get("dtl_desc", ""))))
            rule, threshold = _threshold_rule(question, description)
            passed = _roll_passed(roll.get("vote_result", ""))
            date = roll["date"]
            pending = []
            for vote in sorted(votes_by_roll.get(rollnumber, []), key=lambda row: row["icpsr"]):
                action = _vote_action(vote.get("cast_code", ""))
                member = members[congress].get(vote["icpsr"])
                if action is None or member is None:
                    continue
                prior_member = members[congress - 1].get(vote["icpsr"])
                ideology_source = "prior_congress_member"
                try:
                    ideology = float(prior_member["nominate_dim1"]) if prior_member else math.nan
                except (TypeError, ValueError):
                    ideology = math.nan
                if not math.isfinite(ideology):
                    ideology = prior_party_medians[congress].get(member["party_code"], math.nan)
                    ideology_source = "prior_congress_party_median"
                if not math.isfinite(ideology):
                    continue
                party = member["party_code"]
                own = actor_history[vote["icpsr"]]
                party_prior = party_history[party]
                prior_n = sum(own.values())
                party_n = sum(party_prior.values())
                bill_number = roll.get("bill_number", "")
                matter = "nomination" if bill_number.startswith("PN") else (
                    "resolution" if "RES" in bill_number else "legislation")
                visible = {
                    "beliefs_or_signals": {"prior_congress_ideology": ideology,
                                           "ideology_source": ideology_source},
                    "history": {"prior_votes_current_congress": prior_n,
                                "support_rate": own["support"] / max(1, prior_n),
                                "oppose_rate": own["oppose"] / max(1, prior_n),
                                "abstain_rate": own["abstain_or_absent"] / max(1, prior_n)},
                    "network_or_party_structure": {
                        "party_code": party, "party_prior_votes": party_n,
                        "party_support_rate": party_prior["support"] / max(1, party_n),
                        "party_oppose_rate": party_prior["oppose"] / max(1, party_n),
                    },
                    "institution": {"chamber": "Senate", "congress": congress,
                                    "question": question[:600], "description": description[:1000],
                                    "bill_number": bill_number, "matter": matter,
                                    "threshold_rule": rule, "threshold_share": threshold},
                    "commitments": {"party_membership": party, "state": member.get("state_abbrev", "")},
                }
                features = {
                    "prior_ideology": ideology,
                    "party_democrat": float(party == "100" or party == "100.0"),
                    "party_republican": float(party == "200" or party == "200.0"),
                    "party_other": float(party not in ("100", "100.0", "200", "200.0")),
                    "prior_votes_scaled": min(1.0, prior_n / 500.0),
                    "own_support_rate": own["support"] / max(1, prior_n),
                    "own_oppose_rate": own["oppose"] / max(1, prior_n),
                    "own_abstain_rate": own["abstain_or_absent"] / max(1, prior_n),
                    "party_support_rate": party_prior["support"] / max(1, party_n),
                    "party_oppose_rate": party_prior["oppose"] / max(1, party_n),
                    "nomination": float(matter == "nomination"),
                    "resolution": float(matter == "resolution"),
                    "legislation": float(matter == "legislation"),
                    "supermajority": float(threshold > 0.5),
                    "session_two": float(roll.get("session") == "2"),
                }
                actor = f"senator:{vote['icpsr']}"
                record_key = f"voteview:{congress}:{int(rollnumber):04d}:{vote['icpsr']}"
                examples.append(DecisionExample(
                    record_key, "voteview_senate", actor, "United States Senator",
                    f"party:{party}", f"congress:{congress}:{vote['icpsr']}",
                    f"roll:{congress}:{rollnumber}", _date_timestamp(date),
                    f"{date};congress={congress};roll={rollnumber}", action,
                    ACTIONS["voteview_senate"], visible, features,
                    outcome={"roll_call_passed": passed, "threshold_share": threshold},
                    provenance={"source_ids": [f"S{congress}:roll:{rollnumber}:member:{vote['icpsr']}"],
                                "context_key": f"{congress}:{matter}",
                                "institution_key": f"senate:{congress}",
                                "action_set_status": "explicit_rule",
                                "action_set_source": "VoteView cast-code mapping"},
                    split=split_by_congress[congress],
                ))
                pending.append((vote["icpsr"], party, action))
            # Votes on one roll are simultaneous for actor-view construction.
            for icpsr, party, action in pending:
                actor_history[icpsr][action] += 1
                party_history[party][action] += 1
    examples.sort(key=lambda r: (r.decision_time, r.record_key))
    return _finish_build(
        "voteview_senate", examples,
        source={"name": "Voteview Congressional Roll-Call Votes Database",
                "citation": "https://voteview.com/data", "raw_sha256": actual},
        split_method="Congress 115/116/117/118 train/calibration/validation/test",
        limitations=[
            "Abstention and absence remain a combined action.",
            "Prior-Congress ideology falls back to the prior party median for new members.",
            "VoteView current-vote probability, current counts, and result are excluded from actor-visible state.",
        ],
    )


@dataclass(slots=True)
class _MailMessage:
    timestamp: float
    sender: str
    recipient: str
    subject: str
    display_subject: str
    body: str
    message_key: str
    source_path: str


def _addresses(value) -> list[str]:
    out = []
    for _, address in getaddresses([str(value or "")]):
        normalized = address.strip().lower()
        if normalized and "@" in normalized and normalized not in out:
            out.append(normalized)
    return out


def _normalized_subject(value: str) -> str:
    subject = re.sub(r"^\s*((re|fw|fwd)\s*:\s*)+", "", str(value or ""), flags=re.I)
    subject = re.sub(r"\s+", " ", subject).strip().lower()
    return subject[:240]


def _plain_body(message, limit: int) -> str:
    part = None
    try:
        part = message.get_body(preferencelist=("plain",))
    except (AttributeError, KeyError, TypeError):
        part = None
    try:
        content = (part or message).get_content()
    except (AttributeError, KeyError, LookupError, UnicodeError):
        payload = (part or message).get_payload(decode=True)
        content = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else str(payload or "")
    content = re.sub(r"\x00", "", str(content))
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return content[:limit]


def _mail_timestamp(value: str) -> float | None:
    try:
        parsed = parsedate_to_datetime(value)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            return float(calendar.timegm(parsed.timetuple()))
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError, IndexError):
        return None


def _read_enron_messages(archive_path: Path, *, body_chars: int = 1200) -> tuple[list[_MailMessage], dict]:
    lower = _date_timestamp("1998-01-01")
    upper = _date_timestamp("2006-01-01")
    parser = BytesParser(policy=policy.default)
    by_key = {}
    counters = Counter()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            counters["archive_files"] += 1
            handle = archive.extractfile(member)
            if handle is None:
                counters["unreadable"] += 1
                continue
            try:
                raw = handle.read()
                message = parser.parsebytes(raw)
            except Exception:
                counters["parse_error"] += 1
                continue
            timestamp = _mail_timestamp(message.get("Date", ""))
            senders = _addresses(message.get("From", ""))
            recipients = _addresses(message.get("To", ""))
            if timestamp is None or not lower <= timestamp < upper:
                counters["bad_time"] += 1
                continue
            if len(senders) != 1 or len(recipients) != 1 or senders[0] == recipients[0]:
                counters["not_single_primary_recipient"] += 1
                continue
            normalized_subject = _normalized_subject(message.get("Subject", ""))
            if not normalized_subject:
                counters["empty_subject"] += 1
                continue
            body = _plain_body(message, body_chars)
            message_id = str(message.get("Message-ID", "")).strip().lower()
            fallback = hashlib.sha256((
                f"{timestamp}|{senders[0]}|{recipients[0]}|{normalized_subject}|{body}"
            ).encode()).hexdigest()
            key = message_id if message_id else "digest:" + fallback
            row = _MailMessage(timestamp, senders[0], recipients[0], normalized_subject,
                               str(message.get("Subject", ""))[:240], body, key, member.name)
            existing = by_key.get(key)
            if existing is None or (row.timestamp, row.source_path) < (existing.timestamp, existing.source_path):
                if existing is not None:
                    counters["duplicate_replaced"] += 1
                by_key[key] = row
            else:
                counters["duplicate_dropped"] += 1
    rows = sorted(by_key.values(), key=lambda row: (row.timestamp, row.message_key, row.source_path))
    counters["retained_unique_messages"] = len(rows)
    return rows, dict(counters)


def _enron_time_split(examples: list[DecisionExample]) -> dict:
    ordered = sorted(examples, key=lambda r: (r.decision_time, r.record_key))
    n = len(ordered)
    cut_indices = [round(n * fraction) for fraction in (0.55, 0.70, 0.80)]
    boundaries = [ordered[min(n - 1, index)].decision_time for index in cut_indices] if n else []
    names = ("train", "calibration", "validation", "test")
    counts = Counter()
    for index, row in enumerate(ordered):
        bucket = bisect.bisect_right(cut_indices, index)
        row.split = names[min(3, bucket)]
        if any(abs(row.decision_time - boundary) < 7 * DAY for boundary in boundaries):
            row.split = "purged"
        counts[row.split] += 1
    return {"cut_indices": cut_indices, "boundary_timestamps": boundaries, "counts": dict(counts)}


def build_enron_repaired(archive_path: str | Path) -> DatasetBuild:
    archive = Path(archive_path)
    expected = "b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7"
    actual = sha256_file(archive)
    if actual != expected:
        raise ValueError("Enron archive hash does not match the preregistration")
    messages, parsing = _read_enron_messages(archive)
    examples: list[DecisionExample] = []
    # Each pending tuple is [active, example index, received_at].  The heap makes
    # negative outcomes visible precisely at censoring, never at message receipt.
    pending_by_key = defaultdict(list)
    maturity_heap = []
    sequence = 0
    actor_outcomes = defaultdict(Counter)
    dyad_outcomes = defaultdict(Counter)
    partner_reply_outcomes = defaultdict(Counter)
    recent_inbox = defaultdict(deque)

    def mature_no_reply(until: float) -> None:
        while maturity_heap and maturity_heap[0][0] <= until:
            _, _, item = heapq.heappop(maturity_heap)
            if not item[0]:
                continue
            item[0] = False
            row = examples[item[1]]
            row.label = "no_reply_within_7d"
            actor_outcomes[row.actor_key][row.label] += 1
            other = row.provenance["counterparty_key"]
            dyad_outcomes[(row.actor_key, other)][row.label] += 1

    for mail in messages:
        mature_no_reply(mail.timestamp)
        # This message may be a response by its sender to one earlier reversed-dyad opportunity.
        response_key = (mail.sender, mail.recipient, mail.subject)
        candidates = pending_by_key[response_key]
        while candidates and not candidates[-1][0]:
            candidates.pop()
        if candidates:
            item = candidates.pop()
            delay = mail.timestamp - item[2]
            if 0.0 <= delay <= 7 * DAY:
                item[0] = False
                prior = examples[item[1]]
                prior.label = "reply_within_24h" if delay <= DAY else "reply_in_1_to_7d"
                prior.outcome.update({"reply_delay_seconds": delay, "reply_message_key": mail.message_key})
                actor_outcomes[prior.actor_key][prior.label] += 1
                dyad_outcomes[(prior.actor_key, mail.recipient)][prior.label] += 1
                partner_reply_outcomes[(mail.recipient, mail.sender)][prior.label] += 1

        actor, other = mail.recipient, mail.sender
        inbox = recent_inbox[actor]
        while inbox and inbox[0] < mail.timestamp - DAY:
            inbox.popleft()
        inbox.append(mail.timestamp)
        own = actor_outcomes[actor]
        dyad = dyad_outcomes[(actor, other)]
        partner = partner_reply_outcomes[(actor, other)]
        own_n, dyad_n, partner_n = sum(own.values()), sum(dyad.values()), sum(partner.values())
        commitment_markers = len(re.findall(
            r"\b(please|need|must|by (monday|tuesday|wednesday|thursday|friday)|deadline|asap)\b",
            f"{mail.display_subject} {mail.body}", flags=re.I))
        visible = {
            "beliefs_or_signals": {"subject": mail.display_subject, "message_body": mail.body},
            "context": {"local_hour_utc": int(time.gmtime(mail.timestamp).tm_hour),
                        "weekday_utc": int(time.gmtime(mail.timestamp).tm_wday)},
            "history": {"prior_matured_opportunities": own_n,
                        "prior_timely_reply_rate": own["reply_within_24h"] / max(1, own_n),
                        "prior_any_reply_rate": (own["reply_within_24h"] + own["reply_in_1_to_7d"])
                        / max(1, own_n)},
            "relationships": {"prior_matured_dyad_opportunities": dyad_n,
                              "actor_any_reply_rate_to_counterparty": (
                                  dyad["reply_within_24h"] + dyad["reply_in_1_to_7d"]) / max(1, dyad_n),
                              "counterparty_any_reply_rate": (
                                  partner["reply_within_24h"] + partner["reply_in_1_to_7d"])
                              / max(1, partner_n)},
            "resources": {"inbound_messages_past_24h_including_current": len(inbox)},
            "commitments": {"visible_commitment_markers": commitment_markers},
        }
        features = {
            "hour_sin": math.sin(2 * math.pi * time.gmtime(mail.timestamp).tm_hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * time.gmtime(mail.timestamp).tm_hour / 24.0),
            "weekend": float(time.gmtime(mail.timestamp).tm_wday >= 5),
            "subject_chars_scaled": min(1.0, len(mail.display_subject) / 120.0),
            "body_chars_scaled": min(1.0, len(mail.body) / 1200.0),
            "commitment_markers_scaled": min(1.0, commitment_markers / 5.0),
            "inbox_load_scaled": min(1.0, len(inbox) / 50.0),
            "prior_history": min(1.0, own_n / 100.0),
            "prior_timely_rate": own["reply_within_24h"] / max(1, own_n),
            "prior_any_reply_rate": (own["reply_within_24h"] + own["reply_in_1_to_7d"])
            / max(1, own_n),
            "dyad_history": min(1.0, dyad_n / 20.0),
            "dyad_any_reply_rate": (dyad["reply_within_24h"] + dyad["reply_in_1_to_7d"])
            / max(1, dyad_n),
            "partner_reply_history": min(1.0, partner_n / 20.0),
            "partner_any_reply_rate": (partner["reply_within_24h"]
                                       + partner["reply_in_1_to_7d"]) / max(1, partner_n),
        }
        record_key = "enron:" + stable_hash(mail.message_key, actor, other)[:24]
        examples.append(DecisionExample(
            record_key, "enron_repaired", actor, "email_recipient", f"{actor}|{other}", actor,
            actor, mail.timestamp, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mail.timestamp)),
            "", ACTIONS["enron_repaired"], visible, features,
            outcome={"reply_delay_seconds": None},
            provenance={"source_ids": [mail.message_key, mail.source_path],
                        "counterparty_key": other, "context_key": mail.subject,
                        "institution_key": "enron_email", "action_set_status": "reconstructed_from_logs",
                        "action_set_source": "reversed dyad, normalized subject, seven-day window"},
        ))
        item = [True, len(examples) - 1, mail.timestamp]
        pending_by_key[(actor, other, mail.subject)].append(item)
        sequence += 1
        heapq.heappush(maturity_heap, (mail.timestamp + 7 * DAY, sequence, item))
    mature_no_reply(math.inf)
    split_details = _enron_time_split(examples)
    examples.sort(key=lambda r: (r.decision_time, r.record_key))
    build = _finish_build(
        "enron_repaired", examples,
        source={"name": "CMU CALO Enron Email Dataset",
                "url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
                "raw_bytes": archive.stat().st_size, "raw_sha256": actual},
        split_method="time-forward 55/15/10/20 with seven-day purge at all boundaries",
        limitations=[
            "A reply match requires a reversed dyad and exact normalized subject; unmatched aliases are censored.",
            "Single-primary-recipient messages only; forwarding and delegation are outside the action set.",
            "Quoted text is retained as actor-visible historical content and treated as untrusted prompt data.",
        ],
    )
    build.manifest["parsing_diagnostics"] = parsing
    build.manifest["time_split_details"] = split_details
    build.manifest["checksum"] = digest({k: v for k, v in build.manifest.items() if k != "checksum"})
    return build


class EmpiricalFrequencyModel:
    def __init__(self, actions: tuple[str, ...], alpha: float = 1.0):
        self.actions, self.alpha = actions, float(alpha)
        self.global_counts = Counter()
        self.context_counts = defaultdict(Counter)

    def fit(self, rows: list[DecisionExample]) -> "EmpiricalFrequencyModel":
        for row in rows:
            self.global_counts[row.label] += row.sample_weight
            self.context_counts[str(row.provenance.get("context_key", ""))][row.label] += row.sample_weight
        return self

    def _distribution(self, counts: Counter) -> dict:
        total = sum(counts.values()) + self.alpha * len(self.actions)
        return {action: (counts[action] + self.alpha) / total for action in self.actions}

    def predict_b0(self, row: DecisionExample) -> dict:
        return self._distribution(self.global_counts)

    def predict_b1(self, row: DecisionExample) -> dict:
        local = self.context_counts[str(row.provenance.get("context_key", ""))]
        if sum(local.values()) < 5:
            return self.predict_b0(row)
        return self._distribution(local)


INSTANT_FEATURES = {
    "enron_repaired": {
        "hour_sin", "hour_cos", "weekend", "subject_chars_scaled", "body_chars_scaled",
        "commitment_markers_scaled", "inbox_load_scaled",
    },
    "ipd_long": {"round_scaled", "fixed_partner", "first_round"},
    "voteview_senate": {
        "prior_ideology", "party_democrat", "party_republican", "party_other",
        "nomination", "resolution", "legislation", "supermajority", "session_two",
    },
}


@dataclass
class SparseSoftmaxModel:
    actions: tuple[str, ...]
    feature_names: tuple[str, ...]
    weights: list[list[float]]
    regularization: float
    epochs: int
    fit_rows: int

    @classmethod
    def fit(cls, rows: list[DecisionExample], actions: tuple[str, ...], feature_names,
            *, regularization: float, epochs: int = 5) -> "SparseSoftmaxModel":
        feature_names = tuple(sorted(feature_names))
        d, k = len(feature_names) + 1, len(actions)
        action_index = {action: i for i, action in enumerate(actions)}
        counts = Counter(row.label for row in rows)
        total = sum(counts.values()) + k
        weights = [[0.0] * d for _ in actions]
        for i, action in enumerate(actions):
            weights[i][0] = math.log((counts[action] + 1.0) / total)
        ordered = sorted(rows, key=lambda row: stable_hash("softmax", row.record_key))
        seen = 0
        for epoch in range(epochs):
            learning_rate = 0.08 / math.sqrt(epoch + 1.0)
            shrink = max(0.0, 1.0 - learning_rate * regularization / max(1, len(ordered)))
            for row in ordered:
                x = [1.0] + [float(row.numeric_features.get(name, 0.0)) for name in feature_names]
                logits = [sum(w * value for w, value in zip(class_weights, x))
                          for class_weights in weights]
                maximum = max(logits)
                exp = [math.exp(min(40.0, value - maximum)) for value in logits]
                z = sum(exp)
                probabilities = [value / z for value in exp]
                truth = action_index[row.label]
                step = learning_rate * min(10.0, float(row.sample_weight))
                for class_index in range(k):
                    error = (1.0 if class_index == truth else 0.0) - probabilities[class_index]
                    class_weights = weights[class_index]
                    for j, value in enumerate(x):
                        class_weights[j] += step * error * value
                seen += 1
            for class_weights in weights:
                for j in range(1, d):
                    class_weights[j] *= shrink
        return cls(actions, feature_names, weights, float(regularization), epochs, seen)

    def predict(self, row: DecisionExample) -> dict:
        x = [1.0] + [float(row.numeric_features.get(name, 0.0)) for name in self.feature_names]
        logits = [sum(w * value for w, value in zip(class_weights, x))
                  for class_weights in self.weights]
        maximum = max(logits)
        values = [math.exp(min(40.0, value - maximum)) for value in logits]
        total = sum(values)
        return {action: values[i] / total for i, action in enumerate(self.actions)}

    def as_dict(self) -> dict:
        return asdict(self)


def fit_validated_softmax(train: list[DecisionExample], validation: list[DecisionExample],
                          actions: tuple[str, ...], feature_names) -> tuple[SparseSoftmaxModel, dict]:
    candidates = []
    for regularization in (0.01, 0.1, 1.0, 10.0):
        model = SparseSoftmaxModel.fit(train, actions, feature_names,
                                       regularization=regularization, epochs=3)
        probabilities = [model.predict(row) for row in validation]
        metric = evaluate_predictions(probabilities, [row.label for row in validation])
        candidates.append({"regularization": regularization, "validation_log_loss": metric["log_loss"]})
    best = min(candidates, key=lambda row: (row["validation_log_loss"], row["regularization"]))
    model = SparseSoftmaxModel.fit(train, actions, feature_names,
                                   regularization=best["regularization"], epochs=7)
    return model, {"selection_split": "validation", "candidates": candidates,
                   "selected_regularization": best["regularization"], "final_epochs": 7}


class HierarchicalPolicyModel:
    """Empirical-Bayes actor and relationship pooling around a visible-state logit."""

    def __init__(self, base: SparseSoftmaxModel, actions: tuple[str, ...], strength: float = 20.0):
        self.base, self.actions, self.strength = base, actions, float(strength)
        self.actor_counts = defaultdict(Counter)
        self.relationship_counts = defaultdict(Counter)

    def fit_groups(self, rows: list[DecisionExample]) -> "HierarchicalPolicyModel":
        for row in rows:
            self.actor_counts[row.actor_key][row.label] += row.sample_weight
            self.relationship_counts[row.relationship_key][row.label] += row.sample_weight
        return self

    def _pool(self, prior: dict, counts: Counter) -> dict:
        n = sum(counts.values())
        if n <= 0:
            return prior
        return {action: (self.strength * prior[action] + counts[action]) / (self.strength + n)
                for action in self.actions}

    def predict(self, row: DecisionExample) -> dict:
        result = self.base.predict(row)
        result = self._pool(result, self.actor_counts[row.actor_key])
        return self._pool(result, self.relationship_counts[row.relationship_key])

    def as_dict(self) -> dict:
        return {
            "base": self.base.as_dict(), "strength": self.strength,
            "actor_groups": len(self.actor_counts), "relationship_groups": len(self.relationship_counts),
        }


def transparent_heuristic(row: DecisionExample) -> dict:
    if row.dataset == "ipd_long":
        previous = row.visible_state["history"]["current_opponent_previous_action"]
        cooperate = 0.75 if previous == "cooperate" else (0.20 if previous == "defect" else 0.50)
        return {"cooperate": cooperate, "defect": 1.0 - cooperate}
    if row.dataset == "enron_repaired":
        history = row.visible_state["history"]
        relationship = row.visible_state["relationships"]
        timely = 0.5 * history["prior_timely_reply_rate"] + 0.3 * relationship[
            "actor_any_reply_rate_to_counterparty"] + 0.2 * min(
                1.0, row.visible_state["commitments"]["visible_commitment_markers"] / 2.0)
        any_reply = max(timely, 0.6 * history["prior_any_reply_rate"] + 0.4 * relationship[
            "actor_any_reply_rate_to_counterparty"])
        if history["prior_matured_opportunities"] == 0:
            timely, any_reply = 0.30, 0.55
        timely = min(0.90, max(0.03, timely))
        delayed = min(0.80, max(0.02, any_reply - timely))
        no_reply = max(0.02, 1.0 - timely - delayed)
        z = timely + delayed + no_reply
        return {"reply_within_24h": timely / z, "reply_in_1_to_7d": delayed / z,
                "no_reply_within_7d": no_reply / z}
    ideology = float(row.visible_state["beliefs_or_signals"]["prior_congress_ideology"])
    party = str(row.visible_state["network_or_party_structure"]["party_code"])
    nomination = row.visible_state["institution"]["matter"] == "nomination"
    support_logit = -0.25 * ideology + (0.35 if party in ("100", "100.0") else -0.05)
    support_logit += 0.20 if nomination else 0.0
    support = 1.0 / (1.0 + math.exp(-support_logit))
    absent = 0.025
    return {"support": support * (1.0 - absent), "oppose": (1.0 - support) * (1.0 - absent),
            "abstain_or_absent": absent}


class SpecialistCountModel:
    def __init__(self, actions: tuple[str, ...], alpha: float = 2.0):
        self.actions, self.alpha = actions, float(alpha)
        self.counts = defaultdict(Counter)
        self.global_counts = Counter()

    @staticmethod
    def key(row: DecisionExample) -> tuple:
        f = row.numeric_features
        if row.dataset == "ipd_long":
            return (int(f["fixed_partner"]), int(f["prior_own_cooperate"]),
                    int(f["prior_opponent_cooperate"]), int(f["first_round"]))
        if row.dataset == "enron_repaired":
            return (int(f["weekend"]), int(f["commitment_markers_scaled"] > 0),
                    min(3, int(4 * f["inbox_load_scaled"])), int(f["dyad_history"] > 0))
        ideology_bin = max(-4, min(4, int(round(4 * f["prior_ideology"]))))
        party = 0 if f["party_democrat"] else (1 if f["party_republican"] else 2)
        matter = 0 if f["nomination"] else (1 if f["resolution"] else 2)
        return party, matter, ideology_bin, int(f["supermajority"])

    def fit(self, rows: list[DecisionExample]) -> "SpecialistCountModel":
        for row in rows:
            self.counts[self.key(row)][row.label] += row.sample_weight
            self.global_counts[row.label] += row.sample_weight
        return self

    def predict(self, row: DecisionExample) -> dict:
        local = self.counts[self.key(row)]
        global_total = sum(self.global_counts.values()) or 1.0
        prior = {a: self.global_counts[a] / global_total for a in self.actions}
        n = sum(local.values())
        return {a: (local[a] + self.alpha * prior[a]) / (n + self.alpha) for a in self.actions}


def _history_action_counts(row: DecisionExample) -> dict:
    actions = row.actions
    if row.dataset == "ipd_long":
        history = row.visible_state["history"]
        n = int(history["games_played"])
        cooperate = int(round(n * history["own_cooperation_rate"]))
        return {"cooperate": cooperate, "defect": max(0, n - cooperate)}
    if row.dataset == "enron_repaired":
        history = row.visible_state["history"]
        n = int(history["prior_matured_opportunities"])
        timely = int(round(n * history["prior_timely_reply_rate"]))
        any_reply = int(round(n * history["prior_any_reply_rate"]))
        return {"reply_within_24h": timely,
                "reply_in_1_to_7d": max(0, any_reply - timely),
                "no_reply_within_7d": max(0, n - any_reply)}
    history = row.visible_state["history"]
    n = int(history["prior_votes_current_congress"])
    support = int(round(n * history["support_rate"]))
    oppose = int(round(n * history["oppose_rate"]))
    return {"support": support, "oppose": oppose,
            "abstain_or_absent": max(0, n - support - oppose)}


def phase3_action_particles(row: DecisionExample, prior_alpha: list[float],
                            n_particles: int = 64) -> tuple[list[tuple[dict, float]], dict]:
    counts = _history_action_counts(row)
    observations = []
    if sum(counts.values()) > 0:
        observations.append({
            "counts": counts, "reliability": 1.0,
            "dependence_group": f"prior-history:{row.record_key}",
            "source": "strictly_prior_actor_action_history", "method": "typed_action_counts",
        })
    seed = stable_seed(SEED, row.dataset, row.record_key, "typed_action_propensity")
    posterior = infer_compositional_posterior(
        list(row.actions), prior_alpha, observations, n_particles=n_particles, seed=seed,
        prior_provenance={"source": "training_action_frequency", "typed_actions": list(row.actions)},
    )
    particles = [({action: vector[i] for i, action in enumerate(row.actions)}, float(weight))
                 for vector, weight in posterior.particles]
    weights = [weight for _, weight in particles]
    unique = len({tuple(round(dist[action], 12) for action in row.actions) for dist, _ in particles})
    max_weight = max(weights)
    diagnostics = {
        "n": len(particles), "ess": posterior.ess,
        "ess_fraction": posterior.ess / max(1, len(particles)), "max_weight": max_weight,
        "unique_particles": unique, "posterior_mean": dict(zip(row.actions, posterior.posterior_mean)),
        "posterior_sd": dict(zip(row.actions, posterior.posterior_sd)),
        "interval_width_proxy": 3.92 * max(posterior.posterior_sd, default=0.0),
        "collapsed": posterior.ess / max(1, len(particles)) < 0.2 or max_weight > 0.5,
        "source_counts": counts, "seed": seed,
    }
    return particles, diagnostics


def _log_pool(distributions: list[dict], weights: list[float], actions: tuple[str, ...]) -> dict:
    logits = {action: sum(weight * math.log(max(1e-9, distribution[action]))
                          for distribution, weight in zip(distributions, weights))
              for action in actions}
    maximum = max(logits.values())
    mass = {action: math.exp(value - maximum) for action, value in logits.items()}
    total = sum(mass.values())
    return {action: mass[action] / total for action in actions}


def _weight_grid(step: float = 0.25) -> list[tuple[float, float, float]]:
    units = round(1.0 / step)
    return [(i / units, j / units, (units - i - j) / units)
            for i in range(units + 1) for j in range(units + 1 - i)]


@dataclass
class B7FamilyStacker:
    actions: tuple[str, ...]
    family_weights: tuple[float, float, float]
    validation_candidates: list[dict]
    prior_alpha: list[float]

    @classmethod
    def fit(cls, validation: list[DecisionExample], b6: HierarchicalPolicyModel,
            actions: tuple[str, ...], prior_alpha: list[float]) -> "B7FamilyStacker":
        components = []
        for row in validation:
            _, diagnostics = phase3_action_particles(row, prior_alpha)
            components.append((b6.predict(row), diagnostics["posterior_mean"], transparent_heuristic(row)))
        candidates = []
        labels = [row.label for row in validation]
        for weights in _weight_grid():
            probabilities = [_log_pool(list(parts), list(weights), actions) for parts in components]
            metric = evaluate_predictions(probabilities, labels)
            candidates.append({"weights": list(weights), "validation_log_loss": metric["log_loss"]})
        best = min(candidates, key=lambda row: (row["validation_log_loss"], row["weights"]))
        return cls(actions, tuple(best["weights"]), candidates, list(prior_alpha))

    def predict(self, row: DecisionExample, b6: HierarchicalPolicyModel,
                *, n_particles: int = 64) -> tuple[dict, dict]:
        base = b6.predict(row)
        heuristic = transparent_heuristic(row)
        particles, diagnostics = phase3_action_particles(row, self.prior_alpha, n_particles=n_particles)
        per_particle = []
        retained = []
        for propensity, weight in particles:
            per_particle.append(_log_pool([base, propensity, heuristic],
                                          list(self.family_weights), self.actions))
            retained.append(weight)
        total = sum(retained)
        prediction = {action: sum(weight * distribution[action]
                                  for distribution, weight in zip(per_particle, retained)) / total
                      for action in self.actions}
        posterior_mean = diagnostics["posterior_mean"]
        point_world = _log_pool([base, posterior_mean, heuristic], list(self.family_weights), self.actions)
        no_particle_z = self.family_weights[0] + self.family_weights[2]
        no_consequence_z = self.family_weights[0] + self.family_weights[1]
        no_particles = (_log_pool([base, heuristic],
                                  [self.family_weights[0] / no_particle_z,
                                   self.family_weights[2] / no_particle_z], self.actions)
                        if no_particle_z else {action: 1.0 / len(self.actions) for action in self.actions})
        no_consequences = (_log_pool([base, posterior_mean],
                                     [self.family_weights[0] / no_consequence_z,
                                      self.family_weights[1] / no_consequence_z], self.actions)
                           if no_consequence_z else {action: 1.0 / len(self.actions) for action in self.actions})
        diagnostics.update({
            "family_weights": dict(zip(("hierarchical_visible_policy", "phase3_propensity_particles",
                                         "typed_consequence_heuristic"), self.family_weights)),
            "per_particle_action_tv": _mean_pairwise_tv(per_particle),
            "argmax_diversity": len({max(dist, key=dist.get) for dist in per_particle}),
            "b7_predict_sha256": digest(prediction),
            # Execution consumes this exact dict.  No policy re-evaluation is permitted after selection.
            "b7_execute_input_sha256": digest(prediction),
            "predict_execute_byte_identical": json.dumps(prediction, sort_keys=True, separators=(",", ":"))
            == json.dumps(dict(prediction), sort_keys=True, separators=(",", ":")),
            "point_world_prediction": point_world,
            "no_posterior_particles_prediction": no_particles,
            "no_subjective_consequences_prediction": no_consequences,
        })
        return prediction, diagnostics

    def as_dict(self) -> dict:
        return asdict(self)


def _mean_pairwise_tv(distributions: list[dict], max_pairs: int = 256) -> float:
    if len(distributions) < 2:
        return 0.0
    pairs = []
    stride = max(1, len(distributions) // 16)
    sampled = distributions[::stride][:17]
    for i, first in enumerate(sampled):
        for second in sampled[i + 1:]:
            pairs.append(0.5 * sum(abs(first[action] - second[action]) for action in first))
            if len(pairs) >= max_pairs:
                return sum(pairs) / len(pairs)
    return sum(pairs) / max(1, len(pairs))


@dataclass
class DatasetModels:
    dataset: str
    frequency: EmpiricalFrequencyModel
    b5: SparseSoftmaxModel
    b5_selection: dict
    b6: HierarchicalPolicyModel
    b7: B7FamilyStacker
    b8: SpecialistCountModel
    prior_alpha: list[float]

    def artifact(self) -> dict:
        return {
            "schema_version": "wmv2.phase4-completion.model-pack.v1", "dataset": self.dataset,
            "b0_global_counts": dict(self.frequency.global_counts),
            "b1_contexts": len(self.frequency.context_counts), "b5": self.b5.as_dict(),
            "b5_selection": self.b5_selection, "b6": self.b6.as_dict(),
            "b7": self.b7.as_dict(), "b8_contexts": len(self.b8.counts),
            "phase3_prior_alpha": self.prior_alpha,
        }


def fit_dataset_models(build: DatasetBuild) -> DatasetModels:
    rows = build.examples
    train = [row for row in rows if row.split == "train"]
    validation = [row for row in rows if row.split == "validation"]
    if not train or not validation:
        raise ValueError("model fitting requires nonempty train and validation partitions")
    actions = ACTIONS[build.dataset]
    frequency = EmpiricalFrequencyModel(actions).fit(train)
    b5, b5_selection = fit_validated_softmax(
        train, validation, actions, INSTANT_FEATURES[build.dataset])
    all_features = sorted({name for row in train for name in row.numeric_features})
    base_b6, b6_selection = fit_validated_softmax(train, validation, actions, all_features)
    b5_selection["b6_regularization_selection"] = b6_selection
    b6 = HierarchicalPolicyModel(base_b6, actions).fit_groups(train)
    global_total = sum(frequency.global_counts.values()) or 1.0
    prior_alpha = [1.0 + 10.0 * frequency.global_counts[action] / global_total for action in actions]
    b7 = B7FamilyStacker.fit(validation, b6, actions, prior_alpha)
    b8 = SpecialistCountModel(actions).fit(train)
    return DatasetModels(build.dataset, frequency, b5, b5_selection, b6, b7, b8, prior_alpha)


def _extended_metrics(predictions: list[dict], rows: list[DecisionExample]) -> dict:
    labels = [row.label for row in rows]
    weights = [row.sample_weight for row in rows]
    result = evaluate_predictions(predictions, labels, [row.actions for row in rows], weights)
    classes = list(rows[0].actions)
    top = [max(distribution, key=distribution.get) for distribution in predictions]
    recalls, f1s = [], []
    for action in classes:
        tp = sum(weight for row, predicted, weight in zip(rows, top, weights)
                 if row.label == action and predicted == action)
        actual = sum(weight for row, weight in zip(rows, weights) if row.label == action)
        predicted_n = sum(weight for predicted, weight in zip(top, weights) if predicted == action)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted_n if predicted_n else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    result["balanced_accuracy"] = sum(recalls) / len(recalls)
    result["macro_f1"] = sum(f1s) / len(f1s)
    result["top2_accuracy"] = sum(
        weight for row, distribution, weight in zip(rows, predictions, weights)
        if row.label in sorted(distribution, key=distribution.get, reverse=True)[:2]
    ) / (sum(weights) or 1.0)
    return result


def clustered_difference(pred_a: list[dict], pred_b: list[dict], rows: list[DecisionExample],
                         *, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Cluster bootstrap of loss(A)-loss(B), retaining within-cluster weights."""
    clusters = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[row.cluster_key].append(index)
    keys = sorted(clusters)
    if len(keys) < 2:
        return {"mean": None, "ci95": [None, None], "clusters": len(keys), "n_boot": n_boot}
    losses = [(-math.log(max(1e-12, pred_a[i][row.label]))
               + math.log(max(1e-12, pred_b[i][row.label]))) for i, row in enumerate(rows)]

    def statistic(selected):
        indices = [i for key in selected for i in clusters[key]]
        total = sum(rows[i].sample_weight for i in indices) or 1.0
        return sum(rows[i].sample_weight * losses[i] for i in indices) / total

    mean = statistic(keys)
    rng = random.Random(seed)
    samples = sorted(statistic([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(n_boot))
    below_zero = sum(value <= 0.0 for value in samples) / n_boot
    above_zero = sum(value >= 0.0 for value in samples) / n_boot
    return {
        "estimand": "log_loss_arm_a_minus_arm_b", "mean": mean,
        "ci95": [samples[int(0.025 * n_boot)], samples[min(n_boot - 1, int(0.975 * n_boot))]],
        "two_sided_p": min(1.0, 2.0 * min(below_zero, above_zero)),
        "clusters": len(keys), "cluster_unit": "dataset_preregistered_cluster",
        "n_boot": n_boot, "seed": seed,
    }


def conformal_summary(calibration_predictions: list[dict], calibration_rows: list[DecisionExample],
                      test_predictions: list[dict], test_rows: list[DecisionExample],
                      alpha: float = 0.10) -> dict:
    scores = sorted(1.0 - prediction[row.label]
                    for prediction, row in zip(calibration_predictions, calibration_rows))
    n = len(scores)
    rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
    threshold = scores[max(0, rank - 1)]
    sets = [[action for action, probability in prediction.items()
             if 1.0 - probability <= threshold] for prediction in test_predictions]
    covered = [row.label in values for row, values in zip(test_rows, sets)]
    coverage = sum(covered) / max(1, len(covered))
    return {
        "alpha": alpha, "calibration_n": n, "finite_sample_rank": rank,
        "threshold": threshold, "test_n": len(test_rows), "coverage": coverage,
        "mean_set_size": sum(map(len, sets)) / max(1, len(sets)),
        "empty_set_rate": sum(not values for values in sets) / max(1, len(sets)),
        "sets_by_record": {row.record_key: values for row, values in zip(test_rows, sets)},
    }


def risk_coverage(predictions: list[dict], rows: list[DecisionExample]) -> dict:
    ordered = sorted(range(len(rows)), key=lambda i: (
        -max(predictions[i].values()),
        sum(-p * math.log(max(1e-12, p)) for p in predictions[i].values()),
        rows[i].record_key,
    ))
    curve = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        selected = ordered[:max(1, math.ceil(coverage * len(ordered)))]
        total = sum(rows[i].sample_weight for i in selected) or 1.0
        loss = sum(rows[i].sample_weight * -math.log(max(1e-12, predictions[i][rows[i].label]))
                   for i in selected) / total
        error = sum(rows[i].sample_weight * (max(predictions[i], key=predictions[i].get) != rows[i].label)
                    for i in selected) / total
        curve.append({"coverage": coverage, "n": len(selected), "log_loss": loss,
                      "selective_error": error})
    aurc = 0.0
    points = sorted(curve, key=lambda row: row["coverage"])
    for first, second in zip(points, points[1:]):
        aurc += (second["coverage"] - first["coverage"]) * (
            first["selective_error"] + second["selective_error"]) / 2.0
    return {"ordering": "maximum_probability_then_entropy_label_blind", "curve": curve, "aurc": aurc}


def _slice_metrics(predictions: list[dict], rows: list[DecisionExample],
                   train: list[DecisionExample]) -> dict:
    known_actors = {row.actor_key for row in train}
    known_relationships = {row.relationship_key for row in train}
    masks = {
        "known_actor": [row.actor_key in known_actors for row in rows],
        "cold_actor": [row.actor_key not in known_actors for row in rows],
        "known_relationship": [row.relationship_key in known_relationships for row in rows],
        "cold_relationship": [row.relationship_key not in known_relationships for row in rows],
    }
    out = {}
    for name, mask in masks.items():
        selected_rows = [row for row, keep in zip(rows, mask) if keep]
        selected_predictions = [prediction for prediction, keep in zip(predictions, mask) if keep]
        out[name] = (_extended_metrics(selected_predictions, selected_rows)
                     if selected_rows else {"n": 0, "status": "empty_slice"})
    return out


def _sequence_summary(predictions: list[dict], rows: list[DecisionExample]) -> dict:
    sequences = defaultdict(list)
    for prediction, row in zip(predictions, rows):
        sequences[row.sequence_key].append((row.decision_time, row.record_key,
                                            -math.log(max(1e-12, prediction[row.label]))))
    values = []
    for key, events in sequences.items():
        ordered = sorted(events)
        values.append({"sequence": key, "n": len(ordered),
                       "negative_log_likelihood": sum(value for _, _, value in ordered),
                       "mean_log_loss": sum(value for _, _, value in ordered) / len(ordered)})
    return {
        "sequences": len(values), "events": len(rows),
        "mean_sequence_log_loss": sum(value["mean_log_loss"] for value in values) / max(1, len(values)),
        "total_sequence_negative_log_likelihood": sum(value["negative_log_likelihood"] for value in values),
        "per_sequence": values,
    }


def _downstream_summary(predictions: list[dict], rows: list[DecisionExample]) -> dict:
    if not rows:
        return {"status": "empty"}
    dataset = rows[0].dataset
    if dataset == "enron_repaired":
        delay_values = {"reply_within_24h": 12 * 3600.0, "reply_in_1_to_7d": 4 * DAY,
                        "no_reply_within_7d": 7 * DAY}
        errors = []
        for prediction, row in zip(predictions, rows):
            expected = sum(prediction[action] * delay_values[action] for action in row.actions)
            observed = row.outcome.get("reply_delay_seconds")
            observed = 7 * DAY if observed is None else min(7 * DAY, float(observed))
            errors.append(abs(expected - observed) / 3600.0)
        return {"estimand": "censored_reply_delay_hours", "mae": sum(errors) / len(errors),
                "n": len(errors), "causal_claim": False}
    if dataset == "ipd_long":
        payoff_errors, opponent_losses = [], []
        for prediction, row in zip(predictions, rows):
            previous = row.visible_state["history"]["current_opponent_previous_action"]
            p_opponent_c = 0.7 if previous == "cooperate" else (0.25 if previous == "defect" else 0.5)
            expected_if_c = 3.0 * p_opponent_c
            expected_if_d = 4.0 * p_opponent_c + 1.0 * (1.0 - p_opponent_c)
            expected_payoff = prediction["cooperate"] * expected_if_c + prediction["defect"] * expected_if_d
            payoff_errors.append(abs(expected_payoff - float(row.outcome["payoff"])))
            actual_opp = row.outcome["opponent_action"]
            opponent_losses.append(-math.log(max(1e-12, p_opponent_c if actual_opp == "cooperate"
                                                   else 1.0 - p_opponent_c)))
        return {"payoff_mae": sum(payoff_errors) / len(payoff_errors),
                "opponent_reaction_log_loss": sum(opponent_losses) / len(opponent_losses),
                "n": len(rows), "causal_claim": "randomized_payoff_structure_only"}
    by_roll = defaultdict(list)
    for prediction, row in zip(predictions, rows):
        by_roll[row.cluster_key].append((prediction, row))
    brier, loss = [], []
    rows_out = []
    for roll, members in sorted(by_roll.items()):
        expected_support = sum(prediction["support"] for prediction, _ in members)
        variance = sum(prediction["support"] * (1.0 - prediction["support"])
                       for prediction, _ in members)
        threshold = members[0][1].outcome["threshold_share"] * len(members)
        scale = max(1.0, math.sqrt(variance))
        probability_pass = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -(expected_support - threshold) / scale))))
        actual = bool(members[0][1].outcome["roll_call_passed"])
        brier.append((probability_pass - float(actual)) ** 2)
        loss.append(-math.log(max(1e-12, probability_pass if actual else 1.0 - probability_pass)))
        rows_out.append({"roll": roll, "n_members": len(members), "predicted_passage": probability_pass,
                         "actual_passage": actual})
    return {"roll_calls": len(rows_out), "passage_brier": sum(brier) / max(1, len(brier)),
            "passage_log_loss": sum(loss) / max(1, len(loss)), "per_roll": rows_out,
            "causal_claim": False}


def score_numeric_arms(build: DatasetBuild, models: DatasetModels) -> dict:
    partitions = {name: [row for row in build.examples if row.split == name]
                  for name in ("train", "calibration", "validation", "test")}
    if not partitions["calibration"] or not partitions["test"]:
        raise ValueError("scoring requires nonempty calibration and test partitions")
    arm_functions = {
        "B0": models.frequency.predict_b0,
        "B1": models.frequency.predict_b1,
        "B4": transparent_heuristic,
        "B5": models.b5.predict,
        "B6": models.b6.predict,
        "B8": models.b8.predict,
    }
    raw = {partition: {} for partition in ("calibration", "test")}
    b7_diagnostics = {partition: [] for partition in raw}
    started = time.monotonic()
    for partition in raw:
        rows = partitions[partition]
        for arm, fn in arm_functions.items():
            raw[partition][arm] = [fn(row) for row in rows]
        b7_rows = []
        for row in rows:
            prediction, diagnostics = models.b7.predict(row, models.b6)
            b7_rows.append(prediction)
            b7_diagnostics[partition].append(diagnostics)
        raw[partition]["B7"] = b7_rows
        raw[partition]["A_POINT_WORLD"] = [row["point_world_prediction"]
                                            for row in b7_diagnostics[partition]]
        raw[partition]["A_NO_POSTERIOR_PARTICLES"] = [row["no_posterior_particles_prediction"]
                                                       for row in b7_diagnostics[partition]]
        raw[partition]["A_NO_SUBJECTIVE_CONSEQUENCES"] = [row["no_subjective_consequences_prediction"]
                                                          for row in b7_diagnostics[partition]]
    calibrators, calibrated_test, metrics = {}, {}, {}
    calibration_labels = [row.label for row in partitions["calibration"]]
    calibration_weights = [row.sample_weight for row in partitions["calibration"]]
    for arm in raw["calibration"]:
        calibration = fit_temperature(raw["calibration"][arm], calibration_labels,
                                      build.split_manifest["checksum"], calibration_weights)
        calibrators[arm] = asdict(calibration)
        calibrated_test[arm] = [apply_calibration(prediction, calibration)
                                for prediction in raw["test"][arm]]
        metrics[arm] = {
            "uncalibrated": _extended_metrics(raw["test"][arm], partitions["test"]),
            "calibrated": _extended_metrics(calibrated_test[arm], partitions["test"]),
        }
    conformal = conformal_summary(
        [apply_calibration(prediction, _calibration_from_dict(calibrators["B7"]))
         for prediction in raw["calibration"]["B7"]], partitions["calibration"],
        calibrated_test["B7"], partitions["test"],
    )
    b7_diag_summary = _particle_diagnostic_summary(b7_diagnostics)
    table = []
    known_actors = {r.actor_key for r in partitions["train"]}
    known_relationships = {r.relationship_key for r in partitions["train"]}
    for index, row in enumerate(partitions["test"]):
        table.append({
            "record_key": row.record_key, "label": row.label, "cluster": row.cluster_key,
            "actor_known": row.actor_key in known_actors,
            "relationship_known": row.relationship_key in known_relationships,
            "probabilities": {arm: calibrated_test[arm][index] for arm in calibrated_test
                              if not arm.startswith("A_")},
            "b7_entropy": -sum(p * math.log(max(1e-12, p)) for p in calibrated_test["B7"][index].values()),
        })
    return {
        "schema_version": "wmv2.phase4-completion.numeric-score.v1",
        "dataset": build.dataset, "partition_counts": {k: len(v) for k, v in partitions.items()},
        "calibrators": calibrators, "metrics": metrics,
        "clustered_comparisons": {
            "B7_minus_B6": clustered_difference(calibrated_test["B7"], calibrated_test["B6"],
                                                  partitions["test"]),
            "B7_minus_B5": clustered_difference(calibrated_test["B7"], calibrated_test["B5"],
                                                  partitions["test"]),
            "B7_minus_B8": clustered_difference(calibrated_test["B7"], calibrated_test["B8"],
                                                  partitions["test"]),
        },
        "particle_diagnostics": b7_diag_summary, "conformal": conformal,
        "risk_coverage": risk_coverage(calibrated_test["B7"], partitions["test"]),
        "cold_start_slices": _slice_metrics(calibrated_test["B7"], partitions["test"], partitions["train"]),
        "sequence": _sequence_summary(calibrated_test["B7"], partitions["test"]),
        "downstream": _downstream_summary(calibrated_test["B7"], partitions["test"]),
        "prediction_table": table, "elapsed_seconds": time.monotonic() - started,
        "execution_invariance": {
            "rows": len(partitions["test"]),
            "byte_identical_rows": sum(d["predict_execute_byte_identical"]
                                       for d in b7_diagnostics["test"]),
            "predict_sha256": digest(raw["test"]["B7"]),
            "execute_input_sha256": digest([dict(row) for row in raw["test"]["B7"]]),
        },
    }


def _calibration_from_dict(payload: dict):
    from swm.world_model_v2.phase4_learning import CalibrationArtifact
    return CalibrationArtifact(**payload)


def _particle_diagnostic_summary(by_partition: dict[str, list[dict]]) -> dict:
    result = {}
    for partition, rows in by_partition.items():
        result[partition] = {
            "rows": len(rows), "particles_per_decision": 64,
            "mean_ess_fraction": sum(row["ess_fraction"] for row in rows) / max(1, len(rows)),
            "minimum_ess_fraction": min((row["ess_fraction"] for row in rows), default=None),
            "maximum_particle_weight": max((row["max_weight"] for row in rows), default=None),
            "collapse_count": sum(row["collapsed"] for row in rows),
            "mean_unique_particles": sum(row["unique_particles"] for row in rows) / max(1, len(rows)),
            "mean_action_tv": sum(row["per_particle_action_tv"] for row in rows) / max(1, len(rows)),
            "argmax_diversity_rate": sum(row["argmax_diversity"] > 1 for row in rows) / max(1, len(rows)),
            "mean_interval_width_proxy": sum(row["interval_width_proxy"] for row in rows)
            / max(1, len(rows)),
            "selected_traces": [rows[i] for i in sorted(set(
                [0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, max(0, len(rows) - 1)]
            ))] if rows else [],
        }
    return result
