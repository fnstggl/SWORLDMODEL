"""Semantic action clustering v2 acceptance tests — offline, deterministic, scripted.

Covers: byte-compatibility with cluster-1.0 on rows v1 handled; deterministic target
canonicalization with refusal on ambiguity; the curated equivalence map (merging synonyms,
never materially different acts); strategy-class fallback; novel-action preservation;
unresolved keys only when target ambiguity meets an unknown action; mock-LLM assisted
mapping with strict refusals; deterministic replay from exported mappings (zero LLM calls);
metrics on the locked fixture with the deterministic clusterer; and the SHA-256 lock that
freezes the fixture as a grading artifact."""
import hashlib
import json
from pathlib import Path

import pytest

from swm.world_model_v2.phase4_policy import KNOWN_ACTIONS
from swm.world_model_v2.qualitative_actor import ActionClusterer
from swm.world_model_v2.semantic_clustering import (
    ACTION_EQUIVALENCE, CLUSTER_VERSION_2, NON_EQUIVALENT, ActionClustererV2,
    TargetCanonicalizer, clustering_metrics,
)

FIXTURE = Path("tests/fixtures/semantic_clustering_fixture_v1.json")
#: the lock: any edit to the hand-authored grading fixture must fail loudly here
FIXTURE_SHA256 = "16792039aa97dacafec7fb4a77255f08711d2e6b5239caa841058a8545fa1bef"

SMITHS = ("robert_smith", "jane_smith")


class MapLLM:
    """Scripted mapping backend. `verdict` is a dict, a JSON string, or fn(prompt)->verdict."""

    def __init__(self, verdict=None):
        self.verdict = verdict
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        out = self.verdict(prompt) if callable(self.verdict) else (self.verdict or {})
        return out if isinstance(out, str) else json.dumps(out)


def fixture():
    return json.loads(FIXTURE.read_text())


# ---------------------------------------------------------------- v1 compatibility
def test_rows_v1_handled_produce_identical_v2_keys():
    v1, v2 = ActionClusterer(), ActionClustererV2()
    rows = [
        {"action_name": "approve", "target": "board"},
        {"action_name": "accept", "target": "bob"},
        {"action_name": "wait", "target": ""},
        {"action_name": "delay", "target": ""},
        {"action_name": "escalate_the_campaign", "target": "bob",
         "ontology_anchor": {"family": "negotiation", "name": "escalate",
                             "matched_by": "token_overlap"}},
    ]
    for row in rows:
        assert v2.cluster_key(row) == v1.cluster_key(row)
        # supplying the world's entity ids must not change an already-exact key
        assert v2.cluster_key(row, known_entities=("board", "bob")) == v1.cluster_key(row)
    assert ActionClustererV2.version == CLUSTER_VERSION_2 == "cluster-2.0"
    assert v2.cluster_record(rows[0])["version"] == CLUSTER_VERSION_2


# ---------------------------------------------------------------- target canonicalization
def test_target_canonicalizer_resolves_and_records():
    tc = TargetCanonicalizer(known_entities=["robert_smith"],
                             aliases={"POTUS": "president_karev"})
    for raw, method in (("robert_smith", "exact"), ("Robert Smith", "normalized_id"),
                        ("Mr. Smith", "token_unique"), ("Bob Smith", "token_unique"),
                        ("POTUS", "alias"), ("potus", "alias")):
        rec = tc.resolve(raw)
        assert set(rec) == {"raw", "canonical", "method", "resolved"}
        assert rec["method"] == method and rec["resolved"] is True
        assert rec["canonical"] == ("president_karev" if "otus" in raw.lower()
                                    else "robert_smith")
    assert tc.resolve("") == {"raw": "", "canonical": "", "method": "empty", "resolved": True}
    # unmatched targets keep their normalized (honorific-stripped) form, marked unresolved
    rec = tc.resolve("Ms. Chen")
    assert rec == {"raw": "Ms. Chen", "canonical": "chen", "method": "unmatched",
                   "resolved": False}


def test_target_canonicalizer_refuses_ambiguous_matches():
    tc = TargetCanonicalizer()
    rec = tc.resolve("smith", known_entities=SMITHS)
    assert rec["method"] == "ambiguous" and rec["resolved"] is False
    assert rec["canonical"] == "smith"                      # original preserved, no guessing
    # equally-shared surname stays ambiguous even with a first-name token in play
    assert tc.resolve("Bob Smith", known_entities=SMITHS)["method"] == "ambiguous"
    # a full-name match dominates the tie and resolves uniquely
    rec = tc.resolve("Robert Smith", known_entities=SMITHS)
    assert rec["canonical"] == "robert_smith" and rec["resolved"] is True


def test_canonical_targets_defragment_paraphrases():
    c = ActionClustererV2()
    keys = {c.cluster_key({"action_name": "accept", "target": t},
                          known_entities=("robert_smith",))
            for t in ("robert_smith", "Mr. Smith", "Bob Smith", "ROBERT SMITH")}
    assert keys == {"accept@robert_smith"}                  # the pilot's fragmentation, gone
    assert c.cluster_key({"action_name": "accept", "target": "the Kremlin"},
                         aliases={"the Kremlin": "russian_government"}) == \
        "accept@russian_government"


# ---------------------------------------------------------------- equivalence map
def test_equivalence_map_is_curated_and_conservative():
    assert len(ACTION_EQUIVALENCE) >= 40
    assert set(ACTION_EQUIVALENCE.values()) <= set(KNOWN_ACTIONS)   # targets are ontology
    assert not set(ACTION_EQUIVALENCE) & set(KNOWN_ACTIONS)         # keys never shadow it
    assert not set(ACTION_EQUIVALENCE) & NON_EQUIVALENT             # exclusions honored
    c = ActionClustererV2()
    for free, ontology in (("refuse", "reject"), ("decline_the_offer", "reject"),
                           ("postpone", "delay"), ("publicly_endorse_the_deal", "support"),
                           ("walk_out", "strike"), ("walk_away", "exit"),
                           ("pull_out", "withdraw"), ("do_nothing", "wait")):
        rec = c.cluster_record({"action_name": free, "target": "bob"},
                               known_entities=("bob",))
        assert rec["key"] == f"{ontology}@bob"
        assert rec["method"] in ("equivalence", "lexical")
        assert rec["original"]["action_name"] == free       # phrasing preserved for audit
    # materially different acts are NEVER merged by the deterministic tiers
    assert c.cluster_key({"action_name": "threaten", "target": "bob"}) == "novel:threaten@bob"
    assert c.cluster_key({"action_name": "leak_documents", "target": "press"}) == \
        "novel:leak_documents@press"
    assert c.cluster_key({"action_name": "threaten_to_exit", "target": ""}) == \
        "novel:threaten_to_exit"


# ---------------------------------------------------------------- fallback tiers
def test_strategy_class_fallback_clusters_by_family():
    c = ActionClustererV2()
    rec = c.cluster_record({"action_name": "seek_arbitration", "target": "bob"},
                           known_entities=("bob",))
    assert rec["key"] == "family:negotiation@bob" and rec["method"] == "family"
    # two phrasings of the same strategy land in one class cluster
    assert c.cluster_key({"action_name": "seek_binding_arbitration", "target": "bob"},
                         known_entities=("bob",)) == "family:negotiation@bob"
    # ...but the class never swallows the exact ontology action
    assert c.cluster_key({"action_name": "seek_mediator", "target": "bob"},
                         known_entities=("bob",)) == "seek_mediator@bob"


def test_novel_actions_keep_their_own_cluster_with_original_text():
    c = ActionClustererV2()
    rec = c.cluster_record({"action_name": "Propose Joint Task-Force", "target": "council"},
                           known_entities=("council",))
    assert rec["key"] == "novel:propose_joint_task_force@council"
    assert rec["method"] == "novel"
    assert rec["original"] == {"action_name": "Propose Joint Task-Force",
                               "target": "council", "ontology_anchor": None}
    assert c.cluster_key({"action_name": "propose_joint_task_force", "target": "council"},
                         known_entities=("council",)) == rec["key"]   # self-clusters


def test_unresolved_requires_ambiguous_target_and_unknown_action():
    c = ActionClustererV2()
    rec = c.cluster_record({"action_name": "brief_stakeholders", "target": "smith"},
                           known_entities=SMITHS)
    assert rec["key"] == "unresolved:brief_stakeholders" and rec["method"] == "unresolved"
    assert rec["target_resolution"]["method"] == "ambiguous"
    # a KNOWN action with an ambiguous target keeps the raw target — not unresolved
    rec = c.cluster_record({"action_name": "accept", "target": "smith"},
                           known_entities=SMITHS)
    assert rec["key"] == "accept@smith" and rec["method"] == "exact"
    assert rec["target_resolution"]["resolved"] is False
    # an unknown action with a RESOLVED target is novel — not unresolved
    assert c.cluster_key({"action_name": "brief_stakeholders", "target": "jane_smith"},
                         known_entities=SMITHS) == "novel:brief_stakeholders@jane_smith"


# ---------------------------------------------------------------- LLM-assisted equivalence
def test_llm_assisted_mapping_accepts_only_candidate_verdicts():
    llm = MapLLM({"maps_to": "hold_position", "justification": "same act of not budging",
                  "confidence": "high", "materially_different": False})
    c = ActionClustererV2(llm=llm)
    rec = c.cluster_record({"action_name": "hold_the_line", "target": "bob"},
                           known_entities=("bob",))
    assert rec["key"] == "hold_position@bob" and rec["method"] == "llm"
    assert c.llm_calls == 1 and ("hold_the_line", "bob") in c.llm_cache
    prompt = llm.prompts[0]
    assert "hold_the_line" in prompt and "- hold_position" in prompt
    assert "ONLY choose" in prompt and '"maps_to"' in prompt
    # the cache answers the second identical row — no second call
    assert c.cluster_key({"action_name": "hold_the_line", "target": "bob"},
                         known_entities=("bob",)) == "hold_position@bob"
    assert c.llm_calls == 1


def test_llm_mapping_refusals_materially_different_low_confidence_off_menu():
    refusals = (
        {"maps_to": "escalate_message", "justification": "a threat is coercive",
         "confidence": "high", "materially_different": True},
        {"maps_to": "hold_position", "justification": "maybe",
         "confidence": "low", "materially_different": False},
        {"maps_to": "capitulate_gracefully", "justification": "off the menu",
         "confidence": "high", "materially_different": False},
    )
    for verdict in refusals:
        c = ActionClustererV2(llm=MapLLM(verdict))
        key = c.cluster_key({"action_name": "stand_our_ground", "target": "bob"},
                            known_entities=("bob",))
        assert key == "novel:stand_our_ground@bob"          # refused → never merged
        cached = c.llm_cache[("stand_our_ground", "bob")]
        assert cached["accepted"] is False and cached["maps_to"] == ""
    # a backend crash is not a verdict: nothing cached, nothing merged
    def boom(prompt):
        raise RuntimeError("backend down")
    c = ActionClustererV2(llm=boom)
    assert c.cluster_key({"action_name": "stand_our_ground", "target": ""}) == \
        "novel:stand_our_ground"
    assert c.llm_cache == {}


def test_deterministic_replay_from_exported_mappings():
    accept = {"maps_to": "hold_position", "justification": "same act",
              "confidence": "high", "materially_different": False}
    c1 = ActionClustererV2(llm=MapLLM(accept))
    row = {"action_name": "hold_the_line", "target": "bob"}
    refused_row = {"action_name": "stand_our_ground", "target": "bob"}
    key1 = c1.cluster_key(row, known_entities=("bob",))
    c2_seed = ActionClustererV2(llm=MapLLM({"maps_to": "NONE", "justification": "novel",
                                            "confidence": "high",
                                            "materially_different": True}))
    refused_key = c2_seed.cluster_key(refused_row, known_entities=("bob",))
    exported = json.loads(json.dumps({**c1.export_mappings(),
                                      **c2_seed.export_mappings()}))   # JSON round-trip
    counting = MapLLM(accept)
    replay = ActionClustererV2(llm=counting)
    replay.load_mappings(exported)
    assert replay.cluster_key(row, known_entities=("bob",)) == key1
    assert replay.cluster_key(refused_row, known_entities=("bob",)) == refused_key
    assert counting.prompts == [] and replay.llm_calls == 0    # replay is LLM-free
    # every decision is on the log with version, method, inputs and outputs
    assert len(replay.mapping_log) == 2
    for entry in replay.mapping_log:
        assert entry["version"] == CLUSTER_VERSION_2
        assert entry["method"] and set(entry) == {"version", "method", "inputs", "outputs"}
        assert entry["inputs"]["action_name"] and entry["outputs"]["key"]


# ---------------------------------------------------------------- the locked fixture
def test_metrics_on_locked_fixture_with_deterministic_clusterer():
    fx = fixture()
    assert fx["version"] == "fixture-v1" and fx["locked"] is True
    metrics = clustering_metrics(fx["cases"], ActionClustererV2())   # llm=None: deterministic
    print("deterministic cluster-2.0 fixture metrics:", metrics)
    assert metrics == {"exact_accuracy": 0.3864, "semantic_accuracy": 0.9773,
                       "false_merge_rate": 0.0, "false_split_rate": 0.0333,
                       "unresolved_rate": 0.0455, "n": 44}
    # the repair the pilot asked for: semantic clustering beats exact-string clustering
    # without ever merging cases the fixture marks materially different
    assert metrics["semantic_accuracy"] > metrics["exact_accuracy"]
    assert metrics["false_merge_rate"] == 0.0


def test_fixture_file_is_locked_by_sha256():
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        "the locked grading fixture changed; if the edit was reviewed and intended, "
        "re-lock by updating FIXTURE_SHA256")
    assert len(fixture()["cases"]) == 44
    assert {c["expected"] for c in fixture()["cases"]} == {"same", "different", "unresolved"}
