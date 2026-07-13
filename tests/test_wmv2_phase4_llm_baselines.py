import json

import pytest

from swm.world_model_v2.phase4_learning import read_artifact
from swm.world_model_v2.phase4_llm_baselines import (
    LENSES, RESPONSE_SCHEMA, ResponseEnvelope, assert_complete_collection,
    build_actor_visible_packet, collect_one, collection_manifest, logarithmic_pool,
    request_identity, strict_parse_action_response,
)


ACTIONS = ["cooperate", "defect"]


def response(probabilities=None):
    return json.dumps({
        "schema_version": RESPONSE_SCHEMA,
        "probabilities": probabilities or {"cooperate": 0.4, "defect": 0.6},
        "reason": "visible history is mixed", "uncertainty": "moderate",
    }, separators=(",", ":"))


def packet():
    return build_actor_visible_packet(
        decision_time="session=s;round=3", actor_role="participant",
        visible_state={"history": {"previous": "cooperate"},
                       "relationships": {"fixed": True}}, actions=ACTIONS,
    )


@pytest.mark.parametrize("raw", [
    "prefix " + response(),
    "```json\n" + response() + "\n```",
    '{"schema_version":"phase4.llm-action-response.v1","schema_version":"x",'
    '"probabilities":{"cooperate":.4,"defect":.6},"reason":"x","uncertainty":"x"}',
    response({"cooperate": True, "defect": 0.0}),
    response({"cooperate": float("nan"), "defect": 0.5}),
    response({"cooperate": 0.4}),
    response({"cooperate": 0.4, "defect": 0.5}),
])
def test_strict_parser_rejects_malformed_or_repaired_outputs(raw):
    with pytest.raises(ValueError):
        strict_parse_action_response(raw, ACTIONS)


def test_strict_parser_accepts_only_exact_distribution():
    parsed = strict_parse_action_response(response(), ACTIONS)
    assert parsed["probabilities"] == {"cooperate": 0.4, "defect": 0.6}


def test_packet_boundary_rejects_identifiers_labels_and_outcomes():
    rendered = json.dumps(packet(), sort_keys=True)
    assert "record_id" not in rendered and "observed_action" not in rendered
    with pytest.raises(ValueError, match="non-whitelisted"):
        build_actor_visible_packet(decision_time="t", actor_role="r",
                                   visible_state={"outcome": "defect"}, actions=ACTIONS)


def test_request_hash_changes_with_lens_and_visible_state():
    first, _ = request_identity(packet(), LENSES[0], code_commit="a",
                                dataset_manifest_hash="m", split_checksum="s")
    second, _ = request_identity(packet(), LENSES[1], code_commit="a",
                                 dataset_manifest_hash="m", split_checksum="s")
    changed = packet()
    changed["visible_state"]["history"]["previous"] = "defect"
    third, _ = request_identity(changed, LENSES[0], code_commit="a",
                                dataset_manifest_hash="m", split_checksum="s")
    assert len({first, second, third}) == 3


class FakeClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return ResponseEnvelope(next(self.contents), "provider-id", "frozen-model", {"total_tokens": 9}, 4.2)


def test_raw_is_durable_before_failed_parse_and_cache_replays_without_call(tmp_path):
    client = FakeClient(["not-json", response()])
    result = collect_one(client=client, packet=packet(), lens=LENSES[0], raw_root=tmp_path,
                         code_commit="a", dataset_manifest_hash="m", split_checksum="s",
                         retries=1, sleeper=lambda _: None)
    assert result["valid"] and client.calls == 2
    first = read_artifact(result["attempts"][0]["path"])
    assert first["raw_content"] == "not-json" and not first["secret_fields_persisted"]
    replay = FakeClient([])
    again = collect_one(client=replay, packet=packet(), lens=LENSES[0], raw_root=tmp_path,
                        code_commit="a", dataset_manifest_hash="m", split_checksum="s",
                        retries=1, sleeper=lambda _: None)
    assert again["valid"] and replay.calls == 0


def test_panel_pool_and_manifest_fail_closed():
    pooled = logarithmic_pool([{"cooperate": 0.8, "defect": 0.2},
                               {"cooperate": 0.2, "defect": 0.8}], ACTIONS)
    assert pooled == pytest.approx({"cooperate": 0.5, "defect": 0.5})
    manifest = collection_manifest([], expected_request_hashes=["missing"])
    with pytest.raises(ValueError, match="incomplete"):
        assert_complete_collection(manifest)
