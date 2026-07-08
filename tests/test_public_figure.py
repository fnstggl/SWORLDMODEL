"""Public-figure inference: we resolve a stranger instead of blocking on private data.

Covers the three claims of the change:
  1. VariableInferenceEngine folds web-provenance evidence into the map (bias to infer).
  2. PublicFigureResolver turns online evidence into confidence-tracked variables, and degrades to a
     transparent prior offline — it never blocks.
  3. World.predict on an unknown, unfitted contact returns an UNVALIDATED inference (not an error), and
     the folded web evidence actually moves the prediction.
"""
from swm.entities.public_figure import PublicFigureResolver, default_resolver
from swm.ingestion.store import EventStore
from swm.state.state import Action
from swm.variables.inference import VariableInferenceEngine
from swm.worlds.world import World


def _action(text="hi", channel="email"):
    return Action(action_id="a", actor_id="me", content_features={}, timing={"ts": 1_700_000_000},
                  channel=channel, meta={"text": text})


# --- 1. engine applies web-provenance inference ---------------------------------------------------

def test_engine_applies_web_inference_with_provenance():
    eng = VariableInferenceEngine(platform="email")
    web = {"openness_to_outreach": {"value": 0.85, "confidence": 0.6, "evidence": "backs young founders"}}
    vm = eng.infer("thiel", _action(), web_inference=web)
    v = vm.vars["openness_to_outreach"]
    assert v.provenance == "web"
    assert v.value > 0.8 and v.confidence == 0.6
    assert vm.provenance_report()["by_provenance"].get("web", 0) >= 1


def test_user_context_still_overrides_web():
    eng = VariableInferenceEngine(platform="email")
    web = {"status": {"value": 0.9, "confidence": 0.6}}
    vm = eng.infer("x", _action(), web_inference=web, user_context={"status": (0.2, 0.95)})
    # user provenance outranks web -> the fact you told us wins
    assert vm.vars["status"].provenance == "user"
    assert vm.get("status") == 0.2


# --- 2. resolver: online evidence -> variables; offline -> prior, never blocks --------------------

def test_resolver_offline_returns_prior_not_block():
    prof = default_resolver().resolve("Some Public Figure")
    assert prof.source == "prior"
    # bias to infer: a named public figure is high-status by default, at low confidence
    assert prof.web_variables["status"]["value"] >= 0.6
    assert prof.web_variables["status"]["confidence"] <= 0.4
    assert prof.responsiveness["mean"] > 0.0          # a usable base rate, not a refusal


def test_resolver_reads_web_evidence():
    def search(_q):
        return [
            {"title": "fellowship", "snippet": "backs young founders who drop out; discovered via cold email"},
            {"title": "style", "snippet": "contrarian, provocative, heterodox skeptic; challenges consensus"},
            {"title": "reach", "snippet": "hard to reach, screens heavily, rarely responds to cold outreach"},
        ]
    prof = PublicFigureResolver(search_fn=search).resolve("Peter Thiel")
    assert prof.source == "web+lexical" and len(prof.evidence) == 3
    v = prof.web_variables
    assert v["openness_to_outreach"]["value"] > 0.6      # fellowship signal
    assert v["skepticism"]["value"] > 0.6                # contrarian signal
    assert prof.responsiveness["mean"] < 0.28            # "rarely responds" pulls the base rate down
    assert prof.responsiveness["confidence"] > 0.15      # and we're more sure than the bare prior


def test_infer_fn_is_used_when_provided():
    called = {}
    def infer_fn(name, text, domain, ask):
        called["name"] = name
        return {"base_responsiveness": {"mean": 0.5, "confidence": 0.7, "evidence": "llm read"}}
    prof = PublicFigureResolver(search_fn=lambda q: [{"title": "t", "snippet": "s"}],
                                infer_fn=infer_fn).resolve("X")
    assert called["name"] == "X" and prof.source == "web+llm"
    assert prof.responsiveness["mean"] == 0.5


# --- 3. World.predict never blocks on an unknown contact -----------------------------------------

def test_predict_unfitted_is_unvalidated_not_error():
    w = World(store=EventStore(":memory:"), resolver=default_resolver())
    out = w.predict("stranger", "quick question — can I get 10 minutes?", name="Some Person")
    assert "error" not in out
    assert out["calibration"]["grade"] == "unvalidated"
    assert 0.0 < out["p_mean"] < 1.0
    assert out["report_type"] == "prediction"


def test_web_evidence_moves_the_prediction():
    def unresponsive(_q):
        return [{"title": "reach", "snippet": "unreachable, reclusive, never replies, does not respond"}]
    def responsive(_q):
        return [{"title": "reach", "snippet": "responds to founders, accessible, reads every email, emails back"}]

    low = World(store=EventStore(":memory:"), resolver=PublicFigureResolver(search_fn=unresponsive))
    high = World(store=EventStore(":memory:"), resolver=PublicFigureResolver(search_fn=responsive))
    msg = "hello — would love your take on this?"
    p_low = low.predict("a", msg, name="Recluse")["p_mean"]
    p_high = high.predict("b", msg, name="Reply Guy")["p_mean"]
    assert p_high > p_low                                # observed public behavior shifts the estimate
    assert low.profile("a")["responsiveness"]["mean"] < high.profile("b")["responsiveness"]["mean"]


def test_compare_ranks_without_a_fit():
    w = World(store=EventStore(":memory:"), resolver=default_resolver())
    good = "Loved your essay on X — one contrarian question, is thesis Y obviously wrong to you?"
    bad = "URGENT: please respond ASAP, just following up, circling back per my last email."
    ranked = w.compare("c", [bad, good], name="Someone")["ranked"]
    assert ranked[0]["text"] == good                    # low-friction personalized ask wins
