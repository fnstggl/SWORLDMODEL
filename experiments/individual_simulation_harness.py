"""Individualized simulation harness (Phase 11): recipient_state_t + message_t + context_t
-> simulate recipient attention/reaction/stance over steps -> response distribution -> state_t+1.

Mirrors the aggregate simulation for the individual regime: instead of community segments, ONE
named recipient is simulated as an actor whose attention, fatigue, and relationship stance evolve
over a few steps (notice -> open -> consider -> reply/ignore). The response probability is the
FRACTION OF SIMULATED TRAJECTORIES that end in a reply — not a classifier over features.

Schema (ready for real private data): recipient_id, sender_id, message features, thread history,
prior replies/ignores, relationship state, timing, outcome (reply / positive_reply / ignore /
unsubscribe / sentiment). A loader + metrics + a SYNTHETIC smoke test are implemented. Real proof is
BLOCKED-ON-PRIVATE-DATA — no real outcomes are fabricated.

Baselines: segment mean, raw LLM message-only (BLOCKED), raw LLM + thread context (BLOCKED),
old classifier world model, new simulation world model, hybrid.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from swm.eval.metrics import brier_score, expected_calibration_error, log_loss
from swm.simulation.actors import IndividualActorState
from swm.simulation.policies import sample_poisson

RESULT = "experiments/results/exp013_individual_sim.json"
_SIG = lambda z: 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))  # noqa: E731


# ------------------------------------------------------------------ schema (real-data ready)
@dataclass
class OutboundMessage:
    recipient_id: str
    sender_id: str
    timestamp: float
    features: dict = field(default_factory=dict)     # length, ask_strength, personalization, pushiness...
    thread_len: int = 0
    prior_replies: int = 0
    prior_ignores: int = 0
    relationship: float = 0.0                          # warmth/standing [-1,1]
    outcome: str | None = None                         # "reply"|"positive_reply"|"ignore"|"unsubscribe"

    def to_dict(self) -> dict:
        return asdict(self)


def load_messages(path: str) -> list[OutboundMessage]:
    """Loader for a real corpus (JSONL of OutboundMessage dicts). Returns [] if absent (blocked)."""
    p = Path(path)
    if not p.exists():
        return []
    return [OutboundMessage(**json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ recipient reaction simulation
def simulate_reply(msg: OutboundMessage, *, n_samples: int = 200, seed: int = 0) -> dict:
    """Simulate the recipient over steps: notice -> open -> consider -> reply/ignore. Attention and
    fatigue evolve; relationship + message fit drive the reply hazard each step. P(reply) = fraction
    of trajectories ending in a reply."""
    rng = random.Random(seed)
    actor = IndividualActorState(
        actor_id=msg.recipient_id, sender_id=msg.sender_id,
        relationship_stance=msg.relationship, prior_replies=msg.prior_replies,
        prior_ignores=msg.prior_ignores, thread_len=msg.thread_len,
        fatigue=min(1.0, 0.15 * msg.thread_len))
    f = msg.features
    fit = (0.6 * f.get("personalization", 0.0) + 0.5 * f.get("ask_strength", 0.0)
           - 0.7 * f.get("pushiness", 0.0) - 0.3 * max(0.0, f.get("length", 0.5) - 0.6))
    replies = 0
    for _ in range(n_samples):
        attention = actor.attention * (1.0 - 0.5 * actor.fatigue)
        stance = actor.relationship_stance
        replied = False
        for step in range(3):                              # notice, open, consider
            # hazard of engaging this step, modulated by attention, relationship, fit, novelty decay
            base = actor.responsiveness_prior
            z = math.log(max(1e-6, base) / (1 - max(1e-6, base))) + 1.3 * fit + 0.8 * stance
            p_engage = _SIG(z) * attention * (0.7 ** step)
            if rng.random() < p_engage:
                # engaged this step: chance it converts to a reply rises with fit/stance
                if rng.random() < _SIG(0.5 + 1.2 * fit + 0.6 * stance):
                    replied = True
                    break
            attention *= 0.6                                # attention decays across steps
            stance += 0.05 * fit                            # a good message warms them slightly
        replies += 1 if replied else 0
    p = replies / n_samples
    return {"p_reply": p, "n_samples": n_samples}


# ------------------------------------------------------------------ synthetic smoke test
def _gen_synthetic(n_recipients=120, n_msgs=1500, seed=0):
    rng = random.Random(seed)
    theta = {f"r{i}": min(0.95, max(0.03, rng.betavariate(2, 5))) for i in range(n_recipients)}
    msgs, hist = [], {}
    for _ in range(n_msgs):
        rid = rng.choice(list(theta))
        h = hist.setdefault(rid, {"rep": 0, "ign": 0, "n": 0})
        pers = rng.random(); ask = rng.random(); push = rng.random(); length = rng.random()
        fit = 0.6 * pers + 0.5 * ask - 0.7 * push - 0.3 * max(0.0, length - 0.6)
        p = _SIG(math.log(theta[rid] / (1 - theta[rid])) + 1.3 * fit)
        outcome = "reply" if rng.random() < p else "ignore"
        msgs.append(OutboundMessage(rid, "me", float(h["n"]), {"personalization": pers, "ask_strength": ask,
                    "pushiness": push, "length": length}, thread_len=h["n"], prior_replies=h["rep"],
                    prior_ignores=h["ign"], outcome=outcome))
        h["n"] += 1
        h["rep" if outcome == "reply" else "ign"] += 1
    return msgs


def run():
    real = load_messages("data/individual_outbound.jsonl")
    if real:
        msgs, source = real, "REAL private corpus"
    else:
        msgs, source = _gen_synthetic(), "SYNTHETIC (real data BLOCKED-ON-PRIVATE-DATA)"
    n = len(msgs)
    cut = int(0.7 * n)
    train, test = msgs[:cut], msgs[cut:]
    y = [1 if m.outcome in ("reply", "positive_reply") else 0 for m in test]
    seg = (sum(1 for m in train if m.outcome in ("reply", "positive_reply")) + 1) / (len(train) + 2)

    # tiers
    sim_p = [simulate_reply(m, n_samples=150, seed=i)["p_reply"] for i, m in enumerate(test)]
    seg_p = [seg] * len(test)
    # simple per-recipient empirical (old classifier proxy via prior rate)
    emp_p = [(m.prior_replies + seg * 4) / (m.prior_replies + m.prior_ignores + 4) for m in test]

    def sc(p):
        p = [min(1 - 1e-6, max(1e-6, v)) for v in p]
        return {"log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
                "ece": round(expected_calibration_error(y, p), 4)}

    out = {
        "source": source, "n": n, "n_test": len(test), "test_base_rate": round(sum(y) / len(y), 4),
        "tiers": {
            "segment_mean": sc(seg_p),
            "recipient_empirical(old_classifier_proxy)": sc(emp_p),
            "individual_simulation": sc(sim_p),
            "raw_llm_message_only": "BLOCKED-ON-PRIVATE-DATA (needs real messages + an LLM predictor)",
            "raw_llm_thread_context": "BLOCKED-ON-PRIVATE-DATA",
            "hybrid": "BLOCKED (needs the LLM branch)",
        },
        "note": ("Estimator/engine validated on synthetic data; the REAL individual claim is "
                 "blocked on private outcome data. Schema + loader + metrics are ready "
                 "(data/individual_outbound.jsonl)."),
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"source: {source}\n n={n} test base rate {sum(y)/len(y):.3f}")
    for k, v in out["tiers"].items():
        print(f"  {k:<42} {v if isinstance(v,str) else v}")
    print(f"wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
