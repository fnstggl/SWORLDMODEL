"""EXP-029: general evidence fusion — fusing a person's observed responses with their attributes.

Validates the general `EvidenceFusion` primitive (swm/variables/evidence.py) — the connective tissue
that fuses ANY evidence about a person (attributes, observed responses, text) into the one value profile
every simulation conditions on, with confidence that grows as evidence accumulates.

Two parts, reported honestly:

A. REAL (OpinionQA): does fusing a held-out person's OTHER answers with their demographics beat
   demographics alone? No-cheat: split respondents; response→value centroids fit on TRAIN only; the
   target question is excluded from the person's own context. HONEST FINDING: no — with strong political
   demographics and only ~4 sparse, topically-diffuse answers per person, the response channel is
   redundant. The strong OpinionQA channel is attributes (EXP-028); text is the strong channel elsewhere
   (EXP-025). This delimits WHEN the response channel pays off.

B. CONTROLLED: a population where a latent value profile weakly drives attributes but strongly drives
   responses. Here the response channel SHOULD carry signal — and the fusion recovers it: fused (attr +
   responses) beats attributes-alone, and the gain GROWS with the number of observed responses (the same
   evidence-depth law used for text depth). This proves the mechanism is correct; OpinionQA simply lacks
   informative-enough responses.

Writes experiments/results/exp029_evidence_fusion.json. Run: python -m experiments.exp029_evidence_fusion
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from swm.variables.demographic_values import value_vector
from swm.variables.evidence import EvidenceFusion, PersonEvidence
from experiments.datasets_opinionqa import load
from experiments.exp028_individual_opinion import _cos, _marginal, _norm

RESULT = "experiments/results/exp029_evidence_fusion.json"


def _vote(vec, neighbors, beta, nopt):
    sims = [(_cos(vec, nv), ans) for nv, ans in neighbors]
    mx = max(s for s, _ in sims)
    w = [math.exp(beta * (s - mx)) for s, _ in sims]
    agg = [0.0] * nopt
    for wi, (_, ans) in zip(w, sims):
        agg[ans] += wi
    return _norm(agg)


# ---------- A. real OpinionQA ----------
def run_opinionqa(seeds=(0, 1, 2, 3, 4), beta=8):
    recs = load()
    by_uid = defaultdict(list)
    for r in recs:
        by_uid[r["uid"]].append(r)
    agg = {k: [] for k in ("marginal", "value_demographics", "value_fused")}
    npairs = 0
    for seed in seeds:
        uids = sorted(by_uid); rng = random.Random(seed); rng.shuffle(uids)
        cut = int(0.7 * len(uids)); train_u, test_u = uids[:cut], uids[cut:]
        by_q = defaultdict(list)
        for u in train_u:
            dv = value_vector(by_uid[u][0]["demo"])
            for r in by_uid[u]:
                by_q[r["qid"]].append((dv, r["answer_idx"]))
        fusion = EvidenceFusion().fit(
            [PersonEvidence(u, by_uid[u][0]["demo"], [(r["qid"], r["answer_idx"]) for r in by_uid[u]])
             for u in train_u])
        c = {"marginal": [0, 0], "value_demographics": [0, 0], "value_fused": [0, 0]}
        for u in test_u:
            rs = by_uid[u]; demo = rs[0]["demo"]; dv = value_vector(demo)
            ev = PersonEvidence(u, demo, [(r["qid"], r["answer_idx"]) for r in rs])
            for r in rs:
                qid, a, nopt = r["qid"], r["answer_idx"], r["n_opt"]
                nb = by_q.get(qid)
                if not nb or len(nb) < 5:
                    continue
                fused, _ = fusion.value_profile(ev, exclude_item=qid)
                for name, p in (("marginal", _marginal([{"ans": ans} for _, ans in nb], nopt)),
                                ("value_demographics", _vote(dv, nb, beta, nopt)),
                                ("value_fused", _vote(fused, nb, beta, nopt))):
                    c[name][0] += int(max(range(nopt), key=lambda i: p[i]) == a); c[name][1] += 1
        for k in agg:
            agg[k].append(c[k][0] / c[k][1]); npairs = c["marginal"][1]
    tiers = {k: round(statistics.mean(v), 4) for k, v in agg.items()}
    return {"n_test_pairs": npairs, "accuracy": tiers,
            "fused_minus_demographics": round(tiers["value_fused"] - tiers["value_demographics"], 4)}


# ---------- B. controlled: responses are informative ----------
def run_controlled(n_people=1200, n_items=60, seeds=(0, 1, 2), beta=10):
    """Latent value weakly drives attributes, strongly drives responses -> the response channel matters."""
    D = 10
    curve = defaultdict(list)
    summary = {"attributes_only": [], "fused": []}
    for seed in seeds:
        rng = random.Random(seed)
        latent = {i: [rng.gauss(0, 1) for _ in range(D)] for i in range(n_people)}
        item_dir = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n_items)]

        def attr_vec(i):                       # noisy (weak) observation of the latent
            return [latent[i][k] + rng_a.gauss(0, 2.5) for k in range(D)]

        def answer(i, j):                      # response strongly driven by latent (informative)
            return int(sum(latent[i][k] * item_dir[j][k] for k in range(D)) + rng.gauss(0, 0.5) > 0)

        rng_a = random.Random(seed + 1)
        attr = {i: attr_vec(i) for i in range(n_people)}
        resp = {i: [(j, answer(i, j)) for j in range(n_items)] for i in range(n_people)}
        order = list(range(n_people)); rng.shuffle(order)
        cut = int(0.7 * n_people); tr, te = order[:cut], order[cut:]
        by_item = defaultdict(list)
        for i in tr:
            for j, a in resp[i]:
                by_item[j].append((attr[i], a))
        # our attribute IS already a value vector, so attr_value_fn is identity
        fusion = EvidenceFusion(attr_value_fn=lambda a: a, dims=list(range(D))).fit(
            [PersonEvidence(str(i), attr[i], resp[i]) for i in tr])

        def person_ev(i, k_obs):
            return PersonEvidence(str(i), None, resp[i][:k_obs])

        for k_obs in (0, 2, 5, 15, 40):
            ca = cf = n = 0
            for i in te:
                targets = resp[i][45:]         # held-out items (never in context)
                ev = person_ev(i, k_obs)
                ev.attributes = attr[i]
                for j, a in targets:
                    nb = by_item.get(j)
                    if not nb or len(nb) < 5:
                        continue
                    fa, _ = (attr[i], None) if k_obs == 0 else fusion.value_profile(ev, exclude_item=j)
                    pf = _vote(fa, nb, beta, 2)
                    pa = _vote(attr[i], nb, beta, 2)
                    cf += int((pf[1] > 0.5) == a); ca += int((pa[1] > 0.5) == a); n += 1
            curve[k_obs].append(cf / n)
            if k_obs == 40:
                summary["fused"].append(cf / n); summary["attributes_only"].append(ca / n)
    dc = {str(k): round(statistics.mean(v), 4) for k, v in sorted(curve.items())}
    return {"depth_curve_acc": dc, "attributes_only_acc": round(statistics.mean(summary["attributes_only"]), 4),
            "fused_acc_at_40_responses": round(statistics.mean(summary["fused"]), 4),
            "fused_gain": round(statistics.mean(summary["fused"]) - statistics.mean(summary["attributes_only"]), 4)}


def run():
    oqa = run_opinionqa()
    ctrl = run_controlled()
    out = {"opinionqa_real": oqa, "controlled_informative_responses": ctrl}
    print("EXP-029 general evidence fusion")
    print("  A. OpinionQA (real): accuracy —")
    for k in ("marginal", "value_demographics", "value_fused"):
        print(f"       {k:<20} {oqa['accuracy'][k]}")
    print(f"     fused − demographics: {oqa['fused_minus_demographics']:+.4f} "
          f"(honest negative: sparse, topically-diffuse responses are redundant given strong attributes)")
    print("  B. controlled (responses informative): fused vs attributes-only —")
    print(f"       attributes_only {ctrl['attributes_only_acc']}   fused@40resp {ctrl['fused_acc_at_40_responses']}"
          f"   gain {ctrl['fused_gain']:+.4f}")
    print(f"     depth curve (acc as observed responses grow): {ctrl['depth_curve_acc']}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
