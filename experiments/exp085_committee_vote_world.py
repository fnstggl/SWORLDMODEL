"""EXP-085 — The committee-vote world model: does modeling every member as an agent with MEASURED state
beat the compact party shortcut?

This is the fair test the Maximal World experiment (EXP-083/084) was missing. There, agents carried GUESSED
micro-state and lost. The critique that followed: real fidelity needs MEASURED state (grounding — the +58pt
lever from EXP-082), and a question whose ceiling is high enough that the fidelity shows up. Committee votes
are that case: we can MEASURE each senator's ideology from their real prior-congress voting record, model
them as agents, and predict how the chamber divides — a high-ceiling, structure-dominated question.

Setup (leakage-free): for each divided Senate roll-call, split members 80/20. Fit each model on the observed
80%, predict the held-out 20% — who were never seen for this bill. Every member's ideology is GROUNDED from
the PRIOR congress (measured before this vote, not fit from it).

  BASE RATE  — predict every held-out member by the observed chamber-wide yea-rate. No member identity.
  COMPACT (party shortcut) — predict a held-out member by the observed yea-rate WITHIN THEIR PARTY. The strong,
               simple baseline: "everyone votes their party line." In a polarized chamber this is ~90% right.
  AGENT WORLD — each member is an agent with a MEASURED ideal point; fit a 1-D spatial vote model
               P(yea)=logistic(w·ideology + b) on the observed 80%, predict the held-out member from THEIR
               measured ideology. This is "model every person as an agent," now with grounded state.

The thesis wins if the agent world beats the party shortcut — especially on the CONTESTED votes (within
~60-40), where party-line breaks and individual measured positions decide the outcome. Scored on held-out
member votes (accuracy + Brier) and on the chamber MARGIN, sliced by competitiveness.
Run: python -m experiments.exp085_committee_vote_world   (cache: build_congress_votes first)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from swm.transition.readout import LogisticReadout

DATA = "experiments/results/exp085/senate_bills.json"
RESULT = "experiments/results/exp085_committee_vote_world.json"


def _split(members, rng):
    idx = list(range(len(members)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    obs = [members[i] for i in idx[:cut]]
    hold = [members[i] for i in idx[cut:]]
    return obs, hold


def _agent_model(obs):
    """1-D spatial vote model fit on observed members' MEASURED ideology (the grounded agent world)."""
    X = [[m["x"]] for m in obs]
    y = [m["vote"] for m in obs]
    return LogisticReadout(l2=0.05, epochs=200).fit(X, y)


def _party_rates(obs):
    rate = {}
    for p in set(m["party"] for m in obs):
        v = [m["vote"] for m in obs if m["party"] == p]
        rate[p] = sum(v) / len(v) if v else 0.5
    return rate


def run():
    bills = json.loads(Path(DATA).read_text())
    rng = random.Random(0)

    # accumulate held-out predictions
    acc = {k: {"hit": 0, "brier": 0.0, "n": 0} for k in ("base", "party", "agent")}
    acc_c = {k: {"hit": 0, "brier": 0.0, "n": 0} for k in ("base", "party", "agent")}   # contested only
    defect = {k: {"hit": 0, "n": 0} for k in ("party", "agent")}   # members who broke their party line
    margin = {k: 0.0 for k in ("base", "party", "agent")}
    margin_c = {k: 0.0 for k in ("base", "party", "agent")}
    n_bills = n_bills_c = 0

    for b in bills:
        ms = b["members"]
        if len(ms) < 40:
            continue
        obs, hold = _split(ms, rng)
        if not hold or len(set(m["vote"] for m in obs)) < 2:
            continue
        base_rate = sum(m["vote"] for m in obs) / len(obs)
        prate = _party_rates(obs)
        model = _agent_model(obs)

        preds = {"base": [], "party": [], "agent": []}
        truth = []
        for m in hold:
            truth.append(m["vote"])
            preds["base"].append(base_rate)
            preds["party"].append(prate.get(m["party"], base_rate))
            preds["agent"].append(model.predict_proba([m["x"]]))

        for k in preds:
            for p, t in zip(preds[k], truth):
                acc[k]["hit"] += int((p >= 0.5) == (t == 1))
                acc[k]["brier"] += (p - t) ** 2
                acc[k]["n"] += 1
        # DEFECTORS: held-out members who voted against their party's observed majority — the party shortcut
        # gets these wrong by construction; the grounded agent can catch some (the fidelity win, made vivid).
        for m in hold:
            pmaj = 1 if prate.get(m["party"], base_rate) >= 0.5 else 0
            if m["vote"] != pmaj:
                defect["party"]["hit"] += int(pmaj == m["vote"])                       # always 0
                defect["party"]["n"] += 1
                defect["agent"]["hit"] += int((model.predict_proba([m["x"]]) >= 0.5) == (m["vote"] == 1))
                defect["agent"]["n"] += 1
        # chamber margin: predicted held-out yea-fraction vs actual
        actual_frac = sum(truth) / len(truth)
        for k in preds:
            margin[k] += abs(sum(preds[k]) / len(preds[k]) - actual_frac)
        n_bills += 1

        if b["contested"]:
            for k in preds:
                for p, t in zip(preds[k], truth):
                    acc_c[k]["hit"] += int((p >= 0.5) == (t == 1))
                    acc_c[k]["brier"] += (p - t) ** 2
                    acc_c[k]["n"] += 1
                margin_c[k] += abs(sum(preds[k]) / len(preds[k]) - actual_frac)
            n_bills_c += 1

    def summarize(a, m, nb):
        return {k: {"accuracy": round(a[k]["hit"] / a[k]["n"], 4),
                    "brier": round(a[k]["brier"] / a[k]["n"], 4),
                    "margin_mae": round(m[k] / nb, 4)} for k in a}

    allb = summarize(acc, margin, n_bills)
    cont = summarize(acc_c, margin_c, n_bills_c)
    defector = {k: {"accuracy": round(defect[k]["hit"] / defect[k]["n"], 4), "n": defect[k]["n"]}
                for k in defect}
    out = {"experiment": "Committee-vote world model — grounded per-member agents vs the party shortcut",
           "data": f"VoteView Senate divided roll-calls, {n_bills} bills ({n_bills_c} contested); ideology "
                   f"grounded from PRIOR congress; held-out 20% of members per bill, leakage-free",
           "ALL_VOTES": allb, "CONTESTED_VOTES": cont, "DEFECTORS_against_party_line": defector,
           "agent_vs_party": {"all_accuracy": round(allb["agent"]["accuracy"] - allb["party"]["accuracy"], 4),
                              "contested_accuracy": round(cont["agent"]["accuracy"] - cont["party"]["accuracy"], 4),
                              "all_brier": round(allb["party"]["brier"] - allb["agent"]["brier"], 4),
                              "contested_brier": round(cont["party"]["brier"] - cont["agent"]["brier"], 4),
                              "all_margin_mae": round(allb["party"]["margin_mae"] - allb["agent"]["margin_mae"], 4),
                              "contested_margin_mae": round(cont["party"]["margin_mae"] - cont["agent"]["margin_mae"], 4)}}
    Path(RESULT).write_text(json.dumps(out, indent=1))

    print("EXP-085  committee-vote WORLD MODEL: grounded per-member agents vs the party shortcut")
    print(f"  {n_bills} divided Senate roll-calls ({n_bills_c} contested); held-out 20% of members, "
          f"ideology grounded from prior congress (leakage-free)")
    for label, s in (("ALL VOTES", allb), ("CONTESTED (within ~60-40)", cont)):
        print(f"  --- {label} ---")
        print(f"      {'model':16s} {'vote acc':>9s} {'brier':>7s} {'margin MAE':>11s}")
        for k, name in (("base", "base rate"), ("party", "party shortcut"), ("agent", "AGENT (grounded)")):
            print(f"      {name:16s} {s[k]['accuracy']:9.4f} {s[k]['brier']:7.4f} {s[k]['margin_mae']:11.4f}")
    av = out["agent_vs_party"]
    print(f"  AGENT vs PARTY: accuracy all {av['all_accuracy']:+.4f}, contested {av['contested_accuracy']:+.4f} "
          f"| brier all {av['all_brier']:+.4f}, contested {av['contested_brier']:+.4f}")
    print(f"  DEFECTORS (voted AGAINST party line, n={defector['party']['n']}): party shortcut "
          f"{defector['party']['accuracy']:.4f} (0 by construction) vs AGENT {defector['agent']['accuracy']:.4f} "
          f"<- the grounded agent catches {defector['agent']['accuracy']*100:.1f}% of defections party-line misses")
    verdict = ("AGENT WORLD (grounded) BEATS the party shortcut — measured per-member state wins"
               if av["contested_accuracy"] > 0.005 else
               "grounded agents do not beat the party shortcut")
    print(f"  VERDICT: {verdict}")
    print(f"  wrote {RESULT}")
    return out


if __name__ == "__main__":
    run()
