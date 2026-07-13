# WMV2 Phase 10 — Validation, Failures & Forensics

All numbers regenerate from `experiments/wmv2_phase10_replay.py` (real data) and `wmv2_phase10_execute.py`.

## 1. Real historical replay (Part 22) — US Senate legislative institution

Data: **voteview.com** static roll-call CSVs, 117th + 118th Senate (**1,626 real roll-calls**, 14
unclassifiable excluded). For each roll-call we select the applicable threshold from the vote question
(the institutional RULE-SELECTION step), execute `decisions.evaluate_decision` on the REAL yea/nay counts,
and compare to the RECORDED `vote_result`. Leakage-safe: only each roll-call's own counts + as-of rules.

| Configuration | Accuracy | Meaning |
|---|---|---|
| **matter-aware** (as-of + nuclear-option + treaty/override) | **0.9631** | the real institution's rules |
| ablation: naive uniform-3/5 cloture | 0.7706 | ignores the 2013/2017 nuclear option |
| ablation: majority-only | 0.9533 | ignores all special thresholds |

By threshold type (matter-aware): majority **1040/1088**, nomination-cloture-majority **441/453**,
legislation-cloture-3/5 **77/77 (100%)**, treaty-2/3 **3/3**, veto-override-2/3 **5/5**.

**The institutional rules are load-bearing:** matter-aware beats naive-cloture by **+0.1925**. Getting the
as-of + matter-type cloture rule right (nomination cloture = majority since the nuclear option) is what moves
accuracy from 0.77 to 0.96 — a real demonstration that as-of rule versioning + matter-type rule selection
change reconstructed outcomes.

## 2. Leakage audit (Part 3)

`leakage_audit(us_congress_legislative, "2021-01-01", outcome_events=[{later vote 2024}])` → `clean=True`;
the active reconstruction contains only Art-I rules effective by 1789, and the 2024 outcome is excluded.
Pinned by `test_as_of_versioning_and_leakage`.

## 3. Execution validation (Part 28.B) — `wmv2_phase10_forensic_traces.json`

- **Authorization / invalid-action blocking:** a lobbyist (representative role) attempting a Senate vote is
  **blocked** ("authority does not cover subject 'senate_vote'"); the StateDelta records
  `blocked_invalid_action` and mutates nothing.
- **Decision execution:** an authorized floor vote runs the decision engine on real-style votes, emits a
  StateDelta (stage transition), schedules the next procedural event, and writes the terminal quantity.

## 4. Counterfactuals (Part 23) — causal institutional sensitivity

Same fixed **55 yea / 45 nay**, vary only the institutional rule:

| Counterfactual | Rule | Passed | Terminal |
|---|---|---|---|
| baseline majority | simple_majority | ✓ | passed |
| raise to supermajority 2/3 | supermajority | ✗ | failed |
| raise to cloture 3/5 (of all) | supermajority | ✗ | failed |
| majority of all members | absolute_majority | ✓ | passed |

`coherent=True`: the same votes produce different terminal outcomes under different rules — the institution
**materially affects the terminal outcome** (not ornamental).

## 5. Adversarial distinctions (Part 25) — pinned by tests

- quorum = floor(n/2)+1 (**51 of 100, never 50**); majority-of-ALL-members vs majority-of-present (40y/30n/
  30 absent: present-majority passes, all-member-majority fails); abstention (present) vs absence (not);
  recusal changes the eligible base; **advisory ≠ decision authority**; veto sustained vs overridden (2/3);
  current vs historical rule version (nuclear option); **formal law vs informal custom** (Rule of Four).

## 6. Ablations (Part 24) — what changes execution

- **no institution / generic approve-reject:** the replay's majority-only arm (0.9533) is the "one generic
  threshold" ablation; the matter-aware arm shows special thresholds add +0.0098 overall and are decisive on
  the cloture/treaty/override subset.
- **wrong rule version (naive cloture):** −0.1827 vs matter-aware — an ablation on as-of/matter-type rule
  selection.
- **no authority checking:** removing authorization admits the blocked lobbyist vote (tested to be blocked
  WITH it) — authority is load-bearing for validity.

## 7. Preserved failures & negatives (Part 26) — `wmv2_phase10_failures.json`

1. **naive_cloture_uniform_3_5** — a uniform 3/5 cloture rule reconstructs only 0.77 (worse than majority-
   only) because it ignores the nuclear option. Preserved: naive rule application can hurt.
2. **evidence gap** — agency/queue/election/moderation families are executable but have **no verified
   template** this run → Tier-3 structural selection. Honest gap, not faked.
3. **scotus_cert_count_not_fraction** — the Rule of Four is a fixed count (4 of 9), an informal custom; the
   fraction-based decision engine does not natively execute a count, so SCOTUS cert stays `executable`
   (informal), not historically replayed.

## 8. Cost & latency

Registry build + replay (1626 votes) + execution + report run in seconds of pure-Python CPU. The only
network use was fetching VoteView CSVs (cached under `experiments/results/phase10/voteview/`) and core-agent
primary-source verification — never at institutional runtime. Per-question institution selection is an
O(families × templates) scan, negligible next to rollout.

## 9. Forensic trace (representative)

```
question         : Will a bill pass the US Senate floor? (as-of 2021-06-01, US-federal)
process          : evaluate_quorum_and_threshold  →  select_institution → us_congress_legislative (tier 1)
evidence         : U.S. Const. art. I §5 (quorum) / §7 (passage) — verified
as-of rules      : quorum majority (51/100), passage majority-of-present  [post-as-of amendments excluded]
instance         : stage=floor_first, actors bound (senator / representative)
event            : institutional_action {actor: lobbyist(rep), type: vote, subject: senate_vote}
authorization    : BLOCKED — representative lacks final_decision authority over senate_vote  → StateDelta(no mutation)
event            : institutional_action {actor: chair(senator), type: vote} + decision(55 yea / 45 nay)
decision engine  : quorum_met=True, yes 55 > 45 → passed
StateDelta       : stage floor_first→floor_second ; quantities[bill_enacted]=passed
future event     : institutional_action @ +1 day (next stage)
terminal         : bill_enacted = passed   (raise threshold to 2/3 → terminal = failed)
support tier     : 1 (evidence-backed, production-eligible)   leakage audit: clean
```
