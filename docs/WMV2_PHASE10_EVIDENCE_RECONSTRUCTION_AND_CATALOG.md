# WMV2 Phase 10 — Evidence, Reconstruction & Catalog

## 1. Evidence methodology & primary-source hierarchy

Every production institutional rule is anchored to an `EvidenceRecord` with source provenance and temporal
validity. Source hierarchy (Part 2): constitution > statute > regulation > rules of procedure / court rule /
bylaws / charter > official records (minutes, votes, dockets) > unofficial secondary summaries. **Unofficial
sources may identify a candidate rule but cannot independently establish it.** Nothing here fabricates legal
text, thresholds, membership, or authority — the three templates below were each **core-verified by me
against the primary source this run**.

## 2. As-of rule versioning (Part 3)

`InstitutionTemplate.rules_as_of(as_of)` / `evidence.active_rules(template, as_of)` return only the rules in
force at the as-of date; `leakage_audit` proves the active reconstruction excludes post-as-of amendments and
later outcomes; `amendment_chain` reconstructs supersession. The historical replay's **nuclear-option** case
is the canonical demonstration: cloture on nominations became a simple majority in 2013/2017, so a 2021 as-of
reconstruction must NOT apply the pre-2013 3/5 rule to nomination cloture (doing so costs 19 accuracy points).

## 3. Rule extraction & formalization (Part 4)

`evidence.validate_rule` runs deterministic checks before a rule can be used: known kind, evidence_id present
(unsourced ⇒ unverified), referenced roles/stages/actions exist, threshold fraction in (0,1], no negative/
impossible deadline, effective precedes supersession. The LLM may PROPOSE a formalization from source text;
the validator + core-agent verification gate what becomes a production rule. `test_wmv2_phase10.py`'s
`validate_template_rules(tpl) == {}` pins that all committed rules pass.

**Automatic extraction (continuation, `institutions_v2/extract.py`).** The extraction front-end is now wired
end-to-end: source text → LLM candidate WITH a verbatim source span → **source-span grounding** (a candidate
whose quoted span is not found in the source is rejected — the LLM cannot establish an unsupported rule) →
deterministic `validate_rule` → typed `RuleRecord`. It degrades to a transparent regex/keyword extractor when
no LLM key is present, so it is always runnable. Scored on two materially-different real documents (US Const
Art I §5/§7; Delaware GCL §141(b)) vs the manually-verified ground truth (`wmv2_phase10_extract.py`):
**macro precision 1.0, recall 0.83, zero hallucinated rules** (real DeepSeek run). This closes the "validator
exists but the extraction front-end is not wired" gap.

## 4. Institution families (9, all executable) — `families.py`

| Family | Category | Characteristic executable procedure |
|---|---|---|
| legislative_process | bicameral_legislature | bicameral passage + presentment + veto/override (Art I §7) |
| hierarchical_approval | hierarchical_approval | ordered approval chain; stops on first rejection |
| collective_vote_body | collective_vote_body | one decision under a quorum + threshold rule |
| adjudicative_court | adjudicative_court | trial → appeal → affirm/reverse/remand |
| administrative_agency | administrative_agency | completeness → staff rec → decision → appeal |
| queue_capacity_service | queue_capacity_service | capacity-limited service; real completion timing |
| corporate_board | corporate_board | delegated management vs reserved board vote (with recusal) |
| moderation_appeals | moderation_appeals | report → penalty → appeal → reinstatement |
| **direct_democracy** (continuation) | direct_democracy | popular referendum; **conjunctive DOUBLE majority** (People AND cantons, half-cantons counting half) — a measure can win the People yet fail on the sub-units |

## 5. Real templates reconstructed (5) — each core-verified

### us_congress_legislative — US federal legislature (Art I) — **production-eligible**
- **Evidence:** U.S. Const. art. I **§5** ("a Majority of each shall constitute a Quorum") and **§7**
  (passage by each House → presentment → veto override **two-thirds of each House**). Verified against
  constitution.congress.gov / law.cornell.edu / archives.gov.
- **Rules:** quorum = majority (51 of 100); passage = majority of present; override = 2/3 of present (of a
  quorum, per long practice — recorded as an ambiguity). Valid from 1789-03-04.
- **Formal vs informal:** committee gatekeeping and floor scheduling are recorded as **informal** leadership
  control (a posterior point), NOT a formal passage rule.

### delaware_board_default — Delaware corporation board (DGCL §141(b)) — executable
- **Evidence:** 8 Del. C. **§141(b)**: quorum = majority of directors (default; bylaws may raise, or lower to
  ≥1/3); board act = majority of directors present at a quorum meeting. Verified against delcode.delaware.gov
  / justia. Valid from 1969.
- **Limitation:** statutory DEFAULT — a real company's certificate/bylaws override it (recorded).

### scotus_certiorari — Supreme Court certiorari (Rule of Four) — executable, **informal**
- **Evidence:** the "Rule of Four" (≥4 of 9 Justices grant cert) is a **CUSTOM** since the Judiciary Acts of
  1891/1925 — "not required by the Constitution, any law, or even the Court's own published rules"
  (FJC / SCOTUSblog / Wikipedia). Stored as `informal_practice` (formal=False) — the canonical
  formal-vs-informal institutional distinction.
- **Limitation:** the threshold is a fixed COUNT (4), not a fraction of present; cert votes are secret
  (a Phase-3 posterior point). Not historically replayed.

### scotus_merits — US Supreme Court merits decision (Art III) — **cross_institution_tested** (continuation)
- **Evidence:** U.S. Const. **art. III** + **28 U.S.C. §1** (quorum of six); a case is decided by a **majority
  of participating Justices**; an equally divided Court (4-4) affirms the judgment below (recorded ambiguity).
- **Real replay (SCDB, 2786 cases, terms ≥1990):** majority-rule decision reconstruction **99.4%**; reversal
  rate **0.70** (matches the ~2/3 cert-to-reverse regularity); **NON-VOTING term-deadline timing** — median
  84 days argument→decision, **99.6%** of argued cases decided within the term. Leakage-safe (as-of argument
  date; no post-decision inputs). 2nd institution category (adjudicative court).

### swiss_federal_referendum — Swiss direct democracy (Cst. Art 139–142) — **cross_institution_tested** (continuation)
- **Evidence:** Federal Constitution of the Swiss Confederation **Art. 141** (optional referendum on federal
  acts = single majority of the People), **Art. 139/140** (popular initiative & mandatory referendum on
  constitutional matters = **DOUBLE majority** of People AND cantons), **Art. 142** (required majorities; six
  half-cantons count half). The double-majority is executed by the `direct_democracy` family procedure.
- **Real replay (Swissvotes/BFS, 704 referenda, 1848–2026):** the legal FORM reconstructs the outcome
  regularity — popular initiatives pass **10.7%** vs mandatory referenda **75.1%** (the famous double-majority
  + establishment-opposition effect); an **out-of-sample** forecast (train ≤1990 → test >1990, n=323): acc
  0.64, Brier 0.183 beats the base-rate Brier 0.249; a **NON-VOTING voting-cadence** dimension (median 3
  official voting dates/year). 3rd institution category (direct democracy).
- **Limitation (preserved):** the cached data has the legal form + outcome but **no per-canton vote shares**,
  so the double majority is reconstructed as an outcome REGULARITY by form, NOT executed on canton counts. A
  full double-majority execution replay needs canton-level Swissvotes data (recorded in the failures artifact).

## 6. Formal vs informal structure (Part 12)

Each template separates layers: formal authority (constitution/statute/bylaws) vs informal practice
(leadership agenda control, management information control, the Rule-of-Four custom). Informal influence is
NOT rewritten as formal authority; it is recorded with `formal=False` and flagged as a Phase-3 posterior
point where it affects outcomes.

## 7. Competing rule models & procedural uncertainty (Part 13)

`RuleRecord.alternatives`, `InstitutionTemplate.procedural_uncertainty`, and
`InstitutionInstance.competing_models` carry alternative interpretations (e.g. the override "2/3 of a quorum
vs of full membership" ambiguity; the cert count uncertainty). **Multi-particle execution is now live**
(`institutions_v2/particles.py`, continuation): `execute_competing_models` runs each structurally-distinct
hypothesis SEPARATELY (its own threshold/quorum/base/membership) and aggregates a WEIGHTED terminal
distribution — **incompatible rules are never averaged into one threshold** — with weights drawn from the
REAL Phase-3 posterior (`institutional_hypothesis_posterior` → `infer_compositional_posterior`). On the
veto-override "2/3-of-present vs 2/3-of-all-members" dispute, 66-of-96-present splits **pass 0.60 / fail
0.40** by posterior weight, and `divergence()` flags the interpretation as outcome-determining.

## 8. Unresolved evidence gaps (honest)

The continuation added evidence-backed, **historically-replayed** templates for the adjudicative-court
(scotus_merits) and direct-democracy (swiss_federal_referendum) categories. Still with **no historically-
replayed template** this run: the **administrative agency, capacity queue, platform moderation, and corporate
board** categories — the families are executable but select at Tier 3 (structural, rule uncertainty). This is
recorded as a preserved failure (`wmv2_phase10_failures.json`), not papered over. Also preserved: the Swiss
referendum's missing **per-canton vote shares** (blocks a full double-majority EXECUTION replay). Real
next-round evidence: agency processing-time datasets, board-meeting minutes, platform transparency reports,
and canton-level Swissvotes results.
