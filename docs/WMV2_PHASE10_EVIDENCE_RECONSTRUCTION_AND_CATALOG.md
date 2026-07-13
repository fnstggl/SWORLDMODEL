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

## 4. Institution families (8, all executable) — `families.py`

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

## 5. Real templates reconstructed (3) — each core-verified

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

## 6. Formal vs informal structure (Part 12)

Each template separates layers: formal authority (constitution/statute/bylaws) vs informal practice
(leadership agenda control, management information control, the Rule-of-Four custom). Informal influence is
NOT rewritten as formal authority; it is recorded with `formal=False` and flagged as a Phase-3 posterior
point where it affects outcomes.

## 7. Competing rule models & procedural uncertainty (Part 13)

`RuleRecord.alternatives`, `InstitutionTemplate.procedural_uncertainty`, and
`InstitutionInstance.competing_models` carry alternative interpretations (e.g. the override "2/3 of a quorum
vs of full membership" ambiguity; the cert count uncertainty). Multi-particle execution of competing models
is contract-defined; live branching is remaining work.

## 8. Unresolved evidence gaps (honest)

No template-specific official evidence was core-verified this run for the **administrative agency, queue/
capacity service, court docket procedure, election administration, or platform moderation** categories — the
families are executable but have no evidence-backed template, so they select at Tier 3 (structural, rule
uncertainty). This is recorded as a preserved failure (`wmv2_phase10_failures.json`), not papered over. Real
next-round evidence: agency processing-time datasets, court dockets, official election calendars, published
platform transparency reports.
