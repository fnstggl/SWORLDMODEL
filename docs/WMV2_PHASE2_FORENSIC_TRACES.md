# WMv2 Phase 2 — Forensic Traces

*One deep end-to-end trace per domain. Each shows the exact evidence queries (paired after:/before:), what was retrieved and temporally verified, the claims with their spans, the dependence/contradiction/visibility structure, the immutable bundle hash, and — critically — how the evidence changed the compiled plan, produced observation StateDeltas, and moved the terminal distribution. Machine-readable companion: `experiments/results/wmv2_phase2_forensic_traces.json`.*

Model: DeepSeek V3 + LIVE Google News RSS · 16 domains · 111 calls · ~$0.0748 · 526.4s.

## messaging

**Q:** Will Elon Musk respond publicly to the FTC inquiry about Twitter?  ·  as-of 2023-07-01 → horizon 2023-08-01

- **evidence requirements** (8): If Elon Musk issues any public communication (tweet, press release, interview, s (voi 1.0); value/context of elon_musk.legal_risk_perception (voi 0.9); value/context of elon_musk.desire_for_public_engagement (voi 0.8)
- **paired RSS windows**:
    - `Elon Musk respond publicly FTC inquiry Twitter elon musk ftc twitter x corp after:2023-02-01 before:2023-07-01` → ok 200, 8 items (raw #6ad1f8013517)  [after:2023-02-01 before:2023-07-01]
    - `elon musk legal risk perception after:2023-02-01 before:2023-07-01` → ok 200, 2 items (raw #962e73a5fbd9)  [after:2023-02-01 before:2023-07-01]
    - `elon musk desire public engagement after:2023-02-01 before:2023-07-01` → ok 200, 1 items (raw #9d347af467f3)  [after:2023-02-01 before:2023-07-01]
- **retrieved**: 11 docs; temporal {'likely_pre_asof': 11}; 11 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [official_record] House GOP subpoenas — "House GOP Subpoenas FTC for Twitter Investigation Documents"
    - [opinion] journalists should leave — "Why journalists should finally leave Twitter"
    - [observed_fact] Elon Musk fires — "Elon Musk fires a top Twitter engineer over his declining view count"
    - [official_record] Platformer reports — "Platformer"
- **entities**: ['house gop', 'ftc', 'twitter', 'journalists', 'twitter']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 9}; **leakage flags**: 0
- **immutable bundle hash**: `130c0371a213c0a5`
- **evidence-conditioned plan diff**: 16 structural changes (lean_only=False); institutions 2→2, events 1→9
- **kinds**: ['entity_added', 'entity_added', 'relation_added', 'relation_added', 'relation_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed']
- **observation StateDeltas**: 0
- **terminal**: pre {'does_not_respond_publicly': 0.56, 'responds_publicly': 0.44} → post {'responds_publicly': 0.52, 'does_not_respond_publicly': 0.48} (changed=True); **evidence is causal: True**

## negotiation

**Q:** Will the UAW and Ford reach a tentative labor agreement?  ·  as-of 2023-10-01 → horizon 2023-11-01

- **evidence requirements** (8): agreement_reached if a tentative contract is ratified or announced by both parti (voi 1.0); value/context of uaw.leadership_militancy (voi 0.9); value/context of ford.reservation_price (voi 0.9)
- **paired RSS windows**:
    - `UAW Ford tentative labor agreement uaw ford shawn fain after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #6fd423722b81)  [after:2023-05-04 before:2023-10-01]
    - `uaw leadership militancy after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #b3721709e32f)  [after:2023-05-04 before:2023-10-01]
    - `ford reservation price after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #934dc2dcde4a)  [after:2023-05-04 before:2023-10-01]
- **retrieved**: 24 docs; temporal {'likely_pre_asof': 23, 'uncertain': 1}; 24 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [observed_fact] UAW launches — "UAW launches strike against Big 3 automakers"
    - [opinion] Union Reform Caucus is responsible for — "We Can Thank a Union Reform Caucus for the Militant UAW Strike"
    - [forecast] 2023 Ford F-150 Lightning orders open — "2023 Ford F-150 Lightning Orders Open May 9 to Non-Reservation Holders: Report"
    - [observed_fact] Ford agrees to end — "Ford and GM Agree to End At Least One Tier"
- **entities**: ['uaw', 'big 3 automakers', 'union reform caucus', 'uaw', '2023 ford f-150 lightning']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 10}; **leakage flags**: 0
- **immutable bundle hash**: `6e1b12773386e1b6`
- **evidence-conditioned plan diff**: 18 structural changes (lean_only=False); institutions 3→4, events 1→10
- **kinds**: ['entity_added', 'entity_added', 'institution_added', 'rule_added', 'relation_added', 'relation_added', 'relation_added', 'event_added']
- **observation StateDeltas**: 0
- **terminal**: pre {'no_agreement': 0.52, 'agreement_reached': 0.48} → post {'agreement_reached': 0.56, 'no_agreement': 0.44} (changed=True); **evidence is causal: True**

## organizational_decision

**Q:** Will Disney's board extend Bob Iger's contract as CEO?  ·  as-of 2023-06-01 → horizon 2023-08-01

- **evidence requirements** (8): The board's formal vote outcome as recorded in official minutes or public announ (voi 1.0); value/context of board_members.preference_extend (voi 1.0); value/context of board_members.loyalty_to_iger (voi 0.9)
- **paired RSS windows**:
    - `Disney board extend Bob Iger contract CEO bob iger board chair successor candidate internal after:2023-01-02 before:2023-06-01` → ok 200, 5 items (raw #e93de2e3fe48)  [after:2023-01-02 before:2023-06-01]
    - `board members preference extend after:2023-01-02 before:2023-06-01` → ok 200, 2 items (raw #e67802e7f975)  [after:2023-01-02 before:2023-06-01]
    - `board members loyalty iger after:2023-01-02 before:2023-06-01` → zero_results 200, 0 items (raw #ac6f3b475939)  [after:2023-01-02 before:2023-06-01]
- **retrieved**: 7 docs; temporal {'likely_pre_asof': 7}; 7 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [observed_fact] Disney is experiencing — "Inside Disney it's 'Revenge of the Creatives'"
    - [observed_fact] CEO Bob Iger is taking — "CEO Bob Iger is taking no prisoners"
    - [observed_fact] Baron Davis resigns as — "Baron Davis resigns as superintendent of South Carolina’s Richland School Distri"
    - [official_record] Disney revealed — "Disney Reveals Reason for Firing Former CEO Bob Chapek"
- **entities**: ['disney', 'bob iger', 'baron davis', 'south carolina’s richland school district two', 'disney']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 8}; **leakage flags**: 0
- **immutable bundle hash**: `73353b9eaf265c1c`
- **evidence-conditioned plan diff**: 13 structural changes (lean_only=False); institutions 2→2, events 3→12
- **kinds**: ['entity_added', 'relation_added', 'event_added', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed', 'requirement_fulfilled', 'requirement_fulfilled']
- **observation StateDeltas**: 0
- **terminal**: pre {'not_extend': 0.54, 'extend': 0.46} → post {'not_extend': 0.54, 'extend': 0.46} (changed=False); **evidence is causal: True**

## election

**Q:** Will the Republicans win the Kentucky governor race in 2023?  ·  as-of 2023-10-15 → horizon 2023-11-08

- **evidence requirements** (8): The candidate receiving the most votes in the general election is the winner; if (voi 1.0); value/context of kentucky_voters.true_preference_distribution (voi 0.9); value/context of andy_beshear.approval_rating (voi 0.85)
- **paired RSS windows**:
    - `Republicans win Kentucky governor race 2023 andy beshear daniel cameron kentucky state board of elections after:2023-05-18 before:2023-10-15` → ok 200, 8 items (raw #41265934879a)  [after:2023-05-18 before:2023-10-15]
    - `kentucky voters true preference distribution after:2023-05-18 before:2023-10-15` → zero_results 200, 0 items (raw #d48a168ece9e)  [after:2023-05-18 before:2023-10-15]
    - `andy beshear approval rating after:2023-05-18 before:2023-10-15` → ok 200, 8 items (raw #859032150f4d)  [after:2023-05-18 before:2023-10-15]
- **retrieved**: 16 docs; temporal {'likely_pre_asof': 16}; 15 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [opinion] I am very worried about — "Why I’m very worried about a Gov. Cameron"
    - [observed_fact] Kentucky Gov. Beshear leads re-election race by — "Poll: Kentucky Gov. Beshear leads re-election race by 10 points"
    - [forecast] abortion is set to shape — "How abortion is set to shape the Kentucky governor’s race"
    - [observed_fact] Beshear is — "Poll: Beshear is America’s 5th most popular governor"
- **entities**: ['cameron', 'kentucky beshear', 'abortion', 'kentucky governor’s race', 'beshear']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'exact_duplicate', 'n': 2}]; **contradictions**: none
- **actor visibility**: {'public': 7}; **leakage flags**: 0
- **immutable bundle hash**: `855bc0fe76c466d6`
- **evidence-conditioned plan diff**: 13 structural changes (lean_only=False); institutions 2→2, events 2→9
- **kinds**: ['entity_added', 'relation_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed', 'requirement_fulfilled', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'Democrat_win': 0.56, 'Republican_win': 0.44} → post {'Republican_win': 0.5, 'Democrat_win': 0.5} (changed=True); **evidence is causal: True**

## legislation

**Q:** Will the US Congress pass a stopgap funding bill to avoid a shutdown?

_execution error: KeyError: "unknown entity 'house_of_representatives' (known: ['freedom_caucus', 'house_conservatives_group', 'president', 'senate_majority_leader', 'senate_mino_

## acquisition

**Q:** Will Microsoft complete its acquisition of Activision Blizzard?  ·  as-of 2023-08-01 → horizon 2023-11-01

- **evidence requirements** (7): completed if by 2023-11-01 the transaction has closed (all regulatory approvals  (voi 1.0); value/context of ftc.enforcement_philosophy (voi 0.9); value/context of ec.political_factors (voi 0.8)
- **paired RSS windows**:
    - `Microsoft complete acquisition Activision Blizzard microsoft activision blizzard ftc after:2023-03-04 before:2023-08-01` → ok 200, 8 items (raw #d1987b4a73be)  [after:2023-03-04 before:2023-08-01]
    - `ftc enforcement philosophy after:2023-03-04 before:2023-08-01` → zero_results 200, 0 items (raw #fd79363c5775)  [after:2023-03-04 before:2023-08-01]
    - `political factors after:2023-03-04 before:2023-08-01` → ok 200, 8 items (raw #857c24d6f2d2)  [after:2023-03-04 before:2023-08-01]
- **retrieved**: 16 docs; temporal {'likely_pre_asof': 16}; 16 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [observed_fact] Microsoft-Activision deal moves closer — "Microsoft-Activision deal moves closer"
    - [official_record] judge denies — "judge denies FTC injunction request"
    - [observed_fact] Mercator Institute for China Studies (ME has — "Political and social factors - Mercator Institute for China Studies (MERICS)."
    - [observed_fact] Microsoft heads to court over — "Microsoft heads to court over $69 billion deal that could reshape video gaming"
- **entities**: ['microsoft', 'activision', 'ftc', 'mercator institute for china studies (merics)', 'microsoft']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 7}; **leakage flags**: 0
- **immutable bundle hash**: `125865884af19f80`
- **evidence-conditioned plan diff**: 15 structural changes (lean_only=False); institutions 5→6, events 1→9
- **kinds**: ['entity_added', 'institution_added', 'rule_added', 'relation_added', 'relation_added', 'event_added', 'hypothesis_reweighted', 'lean_changed']
- **observation StateDeltas**: 0
- **terminal**: pre {'not_completed': 0.58, 'completed': 0.42} → post {'completed': 0.56, 'not_completed': 0.44} (changed=True); **evidence is causal: True**

## product_launch

**Q:** Will Apple release the iPhone 15 on its announced date?  ·  as-of 2023-09-01 → horizon 2023-10-01

- **evidence requirements** (8): release_on_date if Apple publicly releases iPhone 15 on the announced date (2023 (voi 1.0); value/context of Apple.supply_chain_readiness (voi 0.9); value/context of Apple.internal_decision (voi 0.8)
- **paired RSS windows**:
    - `Apple release iPhone announced date Apple Foxconn TSMC after:2023-04-04 before:2023-09-01` → ok 200, 3 items (raw #3d53ea1dead9)  [after:2023-04-04 before:2023-09-01]
    - `Apple supply chain readiness after:2023-04-04 before:2023-09-01` → ok 200, 2 items (raw #77e29463f079)  [after:2023-04-04 before:2023-09-01]
    - `Apple internal decision after:2023-04-04 before:2023-09-01` → ok 200, 6 items (raw #021739c134de)  [after:2023-04-04 before:2023-09-01]
- **retrieved**: 11 docs; temporal {'likely_pre_asof': 10, 'uncertain': 1}; 11 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [observed_fact] Apple has — "Apple, Samsung in distinct cash management styles"
    - [observed_fact] Samsung has — "Apple, Samsung in distinct cash management styles"
    - [observed_fact] Foxconn is ready to cash in on — "iPhone manufacturer Foxconn ready to cash in on India's expanding economy"
    - [opinion] Apple's Culture rejects — "Apple's Culture Rejects the Conventional Wisdom of Product Design."
- **entities**: ['apple', 'samsung', 'foxconn', 'india', 'apple']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 10}; **leakage flags**: 0
- **immutable bundle hash**: `0300f7f422fd403b`
- **evidence-conditioned plan diff**: 17 structural changes (lean_only=False); institutions 2→2, events 2→11
- **kinds**: ['entity_added', 'entity_added', 'relation_added', 'relation_added', 'event_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'hypothesis_reweighted']
- **observation StateDeltas**: 0
- **terminal**: pre {'release_on_date': 0.56, 'not_release_on_date': 0.44} → post {'release_on_date': 0.58, 'not_release_on_date': 0.42} (changed=True); **evidence is causal: True**

## social_media_diffusion

**Q:** Will Meta's Threads app reach 100 million users in its first month?  ·  as-of 2023-07-06 → horizon 2023-08-06

- **evidence requirements** (8): If Meta publicly reports or credible third-party data confirms Threads reached 1 (voi 1.0); value/context of threads_app.user_count (voi 1.0); value/context of instagram_users.interest_in_threads (voi 0.8)
- **paired RSS windows**:
    - `Meta Threads app 100 million users first meta platforms threads app instagram after:2023-02-06 before:2023-07-06` → ok 200, 8 items (raw #2e8be39b0204)  [after:2023-02-06 before:2023-07-06]
    - `threads app user count after:2023-02-06 before:2023-07-06` → ok 200, 8 items (raw #f930ff70ced4)  [after:2023-02-06 before:2023-07-06]
    - `instagram users interest threads after:2023-02-06 before:2023-07-06` → ok 200, 8 items (raw #96e04f0f3578)  [after:2023-02-06 before:2023-07-06]
- **retrieved**: 24 docs; temporal {'uncertain': 23, 'likely_pre_asof': 1}; 21 independent sources
- **included claims** (2; 0 excluded, 9 suspicious):
    - [observed_fact] Threads is — "What is Threads? Instagram’s Twitter alternative to launch July 6"
    - [forecast] Threads launch — "What is Threads? Instagram’s Twitter alternative to launch July 6"
- **entities**: ['threads', 'instagram', 'twitter', 'threads', 'threads app']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'exact_duplicate', 'n': 2}]; **contradictions**: none
- **actor visibility**: {'public': 11}; **leakage flags**: 0
- **immutable bundle hash**: `e46a77f672b09966`
- **evidence-conditioned plan diff**: 11 structural changes (lean_only=False); institutions 2→2, events 3→6
- **kinds**: ['event_added', 'hypothesis_reweighted', 'uncertainty_changed', 'requirement_unmet', 'requirement_fulfilled', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'yes': 0.58, 'no': 0.42} → post {'yes': 0.62, 'no': 0.38} (changed=True); **evidence is causal: True**

## protest

**Q:** Will the climate protests disrupt the UN General Assembly in New York?  ·  as-of 2023-09-10 → horizon 2023-09-25

- **evidence requirements** (8): disruption if any protest action forces cancellation, postponement, or significa (voi 1.0); value/context of climate_protesters.radical_flank.willingness_to_escalate (voi 0.9); value/context of un_security_chief.decision_threshold_for_cancellation (voi 0.9)
- **paired RSS windows**:
    - `climate protests disrupt General Assembly York un ga president un security chief nyc mayor after:2023-04-13 before:2023-09-10` → zero_results 200, 0 items (raw #83c92303df56)  [after:2023-04-13 before:2023-09-10]
    - `climate protesters radical flank willingness escalate after:2023-04-13 before:2023-09-10` → zero_results 200, 0 items (raw #737904ec0828)  [after:2023-04-13 before:2023-09-10]
    - `security chief decision threshold cancellation after:2023-04-13 before:2023-09-10` → zero_results 200, 0 items (raw #6c0d527a442a)  [after:2023-04-13 before:2023-09-10]
- **retrieved**: 0 docs; temporal {}; 0 independent sources
- **included claims** (0; 0 excluded, 0 suspicious):
- **entities**: []
- **dependence**: []; **contradictions**: none
- **actor visibility**: {}; **leakage flags**: 0
- **immutable bundle hash**: `af80faec85331112`
- **evidence-conditioned plan diff**: 0 structural changes (lean_only=True); institutions 3→3, events 3→3
- **kinds**: []
- **observation StateDeltas**: 0
- **terminal**: pre {'no_disruption': 0.54, 'disruption': 0.46} → post {'no_disruption': 0.54, 'disruption': 0.46} (changed=False); **evidence is causal: False**

## strike

**Q:** Will the SAG-AFTRA actors strike end with a studio agreement?  ·  as-of 2023-10-01 → horizon 2023-11-15

- **evidence requirements** (8): agreement_reached if a tentative or ratified contract is publicly announced by 2 (voi 1.0); value/context of amptp_leadership.concession_willingness (voi 0.9); value/context of sag_aftra_leadership.internal_hawk_dove_split (voi 0.8)
- **paired RSS windows**:
    - `SAG AFTRA actors strike end studio agreement sag aftra leadership amptp leadership sag aftra membership after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #e9f479ec6bd7)  [after:2023-05-04 before:2023-10-01]
    - `amptp leadership concession willingness after:2023-05-04 before:2023-10-01` → zero_results 200, 0 items (raw #71850c6b59fa)  [after:2023-05-04 before:2023-10-01]
    - `sag aftra leadership internal hawk dove after:2023-05-04 before:2023-10-01` → zero_results 200, 0 items (raw #f1e9dcf993be)  [after:2023-05-04 before:2023-10-01]
- **retrieved**: 8 docs; temporal {'likely_pre_asof': 8}; 8 independent sources
- **included claims** (6; 0 excluded, 1 suspicious):
    - [actor_statement] SAG-AFTRA leaders cite — "SAG-AFTRA leaders cite ‘extremely productive’ contract talks with Hollywood stud"
    - [observed_fact] Actors are going on strike — "Why Actors Are Going on Strike"
    - [observed_fact] Time Magazine published article titled — "Why Actors Are Going on Strike - Time Magazine"
    - [observed_fact] SAG-AFTRA releases — "After SAG-AFTRA Releases List of Contract Proposals"
- **entities**: ['sag-aftra', 'hollywood studios', 'actors', 'time magazine', 'sag-aftra']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 9}; **leakage flags**: 0
- **immutable bundle hash**: `e8d1b564a5295e6d`
- **evidence-conditioned plan diff**: 13 structural changes (lean_only=False); institutions 2→2, events 8→16
- **kinds**: ['hypothesis_reweighted', 'hypothesis_reweighted', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed', 'requirement_fulfilled', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'agreement_reached': 0.5, 'no_agreement': 0.5} → post {'agreement_reached': 0.58, 'no_agreement': 0.42} (changed=True); **evidence is causal: True**

## court_ruling

**Q:** Will a US court rule on the Google antitrust search case?  ·  as-of 2023-09-01 → horizon 2024-01-01

- **evidence requirements** (7): A ruling on the merits (e.g., summary judgment, trial verdict, or dispositive mo (voi 1.0); value/context of us_district_court_dc.judge_mehta.decision_speed_factor (voi 0.8); evidence discriminating structural hypothesis expedited_ruling (voi 0.8)
- **paired RSS windows**:
    - `court rule Google antitrust search case us government google llc us district court dc after:2023-04-04 before:2023-09-01` → zero_results 200, 0 items (raw #5e9607f7f0db)  [after:2023-04-04 before:2023-09-01]
    - `district court judge mehta decision speed after:2023-04-04 before:2023-09-01` → ok 200, 1 items (raw #1e31aa05816d)  [after:2023-04-04 before:2023-09-01]
    - `structural hypothesis expedited ruling evidence discriminating after:2023-04-04 before:2023-09-01` → zero_results 200, 0 items (raw #0b8434076e59)  [after:2023-04-04 before:2023-09-01]
- **retrieved**: 1 docs; temporal {'likely_pre_asof': 1}; 1 independent sources
- **included claims** (2; 0 excluded, 0 suspicious):
    - [observed_fact] Judge tosses — "Judge Tosses FDA's Rule Over Premium Cigars"
    - [official_record] Convenience Store News reports — "Judge Tosses FDA's Rule Over Premium Cigars - Convenience Store News"
- **entities**: ['fda', 'fda', 'premium cigars', 'convenience store news', 'fda']
- **dependence**: [{'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 2}; **leakage flags**: 0
- **immutable bundle hash**: `0bf1085ff674bfae`
- **evidence-conditioned plan diff**: 7 structural changes (lean_only=False); institutions 1→1, events 3→5
- **kinds**: ['lean_changed', 'requirement_unmet', 'requirement_fulfilled', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'no': 0.62, 'yes': 0.38} → post {'no': 0.62, 'yes': 0.38} (changed=False); **evidence is causal: True**

## fundraising

**Q:** Will OpenAI raise a new funding round valuing it above 80 billion?  ·  as-of 2023-10-01 → horizon 2024-01-01

- **evidence requirements** (8): yes if a funding round closes with a valuation >= 80 billion USD by 2024-01-01,  (voi 1.0); value/context of potential_investors.risk_appetite (voi 0.9); value/context of openai.negotiation_leverage (voi 0.8)
- **paired RSS windows**:
    - `OpenAI raise funding round valuing above billion openai sam altman microsoft after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #95e9eac49a32)  [after:2023-05-04 before:2023-10-01]
    - `potential investors risk appetite after:2023-05-04 before:2023-10-01` → ok 200, 8 items (raw #bd56d8a7b49f)  [after:2023-05-04 before:2023-10-01]
    - `openai negotiation leverage after:2023-05-04 before:2023-10-01` → ok 200, 1 items (raw #33fa55891e7d)  [after:2023-05-04 before:2023-10-01]
- **retrieved**: 17 docs; temporal {'likely_pre_asof': 17}; 17 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [official_record] OpenAI seeks new valuation of up to — "OpenAI Seeks New Valuation of Up to $90 Billion in Sale of Existing Shares"
    - [observed_fact] Risk Tolerance and Risk Capacity is topic of article — "Risk Tolerance and Risk Capacity: Why the Difference Matters - Morningstar"
    - [official_record] OpenAI signs deal with — "ChatGPT-maker OpenAI signs deal with AP to license news stories"
    - [observed_fact] Sam Altman has — "Sam Altman’s Tangle of Investments"
- **entities**: ['openai', 'morningstar', 'openai', 'ap', 'sam altman']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 7}; **leakage flags**: 0
- **immutable bundle hash**: `7036ab874c2b7f34`
- **evidence-conditioned plan diff**: 13 structural changes (lean_only=False); institutions 2→2, events 2→9
- **kinds**: ['entity_added', 'relation_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed', 'requirement_fulfilled', 'requirement_fulfilled']
- **observation StateDeltas**: 0
- **terminal**: pre {'no': 0.54, 'yes': 0.46} → post {'yes': 0.54, 'no': 0.46} (changed=True); **evidence is causal: True**

## coalition

**Q:** Will the Dutch parties form a governing coalition after the November election?  ·  as-of 2023-11-25 → horizon 2024-03-01

- **evidence requirements** (6): A coalition is formed if a formal agreement is announced by 2024-03-01; otherwis (voi 1.0); value/context of vvd.negotiation_stance (voi 0.8); value/context of d66.negotiation_stance (voi 0.8)
- **paired RSS windows**:
    - `Dutch parties form governing coalition November election dutch government house of representatives king willem alexander after:2023-06-28 before:2023-11-25` → ok 200, 1 items (raw #4a3f8c169fdf)  [after:2023-06-28 before:2023-11-25]
    - `vvd negotiation stance after:2023-06-28 before:2023-11-25` → zero_results 200, 0 items (raw #b074e1ccd366)  [after:2023-06-28 before:2023-11-25]
    - `d66 negotiation stance after:2023-06-28 before:2023-11-25` → zero_results 200, 0 items (raw #61cc9e1e6a02)  [after:2023-06-28 before:2023-11-25]
- **retrieved**: 1 docs; temporal {'likely_pre_asof': 1}; 1 independent sources
- **included claims** (1; 0 excluded, 0 suspicious):
    - [forecast] Wilders' far-right PVV set to win — "Wilders' far-right PVV set to win — poll"
- **entities**: ['wilders', 'pvv', 'dutch election']
- **dependence**: [{'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 1}; **leakage flags**: 0
- **immutable bundle hash**: `75b475e366327bbe`
- **evidence-conditioned plan diff**: 8 structural changes (lean_only=False); institutions 2→2, events 1→2
- **kinds**: ['entity_added', 'relation_added', 'requirement_fulfilled', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'no_coalition_formed': 0.58, 'coalition_formed': 0.42} → post {'no_coalition_formed': 0.58, 'coalition_formed': 0.42} (changed=False); **evidence is causal: True**

## market

**Q:** Will the US Federal Reserve raise interest rates at its September 2023 meeting?  ·  as-of 2023-09-01 → horizon 2023-09-21

- **evidence requirements** (7): If the FOMC announces an increase in the target range for the federal funds rate (voi 1.0); value/context of fomc.members_preferred_rate_path (voi 0.9); value/context of fomc.internal_consensus (voi 0.8)
- **paired RSS windows**:
    - `Federal Reserve raise interest rates September 2023 meeting fomc jerome powell after:2023-04-04 before:2023-09-01` → ok 200, 8 items (raw #290ff104c0bf)  [after:2023-04-04 before:2023-09-01]
    - `fomc members preferred rate path after:2023-04-04 before:2023-09-01` → zero_results 200, 0 items (raw #bf1d8ecb0526)  [after:2023-04-04 before:2023-09-01]
    - `fomc internal consensus after:2023-04-04 before:2023-09-01` → zero_results 200, 0 items (raw #9a899eb505c5)  [after:2023-04-04 before:2023-09-01]
- **retrieved**: 8 docs; temporal {'likely_pre_asof': 8}; 8 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [forecast] the Fed stop raising rates — "The time could be coming soon"
    - [observed_fact] Federal Reserve resumed — "July 2023 Fed Meeting: Interest Rate Hikes Resume"
    - [observed_fact] Fed approves — "Fed approves hike"
    - [observed_fact] hike takes — "takes interest rates to highest level in more than 22 years"
- **entities**: ['fed', 'federal reserve', 'j p morgan', 'fed', 'fed reserve officials']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 8}; **leakage flags**: 0
- **immutable bundle hash**: `8696614ac549baa5`
- **evidence-conditioned plan diff**: 11 structural changes (lean_only=False); institutions 1→1, events 3→11
- **kinds**: ['hypothesis_reweighted', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed', 'requirement_fulfilled', 'requirement_unmet', 'requirement_unmet']
- **observation StateDeltas**: 0
- **terminal**: pre {'no_raise': 0.56, 'raise': 0.44} → post {'raise': 0.5, 'no_raise': 0.5} (changed=True); **evidence is causal: True**

## reputation_crisis

**Q:** Will Bud Light sales recover after the boycott controversy?  ·  as-of 2023-07-01 → horizon 2023-10-01

- **evidence requirements** (8): The outcome is the ratio of Bud Light weekly sales volume in the last week of Se (voi 1.0); value/context of us_beer_drinkers.conservative_males_21_40.boycott_persistence (voi 0.9); value/context of anheuser_busch_inbev.crisis_response_effectiveness (voi 0.8)
- **paired RSS windows**:
    - `Bud Light sales recover boycott controversy anheuser busch inbev bud light brand dylan mulvaney after:2023-02-01 before:2023-07-01` → ok 200, 7 items (raw #cb8e9a513b8d)  [after:2023-02-01 before:2023-07-01]
    - `beer drinkers conservative males boycott persistence after:2023-02-01 before:2023-07-01` → zero_results 200, 0 items (raw #94ea48f678bc)  [after:2023-02-01 before:2023-07-01]
    - `anheuser busch inbev crisis response effectiveness after:2023-02-01 before:2023-07-01` → ok 200, 3 items (raw #91a57b5e3709)  [after:2023-02-01 before:2023-07-01]
- **retrieved**: 10 docs; temporal {'likely_pre_asof': 10}; 10 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [observed_fact] Bud Light fumbles — "Bud Light fumbles"
    - [opinion] inclusive advertising are here to stay — "inclusive advertising are here to stay"
    - [observed_fact] AB InBev is helping address — "5 ways AB InBev is helping address global water challenges"
    - [actor_statement] CEO distances — "CEO distances Anheuser-Busch from Bud Light Dylan Mulvaney controversy: 'Not a f"
- **entities**: ['bud light', 'ab inbev', 'bud light', 'anheuser-busch', 'bud light']
- **dependence**: [{'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}, {'type': 'independent', 'n': 1}]; **contradictions**: none
- **actor visibility**: {'public': 9}; **leakage flags**: 0
- **immutable bundle hash**: `6ec160d29970eb95`
- **evidence-conditioned plan diff**: 15 structural changes (lean_only=False); institutions 2→2, events 1→11
- **kinds**: ['entity_added', 'relation_added', 'event_added', 'event_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'uncertainty_changed', 'requirement_fulfilled']
- **observation StateDeltas**: 0
- **terminal**: pre None → post None (changed=False); **evidence is causal: True**

## best_action

**Q:** Should the Writers Guild accept the studios' latest contract offer?  ·  as-of 2023-09-20 → horizon 2023-10-01

- **evidence requirements** (8): majority vote of WGA membership (voi 1.0); value/context of AMPTP.offer_details (voi 0.9); value/context of WGA_members.segments.strike_hawks.reservation_wage (voi 0.8)
- **paired RSS windows**:
    - `Writers Guild accept studios latest contract offer WGA AMPTP WGA leadership after:2023-04-23 before:2023-09-20` → ok 200, 8 items (raw #5ee9b7492ecd)  [after:2023-04-23 before:2023-09-20]
    - `AMPTP offer details after:2023-04-23 before:2023-09-20` → ok 200, 8 items (raw #ff2e172bcf0c)  [after:2023-04-23 before:2023-09-20]
    - `WGA members segments strike hawks reservation after:2023-04-23 before:2023-09-20` → zero_results 200, 0 items (raw #3675cf45fc83)  [after:2023-04-23 before:2023-09-20]
- **retrieved**: 16 docs; temporal {'likely_pre_asof': 16}; 13 independent sources
- **included claims** (6; 0 excluded, 0 suspicious):
    - [official_record] Studios reveal — "Studios Reveal New AI, Data Transparency & Residuals Proposals To WGA"
    - [official_record] Guild meets with — "Guild Meets With CEOs"
    - [actor_statement] Writers Guild replies to — "Writers Guild Replies to Studios’ Counteroffer: Not “Nearly Enough”"
    - [actor_statement] Writers Guild states that the counteroffer is — "Not “Nearly Enough”"
- **entities**: ['studios', 'wga', 'guild', 'ceos', 'writers guild']
- **dependence**: [{'type': 'exact_duplicate', 'n': 2}, {'type': 'independent', 'n': 1}, {'type': 'exact_duplicate', 'n': 2}, {'type': 'independent', 'n': 1}, {'type': 'exact_duplicate', 'n': 2}]; **contradictions**: none
- **actor visibility**: {'public': 10}; **leakage flags**: 0
- **immutable bundle hash**: `98c78764be4e5d6c`
- **evidence-conditioned plan diff**: 15 structural changes (lean_only=False); institutions 2→2, events 3→12
- **kinds**: ['entity_added', 'relation_added', 'relation_added', 'event_added', 'hypothesis_reweighted', 'hypothesis_reweighted', 'lean_changed', 'uncertainty_changed']
- **observation StateDeltas**: 0
- **terminal**: pre {'accept': 0.5, 'reject': 0.5} → post {'reject': 0.52, 'accept': 0.48} (changed=True); **evidence is causal: True**

---

Across 16 domains: every trace issued paired after:/before: RSS queries, verified temporal validity independently of the RSS date, extracted span-validated claims, and recorded an evidence-conditioned plan diff. Traces where evidence was admitted show structural plan changes and terminal movement — the system uses contemporaneous evidence to change the world, not a current-search summary to nudge a lean.