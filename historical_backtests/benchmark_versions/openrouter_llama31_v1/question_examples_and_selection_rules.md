# openrouter_llama31_v1 — question examples and frozen selection rules

Model: `meta-llama/llama-3.1-70b-instruct` (released 2024-07-23). Every benchmark question OPENED
strictly after 2024-07-23T23:59:59Z, was genuinely unresolved at all four cutoffs (market price in
[0.05, 0.95] at each), and resolved decisively before the vault freeze.

## Frozen selection rules (implemented in `framework/vault_build.py`, frozen before selection)

1. Source: Polymarket gamma archive, volume-descending, closed binary yes/no markets.
2. `question_open > 2024-07-23T23:59:59Z` proven by the market's own createdAt/startDate.
3. Decisive UMA resolution (outcomePrices exactly {0,1}) before the freeze timestamp.
4. Volume ≥ $20k; unresolved lifetime ≥ 14 days; ≥ 8 archived price points.
5. Exclusions (deterministic tokens): sports, exact scores, celebrity trivia, pure crypto/stock
   price thresholds, mechanical tweet/mention counters.
6. Cutoffs: fractions (0.15, 0.35, 0.55, 0.75) of the unresolved lifetime, each ≥1 day from both
   ends; price at EVERY cutoff within [0.05, 0.95].
7. Cluster cap ≤ 3 per gamma event slug.
8. Causal-scale quotas (frozen): single_decision_maker ≥15, small_group_decision ≥15,
   multi_actor_strategic ≥20, institutional_process ≥15, broad_aggregate ≥20, mixed_scale ≥15 —
   quota-first greedy fill in chronological order, then chronological fill to 100.
9. Splits: chronological 40 calibration / 20 validation / 40 rotating_locked. Selection seed
   20260717 frozen before forecasting. No manual cherry-picking; the composition report is
   generated, not curated.

## 50+ example question forms (the required shape — full, resolvable, dated)

### A. Single-decision-maker
1. Will the President of the United States veto the bill before March 31, 2025?
2. Will the CEO of Company X resign before December 31, 2025?
3. Will Elon Musk publicly announce that SpaceX has filed for an IPO before December 31, 2025?
4. Will the governor sign Bill X into law by June 30, 2025?
5. Will Judge X issue an injunction blocking Policy Y before the stated deadline?
6. Will President Trump pardon Individual X before July 4, 2025?
7. Will the Fed Chair announce his resignation before the June FOMC meeting?
8. Will Prime Minister X call a snap election before September 30, 2025?
9. Will the Pope name a new Secretary of State before year-end?

### B. Small-group decision
10. Will the Federal Trade Commission file a lawsuit to block the Company A–Company B merger
    before the transaction closes?
11. Will the Supreme Court rule that Law X is unconstitutional by June 30, 2025?
12. Will Company X's board remove its CEO before December 31, 2025?
13. Will the Federal Reserve cut its target rate at the September 2024 meeting?
14. Will the Senate confirm Nominee X before the end of the congressional session?
15. Will Israel's security cabinet approve the proposed ceasefire agreement by the deadline?
16. Will OPEC+ announce a production cut at its next ministerial meeting?
17. Will the FOMC cut the federal-funds target range at least twice during 2025?

### C. Multi-actor strategic
18. Will Russia and Ukraine sign a formal ceasefire agreement before December 31, 2025?
19. Will the United States and Iran announce a new nuclear agreement before June 30, 2025?
20. Will Company A complete its acquisition of Company B by the contractual deadline?
21. Will TikTok remain available to a majority of United States users through June 30, 2025?
22. Will NATO invite Country X to begin formal accession talks before the summit ends?
23. Will SpaceX complete an initial public offering before December 31, 2025?
24. Will the US and China announce a trade agreement before August 1, 2025?
25. Will a US–Houthi maritime ceasefire hold through September 30, 2025?
26. Will Hamas release all living hostages before March 1, 2025?

### D. Institutional-process
27. Will Congress enact the proposed foreign-aid package before April 30, 2025?
28. Will the European Union approve the AI regulation before the end of the parliamentary term?
29. Will the regulator grant final approval to Drug X before December 31, 2025?
30. Will the proposed constitutional amendment receive the required supermajority by the deadline?
31. Will the merger receive approval from every required regulator before the termination date?
32. Will the government avoid a shutdown past the continuing-resolution deadline?
33. Will the Senate pass the reconciliation bill before the August recess?
34. Will the House vote to impeach Official X during this Congress?

### E. Broad aggregate
35. Will more than 50 countries formally recognize the State of Palestine before Dec 31, 2025?
36. Will United States electric-vehicle sales exceed 10% of new vehicle sales in 2025?
37. Will nationwide support for Candidate X exceed 50% in the final pre-election polling average?
38. Will global refugee displacement exceed the stated threshold by year-end?
39. Will more than ten major universities adopt Policy X before the 2025 academic year?
40. Will United States union membership increase year over year in 2025?
41. Will annual global AI data-center electricity consumption exceed the stated threshold?
42. Will US CPI year-over-year inflation fall below 2.5% before October 2025?
43. Will measles cases in the US exceed 2,000 in 2025?

### F. Mixed-scale
44. Will Candidate X win the national election?
45. Will the nationwide strike cause the government to withdraw the reform before the deadline?
46. Will a major social platform reverse its policy after sustained user and advertiser pressure?
47. Will Country X enter a recession before December 31, 2025?
48. Will the government impose a nationwide ban after the regulator's recommendation?
49. Will a proposed peace agreement survive for at least 30 consecutive days?
50. Will turnout exceed 60% and Candidate Y concede within 48 hours of the polls closing?
51. Will the central bank intervene after the currency falls more than 10% in a quarter?
52. Will the referendum pass with more than 55% of the vote?

These are FORM examples; no example enters the benchmark unless its historical timestamps,
objective resolution, archived evidence, and temporal eligibility are proven by the frozen rules
above. The actually-selected 100 questions and generated composition are in
`question_vault.json` + `composition_report.json` (sealed).
