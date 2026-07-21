"""Phase-integration relevance corpus (Part B).

Each question carries INDEPENDENT per-phase relevance labels derived from explicit scenario characteristics —
NOT from the compiler's own output (the compiler must never be its own ground truth). Labels mark which phases
are causally required. Irrelevant controls (no strategic actor / no institution / no population / no network /
linear dynamics) test false activation. Paraphrase variants test robustness.

relevance flags per row: p4 (strategic actor), p6 (registered causal mechanism), p7 (nonlinear structure),
p9pop (population heterogeneity), p9net (relational propagation), p10 (institutional rules/authority),
p11 (structural shock in the window).
"""
from __future__ import annotations

# (qid, question, as_of, horizon, {relevance flags})
QUESTIONS = [
    # ---- Phase 10 institutions required (formal rules/authority/procedure issue the decision) ----
    ("sen_confirm", "Will the US Senate confirm the President's Supreme Court nominee at the scheduled vote?", "2024-06-01", "2024-06-30", {"p4", "p10"}),
    ("house_bill", "Will the House of Representatives pass the appropriations bill before the deadline?", "2024-09-01", "2024-09-30", {"p4", "p10"}),
    ("fed_fomc_vote", "Will the FOMC vote to cut the federal funds rate at its next meeting?", "2024-09-01", "2024-09-19", {"p4", "p10"}),
    ("eu_merger", "Will the European Commission approve the proposed merger after its regulatory review?", "2024-03-01", "2024-09-30", {"p10"}),
    ("scotus_rule", "Will the Supreme Court rule in favor of the plaintiff in the pending case?", "2024-04-01", "2024-06-30", {"p10"}),
    ("un_sc_vote", "Will the UN Security Council adopt the ceasefire resolution at its session?", "2024-03-01", "2024-04-30", {"p4", "p10"}),
    ("board_ceo", "Will the company's board of directors approve the proposed CEO succession at its meeting?", "2024-05-01", "2024-06-30", {"p4", "p10"}),
    ("union_ratify", "Will the union membership ratify the tentative labor agreement in the certification vote?", "2024-10-01", "2024-11-15", {"p4", "p9pop", "p10"}),
    # ---- Phase 4 strategic actors required (negotiation / bargaining / coordination) ----
    ("hostage_deal", "Will Israel and Hamas reach a hostage-release agreement in the current negotiations?", "2024-08-01", "2024-10-31", {"p4"}),
    ("opec_cut", "Will OPEC+ members agree to extend production cuts at their meeting?", "2023-11-01", "2023-12-31", {"p4", "p10"}),
    ("ukraine_talks", "Will Russia and Ukraine agree to a temporary ceasefire in the ongoing talks?", "2024-06-01", "2024-12-31", {"p4"}),
    ("studio_strike", "Will the studios and the writers' guild settle the strike through negotiation?", "2023-09-01", "2023-11-30", {"p4", "p9pop"}),
    ("coalition_govt", "Will the parties form a governing coalition after the inconclusive election?", "2024-06-01", "2024-08-31", {"p4", "p10"}),
    # ---- Phase 9 populations required (aggregate heterogeneous behavior) ----
    ("turnout_high", "Will voter turnout in the state exceed 60% in the upcoming election?", "2024-10-01", "2024-11-05", {"p9pop"}),
    ("ev_adopt", "Will electric vehicles exceed 10% of new car sales in the country this year?", "2024-01-01", "2024-12-31", {"p9pop"}),
    ("app_dau", "Will the new social app surpass 5 million daily active users within three months of launch?", "2024-03-01", "2024-06-30", {"p9pop", "p9net"}),
    ("referendum_yes", "Will the referendum pass with a majority of votes cast?", "2024-05-01", "2024-06-30", {"p9pop", "p10"}),
    ("vax_uptake", "Will more than half of eligible adults get the updated vaccine this season?", "2024-09-01", "2024-12-31", {"p9pop"}),
    # ---- Phase 9 networks required (relational propagation / diffusion) ----
    ("meme_viral", "Will the campaign hashtag reach over one million shares on the platform within a week?", "2024-04-01", "2024-04-15", {"p9net", "p7"}),
    ("bank_run", "Will depositors withdraw enough funds to trigger the regional bank's collapse within the month?", "2024-03-01", "2024-03-31", {"p9net", "p7"}),
    ("protest_spread", "Will the protests spread to at least five major cities within two weeks?", "2024-07-01", "2024-07-31", {"p9net", "p9pop", "p7"}),
    ("supply_contagion", "Will the supplier's bankruptcy cause at least three downstream firms to halt production?", "2024-05-01", "2024-08-31", {"p9net"}),
    ("rumor_spread", "Will the misinformation about the product reach mainstream news within days?", "2024-06-01", "2024-06-15", {"p9net"}),
    # ---- Phase 7 nonlinear required (threshold / saturation / self-excitation / tipping) ----
    ("crowd_tip", "Will the crowdfunding campaign hit its funding threshold, triggering full release, before the deadline?", "2024-02-01", "2024-03-31", {"p7", "p9pop"}),
    ("wildfire", "Will the wildfire exceed 100,000 acres given the current spread dynamics?", "2024-08-01", "2024-08-31", {"p7"}),
    ("outage_cascade", "Will the initial grid failure cascade into a regional blackout affecting over a million people?", "2024-07-01", "2024-07-15", {"p7", "p9net"}),
    ("adoption_tip", "Will the messaging app reach the network tipping point and become the dominant platform in the country this year?", "2024-01-01", "2024-12-31", {"p7", "p9net", "p9pop"}),
    ("hype_saturate", "Will pre-orders for the console saturate at under two million given diminishing marketing returns?", "2024-05-01", "2024-09-30", {"p7", "p9pop"}),
    # ---- Phase 6 registered mechanism required (a known social/economic causal process) ----
    ("price_pass", "Will the tariff increase be passed through to consumer prices within six months?", "2024-04-01", "2024-10-31", {"p6"}),
    ("wage_bargain", "Will the minimum-wage increase reduce employment in the affected sector this year?", "2024-01-01", "2024-12-31", {"p6", "p9pop"}),
    ("run_dynamics", "Will the stablecoin lose its peg given redemption pressure this month?", "2024-05-01", "2024-05-31", {"p6", "p7", "p9net"}),
    # ---- Phase 11 shock / recompilation required (structure changes mid-window) ----
    ("assad_shock", "Will the Syrian government retain control of Damascus through the end of the year?", "2024-11-25", "2024-12-31", {"p11", "p4"}),
    ("ceo_ouster", "Will the founder remain CEO of the AI startup through the end of the quarter?", "2023-11-15", "2023-12-31", {"p11", "p4", "p10"}),
    ("biden_drop", "Will the incumbent remain his party's presidential nominee through the convention?", "2024-07-05", "2024-08-22", {"p11", "p4"}),
    ("coup_shock", "Will the country's elected president still hold office at year end?", "2023-07-20", "2023-12-31", {"p11", "p4"}),
    ("merger_collapse", "Will the announced acquisition close, given a possible regulatory block mid-process?", "2024-01-01", "2024-12-31", {"p11", "p10"}),
    # ---- irrelevant controls (no strategic actor / institution / population / network / linear) ----
    ("ctrl_temp", "Will the daily high temperature in the city exceed 35 degrees Celsius tomorrow?", "2024-07-01", "2024-07-02", set()),
    ("ctrl_gold", "Will the price of gold close above 2500 dollars per ounce by year end?", "2024-07-01", "2024-12-31", set()),
    ("ctrl_eclipse", "Will a total solar eclipse be visible from the city on the given date?", "2024-01-01", "2024-04-30", set()),
    ("ctrl_btc", "Will Bitcoin trade above 100000 dollars at any point this year?", "2024-06-01", "2024-12-31", set()),
    ("ctrl_rain", "Will it rain in the capital on New Year's Day?", "2024-12-01", "2025-01-01", set()),
    ("ctrl_quake", "Will a magnitude 7 or greater earthquake occur in the region this year?", "2024-01-01", "2024-12-31", set()),
    ("ctrl_oil", "Will WTI crude oil exceed 100 dollars per barrel this year?", "2024-04-01", "2024-12-31", set()),
    ("ctrl_marathon", "Will the men's marathon world record be broken at the race?", "2024-09-01", "2024-10-31", set()),
    ("ctrl_snow", "Will the ski resort record over 300 inches of snowfall this season?", "2024-11-01", "2025-04-30", set()),
    ("ctrl_gdp_print", "Will the country's reported quarterly GDP growth be positive?", "2024-04-01", "2024-08-31", set()),
]

PHASE_FLAGS = ["p4", "p6", "p7", "p9pop", "p9net", "p10", "p11"]
