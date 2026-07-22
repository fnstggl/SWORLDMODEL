"""Part 10 — independent activation corpus: 200 non-scripted questions, 20 causal scenario families,
12 domains, balanced relevant/irrelevant controls per phase.

Labels are authored INDEPENDENTLY of the runtime's relevance detector, from the question's causal
structure alone (no runtime output was consulted; no historical outcomes are used; the questions are
generic realistic scenarios, not benchmark IDs the runtime could memorize).

Label semantics (per phase, `True` = the phase is causally REQUIRED for a faithful world model):
  p4    — deliberate choices by identifiable strategic actors materially drive the outcome
  p6    — a social/behavioral causal mechanism (persuasion, mobilization, bargaining, adoption,
          compliance…) must be represented for the forecast to be causally grounded
  p7    — thresholds / tipping / saturation / cascades / self-excitation / feedback shape the dynamics
  p9pop — the outcome aggregates heterogeneous behavior of many people
  p9net — transmission across relationships (communication/exposure/trust/influence/authority) matters
  p10   — the outcome itself is decided by a rule-governed institutional procedure
  p11   — plausible structural change (new actors, rule changes, coalition shifts) before the horizon

Row format: (qid, question, as_of, horizon, domain, family, set_of_true_phase_flags)
"""
from __future__ import annotations

PHASE_FLAGS = ["p4", "p6", "p7", "p9pop", "p9net", "p10", "p11"]

_A, _H = "2025-06-01", "2025-09-01"


def _rows():
    R = []

    def add(fam, dom, items):
        for i, (q, flags) in enumerate(items):
            R.append((f"{fam}_{i+1}", q, _A, _H, dom, fam, set(flags)))

    # ---- 1. legislative_vote (politics; p10/p4/p6, p11 sometimes) ----
    add("legvote", "politics", [
        ("Will the national assembly pass the budget reconciliation bill before the summer recess?", {"p4", "p6", "p10", "p11"}),
        ("Will the upper chamber ratify the trade treaty at its July session?", {"p4", "p6", "p10"}),
        ("Will the city council approve the downtown rezoning ordinance by the end of August?", {"p4", "p6", "p10"}),
        ("Will the provincial parliament override the governor's veto of the water bill?", {"p4", "p6", "p10"}),
        ("Will the education funding amendment clear the two-thirds supermajority requirement?", {"p4", "p6", "p10"}),
        ("Will the senate confirm the central-bank nominee before the recess?", {"p4", "p6", "p10"}),
        ("Will the housing committee report the tenancy reform bill to the floor by mid-July?", {"p4", "p6", "p10"}),
        ("Will the coalition government hold together long enough to pass its flagship labor law?", {"p4", "p6", "p10", "p11"}),
        ("Will the impeachment resolution reach the required threshold in the lower house?", {"p4", "p6", "p10", "p11"}),
        ("Will the county board adopt the new public-transit levy at its August meeting?", {"p4", "p6", "p10"}),
    ])
    # ---- 2. judicial_ruling (law; p10/p4) ----
    add("court", "law", [
        ("Will the constitutional court strike down the data-retention statute this term?", {"p4", "p10"}),
        ("Will the appeals panel uphold the antitrust injunction against the platform company?", {"p4", "p10"}),
        ("Will the labor tribunal certify the warehouse workers' class action by September?", {"p4", "p10"}),
        ("Will the supreme court grant review of the emissions-standards case?", {"p4", "p10"}),
        ("Will the arbitration panel rule for the investor in the mining-concession dispute?", {"p4", "p10"}),
        ("Will the electoral commission disqualify the challenger's candidacy petition?", {"p4", "p10", "p11"}),
        ("Will the bankruptcy judge approve the airline's restructuring plan at the July hearing?", {"p4", "p10"}),
        ("Will the patent office's review board invalidate the disputed battery patent?", {"p4", "p10"}),
        ("Will the human-rights court find the detention policy unlawful this session?", {"p4", "p10"}),
        ("Will the regional court order a halt to the pipeline construction pending review?", {"p4", "p10"}),
    ])
    # ---- 3. regulatory_approval (regulation; p10/p4) ----
    add("regapproval", "regulation", [
        ("Will the drug regulator approve the new obesity therapy by the end of Q3?", {"p4", "p10"}),
        ("Will the aviation authority certify the electric commuter aircraft this year?", {"p4", "p10"}),
        ("Will the competition authority clear the grocery-chain merger without divestitures?", {"p4", "p10"}),
        ("Will the energy regulator grant the offshore wind farm its final operating license?", {"p4", "p10"}),
        ("Will the food-safety agency authorize the cultivated-meat product for retail sale?", {"p4", "p10"}),
        ("Will the telecom regulator approve the spectrum transfer between the two carriers?", {"p4", "p10"}),
        ("Will the securities regulator approve the exchange's new listing framework?", {"p4", "p10"}),
        ("Will the environment ministry issue the mine's water permit despite objections?", {"p4", "p10", "p11"}),
        ("Will the medical-devices board clear the implant after the advisory-panel vote?", {"p4", "p10"}),
        ("Will the banking supervisor approve the regional lender's acquisition bid?", {"p4", "p10"}),
    ])
    # ---- 4. election_outcome (elections; p4/p6/p9pop/p10 formal, p9net media) ----
    add("election", "elections", [
        ("Will the incumbent mayor win re-election in the October runoff?", {"p4", "p6", "p9pop", "p9net", "p10"}),
        ("Will the opposition alliance win a majority in the regional assembly election?", {"p4", "p6", "p9pop", "p9net", "p10", "p11"}),
        ("Will turnout in the municipal election exceed forty percent?", {"p6", "p9pop", "p9net"}),
        ("Will the governing party lose its supermajority in the September vote?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the independent candidate qualify for the presidential debate stage?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the union's endorsed slate win the pension-board election?", {"p4", "p6", "p9pop", "p9net", "p10"}),
        ("Will the student government referendum on fees pass with the required quorum?", {"p6", "p9pop", "p10"}),
        ("Will the challenger concede within a week of the vote if exit polls show a wide margin?", {"p4", "p6", "p9pop"}),
        ("Will the diaspora vote share exceed ten percent of total ballots cast?", {"p6", "p9pop"}),
        ("Will the ruling coalition retain the swing province in the by-election?", {"p4", "p6", "p9pop", "p9net", "p10"}),
    ])
    # ---- 5. referendum (elections; p9pop/p10/p6) ----
    add("referendum", "elections", [
        ("Will the constitutional referendum on term limits be approved by double majority?", {"p6", "p9pop", "p10"}),
        ("Will the city's transit-bond measure pass at the November ballot?", {"p6", "p9pop", "p10"}),
        ("Will the canton approve the energy-transition initiative despite rural opposition?", {"p6", "p9pop", "p10"}),
        ("Will the independence plebiscite reach the participation threshold to be valid?", {"p6", "p7", "p9pop", "p10"}),
        ("Will the school-district consolidation measure survive the recount procedure?", {"p6", "p9pop", "p10"}),
        ("Will the minimum-wage initiative gather enough verified signatures to qualify?", {"p6", "p9pop", "p9net", "p10"}),
        ("Will the coastal-protection levy pass in at least six of the nine districts?", {"p6", "p9pop", "p10"}),
        ("Will the data-privacy charter amendment be adopted by the required margin?", {"p6", "p9pop", "p10"}),
        ("Will the stadium-funding proposition fail for the third consecutive time?", {"p6", "p9pop", "p10"}),
        ("Will the rank-choice-voting measure pass despite the late opposition campaign?", {"p6", "p9pop", "p9net", "p10", "p11"}),
    ])
    # ---- 6. labor_negotiation (labor; p4/p6, p9pop when mass action, p11 sometimes) ----
    add("labor", "labor", [
        ("Will the dockworkers' union and the port operator reach a contract before the strike deadline?", {"p4", "p6", "p11"}),
        ("Will the nurses' union ratify the tentative agreement in the membership vote?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the auto plant's walkout spread to the supplier network within a month?", {"p4", "p6", "p7", "p9pop", "p9net"}),
        ("Will the teachers' strike end before the school year begins?", {"p4", "p6", "p9pop", "p11"}),
        ("Will the airline pilots accept binding arbitration over the scheduling dispute?", {"p4", "p6", "p10"}),
        ("Will the delivery-platform couriers vote to unionize the metropolitan region?", {"p4", "p6", "p9pop", "p9net", "p10"}),
        ("Will the rail union call a national work stoppage during the holiday season?", {"p4", "p6", "p9pop"}),
        ("Will the studio writers reach a residuals deal before the production season?", {"p4", "p6"}),
        ("Will the mining company's lockout trigger sympathy actions at other sites?", {"p4", "p6", "p7", "p9net"}),
        ("Will the public-sector wage talks conclude without mandatory mediation?", {"p4", "p6", "p10"}),
    ])
    # ---- 7. corporate_strategy (business; p4, p10 when regulatory/board) ----
    add("corp", "business", [
        ("Will the two software firms complete their announced merger by year end?", {"p4", "p10", "p11"}),
        ("Will the activist investor win at least two board seats at the annual meeting?", {"p4", "p6", "p10"}),
        ("Will the retailer spin off its logistics arm following the strategic review?", {"p4"}),
        ("Will the chipmaker's hostile bid for its rival succeed at the tender deadline?", {"p4", "p6", "p10", "p11"}),
        ("Will the struggling carrier file for court-supervised restructuring this quarter?", {"p4", "p10"}),
        ("Will the founder return as chief executive after the boardroom dispute?", {"p4", "p10", "p11"}),
        ("Will the energy major divest its refining unit under shareholder pressure?", {"p4", "p6", "p10"}),
        ("Will the bank cut its dividend at the next board meeting?", {"p4", "p10"}),
        ("Will the conglomerate's breakup plan win proxy-adviser support?", {"p4", "p6", "p10"}),
        ("Will the joint venture between the automakers be dissolved before the model launch?", {"p4", "p11"}),
    ])
    # ---- 8. product_adoption (technology; p6/p7/p9pop/p9net) ----
    add("adoption", "technology", [
        ("Will the payment app reach ten million active users by the end of the year?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will household adoption of the smart meter exceed a third of the pilot region?", {"p6", "p7", "p9pop"}),
        ("Will the open-source model become the most-downloaded on the hosting hub this quarter?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the electric scooter service hit its subscriber target in the new market?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the messaging platform's new feature reach half its user base within 90 days?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the vaccine booster campaign reach seventy percent uptake among seniors?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the rural broadband program sign up the majority of eligible households?", {"p6", "p7", "p9pop"}),
        ("Will the browser's privacy mode become the default choice for most new installs?", {"p6", "p9pop"}),
        ("Will the language-learning app double its paying cohort after the price cut?", {"p6", "p9pop"}),
        ("Will the heat-pump subsidy push installations past the saturation of early adopters?", {"p6", "p7", "p9pop", "p9net"}),
    ])
    # ---- 9. information_diffusion (media; p6/p7/p9pop/p9net) ----
    add("diffusion", "media", [
        ("Will the whistleblower's video surpass fifty million views within two weeks?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the boycott hashtag spread from the fan community to mainstream outlets?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the debunked health claim continue to circulate after the platform's label?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the leaked memo trigger coverage in at least three national newspapers?", {"p4", "p6", "p9net"}),
        ("Will the charity's matching campaign go viral enough to hit its goal early?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the satirical clip be mistaken for real footage by a major broadcaster?", {"p6", "p9net"}),
        ("Will the recall notice reach the majority of affected owners within a month?", {"p6", "p9pop", "p9net"}),
        ("Will the influencer's product endorsement move the brand into the top search rank?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the rumor about the bank's solvency trigger deposit withdrawals above 5%?", {"p6", "p7", "p9pop", "p9net", "p11"}),
        ("Will the protest livestream draw a larger online audience than the official address?", {"p6", "p9pop", "p9net"}),
    ])
    # ---- 10. financial_system (finance; mixed; p7 contagion, p10 for policy) ----
    add("finsys", "finance", [
        ("Will the central bank cut its policy rate at the September meeting?", {"p4", "p10"}),
        ("Will the mid-sized lender face a run requiring emergency liquidity support?", {"p6", "p7", "p9pop", "p9net", "p11"}),
        ("Will the sovereign's restructuring talks with bondholders conclude before the coupon date?", {"p4", "p6", "p11"}),
        ("Will contagion from the property developer's default spread to two more issuers?", {"p6", "p7", "p9net", "p11"}),
        ("Will the stablecoin keep its peg through the redemption wave?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the deposit-insurance board raise its coverage cap at the emergency session?", {"p4", "p10", "p11"}),
        ("Will the clearinghouse trigger its default-management auction this quarter?", {"p7", "p10", "p11"}),
        ("Will margin calls force the fund to liquidate its concentrated position?", {"p4", "p7"}),
        ("Will the currency board maintain the peg despite reserve depletion?", {"p4", "p7", "p10", "p11"}),
        ("Will the parliament pass the bank-resolution reform before the IMF review?", {"p4", "p6", "p10"}),
    ])
    # ---- 11. protest_mobilization (civil society; p6/p7/p9pop/p9net, p4 leaders, p11) ----
    add("protest", "civil_society", [
        ("Will the fuel-price protests draw over one hundred thousand participants nationwide?", {"p4", "p6", "p7", "p9pop", "p9net"}),
        ("Will the student occupation spread to the majority of the country's campuses?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the farmers' road blockades force the ministry to withdraw the levy?", {"p4", "p6", "p9pop", "p11"}),
        ("Will the general strike call achieve majority compliance in the capital?", {"p4", "p6", "p9pop", "p9net"}),
        ("Will the opposition rally exceed the turnout of the government's counter-rally?", {"p4", "p6", "p9pop", "p9net"}),
        ("Will the mutual-aid network expand to all affected flood districts within a month?", {"p6", "p9net"}),
        ("Will the curfew halve nightly protest attendance within two weeks?", {"p6", "p7", "p9pop", "p10"}),
        ("Will the coalition of unions and student groups hold together through August?", {"p4", "p6", "p9net", "p11"}),
        ("Will the petition against the dam cross one million verified signatures?", {"p6", "p9pop", "p9net"}),
        ("Will the city grant the march permit after the court's assembly ruling?", {"p4", "p10"}),
    ])
    # ---- 12. international_negotiation (geopolitics; p4/p6, p11) ----
    add("intl", "geopolitics", [
        ("Will the two governments sign the maritime-border agreement at the summit?", {"p4", "p6", "p11"}),
        ("Will the ceasefire talks produce a monitored humanitarian corridor by August?", {"p4", "p6", "p11"}),
        ("Will the sanctions coalition add secondary measures against the intermediaries?", {"p4", "p6", "p10", "p11"}),
        ("Will the grain-export corridor deal be renewed before it lapses?", {"p4", "p6", "p11"}),
        ("Will the alliance admit the applicant state at the autumn ministerial?", {"p4", "p6", "p10"}),
        ("Will the hostage negotiation conclude with an exchange before the deadline?", {"p4", "p6", "p11"}),
        ("Will the climate summit adopt a binding methane-reduction annex?", {"p4", "p6", "p10"}),
        ("Will the border dispute be referred to international arbitration this year?", {"p4", "p10", "p11"}),
        ("Will the trade bloc conclude its digital-services chapter with the island economy?", {"p4", "p6", "p10"}),
        ("Will the peacekeeping mandate be renewed without the contested amendment?", {"p4", "p10", "p11"}),
    ])
    # ---- 13. epidemic_health (health; p6/p7/p9pop/p9net; p10 for authority decisions) ----
    add("health", "health", [
        ("Will the seasonal outbreak push hospital occupancy past the surge threshold?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the school-entry vaccination rate recover to its pre-controversy level?", {"p6", "p9pop", "p9net"}),
        ("Will the antimicrobial-resistance cluster spread beyond the two index hospitals?", {"p7", "p9net"}),
        ("Will the health authority declare the mosquito-borne outbreak a public emergency?", {"p4", "p7", "p10", "p11"}),
        ("Will mask usage on transit exceed half of riders during the winter wave?", {"p6", "p9pop", "p9net"}),
        ("Will the contact-tracing app reach the adoption level needed for effectiveness?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the clinic boycott reduce screening attendance by more than a quarter?", {"p6", "p9pop", "p9net"}),
        ("Will the new treatment guideline be adopted by the majority of regional trusts?", {"p6", "p9net", "p10"}),
        ("Will the blood-donation drive close the projected summer shortfall?", {"p6", "p9pop"}),
        ("Will the outbreak burn out below the herd-immunity threshold without intervention?", {"p7", "p9pop", "p9net"}),
    ])
    # ---- 14. energy_infrastructure (energy; p4/p10 approvals; p7 physical-social mixes) ----
    add("energy", "energy", [
        ("Will the interconnector project receive both national grid approvals by Q4?", {"p4", "p10"}),
        ("Will the nuclear plant's life-extension pass the safety board review?", {"p4", "p10"}),
        ("Will rooftop-solar installations exceed the feeder capacity limit in the pilot city?", {"p6", "p7", "p9pop"}),
        ("Will the grid operator invoke rolling blackouts during the heat season?", {"p7", "p10", "p11"}),
        ("Will the community successfully block the substation siting at the appeals stage?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the gas-storage facility reach its mandated fill level before winter?", {"p4", "p10"}),
        ("Will the coal region's transition fund be approved by the joint committee?", {"p4", "p10"}),
        ("Will demand-response enrollment reach the operator's target this summer?", {"p6", "p9pop"}),
        ("Will the pipeline consortium lose a second anchor investor before financing closes?", {"p4", "p11"}),
        ("Will the offshore lease auction clear at above the reserve price?", {"p4", "p10"}),
    ])
    # ---- 15. sports_individual (sports; mostly controls for social phases) ----
    add("sport", "sports", [
        ("Will the marathon world record be broken at the autumn majors?", set()),
        ("Will the top seed win the season's final grand-slam tournament?", set()),
        ("Will the injured striker return before the knockout round?", set()),
        ("Will the promoted club avoid relegation this season?", set()),
        ("Will the national team qualify for the continental finals?", set()),
        ("Will the veteran swimmer make the podium at the world championships?", {"p11"}),
        ("Will the league's disciplinary panel suspend the captain for the derby?", {"p4", "p10"}),
        ("Will the players' association vote to accept the revenue-sharing deal?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the host city's stadium be ready for the opening ceremony?", {"p4"}),
        ("Will the doping tribunal uphold the sprinter's four-year ban?", {"p4", "p10"}),
    ])
    # ---- 16. weather_natural (weather; pure controls) ----
    add("weather", "weather", [
        ("Will the capital record its hottest July temperature on record?", set()),
        ("Will the hurricane season produce more than fifteen named storms?", set()),
        ("Will the monsoon arrive within a week of its climatological onset date?", set()),
        ("Will the reservoir refill above sixty percent after the spring melt?", set()),
        ("Will the wildfire season burn more area than the five-year median?", {"p7"}),
        ("Will the river crest above the major-flood stage at the delta gauge?", {"p7"}),
        ("Will the city see measurable snowfall before December?", set()),
        ("Will the drought classification be lifted for the central valley by autumn?", set()),
        ("Will the typhoon make landfall north of the strait?", set()),
        ("Will the pollen season end earlier than usual this year?", set()),
    ])
    # ---- 17. geophysics_astronomy (science; pure controls) ----
    add("geo", "science", [
        ("Will a magnitude-six earthquake strike the subduction zone this year?", set()),
        ("Will the volcano's alert level be raised to orange before September?", {"p10"}),
        ("Will the comet brighten enough for naked-eye visibility at perihelion?", set()),
        ("Will the solar cycle's sunspot count peak above the official forecast?", set()),
        ("Will the glacier's terminus retreat more than last year's measurement?", set()),
        ("Will the meteor shower's peak rate exceed one hundred per hour?", set()),
        ("Will the drought index cross its historical minimum at the basin gauge?", set()),
        ("Will the aurora be visible from the mid-latitude capital this quarter?", set()),
        ("Will the coral survey record bleaching beyond the previous extent?", {"p7"}),
        ("Will the seismic swarm subside without a larger event?", set()),
    ])
    # ---- 18. market_price (finance; controls for social phases mostly) ----
    add("price", "finance", [
        ("Will the benchmark crude price close above ninety dollars this quarter?", set()),
        ("Will the exchange rate breach the intervention band before autumn?", {"p4", "p10", "p11"}),
        ("Will the gold price set a new nominal high by September?", set()),
        ("Will the equity index enter a technical correction during earnings season?", {"p7"}),
        ("Will the grain future spike above the export-ban trigger level?", {"p11"}),
        ("Will the freight-rate index double from its spring low?", set()),
        ("Will the carbon-allowance price hold above the floor after the auction?", {"p10"}),
        ("Will the housing-price index post a third consecutive monthly decline?", {"p9pop"}),
        ("Will the memecoin lose ninety percent of its peak value by year end?", {"p6", "p7", "p9pop", "p9net"}),
        ("Will the bond spread tighten after the ratings review?", {"p10"}),
    ])
    # ---- 19. organizational_governance (organizations; p4/p10, p9net authority) ----
    add("org", "organizations", [
        ("Will the university senate adopt the new tenure-review procedure?", {"p4", "p6", "p10"}),
        ("Will the hospital board approve the merger with the regional network?", {"p4", "p10"}),
        ("Will the cooperative's members vote to demutualize at the special meeting?", {"p4", "p6", "p9pop", "p10"}),
        ("Will the standards body ratify the interoperability spec this cycle?", {"p4", "p6", "p10"}),
        ("Will the diocese close a quarter of its parishes under the consolidation plan?", {"p4", "p10"}),
        ("Will the party congress amend the leadership-selection rules?", {"p4", "p6", "p10", "p11"}),
        ("Will the foundation's trustees divest the endowment from fossil holdings?", {"p4", "p6", "p10"}),
        ("Will the professional league expand by two franchises at the owners' vote?", {"p4", "p10"}),
        ("Will the open-source foundation adopt the contested license change?", {"p4", "p6", "p9net", "p10"}),
        ("Will the condo association pass the special assessment for the retrofit?", {"p4", "p6", "p9pop", "p10"}),
    ])
    # ---- 20. crisis_dynamics (geopolitics/mixed; p7/p11 heavy) ----
    add("crisis", "geopolitics", [
        ("Will the border skirmishes escalate into a declared conflict this summer?", {"p4", "p7", "p11"}),
        ("Will the junta call elections after the mediation framework is signed?", {"p4", "p10", "p11"}),
        ("Will the separatist region hold its announced independence vote despite the ban?", {"p4", "p6", "p9pop", "p10", "p11"}),
        ("Will the refugee flow across the mountain corridor double within two months?", {"p7", "p9pop", "p9net", "p11"}),
        ("Will the governing coalition collapse before the confidence vote?", {"p4", "p10", "p11"}),
        ("Will the peace monitors verify the heavy-weapons withdrawal on schedule?", {"p4", "p10", "p11"}),
        ("Will the blockade trigger fuel rationing in the landlocked capital?", {"p7", "p9pop", "p11"}),
        ("Will the assassination attempt reshape the presidential race's field?", {"p4", "p11"}),
        ("Will the mutiny spread beyond the two garrisons that declared it?", {"p4", "p7", "p9net", "p11"}),
        ("Will the interim council agree on a transition timetable with the opposition?", {"p4", "p6", "p10", "p11"}),
    ])
    return R


QUESTIONS = _rows()

assert len(QUESTIONS) == 200, len(QUESTIONS)
assert len({q[0] for q in QUESTIONS}) == 200
assert len({q[5] for q in QUESTIONS}) == 20                    # >= 20 scenario families
assert len({q[4] for q in QUESTIONS}) >= 10                    # >= 10 domains
