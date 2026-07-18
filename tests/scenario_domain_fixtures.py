"""Cross-domain scenario factories for the §19 generality suite.

Fifteen materially different decision scenarios, each with its OWN ScenarioSemanticModel
(types named for THAT domain, pairwise-disjoint record vocabularies), 2-4 actors, an
institution where apt, outcome predicates, and a scripted reaction. Every scenario follows
ONE architecturally-honest causal template — the decision-maker performs a concrete act whose
DIRECT effect is a scenario event; a consequential OTHER actor observes it and reacts through
the same kernel, and only their reaction creates the terminal outcome record (the composite-
escrow scenario is the deliberate exception: one act with dual institutional direct effects).

Nothing here is memorized in production source: the same factory shape parameterizes the
randomized test, which invents type names at test time and still runs end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.phase13.contracts import DecisionProblem
from swm.world_model_v2.phase13.scenario_actions.candidates import (ConcreteAction, ConditionSpec,
                                                                    PlanStep)
from swm.world_model_v2.scenario_schema import ScenarioSemanticModel
from tests.scenario_fixtures import T0, DAY, build_context, build_world

HORIZON = T0 + 60 * DAY


# ---------------------------------------------------------------- scenario container
@dataclass
class Scenario:
    key: str
    schema: ScenarioSemanticModel
    maker: str
    actors: list
    decider: str
    goal_text: str
    event_type: str
    outcome_record: str
    trigger: str
    reaction: dict
    distractor: str = ""
    resource: str = ""
    authority: list = field(default_factory=list)
    composite_ops: list = None                 # domain 15: single act, dual institutional effect
    composite_outcomes: list = None            # extra outcome record types the composite act creates

    # -------------------------------------------------- decision contract
    def problem(self, decision_id=None) -> DecisionProblem:
        return DecisionProblem(decision_id=decision_id or f"{self.key}_d",
                               decision_maker=self.maker,
                               authority=list(self.authority) or [self.key],
                               horizon="2024-06-01T00:00:00Z")

    def context(self, *, n_particles=3, script=None, maker_resources=None, report=None):
        return build_context(self.schema, self.actors, script=script or self.reaction,
                             maker_resources=maker_resources, n_particles=n_particles,
                             report=report)

    def world(self):
        return build_world(self.schema, self.actors)

    # -------------------------------------------------- candidate builders (precompiled)
    def _emit(self, cid, content, *, targets=None, visibility="participants", timing_ts=None,
              resource_commitments=None):
        targets = targets or [self.decider]
        step = PlanStep(step_id=f"{cid}_s1", intent=f"{self.maker} acts toward {self.decider}",
                        target_ids=list(targets), channel="direct", exact_content=content,
                        visibility=visibility, timing_ts=timing_ts,
                        resource_commitments=dict(resource_commitments or {}))
        step.compiled_ops = [{"op": "emit_semantic_event", "semantic_type_id": self.event_type,
                              "exact_content": content, "structured_fields": {"summary": content[:40]},
                              "direct_targets": list(targets), "intended_visibility": visibility}]
        step.compile_meta = {"compiler": "test_precompiled"}
        return ConcreteAction(candidate_id=cid, actor_id=self.maker, title=cid,
                              strategy_class=cid, steps=[step])

    def optimal(self, cid="optimal", title="direct approach"):
        c = self._emit(cid, f"{self.trigger}: the substantive concrete proposal.")
        c.title = title
        return c

    def content_twin(self, cid, content, title=None):
        c = self._emit(cid, content)
        c.title = title or cid
        return c

    def decoy(self, cid="decoy", title="Bold Visionary Masterstroke (all sizzle)"):
        c = self._emit(cid, "a vague gesture that says nothing actionable at all.")
        c.title = title
        return c

    def wrong_actor_candidate(self, cid="wrong_actor"):
        c = self.optimal(cid)
        c.actor_id = self.decider           # not the contract's decision_maker
        return c

    def resource_hog(self, cid="res_hog", amount=999.0):
        return self._emit(cid, f"{self.trigger}: proposal", resource_commitments={self.resource: amount})

    def after_horizon(self, cid="late"):
        return self._emit(cid, f"{self.trigger}: proposal", timing_ts=HORIZON + 30 * DAY)

    def nl_action(self) -> str:
        return (f"personally walk over to {self.decider} and make the case for "
                f"{self.trigger.lower()} over a long lunch, off the record")

    def composite(self, cid="composite"):
        """Domain-15 only: a single step with dual institutional direct effects."""
        step = PlanStep(step_id=f"{cid}_s1", intent="lodge both instruments in one act",
                        target_ids=[self.decider], channel="registry", visibility="participants",
                        exact_content="escrow the deposit AND file the conditional withdrawal, simultaneously")
        step.compiled_ops = list(self.composite_ops)
        step.compile_meta = {"compiler": "test_precompiled"}
        return ConcreteAction(cid, self.maker, title="composite escrow+withdrawal",
                              strategy_class="dual_instrument", steps=[step])

    def composite_partial(self, cid="partial"):
        """Only one of the two instruments — fails the joint outcome."""
        step = PlanStep(step_id=f"{cid}_s1", intent="escrow only", target_ids=[self.decider],
                        channel="registry", visibility="participants", exact_content="escrow only")
        step.compiled_ops = [dict(self.composite_ops[0])]
        step.compile_meta = {"compiler": "test_precompiled"}
        return ConcreteAction(cid, self.maker, title="escrow only", steps=[step])


# ---------------------------------------------------------------- generic builder
def _social_scenario(key, *, maker, decider, entity_type, event_type, proposal_record,
                     outcome_record, trigger, goal_text, distractor="", extra_actors=(),
                     institution=None, decision_record=None, resource=None, authority=None):
    """Assemble a ScenarioSemanticModel plus a decider reaction that creates the outcome record
    ONLY when the decider observes the trigger phrase — the social effect travels through the
    actor loop, never a coefficient."""
    entity_types = {entity_type: {"description": f"{key} principal", "fields": {"name": "str"}}}
    fact_types = {proposal_record: {"description": f"the {key} proposal",
                                    "fields": {"summary": "str", "status": "str"}},
                  outcome_record: {"description": f"the {key} terminal outcome",
                                   "fields": {"detail": "str", "status": "str"}}}
    if decision_record:
        fact_types[decision_record] = {"description": f"one {key} decision-holder's record",
                                       "fields": {"position": "str", "matter": "str"}}
    sem_events = {event_type: {"description": f"{maker}'s concrete act", "fields": {"summary": "str"},
                               "typical_visibility": "participants"}}
    institutional = {}
    if institution:
        institutional[institution] = {"procedure": f"{key} procedure", "decision_holders": [decider],
                                      "decision_record_type": decision_record or "",
                                      "aggregation": {"kind": "single_authority"}, "assumed": True}
    resources = {resource: {"unit": "units", "conserved": True}} if resource else {}
    actors = [maker] + ([distractor] if distractor else []) + list(extra_actors) + [decider]
    actors = list(dict.fromkeys(actors))                # dedup, order-preserving
    roles = {maker: {"role": f"{key}_lead", "why_consequential": "decides",
                     "affordances": [f"advance the {key}", "hold back", "escalate"]},
             decider: {"role": f"{key}_counterparty", "why_consequential": "produces the outcome",
                       "affordances": [f"grant the {key} outcome", "decline", "defer"]}}
    for a in actors:
        roles.setdefault(a, {"role": f"{key}_participant", "why_consequential": "adjacent",
                             "affordances": ["observe", "advise"]})
    schema = ScenarioSemanticModel(
        question=f"Will {maker} achieve the {key} outcome?",
        prediction_timestamp=T0, horizon=HORIZON,
        entity_types=entity_types, fact_types=fact_types, semantic_event_types=sem_events,
        institutional_definitions=institutional, resource_definitions=resources,
        actor_roles=roles,
        outcome_predicates=[{"predicate_id": f"{key}_secured", "record_type": outcome_record,
                             "op": "exists", "option_true": "secured", "option_false": "not_secured"}],
        information_rules={"default_channel": "direct", "default_delay_s": 60.0},
        provenance={"compiler": "test"}).freeze()

    def reaction(world, situation):
        if trigger in situation:
            return [{"op": "create_or_update_record", "record_type": outcome_record,
                     "record_id": f"{key}_out", "status": "secured",
                     "fields": {"detail": "produced by the counterparty's own reaction",
                                "status": "secured"}}]
        return None

    return Scenario(key=key, schema=schema, maker=maker, actors=actors, decider=decider,
                    goal_text=goal_text, event_type=event_type, outcome_record=outcome_record,
                    trigger=trigger, reaction={decider: reaction}, distractor=distractor,
                    resource=resource or "", authority=authority or [key])


# ---------------------------------------------------------------- the fifteen domains
def founder_launch():
    return _social_scenario(
        "founder_launch", maker="founder", decider="head_of_growth", distractor="board_chair",
        entity_type="venture_company", event_type="launch_gtm_briefing",
        proposal_record="go_to_market_plan", outcome_record="market_traction_milestone",
        trigger="LEAN LAUNCH", resource="engineering_sprints",
        goal_text="hit the first traction milestone this quarter")


def partnership_outreach():
    return _social_scenario(
        "partnership_outreach", maker="startup_ceo", decider="lead_investor", distractor="angel_advisor",
        entity_type="seed_stage_startup", event_type="investor_pitch_meeting",
        proposal_record="investor_outreach_packet", outcome_record="signed_term_sheet",
        trigger="CONVICTION", resource="cash_runway_weeks",
        goal_text="secure a term sheet from the lead investor")


def negotiation():
    return _social_scenario(
        "negotiation", maker="buyer_agent", decider="seller_agent", distractor="escrow_officer",
        entity_type="transaction_party", event_type="counteroffer_tendered",
        proposal_record="purchase_negotiation_position", outcome_record="executed_sale_agreement",
        trigger="SPLIT THE DIFFERENCE", goal_text="close the sale within the price band")


def hiring_team():
    return _social_scenario(
        "hiring_team", maker="hiring_manager", decider="candidate_engineer", distractor="recruiting_partner",
        entity_type="engineering_org", event_type="offer_extended",
        proposal_record="compensation_offer_packet", outcome_record="accepted_employment_agreement",
        trigger="EQUITY REFRESH", goal_text="get the senior candidate to accept")


def product_pricing():
    return _social_scenario(
        "product_pricing", maker="pricing_lead", decider="enterprise_customer",
        entity_type="saas_product_line", event_type="pricing_proposal_shared",
        proposal_record="tiered_pricing_proposal", outcome_record="closed_annual_contract",
        trigger="USAGE BASED TIER", goal_text="close the enterprise account this cycle")


def institutional_procedure():
    return _social_scenario(
        "institutional_procedure", maker="applicant", decider="review_board_chair", distractor="board_secretary",
        entity_type="applicant_firm", event_type="license_application_filed",
        proposal_record="license_application_dossier", outcome_record="issued_operating_license",
        trigger="COMPLETE DOSSIER", goal_text="obtain the operating license",
        institution="licensing_board", decision_record="board_member_ballot")


def coalition_group():
    return _social_scenario(
        "coalition_group", maker="coalition_organizer", decider="swing_member", distractor="opposition_whip",
        entity_type="legislative_faction", event_type="coalition_appeal_made",
        proposal_record="coalition_platform_memo", outcome_record="ratified_bloc_agreement",
        trigger="SHARED PLANK", goal_text="ratify the bloc agreement",
        institution="voting_bloc", decision_record="bloc_member_pledge")


def operational_allocation():
    return _social_scenario(
        "operational_allocation", maker="operations_director", decider="warehouse_lead",
        entity_type="distribution_hub", event_type="allocation_directive_issued",
        proposal_record="capacity_allocation_plan", outcome_record="fulfilled_shipment_commitment",
        trigger="NIGHT SHIFT SURGE", resource="labor_hours",
        goal_text="fulfil the shipment commitment on time")


def legal_regulatory():
    return _social_scenario(
        "legal_regulatory", maker="general_counsel", decider="regulator_examiner", distractor="outside_counsel",
        entity_type="regulated_entity", event_type="response_brief_submitted",
        proposal_record="compliance_response_brief", outcome_record="closed_enforcement_matter",
        trigger="REMEDIATION PLAN", goal_text="resolve the enforcement matter without penalty",
        institution="regulatory_panel", decision_record="examiner_finding")


def personal_relationship():
    return _social_scenario(
        "personal_relationship", maker="partner_a", decider="partner_b",
        entity_type="household", event_type="heartfelt_conversation_opened",
        proposal_record="relationship_repair_overture", outcome_record="renewed_commitment_understanding",
        trigger="I HEAR YOU", goal_text="rebuild trust after the argument")


def crisis_response():
    return _social_scenario(
        "crisis_response", maker="incident_commander", decider="community_liaison", distractor="press_officer",
        entity_type="affected_region", event_type="public_safety_advisory_issued",
        proposal_record="crisis_response_directive", outcome_record="restored_service_confirmation",
        trigger="EVACUATE ZONE A", goal_text="restore safe service to the region")


def information_gathering():
    return _social_scenario(
        "information_gathering", maker="analyst_lead", decider="field_source",
        entity_type="research_program", event_type="information_request_sent",
        proposal_record="intelligence_gap_memo", outcome_record="corroborated_finding_record",
        trigger="ON THE RECORD", goal_text="corroborate the finding before deciding")


def timing_sensitive():
    return _social_scenario(
        "timing_sensitive", maker="trading_desk_head", decider="counterparty_desk",
        entity_type="trading_book", event_type="block_trade_offer_sent",
        proposal_record="execution_timing_plan", outcome_record="filled_block_order",
        trigger="BEFORE THE BELL", goal_text="fill the block before the window closes")


def contingent_policy():
    return _social_scenario(
        "contingent_policy", maker="policy_owner", decider="ops_monitor",
        entity_type="service_platform", event_type="contingency_trigger_armed",
        proposal_record="escalation_runbook", outcome_record="executed_failover_action",
        trigger="LATENCY BREACH", goal_text="execute failover only if latency breaches")


def composite_escrow():
    """A single act with dual institutional semantics: escrow the deposit AND file a conditional
    withdrawal notice at once — not representable by any single legacy verb."""
    maker, custodian, clerk = "deal_principal", "neutral_custodian", "court_clerk"
    schema = ScenarioSemanticModel(
        question=f"Will {maker} secure a protected exit position?",
        prediction_timestamp=T0, horizon=HORIZON,
        entity_types={"disputed_asset": {"description": "the asset", "fields": {"name": "str"}}},
        fact_types={
            "escrow_deposit_record": {"description": "deposit lodged with the custodian",
                                      "fields": {"amount": "str", "status": "str"}},
            "conditional_withdrawal_notice": {"description": "withdrawal filed with the registry",
                                              "fields": {"condition": "str", "status": "str"}},
            "custodian_release_order": {"description": "custodian's own decision record",
                                        "fields": {"position": "str", "matter": "str"}},
            "registry_docket_entry": {"description": "registry clerk's own decision record",
                                      "fields": {"position": "str", "matter": "str"}}},
        semantic_event_types={"dual_instrument_lodged": {"description": "the composite filing",
                                                         "fields": {"summary": "str"},
                                                         "typical_visibility": "participants"}},
        institutional_definitions={
            "escrow_house": {"procedure": "holds deposits", "decision_holders": [custodian],
                             "decision_record_type": "custodian_release_order",
                             "aggregation": {"kind": "single_authority"}, "assumed": True},
            "court_registry": {"procedure": "dockets filings", "decision_holders": [clerk],
                               "decision_record_type": "registry_docket_entry",
                               "aggregation": {"kind": "single_authority"}, "assumed": True}},
        resource_definitions={"deposit_funds": {"unit": "usd", "conserved": True}},
        actor_roles={maker: {"role": "principal", "why_consequential": "decides",
                             "affordances": ["lodge instruments", "wait"]},
                     custodian: {"role": "escrow custodian", "why_consequential": "holds deposit",
                                 "affordances": ["accept deposit", "release"]},
                     clerk: {"role": "registry clerk", "why_consequential": "dockets notice",
                             "affordances": ["docket", "reject"]}},
        outcome_predicates=[
            {"predicate_id": "deposit_escrowed", "record_type": "escrow_deposit_record",
             "op": "exists", "option_true": "escrowed", "option_false": "not_escrowed"},
            {"predicate_id": "withdrawal_filed", "record_type": "conditional_withdrawal_notice",
             "op": "exists", "option_true": "filed", "option_false": "not_filed"}],
        information_rules={"default_channel": "registry", "default_delay_s": 60.0},
        provenance={"compiler": "test"}).freeze()
    dual_ops = [
        {"op": "create_or_update_record", "record_type": "escrow_deposit_record", "record_id": "esc1",
         "status": "held", "fields": {"amount": "500k", "status": "held"}},
        {"op": "create_or_update_record", "record_type": "conditional_withdrawal_notice", "record_id": "wd1",
         "status": "filed", "fields": {"condition": "release on default", "status": "filed"}}]
    return Scenario(key="composite_escrow", schema=schema, maker=maker,
                    actors=[maker, custodian, clerk], decider=custodian,
                    goal_text="secure a protected exit (escrow + conditional withdrawal at once)",
                    event_type="dual_instrument_lodged", outcome_record="escrow_deposit_record",
                    trigger="DUAL LODGE", reaction={}, resource="deposit_funds",
                    authority=["deal_principal"], composite_ops=dual_ops,
                    composite_outcomes=["escrow_deposit_record", "conditional_withdrawal_notice"])


ALL_FACTORIES = [
    founder_launch, partnership_outreach, negotiation, hiring_team, product_pricing,
    institutional_procedure, coalition_group, operational_allocation, legal_regulatory,
    personal_relationship, crisis_response, information_gathering, timing_sensitive,
    contingent_policy, composite_escrow,
]


def all_scenarios() -> list:
    return [f() for f in ALL_FACTORIES]


def scenario_by_key(key: str) -> Scenario:
    for f in ALL_FACTORIES:
        if f.__name__ == key:
            return f()
    raise KeyError(key)


# ---------------------------------------------------------------- randomized scenario builder
_SYLLABLES = ("bo", "ka", "ze", "mir", "lon", "tuv", "wex", "pol", "dra", "nem", "qua", "fim",
              "sor", "gild", "vun", "yat", "rho", "cel", "obu", "zeph")


def random_scenario(seed: int) -> Scenario:
    """Invent every scenario type name at test time — no memorized literals."""
    import random
    rng = random.Random(seed)

    def word(n=2):
        return "".join(rng.choices(_SYLLABLES, k=n))

    maker, decider = f"actor_{word(1)}", f"actor_{word(1)}"
    while decider == maker:
        decider = f"actor_{word(1)}"
    event_type = f"{word(2)}_signal"
    proposal_record = f"{word(2)}_proposal"
    outcome_record = f"{word(2)}_outcome"
    entity_type = f"{word(2)}_entity"
    trigger = word(2).upper()
    return _social_scenario(
        key=f"rand_{word(2)}", maker=maker, decider=decider, entity_type=entity_type,
        event_type=event_type, proposal_record=proposal_record, outcome_record=outcome_record,
        trigger=trigger, goal_text=f"achieve the {outcome_record}", authority=["mandate"])
