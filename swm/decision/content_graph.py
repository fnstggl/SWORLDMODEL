"""MESSAGE CONTENT GRAPH — search over IDEAS and INFORMATION before language.

The reply-first planner (reply_first.py) already inverts generate-and-rank: it starts from the
desired reply and searches over BEAT STRUCTURES. But a beat is still a *slot for a sentence* — the
search reaches language almost immediately. The owner's directive is sharper: "The system is still
searching over emails. It should search over ideas and information first, then turn the winning plan
into language."

This module adds the missing upstream stage. The unit of search is not a sentence and not a beat —
it is a CONTENT UNIT: one true ingredient (an identity fact, a framing of the problem, a non-obvious
insight, one framing of the real evidence, a recipient-specific reason to care, a shape of the ask,
a tone note). The stage runs:

  A. BUILD    — from the sender's real facts + the recipient dossier, generate the full INGREDIENT
                SET: several problem formulations, several insights, several evidence FRAMINGS of the
                *same true numbers* (never new numbers), recipient-specific relevance reasons,
                several request shapes, tone notes. Deterministic validation quarantines — loudly —
                any unit whose numbers are not in the facts, whose text is too long, or whose
                identity/credibility claim is not traceable to a real fact.
  C. PLAN     — search over WHICH units belong, WHICH to omit (with a reason), the ORDER, and the
                ONE carrying idea. Several diverse SemanticPlans + one deliberately minimal plan
                (carrying idea + ask only) as the brevity baseline. Deterministic plan checks:
                exactly one request unit, at most one number-bearing evidence unit (the one-number
                rule), at least one relevance unit unless its omission is stated.
  D. VERBALIZE— only now is a plan turned into language: n complete emails per plan, under the SAME
                hard writing rules the reply-first planner uses.
  E. ADVERSARIAL DELETION — for the surviving finalist: find the strongest reason the recipient
                would ignore it and make ONE targeted repair; then probe every sentence's deletion.
                Each variant that passes the truth gate is a candidate the CALLER ranks blind — this
                stage never certifies a winner (no self-certification).

Everything degrades: with no chat_fn, BUILD derives units mechanically from the facts (so the graph
is testable offline) and VERBALIZE returns nothing — you cannot turn a plan into language without a
writer. The reply-first planner wires these in as ADDITIONAL seed candidates + a final deletion
gauntlet, all flowing through the SAME three separated judges (truth / language / blind outcome).
Every LLM call is traced (stages: content_graph, semantic_plan, verbalize, adversarial_deletion).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from swm.decision.iterative_editor import _strip_subject
from swm.decision.llm_moves import _call, allowed_numbers, number_violations, numbers_in

#: the ingredient taxonomy — what a single true unit of content can BE
UNIT_KINDS = ("identity_fact", "credibility_fact", "problem_formulation", "insight", "evidence",
              "recipient_relevance", "request", "tone_note")
#: kinds whose factual claim must be traceable to a real sender fact (not free invention)
_TRACEABLE_KINDS = ("identity_fact", "credibility_fact")
_MAX_UNIT_CHARS = 220

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = set("a an the and or but if to of in on for with is are was were be been being this that it "
            "i you he she we they my your our their me him her them as at by from about into so not "
            "no do does did would could should will can im youre its thats what who how have has had "
            "not our per was".split())


# ---------------------------------------------------------------- small deterministic helpers
def _sentences(text: str) -> list:
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def _tokens(s: str) -> set:
    return {w for w in _TOKEN.findall((s or "").lower()) if len(w) > 2 and w not in _STOP}


def _fact_traceable(text: str, facts: list, *, thresh: float = 0.5) -> bool:
    """LLM-FREE traceability: an identity/credibility claim must be a substring of some fact OR share
    >= `thresh` of its content tokens with some fact. Blocks 'a Princeton admit featured in the NYT'
    style invention when the facts say only 'starting Princeton'."""
    t = (text or "").strip().lower()
    if not t:
        return False
    ut = _tokens(text)
    for f in (facts or []):
        fl = str(f).lower()
        if t in fl or fl in t:
            return True
        if ut and len(ut & _tokens(f)) / len(ut) >= thresh:
            return True
    return False


def _unit_has_number(unit: "ContentUnit") -> bool:
    """A unit 'carries a number' if its text or declared numbers_used hold a distinctive number
    (small counting integers 0-12 do not count — '3 sentences' is not a statistic)."""
    if numbers_in(unit.text):
        return True
    return bool(numbers_in(" ".join(str(n) for n in (unit.numbers_used or []))))


def _classify_offline(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(building|build|starting|start|years?[ -]old|year[ -]old|run|running|"
                 r"started|founder|studying|student)\b", t):
        return "identity_fact"
    if re.search(r"\d[\d,.]*\s*%|\d[\d,.]*\s*[xmkb]\b|per dollar|goodput|throughput|benchmark|"
                 r"latency", t):
        return "evidence"
    return "credibility_fact"


def _trace(trace_path: str, stage: str, prompt: str, response: str):
    if trace_path:
        with open(trace_path, "a") as f:
            f.write(json.dumps({"stage": stage, "prompt": prompt, "response": response}) + "\n")


def _llm_json(chat_fn, prompt, *, trace_path=None, stage="content_graph", max_tokens=900,
              temperature=0.5):
    """One traced LLM call, tolerant JSON-object parse. Returns (obj_or_None, raw_text)."""
    try:
        raw = _call(chat_fn, prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception:  # noqa: BLE001 — a dead backend degrades to the offline path / no units
        raw = ""
    _trace(trace_path, stage, prompt, raw)
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None, raw
    try:
        return json.loads(m.group(0)), raw
    except ValueError:
        return None, raw


def _llm_list(chat_fn, prompt, *, trace_path=None, stage="verbalize", max_tokens=900,
              temperature=0.6):
    """One traced LLM call, tolerant JSON-array-of-strings parse (falls back to line splitting)."""
    try:
        raw = _call(chat_fn, prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception:  # noqa: BLE001
        raw = ""
    _trace(trace_path, stage, prompt, raw)
    m = re.search(r"\[.*\]", raw or "", re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:  # noqa: BLE001
            pass
    out = []
    for ln in (raw or "").splitlines():
        ln = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', "", ln).strip().strip('"').strip()
        if ln and not ln.startswith("["):
            out.append(ln)
    return out


# ---------------------------------------------------------------- (A) the content unit
@dataclass
class ContentUnit:
    """One true ingredient of the message, in plain words — searched over BEFORE any sentence."""
    kind: str                                    # one of UNIT_KINDS
    text: str                                    # the ingredient in plain words
    belief_target: str = ""                      # what the reader should believe after reading it
    source: str = "derived"                      # sender_fact | derived | dossier | derived_offline
    numbers_used: list = field(default_factory=list)
    unit_id: str = ""

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "kind": self.kind, "text": self.text,
                "belief_target": self.belief_target, "source": self.source,
                "numbers_used": list(self.numbers_used)}


@dataclass
class ContentGraph:
    units: list = field(default_factory=list)          # [ContentUnit]
    rejected_units: list = field(default_factory=list)  # [{"unit": {...}, "reason": str}]
    source: str = "llm"                                # llm | offline

    def by_id(self) -> dict:
        return {u.unit_id: u for u in self.units}

    def kinds(self) -> set:
        return {u.kind for u in self.units}

    def of_kind(self, kind: str) -> list:
        return [u for u in self.units if u.kind == kind]

    def summary(self) -> dict:
        return {"source": self.source, "n_units": len(self.units),
                "n_rejected": len(self.rejected_units),
                "units": [u.to_dict() for u in self.units],
                "rejected_units": self.rejected_units}


def _validate_unit(kind, text, numbers_used, allowed, facts) -> tuple:
    """Deterministic admissibility for ONE unit. Returns (ok, reason). Rejections are LOUD (the
    caller drops the unit into rejected_units): a fabricated number, over-long text, or an identity/
    credibility claim the facts do not support never enters the ingredient set."""
    text = (text or "").strip()
    if kind not in UNIT_KINDS:
        return False, f"unknown kind '{kind}'"
    if not text:
        return False, "empty text"
    if len(text) > _MAX_UNIT_CHARS:
        return False, f"text {len(text)} chars > {_MAX_UNIT_CHARS} limit"
    bad = number_violations(text, allowed)
    if bad:
        return False, f"number(s) not in sender facts: {bad}"
    for n in (numbers_used or []):
        nb = number_violations(str(n), allowed)
        if nb:
            return False, f"declared number(s) not in sender facts: {nb}"
    if kind in _TRACEABLE_KINDS and not _fact_traceable(text, facts):
        return False, "identity/credibility claim not traceable to any sender fact"
    return True, ""


_BUILD_SYSTEM = (
    "You are decomposing a cold outreach into its TRUE INGREDIENTS — not sentences, not an email. "
    "Each ingredient is one plain-language unit of content the reader could believe on its own. You "
    "will produce a rich, redundant SET so a downstream planner can search over which to keep, drop, "
    "and order. Rules: (1) every fact, number, dataset, and result must come from the sender facts "
    "verbatim in meaning — you may REFRAME the same true numbers many ways, but you may NEVER "
    "introduce a number, client, or result not in the facts; (2) identity and credibility units must "
    "restate something actually in the facts; (3) plain words a stranger instantly gets — no jargon "
    "compounds, no product-category names.")


def build_content_graph(chat_fn, brief, dossier, *, trace_path=None) -> ContentGraph:
    """(A) Build the ingredient set from the SenderBrief facts + recipient dossier.

    ONE LLM call plus one retry. Every returned unit passes deterministic validation
    (allowed_numbers / number_violations against the brief; text <= 220 chars; identity/credibility
    traceable to a fact) or is quarantined LOUDLY into `rejected_units`. Offline (chat_fn=None):
    units are derived mechanically from brief.facts + thesis + dossier evidence titles, labeled
    source='derived_offline'."""
    facts = list(getattr(brief, "facts", []) or [])
    allowed = allowed_numbers(brief.to_prompt() if brief else "")

    if chat_fn is None:
        return _offline_graph(brief, dossier, allowed, facts)

    dossier_txt = dossier.render() if dossier is not None else "(no recipient dossier)"
    prompt = (
        _BUILD_SYSTEM + "\n\n" + dossier_txt + "\n\nSender facts (ground truth; nothing beyond "
        f"these):\n{brief.to_prompt()}\n\n"
        "Produce a broad ingredient set covering: 2-3 different PROBLEM FORMULATIONS (different ways "
        "to frame the real problem), 2-3 non-obvious INSIGHTS, 2-3 EVIDENCE framings of the SAME "
        "true numbers (never new numbers), the IDENTITY and CREDIBILITY facts as separate units, "
        "2-3 RECIPIENT_RELEVANCE reasons this recipient specifically should care (grounded in the "
        "dossier), 2-3 REQUEST shapes (materially different sender asks), and 1-2 TONE_NOTE units. "
        'Return ONLY JSON: {"units": [{"kind": one of '
        f"{list(UNIT_KINDS)}, "
        '"text": "the ingredient in plain words (<=220 chars)", "belief_target": "what the reader '
        'should believe after reading it", "numbers_used": ["only numbers copied from the facts"]}]}')

    obj, _ = _llm_json(chat_fn, prompt, trace_path=trace_path, stage="content_graph",
                       max_tokens=1100, temperature=0.5)
    rows = (obj or {}).get("units") if isinstance(obj, dict) else None
    if not rows:
        # one retry, terser instruction
        obj, _ = _llm_json(chat_fn, prompt + "\nReturn the COMPLETE JSON object with a non-empty "
                           '"units" array.', trace_path=trace_path, stage="content_graph",
                           max_tokens=1100, temperature=0.4)
        rows = (obj or {}).get("units") if isinstance(obj, dict) else None
    if not rows:
        # LLM unusable — fall back to the mechanical graph so the stage still contributes ingredients
        return _offline_graph(brief, dossier, allowed, facts)

    units, rejected = [], []
    for r in rows:
        if not isinstance(r, dict):
            rejected.append({"unit": {"raw": str(r)[:120]}, "reason": "not an object"})
            continue
        kind = str(r.get("kind", "")).strip()
        text = str(r.get("text", "")).strip()
        nums = r.get("numbers_used") or []
        if not isinstance(nums, list):
            nums = [nums]
        ok, reason = _validate_unit(kind, text, nums, allowed, facts)
        if not ok:
            rejected.append({"unit": {"kind": kind, "text": text[:120]}, "reason": reason})
            continue
        src = ("sender_fact" if kind in ("identity_fact", "credibility_fact", "evidence")
               else "dossier" if kind == "recipient_relevance" else "derived")
        units.append(ContentUnit(kind=kind, text=text[:_MAX_UNIT_CHARS],
                                 belief_target=str(r.get("belief_target", ""))[:200], source=src,
                                 numbers_used=[str(n) for n in nums]))
    for i, u in enumerate(units):
        u.unit_id = f"u{i}"
    _trace(trace_path, "content_graph", "[validation]",
           json.dumps({"accepted": len(units), "rejected": rejected}))
    return ContentGraph(units=units, rejected_units=rejected, source="llm")


def _offline_graph(brief, dossier, allowed, facts) -> ContentGraph:
    """Mechanical ingredient derivation (no LLM): one unit per fact, thesis -> problem_formulation,
    ask -> request, dossier evidence titles -> recipient_relevance. Same validation as the LLM path
    (so a fact holding an out-of-facts number would still be quarantined)."""
    raw = []
    for f in facts:
        txt = str(f)
        kind = _classify_offline(txt)
        bt = {"identity_fact": "the sender is a real, specific person",
              "evidence": "the result is real and measured, not a claim",
              "credibility_fact": "the sender is being straight about limits"}.get(kind, "")
        raw.append((kind, txt, bt, "derived_offline",
                    [m for m in re.findall(r"\d[\d,]*(?:\.\d+)?", txt)]))
    thesis = getattr(brief, "thesis", "") or ""
    if thesis:
        raw.append(("problem_formulation", thesis, "the core problem is real and mis-framed by "
                    "others", "derived_offline", []))
        raw.append(("insight", thesis, "there is one genuinely non-obvious claim here",
                    "derived_offline", []))
    raw.append(("request", "Want the one-page memo?", "replying is one cheap keystroke",
                "derived_offline", []))
    for src, txt in (getattr(dossier, "evidence", []) or [])[:3]:
        raw.append(("recipient_relevance", f"Relevant to you: {str(txt)[:120]}",
                    "this is about the recipient's own stated interest", "dossier", []))

    units, rejected = [], []
    for kind, txt, bt, src, nums in raw:
        ok, reason = _validate_unit(kind, txt, nums, allowed, facts)
        if not ok:
            rejected.append({"unit": {"kind": kind, "text": txt[:120]}, "reason": reason})
            continue
        units.append(ContentUnit(kind=kind, text=txt[:_MAX_UNIT_CHARS], belief_target=bt,
                                 source=src, numbers_used=[str(n) for n in nums]))
    for i, u in enumerate(units):
        u.unit_id = f"u{i}"
    return ContentGraph(units=units, rejected_units=rejected, source="offline")


# ---------------------------------------------------------------- (C) semantic plans
@dataclass
class SemanticPlan:
    plan_id: str
    carrying_idea: str                            # unit_id of the ONE idea the message rides on
    included: list = field(default_factory=list)  # [unit_id] in reading order
    omitted: list = field(default_factory=list)   # [{"unit_id","reason"}]
    belief_targets: dict = field(default_factory=dict)   # unit_id -> what the reader should believe
    predicted_shortest_form: str = ""

    def to_dict(self) -> dict:
        return {"plan_id": self.plan_id, "carrying_idea": self.carrying_idea,
                "included": list(self.included), "omitted": list(self.omitted),
                "belief_targets": dict(self.belief_targets),
                "predicted_shortest_form": self.predicted_shortest_form}


def validate_plan(plan: SemanticPlan, units_by_id: dict, *, graph_kinds=None) -> tuple:
    """Deterministic plan admissibility. Returns (ok, reason). Enforces:
      * every included id exists;
      * EXACTLY ONE request unit;
      * AT MOST ONE number-bearing evidence unit (the one-number rule);
      * >= 1 recipient_relevance unit, UNLESS the plan explicitly omits one with a reason (or the
        graph has no relevance unit to include). The deliberately-minimal brevity plan is exempt
        from the relevance requirement."""
    unknown = [u for u in plan.included if u not in units_by_id]
    if unknown:
        return False, f"references unknown unit ids: {unknown}"
    kinds = [units_by_id[u].kind for u in plan.included]
    n_req = kinds.count("request")
    if n_req != 1:
        return False, f"must include exactly one request unit (has {n_req})"
    n_evi_num = sum(1 for u in plan.included
                    if units_by_id[u].kind == "evidence" and _unit_has_number(units_by_id[u]))
    if n_evi_num > 1:
        return False, f"one-number rule: {n_evi_num} number-bearing evidence units (max 1)"
    if plan.plan_id != "minimal" and kinds.count("recipient_relevance") < 1:
        rel_available = ("recipient_relevance" in graph_kinds) if graph_kinds is not None else True
        omitted_rel = any(
            str(o.get("reason", "")).strip()
            and units_by_id.get(o.get("unit_id"), ContentUnit("", "")).kind == "recipient_relevance"
            for o in plan.omitted)
        if rel_available and not omitted_rel:
            return False, "no recipient_relevance unit and no stated reason for omitting one"
    return True, ""


def _minimal_plan(units_by_id: dict) -> SemanticPlan | None:
    """The brevity baseline: the carrying idea + the ask, nothing else."""
    idea = next((uid for uid, u in units_by_id.items()
                 if u.kind in ("insight", "problem_formulation")), None)
    req = next((uid for uid, u in units_by_id.items() if u.kind == "request"), None)
    if idea is None or req is None or idea == req:
        return None
    return SemanticPlan(plan_id="minimal", carrying_idea=idea, included=[idea, req],
                        omitted=[], belief_targets={idea: units_by_id[idea].belief_target,
                                                    req: units_by_id[req].belief_target},
                        predicted_shortest_form="")


def plan_semantics(chat_fn, units, target_replies, *, k=4, trace_path=None) -> list:
    """(C) Search over which units belong, which to omit, order, and the ONE carrying idea.

    Generates k diverse SemanticPlans via one LLM call (plus a minimal brevity plan built
    deterministically). Every plan passes `validate_plan` or is dropped. Offline (chat_fn=None): one
    deterministic full plan (sensible kind order) + the minimal plan — verbalization needs a writer,
    so these matter only when a chat_fn is present downstream."""
    units = list(units or [])
    if not units:
        return []
    by_id = {u.unit_id: u for u in units}
    graph_kinds = {u.kind for u in units}
    plans: list = []

    if chat_fn is not None:
        listing = "\n".join(f"{u.unit_id} [{u.kind}]: {u.text}" for u in units)
        replies = "\n".join(f'- "{r.get("reply", "")}" ({r.get("outcome", "")})'
                            for r in (target_replies or []))
        prompt = (
            "You are PLANNING a cold message at the level of IDEAS, not sentences. Below is a set of "
            "true content units (ingredients). Choose subsets/orderings that would most make the "
            "recipient WANT to send one of the target replies. Search the space: different carrying "
            "ideas, different inclusions, different orders.\n\n"
            f"Content units:\n{listing}\n\nReplies we want to earn:\n{replies or '(none given)'}\n\n"
            f"Return {k} DIVERSE plans. Each plan: one carrying idea (the single unit the whole "
            "message rides on), the included unit ids IN READING ORDER, the omitted units WITH a "
            "one-clause reason each, a per-unit belief_target (what the reader believes after that "
            "unit), and a predicted_shortest_form (the plan in <=25 words, no email formatting). "
            "Constraints every plan must respect: exactly ONE request unit; at most ONE evidence "
            "unit that carries a number; include a recipient_relevance unit unless you state why you "
            'omit it. Return ONLY JSON: {"plans": [{"carrying_idea": "uID", "included": ["uID",...], '
            '"omitted": [{"unit_id": "uID", "reason": "..."}], "belief_targets": {"uID": "..."}, '
            '"predicted_shortest_form": "..."}]}')
        obj, _ = _llm_json(chat_fn, prompt, trace_path=trace_path, stage="semantic_plan",
                           max_tokens=1200, temperature=0.6)
        rows = (obj or {}).get("plans") if isinstance(obj, dict) else None
        for i, r in enumerate(rows or []):
            if not isinstance(r, dict):
                continue
            included = [str(x) for x in (r.get("included") or []) if str(x) in by_id]
            omitted = [{"unit_id": str(o.get("unit_id", "")), "reason": str(o.get("reason", ""))}
                       for o in (r.get("omitted") or []) if isinstance(o, dict)]
            carrying = str(r.get("carrying_idea", "")) or (included[0] if included else "")
            bt = {str(kk): str(vv) for kk, vv in (r.get("belief_targets") or {}).items()
                  if str(kk) in by_id}
            plans.append(SemanticPlan(plan_id=f"p{i}", carrying_idea=carrying, included=included,
                                      omitted=omitted, belief_targets=bt,
                                      predicted_shortest_form=str(
                                          r.get("predicted_shortest_form", ""))[:200]))
    else:
        # deterministic full plan: identity -> problem -> insight -> evidence -> relevance -> request
        order = ("identity_fact", "credibility_fact", "problem_formulation", "insight", "evidence",
                 "recipient_relevance", "request")
        seen_evi_num = False
        included = []
        for kind in order:
            for u in units:
                if u.kind != kind:
                    continue
                if kind == "evidence" and _unit_has_number(u):
                    if seen_evi_num:
                        continue
                    seen_evi_num = True
                if kind == "request" and any(by_id[x].kind == "request" for x in included):
                    continue
                included.append(u.unit_id)
        carrying = next((x for x in included
                         if by_id[x].kind in ("insight", "problem_formulation")),
                        included[0] if included else "")
        if included:
            plans.append(SemanticPlan(plan_id="p0", carrying_idea=carrying, included=included,
                                      belief_targets={x: by_id[x].belief_target for x in included}))

    mp = _minimal_plan(by_id)
    if mp is not None:
        plans.append(mp)

    valid = []
    for p in plans:
        ok, reason = validate_plan(p, by_id, graph_kinds=graph_kinds)
        _trace(trace_path, "semantic_plan", f"[validate {p.plan_id}]",
               json.dumps({"ok": ok, "reason": reason, "plan": p.to_dict()}))
        if ok:
            valid.append(p)
    return valid


# ---------------------------------------------------------------- (D) verbalization
_FALLBACK_WRITING_RULES = (
    "Writing rules (hard): every sentence is something a sharp busy person would actually type. NO "
    "jargon compounds — say what the thing DOES in plain words. AT MOST ONE number in the whole "
    "email, translated into what it means for the reader, with its provenance in the same sentence; "
    "numbers only from the sender facts. The email is written BY the sender TO the recipient: the "
    "closing line is the sender's own ask, never the recipient's hoped-for reply. No ceremonious "
    "permission constructions. No em dashes unless truly necessary. 45-85 words total.")


def _writing_rules() -> str:
    try:  # reuse the reply-first rules verbatim; lazy import avoids any import cycle
        from swm.decision.reply_first import ReplyFirstPlanner
        return ReplyFirstPlanner._WRITING_RULES
    except Exception:  # noqa: BLE001
        return _FALLBACK_WRITING_RULES


def verbalize(chat_fn, plan: SemanticPlan, units, brief, *, n=2, trace_path=None) -> list:
    """(D) Turn ONE plan into n complete emails under the reply-first hard writing rules. This is the
    ONLY stage that produces language. Offline (chat_fn=None) returns [] — a plan cannot become
    language without a writer; the plan/graph search still happened, it simply contributes no draft.
    Number/contract admissibility is enforced downstream by the caller's truth gate."""
    if chat_fn is None or plan is None:
        return []
    by_id = {u.unit_id: u for u in (units or [])}
    idea = by_id.get(plan.carrying_idea)
    ordered = [by_id[x] for x in plan.included if x in by_id]
    if not ordered:
        return []
    ingredient_lines = "\n".join(
        f"{i + 1}. [{u.kind}] {u.text}"
        + (f"  (reader should then believe: {plan.belief_targets.get(u.unit_id, u.belief_target)})"
           if (plan.belief_targets.get(u.unit_id) or u.belief_target) else "")
        for i, u in enumerate(ordered))
    prompt = (
        f"Sender facts (ground truth; nothing beyond these):\n{brief.to_prompt() if brief else ''}\n\n"
        "Write the email from this ALREADY-CHOSEN plan of ideas. Do not add ideas, do not drop the "
        "ask, keep the ingredients in THIS order:\n" + ingredient_lines + "\n"
        + (f"The whole message rides on this one idea: {idea.text}\n" if idea else "")
        + f"{_writing_rules()}\n"
        f"Write {n} DISTINCT complete versions (same plan, different wording). "
        f'Return ONLY a JSON array of {n} strings, each one complete email (greeting through '
        "sign-off). No prose outside the JSON.")
    drafts = _llm_list(chat_fn, prompt, trace_path=trace_path, stage="verbalize",
                       max_tokens=260 * max(1, n) + 200, temperature=0.6)
    out = []
    for d in drafts[:n]:
        t = _strip_subject((d or "").strip().strip('"'))
        if t:
            out.append(t)
    return out


# ---------------------------------------------------------------- (E) adversarial deletion
def _ask_ignore_reason(chat_fn, text, trace_path) -> str:
    prompt = ("A busy, skeptical, high-status recipient just received this cold email from a "
              "stranger. In ONE sentence, what is the STRONGEST reason they would ignore it and not "
              f"reply?\n--- EMAIL ---\n{text}\n--- END ---\nReturn ONLY that one sentence.")
    try:
        raw = _call(chat_fn, prompt, max_tokens=120, temperature=0.3)
    except Exception:  # noqa: BLE001
        raw = ""
    _trace(trace_path, "adversarial_deletion", prompt, raw)
    return (raw or "").strip().strip('"').strip()


def _repair_for_reason(chat_fn, text, reason, brief, trace_path) -> str:
    prompt = (f"Sender facts (ground truth; nothing beyond these):\n"
              f"{brief.to_prompt() if brief else ''}\n--- EMAIL ---\n{text}\n--- END ---\n"
              f"A recipient's single strongest reason to ignore this: \"{reason}\"\n"
              "Rewrite the WHOLE email ONCE to defuse exactly that reason and nothing else. Keep "
              "every fact, keep the sign-off, keep it a real human's quick note. Any number must "
              "come from the sender facts. Return ONLY the email text.")
    try:
        raw = _call(chat_fn, prompt, max_tokens=280, temperature=0.3)
    except Exception:  # noqa: BLE001
        raw = ""
    _trace(trace_path, "adversarial_deletion", prompt, raw)
    return _strip_subject((raw or "").strip().strip('"'))


def adversarial_deletion(chat_fn, text, brief, *, judge=None, trace_path=None) -> dict:
    """(E) Attack the surviving finalist, then hand the CALLER a set of candidates to rank blind.

    (1) REPAIR: ask the recipient's strongest reason to ignore the message, then make ONE targeted
        rewrite that defuses exactly that reason. The rewrite is admitted only if it passes
        number_violations + the cold-outreach contract (identity_window=None) [+ `judge` if given].
    (2) DELETION PROBES: for EVERY body sentence, produce the variant with that one sentence removed
        (deterministic surgery; a bare trailing signature like 'Beckett' is held out and re-appended
        so a probe never strips the name — the reply_first run-4 forensic). `deletions` has exactly
        one entry per body sentence (full coverage), each tagged `truth_ok` = admissible-as-candidate.

    Returns {original, repaired, repaired_ok, reason_ignored, deletions}. Admissible variants
    (repaired if repaired_ok; deletions with truth_ok) are the candidates. This function NEVER picks
    a winner — no self-certification; the caller ranks the survivors blind under its own judges."""
    original = (text or "").strip()
    allowed = allowed_numbers(brief.to_prompt() if brief else "")

    def _admissible(t: str) -> bool:
        t = (t or "").strip()
        if not t or number_violations(t, allowed):
            return False
        from swm.decision.outreach_contract import validate
        if not validate(t, brief, identity_window=None).ok:
            return False
        if judge is not None:
            try:
                if not judge(t).get("ok"):
                    return False
            except Exception:  # noqa: BLE001 — an unverifiable variant is not a candidate
                return False
        return True

    # (1) one strongest-reason repair
    repaired, repaired_ok, reason = "", False, ""
    if chat_fn is not None and original:
        reason = _ask_ignore_reason(chat_fn, original, trace_path)
        cand = _repair_for_reason(chat_fn, original, reason, brief, trace_path)
        if cand and cand != original and _admissible(cand):
            repaired, repaired_ok = cand, True

    # (2) claim-deletion probes — every body sentence removed exactly once, signature preserved
    sents = _sentences(original)
    signoff = None
    if sents and len(sents[-1].split()) <= 3 and not sents[-1].endswith((".", "!", "?")):
        signoff = sents.pop()
    tail = [signoff] if signoff else []
    deletions = []
    for i in range(len(sents)):
        variant = " ".join(sents[:i] + sents[i + 1:] + tail)
        deletions.append({"removed_sentence": sents[i], "text": variant,
                          "truth_ok": _admissible(variant)})

    _trace(trace_path, "adversarial_deletion", "[result]",
           json.dumps({"reason_ignored": reason, "repaired_ok": repaired_ok,
                       "n_deletion_probes": len(deletions),
                       "n_admissible_deletions": sum(1 for d in deletions if d["truth_ok"])}))
    return {"original": original, "repaired": repaired, "repaired_ok": repaired_ok,
            "reason_ignored": reason, "deletions": deletions}
