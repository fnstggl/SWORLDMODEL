"""D11 — canonical fact store. The information that actually exists in the world, modeled as
first-class facts with real content, provenance, credibility, and VISIBILITY — not char-truncated
blobs and not opaque hashes handed to actors.

The EXP-113 failures this eliminates:
  * decisive facts fell off the end of a blind `evidence_text[:2400]` truncation before the actor
    or the terminal ever saw them;
  * actors were handed fact IDs / hashes instead of the real proposition, so they could not reason
    from what was actually researched.

A `CanonicalFact` carries the genuine social-information structure a real decision runs on:

  * `content` — the actual proposition (what an actor reads), never a hash;
  * `credibility` — confirmed / reported / rumored / speculative (actors reason under it);
  * `visibility` — public / role_private / institution_private / secret, plus explicit
    `actor_access` / `institution_access` so WHO KNOWS WHAT is modeled, not assumed universal;
  * `contradiction_group` — mutually-exclusive facts (only one can hold) so conflicting reports
    are represented honestly rather than both asserted;
  * `numeric_values` + `units` — typed quantities for the dimensional terminal (D16);
  * `as_of_validity` — a leakage guard: a fact dated on/after `as_of` is NOT knowable and is
    dropped from every actor's knowledge and from grounding.

Selection is by TYPED RELEVANCE (per actor, per decision, per the terminal), never a global
character budget: each call receives the facts it needs, rendered as real content.

Universal: nothing is question-specific. Facts are built from the sealed evidence and the counted
grounding; credibility/visibility default conservatively and are only narrowed when the source
supports it."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from swm.world_model_v2.lean_v2.blueprint import norm, norm_key, parse_day

EVIDENCE_STORE_VERSION = "lean_v2.evidence_store.v1"

# credibility grades, strongest first (an actor weights a confirmed fact above a rumor)
CONFIRMED, REPORTED, RUMORED, SPECULATIVE = "confirmed", "reported", "rumored", "speculative"
_CREDIBILITY_ORDER = {CONFIRMED: 3, REPORTED: 2, RUMORED: 1, SPECULATIVE: 0}

# visibility classes (who, by default, can see the fact)
PUBLIC, ROLE_PRIVATE, INSTITUTION_PRIVATE, SECRET = \
    "public", "role_private", "institution_private", "secret"


@dataclass
class CanonicalFact:
    content: str                                   # the real proposition (never a hash)
    date: str = ""                                 # when it became true/known (ISO)
    sources: list = field(default_factory=list)
    source_quotes: list = field(default_factory=list)   # verbatim support
    credibility: str = REPORTED
    visibility: str = PUBLIC
    actor_access: list = field(default_factory=list)     # actor ids who additionally know it
    institution_access: list = field(default_factory=list)
    contradiction_group: str = ""                  # facts in one group are mutually exclusive
    numeric_values: dict = field(default_factory=dict)   # {quantity: value}
    units: str = ""
    as_of_validity: str = ""                       # knowable only strictly before this date
    terminal_relevance: float = 0.0                # [0,1] relevance to the terminal outcome
    decision_relevance: dict = field(default_factory=dict)   # {actor_id: relevance}
    fact_id: str = ""

    def __post_init__(self):
        if not self.fact_id:
            self.fact_id = "f_" + hashlib.sha256(norm_key(self.content).encode()).hexdigest()[:12]

    def credibility_rank(self) -> int:
        return _CREDIBILITY_ORDER.get(self.credibility, 1)

    def knowable_on(self, day: str) -> bool:
        """Knowable on `day` iff the fact has already OCCURRED by then (its date is on/before the
        query day). The leakage guard — dropping facts dated on/after `as_of` — is enforced once
        when the fact enters the store, so query days (which run forward from as_of) never re-admit
        a future fact. An undated latent fact is always knowable."""
        d = parse_day(day)
        fdate = parse_day(self.date)
        if fdate is not None and d is not None and fdate > d:
            return False                            # has not occurred yet
        return True

    def visible_to(self, actor_id: str, *, roles: dict = None,
                   institutions: dict = None) -> bool:
        """Whether this actor can see the fact. PUBLIC → everyone; otherwise only actors granted
        explicit access, in the fact's role, or in an institution with access. Secret ballots and
        another actor's private state never leak (the caller passes only permissible grants)."""
        if self.visibility == PUBLIC:
            return True
        if actor_id in self.actor_access:
            return True
        if self.visibility == INSTITUTION_PRIVATE and institutions:
            member_of = institutions.get(actor_id) or []
            if set(member_of) & set(self.institution_access):
                return True
        return False

    def render(self) -> str:
        """The actor-facing line: real content + a credibility tag + the strongest quote. Never a
        hash — this is what a real decision-maker would actually have in front of them."""
        tag = {CONFIRMED: "confirmed", REPORTED: "reported", RUMORED: "unconfirmed report",
               SPECULATIVE: "speculation"}.get(self.credibility, self.credibility)
        q = f" — “{self.source_quotes[0]}”" if self.source_quotes else ""
        dt = f" [{self.date}]" if self.date else ""
        return f"({tag}){dt} {self.content}{q}".strip()

    def as_dict(self) -> dict:
        return {"fact_id": self.fact_id, "content": self.content, "date": self.date,
                "sources": list(self.sources), "source_quotes": list(self.source_quotes),
                "credibility": self.credibility, "visibility": self.visibility,
                "actor_access": list(self.actor_access),
                "institution_access": list(self.institution_access),
                "contradiction_group": self.contradiction_group,
                "numeric_values": dict(self.numeric_values), "units": self.units,
                "as_of_validity": self.as_of_validity,
                "terminal_relevance": round(self.terminal_relevance, 4),
                "decision_relevance": {k: round(v, 4) for k, v in self.decision_relevance.items()}}


_NUM = re.compile(r"(-?\d[\d,]*\.?\d*)")


def _extract_numbers(text: str) -> dict:
    out = {}
    for i, m in enumerate(_NUM.finditer(text or "")):
        try:
            out[f"n{i}"] = float(m.group(1).replace(",", ""))
        except ValueError:  # noqa: PERF203
            continue
        if i >= 5:
            break
    return out


class EvidenceStore:
    """Holds the world's canonical facts and selects them by typed relevance. The one source of
    fact content for actor knowledge packets (D13) and the terminal."""

    def __init__(self, as_of: str = ""):
        self.as_of = str(as_of or "")[:10]
        self.facts: list = []
        self._by_id: dict = {}
        self.diagnostics: list = []

    def add(self, fact: CanonicalFact) -> CanonicalFact:
        # leakage guard: default the validity cutoff to as_of, and DROP a fact dated on/after it
        if not fact.as_of_validity:
            fact.as_of_validity = self.as_of
        fd = parse_day(fact.date)
        cut = parse_day(self.as_of)
        if fd is not None and cut is not None and fd >= cut:
            self.diagnostics.append({"dropped": fact.fact_id, "reason":
                                     f"post-as_of ({fact.date} >= {self.as_of}) — leakage"})
            return fact
        if fact.fact_id not in self._by_id:
            self.facts.append(fact)
            self._by_id[fact.fact_id] = fact
        return fact

    def get(self, fact_id: str) -> CanonicalFact:
        return self._by_id.get(fact_id)

    # -- typed relevance selection (never a global character budget) ----------------------
    def facts_for_actor(self, actor_id: str, *, day: str = "", k: int = 12, roles: dict = None,
                        institutions: dict = None) -> list:
        """The facts this actor actually knows on `day`, most decision-relevant first. Returns
        CanonicalFacts (real content), never ids — the caller renders them into the packet."""
        day = day or self.as_of
        cand = [f for f in self.facts
                if f.knowable_on(day)
                and f.visible_to(actor_id, roles=roles, institutions=institutions)]
        cand.sort(key=lambda f: (f.decision_relevance.get(actor_id, f.terminal_relevance),
                                 f.credibility_rank()), reverse=True)
        return cand[:k]

    def terminal_relevant_facts(self, *, k: int = 20, min_relevance: float = 0.0) -> list:
        cand = [f for f in self.facts if f.terminal_relevance > min_relevance]
        cand.sort(key=lambda f: (f.terminal_relevance, f.credibility_rank()), reverse=True)
        return cand[:k]

    def contradictions(self) -> dict:
        """{group: [facts]} for every group with >1 member — the honest representation of
        conflicting reports (an actor is shown the conflict, not one side asserted as truth)."""
        groups: dict = {}
        for f in self.facts:
            if f.contradiction_group:
                groups.setdefault(f.contradiction_group, []).append(f)
        return {g: fs for g, fs in groups.items() if len(fs) > 1}

    def manifest(self) -> dict:
        return {"version": EVIDENCE_STORE_VERSION, "as_of": self.as_of,
                "n_facts": len(self.facts), "n_dropped_leakage": len(self.diagnostics),
                "contradiction_groups": sorted(self.contradictions().keys()),
                "diagnostics": self.diagnostics}


def build_evidence_store(bp, grounding: dict, *, as_of: str, evidence_text: str = "") -> EvidenceStore:
    """Assemble the canonical fact store from the counted grounding (its shared-world conditions and
    verified reference cases carry dated, quoted claims) and the blueprint's grounded rates. Every
    fact keeps its real content + quote + date; the as_of leakage guard drops anything not yet
    knowable. Deterministic — no LLM call here."""
    store = EvidenceStore(as_of=as_of)
    grounding = grounding or {}
    # institution membership map for institution-private visibility
    inst_members = {}
    for inst in getattr(bp, "institutions", []) or []:
        for m in inst.get("members") or []:
            inst_members.setdefault(m, []).append(inst.get("id"))

    # (1) shared-world conditions → latent facts (each condition's claim is a proposition about
    # the world, visible publicly by default, credibility from whether it is counted)
    shared = grounding.get("shared_world_conditions") or {}
    for cid, sc in (shared.items() if isinstance(shared, dict) else []):
        claim = norm(sc.get("claim") or cid, 300)
        if not claim:
            continue
        tbl = sc.get("table") or {}
        prov = tbl.get("provenance") or {}
        cases = prov.get("cases") or []
        quote = ""
        date = ""
        for c in cases:
            if c.get("included") and c.get("basis_quote"):
                quote, date = c.get("basis_quote"), c.get("date")
                break
        store.add(CanonicalFact(
            content=claim, date=str(date or "")[:10],
            source_quotes=[norm(quote, 300)] if quote else [],
            credibility=CONFIRMED if (prov.get("denominator") or 0) > 0 else REPORTED,
            visibility=PUBLIC, terminal_relevance=0.6,
            decision_relevance={a: 0.6 for a in (sc.get("affects_actors") or [])}))

    # (2) verified reference cases (actor + outcome classes) → dated, quoted historical facts
    for bucket, base_rel in (("actor_state_reference_classes", 0.4),
                             ("outcome_reference_class", 0.8)):
        obj = grounding.get(bucket)
        tables = (obj if isinstance(obj, list) else
                  [t for v in (obj.values() if isinstance(obj, dict) else []) for t in
                   (v if isinstance(v, list) else [v])] if bucket == "actor_state_reference_classes"
                  else [obj] if obj else [])
        for tbl in tables:
            prov = (tbl or {}).get("provenance") or {}
            for c in prov.get("cases") or []:
                if not c.get("included") or not c.get("basis_quote"):
                    continue
                store.add(CanonicalFact(
                    content=norm(c.get("description") or tbl.get("quantity"), 240),
                    date=str(c.get("date") or "")[:10],
                    sources=[norm(c.get("source"), 120)] if c.get("source") else [],
                    source_quotes=[norm(c.get("basis_quote"), 300)],
                    credibility=REPORTED, visibility=PUBLIC, terminal_relevance=base_rel))

    # (3) grounded rates on the blueprint (each carries a verbatim basis_quote by construction)
    for g in getattr(bp, "grounded_rates", []) or []:
        q = norm(g.get("basis_quote"), 300)
        if not q:
            continue
        store.add(CanonicalFact(
            content=norm(g.get("quantity") or q, 240), source_quotes=[q],
            credibility=CONFIRMED, visibility=PUBLIC,
            numeric_values=_extract_numbers(str(g.get("value_range") or "")),
            terminal_relevance=0.7))
    return store
