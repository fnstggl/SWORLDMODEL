"""Semantic action clustering v2 — defragmenting the counted decision distributions.

The pilot evaluation of the qualitative-actor pipeline (qualitative_actor.ActionClusterer,
version "cluster-1.0": exact ``(action_name, target)`` key with an ontology-anchor override)
found two systematic scoring failures: semantically identical actions ("decline_the_offer" vs
"reject") landed in different clusters, and target-string paraphrases ("Bob Smith" vs
"bob_smith" vs "Mr. Smith") fragmented the branch-count distribution — so raw frequencies
were split across spurious clusters and semantically correct decisions scored as wrong.

Version "cluster-2.0" repairs both while keeping every merge auditable and REFUSABLE:

  1. exact typed match — ontology_anchor override first, then exact ontology names;
     byte-compatible with cluster-1.0 keys on every row v1 handled;
  2. canonical target resolution (:class:`TargetCanonicalizer`) — deterministic mapping of
     free-text targets onto known entity ids (lowercase/underscore normalization, honorific
     stripping, full-id match, unique token-containment) that refuses ambiguous matches;
  3. curated ontology equivalence (:data:`ACTION_EQUIVALENCE`) — synonym → ontology action,
     never merging materially different acts (threaten ≠ escalate_message, leak ≠
     reveal_information — see :data:`NON_EQUIVALENT`);
  4. optional LLM-assisted equivalence — candidates-only prompt, strict JSON verdict, refused
     on ``materially_different`` / low confidence / off-menu answers, and cached keyed by
     ``(name, target)`` for deterministic replay (:meth:`ActionClustererV2.export_mappings` /
     :meth:`ActionClustererV2.load_mappings`);
  5. strategy-class fallback — the ontology FAMILY of the nearest action name (token overlap
     >= 1/2 against family action names) as ``family:<family>@<target>``;
  6. genuinely novel actions keep their own cluster (``novel:<name>@<target>``) with the
     original phrasing preserved in the record;
  7. ``unresolved:<name>`` when the target is irreducibly ambiguous AND the action unknown.

Every decision is recorded (``mapping_log``; :meth:`ActionClustererV2.cluster_record`) with
version, method, inputs and outputs. No numeric behavioral fields anywhere: LLM confidence is
categorical, thresholds are structural constants, and :func:`clustering_metrics` is
evaluation-only. :class:`ActionClustererV2` is a drop-in ``clusterer`` for
``qualitative_actor.aggregate_actor_decisions`` (same ``cluster_key(row)`` / ``version``
surface)."""
from __future__ import annotations

import json
import re

from swm.world_model_v2.phase4_policy import ACTION_ONTOLOGY, KNOWN_ACTIONS

CLUSTER_VERSION_2 = "cluster-2.0"

_NORM = re.compile(r"[^a-z0-9]+")

#: honorifics/titles stripped from free-text targets before entity matching — a title is a
#: form of address, not identity ("Mr. Smith", "Senator Doe" name the same ids as "smith",
#: "doe"). Stripping runs AFTER full-id matching, so ids like "president_karev" still match.
HONORIFICS = frozenset((
    "mr", "mrs", "ms", "mx", "miss", "sir", "dame", "madam", "lord", "lady", "hon", "rev",
    "dr", "doc", "doctor", "prof", "professor", "president", "premier", "chancellor",
    "senator", "sen", "secretary", "minister", "governor", "gov", "mayor", "judge", "justice",
    "chair", "chairman", "chairwoman", "chairperson", "speaker", "ceo", "cfo", "coo",
    "director", "general", "gen", "colonel", "col", "captain", "capt", "admiral",
    "ambassador", "amb", "rep", "representative", "congressman", "congresswoman",
    "king", "queen", "prince", "princess", "pope",
))

#: filler tokens carrying no action meaning in snake_case phrases. Deliberately minimal:
#: directional particles (off/out/away/back/down/up/in) are meaning-bearing and STAY —
#: "walk_out" (strike) and "walk_away" (exit) must never collapse into "walk".
_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "for", "and", "or", "my", "our", "their", "his", "her",
    "its", "this", "that", "at", "on", "with", "about", "toward", "towards", "regarding",
    "now", "immediately", "formally", "publicly", "privately", "officially", "quietly",
))

#: filler tokens additionally ignored when matching TARGET tokens against entity ids
_TARGET_STOP = frozenset(("the", "a", "an", "of", "and"))

#: free-text action tokens that must NEVER be lexically merged into the ontology —
#: superficially similar but MATERIALLY DIFFERENT acts. A threat is a coercive act, not
#: escalate_message; a leak is unauthorized disclosure, not reveal_information; block/table/
#: sanction/release/confirm/hold/compromise/betray are ambiguous across several distinct
#: acts. Names containing one of these tokens can reach the ontology only through the LLM
#: judge (which may itself refuse), never through token matching or family fallback.
NON_EQUIVALENT = frozenset((
    "threaten", "leak", "bribe", "blackmail", "sue", "sanction",
    "block", "table", "compromise", "betray", "release", "confirm", "hold",
))

#: curated synonym map: common free-text action names → ontology actions (phase4_policy
#: ACTION_ONTOLOGY). Every entry asserts the two names denote the SAME real-world act by the
#: same actor; near-misses are deliberately absent (see NON_EQUIVALENT). Multi-token keys
#: also serve as lexical patterns inside longer phrases ("put_off_the_vote" → delay).
ACTION_EQUIVALENCE = {
    # --- negotiation
    "agree": "accept", "accept_offer": "accept", "say_yes": "accept", "consent": "accept",
    "refuse": "reject", "decline": "reject", "turn_down": "reject", "rebuff": "reject",
    "counter": "counteroffer", "counter_offer": "counteroffer",
    "counterproposal": "counteroffer",
    "give_ground": "concede", "back_down": "concede",
    "stand_firm": "hold_position", "hold_firm": "hold_position",
    "stay_the_course": "hold_position",
    "postpone": "delay", "put_off": "delay", "stall": "delay", "hold_off": "delay",
    "pause": "delay",
    "walk_away": "exit", "quit": "exit", "resign": "exit", "resign_from": "exit",
    "request_mediation": "seek_mediator", "bring_in_mediator": "seek_mediator",
    # --- participation
    "endorse": "support", "back": "support", "champion": "support", "advocate": "support",
    "resist": "oppose", "object": "oppose", "push_back": "oppose",
    "sit_out": "abstain", "recuse": "abstain",
    "rally": "mobilize",
    "walk_out": "strike",                       # a walkout IS a strike; walk_away is exit
    "demonstrate": "protest", "march": "protest",
    "lobby": "persuade", "convince": "persuade", "sway": "persuade",
    "switch_sides": "defect",
    "collaborate": "coordinate", "team_up": "coordinate",
    "pull_out": "withdraw", "drop_out": "withdraw",
    # --- messaging
    "respond": "reply_now", "write_back": "reply_now",
    "message": "follow_up", "contact": "follow_up", "reach_out": "follow_up",
    "ping": "follow_up", "check_in": "follow_up",
    "confirm_receipt": "acknowledge",           # bare "confirm" is ambiguous → NON_EQUIVALENT
    "seek_clarification": "clarify", "clarification": "clarify",
    "hand_off": "delegate", "assign": "delegate",
    "disregard": "ignore", "brush_off": "ignore",
    "disclose": "reveal_information",           # voluntary disclosure; "leak" stays novel
    "keep_secret": "withhold_information",
    # --- institutional
    "ratify": "approve", "greenlight": "approve", "sign_off": "approve",
    "revise": "amend", "modify": "amend",
    "punt": "defer",
    "put_on_agenda": "place_on_agenda", "add_to_agenda": "place_on_agenda",
    # --- organizational / market
    "recruit": "hire", "onboard": "hire",
    "dismiss": "fire", "terminate": "fire", "lay_off": "fire", "sack": "fire",
    "buy": "purchase", "procure": "purchase",
    "divest": "sell",
    "roll_out": "launch", "go_live": "launch",
    "rescind_offer": "withdraw_offer",
    "take_over": "acquire",
    "fund": "allocate_budget",
    # --- generic
    "do_nothing": "wait", "sit_tight": "wait", "stand_by": "wait",
}


def _norm(text) -> str:
    return _NORM.sub("_", str(text or "").lower()).strip("_")


def _tokens(text, stop=_STOPWORDS) -> frozenset:
    return frozenset(t for t in _norm(text).split("_") if t and t not in stop)


# ------------------------------------------------------------------- target canonicalization
class TargetCanonicalizer:
    """Deterministic canonical entity resolution for free-text targets.

    Given known entity ids (e.g. ``world.entities`` keys) and an optional explicit alias map,
    resolution attempts in order: alias map → exact id → normalized id → honorific-stripped
    id → UNIQUE token-containment (last-name style: "Mr. Smith" → robert_smith when exactly
    one known entity shares the significant tokens; overlap ties are REFUSED). Every attempt
    returns a full record ``{raw, canonical, method, resolved}`` — an ambiguous match keeps
    the original string with ``resolved=False`` rather than guessing, and an unmatched target
    keeps its normalized form so casing/spacing paraphrases still cluster together."""

    def __init__(self, known_entities=(), aliases: dict | None = None):
        self.known_entities = tuple(str(e) for e in known_entities)
        self.aliases = dict(aliases or {})

    def resolve(self, raw, known_entities=(), aliases: dict | None = None) -> dict:
        raw = str(raw or "")
        if not raw:
            return {"raw": "", "canonical": "", "method": "empty", "resolved": True}
        known = tuple(str(e) for e in (known_entities or self.known_entities))
        amap = {**self.aliases, **(aliases or {})}
        if raw in amap:
            return {"raw": raw, "canonical": str(amap[raw]), "method": "alias",
                    "resolved": True}
        norm = _norm(raw)
        by_norm = {_norm(k): str(v) for k, v in amap.items()}
        if norm in by_norm:
            return {"raw": raw, "canonical": by_norm[norm], "method": "alias",
                    "resolved": True}
        kset = set(known)
        if raw in kset:
            return {"raw": raw, "canonical": raw, "method": "exact", "resolved": True}
        if norm in kset:
            return {"raw": raw, "canonical": norm, "method": "normalized_id",
                    "resolved": True}
        tokens = [t for t in norm.split("_") if t and t not in _TARGET_STOP]
        core = [t for t in tokens if t not in HONORIFICS] or tokens
        stripped = "_".join(core)
        if stripped in kset:
            return {"raw": raw, "canonical": stripped, "method": "honorific_stripped",
                    "resolved": True}
        if core and kset:
            cset = set(core)
            overlaps = {e: len(cset & set(_norm(e).split("_"))) for e in kset}
            overlaps = {e: n for e, n in overlaps.items() if n}
            if overlaps:
                best = max(overlaps.values())
                winners = sorted(e for e, n in overlaps.items() if n == best)
                if len(winners) == 1:
                    return {"raw": raw, "canonical": winners[0], "method": "token_unique",
                            "resolved": True}
                # several known entities match equally well — refuse to guess
                return {"raw": raw, "canonical": raw, "method": "ambiguous",
                        "resolved": False}
        return {"raw": raw, "canonical": stripped or norm, "method": "unmatched",
                "resolved": False}


# ------------------------------------------------------------------- the v2 clusterer
_LLM_MAP_PROMPT = """You are judging whether one free-text action name denotes SEMANTICALLY THE SAME real-world
act as one of the candidate ontology actions below, for clustering independently-made decisions. Map ONLY when
the free-text action and a candidate would be the same act by the same actor. Superficially similar but
materially different acts (threatening vs escalating a message, leaking vs officially revealing) must NOT be
merged — for those, answer NONE or set materially_different true.

FREE-TEXT ACTION: {name}
TARGET (context only, do not map it): {target}
CANDIDATE ONTOLOGY ACTIONS (you may ONLY choose from these):
{candidates}

Return ONLY one JSON object:
{{"maps_to": "<one candidate name or NONE>", "justification": "<one short sentence>",
 "confidence": "high|medium|low", "materially_different": true|false}}"""


def _parse_json_obj(text) -> dict | None:
    try:
        r = json.loads(text)
        return r if isinstance(r, dict) else None
    except (TypeError, ValueError):
        m = re.search(r"\{.*\}", str(text or ""), flags=re.S)
        if m:
            try:
                r = json.loads(m.group(0))
                return r if isinstance(r, dict) else None
            except ValueError:
                return None
    return None


class ActionClustererV2:
    """Versioned semantic clustering of selected actions (version "cluster-2.0").

    Mapping hierarchy (each attempt recorded in ``mapping_log`` and the returned record):
    ontology_anchor → exact ontology name → curated equivalence (full name, then unique
    lexical subset) → optional LLM equivalence (cached, refusable) → strategy-class family
    fallback → novel → unresolved. Targets are canonicalized independently by
    :class:`TargetCanonicalizer`. Rows that v1 (cluster-1.0) handled — exact ontology names,
    exact targets — produce byte-identical keys, so v2 distributions remain comparable.

    ``llm`` is an optional ``fn(prompt) -> text`` backend; every LLM verdict (accepted OR
    refused) is cached keyed ``(name, target)`` and exportable via :meth:`export_mappings` /
    :meth:`load_mappings`, so replays are deterministic and LLM-free."""

    version = CLUSTER_VERSION_2

    def __init__(self, llm=None, *, known_entities=(), aliases: dict | None = None):
        self.llm = llm
        self.targets = TargetCanonicalizer(known_entities, aliases)
        self.llm_cache: dict = {}          # (name, canonical_target) -> verdict record
        self.mapping_log: list = []        # every mapping decision, append-only
        self.llm_calls = 0

    # ---- public surface (drop-in for aggregate_actor_decisions) ----------------------
    def cluster_key(self, row: dict, known_entities=(), aliases: dict | None = None) -> str:
        return self.cluster_record(row, known_entities=known_entities, aliases=aliases)["key"]

    def cluster_record(self, row: dict, known_entities=(),
                       aliases: dict | None = None) -> dict:
        raw_name = str(row.get("action_name", "") or "")
        anchor = row.get("ontology_anchor") if isinstance(row.get("ontology_anchor"),
                                                          dict) else {}
        raw_target = str(row.get("target", "") or "")
        tres = self.targets.resolve(raw_target, known_entities=known_entities,
                                    aliases=aliases)
        target = tres["canonical"]
        name = _norm(raw_name) or "unnamed"
        resolved, method = None, ""
        if anchor and anchor.get("name"):
            resolved, method = str(anchor["name"]), "ontology_anchor"   # exactly like v1
        elif name in KNOWN_ACTIONS:
            resolved, method = name, "exact"
        else:
            hit = self._equivalent(name)
            if hit is None:
                hit = self._llm_equivalent(name, target)
            if hit is None:
                family = self._family(name)
                if family:
                    hit = (f"family:{family}", "family")
            if hit is not None:
                resolved, method = hit
        if resolved is None:
            if tres["method"] == "ambiguous":
                # neither the act nor the target can be pinned down — refuse to cluster
                resolved, method, target = f"unresolved:{name}", "unresolved", ""
            else:
                resolved, method = f"novel:{name}", "novel"
        key = f"{resolved}@{target}" if target else resolved
        record = {
            "key": key, "method": method, "target_resolution": tres,
            "version": self.version,
            "original": {"action_name": raw_name, "target": raw_target,
                         "ontology_anchor": dict(anchor) if anchor else None},
        }
        self.mapping_log.append({
            "version": self.version, "method": method,
            "inputs": {"action_name": raw_name, "target": raw_target,
                       "ontology_anchor": str((anchor or {}).get("name", ""))},
            "outputs": {"key": key, "canonical_target": target,
                        "target_method": tres["method"],
                        "target_resolved": tres["resolved"]},
        })
        return record

    # ---- deterministic replay --------------------------------------------------------
    def export_mappings(self) -> dict:
        """Every cached LLM verdict (accepted and refused), JSON-serializable for replay."""
        return {f"{n}@@{t}": dict(rec) for (n, t), rec in self.llm_cache.items()}

    def load_mappings(self, mappings: dict):
        """Load previously exported verdicts; replayed rows then never touch the LLM."""
        for k, rec in (mappings or {}).items():
            if isinstance(k, tuple):
                n, t = k
            else:
                n, _, t = str(k).partition("@@")
            self.llm_cache[(str(n), str(t))] = dict(rec)

    # ---- name resolution tiers -------------------------------------------------------
    def _equivalent(self, name: str):
        """Curated-map resolution: exact full-name entry first; then a UNIQUE lexical subset
        match (ontology names and equivalence keys as token patterns inside longer phrases,
        most-specific pattern wins; ties are refused). Names carrying a NON_EQUIVALENT token
        never lexically merge — materially different acts stay apart."""
        if name in ACTION_EQUIVALENCE:
            return ACTION_EQUIVALENCE[name], "equivalence"
        tokens = _tokens(name)
        if not tokens or tokens & NON_EQUIVALENT:
            return None
        hits: dict = {}                    # ontology name -> most specific pattern size
        for oname in KNOWN_ACTIONS:
            pat = _tokens(oname)
            if pat and pat <= tokens:
                hits[oname] = max(hits.get(oname, 0), len(pat))
        for key, oname in ACTION_EQUIVALENCE.items():
            pat = _tokens(key)
            if pat and pat <= tokens:
                hits[oname] = max(hits.get(oname, 0), len(pat))
        if not hits:
            return None
        best = max(hits.values())
        winners = sorted(o for o, n in hits.items() if n == best)
        if len(winners) == 1:
            return winners[0], "lexical"
        return None                        # ambiguous — refuse, fall through

    def _llm_equivalent(self, name: str, target: str):
        """Optional LLM-assisted equivalence. The LLM sees ONLY the provided candidates and
        must return a strict JSON verdict; the mapping is REFUSED when materially_different,
        when confidence is low (or missing), or when maps_to is off-menu. Every verdict —
        acceptance or refusal — is cached keyed (name, target) for deterministic replay."""
        cache_key = (name, target)
        rec = self.llm_cache.get(cache_key)
        if rec is None:
            if self.llm is None:
                return None
            candidates = sorted(KNOWN_ACTIONS)
            prompt = _LLM_MAP_PROMPT.format(
                name=name, target=target or "(none)",
                candidates="\n".join(f"- {c}" for c in candidates))
            self.llm_calls += 1
            try:
                raw = self.llm(prompt)
            except Exception:  # noqa: BLE001 — backend failure is not a verdict; not cached
                return None
            r = _parse_json_obj(raw) or {}
            maps_to = str(r.get("maps_to", "NONE") or "NONE")
            confidence = str(r.get("confidence", "low") or "low").lower()
            materially = bool(r.get("materially_different", True))
            accepted = (maps_to in KNOWN_ACTIONS and not materially
                        and confidence in ("high", "medium"))
            rec = {"name": name, "target": target, "version": self.version,
                   "maps_to": maps_to if accepted else "", "raw_maps_to": maps_to,
                   "confidence": confidence, "materially_different": materially,
                   "justification": str(r.get("justification", ""))[:300],
                   "accepted": accepted}
            self.llm_cache[cache_key] = rec
        if rec.get("accepted") and rec.get("maps_to"):
            return str(rec["maps_to"]), "llm"
        return None

    def _family(self, name: str):
        """Strategy-class fallback: the ontology FAMILY of the nearest action name, when at
        least half of some family action name's tokens appear in the free text and the best
        overlap names a single family. NON_EQUIVALENT tokens refuse this tier too."""
        tokens = _tokens(name)
        if not tokens or tokens & NON_EQUIVALENT:
            return None
        scored: dict = {}                  # overlap ratio -> families at that ratio
        for family, names in ACTION_ONTOLOGY.items():
            for oname in names:
                pat = _tokens(oname)
                if not pat:
                    continue
                overlap = len(pat & tokens) / len(pat)
                if overlap >= 0.5:
                    scored.setdefault(overlap, set()).add(family)
        if not scored:
            return None
        families = scored[max(scored)]
        return sorted(families)[0] if len(families) == 1 else None


# ------------------------------------------------------------------- locked-fixture metrics
def _v1_key(row: dict) -> str:
    """The cluster-1.0 key (exact-match baseline), replicated verbatim from
    qualitative_actor.ActionClusterer so the metrics need not import the actor runtime."""
    anchor = (row.get("ontology_anchor") or {}).get("name")
    name = anchor or row.get("action_name", "")
    target = row.get("target", "")
    return f"{name}@{target}" if target else str(name)


def clustering_metrics(fixture_rows, clusterer) -> dict:
    """Evaluate a clusterer against locked fixture cases.

    Each case holds two actions (``a``/``b``, alias ``action_a``/``action_b``), optional
    ``known_entities`` / ``aliases``, and ``expected`` in {"same", "different", "unresolved"}.
    The clusterer's prediction is "unresolved" when either key is unresolved-prefixed, else
    "same"/"different" by key equality; the exact baseline is the v1 (cluster-1.0) key.
    Returns {exact_accuracy, semantic_accuracy, false_merge_rate, false_split_rate,
    unresolved_rate, n} where false_merge = expected different but same v2 key (rate over
    expected-different cases), false_split = expected same but different v2 keys (rate over
    expected-same cases), and unresolved_rate is over all cases."""
    n = exact_ok = sem_ok = unresolved = 0
    merges = splits = n_diff = n_same = 0
    for case in fixture_rows:
        a = case.get("action_a") or case.get("a") or {}
        b = case.get("action_b") or case.get("b") or {}
        expected = str(case.get("expected", ""))
        known = tuple(case.get("known_entities") or ())
        aliases = case.get("aliases") or None
        ka = clusterer.cluster_key(a, known_entities=known, aliases=aliases)
        kb = clusterer.cluster_key(b, known_entities=known, aliases=aliases)
        if ka.startswith("unresolved:") or kb.startswith("unresolved:"):
            predicted = "unresolved"
        else:
            predicted = "same" if ka == kb else "different"
        exact_predicted = "same" if _v1_key(a) == _v1_key(b) else "different"
        n += 1
        exact_ok += exact_predicted == expected
        sem_ok += predicted == expected
        unresolved += predicted == "unresolved"
        if expected == "different":
            n_diff += 1
            merges += ka == kb
        elif expected == "same":
            n_same += 1
            splits += ka != kb
    return {
        "exact_accuracy": round(exact_ok / n, 4) if n else 0.0,
        "semantic_accuracy": round(sem_ok / n, 4) if n else 0.0,
        "false_merge_rate": round(merges / n_diff, 4) if n_diff else 0.0,
        "false_split_rate": round(splits / n_same, 4) if n_same else 0.0,
        "unresolved_rate": round(unresolved / n, 4) if n else 0.0,
        "n": n,
    }
