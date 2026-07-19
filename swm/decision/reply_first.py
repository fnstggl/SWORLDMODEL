"""REPLY-FIRST BEAT PLANNER — the default cold-outreach architecture: design the message around a
concrete human reaction, then search over BEATS, then wording; three separated judges certify.

Why this replaces generate-and-rank as the default: both prior methods (full drafts, line editing)
still searched for "the text an imagined recipient scores highest" — more search just exploited the
imaginary judge (jargon compounds, competing benchmark numbers, a fabricated Princeton line the
judge liked). This planner inverts the direction:

  STEP 1  DESIRED REPLY   — start from the exact positive replies wanted ("Interesting. Send it." /
                            "What's the main technical insight?" / "Talk to X on our team." /
                            "I'd be open to a short conversation.").
  STEP 2  BACKWARD REQS   — for each target reply: what must the recipient have JUST READ for that
                            response to feel like their natural next keystroke? Decomposed into:
                            worthwhile / surprising / believable / sender-noticed / effortless —
                            concrete requirements grounded in the dossier + sender facts.
  STEP 3  BEAT STRUCTURES — a message is functional beats (recognition, identity, surprising_idea,
                            evidence, relevance, request). Several orderings are instantiated as
                            complete drafts, one plain human sentence per beat, with hard writing
                            rules: no jargon compounds (say what the thing does), AT MOST ONE
                            number (translated into reader consequence), a request typed the way a
                            real person types (their words, not ceremony), numbers only from facts.
  STEP 4  BEAT SEARCH     — necessity (drop each beat), merge, and request-swap variants; only
                            gate-passing candidates are compared, blind, by the outcome judge.
  STEP 5  WORDING         — one capped iterative-editor pass inside the winning structure
                            (fabrication guard on).
  STEP 6  THREE JUDGES    — TRUTH (fabricated-vs-facts + numeric guard + contract; unchanged — the
                            judge that caught the false Princeton line): hard gate. LANGUAGE
                            (human-register judge with the user-preference learning hook): hard
                            gate. OUTCOME (the qualitative persona ensemble, BLIND shuffled labels,
                            counts internal): ranks ONLY gate-survivors and can never see who
                            authored what. The strategy-inventing calls and the certifying calls
                            are different roles on different prompts.
  STEP 7  ONE OUTPUT      — a single recommended message. Ties break by language score, then
                            brevity, then deterministic seed order. Human-facing output carries NO
                            simulated reply percentages: the label is "best-supported candidate
                            under the current assumptions; no reliable distinction between
                            finalists" whenever the outcome judge cannot separate them. Full
                            distributions stay in machine-readable artifacts only.

Every LLM call is traced (stage, prompt, response) so a run is fully auditable end to end.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from swm.decision.llm_moves import _call, allowed_numbers, number_violations

BEATS = ("recognition", "identity", "surprising_idea", "evidence", "relevance", "request")

BEAT_ROLE = {
    "recognition": "one short sentence the recipient would nod at — a concern or observation THEY "
                   "already hold (grounded in the dossier; never a fabricated quote)",
    "identity": "who the sender is and what they are building, in plain words a stranger instantly "
                "gets — identity, not credentials",
    "surprising_idea": "the one genuinely non-obvious claim, stated plainly enough to be "
                       "disagreed with",
    "evidence": "the SINGLE strongest number, translated into what it means for the reader, with "
                "its provenance in the same sentence",
    "relevance": "why this recipient specifically — their stated interests, not flattery",
    "request": "the SENDER'S closing ask, in the sender's own words, that makes the desired reply "
               "the recipient's cheapest keystroke — offer, don't command; NEVER paste the "
               "recipient's desired reply itself as the closing line",
}

#: canonical structure library (the search also tests drops/merges/reorders of these)
STRUCTURES = [
    ("identity", "surprising_idea", "evidence", "request"),
    ("surprising_idea", "identity", "evidence", "request"),
    ("recognition", "surprising_idea", "evidence", "identity", "request"),
    ("evidence", "surprising_idea", "identity", "request"),
    ("surprising_idea", "evidence", "relevance", "identity", "request"),
]

DEFAULT_TARGET_REPLIES = [
    {"reply": "Interesting. Send it.", "outcome": "requests_material"},
    {"reply": "What's the main technical insight?", "outcome": "curious_reply"},
    {"reply": "Talk to the relevant person on our team.", "outcome": "refers_to_other"},
    {"reply": "I'd be open to a short conversation.", "outcome": "meeting_offer"},
]

_NO_PERCENT_LABEL = ("best-supported candidate under the current assumptions; no reliable "
                     "distinction between finalists (uncalibrated evaluator — real outreach "
                     "outcomes, not more simulation, would sharpen this)")


@dataclass
class PlannerResult:
    winner_text: str
    winner_origin: str
    finalists: list                     # [{label, text, gates, ordinal_note}] — no percentages
    label: str
    trace_path: str = ""
    n_llm_calls: int = 0

    def summary(self) -> dict:
        return {"report_type": "reply_first_single_output",
                "recommended_message": self.winner_text,
                "origin": self.winner_origin,
                "finalists_considered": [{"label": f["label"],
                                          "gates": f.get("gates"),
                                          "note": f.get("ordinal_note", "")}
                                         for f in self.finalists],
                "label": self.label, "n_llm_calls": self.n_llm_calls}


class ReplyFirstPlanner:
    def __init__(self, chat_fn=None, *, sender_brief, dossier, hypotheses=None,
                 recipient_notes: str = "", seed: int = 0, trace_path: str = None,
                 max_llm_calls: int = 150, persona_draws: int = 3,
                 use_content_graph: bool = True):
        self.chat = chat_fn
        self.brief = sender_brief
        self.dossier = dossier
        self.hypotheses = hypotheses or []
        self.notes = recipient_notes
        self.seed = seed
        self.trace_path = trace_path
        self.max_llm_calls = max_llm_calls
        self.persona_draws = persona_draws
        # semantic-planning stage (content_graph.py): search over ideas/information and turn the
        # winning plan into language, as ADDITIONAL seed candidates + a final deletion gauntlet, all
        # flowing through the same three judges. Contributes nothing offline (a plan needs a writer
        # to become language) or when off, so the offline pipeline stays byte-identical either way.
        self.use_content_graph = use_content_graph
        self.calls = 0
        self.allowed = allowed_numbers(sender_brief.to_prompt() if sender_brief else "",
                                       recipient_notes)
        self._truth_judge = None
        from swm.decision.language_judge import llm_language_judge
        self.language = llm_language_judge(chat_fn)

    # ---------------------------------------------------------------- traced LLM call
    def _llm(self, stage: str, prompt: str, *, max_tokens: int = 480,
             temperature: float = 0.4) -> str:
        if self.chat is None or self.calls >= self.max_llm_calls:
            return ""
        self.calls += 1
        try:
            out = _call(self.chat, prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception:  # noqa: BLE001
            out = ""
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps({"call": self.calls, "stage": stage, "prompt": prompt,
                                    "response": out}) + "\n")
        return out

    def _jobj(self, raw):
        m = re.search(r"\{.*\}", raw or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except ValueError:
            return None

    # ---------------------------------------------------------------- STEP 1-2: backward planning
    def desired_replies(self) -> list:
        raw = self._llm("step1_desired_replies", (
            f"{self.dossier.render()}\n\nThe sender's goal: {self.brief.ask}.\n"
            "List the 4 most valuable POSITIVE replies this recipient could realistically type to "
            "a cold email serving that goal — verbatim, in the recipient's own voice (short, like "
            "real typed replies). Map each to one outcome category from: requests_material, "
            "curious_reply, refers_to_other, meeting_offer.\n"
            'Return ONLY a JSON object {"replies": [{"reply": "...", "outcome": "..."}]}.'),
            max_tokens=300, temperature=0.3)
        obj = self._jobj(raw) or {}
        rows = [r for r in obj.get("replies", []) if isinstance(r, dict) and r.get("reply")]
        return rows[:4] if len(rows) >= 2 else list(DEFAULT_TARGET_REPLIES)

    def backward_requirements(self, replies: list) -> dict:
        raw = self._llm("step2_backward_requirements", (
            f"{self.dossier.render()}\n\nSender facts (ground truth; nothing beyond these):\n"
            f"{self.brief.to_prompt()}\n\n"
            "Target replies we want to make feel like the recipient's NATURAL next keystroke:\n" +
            "\n".join(f"- \"{r['reply']}\"" for r in replies) + "\n\n"
            "Work BACKWARD: immediately before typing such a reply, what must this person have "
            "just read and believed? Give concrete, plain-language requirements grounded in the "
            "dossier and ONLY the sender facts:\n"
            "worthwhile: what makes replying worth his seconds; surprising: what makes the thesis "
            "genuinely non-obvious TO HIM; believable: what makes the claim credible on first read "
            "(which single number, framed how); noticed: what makes this sender worth registering "
            "as a real person; effortless: what exact reply format costs him the least.\n"
            'Return ONLY JSON: {"worthwhile": "...", "surprising": "...", "believable": "...", '
            '"noticed": "...", "effortless": "..."}'), max_tokens=420, temperature=0.3)
        obj = self._jobj(raw)
        if not obj:
            obj = {"worthwhile": "a genuinely non-obvious claim about his own field",
                   "surprising": "the bottleneck is planning, not power",
                   "believable": "one number with its provenance in the same sentence",
                   "noticed": "a real person plainly identified in the first line",
                   "effortless": "a reply he can type in five words"}
        return obj

    # ---------------------------------------------------------------- STEP 3: instantiate structures
    _WRITING_RULES = (
        "Writing rules (hard): every sentence is something a sharp busy person would actually type. "
        "NO jargon compounds — never name a product category ('constraint-aware orchestration'); "
        "say what the thing DOES in plain words. AT MOST ONE number in the whole email — pick the "
        "single strongest, translate it into what it means for the reader, and keep its provenance "
        "in the same sentence. Numbers only from the sender facts. The email is written BY the "
        "sender TO the recipient: the closing line is the sender's own ask (e.g. offering the "
        "one-pager), never the recipient's hoped-for reply pasted verbatim. The request must read "
        "like a human typed it to a peer (their own words, no ceremonious permission "
        "constructions). No em dashes unless truly necessary. 45-85 words total.")

    def instantiate(self, structure: tuple, reqs: dict) -> str:
        beats_desc = "\n".join(f"{i + 1}. {b}: {BEAT_ROLE[b]}" for i, b in enumerate(structure))
        raw = self._llm("step3_instantiate", (
            f"{self.dossier.render()}\n\nSender facts (ground truth):\n{self.brief.to_prompt()}\n\n"
            "Backward requirements the email must satisfy (derived from the replies we want):\n"
            f"{json.dumps(reqs, indent=1)}\n\n"
            f"Write ONE complete cold email as exactly this beat sequence:\n{beats_desc}\n"
            f"{self._WRITING_RULES}\n"
            "Return ONLY the email text (greeting through sign-off)."),
            max_tokens=260, temperature=0.6)
        from swm.decision.iterative_editor import _strip_subject
        return _strip_subject((raw or "").strip().strip('"'))

    # ---------------------------------------------------------------- STEP 6: the three judges
    def truth(self, text: str) -> dict:
        """TRUTH JUDGE (unchanged in spirit — the judge that caught the fabricated Princeton line):
        numeric fact guard + contract + fabricated-vs-facts sentence judge. Hard gate."""
        bad = number_violations(text, self.allowed)
        if bad:
            return {"ok": False, "violations": [f"number not in facts: {bad}"]}
        from swm.decision.outreach_contract import validate
        # identity_window=None: identity must exist somewhere (debate-bait still fails), but its
        # POSITION is a structure choice for the blind outcome ranking, not a contract rule —
        # run-2 forensic: the v3 first-two-sentences rule killed 4 of 5 structures at the gate.
        cv = validate(text, self.brief, identity_window=None)
        if not cv.ok:
            return {"ok": False, "violations": [f"contract: {cv.missing}"]}
        if self.chat is None:
            return {"ok": True, "violations": [], "source": "numeric+contract only (offline)"}
        try:
            if self._truth_judge is None:
                from swm.decision.llm_moves import llm_sentence_judge
                self._truth_judge = llm_sentence_judge(self.chat,
                                                       facts_text=self.brief.to_prompt())
            self.calls += 1
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
            verdicts = self._truth_judge(sents)
            fab = [{"sentence": s, "reason": v.get("reason", "")}
                   for s, v in zip(sents, verdicts) if v.get("fabricated")]
            return {"ok": not fab, "violations": fab}
        except Exception:  # noqa: BLE001 — unverifiable text does not pass the truth gate
            return {"ok": False, "violations": ["truth judge unavailable; failing closed"]}

    def outcome_rank(self, finalists: list) -> dict:
        """OUTCOME JUDGE — blind: shuffled anonymous labels, persona-ensemble counts kept internal.
        Returns ordinal info only: {'order': [labels], 'separable': bool}. Never sees which method
        authored a candidate; never reports a percentage upward."""
        if self.chat is None or not self.hypotheses:
            return {"order": [f["label"] for f in finalists], "separable": False,
                    "note": "offline: no outcome judgment; original order retained"}
        from swm.decision.persona_response import (DEFAULT_OUTCOME_UTILITIES, ensemble_evaluate)
        rng = random.Random(self.seed)
        blind = list(finalists)
        rng.shuffle(blind)
        results = {}
        for i, f in enumerate(blind):
            ens = ensemble_evaluate(self.chat, self.dossier, self.hypotheses, f["text"],
                                    draws_per_hypothesis=self.persona_draws)
            self.calls += ens.n_draws
            results[f["label"]] = ens
        eu = {k: v.expected_utility(DEFAULT_OUTCOME_UTILITIES) for k, v in results.items()}
        order = sorted(eu, key=eu.get, reverse=True)
        n = next(iter(results.values())).n_draws or 1
        noise = 1.5 / (n ** 0.5)
        separable = (len(order) >= 2 and eu[order[0]] - eu[order[1]] > noise)
        # full distributions go to the machine-readable trace ONLY (no percentages upward)
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps({"stage": "step6_outcome_internal",
                                    "counts": {k: v.counts for k, v in results.items()},
                                    "note": "internal persona counts; never shown as probabilities"}
                                   ) + "\n")
        return {"order": order, "separable": separable}

    # ---------------------------------------------------------------- STEP 4: beat-level search
    def beat_variants(self, structure: tuple, text: str) -> list:
        """Necessity (drop each beat), request-swap — as complete drafts. Request swaps come
        FIRST after the base so a capped ranking pool can never silently exclude them (run-1
        forensic: a [:4] slice dropped both request swaps while the seed carried an inverted ask)."""
        base = [{"label": f"S{'-'.join(b[:2] for b in structure)}", "text": text,
                 "origin": "structure"}]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        # a bare trailing signature ("Beckett", "— Beckett", "Best, Beckett") is not a sentence:
        # hold it out of the surgery and re-append, or a request swap replaces the NAME instead of
        # the ask and the variant ships unsigned (run-4 forensic)
        signoff = None
        if sents and len(sents[-1].split()) <= 3 and not sents[-1].endswith((".", "!", "?")):
            signoff = sents.pop()
        tail = [signoff] if signoff else []
        # request swap: two alternative reply-shapes, written by the LLM under the same rules
        raw = self._llm("step4_request_swap", (
            f"Sender facts:\n{self.brief.to_prompt()}\n--- EMAIL ---\n{text}\n--- END ---\n"
            "Write TWO alternative CLOSING lines for this email, each a materially different "
            "reply-shape (e.g. asking one pointed question he can answer from expertise vs. "
            "offering the one-pager in the sender's own words). Each is the SENDER's own typed "
            "words to the recipient — never the reply the sender hopes to receive. Each must read "
            "like a busy human typed it. Return ONLY JSON: {\"a\": \"...\", \"b\": \"...\"}"),
            max_tokens=160, temperature=0.6)
        obj = self._jobj(raw) or {}
        swaps = []
        for key in ("a", "b"):
            alt = str(obj.get(key, "")).strip()
            if alt and sents:
                t = " ".join(sents[:-1] + [alt] + tail)
                swaps.append({"label": f"request_{key}", "text": t, "origin": "request_swap"})
        # drop each non-request beat (necessity test) — deterministic surgery on sentences when the
        # count lines up; otherwise skipped (the wording pass handles ragged cases)
        drops = []
        if len(sents) >= len(structure):
            for i, b in enumerate(structure):
                if b == "request":
                    continue
                t = " ".join(sents[:i] + sents[i + 1:] + tail)
                drops.append({"label": f"drop_{b}", "text": t, "origin": "necessity"})
        return base + swaps + drops

    # ---------------------------------------------------------------- gate pool + flag repair
    def _gate_pool(self, cands: list) -> list:
        """Run truth + language on each candidate; return the pool the outcome judge may rank.
        STRICTLY clean candidates (truth ok AND language ok with zero flags) always outrank
        flagged-but-high-score ones: near-misses (score >= 0.55) are used ONLY when nothing is
        strictly clean. Run-1 forensic: the near-miss rule admitted a flagged winner while the
        judge's flags went unused."""
        strict, near = [], []
        for c in cands:
            tr = self.truth(c["text"])
            lv = self.language(c["text"])
            c["gates"] = {"truth": tr["ok"], "language": lv.ok,
                          "language_score": lv.score,
                          "problems": tr.get("violations", []) + lv.flags}
            c["_lang_flags"] = lv.flags
            if tr["ok"] and lv.ok:
                strict.append(c)
            elif tr["ok"] and lv.score >= 0.55:
                near.append(c)
        return strict if strict else near

    def repair_language(self, cand: dict) -> dict:
        """Targeted revision that resolves the language judge's flags and nothing else, then
        re-gates. Up to two attempts; each attempt must MUST-fix every flagged item. A repair is
        kept if it comes back strictly clean, or if it strictly reduces the flag count without a
        truth failure or a language-score drop (monotone improvement — run-3 forensic: all-or-
        nothing acceptance rejected every partial fix, so the winner shipped with flags the judge
        had raised a dozen times). Never returns anything worse than the input."""
        if self.chat is None or not (cand.get("_lang_flags") or []):
            return cand
        from swm.decision.iterative_editor import _strip_subject
        best = cand
        for _ in range(2):
            flags = best.get("_lang_flags") or []
            if not flags:
                break
            raw = self._llm("step6_language_repair", (
                f"Sender facts (ground truth; nothing beyond these):\n{self.brief.to_prompt()}\n"
                f"--- EMAIL ---\n{best['text']}\n--- END ---\n"
                "A language judge flagged these problems. You MUST eliminate every one — replace "
                "each jargon compound with plain words for what the thing does, keep only the "
                "single strongest number and cut the others, and rewrite any pitch-deck or "
                "ceremonious phrasing the way a busy person would type it:\n" +
                "\n".join(f"- MUST FIX: {f.get('problem', '')} (in: "
                          f"\"{str(f.get('sentence', ''))[:90]}\")" for f in flags) +
                f"\nChange ONLY what resolves the flags. Keep every fact, the same beats in the "
                f"same order, and the same ask. {self._WRITING_RULES}\n"
                "Return ONLY the email text."), max_tokens=260, temperature=0.3)
            fixed = _strip_subject((raw or "").strip().strip('"'))
            if not fixed or fixed == best["text"]:
                break
            tr = self.truth(fixed)
            if not tr["ok"]:
                continue
            lv = self.language(fixed)
            if lv.ok:
                return {**best, "text": fixed, "label": cand["label"] + "+lang_repair",
                        "gates": {"truth": True, "language": True,
                                  "language_score": lv.score, "problems": []},
                        "_lang_flags": []}
            if (len(lv.flags) < len(flags)
                    and lv.score >= best["gates"]["language_score"] - 0.05
                    and lv.score >= 0.55):
                best = {**best, "text": fixed, "label": cand["label"] + "+lang_repair",
                        "gates": {"truth": True, "language": False,
                                  "language_score": lv.score, "problems": lv.flags},
                        "_lang_flags": lv.flags}
        return best

    # ---------------------------------------------------------------- STEP 5: wording pass
    def wording_pass(self, text: str) -> str:
        if self.chat is None:
            return text
        from swm.decision.iterative_editor import IterativeEditor
        ed = IterativeEditor(self.chat, sender_brief=self.brief, recipient_notes=self.notes,
                             dossier_text=self.dossier.render(), recipient_vars={},
                             base_mean=0.2, max_passes=1, beam_size=1,
                             max_llm_calls=min(30, self.max_llm_calls - self.calls),
                             trace_path=self.trace_path)
        out = ed.run([{"label": "planner_winner", "text": text}])
        self.calls += ed.calls
        beam = out.get("beam") or []
        return beam[0].text if beam else text

    # ---------------------------------------------------------------- semantic-planning stage
    def _cg_chat(self):
        """A budget- and trace-aware chat wrapper handed to the content-graph stage: its calls count
        against max_llm_calls (fail-closed to '' once exhausted). None stays None (offline)."""
        if self.chat is None:
            return None

        def wrapped(prompt, **kw):
            if self.calls >= self.max_llm_calls:
                return ""
            self.calls += 1
            try:
                return _call(self.chat, prompt, **kw)
            except Exception:  # noqa: BLE001 — a dead backend degrades the stage, never crashes it
                return ""
        return wrapped

    def _content_graph_seeds(self, replies: list, *, max_seeds: int = 4) -> list:
        """Build the graph -> plan semantics -> verbalize, returning CG seed candidates labeled
        'CG:<plan_id>.<j>'. Bounded: the wrapper caps LLM calls at max_llm_calls and the seed count
        is capped; returns [] on any failure so the beat-structure seeds always still run."""
        from swm.decision.content_graph import (build_content_graph, plan_semantics, verbalize)
        cg_chat = self._cg_chat()
        graph = build_content_graph(cg_chat, self.brief, self.dossier, trace_path=self.trace_path)
        if not graph.units:
            return []
        plans = plan_semantics(cg_chat, graph.units, replies, k=4, trace_path=self.trace_path)
        seeds = []
        for plan in plans:
            if len(seeds) >= max_seeds or self.calls >= self.max_llm_calls:
                break
            for j, txt in enumerate(verbalize(cg_chat, plan, graph.units, self.brief, n=2,
                                              trace_path=self.trace_path)):
                seeds.append({"label": f"CG:{plan.plan_id}.{j}", "text": txt,
                              "origin": "content_graph", "plan_id": plan.plan_id})
        return seeds[:max_seeds]

    def _adversarial_variants(self, text: str, *, max_variants: int = 3) -> list:
        """Adversarial-deletion candidates for the finalist: the strongest-reason repair plus a few
        surviving single-sentence deletions, labeled 'AD:...'. This stage never picks a winner —
        deterministic admissibility only (numbers + contract); the SAME truth + language gates and
        the blind outcome judge re-judge these in the final gauntlet."""
        from swm.decision.content_graph import adversarial_deletion
        ad = adversarial_deletion(self._cg_chat(), text, self.brief, judge=None,
                                  trace_path=self.trace_path)
        cands = []
        if ad.get("repaired_ok"):
            cands.append({"label": "AD:repair", "text": ad["repaired"], "origin": "adversarial"})
        for i, d in enumerate(ad.get("deletions", [])):
            if d.get("truth_ok"):
                cands.append({"label": f"AD:del_{i}", "text": d["text"], "origin": "adversarial"})
        return cands[:max_variants]

    # ---------------------------------------------------------------- the full pipeline
    def run(self) -> PlannerResult:
        replies = self.desired_replies()
        reqs = self.backward_requirements(replies)

        # instantiate structures -> gate -> collect candidates
        candidates = []
        for st in STRUCTURES:
            t = self.instantiate(st, reqs)
            if not t:
                continue
            candidates.append({"label": f"S:{'>'.join(b[:4] for b in st)}", "text": t,
                               "origin": "structure", "structure": st})

        # SEMANTIC-PLANNING STAGE: search over ideas/information (content units -> best subset and
        # ordering -> several verbalizations) BEFORE the beat structures, and add the results as
        # additional seed candidates. Skipped offline (no writer to verbalize a plan) and when the
        # flag is off, so the offline pipeline stays byte-identical to the pre-content-graph default.
        cg_seeds = []
        if self.use_content_graph and self.chat is not None:
            cg_seeds = self._content_graph_seeds(replies)
        candidates = cg_seeds + candidates

        from swm.decision.outreach_contract import plain_baseline_draft
        candidates.append({"label": "plain_baseline", "origin": "baseline",
                           "text": plain_baseline_draft(self.brief,
                                                        getattr(self.dossier, "name", ""))})

        gated = self._gate_pool(candidates)
        if not gated:
            gated = [candidates[-1]]                       # plain baseline always survives truth

        # blind outcome ranking of the seed candidates -> beat search on the top seed. Widen the
        # ranking pool only when content-graph seeds are present, so structures still get ranked.
        pool_cap = 8 if cg_seeds else 6
        rank1 = self.outcome_rank([{"label": c["label"], "text": c["text"]}
                                   for c in gated[:pool_cap]])
        top_label = rank1["order"][0]
        top = next(c for c in gated if c["label"] == top_label)
        variants = self.beat_variants(top.get("structure", STRUCTURES[0]), top["text"])
        gated_variants = self._gate_pool(variants)
        if not gated_variants:
            gated_variants = [variants[0]]
        rank2 = self.outcome_rank([{"label": v["label"], "text": v["text"]}
                                   for v in gated_variants[:6]])
        best_v = next(v for v in gated_variants if v["label"] == rank2["order"][0])

        # wording pass inside the winning structure, then the final gauntlet: gate, repair any
        # flagged finalist once (flags become edits), and only then rank blind
        polished = self.wording_pass(best_v["text"])
        finalists = [{"label": "polished", "text": polished},
                     {"label": "pre_polish", "text": best_v["text"]}]

        # ADVERSARIAL DELETION on the finalist: a strongest-reason repair + per-sentence deletion
        # probes; the surviving variants join the SAME final gauntlet (truth + language + blind
        # outcome). Skipped offline / when the flag is off -> byte-identical offline behavior.
        if self.use_content_graph and self.chat is not None:
            finalists += self._adversarial_variants(polished)

        final_gated = self._gate_pool(finalists)
        if not final_gated:
            final_gated = [finalists[1]]
        final_gated = [self.repair_language(f) for f in final_gated]
        finalists = final_gated + [f for f in finalists
                                   if not any(g["label"].startswith(f["label"])
                                              for g in final_gated)]
        rank3 = self.outcome_rank(final_gated)

        # STEP 7: ONE output. Tie-break: strictly-clean first, then language score, then brevity.
        if rank3.get("separable"):
            win_label = rank3["order"][0]
            note = "outcome judge separated the finalists"
        else:
            final_gated.sort(key=lambda f: (0 if f["gates"]["language"] else 1,
                                            -f["gates"]["language_score"],
                                            len(f["text"].split())))
            win_label = final_gated[0]["label"]
            note = _NO_PERCENT_LABEL
        winner = next(f for f in final_gated if f["label"] == win_label)
        for f in finalists:
            f["ordinal_note"] = ("selected" if f["label"] == win_label else
                                 "finalist; not reliably distinguishable" if not rank3.get("separable")
                                 else "ranked below the winner by the outcome judge")
        return PlannerResult(winner_text=winner["text"],
                             winner_origin=f"{top['label']} -> {best_v['label']} -> {win_label}",
                             finalists=finalists, label=note,
                             trace_path=self.trace_path or "", n_llm_calls=self.calls)
