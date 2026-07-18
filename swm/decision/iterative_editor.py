"""ITERATIVE EDITOR — an exacting human editor's loop, mechanized (the second, experimental search
path for outreach wording; it does not replace the full-draft generator, it competes with it).

The full-draft path samples complete emails and picks the best of "pretty good". A human editor
works differently: start from the strongest draft, then interrogate it one location at a time —
this line sounds phony, can this be said plainer, is this line needed at all, is a beat missing —
while constantly re-reading the WHOLE message, because a locally better sentence can make the email
worse as one social interaction. This module imitates that loop:

  1. SEED       — several strategy-diverse complete drafts (incl. the plain-human baseline); the
                  strongest becomes the working state (others stay in the beam).
  2. DIAGNOSE   — a whole-message diagnostic: artificial/annoying/boastful/vague/dense/unnecessary
                  lines, missing beats, weak ordering, length, unsupported claims, ask difficulty,
                  and whether it works as ONE coherent social interaction.
  3. MUTATE     — one location at a time: materially different alternatives (keep / rewrite plainer /
                  shorten / reframe / merge with neighbor / DELETE / insert a missing line), never
                  cosmetic paraphrase. Every alternative is fact-guarded (numbers must come from the
                  sender facts) and rendered as a FULL candidate email.
  4. JUDGE      — an independent comparative judge reads the complete variants in context and picks
                  the one most likely to earn a POSITIVE downstream response from this recipient —
                  full-message judgment, never isolated-sentence judgment.
  5. GUARD      — after every accepted mutation the complete email is rescored on eight axes
                  (comprehension, credibility, naturalness, relevance, cognitive effort,
                  positive-response value, annoyance risk, coherence); a locally-improved line that
                  worsens the whole email is REJECTED and recorded.
  6. ENDGAME    — when a pass stops improving: explicit tests for deleting each remaining line,
                  reordering beats, adding a missing credibility/relevance/permission beat,
                  shortening the whole message, replacing the ask, changing the opening, and
                  changing the strategic frame.
  7. ESCAPE     — a small beam of surviving states plus periodic structural moves (two-line
                  mutations via the endgame, an informed complete rewrite from the accumulated
                  critiques, crossover between the strongest parts of different drafts) so the
                  search is not trapped in one locally optimal draft.

HONESTY: the internal 8-axis score is the editor's compass, not the arbiter — final candidates are
compared under the same persona-ensemble/funnel/gate evaluator as every other approach, and the
output is 'best-supported among tested', never a theoretical optimum. The full edit trace (every
alternative, selection, reason, before/after scores, rejections) is machine-readable JSONL.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from swm.decision.llm_moves import _call, allowed_numbers, number_violations

AXES = ("cold_reader_comprehension", "credibility", "naturalness", "recipient_relevance",
        "cognitive_effort", "positive_response_value", "negative_response_risk", "coherence")

#: composite = mean(positive axes) − 0.5·cognitive_effort − 0.7·negative_risk (documented weights;
#: the internal compass only — NEVER reported as a response probability)
_POSITIVE = ("cold_reader_comprehension", "credibility", "naturalness", "recipient_relevance",
             "positive_response_value", "coherence")


def composite(scores: dict) -> float:
    pos = sum(float(scores.get(a, 0.0)) for a in _POSITIVE) / len(_POSITIVE)
    return round(pos - 0.5 * float(scores.get("cognitive_effort", 0.0))
                 - 0.7 * float(scores.get("negative_response_risk", 0.0)), 4)


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def _join(sents: list) -> str:
    return " ".join(s for s in sents if s and s.strip())


def _json_obj(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


@dataclass
class EditState:
    label: str
    text: str
    scores: dict = field(default_factory=dict)

    @property
    def value(self) -> float:
        return composite(self.scores) if self.scores else -9.9


class IterativeEditor:
    """chat_fn=None runs a deterministic offline mode (funnel/critic proxy scores, rule-based
    alternatives) so the mechanics are testable without an LLM; live mode uses the LLM for
    diagnosis, alternatives, the comparative judge, and the 8-axis rescore."""

    def __init__(self, chat_fn=None, *, sender_brief=None, recipient_notes: str = "",
                 dossier_text: str = "", recipient_vars: dict = None, base_mean: float = 0.2,
                 seed: int = 0, max_passes: int = 2, beam_size: int = 3,
                 max_llm_calls: int = 110, trace_path: str = None):
        self.chat = chat_fn
        self.brief = sender_brief
        self.notes = recipient_notes
        self.dossier_text = dossier_text
        self.rvars = dict(recipient_vars or {})
        self.base_mean = base_mean
        self.seed = seed
        self.max_passes = max_passes
        self.beam_size = beam_size
        self.max_llm_calls = max_llm_calls
        self.calls = 0
        self.trace: list = []
        self.trace_path = trace_path
        self.critiques: list = []                       # accumulated diagnosis text (informs rewrite)
        self.allowed = allowed_numbers(sender_brief.to_prompt() if sender_brief else "",
                                       recipient_notes)

    # ---------------------------------------------------------------- plumbing
    def _llm(self, prompt: str, *, max_tokens: int = 500, temperature: float = 0.4) -> str:
        if self.chat is None or self.calls >= self.max_llm_calls:
            return ""
        self.calls += 1
        try:
            return _call(self.chat, prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception:  # noqa: BLE001 — a failed call degrades to "no change", never crashes
            return ""

    def _record(self, row: dict):
        self.trace.append(row)
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def _guard(self, text: str) -> bool:
        """Deterministic admissibility: numbers from the facts only + the content contract holds."""
        if number_violations(text, self.allowed):
            return False
        from swm.decision.outreach_contract import validate
        return validate(text, self.brief).ok

    # ---------------------------------------------------------------- scoring (the compass)
    def score(self, text: str) -> dict:
        raw = self._llm(
            f"You are scoring a cold email to this recipient.\n{self.dossier_text}\n"
            f"Recipient notes: {self.notes}\n--- EMAIL ---\n{text}\n--- END ---\n"
            "Score each axis 0.00-1.00, judging the COMPLETE email as one social interaction from "
            "a busy stranger's chair:\n"
            "cold_reader_comprehension (a stranger instantly gets who/what/why), credibility "
            "(claims believable, nothing boastful/unsupported), naturalness (a real busy human "
            "wrote it), recipient_relevance (why THIS recipient), cognitive_effort (how much work "
            "replying takes: 0=trivial, 1=heavy), positive_response_value (how likely a POSITIVE "
            "reply is), negative_response_risk (annoyance/dismissal risk), coherence (one clean "
            "interaction: identity->idea->evidence->ask).\n"
            'Return ONLY JSON like {"cold_reader_comprehension":0.8,...} with all eight keys.',
            max_tokens=260, temperature=0.0)
        obj = _json_obj(raw)
        if obj and all(a in obj for a in AXES):
            return {a: max(0.0, min(1.0, float(obj[a]))) for a in AXES}
        return self._proxy_score(text)

    def _proxy_score(self, text: str) -> dict:
        """Deterministic offline proxy from the funnel + critic + contract (keeps tests LLM-free)."""
        from swm.decision.compositional_search import encode_text_to_strategy
        from swm.decision.outreach_contract import validate
        from swm.decision.response_funnel import FunnelScorer
        from swm.decision.semantic_critic import SemanticCritic
        strat = encode_text_to_strategy(text)
        sc = FunnelScorer(recipient=self.rvars, base_responsiveness=self.base_mean,
                          n_weight_samples=24, seed=self.seed)
        d = sc.score_dist(strat)
        crit = SemanticCritic(allowed_numbers=self.allowed).critique(text)
        cv = validate(text, self.brief)
        n_words = len(text.split())
        return {"cold_reader_comprehension": d.stage_trace.get("understand", 0.5),
                "credibility": d.stage_trace.get("believe", 0.5),
                "naturalness": crit.naturalness,
                "recipient_relevance": d.stage_trace.get("relevant", 0.5),
                "cognitive_effort": strat.get("cognitive_effort", 0.3),
                "positive_response_value": d.mean,
                "negative_response_risk": d.mean_neg,
                "coherence": (crit.coherence if cv.ok else 0.3) * (1.0 if n_words <= 130 else 0.7)}

    # ---------------------------------------------------------------- diagnosis
    def diagnose(self, text: str) -> dict:
        raw = self._llm(
            f"You are an exacting human editor reviewing a cold email to this recipient.\n"
            f"{self.dossier_text}\n--- EMAIL ---\n{text}\n--- END ---\n"
            "Diagnose the WHOLE message: which lines sound artificial, annoying, boastful, vague, "
            "confusing, overly dense, or unnecessary; what beat is missing (identity? credibility? "
            "relevance? permission-style ask?); is the ordering weak; is it too long; which claims "
            "read unsupported; is the ask too hard or unclear; where does it fail as ONE coherent "
            "social interaction?\n"
            'Return ONLY JSON: {"line_issues":[{"line": "<verbatim line>", "issues": ["..."]}], '
            '"missing_beats": ["..."], "ordering_weak": bool, "too_long": bool, '
            '"unsupported_claims": ["..."], "ask_problem": "or empty", "coherence_note": "..."}',
            max_tokens=520, temperature=0.2)
        obj = _json_obj(raw)
        if obj is None:                                   # offline: contract + critic diagnosis
            from swm.decision.outreach_contract import validate
            from swm.decision.semantic_critic import SemanticCritic
            crit = SemanticCritic(allowed_numbers=self.allowed).critique(text)
            cv = validate(text, self.brief)
            obj = {"line_issues": [{"line": f["sentence"], "issues": [f["issue"]]}
                                   for f in crit.flags()],
                   "missing_beats": cv.missing, "ordering_weak": False,
                   "too_long": len(text.split()) > 130,
                   "unsupported_claims": [f for f in cv.flags if "unanchored" in f],
                   "ask_problem": next((f for f in cv.flags if "next_step" in f or "diligence" in f), ""),
                   "coherence_note": ""}
        self.critiques.append(obj)
        return obj

    # ---------------------------------------------------------------- per-location alternatives
    def line_alternatives(self, sents: list, idx: int, issues: list) -> list:
        """Materially different alternatives for location idx, each returned as a FULL email variant:
        keep / rewrite / shorten / reframe / merge-with-next / delete / insert-after."""
        line = sents[idx]
        variants = [{"label": "KEEP", "kind": "keep", "text": _join(sents)}]
        raw = self._llm(
            f"You are editing ONE location of a cold email to this recipient.\n{self.dossier_text}\n"
            f"Sender facts (numbers may ONLY come from these):\n"
            f"{self.brief.to_prompt() if self.brief else ''}\n"
            f"--- FULL EMAIL ---\n{_join(sents)}\n--- END ---\n"
            f"TARGET LINE: \"{line}\"\n"
            f"Editor's concerns about it: {'; '.join(map(str, issues)) or 'none noted'}\n"
            "Write MATERIALLY DIFFERENT alternatives (no cosmetic paraphrase):\n"
            "rewrite: the line said plainly, as a busy human would;\n"
            "shorten: the line at half the words, keeping what matters;\n"
            "reframe: the line serving a DIFFERENT purpose/framing that helps the whole email;\n"
            "merge: ONE line combining this line with the following line (write the merged line);\n"
            "insert_after: a NEW line adding a missing beat after this one (or empty if none needed).\n"
            'Return ONLY JSON: {"rewrite": "...", "shorten": "...", "reframe": "...", '
            '"merge": "...", "insert_after": "..."}',
            max_tokens=420, temperature=0.6)
        obj = _json_obj(raw) or {}

        def add(kind, new_sents, note=""):
            t = _join(new_sents)
            if t and t != variants[0]["text"] and self._guard(t):
                variants.append({"label": kind.upper(), "kind": kind, "text": t, "note": note})

        for kind in ("rewrite", "shorten", "reframe"):
            alt = str(obj.get(kind, "")).strip()
            if alt:
                add(kind, sents[:idx] + [alt] + sents[idx + 1:])
        if str(obj.get("merge", "")).strip() and idx + 1 < len(sents):
            add("merge", sents[:idx] + [str(obj["merge"]).strip()] + sents[idx + 2:])
        ins = str(obj.get("insert_after", "")).strip()
        if ins:
            add("insert_after", sents[:idx + 1] + [ins] + sents[idx + 1:])
        # deletion is ALWAYS on the table (deterministic; a human editor's favorite move)
        add("delete", sents[:idx] + sents[idx + 1:])
        return variants

    # ---------------------------------------------------------------- the comparative judge
    def judge(self, variants: list, purpose: str) -> tuple:
        """Independent judge over FULL email variants in context; returns (label, why). Offline:
        the proxy composite decides (deterministic)."""
        if self.chat is None or self.calls >= self.max_llm_calls:
            best = max(variants, key=lambda v: composite(self._proxy_score(v["text"])))
            return best["label"], "offline proxy composite"
        listing = "\n\n".join(f"[{v['label']}]\n{v['text']}" for v in variants)
        raw = self._llm(
            f"You are an independent judge of cold outreach.\n{self.dossier_text}\n"
            f"Recipient notes: {self.notes}\n"
            f"Decision purpose: {purpose}\n"
            f"Below are COMPLETE variants of the same email. Judge each as a whole social "
            f"interaction (never the changed sentence in isolation) and pick the variant MOST "
            f"likely to earn a POSITIVE downstream response from this specific recipient (a reply "
            f"engaging with the substance, a request for the memo, a referral) while risking the "
            f"least annoyance.\n\n{listing}\n\n"
            'Return ONLY JSON: {"choice": "<label>", "why": "one short sentence"}',
            max_tokens=200, temperature=0.0)
        obj = _json_obj(raw) or {}
        labels = {v["label"] for v in variants}
        choice = str(obj.get("choice", "")).strip().upper()
        if choice not in labels:
            return "KEEP", "judge unparseable; keeping current line (fail-closed)"
        return choice, str(obj.get("why", ""))[:200]

    # ---------------------------------------------------------------- one improvement pass
    def improve_pass(self, state: EditState, phase: str) -> bool:
        changed = False
        diag = self.diagnose(state.text)
        issue_by_line = {}
        for li in diag.get("line_issues", []):
            issue_by_line[str(li.get("line", ""))[:60]] = li.get("issues", [])
        idx = 0
        while idx < len(_sentences(state.text)):
            sents = _sentences(state.text)
            line = sents[idx]
            issues = next((v for k, v in issue_by_line.items() if k and k in line), [])
            variants = self.line_alternatives(sents, idx, issues)
            if len(variants) <= 1:
                idx += 1
                continue
            choice, why = self.judge(variants, f"improve the line: \"{line[:70]}\"")
            picked = next(v for v in variants if v["label"] == choice)
            row = {"phase": phase, "location": f"line {idx}: {line[:70]}",
                   "email_before": state.text, "n_alternatives": len(variants),
                   "alternatives": [{"label": v["label"], "kind": v["kind"]} for v in variants],
                   "selected": choice, "judge_reason": why,
                   "scores_before": state.scores}
            if choice != "KEEP" and picked["text"] != state.text:
                after = self.score(picked["text"])
                row["scores_after"] = after
                # the whole-email guard: a locally-better line must not worsen the message
                if composite(after) >= composite(state.scores) - 1e-9:
                    state.text = picked["text"]
                    state.scores = after
                    row["accepted"] = True
                    changed = True
                else:
                    row["accepted"] = False
                    row["reject_reason"] = (f"local improvement worsened the whole email "
                                            f"({composite(after)} < {composite(state.scores)})")
            else:
                row["accepted"] = False
                row["reject_reason"] = "judge kept the existing line"
            self._record(row)
            if choice not in ("DELETE", "MERGE"):
                idx += 1                                   # deletion/merge re-examines the same index
        return changed

    # ---------------------------------------------------------------- endgame sweeps
    def endgame(self, state: EditState):
        sents = _sentences(state.text)
        # 1. deletion sweep: every single-line deletion as a full variant, one comparative judgment
        variants = [{"label": "KEEP", "kind": "keep", "text": state.text}]
        for i in range(len(sents)):
            t = _join(sents[:i] + sents[i + 1:])
            if self._guard(t):
                variants.append({"label": f"DROP_{i}", "kind": "delete", "text": t})
        self._endgame_step(state, variants, "deletion sweep: is any remaining line unnecessary?")
        # 2-7. one LLM proposal each: reorder, add-beat, shorten-whole, replace-ask, new opening, reframe
        for kind, instruction in (
            ("reorder", "Reorder the beats for the strongest flow (same sentences, better order)."),
            ("add_beat", "Add ONE missing beat if any (credibility, relevance, or a clearer "
                         "permission-style ask); otherwise return the email unchanged."),
            ("shorten", "Cut the email to its shortest fully-effective form (aim 20% fewer words)."),
            ("replace_ask", "Replace the closing ask with a materially different, easier ask."),
            ("new_opening", "Replace the opening line with a materially different one."),
            ("reframe", "Rewrite with a DIFFERENT overall strategic frame (e.g. lead with the "
                        "question, or the evidence, or the shared interest), keeping the facts."),
        ):
            raw = self._llm(
                f"{self.dossier_text}\nSender facts (numbers only from these):\n"
                f"{self.brief.to_prompt() if self.brief else ''}\n"
                f"--- EMAIL ---\n{state.text}\n--- END ---\n{instruction}\n"
                "Return ONLY the complete resulting email as plain text.",
                max_tokens=320, temperature=0.5)
            alt = (raw or "").strip().strip('"')
            if alt and alt != state.text and self._guard(alt):
                self._endgame_step(state, [{"label": "KEEP", "kind": "keep", "text": state.text},
                                           {"label": kind.upper(), "kind": kind, "text": alt}],
                                   f"endgame: {instruction[:60]}")

    def _endgame_step(self, state: EditState, variants: list, purpose: str):
        if len(variants) <= 1:
            return
        choice, why = self.judge(variants, purpose)
        picked = next(v for v in variants if v["label"] == choice)
        row = {"phase": "endgame", "location": purpose, "email_before": state.text,
               "alternatives": [{"label": v["label"], "kind": v["kind"]} for v in variants],
               "selected": choice, "judge_reason": why, "scores_before": state.scores}
        if choice != "KEEP" and picked["text"] != state.text:
            after = self.score(picked["text"])
            row["scores_after"] = after
            if composite(after) >= composite(state.scores) - 1e-9:
                state.text, state.scores, row["accepted"] = picked["text"], after, True
            else:
                row["accepted"] = False
                row["reject_reason"] = "worsened whole-email score"
        else:
            row["accepted"] = False
            row["reject_reason"] = "judge kept current email"
        self._record(row)

    # ---------------------------------------------------------------- structural escape moves
    def structural_round(self, beam: list) -> list:
        """Informed complete rewrite (from accumulated critiques) + crossover of the two strongest
        drafts; admitted through the same judge + whole-email guard into the beam."""
        cands = []
        crit_text = json.dumps(self.critiques[-3:], default=str)[:1200]
        raw = self._llm(
            f"{self.dossier_text}\nSender facts (numbers only from these):\n"
            f"{self.brief.to_prompt() if self.brief else ''}\n"
            f"Accumulated editorial critiques of prior drafts:\n{crit_text}\n"
            f"Current best draft:\n{beam[0].text}\n"
            "Write a COMPLETE new email that fixes every recurring critique. Materially different "
            "structure allowed. Return ONLY the email text.",
            max_tokens=340, temperature=0.7)
        if raw and self._guard(raw.strip()):
            cands.append(EditState(label="informed_rewrite", text=raw.strip()))
        if len(beam) >= 2:
            raw2 = self._llm(
                f"{self.dossier_text}\nSender facts (numbers only from these):\n"
                f"{self.brief.to_prompt() if self.brief else ''}\n"
                f"DRAFT A:\n{beam[0].text}\n\nDRAFT B:\n{beam[1].text}\n"
                "Write ONE email combining the strongest parts of A and B (crossover, not a blend "
                "of everything). Return ONLY the email text.",
                max_tokens=340, temperature=0.6)
            if raw2 and self._guard(raw2.strip()):
                cands.append(EditState(label="crossover", text=raw2.strip()))
        for c in cands:
            c.scores = self.score(c.text)
            self._record({"phase": "structural", "location": c.label, "email_before": beam[0].text,
                          "alternatives": [{"label": c.label, "kind": c.label}],
                          "selected": c.label, "scores_after": c.scores,
                          "accepted": True, "note": "admitted to beam; ranked by score"})
        merged = beam + cands
        merged.sort(key=lambda s: -s.value)
        return merged[:self.beam_size]

    # ---------------------------------------------------------------- the full loop
    def run(self, seeds: list) -> dict:
        """seeds: [{label, text}] — strategy-diverse complete drafts (incl. plain baseline).
        Returns {'beam': [EditState best-first], 'trace': [...], 'llm_calls': n}."""
        states = []
        for s in seeds:
            if self._guard(s["text"]):
                states.append(EditState(label=s["label"], text=s["text"],
                                        scores=self.score(s["text"])))
        if not states:
            return {"beam": [], "trace": self.trace, "llm_calls": self.calls}
        states.sort(key=lambda s: -s.value)
        self._record({"phase": "seed", "location": "seed selection",
                      "alternatives": [{"label": s.label, "score": s.value} for s in states],
                      "selected": states[0].label, "accepted": True})
        beam = states[:self.beam_size]
        work = beam[0]
        for p in range(self.max_passes):
            changed = self.improve_pass(work, phase=f"pass{p + 1}")
            if not changed:
                break
        self.endgame(work)
        beam.sort(key=lambda s: -s.value)
        beam = self.structural_round(beam)
        # one final polish pass on the beam head if a structural move took the lead
        if beam[0] is not work and self.calls < self.max_llm_calls:
            self.improve_pass(beam[0], phase="post_structural")
        beam.sort(key=lambda s: -s.value)
        self._record({"phase": "final", "location": "beam",
                      "alternatives": [{"label": s.label, "score": s.value,
                                        "text": s.text} for s in beam],
                      "selected": beam[0].label, "accepted": True})
        return {"beam": beam, "trace": self.trace, "llm_calls": self.calls}
