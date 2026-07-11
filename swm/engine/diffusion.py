"""Diffusion/virality — the question class where INTERACTION genuinely drives the outcome.

"How far will this message/news/crisis spread? Who carries it? When does it inflect?" is the one class our
own ablations said needs a real interaction structure (binary event forecasting did not — interaction added
nothing there; spread IS interaction). This is the OASIS/MiroFish concept — agents on a follower graph,
emergent cascades — rebuilt lean on this engine's constitution:

  - COGNITION IS REASONING: each audience archetype's reshare propensity comes from N sampled LLM
    decisions about the ACTUAL content ("you see this post — do you amplify it?"), exactly like
    individual-mode — never a hand-set coefficient. Early- vs late-exposure decisions are sampled
    separately (novelty decay is measured from the persona, not assumed).
  - THE CASCADE IS MECHANICS, NOT VIBES: a Monte-Carlo branching process over a heavy-tailed follower
    graph (the one structural assumption, stated openly) — exposure → per-archetype sampled decision →
    new exposures — run to quiescence, many worlds.
  - NATIVE OUTPUT: a reach DISTRIBUTION (quantiles + P(reach > thresholds)), the NARRATIVE LEADERS
    (archetypes ranked by expected amplifications generated), and the INFLECTION round.
  - GRADE-OR-ABSTAIN: the class ships flagged ungraded until backtested on real cascade data (e.g.
    Upworthy-style share counts) — per the constitution, a believable cascade is not a calibrated one.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from swm.engine.grounding import parse_json

_AUDIENCE_PROMPT = """We are simulating how a piece of content spreads through a real audience.
CONTENT: {artifact}
GROUNDED SCENE:
{scene}

Identify 4-6 audience ARCHETYPES this would plausibly reach (concrete: who they are, why they'd care),
each with its share of the plausible audience (sum ~1) and its REACH multiplier (how many followers a
member has vs average: an influencer archetype 5-20x, a lurker 0.3x).
Return ONLY JSON: {{"archetypes": [{{"name": "...", "share": <0..1>, "reach_mult": <float>,
"sketch": "<2 sentences: who this is and what they care about>"}}]}}"""

_DECIDE_PROMPT = """You are one specific person seeing a piece of content in your feed.
WHO YOU ARE: {sketch}
{timing_note}
THE CONTENT ({channel}): "{artifact}"

Be this person. Do you actually AMPLIFY it (reshare/repost/forward — spend your social capital on it),
or just scroll past? Most content gets scrolled past.
Return ONLY JSON: {{"amplify": true|false, "sentiment": <-1..1 your reaction>, "why": "<one sentence>"}}"""

_EARLY = "TIMING: you're seeing this EARLY — it's novel, few of your contacts have posted it."
_LATE = "TIMING: you're seeing this LATE — it's been circulating for days; your contacts have already seen it."


@dataclass
class DiffusionForecast:
    reach: dict = field(default_factory=dict)          # {"p10","p50","p90"} as audience fractions
    p_over: dict = field(default_factory=dict)         # {"0.05":p, "0.2":p, "0.5":p} P(reach > frac)
    narrative_leaders: list = field(default_factory=list)   # archetypes by expected amplifications
    inflection_round: float = None                     # median round of peak new exposures
    sentiment: float = None                            # exposure-weighted mean sentiment
    archetypes: list = field(default_factory=list)     # audit: per-archetype sampled propensities + whys
    n_calls: int = 0
    n_worlds: int = 0


@dataclass
class DiffusionSimulator:
    llm_hot: object
    llm: object = None
    reps_per_archetype: int = 6                        # sampled decisions per (archetype × timing)
    n_nodes: int = 400                                 # audience graph size (archetype-proportional)
    n_worlds: int = 200                                # Monte-Carlo cascades
    n_seeds: int = 3                                   # initially-exposed nodes
    max_workers: int = 8
    seed: int = 0

    def simulate(self, artifact: str, dossier, *, channel="social post") -> DiffusionForecast:
        llm_cold = self.llm or self.llm_hot
        raw = parse_json(llm_cold(_AUDIENCE_PROMPT.format(artifact=artifact[:1200],
                                                          scene=dossier.brief()))) or {}
        arch = [a for a in raw.get("archetypes", []) if isinstance(a, dict) and a.get("name")][:6]
        if not arch:
            return DiffusionForecast()
        z = sum(max(0.0, float(a.get("share", 0) or 0)) for a in arch) or 1.0

        # ---- per-archetype propensities from SAMPLED reasoned decisions on the actual content ----
        jobs = [(i, timing) for i in range(len(arch)) for timing in ("early", "late")
                for _ in range(self.reps_per_archetype)]

        def one(job):
            i, timing = job
            r = parse_json(self.llm_hot(_DECIDE_PROMPT.format(
                sketch=arch[i].get("sketch", arch[i]["name"]), channel=channel,
                timing_note=_EARLY if timing == "early" else _LATE, artifact=artifact[:1200])))
            if not r or not isinstance(r.get("amplify"), bool):
                return None
            try:
                senti = max(-1.0, min(1.0, float(r.get("sentiment", 0) or 0)))
            except (TypeError, ValueError):
                senti = 0.0
            return (i, timing, 1 if r["amplify"] else 0, senti, str(r.get("why", ""))[:120])

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = [r for r in ex.map(one, jobs) if r is not None]
        if not results:
            return DiffusionForecast(n_calls=len(jobs))

        audit = []
        for i, a in enumerate(arch):
            for timing in ("early", "late"):
                rs = [r for r in results if r[0] == i and r[1] == timing]
                yes = sum(r[2] for r in rs)
                a[f"p_{timing}"] = (yes + 0.5) / (len(rs) + 1.0) if rs else 0.05   # Laplace-smoothed
            a["share"] = max(0.0, float(a.get("share", 0) or 0)) / z
            a["reach_mult"] = max(0.1, float(a.get("reach_mult", 1.0) or 1.0))
            a["sentiment"] = (sum(r[3] for r in results if r[0] == i) /
                              max(1, sum(1 for r in results if r[0] == i)))
            audit.append({"name": a["name"], "share": round(a["share"], 3),
                          "p_early": round(a["p_early"], 3), "p_late": round(a["p_late"], 3),
                          "reach_mult": a["reach_mult"], "sentiment": round(a["sentiment"], 2),
                          "why": next((r[4] for r in results if r[0] == i and r[2] == 1), "")})

        # ---- the cascade mechanics: Monte-Carlo branching over a heavy-tailed follower graph ----
        rng = random.Random(self.seed)
        nodes = []                                      # node -> archetype index
        for i, a in enumerate(arch):
            nodes += [i] * max(1, int(round(a["share"] * self.n_nodes)))
        n = len(nodes)
        degrees = [max(1, int(arch[nodes[v]]["reach_mult"] * (1.0 / max(0.02, rng.random())) ** 0.5))
                   for v in range(n)]                   # heavy-tailed audience sizes (stated assumption)

        finals, peaks, amps_by_arch = [], [], [0.0] * len(arch)
        senti_accum = senti_n = 0.0
        for _ in range(self.n_worlds):
            wrng = random.Random(rng.random())
            exposed = set(wrng.sample(range(n), min(self.n_seeds, n)))
            frontier = list(exposed)
            new_counts, rnd = [], 0
            while frontier and rnd < 12:
                rnd += 1
                nxt = []
                for v in frontier:
                    a = arch[nodes[v]]
                    p = a["p_early"] if rnd <= 2 else a["p_late"]
                    if wrng.random() < p:               # this person amplifies → exposes their audience
                        amps_by_arch[nodes[v]] += 1
                        senti_accum += a["sentiment"]; senti_n += 1
                        k = min(degrees[v], n - 1)
                        for w in wrng.sample(range(n), k):
                            if w not in exposed:
                                exposed.add(w)
                                nxt.append(w)
                new_counts.append(len(nxt))
                frontier = nxt
            finals.append(len(exposed) / n)
            peaks.append(1 + new_counts.index(max(new_counts)) if new_counts and max(new_counts) > 0 else 1)

        finals.sort()
        q = lambda f: finals[min(len(finals) - 1, int(f * len(finals)))]
        leaders = sorted(({"archetype": arch[i]["name"],
                           "amplifications_per_world": round(amps_by_arch[i] / self.n_worlds, 2)}
                          for i in range(len(arch))), key=lambda x: -x["amplifications_per_world"])
        peaks.sort()
        return DiffusionForecast(
            reach={"p10": round(q(0.10), 4), "p50": round(q(0.50), 4), "p90": round(q(0.90), 4)},
            p_over={t: round(sum(1 for x in finals if x > float(t)) / len(finals), 3)
                    for t in ("0.05", "0.2", "0.5")},
            narrative_leaders=leaders, inflection_round=peaks[len(peaks) // 2],
            sentiment=round(senti_accum / senti_n, 3) if senti_n else None,
            archetypes=audit, n_calls=1 + len(jobs), n_worlds=self.n_worlds)
