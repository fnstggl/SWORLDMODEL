"""The society rollout — Stage 3: roll forward under uncertainty, in real calendar time.

Each ROUND is a real dated step (today → resolve_by at the cast's cadence): agents are told the date and
instructed to consider only changes plausible within that interval — one week of rollout is one week of
change, never an arbitrary "t+1". Each round, every persona REASONS (swm/engine/agents.py) from its
grounded dossier + the PUBLIC SIGNAL of what just happened, and emits a decision distribution + optionally
a public statement. Interaction is real: the next round's public signal is built from a SAMPLED realization
of this round's aggregate (a poll reading drawn from the current distribution) plus sampled statements — so
cascades and bandwagons can happen, and different sampled histories genuinely diverge.

Uncertainty enters through four honest doors, and the output separates them:
  - persona sampling      (variants per actor — who exactly the population is)
  - private information   (rotated evidence slices — who has seen what)
  - interaction stochasticity (B independent BRANCHES with different sampled public histories)
  - decision noise        (temperature on the reasoning calls)
The forecast distribution is the branch-weighted persona-weighted aggregate; the branch spread is reported
as the interaction-uncertainty interval. NO homophily/consensus_pull constants anywhere — if agents herd,
it is because they read the same public signal and reasoned their way to it.
"""
from __future__ import annotations

import random
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from swm.engine.agents import decide, draw_variants, slice_private_facts
from swm.engine.calibrate import pool_distribution


def _dates(horizon_days, cadence_days, max_rounds, today=""):
    """Real dated rounds: today stepping by cadence, capped, always ending at the horizon."""
    t0 = _time.mktime(_time.strptime(today, "%Y-%m-%d")) if today else _time.time()
    n = max(1, min(max_rounds, int(round(horizon_days / max(cadence_days, 0.5)))))
    step = horizon_days / n * 86400.0
    return [_time.strftime("%Y-%m-%d", _time.localtime(t0 + (i + 1) * step)) for i in range(n)]


def _sample_reading(rng, probs):
    """A noisy public 'poll reading' drawn around the current aggregate — what the world would actually
    see published, not the true latent distribution (sampling error is real interaction noise)."""
    noisy = {o: max(1e-6, p + rng.gauss(0, 0.04)) for o, p in probs.items()}
    z = sum(noisy.values())
    return {o: v / z for o, v in noisy.items()}


@dataclass
class SocietyResult:
    distribution: dict                    # option -> p (the native answer)
    branch_distributions: list            # per-branch final distributions (interaction spread)
    interval: dict                        # option -> [min, max] across branches
    rounds: list                          # the real dates simulated
    audit: list = field(default_factory=list)   # per-persona final probs + statements + why
    n_personas: int = 0
    n_calls: int = 0


@dataclass
class SocietyRollout:
    llm_hot: object                       # decision-call backend (temperature > 0)
    llm: object = None                    # cold backend for variant drawing (defaults to llm_hot)
    branches: int = 3
    max_rounds: int = 2
    max_workers: int = 8
    seed: int = 0

    def run(self, question, cast, dossier, *, today="") -> SocietyResult:
        llm_cold = self.llm or self.llm_hot
        rng = random.Random(self.seed)
        options = cast.options()
        dates = _dates(cast.horizon_days, cast.cadence_days, self.max_rounds, today)
        resolve_note = (f"The outcome is decided on {cast.resolve_by}." if cast.resolve_by else
                        f"The outcome resolves about {cast.horizon_days:.0f} days from today.")

        # ---- Stage 2: instantiate every actor as diverse personas with private evidence slices ----
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            drawn = list(ex.map(lambda a: draw_variants(llm_cold, a, question, dossier.brief()),
                                cast.actors))
        personas = [p for group in drawn for p in group]
        # The DECIDING facts are COMMON knowledge (in a covered race everyone knows the front-runner) — never
        # rotate the deciding signal away from an agent, or it reasons in a vacuum (the 'underconfident on a
        # clear favorite' failure). Every persona sees all grounded facts; diversity comes from the variant
        # sketch + temperature + a little rotated PERIPHERAL color, not from hiding the standing.
        all_facts = [f"{f['fact']}: {f.get('detail', '')}" for f in dossier.facts]
        for i, p in enumerate(personas):
            peripheral = slice_private_facts(dossier.facts, i, keep=0.4)
            p.private_facts = all_facts + [f"(you weigh this personally: {x})" for x in peripheral[:2]]
        standing = dossier.standing
        n_calls = len(cast.actors)

        # ---- Stage 3: B branches × dated rounds; each round every persona reasons; the branch's next
        # public signal is a SAMPLED realization of this round's aggregate + sampled statements ----
        branch_finals, audit = [], []
        for b in range(self.branches):
            brng = random.Random(self.seed * 997 + b)
            public = ((f"CURRENT STANDING: {standing}. " if standing else "") +
                      (cast.interaction or "the situation as grounded above"))
            last = None
            for ri, date in enumerate(dates):
                interval_days = cast.horizon_days / len(dates)
                time_note = (f"{resolve_note} This round advances the world to {date} "
                             f"(~{interval_days:.0f} days of real change — assume only what can "
                             f"plausibly happen in that time).")
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    results = list(ex.map(
                        lambda p: decide(self.llm_hot, p, question, options, date=date,
                                         public=public, time_note=time_note), personas))
                n_calls += len(personas)
                voted = [(p, r) for p, r in zip(personas, results) if r is not None]
                if not voted:
                    break
                # LOG-LINEAR opinion pool (weighted geometric mean), not a linear mean — agreement sharpens,
                # a dissenter widens; a single certain persona can't force 0/1 (finite-sample smoothing).
                agg = pool_distribution([r["probs"] for _, r in voted],
                                        [p.weight for p, _ in voted])
                last = (voted, agg)
                if ri < len(dates) - 1:                     # build the next round's public world
                    reading = _sample_reading(brng, agg)
                    lead = " / ".join(f"{o} {v:.0%}" for o, v in
                                      sorted(reading.items(), key=lambda kv: -kv[1])[:4])
                    statements = [r["statement"] for _, r in voted if r["statement"]]
                    brng.shuffle(statements)
                    said = ("  Heard around: " + " | ".join(f'"{s}"' for s in statements[:3])) \
                        if statements else ""
                    public = f"As of {date}, published polling/read of the race: {lead}.{said}"
            if last is not None:
                voted, agg = last
                branch_finals.append(agg)
                if b == 0:                                   # audit one branch fully
                    audit = [{"persona": f"{p.actor_name} [{p.variant[:60]}]",
                              "weight": round(p.weight, 4),
                              "probs": {o: round(r['probs'].get(o, 0.0), 3) for o in options},
                              "why": r["why"], "statement": r["statement"]} for p, r in voted]

        if not branch_finals:
            return SocietyResult({}, [], {}, dates, [], len(personas), n_calls)
        dist = {o: sum(bf.get(o, 0.0) for bf in branch_finals) / len(branch_finals) for o in options}
        interval = {o: [round(min(bf.get(o, 0.0) for bf in branch_finals), 4),
                        round(max(bf.get(o, 0.0) for bf in branch_finals), 4)] for o in options}
        return SocietyResult(distribution={o: round(p, 4) for o, p in dist.items()},
                             branch_distributions=[{o: round(v, 4) for o, v in bf.items()}
                                                   for bf in branch_finals],
                             interval=interval, rounds=dates, audit=audit,
                             n_personas=len(personas), n_calls=n_calls)
