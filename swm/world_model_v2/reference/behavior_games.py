"""Reference World B — economic-game populations (BehaviorBench moblab/game_behavior, benchmark use).

The prediction target per game is the DISTRIBUTION of real human choices (one condition, ~200 people,
split train/test). This world exercises what Enron structurally could not: STRATEGIC INTERACTION —
an actor's choice driven by simulated beliefs about another actor's response.

Structure per game (all through the universal V2 runtime — entities, latents, events, deltas, terminal
readout; nothing game-branching in the engine):

  * latent SOCIAL PREFERENCES per particle, priors FITTED ON TRAIN (smoothed empirical; status=fitted):
    altruism (dictator train), fairness threshold (responder train), banker return fraction (banker train),
    level-k depth mixture (guessing train), conditional-cooperation slope (public-goods train)
  * CROSS-GAME TRANSFER: the proposer's world contains a simulated RESPONDER whose threshold latent is
    fitted from the responder game's train sample; the investor's world contains a simulated BANKER fitted
    from the banker train — interaction is a mechanism, not a metaphor
  * one LLM INTERPRETATION of the game situation (universal interpret()); dims shift prior means within
    bounded ±15%-of-range (ablatable)
  * decision noise (trembling/softmax β) fitted per game ON TRAIN by 1-D grid search

Ablations: interp_on=False (no semantic reading), latent=False (point preferences), interaction=False
(no simulated partner — preference-only policy). Persistence & long-horizon rollout are STRUCTURALLY NOT
EXERCISED by one-shot games (logged; OmniBehavior/Higgs carry those tests).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

GAME_SPECS = {
    "dictator":            {"lo": 0.0, "hi": 100.0, "partner": None},
    "ultimatum_responder": {"lo": 0.0, "hi": 100.0, "partner": None},
    "ultimatum_proposer":  {"lo": 0.0, "hi": 100.0, "partner": "responder"},
    "trust_investor":      {"lo": 0.0, "hi": 100.0, "partner": "banker"},
    "trust_banker":        {"lo": 0.0, "hi": 150.0, "partner": None},
    "guessing":            {"lo": 0.0, "hi": 100.0, "partner": "population"},
    "public_goods":        {"lo": 0.0, "hi": 20.0,  "partner": "population"},
}


# ---------------------------------------------------------------- fitted population priors (train only)
@dataclass
class FittedDist:
    """Smoothed empirical distribution over a bounded range (Silverman-bandwidth gaussian kernel around
    train points). status: fitted (train sample). sample() is the population prior draw."""
    points: list
    lo: float
    hi: float
    bw: float
    status: str = "fitted (train sample, KDE-smoothed)"

    def sample(self, rng, *, shift=0.0):
        v = self.points[rng.randrange(len(self.points))] + rng.gauss(0.0, self.bw) + shift
        return min(self.hi, max(self.lo, v))

    def mean(self):
        return sum(self.points) / len(self.points)


def fit_dist(sample, lo, hi) -> FittedDist:
    pts = [min(hi, max(lo, float(v))) for v in sample]
    n = len(pts) or 1
    mu = sum(pts) / n
    sd = (sum((x - mu) ** 2 for x in pts) / max(1, n - 1)) ** 0.5
    bw = max((hi - lo) * 0.01, 1.06 * sd * n ** (-0.2))     # Silverman
    return FittedDist(points=pts, lo=lo, hi=hi, bw=bw)


@dataclass
class GamePriors:
    """Every latent's parameter source, fitted on TRAIN samples only. Cross-game: a latent fitted in one
    game parameterizes SIMULATED PARTNERS in another (the interaction mechanism's grounding)."""
    dists: dict                            # game -> FittedDist of its own train answers
    altruism: FittedDist = None            # dictator train / 100 → [0,1]
    threshold: FittedDist = None           # responder train (dollars)
    banker_frac: FittedDist = None         # banker train / 150 → [0,1]
    levelk_w: list = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])   # fitted on guessing train
    pg_slope: float = 0.7                  # conditional-cooperation slope, fitted on public_goods train
    beta: dict = field(default_factory=dict)     # per-game trembling sd (fraction of range), fitted


def _w1(a, b):
    """1-D Wasserstein-1 between empirical samples via 50 matched quantiles."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    n = 50
    return sum(abs(a[min(len(a) - 1, int(t / n * len(a)))] -
                   b[min(len(b) - 1, int(t / n * len(b)))]) for t in range(n)) / n


def fit_priors(train_by_game: dict, *, seed=7) -> GamePriors:
    gp = GamePriors(dists={g: fit_dist(s, GAME_SPECS[g]["lo"], GAME_SPECS[g]["hi"])
                           for g, s in train_by_game.items()})
    if "dictator" in train_by_game:
        gp.altruism = fit_dist([v / 100.0 for v in train_by_game["dictator"]], 0.0, 1.0)
    if "ultimatum_responder" in train_by_game:
        gp.threshold = gp.dists["ultimatum_responder"]
    if "trust_banker" in train_by_game:
        gp.banker_frac = fit_dist([v / 150.0 for v in train_by_game["trust_banker"]], 0.0, 1.0)
    # level-k mixture on guessing train: grid over simplex (coarse, 4 levels) minimizing W1
    if "guessing" in train_by_game:
        tr = train_by_game["guessing"]
        rng = random.Random(seed)
        best, best_w = None, gp.levelk_w
        grid = [i / 10 for i in range(11)]
        for w0 in grid:
            for w1_ in grid:
                for w2 in grid:
                    w3 = 1.0 - w0 - w1_ - w2
                    if w3 < -1e-9:
                        continue
                    w3 = max(0.0, w3)
                    sim = [_levelk_guess([w0, w1_, w2, w3], rng) for _ in range(300)]
                    d = _w1(sim, tr)
                    if best is None or d < best:
                        best, best_w = d, [w0, w1_, w2, w3]
        gp.levelk_w = best_w
    # conditional-cooperation slope on public_goods train
    if "public_goods" in train_by_game and gp.altruism is not None:
        tr = train_by_game["public_goods"]
        rng = random.Random(seed + 1)
        belief = gp.dists["public_goods"]
        best, best_s = None, gp.pg_slope
        for s10 in range(0, 11):
            s = s10 / 10
            sim = [min(20.0, max(0.0, s * belief.sample(rng) + (1 - s) * 20.0 * gp.altruism.sample(rng)))
                   for _ in range(300)]
            d = _w1(sim, tr)
            if best is None or d < best:
                best, best_s = d, s
        gp.pg_slope = best_s
    return gp


def _levelk_guess(w, rng):
    r, acc, k = rng.random(), 0.0, 3
    for i, wi in enumerate(w):
        acc += wi
        if r <= acc:
            k = i
            break
    if k == 0:
        return rng.uniform(0, 100)
    return min(100.0, max(0.0, 50.0 * (2.0 / 3.0) ** k + rng.gauss(0, 6.0)))


def fit_beta(game, gp: GamePriors, train_sample, interp, *, interaction=True, seed=11) -> float:
    """Trembling noise (sd as fraction of range) fitted ON TRAIN: simulate the arm's own policy at each
    candidate β, keep the β whose simulated distribution is W1-closest to the TRAIN sample."""
    best, best_b = None, 0.05
    for b in (0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25):
        sim = simulate_game(game, gp, interp, n=300, seed=seed, beta=b, interaction=interaction,
                            latent=True, interp_on=interp is not None)
        d = _w1(sim, train_sample)
        if best is None or d < best:
            best, best_b = d, b
    return best_b


# ---------------------------------------------------------------- the per-particle structural policies
def _interp_shift(interp, dim, rng_frac=0.15):
    """Bounded prior-mean shift from one interpretation dim: ±rng_frac of range at the extremes."""
    return rng_frac * (getattr(interp, dim) - 0.5) * 2.0 if interp is not None else 0.0


def particle_action(game, gp: GamePriors, interp, rng, *, beta, interaction=True, latent=True):
    """ONE actor's choice in one sampled world. Latents are per-particle draws (latent=True) or prior
    means (latent=False). interaction=True simulates the partner population inside the actor's beliefs
    (m=40 internal draws from the CROSS-GAME fitted latent)."""
    lo, hi = GAME_SPECS[game]["lo"], GAME_SPECS[game]["hi"]

    def draw(dist, shift=0.0):
        return dist.sample(rng, shift=shift) if latent else min(dist.hi, max(dist.lo, dist.mean() + shift))

    tremble = rng.gauss(0.0, beta * (hi - lo))
    a_shift = _interp_shift(interp, "benefit_of_action", 0.15)          # generosity/engagement reading
    r_shift = _interp_shift(interp, "risk_of_inaction", 0.15)

    if game == "dictator":
        give = 100.0 * draw(gp.altruism, shift=0.15 * a_shift)
        return min(hi, max(lo, give + tremble))
    if game == "ultimatum_responder":
        t = draw(gp.threshold, shift=15.0 * _interp_shift(interp, "obligation", 0.15))
        return min(hi, max(lo, t + tremble))
    if game == "trust_banker":
        f = draw(gp.banker_frac, shift=0.15 * a_shift)
        return min(hi, max(lo, 150.0 * f + tremble))
    if game == "ultimatum_proposer":
        alt = draw(gp.altruism, shift=0.15 * a_shift)
        if not interaction:                                  # preference-only giving (no partner model)
            return min(hi, max(lo, 100.0 * alt + tremble))
        thr = [gp.threshold.sample(rng) for _ in range(40)]  # simulated responder population (fitted)
        best_o, best_u = 50.0, -1e9
        for o in range(0, 101, 5):
            p_acc = sum(1 for t in thr if o >= t) / len(thr)
            u = (100.0 - o) * p_acc + 60.0 * alt * (o / 100.0) * p_acc
            if u > best_u:
                best_u, best_o = u, float(o)
        return min(hi, max(lo, best_o + tremble))
    if game == "trust_investor":
        if not interaction:
            alt = draw(gp.altruism, shift=0.15 * a_shift)
            return min(hi, max(lo, 100.0 * alt + tremble))
        fr = [gp.banker_frac.sample(rng) for _ in range(40)]  # simulated banker population (fitted)
        ef = sum(fr) / len(fr)
        sf = (sum((x - ef) ** 2 for x in fr) / len(fr)) ** 0.5
        # risk preference: BROAD prior spanning risk-seeking to risk-averse (labeled prior) —
        # human trust data is bimodal all-or-nothing, which EU-with-narrow-risk cannot express
        lam = min(2.0, max(-1.0, rng.gauss(0.5, 1.0))) if latent else 0.5
        lam = lam - r_shift
        best_x, best_u = 0.0, -1e9
        for x in range(0, 101, 5):
            u = (100.0 - x) + 3.0 * x * ef - lam * 3.0 * x * sf
            if u > best_u:
                best_u, best_x = u, float(x)
        return min(hi, max(lo, best_x + tremble))
    if game == "guessing":
        if not interaction:                                  # no iterated reasoning about the population
            return min(hi, max(lo, rng.uniform(0, 100)))
        return _levelk_guess(gp.levelk_w if latent else
                             [1.0 if i == max(range(4), key=lambda j: gp.levelk_w[j]) else 0.0
                              for i in range(4)], rng)
    if game == "public_goods":
        alt = draw(gp.altruism, shift=0.15 * a_shift)
        if not interaction:
            return min(hi, max(lo, 20.0 * alt + tremble))
        belief = gp.dists["public_goods"].sample(rng)        # belief about others' contribution (fitted)
        c = gp.pg_slope * belief + (1 - gp.pg_slope) * 20.0 * alt
        return min(hi, max(lo, c + tremble))
    raise KeyError(game)


def simulate_game(game, gp, interp, *, n, seed, beta, interaction=True, latent=True, interp_on=True):
    """Fast path (no world objects): n particle actions. Used for β fitting and the big-n readout."""
    rng = random.Random(seed)
    it = interp if interp_on else None
    return [particle_action(game, gp, it, rng, beta=beta, interaction=interaction, latent=latent)
            for _ in range(n)]


# ---------------------------------------------------------------- the V2 world (execution-trace path)
def v2_game_world(game, gp: GamePriors, interp, *, n_particles=24, seed=0, beta=0.05,
                  interaction=True, latent=True, interp_on=True, t0=1.0e9):
    """The anti-cheating path: the SAME policy runs through the universal runtime — typed entities,
    a real decision event, a partner-response event where one exists, machine-readable deltas, terminal
    readout. Returns {'sample', 'trace_branches', 'n_deltas'} — sample comes from terminal states."""
    from swm.world_model_v2.contracts import OutcomeContract
    from swm.world_model_v2.events import Event, EventQueue, register_event_type
    from swm.world_model_v2.init_state import InitialStateModel, LatentVariableRecord
    from swm.world_model_v2.rollout import WorldModelV2Run
    from swm.world_model_v2.state import Entity, F, SimulationClock, WorldState
    from swm.world_model_v2.transitions import TransitionOperator, TransitionProposal, StateDelta

    partner = GAME_SPECS[game]["partner"]
    base = WorldState(world_id=f"bb_{game}"[:12], branch_id="root",
                      clock=SimulationClock(now=t0, as_of=t0))
    actor = Entity(identity="actor")
    actor.set("current_action", F(None, status="assumed"))
    base.entities["actor"] = actor
    if partner == "responder":
        p = Entity(identity="partner")
        p.set("current_action", F(None, status="assumed"))
        base.entities["partner"] = p
    latents = [LatentVariableRecord(path="actor.attention",
                                    candidates={"mean": 0.7, "sd": 0.15, "lo": 0.3, "hi": 1.0},
                                    method="prior")]
    init = InitialStateModel(base_world=base, latents=latents)
    register_event_type("game_decision", scheduling="scheduled", validated=True)
    register_event_type("partner_response", scheduling="scheduled", validated=True)

    class GameDecision(TransitionOperator):
        name = "game_decision_structural"

        def applicable(self, world, event):
            return event.etype == "game_decision" and world.entity("actor").value("current_action") is None

        def propose(self, world, event, rng):
            v = particle_action(game, gp, interp if interp_on else None, rng, beta=beta,
                                interaction=interaction, latent=latent)
            return TransitionProposal(operator=self.name, action={"actor": "actor", "type": "choose",
                                                                  "value": round(v, 2)},
                                      reason_codes=[f"game={game}",
                                                    "interaction" if interaction else "no_interaction"])

        def apply(self, world, proposal):
            a = world.entity("actor")
            before = a.value("current_action")
            a.set("current_action", F(proposal.action["value"], status="derived", method=self.name,
                                      updated_at=world.clock.now))
            world.quantities["choice"] = type("Q", (), {"value": proposal.action["value"]})()
            d = StateDelta(at=world.clock.now, event_type="game_decision", operator=self.name,
                           reason_codes=proposal.reason_codes)
            return d.change("actor.current_action", before, proposal.action["value"])

    class PartnerResponse(TransitionOperator):
        """The ultimatum responder actually RESPONDS inside the proposer's world (fitted threshold draw)
        — the interaction is realized in the terminal state, not just anticipated."""
        name = "partner_response_fitted"

        def applicable(self, world, event):
            return (event.etype == "partner_response" and "partner" in world.entities
                    and world.entity("actor").value("current_action") is not None)

        def propose(self, world, event, rng):
            offer = float(world.entity("actor").value("current_action") or 0.0)
            thr = gp.threshold.sample(rng)
            return TransitionProposal(operator=self.name,
                                      action={"actor": "partner",
                                              "type": "accept" if offer >= thr else "reject"},
                                      reason_codes=[f"threshold={thr:.0f}"])

        def apply(self, world, proposal):
            p = world.entity("partner")
            before = p.value("current_action")
            p.set("current_action", F(proposal.action["type"], status="derived", method=self.name,
                                      updated_at=world.clock.now))
            d = StateDelta(at=world.clock.now, event_type="partner_response", operator=self.name,
                           reason_codes=proposal.reason_codes)
            return d.change("partner.current_action", before, proposal.action["type"])

    def build_queue(world):
        q = EventQueue(horizon_ts=t0 + 3600.0)
        q.schedule(Event(ts=t0 + 60.0, etype="game_decision", participants=["actor"]))
        if partner == "responder":
            q.schedule(Event(ts=t0 + 120.0, etype="partner_response", participants=["partner"]))
        return q

    contract = OutcomeContract(
        family="continuous", options=["chosen", "no_choice"],
        resolution_rule="actor made a numeric choice",
        readout=lambda w: "chosen" if w.entity("actor").value("current_action") is not None else "no_choice",
        horizon_ts=t0 + 3600.0).validate()
    run = WorldModelV2Run(initial=init, queue_builder=build_queue,
                          operators=[GameDecision(), PartnerResponse()],
                          contract=contract, n_particles=n_particles)
    _, branches = run.run(seed=seed)
    sample = [b.world.quantities["choice"].value for b in branches if "choice" in b.world.quantities]
    return {"sample": sample, "trace_branches": branches,
            "n_deltas": sum(len(b.log) for b in branches)}
