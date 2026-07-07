"""PopulationSimulator — Level 3: large-scale demographic prediction as a coupled simulation.

GENERAL, not election-specific. The unit is a DEMOGRAPHIC CELL (a subgroup: "white, college, 30-44,
south" — or any partition), carrying a stance, a size weight, a responsiveness, and a PARTICIPATION
propensity (turnout). "Who wins the election" is ONE instance of the shape "what does a large population
do", alongside: what share adopts a product, supports a policy, joins a movement, boycotts a brand. The
same machinery answers all of them; only the AGGREGATOR differs.

Three parts, matching the gap this closes:

  (a) REAL CELLS at scale — hundreds of subgroups, each a mean-field agent (belief, responsiveness,
      influence=size×turnout), built from real survey/census data.

  (b) COUPLING THAT CAN BITE — two coupled channels over the horizon:
        - OPINION coupling (mean-field): conformity + social proof / bandwagon (MeanFieldRollout);
        - PARTICIPATION coupling (the new piece): a cell's turnout is not fixed — enthusiasm rises when
          the aggregate is moving its way (mobilization) and falls when it is losing (discouragement).
      Differential, stance-coupled turnout is exactly where large-scale outcomes stop being a linear pool
      of marginals — an enthusiastic bloc that turns out amplifies itself. This is the general analogue of
      turnout surges, viral adoption, and protest cascades.

  (c) AGGREGATION LAYER — pluggable. `share_aggregator` (participation-weighted population share: the
      general default) or `winner_take_all_aggregator` (regional winner roll-up: the electoral-college
      shape, and any "majority per region, then count regions" outcome).

Held honestly to the KPI in `population_metrics`: the coupled outcome must beat the MARGINAL-AVERAGE
composite (the same cells, no interaction) — otherwise it is a fancy poll average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.simulation.mean_field import Agent, MeanFieldRollout


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


@dataclass
class DemographicCell:
    cell_id: str
    weight: float                       # subgroup size (share of the population)
    stance: float                       # current support / adoption probability in [0,1]
    responsiveness: float = 0.3         # how fast this cell updates (openness/volatility)
    turnout: float = 1.0                # participation propensity (who actually acts) in [0,1]
    region: str = ""                    # for the winner-take-all aggregator (optional)
    enthusiasm: float = 0.5             # mobilization state; couples to turnout over the horizon


# ---- aggregators (pluggable) ------------------------------------------------------------------------
def share_aggregator(cells) -> float:
    """GENERAL default: participation-weighted population share (support/adoption/turnout-weighted %)."""
    num = sum(c.weight * c.turnout * c.stance for c in cells)
    den = sum(c.weight * c.turnout for c in cells) or 1.0
    return num / den


def marginal_share(cells) -> float:
    """The NO-INTERACTION composite the coupled sim must beat: raw size-weighted mean of cell stances,
    frozen at their as-of values, ignoring participation dynamics — a linear pool of marginals."""
    den = sum(c.weight for c in cells) or 1.0
    return sum(c.weight * c.stance for c in cells) / den


def winner_take_all_aggregator(cells) -> dict:
    """Regional roll-up (the electoral-college shape, fully general): within each region take the
    participation-weighted majority, then report the fraction of regions (optionally region-weighted) won.
    Returns the share of region-weight on the YES side + per-region detail."""
    regions = {}
    for c in cells:
        r = regions.setdefault(c.region or "all", {"num": 0.0, "den": 0.0, "w": 0.0})
        r["num"] += c.weight * c.turnout * c.stance
        r["den"] += c.weight * c.turnout
        r["w"] += c.weight
    won_w = tot_w = 0.0
    detail = {}
    for name, r in regions.items():
        share = r["num"] / (r["den"] or 1.0)
        detail[name] = round(share, 4)
        tot_w += r["w"]
        if share > 0.5:
            won_w += r["w"]
    return {"region_share_won": (won_w / tot_w) if tot_w else 0.0, "by_region": detail}


@dataclass
class PopulationSimulator:
    """Roll a demographic population forward with coupled opinion + participation, then aggregate.
    `aggregator(cells) -> float | dict`. `turnout_coupling` sets how strongly enthusiasm moves turnout."""
    rollout: MeanFieldRollout = field(default_factory=lambda: MeanFieldRollout(k_social=0.15, k_proof=0.0))
    aggregator: object = share_aggregator
    turnout_coupling: float = 0.0       # 0 = fixed turnout; >0 = stance-coupled mobilization (cascade)
    enthusiasm_gain: float = 0.5        # how much a cell's enthusiasm feeds its turnout

    @staticmethod
    def _scalar(out):
        return out["region_share_won"] if isinstance(out, dict) else out

    def _copy(self, cells):
        return [DemographicCell(c.cell_id, c.weight, c.stance, c.responsiveness, c.turnout,
                                c.region, c.enthusiasm) for c in cells]

    def simulate(self, cells, steps: int = 6, events=None) -> dict:
        """Return the MARGINAL (no-interaction) outcome AND the COUPLED outcome + trajectory. The marginal
        baseline is the SAME aggregator applied to the FROZEN cells (no opinion coupling, no mobilization);
        the coupled outcome applies it to the evolved cells — so the difference is purely the dynamics.
        Mutates a copy, never the input."""
        frozen, work = self._copy(cells), self._copy(cells)
        base_turnout = [c.turnout for c in work]         # baseline participation (enthusiasm 0.5 = unchanged)
        marginal_out = self.aggregator(frozen)
        # opinion coupling via mean-field over cell agents (influence = size × current turnout)
        agents = [Agent(belief=_clamp(c.stance), responsiveness=c.responsiveness,
                        influence=max(1e-3, c.weight * c.turnout)) for c in work]
        traj = []
        for t in range(steps):
            ev = 0.0 if not events else (events[t] if t < len(events) else events[-1])
            agg = self.rollout.step(agents, ev)
            for i, (c, a) in enumerate(zip(work, agents)):
                c.stance = a.belief
                if self.turnout_coupling > 0:            # PARTICIPATION coupling (mobilization cascade)
                    momentum = (a.belief - 0.5) * (agg - 0.5)   # >0: your side is winning -> mobilize
                    c.enthusiasm = _clamp(c.enthusiasm + self.turnout_coupling * momentum)
                    # enthusiasm 0.5 => baseline; >0.5 mobilizes, <0.5 demobilizes (neutral-preserving)
                    c.turnout = _clamp(base_turnout[i] * (1 + self.enthusiasm_gain * 2 * (c.enthusiasm - 0.5)))
                    a.influence = max(1e-3, c.weight * c.turnout)
            traj.append(round(self._scalar(self.aggregator(work)), 4))
        coupled_out = self.aggregator(work)
        return {"marginal": self._scalar(marginal_out), "coupled": self._scalar(coupled_out),
                "marginal_detail": marginal_out, "coupled_detail": coupled_out, "trajectory": traj,
                "final_cells": [(c.cell_id, round(c.stance, 3), round(c.turnout, 3)) for c in work]}


def cells_from_rows(rows, cell_key, stance_key, *, responsiveness=0.3, turnout_key=None,
                    region_key=None) -> list:
    """Build demographic cells by grouping labeled rows. `cell_key(row)->tuple`, `stance_key(row)->0/1`,
    optional `turnout_key(row)->0..1`, `region_key(row)->str`. Cell stance = mean outcome; weight = count."""
    agg = {}
    for r in rows:
        k = cell_key(r)
        a = agg.setdefault(k, {"n": 0, "s": 0.0, "t": 0.0, "region": region_key(r) if region_key else ""})
        a["n"] += 1
        a["s"] += float(stance_key(r))
        a["t"] += float(turnout_key(r)) if turnout_key else 1.0
    cells = []
    for k, a in agg.items():
        cells.append(DemographicCell(cell_id=str(k), weight=float(a["n"]), stance=a["s"] / a["n"],
                                     responsiveness=responsiveness, turnout=a["t"] / a["n"],
                                     region=a["region"]))
    return cells
