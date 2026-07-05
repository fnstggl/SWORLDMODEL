"""TrajectoryState — the mutable world state carried through ONE simulated trajectory (Phase 2).

Each Monte-Carlo trajectory owns a TrajectoryState that evolves as events fire: the score
accumulates, segments gain exposure and attention shifts, social proof rises, the front-page flag
flips, novelty decays. The engine samples many trajectories; the outcome distribution over their
`accumulated_score` is where the prediction comes from — NOT a classifier over the initial state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.simulation.actors import SegmentAgent


@dataclass
class TrajectoryState:
    timestep: int = 0
    time: float = 0.0
    accumulated_score: float = 0.0        # simulated points so far
    on_front_page: bool = False
    peak_velocity: float = 0.0            # max points/step seen (drives front-page transition)
    social_proof: float = 0.0             # rises with score; feeds bandwagon upvoting
    novelty: float = 1.0                  # decays each step (fatigue)
    exposure_pool: float = 0.0            # current size of the audience being exposed
    segments: list[SegmentAgent] = field(default_factory=list)
    context: dict = field(default_factory=dict)     # topic salience, domain reputation, drift
    accumulated_outcomes: list = field(default_factory=list)   # (timestep, event_type, delta_score)
    probability_weight: float = 1.0
    reactions: list = field(default_factory=list)   # Reaction objects, for diagnostics

    def snapshot(self) -> dict:
        return {"t": self.timestep, "score": round(self.accumulated_score, 2),
                "front_page": self.on_front_page, "social_proof": round(self.social_proof, 3),
                "novelty": round(self.novelty, 3), "exposure": round(self.exposure_pool, 1)}
