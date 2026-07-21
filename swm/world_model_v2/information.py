"""Information state — Phase 1.7. Different actors KNOW different things; that difference is data.

The ledger holds typed information items (public / private / latent-unknown / misinformation) with source
credibility and arrival times, and per-actor EXPOSURE records: whether this actor actually observed the item,
when, through which edge/channel. `visible_to(actor)` is the ONLY way an agent-decision or belief-update
operator reads information — no universal dossier. Beliefs-about-beliefs are ordinary typed items whose
`about` targets another actor's belief. Memory/forgetting: exposures carry salience that background time
transitions decay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.world_model_v2.state import Provenance


@dataclass
class InformationItem:
    item_id: str
    content: str                          # the proposition (typed where possible: short claim text)
    kind: str = "public"                  # public | private | latent_unknown | misinformation
    truth: object = None                  # True/False/None(unknown) — ground truth IN THE WORLD, if defined
    source: str = ""                      # originating entity/outlet
    credibility: float = 0.6              # source credibility (belief-update likelihood input)
    created_at: float = 0.0
    about: str = ""                       # optional: entity/belief this is about (beliefs-about-beliefs)
    prov: Provenance = field(default_factory=Provenance)


@dataclass
class Exposure:
    actor_id: str
    item_id: str
    at: float                             # when this actor actually observed it
    channel: str = ""                     # edge/channel it arrived through
    salience: float = 0.6                 # decays with time (memory)
    observed: bool = True                 # False = delivered but not yet seen (inbox != read)


@dataclass
class InformationLedger:
    items: dict = field(default_factory=dict)      # item_id -> InformationItem
    exposures: list = field(default_factory=list)  # [Exposure]

    def publish(self, item: InformationItem):
        self.items[item.item_id] = item
        return item

    def expose(self, actor_id: str, item_id: str, at: float, *, channel="", salience=0.6,
               observed=True) -> Exposure:
        if item_id not in self.items:
            raise KeyError(f"unknown information item {item_id!r}")
        e = Exposure(actor_id=actor_id, item_id=item_id, at=at, channel=channel,
                     salience=salience, observed=observed)
        self.exposures.append(e)
        return e

    def visible_to(self, actor_id: str, *, at: float = None, observed_only=True) -> list:
        """THE actor-specific information set: items this actor has actually been exposed to (by `at`)."""
        out = []
        for e in self.exposures:
            if e.actor_id != actor_id or (observed_only and not e.observed):
                continue
            if at is not None and e.at > at:
                continue
            it = self.items.get(e.item_id)
            if it is not None:
                out.append((it, e))
        return out

    def unseen_by(self, actor_id: str) -> list:
        seen = {e.item_id for e in self.exposures if e.actor_id == actor_id and e.observed}
        return [it for iid, it in self.items.items() if iid not in seen]

    def decay(self, elapsed_days: float, *, half_life_days: float = 10.0):
        """Memory: exposure salience decays exponentially over THIS elapsed interval (called once per
        background time step — idempotent per interval, not per total age)."""
        f = 0.5 ** (max(0.0, elapsed_days) / max(0.1, half_life_days))
        for e in self.exposures:
            e.salience *= f
