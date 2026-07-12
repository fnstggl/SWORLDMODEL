"""Universal typed world state — Phase 1. The world is data with provenance, not prose.

`StateField` is the atom: a value OR a distribution, with observed/inferred/assumed/sampled status, source
references, confidence, timestamp, inference method, dependencies, calibration status. `Entity` composes
fields for persons/institutions under the universal schema — the compiler populates only the causally-relevant
subset (sparsity is a feature, not laziness). Scenario-specific typed extensions go through a controlled
registry (never arbitrary untyped keys as the route to generality). `WorldState` is the shared world every
mechanism reads and writes: entities + populations + network + institutions + quantities + information +
event queue, stamped with identity, branch, RFC3339 time, as-of cutoff, parentage, evidence hash, versions.
"""
from __future__ import annotations

import copy
import hashlib
import time as _time
from dataclasses import dataclass, field

STATUSES = ("observed", "inferred", "assumed", "sampled", "derived")


def rfc3339(ts: float) -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(ts))


def parse_time(s) -> float:
    """RFC3339 or YYYY-MM-DD → unix ts. Raises on garbage — time is load-bearing here."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _time.mktime(_time.strptime(s[:len(_time.strftime(fmt))], fmt)) - _time.timezone
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {s!r}")


@dataclass
class Provenance:
    """The full StateEstimate envelope. Status SEMANTICS affect execution, not just labeling:
    initialization samples only non-observed fields; particle sampling never perturbs `observed`; assimilation
    may overwrite inferred/assumed but must version observed; sensitivity analysis targets sampled/inferred;
    abstention triggers when high-sensitivity fields sit at `assumed` with low confidence."""
    status: str = "assumed"               # observed | derived | inferred | assumed | sampled
    sources: list = field(default_factory=list)    # evidence references (citations, dataset ids)
    confidence: float = 0.5
    method: str = ""                      # how it was established (rule id, prompt hash, dataset, sensor)
    updated_at: float = 0.0               # unix ts of last update
    dependencies: list = field(default_factory=list)  # other field paths this was derived from
    calibrated: bool = False              # does this field's estimator have empirical calibration?
    uncertainty_type: str = ""            # one of uncertainty.UNCERTAINTY_TYPES (where classified)
    valid_from: float = 0.0               # temporal validity window
    valid_until: float = 0.0              # 0.0 = open-ended
    sensitivity: float = 0.5              # estimated outcome sensitivity (fidelity/abstention input)
    lineage: list = field(default_factory=list)   # prior (method, updated_at) entries — the field's history

    def as_dict(self):
        return {"status": self.status, "sources": self.sources[:4], "confidence": round(self.confidence, 3),
                "method": self.method, "updated_at": rfc3339(self.updated_at) if self.updated_at else "",
                "dependencies": self.dependencies, "calibrated": self.calibrated,
                "uncertainty_type": self.uncertainty_type, "sensitivity": self.sensitivity,
                "lineage": self.lineage[-3:]}


@dataclass
class StateField:
    """Value OR distribution, never both implicit. `dist` is {value: prob} for discrete or
    {"mean":…, "sd":…, "lo":…, "hi":…} for continuous. `value` is the point/sampled realization."""
    value: object = None
    dist: dict = None
    prov: Provenance = field(default_factory=Provenance)

    def is_uncertain(self) -> bool:
        return self.dist is not None

    def sample(self, rng):
        """Draw a realization; marks nothing (sampling happens at particle construction, which stamps prov)."""
        if self.dist is None:
            return self.value
        if "mean" in self.dist:
            v = rng.gauss(float(self.dist["mean"]), float(self.dist.get("sd", 0.0)))
            lo, hi = self.dist.get("lo"), self.dist.get("hi")
            if lo is not None:
                v = max(float(lo), v)
            if hi is not None:
                v = min(float(hi), v)
            return v
        opts = list(self.dist.items())
        r, acc = rng.random() * sum(p for _, p in opts), 0.0
        for v, p in opts:
            acc += p
            if r <= acc:
                return v
        return opts[-1][0]

    def observed(self):
        return self.prov.status == "observed"


def F(value=None, *, dist=None, status="assumed", sources=None, confidence=0.5, method="",
      updated_at=0.0, calibrated=False) -> StateField:
    """Shorthand constructor used everywhere."""
    return StateField(value=value, dist=dist,
                      prov=Provenance(status=status, sources=sources or [], confidence=confidence,
                                      method=method, updated_at=updated_at, calibrated=calibrated))


# ---------------------------------------------------------------- entities + extension registry
# the UNIVERSAL schema: which fields exist. The compiler populates only the causally-relevant subset.
ENTITY_FIELDS = ("identity", "entity_type", "roles", "goals", "preferences", "beliefs", "resources",
                 "authority", "commitments", "attention", "affect", "private_information", "relationships",
                 "memory", "past_actions", "current_action", "planned_actions", "constraints",
                 "information_set",
                 # typed namespace for compiler-proposed latent scalars whose semantic name is scenario-
                 # specific (e.g. "attention_to_scheduling"): a dict {name: StateField}. Keeps generality
                 # (arbitrary latents) WITHOUT arbitrary top-level untyped keys — every value is a
                 # provenance-stamped StateField, just under a declared namespace.
                 "latent_state")

_ENTITY_EXTENSIONS: dict = {}             # name -> {"fields": {fname: description}, "entity_types": [...]}


def register_entity_extension(name: str, *, fields: dict, entity_types=("person", "institution")):
    """Controlled extension registry: scenario-specific TYPED fields (e.g. sports package adds
    'fitness_level'). Extensions declare their fields up front; arbitrary untyped keys are rejected at
    set-time. Provenance/uncertainty machinery applies to extension fields identically."""
    if not name or not fields:
        raise ValueError("extension needs a name and typed fields")
    _ENTITY_EXTENSIONS[name] = {"fields": dict(fields), "entity_types": tuple(entity_types)}
    return name


def extension_fields(entity_type: str) -> set:
    out = set()
    for ext in _ENTITY_EXTENSIONS.values():
        if entity_type in ext["entity_types"]:
            out |= set(ext["fields"])
    return out


@dataclass
class Entity:
    """A person or institution. Every populated field is a StateField (typed + provenance). Beliefs are a
    dict of {proposition: StateField(True/False/level)}; resources {name: StateField(Quantity-like)};
    commitments a list of typed dicts; relationships live in the world's network (not here)."""
    identity: str
    entity_type: str = "person"           # person | institution
    fields: dict = field(default_factory=dict)     # fname -> StateField | {key: StateField}

    def set(self, fname: str, value, *, key=None):
        allowed = set(ENTITY_FIELDS) | extension_fields(self.entity_type)
        if fname not in allowed:
            raise KeyError(f"{fname!r} is not in the universal schema or a registered extension "
                           f"(register_entity_extension first — no arbitrary untyped keys)")
        sf = value if isinstance(value, StateField) else F(value)
        if key is not None:
            self.fields.setdefault(fname, {})[key] = sf
        else:
            self.fields[fname] = sf
        return sf

    def get(self, fname: str, key=None, default=None):
        v = self.fields.get(fname)
        if v is None:
            return default
        if key is not None:
            return v.get(key, default) if isinstance(v, dict) else default
        return v

    def value(self, fname: str, key=None, default=None):
        sf = self.get(fname, key)
        return sf.value if isinstance(sf, StateField) else default


# ---------------------------------------------------------------- the shared world
@dataclass
class SimulationClock:
    now: float                            # unix ts — REAL calendar time
    as_of: float                          # information cutoff (nothing after this may enter initialization)

    def now_rfc3339(self) -> str:
        return rfc3339(self.now)

    def advance_to(self, ts: float):
        if ts < self.now:
            raise ValueError(f"clock cannot go backward ({rfc3339(ts)} < {self.now_rfc3339()})")
        elapsed = ts - self.now
        self.now = ts
        return elapsed


@dataclass
class WorldState:
    world_id: str
    branch_id: str
    clock: SimulationClock
    entities: dict = field(default_factory=dict)          # id -> Entity
    populations: dict = field(default_factory=dict)       # id -> population.Population
    network: object = None                                # network.RelationGraph
    institutions: dict = field(default_factory=dict)      # id -> institutions.RuleSystem
    quantities: dict = field(default_factory=dict)        # name -> quantities.Quantity
    information: object = None                            # information.InformationLedger
    parent_version: str = ""                              # hash of the parent state (branch lineage)
    evidence_hash: str = ""                               # snapshot hash of the grounding evidence
    versions: dict = field(default_factory=dict)          # {"code": commit, "model": id, "config": hash}
    provenance_note: str = ""
    uncertainty_meta: dict = field(default_factory=dict)  # particle weight, sampled-latent record refs
    omissions: list = field(default_factory=list)         # recorded drops/unsupported elements (loud, not silent)

    def version_hash(self) -> str:
        payload = f"{self.world_id}|{self.branch_id}|{self.clock.now}|{len(self.entities)}|{self.evidence_hash}"
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def clone(self, *, branch_id: str) -> "WorldState":
        """Deep-copy for a counterfactual branch: identical latent world, new branch id, parent recorded."""
        w = copy.deepcopy(self)
        w.parent_version = self.version_hash()
        w.branch_id = branch_id
        return w

    def entity(self, eid: str) -> Entity:
        if eid not in self.entities:
            raise KeyError(f"unknown entity {eid!r} (known: {sorted(self.entities)[:8]})")
        return self.entities[eid]


@dataclass
class WorldBranch:
    branch_id: str
    world: WorldState
    weight: float = 1.0                   # particle weight (assimilation reweights)
    log: list = field(default_factory=list)     # [StateDelta] — the machine-readable history
    terminal: bool = False


@dataclass
class WorldTrajectory:
    trajectory_id: str
    branches: list = field(default_factory=list)   # [WorldBranch]
    seed: int = 0
