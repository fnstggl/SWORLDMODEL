"""Continuous-time FIRST-PASSAGE hazard scheduling (§15–§16 of the event-driven temporal
architecture) — replaces visible evenly-spaced `hazard_round` grids.

The mathematically correct treatment of a stochastic process with a state-dependent rate:

  1. define the hazard intensity λ(t, state) (per-day), possibly piecewise over lifetime
     fractions (the fitted family curve) and modulated by live state;
  2. sample ONE persistent threshold  E ~ Exp(1)  per (branch-particle, process) — seeded from
     the PARTICLE ROOT so matched counterfactual arms share it (§16, invariants 27/35);
  3. integrate the cumulative hazard  Λ(t) = ∫ λ  over REAL elapsed time;
  4. the event occurs when Λ crosses E — the projected crossing time is scheduled as a real
     queue event;
  5. when observable state changes the rate, the accumulated Λ and the threshold are PRESERVED
     and only the remaining time is re-projected (never a fresh redraw under the new rate —
     redraw bias is exactly the artifact this module removes);
  6. numerical integration, where needed, is INTERNAL: it never creates world events, never
     triggers actor decisions, and records its error bounds.

Calibration identity: a per-particle target absorbed-mass T over the window corresponds to
total cumulative hazard  Λ_total = −ln(1−T); spreading Λ_total across lifetime-fraction buckets
by the fitted family curve's event-mass weights reproduces T exactly at the horizon — the same
mass-conservation contract the old evenly-spaced exponent chains had, now without the grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from swm.world_model_v2.temporal_model import particle_rng

_BUCKETS = 5


def mass_weights_from_curve(curve) -> list:
    """Fitted bucket hazards → normalized per-bucket EVENT-MASS weights (shape of when events
    happen inside the window). Uniform when no curve was fitted."""
    if not curve or len(curve) != _BUCKETS or not any(h > 0 for h in curve):
        return [1.0 / _BUCKETS] * _BUCKETS
    surv, mass = 1.0, []
    for h in curve:
        h = max(0.0, min(0.999999, float(h)))
        mass.append(surv * h)
        surv *= (1.0 - h)
    z = sum(mass) or 1.0
    return [m / z for m in mass]


def rates_from_target_mass(target_mass: float, *, span_s: float, curve=None) -> list:
    """Piecewise-constant per-day intensities over the window's lifetime buckets whose total
    cumulative hazard is exactly −ln(1−T) — first-passage probability by the horizon == T."""
    t = max(0.0, min(0.999999, float(target_mass)))
    lam_total = -math.log(1.0 - t) if t > 0 else 0.0
    weights = mass_weights_from_curve(curve)
    bucket_days = max(1e-9, (span_s / _BUCKETS) / 86400.0)
    return [lam_total * w / bucket_days for w in weights]


def rates_from_bucket_hazards(curve, *, span_s: float) -> list:
    """Discrete per-bucket hazards h_b → continuous per-day rates λ_b = −ln(1−h_b)/bucket_days
    (exact survival-equivalent piecewise-constant intensity)."""
    bucket_days = max(1e-9, (span_s / _BUCKETS) / 86400.0)
    out = []
    for h in (curve or [0.0] * _BUCKETS):
        h = max(0.0, min(0.999999, float(h)))
        out.append(-math.log(1.0 - h) / bucket_days)
    while len(out) < _BUCKETS:
        out.append(out[-1] if out else 0.0)
    return out[:_BUCKETS]


@dataclass
class CumulativeHazardState:
    """One branch's live first-passage state for one stochastic process.

    Persisted on the world (`world.temporal_hazards[process_id]`); deep-copied with the world on
    cloning, so a counterfactual arm inherits the SAME threshold and the SAME accumulated hazard
    at the branch point — arms diverge only where an action causally changes the modulation.

      threshold      the branch's Exp(1) draw (particle-rooted; never resampled)
      accumulated    Λ integrated so far over real elapsed time
      last_ts        integration frontier
      base_rates     piecewise per-day intensity over lifetime-fraction buckets
      modulation     live multiplicative factor from consumed state (rate changes preserve Λ)
      generation     bumps on every re-projection; stale scheduled crossings are skipped
    """
    process_id: str
    as_of: float
    horizon_ts: float
    base_rates: list
    threshold: float
    accumulated: float = 0.0
    last_ts: float = None
    modulation: float = 1.0
    generation: int = 0
    fired: bool = False
    reads: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    n_reprojections: int = 0
    integration_error_bound: float = 0.0

    def __post_init__(self):
        if self.last_ts is None:
            self.last_ts = float(self.as_of)

    # -- intensity ---------------------------------------------------------------------------
    def _bucket_edges(self):
        span = max(1e-9, self.horizon_ts - self.as_of)
        return [self.as_of + span * b / _BUCKETS for b in range(_BUCKETS + 1)]

    def rate_at(self, ts: float) -> float:
        span = max(1e-9, self.horizon_ts - self.as_of)
        frac = min(0.999999, max(0.0, (ts - self.as_of) / span))
        b = min(int(frac * _BUCKETS), _BUCKETS - 1)
        return max(0.0, float(self.base_rates[b])) * max(0.0, self.modulation)

    # -- exact piecewise integration ---------------------------------------------------------
    def _integral(self, t0: float, t1: float) -> float:
        """∫ λ dt over [t0, t1] — exact for the piecewise-constant intensity (no numerical
        error; integration_error_bound stays 0 unless a non-piecewise form is added)."""
        if t1 <= t0:
            return 0.0
        edges = self._bucket_edges()
        total = 0.0
        for b in range(_BUCKETS):
            lo, hi = edges[b], edges[b + 1]
            a, c = max(t0, lo), min(t1, hi)
            if c > a:
                total += max(0.0, float(self.base_rates[b])) * ((c - a) / 86400.0)
        # beyond-horizon exposure keeps the last bucket's rate (censoring handled by caller)
        if t1 > edges[-1]:
            total += max(0.0, float(self.base_rates[-1])) * ((t1 - max(t0, edges[-1])) / 86400.0)
        return total * max(0.0, self.modulation)

    def accumulate_to(self, ts: float) -> float:
        """Advance the integration frontier to `ts`, preserving accumulated hazard."""
        ts = float(ts)
        if ts > self.last_ts:
            self.accumulated += self._integral(self.last_ts, ts)
            self.last_ts = ts
        return self.accumulated

    # -- projection --------------------------------------------------------------------------
    def project_crossing(self) -> float:
        """The timestamp where Λ first reaches the threshold, from the current frontier under
        the CURRENT modulation — or None if the crossing lies beyond the horizon. Solves the
        piecewise-constant integral exactly (inverse-CDF per bucket)."""
        if self.fired:
            return None
        need = self.threshold - self.accumulated
        if need <= 0:
            return self.last_ts
        t = self.last_ts
        edges = self._bucket_edges()
        for b in range(_BUCKETS):
            lo, hi = edges[b], edges[b + 1]
            a = max(t, lo)
            if a >= hi:
                continue
            rate = max(0.0, float(self.base_rates[b])) * max(0.0, self.modulation)
            if rate <= 0:
                continue
            seg = rate * ((hi - a) / 86400.0)
            if seg >= need:
                return a + (need / rate) * 86400.0
            need -= seg
        return None                                        # censored beyond horizon

    def on_state_change(self, now_ts: float, new_modulation: float) -> float:
        """A declared read field changed: integrate up to now under the OLD rate, keep Λ and
        the threshold, switch modulation, and return the re-projected crossing (or None).
        This is the §16 contract — no resampling bias, matched streams preserved."""
        self.accumulate_to(now_ts)
        self.modulation = max(0.0, float(new_modulation))
        self.generation += 1
        self.n_reprojections += 1
        return self.project_crossing()


def ensure_hazard_state(world, process_id: str, *, as_of: float, horizon_ts: float,
                        base_rates: list, reads=(), payload=None) -> CumulativeHazardState:
    """Get-or-create the branch's first-passage state for one process. The threshold seeds from
    the PARTICLE ROOT (temporal_model.particle_rng), so `b3` and `b3:armA` share it."""
    store = getattr(world, "temporal_hazards", None)
    if store is None:
        store = {}
        world.temporal_hazards = store
    st = store.get(process_id)
    if st is None:
        rng = particle_rng(world, f"hazard_threshold:{process_id}")
        st = CumulativeHazardState(process_id=process_id, as_of=float(as_of),
                                   horizon_ts=float(horizon_ts),
                                   base_rates=[max(0.0, float(r)) for r in base_rates],
                                   threshold=rng.expovariate(1.0),
                                   last_ts=float(max(as_of, getattr(world.clock, "now", as_of))),
                                   reads=list(reads), payload=dict(payload or {}))
        store[process_id] = st
    return st


def schedule_crossing(queue, world, st: CumulativeHazardState, *, etype: str,
                      participants=(), extra_payload=None):
    """Schedule (or reschedule) the process's projected first-passage crossing as a real queue
    event. The event carries the process id + generation; the runtime drops stale generations
    on pop, so a re-projection never double-fires."""
    from swm.world_model_v2.events import Event
    ts = st.project_crossing()
    if ts is None or ts > st.horizon_ts:
        return None
    payload = {**st.payload, **(extra_payload or {}),
               "hazard_process_id": st.process_id, "hazard_generation": st.generation}
    ev = Event(ts=max(float(ts), float(world.clock.now)), etype=etype,
               participants=list(participants), payload=payload,
               source=f"first_passage:{st.process_id}")
    queue.schedule(ev)
    return ev


def crossing_is_current(world, event) -> bool:
    """True iff a popped first-passage event still matches its process's live generation (a
    re-projection bumps the generation, invalidating previously scheduled crossings)."""
    pid = (event.payload or {}).get("hazard_process_id")
    if not pid:
        return True
    st = (getattr(world, "temporal_hazards", None) or {}).get(pid)
    if st is None or st.fired:
        return False
    return int((event.payload or {}).get("hazard_generation", -1)) == int(st.generation)


# ---------------------------------------------------------------- modulation hooks
#: process-family modulation functions: name -> fn(world, state) -> factor. Registered by the
#: owning module (event_time registers "event_time_mode"); the runtime resolves through here so
#: hazard semantics stay with the mechanism that owns them.
MODULATION_HOOKS: dict = {}


def register_modulation_hook(name: str, fn):
    MODULATION_HOOKS[str(name)] = fn
    return name


def resolve_modulation(world, st) -> float:
    """The process's LIVE modulation: its registered hook when named, else the generic bounded
    consume-state factor (×2 per full weight at an extreme state, clamped [0.25, 4])."""
    hook = MODULATION_HOOKS.get(str((st.payload or {}).get("modulation_hook", "")))
    if hook is not None:
        try:
            return max(0.0, float(hook(world, st)))
        except Exception:  # noqa: BLE001 — a broken hook must not kill the branch
            return st.modulation
    consume = (st.payload or {}).get("consume") or []
    logf, used = 0.0, 0
    for m in consume:
        var, w = str(m.get("var", "")), float(m.get("weight", 0.0) or 0.0)
        q = world.quantities.get(var)
        v = getattr(q, "value", None)
        if w <= 0.0 or not isinstance(v, (int, float)):
            continue
        v = max(0.0, min(1.0, float(v)))
        if m.get("invert"):
            v = 1.0 - v
        logf += w * (v - 0.5) * 2.0 * math.log(2.0)
        used += 1
    base = float((st.payload or {}).get("hr_factor", 1.0))
    return base * (max(0.25, min(4.0, math.exp(logf))) if used else 1.0)
