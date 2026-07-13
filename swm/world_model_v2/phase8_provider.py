"""Phase 8 — automatic persistence-context construction (the final usability gap).

Callers of the canonical `pipeline.simulate()` should not have to hand-build a `PersistenceContext`. This
module provides a typed factory that resolves identity, constructs/retrieves the right transactional store,
loads compatible history/checkpoints, and returns a ready context — or degrades honestly (never abstains)
when identity or storage is unavailable.

Not a hidden global singleton: `PersistenceContextProvider` is an ordinary object the pipeline constructs
from environment config (or the caller injects). Durable state lives in the backend, never in the provider.
Configuration:
  * SWM_PERSISTENCE_BACKEND = memory (default) | sqlite | jsonl
  * SWM_PERSISTENCE_DIR     = directory for sqlite/jsonl files (required for those backends)
Tests inject `backend_kind="memory"` or a temp `db_dir` so they never touch a production database.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from swm.world_model_v2.phase8_filtering import DecayedBetaBernoulliFilter
from swm.world_model_v2.phase8_identity import IdentityResolver, scenario_id, world_id


def _default_filter_registrar(store):
    """Register the broad-prior engagement filter (the empirically-supported default family). Additional
    families are registered by the caller/environment; absence of a filter simply means that family is not
    replayed — never an abstention."""
    store.register_filter("engagement_propensity", lambda k, obs, ao, sd: DecayedBetaBernoulliFilter(
        key=k, prior_mean=0.2, prior_strength=8.0, decay=0.85).filter(
            [(e, 1 if o else 0, t) for e, o, t in obs], as_of=ao))


@dataclass
class PersistenceContextProvider:
    """Builds a `PersistenceContext` automatically for a canonical request. Caches stores per
    (world, scenario) within one provider instance (so repeated calls in a process reuse the same durable
    store), but the authoritative state is the backend, not this object."""
    backend_kind: str = ""                   # memory | sqlite | jsonl ("" → env → memory)
    db_dir: str = ""                         # for sqlite/jsonl
    filter_registrar: object = None
    resolver: IdentityResolver = field(default_factory=IdentityResolver)
    _stores: dict = field(default_factory=dict)
    _lock: object = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self):
        self.backend_kind = self.backend_kind or os.environ.get("SWM_PERSISTENCE_BACKEND", "memory")
        self.db_dir = self.db_dir or os.environ.get("SWM_PERSISTENCE_DIR", "")
        self.filter_registrar = self.filter_registrar or _default_filter_registrar

    def for_request(self, question: str, as_of: str, *, actor_tokens=None):
        """Return (PersistenceContext | None, meta). When actor_tokens are empty (anonymous/stateless) or
        storage is unavailable, returns (None, meta) so the pipeline runs the ordinary non-persistent path —
        an honest degradation, never an abstention. Otherwise resolves identity, builds/reuses the store,
        and returns a context ready for `run_with_persistence`."""
        meta = {"provider": type(self).__name__, "backend": self.backend_kind, "degraded": None,
                "world_id": "", "scenario_id": "", "actor_ids": []}
        if not actor_tokens:
            meta["degraded"] = "anonymous_no_durable_identity"       # stateless request → no persistence
            return None, meta
        wid, sid = world_id(question), scenario_id(question, as_of)
        meta["world_id"], meta["scenario_id"] = wid, sid
        # resolve actor identity (probabilistic linkage preserved by the resolver; here we take the top id)
        actor_ids, link_unc = [], {}
        for tok in actor_tokens:
            hyps = self.resolver.resolve(tok)
            actor_ids.append(hyps[0].canonical_id)
            if len(hyps) > 1:
                link_unc[tok] = self.resolver.link_uncertainty(tok)
        meta["actor_ids"] = actor_ids
        if link_unc:
            meta["identity_link_uncertainty"] = link_unc
        try:
            store = self._get_store(wid, sid)
        except Exception as e:  # noqa: BLE001 — storage unavailable must DEGRADE, not crash/abstain
            meta["degraded"] = f"storage_unavailable: {type(e).__name__}: {e}"[:160]
            return None, meta
        from swm.world_model_v2.phase8_pipeline import PersistenceContext
        return PersistenceContext(store=store), meta

    def _get_store(self, wid, sid):
        from swm.world_model_v2.phase8_events import EventLog
        from swm.world_model_v2.phase8_service import PersistentStore
        key = (wid, sid)
        with self._lock:
            if key in self._stores:
                return self._stores[key]
            backend = self._make_backend(wid, sid)
            log = EventLog(wid, sid, backend=backend)
            store = PersistentStore(wid, sid, log=log)
            self.filter_registrar(store)
            self._stores[key] = store
            return store

    def _make_backend(self, wid, sid):
        if self.backend_kind == "memory":
            return None                                              # in-memory EventLog (no durable file)
        if not self.db_dir:
            raise RuntimeError(f"backend {self.backend_kind!r} requires SWM_PERSISTENCE_DIR / db_dir")
        base = os.path.join(self.db_dir, f"{wid}_{sid}")
        if self.backend_kind == "sqlite":
            from swm.world_model_v2.phase8_storage import SqliteBackend
            return SqliteBackend(base + ".db")
        if self.backend_kind == "jsonl":
            from swm.world_model_v2.phase8_storage import JsonlBackend
            return JsonlBackend(base + ".jsonl")
        raise RuntimeError(f"unknown persistence backend {self.backend_kind!r}")


def build_provider(**overrides) -> PersistenceContextProvider:
    """Construct a provider from environment config (or explicit overrides). Called by the pipeline per
    request when the caller supplies no explicit provider/context — no module-level mutable singleton."""
    return PersistenceContextProvider(**overrides)
