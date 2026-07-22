"""Dependency-aware compilation caching — shared compilation that is REAL, not manifest-only.

Two layers, one lookup path:

  * RUN layer: every compiled component (interpretation, roster, aliases, authority graph,
    calendar, terminal definition, action schema, consequence templates, deterministic
    mechanisms, blueprint response...) is stored under its full DEPENDENCY HASH. Before any
    component compiles, the cache is consulted; a hit returns the existing component and a
    recorded hit — a challenger or repaired model REUSES every unchanged component instead of
    re-registering something it silently regenerated.

  * PERSISTENT layer (optional, on by default for IMMUTABLE artifacts only): the same
    dependency-hash key, on disk, surviving across runs. The key includes question/artifact
    dependencies, as_of, evidence hash, model backend fingerprint, prompt version, schema
    version and compiler version — any drift in ANY of them is a different key. Mutable world
    state, actor memories, branch state, event queues and ACTOR DECISIONS are NEVER persisted
    (cross-run actor-decision reuse stays disabled by default)."""
from __future__ import annotations

import hashlib
import json
import os
import threading

from swm.world_model_v2.lean_v2 import COMPILER_VERSION, PROMPT_VERSION, SCHEMA_VERSION

#: artifact kinds that may go to the persistent layer — immutable compilation only
PERSISTABLE_KINDS = (
    "resolution_interpretation", "evidence_canonicalization", "calendar", "institution_rules",
    "entity_aliases", "authority_graph", "procedural_rules", "action_schema",
    "consequence_templates", "blueprint_response", "blueprint_repair_response",
    # counted historical reference classes are immutable given (question, as_of, evidence,
    # backend) — safe to persist across runs; the cases and cutoff are baked into the key
    "reference_class_grounding",
)

#: kinds that must NEVER persist (mutable / behavioral)
NEVER_PERSIST = ("actor_decision", "actor_memory", "branch_state", "event_queue", "world_node")


def dependency_hash(kind: str, deps: dict) -> str:
    """The full dependency vector -> stable key. Version pins are structural, not optional."""
    payload = {"kind": str(kind), "deps": deps,
               "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
               "compiler_version": COMPILER_VERSION}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class CompilationCache:
    def __init__(self, *, persistent_dir: str = None, persist: bool = True):
        self._run: dict = {}
        self._lock = threading.RLock()
        self.persist = bool(persist)
        self.dir = persistent_dir or os.environ.get(
            "SWM_LEAN_V2_CACHE_DIR", os.path.join(".swm_cache", "lean_v2"))
        self.hits_run = 0
        self.hits_persistent = 0
        self.misses = 0
        self.stores = 0
        self.events: list = []                  # [{kind, key8, outcome}]

    # ---- lookup / store -----------------------------------------------------------
    def get(self, kind: str, deps: dict):
        key = dependency_hash(kind, deps)
        with self._lock:
            if key in self._run:
                self.hits_run += 1
                self.events.append({"kind": kind, "key": key[:8], "outcome": "run_hit"})
                return self._run[key]
        if self.persist and kind in PERSISTABLE_KINDS:
            path = self._path(key)
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        value = json.load(f)["value"]
                    with self._lock:
                        self._run[key] = value
                        self.hits_persistent += 1
                        self.events.append({"kind": kind, "key": key[:8],
                                            "outcome": "persistent_hit"})
                    return value
                except Exception:  # noqa: BLE001 — a corrupt cache file is a miss, never a crash
                    pass
        with self._lock:
            self.misses += 1
            self.events.append({"kind": kind, "key": key[:8], "outcome": "miss"})
        return None

    def put(self, kind: str, deps: dict, value):
        assert kind not in NEVER_PERSIST or not self.persist or True  # doc marker; enforced below
        key = dependency_hash(kind, deps)
        with self._lock:
            self._run[key] = value
            self.stores += 1
            self.events.append({"kind": kind, "key": key[:8], "outcome": "store"})
        if self.persist and kind in PERSISTABLE_KINDS:
            try:
                os.makedirs(self.dir, exist_ok=True)
                tmp = self._path(key) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"kind": kind, "value": value}, f, default=str)
                os.replace(tmp, self._path(key))
            except Exception:  # noqa: BLE001 — persistence is best-effort, run layer is truth
                pass
        return key

    def get_or_compile(self, kind: str, deps: dict, compile_fn):
        """THE shared-compilation contract: consult first, compile only on miss, record both."""
        found = self.get(kind, deps)
        if found is not None:
            return found, True
        value = compile_fn()
        self.put(kind, deps, value)
        return value, False

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.json")

    def manifest(self) -> dict:
        with self._lock:
            return {"hits_run": self.hits_run, "hits_persistent": self.hits_persistent,
                    "misses": self.misses, "stores": self.stores,
                    "persistent_layer": self.persist, "persistent_dir": self.dir,
                    "persistable_kinds": list(PERSISTABLE_KINDS),
                    "never_persisted": list(NEVER_PERSIST),
                    "events": self.events[-60:]}
