"""Phase 8 completion — transactional production storage (Part 6).

The base Phase-8 log was append-only JSONL — portable and fine for tests, but a single unsynchronized file
is not a production backend (no atomic checkpoint commit, no multi-process safety, no transactional
rollback). This module adds a typed storage interface and a transactional **SQLite (WAL)** backend, while
keeping JSONL as a portable/testing backend.

    PersistentStorageBackend  (typed interface)
      ├── JsonlBackend    portable, testing — append-only events + sidecar checkpoints
      └── SqliteBackend   PRODUCTION — WAL journal, atomic event append (idempotent), atomic checkpoint
                          commit, transactional rollback, multi-process writers, concurrent readers, crash
                          recovery (WAL), integrity verification, compaction, bounded checkpoint growth.

Determinism + integrity are preserved: events carry the same content-hash id and running watermark; the
event↔checkpoint watermark link is stored so a mismatch is detectable.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


class StorageError(Exception):
    pass


class PersistentStorageBackend:
    """Typed durable-storage interface. An ``EventLog``/``PersistentStore`` talks only to this."""

    def append_event(self, event_dict: dict, watermark: str) -> bool:
        """Atomically persist one event with the running watermark. Returns True if newly stored, False if a
        row with this event_id already exists (idempotent — retry-safe)."""
        raise NotImplementedError

    def load_events(self) -> list:
        """Return all stored event dicts in insertion (seq) order — for reconstructing the in-memory index."""
        raise NotImplementedError

    def commit_checkpoint(self, cp_dict: dict) -> str:
        """Atomically commit a checkpoint (all-or-nothing). Returns the stored checkpoint id."""
        raise NotImplementedError

    def latest_checkpoint(self, as_of: float | None = None) -> dict | None:
        """Return the most recent checkpoint with checkpoint.as_of ≤ as_of (or the latest if None)."""
        raise NotImplementedError

    def verify_integrity(self) -> dict:
        raise NotImplementedError

    def compact(self) -> dict:
        return {"compacted": False, "note": "no-op for this backend"}

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ JSONL (portable / testing)
@dataclass
class JsonlBackend(PersistentStorageBackend):
    events_path: str
    checkpoints_path: str = ""

    def __post_init__(self):
        if not self.checkpoints_path:
            self.checkpoints_path = str(Path(self.events_path).with_suffix(".checkpoints.jsonl"))
        self._seen = set()
        p = Path(self.events_path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    self._seen.add(json.loads(line).get("event_id"))

    def append_event(self, event_dict, watermark):
        if event_dict.get("event_id") in self._seen:
            return False
        p = Path(self.events_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps({**event_dict, "_watermark": watermark}, sort_keys=True, default=str) + "\n")
        self._seen.add(event_dict.get("event_id"))
        return True

    def load_events(self):
        p = Path(self.events_path)
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                d.pop("_watermark", None)
                out.append(d)
        return out

    def commit_checkpoint(self, cp_dict):
        p = Path(self.checkpoints_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(cp_dict, sort_keys=True, default=str) + "\n")
        return cp_dict.get("integrity_hash", "")

    def latest_checkpoint(self, as_of=None):
        p = Path(self.checkpoints_path)
        if not p.exists():
            return None
        best = None
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            cp = json.loads(line)
            if as_of is not None and cp.get("as_of", 0) > as_of:
                continue
            if best is None or cp.get("as_of", 0) >= best.get("as_of", 0):
                best = cp
        return best

    def verify_integrity(self):
        return {"ok": True, "backend": "jsonl", "n_events": len(self._seen)}


# ------------------------------------------------------------------ SQLite WAL (production)
class SqliteBackend(PersistentStorageBackend):
    """Transactional production backend. WAL journaling gives concurrent readers + a single writer per
    connection while multiple processes may open their own connections; ``busy_timeout`` serializes
    contending writers. Event append is ``INSERT OR IGNORE`` on the content-hash primary key (idempotent);
    checkpoint commit is a single transaction (atomic, rollback-safe). SQLite's WAL provides crash recovery
    automatically."""

    def __init__(self, db_path: str, *, busy_timeout_ms: int = 5000):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._busy = busy_timeout_ms
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=self._busy / 1000.0, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute(f"PRAGMA busy_timeout={self._busy}")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    def _init_schema(self):
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, seq INTEGER, event_time REAL, observed_time REAL,
                actor_ids TEXT, kind TEXT, watermark TEXT, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(observed_time);
            CREATE INDEX IF NOT EXISTS idx_events_seq ON events(seq);
            CREATE TABLE IF NOT EXISTS checkpoints (
                integrity_hash TEXT PRIMARY KEY, as_of REAL, event_watermark TEXT, schema_version TEXT,
                created_seq INTEGER, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_cp_asof ON checkpoints(as_of);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        """)

    def append_event(self, event_dict, watermark):
        c = self._conn()
        cur = c.execute(
            "INSERT OR IGNORE INTO events(event_id, seq, event_time, observed_time, actor_ids, kind, "
            "watermark, payload) VALUES (?,?,?,?,?,?,?,?)",
            (event_dict["event_id"], event_dict.get("seq", 0), float(event_dict.get("event_time", 0.0)),
             float(event_dict.get("observed_time", 0.0)), json.dumps(event_dict.get("actor_ids", [])),
             event_dict.get("kind", "observation"), watermark,
             json.dumps(event_dict, sort_keys=True, default=str)))
        return cur.rowcount > 0                                  # 0 if IGNORE-d (duplicate)

    def load_events(self):
        c = self._conn()
        rows = c.execute("SELECT payload FROM events ORDER BY seq ASC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def commit_checkpoint(self, cp_dict):
        c = self._conn()
        try:
            c.execute("BEGIN")
            c.execute(
                "INSERT OR REPLACE INTO checkpoints(integrity_hash, as_of, event_watermark, schema_version, "
                "created_seq, payload) VALUES (?,?,?,?,?,?)",
                (cp_dict.get("integrity_hash", ""), float(cp_dict.get("as_of", 0.0)),
                 cp_dict.get("event_watermark", ""), cp_dict.get("schema_version", ""),
                 int(cp_dict.get("_seq", 0)), json.dumps(cp_dict, sort_keys=True, default=str)))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")                                # transactional rollback on failure
            raise
        return cp_dict.get("integrity_hash", "")

    def latest_checkpoint(self, as_of=None):
        c = self._conn()
        if as_of is None:
            row = c.execute("SELECT payload FROM checkpoints ORDER BY as_of DESC, created_seq DESC "
                            "LIMIT 1").fetchone()
        else:
            row = c.execute("SELECT payload FROM checkpoints WHERE as_of <= ? ORDER BY as_of DESC, "
                            "created_seq DESC LIMIT 1", (float(as_of),)).fetchone()
        return json.loads(row[0]) if row else None

    def verify_integrity(self):
        """Recompute the running watermark chain over stored events and confirm it matches the last stored
        watermark, and that the latest checkpoint's watermark corresponds to a real event row (or genesis)."""
        import hashlib
        c = self._conn()
        rows = c.execute("SELECT event_id, watermark FROM events ORDER BY seq ASC").fetchall()
        chain = ""
        ok = True
        for eid, wm in rows:
            chain = hashlib.sha256((chain + "|" + eid).encode()).hexdigest()[:24]
            if wm and wm != chain:
                ok = False
        cp = c.execute("SELECT event_watermark FROM checkpoints ORDER BY created_seq DESC LIMIT 1").fetchone()
        cp_wm = cp[0] if cp else ""
        cp_ok = (not cp_wm) or cp_wm == chain or any(w == cp_wm for _, w in rows)
        return {"ok": ok and cp_ok, "backend": "sqlite", "n_events": len(rows),
                "recomputed_watermark": chain, "latest_checkpoint_watermark": cp_wm,
                "event_checkpoint_watermark_consistent": cp_ok}

    def compact(self, *, keep_checkpoints: int = 5):
        """Bound checkpoint growth (keep the newest N) and VACUUM. The event log is never pruned (source of
        truth); only redundant intermediate checkpoints are trimmed — lineage of the survivors is preserved."""
        c = self._conn()
        n_before = c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        c.execute("DELETE FROM checkpoints WHERE integrity_hash NOT IN "
                  "(SELECT integrity_hash FROM checkpoints ORDER BY created_seq DESC LIMIT ?)",
                  (keep_checkpoints,))
        c.execute("VACUUM")
        n_after = c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        return {"compacted": True, "checkpoints_before": n_before, "checkpoints_after": n_after,
                "events_preserved": True, "db_size_bytes": Path(self.db_path).stat().st_size}

    def stats(self) -> dict:
        c = self._conn()
        ne = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        nc = c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        return {"n_events": ne, "n_checkpoints": nc,
                "db_size_bytes": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0}

    def close(self):
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None
