"""Shared low-level IO primitives: atomic writes, hashing, JSON/JSONL, retry/backoff.

Kept dependency-free (stdlib only) so every stage — including acquisition before the
heavy libs are installed — can rely on it. These are the building blocks that make the
pipeline *resumable* and *auditable*: every durable write is atomic, every file is
hashable, and every fallible operation retries with bounded exponential backoff and a
structured error record.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

_CHUNK = 1024 * 1024


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------------
# Atomic writes (write to a temp file in the same dir, fsync, then os.replace)
# --------------------------------------------------------------------------------------
def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


# --------------------------------------------------------------------------------------
# JSON / JSONL
# --------------------------------------------------------------------------------------
def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str))
            f.write("\n")
            n += 1
    os.replace(tmp, path)
    return n


# --------------------------------------------------------------------------------------
# Retry with bounded exponential backoff
# --------------------------------------------------------------------------------------
@dataclass
class RetryError:
    attempt: int
    error_type: str
    message: str
    next_action: str


class RetryExhausted(RuntimeError):
    def __init__(self, errors: list[RetryError]):
        self.errors = errors
        last = errors[-1] if errors else None
        super().__init__(f"retry exhausted after {len(errors)} attempts: "
                         f"{last.error_type if last else 'unknown'}")


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[RetryError], None] | None = None,
) -> T:
    """Call ``fn`` with retries. Delays: base_delay * 2**(n-1), capped at max_delay.

    ``give_up_on`` exceptions (e.g. auth/gated errors) are NOT retried — they re-raise
    immediately so the caller can classify the dataset as blocked. Every failed attempt
    produces a :class:`RetryError` (short, secret-free) passed to ``on_error``.
    """
    errors: list[RetryError] = []
    for n in range(1, attempts + 1):
        try:
            return fn()
        except give_up_on:
            raise
        except retry_on as e:  # noqa: BLE001 - deliberate broad-but-bounded catch
            msg = _short(str(e))
            last = n == attempts
            rec = RetryError(
                attempt=n,
                error_type=type(e).__name__,
                message=msg,
                next_action="give up" if last else f"retry in {min(base_delay * 2**(n-1), max_delay):.0f}s",
            )
            errors.append(rec)
            if on_error:
                on_error(rec)
            if last:
                raise RetryExhausted(errors) from e
            sleep(min(base_delay * 2 ** (n - 1), max_delay))
    raise RetryExhausted(errors)  # pragma: no cover


def _short(msg: str, limit: int = 240) -> str:
    msg = " ".join(msg.split())
    return msg if len(msg) <= limit else msg[:limit] + "…"


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
def human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total
