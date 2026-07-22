"""Base types + exceptions shared by all source adapters.

An adapter knows how to fetch one dataset from one kind of official source (Hugging Face,
git, or a plain HTTP(S) URL) into a destination directory, returning the list of files it
wrote with sizes + checksums. Adapters must be:

* **resumable** — skip files already present with a matching size/checksum,
* **honest about access** — raise :class:`AccessBlocked` (never retried) for gated/auth
  failures so the orchestrator can classify the dataset ``blocked`` with the exact human
  action required,
* **secret-free** — never put tokens into returned paths, errors, or file contents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class AcquisitionError(RuntimeError):
    """Transient/recoverable acquisition failure (network, 5xx, timeout). Retryable."""


class AccessBlocked(RuntimeError):
    """Access requires a human action (login, gated approval, DUA, application).

    NOT retryable. Carries the exact requirement so the orchestrator can record it.
    """

    def __init__(self, requirement: str, *, requires_token: bool = False):
        self.requirement = requirement
        self.requires_token = requires_token
        super().__init__(requirement)


class SourceUnavailable(RuntimeError):
    """The official source does not (or no longer) publicly hosts the data. NOT retryable."""


@dataclass
class FileRecord:
    path: str          # relative to the dataset raw dir
    sha256: str
    size_bytes: int
    role: str = "data"

    def as_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "role": self.role}


@dataclass
class FetchResult:
    files: list[FileRecord] = field(default_factory=list)
    resume_state: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


# progress callback: (bytes_done, bytes_total_or_None, label)
ProgressCB = Callable[[int, "int | None", str], None]


def noop_progress(done: int, total: int | None, label: str) -> None:  # pragma: no cover
    pass


class Adapter:
    """Adapter interface. Concrete adapters implement :meth:`fetch` and :meth:`estimate`."""

    name = "base"

    def estimate(self, spec: dict) -> int | None:
        """Best-effort estimate of total download bytes, or None if unknown."""
        return None

    def fetch(self, spec: dict, dest: Path, *, token: str | None = None,
              max_bytes: int | None = None, progress: ProgressCB = noop_progress,
              limit: int | None = None) -> FetchResult:
        raise NotImplementedError
