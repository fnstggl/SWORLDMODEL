"""Small driver to acquire a batch of datasets (used for the initial real acquisition).

Usage: python -m machine_learning.scripts.acquire_batch id1 id2 ...
Prints a concise per-dataset outcome. Safe to re-run (resumable).
"""
from __future__ import annotations

import sys

from machine_learning.acquisition.download import acquire
from machine_learning.io_utils import human_bytes

TS = "2026-07-22T00:00:00Z"  # fixed for deterministic manifests in this run


def main(ids: list[str]) -> None:
    for did in ids:
        try:
            man = acquire(did, timestamp=TS)
            print(f"[{did}] status={man['status']} files={len(man.get('files', []))} "
                  f"bytes={human_bytes(man.get('total_bytes', 0))} "
                  f"notes={'; '.join(man.get('notes', [])[-2:])}")
        except Exception as e:  # noqa: BLE001 - one dataset must not abort the batch
            print(f"[{did}] UNEXPECTED ERROR: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main(sys.argv[1:])
