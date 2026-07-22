"""Resume helpers: locate the latest checkpoint in a run directory."""
from __future__ import annotations

from pathlib import Path


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    pointer = run_dir / "latest"
    if pointer.exists():
        cand = run_dir / pointer.read_text().strip()
        if cand.exists():
            return cand
    ckpts = sorted(run_dir.glob("checkpoint-*"))
    return ckpts[-1] if ckpts else None


def resume_step(run_dir: str | Path) -> int:
    ck = latest_checkpoint(run_dir)
    if ck is None:
        return 0
    import json
    st = json.loads((ck / "trainer_state.json").read_text())
    return int(st.get("step", 0))
