"""Drive a REAL Lean V2 run and capture it as a replayable recording.

This runs the canonical, unmodified consumer path
``unified_runtime.simulate_world(..., execution_profile="lean_v2")`` on a real, sealed
forecasting question, under the transparent :mod:`swm_viz.recorder` observers, then folds the
verbatim capture + provenance into ``swm_viz/recordings/<slug>.json`` via
:func:`swm_viz.build_recording.build_recording`.

The simulation code is never touched. The as-of background is the kind of frozen, time-locked
evidence a product user supplies; it is injected through the runtime's own ``prebuilt_bundle``
sealed-replay port (no live retrieval), so the only external dependency is the LLM backend.

Run:  DEEPSEEK_API_KEY=... python -m swm_viz.record_run [case_id]
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

REC_DIR = Path("swm_viz/recordings")
CACHE_DIR = REC_DIR / "_compile_cache"

MODEL = "deepseek-v4-flash"
# The benchmark arms ran at 3600 to bound cost; that ceiling TRUNCATES the multi-actor
# state-generation JSON, so no private-state variants materialize and the waves have nothing
# to decide. A larger ceiling lets the SAME simulation finish its replies. This is a property
# of the injected backend (like temperature/model), never a change to swm/ simulation code.
MAX_TOKENS = 8000
TEMPERATURE = 0.2
SEED = 0

# ---- bundled real, sealed cases (as-of background = frozen, time-locked evidence) --------
CASES: dict[str, dict] = {
    "banxico": {
        "slug": "banxico_unanimous_2026",
        "title": "Banxico — unanimous June 2026 rate vote?",
        "question": ("Will the Banxico Governing Board's June 25, 2026, interest rate "
                     "decision be unanimous (5-0 vote)?"),
        "as_of": "2026-05-14",
        "horizon": "2026-06-25",
        "resolution_criteria": (
            "Resolves YES if and only if the Bank of Mexico (Banxico) Governing Board's "
            "monetary policy decision announced on June 25, 2026 is a unanimous 5-0 vote "
            "(all five members voting the same way). Any split (e.g. 4-1, 3-2) resolves NO."),
        "background": (
            "On May 7, 2026, the Bank of Mexico (Banxico) Governing Board cut its benchmark "
            "interest rate to 6.50% in a 3-2 split vote. Board members Jonathan Heath and "
            "Irene Espinosa Cantellano dissented, preferring to hold the rate at 6.75%. The "
            "central bank cited a contraction in economic activity and revised its Q2 2026 "
            "inflation forecast upward to 4.1%, while targeting convergence to its 3% goal by "
            "Q2 2027. In its statement the board signaled that its roughly two-year easing "
            "cycle had come to an end and that it would take a more cautious, data-dependent "
            "stance going forward. The five voting members are Governor Victoria Rodriguez "
            "Ceja and Deputy Governors Galia Borja Gomez, Omar Mejia Torres, Jonathan Heath, "
            "and Irene Espinosa Cantellano. The next scheduled monetary policy decision is "
            "June 25, 2026."),
        "note": ("As-of (2026-05-14) background reconstructed from the public, sourced facts "
                 "of the May 7 2026 Banxico decision; injected as sealed-replay evidence."),
    },
}


def _budget() -> dict:
    return {"max_wall_s": 900.0, "max_calls": 160}


def run_case(case_id: str = "banxico", *, case: dict = None) -> Path:
    from experiments.btf3_frozen_bundle import frozen_background_bundle
    from swm.api.deepseek_backend import deepseek_chat_fn
    from swm.world_model_v2.unified_runtime import simulate_world

    from swm_viz.build_recording import build_recording
    from swm_viz.recorder import capture_lean_v2

    case = case or CASES[case_id]
    case_id = case.get("slug", case_id)
    REC_DIR.mkdir(parents=True, exist_ok=True)

    evidence = (f"Resolution criteria: {case['resolution_criteria']}\n\n"
                f"Background (as of {case['as_of']}): {case['background']}")
    as_of_ts = (datetime.fromisoformat(case["as_of"]).replace(tzinfo=timezone.utc).timestamp())
    bundle = frozen_background_bundle(
        case["question"], as_of_ts=as_of_ts, background=case["background"],
        resolution_criteria=case["resolution_criteria"], seed=SEED)

    base_llm = deepseek_chat_fn(MODEL, system="Reply ONLY JSON.", max_tokens=MAX_TOKENS,
                                temperature=TEMPERATURE)

    print(f"[record_run] launching REAL lean_v2 run for '{case_id}' "
          f"(model={MODEL}) — this makes live LLM calls…", flush=True)
    t0 = time.time()
    with capture_lean_v2() as cap:
        res = simulate_world(
            case["question"], llm=base_llm, evidence=evidence,
            as_of=case["as_of"], horizon=case["horizon"], seed=SEED,
            prebuilt_bundle=bundle,
            execution_policy={"lean_v2": {
                "budget": _budget(),
                "backend_fingerprint": MODEL,
                # capture EVERY call live (no cross-run compile cache short-circuit)
                "persistent_cache": False,
                # deterministic, readable ordering for the visualization
                "max_workers": 1}},
            execution_profile="lean_v2")
    wall = round(time.time() - t0, 1)
    print(f"[record_run] run complete in {wall}s — {len(cap.calls)} LLM calls captured, "
          f"status={getattr(res, 'simulation_status', '?')}", flush=True)

    # serialize the compiled world + the full result (with provenance) as plain data
    blueprint_dict = {}
    if cap.blueprints:
        try:
            blueprint_dict = cap.blueprints[0].as_dict()
        except Exception as e:  # noqa: BLE001
            print(f"[record_run] blueprint serialize failed: {e}", flush=True)
    result_dict = _result_to_dict(res)

    meta = {
        "case_id": case_id,
        "slug": case["slug"],
        "title": case["title"],
        "question": case["question"],
        "as_of": case["as_of"],
        "horizon": case["horizon"],
        "resolution_criteria": case["resolution_criteria"],
        "background": case["background"],
        "background_note": case["note"],
        "model": MODEL,
        "profile": "lean_v2",
        "wall_clock_s": wall,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runtime_version": (res.provenance or {}).get("runtime")
        if getattr(res, "provenance", None) else "",
    }
    # dump the raw capture FIRST — a recording can then be rebuilt offline (no sim, no API),
    # so a downstream builder bug never wastes a real run
    raw_dir = REC_DIR / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{case['slug']}.raw.json"
    raw_path.write_text(json.dumps(
        {"meta": meta, "calls": cap.calls, "blueprint_dict": blueprint_dict,
         "result_dict": result_dict}, default=str))
    print(f"[record_run] raw capture saved -> {raw_path} "
          f"({raw_path.stat().st_size // 1024} KB)", flush=True)

    recording = build_recording(meta=meta, calls=cap.calls,
                                blueprint_dict=blueprint_dict, result_dict=result_dict)
    out = REC_DIR / f"{case['slug']}.json"
    out.write_text(json.dumps(recording, indent=1, default=str))
    _update_index(recording, out)
    print(f"[record_run] wrote recording -> {out} "
          f"({out.stat().st_size // 1024} KB, {len(recording['events'])} events)",
          flush=True)
    return out


def _slugify(text: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return (s[:48] or "question")


def run_custom(spec_path: str) -> Path:
    """Record a REAL lean_v2 run for an arbitrary question described in a JSON file.

    The JSON needs: question, as_of (YYYY-MM-DD), horizon (YYYY-MM-DD), and an as-of
    `background` (the frozen, time-locked facts a user knows as of `as_of`). Optional:
    resolution_criteria, title, slug. Example file:

        {"question": "Will X happen by ...?",
         "as_of": "2026-05-14", "horizon": "2026-06-25",
         "resolution_criteria": "Resolves YES iff ...",
         "background": "As of 2026-05-14, ..."}
    """
    spec = json.loads(Path(spec_path).read_text())
    for k in ("question", "as_of", "horizon", "background"):
        if not spec.get(k):
            raise SystemExit(f"custom question JSON is missing required field: {k!r}")
    slug = spec.get("slug") or _slugify(spec["question"])
    case = {
        "slug": slug,
        "title": spec.get("title") or spec["question"][:70],
        "question": spec["question"],
        "as_of": spec["as_of"],
        "horizon": spec["horizon"],
        "resolution_criteria": spec.get("resolution_criteria", ""),
        "background": spec["background"],
        "note": spec.get("note", "user-supplied as-of background (sealed-replay evidence)"),
    }
    return run_case(case=case)


def _result_to_dict(res) -> dict:
    import dataclasses
    if dataclasses.is_dataclass(res):
        return dataclasses.asdict(res)
    return dict(getattr(res, "__dict__", {}) or {})


def rebuild(slug: str = "banxico_unanimous_2026") -> Path:
    """Rebuild a recording from its saved raw capture — offline, no simulation, no API."""
    from swm_viz.build_recording import build_recording
    raw = json.loads((REC_DIR / "_raw" / f"{slug}.raw.json").read_text())
    recording = build_recording(
        meta=raw["meta"], calls=raw["calls"],
        blueprint_dict=raw.get("blueprint_dict") or {}, result_dict=raw["result_dict"])
    out = REC_DIR / f"{slug}.json"
    out.write_text(json.dumps(recording, indent=1, default=str))
    _update_index(recording, out)
    print(f"[rebuild] {out} — {len(recording['events'])} events, "
          f"{len(recording['llm_calls'])} calls")
    return out


def _update_index(recording: dict, path: Path) -> None:
    idx_path = REC_DIR / "index.json"
    try:
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"recordings": []}
    except Exception:  # noqa: BLE001
        idx = {"recordings": []}
    m = recording["meta"]
    row = {
        "slug": m.get("slug"),
        "file": path.name,
        "title": m.get("title"),
        "question": m.get("question"),
        "as_of": m.get("as_of"),
        "horizon": m.get("horizon"),
        "status": m.get("status"),
        "headline_probability": (recording.get("forecast") or {}).get("headline_probability"),
        "n_events": len(recording.get("events") or []),
        "n_llm_calls": len(recording.get("llm_calls") or []),
        "n_actors": len(recording.get("cast") or []),
        "recorded_at": m.get("recorded_at"),
    }
    idx["recordings"] = [r for r in idx["recordings"] if r.get("slug") != row["slug"]]
    idx["recordings"].append(row)
    idx["recordings"].sort(key=lambda r: r.get("recorded_at") or "", reverse=True)
    idx_path.write_text(json.dumps(idx, indent=1, default=str))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "banxico"
    if cmd == "rebuild":
        rebuild(sys.argv[2] if len(sys.argv) > 2 else "banxico_unanimous_2026")
    elif cmd == "custom":
        if len(sys.argv) < 3:
            raise SystemExit("usage: python -m swm_viz.record_run custom <question.json>")
        run_custom(sys.argv[2])
    else:
        run_case(cmd)
