# swm_viz — Social World Model, Lean V2 replay viewer

A read-only, replayable **visualization of the real World-Model-V2 (Lean V2) simulation**. It
shows, for an actual run of the *main wired* consumer path
(`unified_runtime.simulate_world(..., execution_profile="lean_v2")`):

- **exactly how each agent is modelled** — their role, authority, and every generated
  private-state hypothesis (beliefs, goals, stances, relationships, grounded weights);
- **the exact LLM calls** given to them — the full prompt the model saw and the full reply it
  produced, verbatim, for every call in the run;
- **how they interact** — the relationship graph, and each actor's decision/vote landing on the
  shared institution tally;
- **each state transition, step by step** — with **play / pause / step +1 / scrub / replay**;
- an **openable event log** on the right of everything that happened.

Nothing here is a mock. The viewer only ever renders data a real simulation produced, and it
**does not modify any simulation code** under `swm/`.

---

## How it stays faithful (and non-invasive)

Every external call in a Lean V2 run flows through one chokepoint —
`LLMGateway.call(stage, prompt) -> reply` — and the compiled world comes from
`compile_blueprint(...)`. `recorder.py` transparently **wraps** those two functions at runtime:
each wrapper calls the original and returns its result **unchanged**, only *reading* the prompt,
reply, and compiled world on the way through. The simulation runs identically; we just observe
it. All patches are restored on exit. This is exactly "read/take in the same LLM API inputs and
outputs and states" — never a behavior change.

The remaining structured truth (each actor's modelled states, grounded weights, the ordered
decision/vote trace, the forecast decomposition, unresolved mass) is read from the run's own
`result.provenance["lean_v2"]`, which the simulation already produces.

```
recorder.py        transparent observers over LLMGateway.call + compile_blueprint
record_run.py      drives a REAL lean_v2 run (DeepSeek) and saves a raw capture + recording
build_recording.py folds capture + provenance into one ordered, replayable recording.json
server.py          stdlib localhost server (frontend + recordings + optional live-run trigger)
frontend/          the visual plane (index.html / app.js / styles.css) — zero dependencies
recordings/        produced recordings (real data) + index.json; _raw/ holds raw captures
```

---

## Quick start

```bash
# 1) record a real run (makes live LLM calls; DEEPSEEK_API_KEY must be set)
python -m swm_viz.record_run banxico

# 2) serve the viewer
python -m swm_viz.server            # -> http://127.0.0.1:8756/
```

Open the URL, pick a recording, and press **Play** (or **▶▎** to step one transform at a time).
Click any LLM-call row in the log to see the **exact prompt & reply**; click an actor on the
plane to see **exactly how they are modelled**.

### Rebuild a recording without re-running the sim

The raw capture is saved under `recordings/_raw/`, so you can iterate on the recording format
(or the viewer) for free:

```bash
python -m swm_viz.record_run rebuild banxico_unanimous_2026
```

### Trigger a fresh run from the running server

```bash
curl -XPOST localhost:8756/api/run -d '{"case_id":"banxico"}'
curl localhost:8756/api/run/status
```

---

## Running your own question

Fastest way — a JSON file, no code edit (see `example_question.json`):

```bash
python -m swm_viz.record_run custom swm_viz/example_question.json
```

The JSON needs `question`, `as_of`, `horizon`, and an as-of `background` (the frozen,
time-locked facts a user knows as of `as_of`); `resolution_criteria`, `title`, `slug` are
optional. The recording appears in the viewer's dropdown after you refresh.

Or add a permanent entry to `CASES` in `record_run.py` and run
`python -m swm_viz.record_run <case_id>`. Either way the background is injected through the
runtime's own sealed-replay `prebuilt_bundle` port, so no live retrieval is needed — the only
external dependency is the LLM backend.

## Notes on this run

- Model: `deepseek-v4-flash` (the consequential family used by the Lean V2 benchmark arms).
- The injected backend uses a larger `max_tokens` than the cost-bounded benchmark (3600), which
  truncated the multi-actor state-generation JSON; the larger ceiling lets the *same* simulation
  finish its replies. That is a backend/provider parameter, not a change to `swm/`.
- `max_workers=1` and no cross-run compile cache are set (via `execution_policy`) so the run is
  captured live and in a clean, readable order — again, configuration, not code change.
