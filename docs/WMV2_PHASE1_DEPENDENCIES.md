# WMv2 Phase 1 — Dependency Manifest (B16)

*The production Phase-1 path — `question → SimulationResult` — has **zero third-party runtime
dependencies**. The entire compiler, world state, mechanisms, rollout, and result contract run on the
Python 3.11 standard library. The only external service is the LLM used for **qualitative** decomposition.*

## Runtime environment

| item | value |
|---|---|
| Python | 3.11.15 (CPython) |
| Core package deps | **none** — `pyproject.toml` `dependencies = []` (dependency-free by design) |
| Platform | linux; pure-Python, no compiled extensions |

## Production import surface (`swm/world_model_v2/` + `swm/facade.py` + `swm/api/deepseek_backend.py`)

Standard library only:

| module | used for |
|---|---|
| `dataclasses` | every plan/state/result type |
| `json` | LLM decomposition parse, plan/result serialization, persistence |
| `hashlib` | `plan_hash`, prompt/evidence hashes, lock versions |
| `random` | particle sampling, Beta/Gamma draws (Marsaglia–Tsang), rollout RNG |
| `math`, `statistics` | Beta/Gamma math, entropy, uncertainty decomposition |
| `heapq` | the event queue (event-driven rollout on real calendar time) |
| `re` | JSON salvage, readout parsing, static domain-branch gate |
| `copy` | plan deep-copy (ablations; safe transforms) |
| `pathlib` | persistence + ledger paths |
| `time` | latency accounting, ledger timestamps (injected, not `Date.now`) |
| `subprocess` | `git rev-parse` for the code-commit provenance stamp |
| `urllib.request` / `urllib.error` | the DeepSeek HTTPS call (`swm/api/deepseek_backend.py`) |
| `os` | reads `DEEPSEEK_API_KEY` |

**No** `numpy`, `pandas`, `torch`, `scikit-learn`, `scipy`, `requests`, `httpx`, `openai`, or `anthropic`
is imported on the production path (verified by grep over `swm/world_model_v2/*.py`). Numerical primitives
(Beta/Gamma sampling, entropy, pooling) are implemented in-tree so the core stays dependency-free and
auditable.

## External service

| dependency | role | contract |
|---|---|---|
| **DeepSeek V3** (`deepseek-chat`) | LLM for **qualitative** world decomposition only — actors, relationships, mechanism *names*, a directional lean, structural hypotheses, sensitivities | called over HTTPS via `urllib`; needs `DEEPSEEK_API_KEY`. The LLM **may not** mint any number that becomes a probability, latent value, edge strength, weight, coefficient, rule, intervention effect, or terminal outcome probability — those come from typed mechanisms / broad priors. |

The pipeline accepts **any** `llm: Callable[[str], str]`; DeepSeek is the default backend, not a hard
dependency. Tests inject a scripted `llm=lambda p: json.dumps(...)` and exercise the full path with no
network.

## Optional / non-production extras (`pyproject.toml`)

| extra | packages | scope |
|---|---|---|
| `api` | `fastapi`, `uvicorn`, `anthropic` | the HTTP server surface — **not** the Phase-1 simulation path |
| `dev` | `pytest`, `httpx` | test + tooling only |

The Phase-1 test suites (`tests/test_world_model_v2.py`, `tests/test_wmv2_tier_a_fixes.py`,
`tests/test_forward_ledger_v2.py`, `tests/test_calibration.py`) require only `pytest`. Two repo-wide tests
outside Phase-1 need the `api` extra (`fastapi`) or a generated data file
(`data/dataset_registry.json`, gitignored) and fail environmentally in a bare checkout — neither touches
the Phase-1 path.

## Reproducibility inputs

A run is reproducible from: the code commit (stamped via `git rev-parse`), the compiler version
(`phase1-no-abstention-1.0`), the prompt hash, the evidence bundle hash, the seed, and `plan_hash`. No
`Date.now()`/`Math.random()`-style nondeterminism enters a stored artifact; timestamps are injected and
RNG is seeded. Validation harnesses are resumable via per-question caches under
`experiments/results/phase1_*`.
