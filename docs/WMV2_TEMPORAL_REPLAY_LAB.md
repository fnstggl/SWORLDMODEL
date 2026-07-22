# WMV2 Temporal Replay Laboratory — sealed historical backtesting

**The standard.** At forecast time T, no component of the forecasting system may access anything created
after T — and where that cannot be *proven* (a current LLM's weights contain the outcome), the event is
causally blinded, actively probed for leakage per row, and classified; only contamination-safe rows enter
the headline. "Pretend it is October 2024" prompting is retained ONLY as a diagnostic arm and is labeled
`contamination_not_excluded` by construction.

## Architecture (five separated parts)

```
historical event vault      experiments/replay_vault/events.json      (PUBLIC — no outcomes)
        ↓
time-locked evidence        Phase-2 orchestrator: strict as-of retrieval, per-doc temporal verification,
                            claim-level leakage audit (resolution terms, future dates, retrospective
                            language), timestamp-basis grading, snapshot refs
        ↓
blinded / prompted compiler swm/replay/blinding.py — pseudonym mapping over ALL LLM-visible text
        ↓
full unified simulation     simulate_world(..., prebuilt_bundle=frozen_bundle) — every phase, one path
        ↓
sealed resolution + scorer  experiments/replay_vault/SEALED_resolutions.json — loadable ONLY with
                            REPLAY_SCORER=1; scorer is a separate process (experiments/replay_score.py)
```

**Sealing (honest scope).** Process-level, not hardware isolation: the forecaster never reads the sealed
store (a loud `PermissionError` without `REPLAY_SCORER=1`), every forecast row is content-hashed at freeze
time (`freeze_hash`) and the scorer verifies hashes so post-resolution edits are detectable. The runtime
fingerprint is stamped on every row. The limitation (same machine, no network-level egress lock on the LLM
API) is recorded in the artifact — that is exactly why the blinding + probe layer exists.

## Arms

1. **Verified pre-cutoff model checkpoint** — *unavailable for this backend*; recorded on every row rather
   than papered over. This is the cleanest arm and the first scale-up item.
2. **`blinded_current_llm`** (the trusted arm): evidence gathered strictly as-of by CODE with real names
   (retrieval is code, not the LLM), then every LLM-visible text — question and evidence — passes through a
   stable pseudonym mapping (Candidate A / Central Bank B / Federation K…), calendar years stripped, causal
   structure (roles, trajectories, thresholds, institutions) preserved. Mappings carry no outcomes and are
   stored forecaster-side; blinding is never assumed secure (see probes).
3. **`cutoff_prompted_unblinded`** (diagnostic only): the current product path on real identities with the
   as-of cutoff. Never mixed into the clean headline.

## Per-row leakage probes (active detection, not a global claim)

- **Name-only probe** — real question, "do you know how this resolved as fact?" A confident correct answer
  (scored later against the sealed outcome) → `known_contaminated`.
- **Recognition probe** — the blinded packet; if the model identifies the real event (names a real entity
  from the mapping, or high confidence) → `contamination_susceptible`.
- **No-evidence probe** — blinded question, no evidence; recorded for base-rate-collapse analysis.

Row classes: `known_contaminated` / `contamination_susceptible` / `uncertain_leakage` /
`low_leakage_risk`. **Headline = blinded arm ∩ low_leakage_risk only.**

## Scoring discipline

- **Event-family clustering**: correlated questions (e.g. the two 2024-presidential contracts) share one
  cluster; the bootstrap resamples **clusters**, never rows, so correlated snapshots cannot inflate n.
- **Multi-cutoff snapshots**: each event is forecast at a far (−21d) and near cutoff — update behavior is
  measurable, and both snapshots stay in the same cluster.
- Brier + log-loss + directional accuracy vs the p=0.5 base-rate baseline. A market baseline is reported
  **null** — no defensible as-of market snapshot is stored, and fabricating one would poison the comparison.
- Vault: 18 resolved events / 16 families across elections, central-bank decisions, geopolitics, tech,
  finance, sports, spaceflight (2024–2025, post-knowledge-cutoff-adjacent, all objectively resolved).

## Results

Results are written by the scorer to `experiments/results/replay/scores.json` (headline, diagnostic arm,
leakage census, cluster CIs) — see the PR description for the run summary. Read that artifact, not this
doc, for numbers: the doc describes the *method*, the artifact carries the *measurements*, and the two are
kept separate deliberately.

## Scale-up path (in order)

1. Pre-cutoff checkpoint arm (Approach A) via a time-indexed model zoo → the cleanest headline.
2. Archived-bytes evidence (Wayback/Common Crawl snapshot enforcement server-side: `proven_available_at <=
   cutoff` as a store guarantee rather than a per-doc verification grade).
3. Counterfactual-evidence and identity-permutation probes per row; temporal-fact probes.
4. Market-archive ingestion (Kalshi/Polymarket/Metaculus) with contract-graph clustering and market-blind
   vs market-informed arms.
5. Trajectory scoring (intermediate outcomes per event family), then the 100-worlds × 4-cutoffs × 2-arms
   milestone, then thousands of clusters.
6. The prospective live ledger stays the final authority for current-product claims.
