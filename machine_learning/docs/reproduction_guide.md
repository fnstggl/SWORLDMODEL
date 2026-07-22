# Reproduction guide

Every artifact in this pipeline is re-derivable from committed code + public source data. This guide
shows how the determinism is achieved and how to reproduce a record, a manifest, and a training run,
and how to trace any record's full lineage.

## Deterministic record_ids + content_hashes

`canonical.make_record` builds both from the record's **semantic content only** — never from
wall-clock time or run-specific values (`canonical.py`):

- `record_id = "<dataset_id>:<task_type>:<stable-hash>"`, where the hash is over
  `[dataset_id, task_type, episode_id, sequence_index, raw_record_locator, payload.target]`.
- `content_hash = sha256(canonical_json(record without provenance/split_metadata and without
  source.normalization_timestamp))`. It also becomes `split_metadata.dedup_hash`.

`canonical_json` is sorted-key, compact, non-ASCII-preserving, so identical content always serializes
identically. Consequence: re-running a converter (same version) on the same raw input reproduces the
**exact same ids**. Timestamps live only in `source.normalization_timestamp` and
`provenance.conversion_timestamp`, both excluded from the hashes.

## Fixed seeds

Training sets `random`, `numpy`, `torch`, and `torch.cuda` seeds from `train.seed` (default 42) via
`train_qlora.set_seed`. The seed is recorded in `run_manifest.json`. (Full bit-exactness across
different GPUs/library builds is not guaranteed by seeds alone — the manifest records the device +
package versions so any divergence is explainable.)

## run_manifest.json fields

Written to `<run_dir>/run_manifest.json` at launch and updated on completion. It records everything
needed to reproduce or audit a run:

`name` · `config` (the full resolved config) · `base_model` + `base_model_revision` · `tokenizer` +
`tokenizer_revision` · `quantization_effective` · `data_view` + `data_manifest_hash` ·
`n_train_records` · `code_commit` (git HEAD of the converter/training code) · `seed` ·
`planned_total_steps` · `trainable_parameters` · `package_versions` · `device` + `cuda` · and a
`result` block (steps, final/best loss, early-stopped, finish time).

## Reproduce a manifest (manifest_hash)

A view's records are deterministic given the split tables + the view config. `sampling/manifests.py`
computes `manifest_hash = sha256 over the sorted record_ids` (16 hex chars).

```bash
python -m machine_learning.cli manifests build actor_choice_v1
# -> prints included datasets, counts, est_tokens, and hash=<manifest_hash>
```

Same registry + approvals + split tables + view config ⇒ same `manifest_hash`. A changed hash means
one of those inputs changed — diff the committed `reports`/`artifacts` summary
(`<view>.summary.json`) to see which datasets entered/left and why.

## Reproduce a training run

1. Check out the `code_commit` from the target run's `run_manifest.json`.
2. Restore the same registry + `training_approvals.yaml` state (versioned) so eligibility matches.
3. Re-acquire + re-normalize the same datasets (ids are deterministic) and re-split.
4. `manifests build <view>` — confirm the `manifest_hash` matches the run manifest's
   `data_manifest_hash`.
5. `train run <config> --launch` with the same `seed`, base_model, and revision from the manifest.

Checkpoints + `resume=True` make the loop itself restart-safe (see
[`training_guide.md`](training_guide.md)).

## Trace a record's lineage

```bash
python -m machine_learning.cli provenance show <record_id>
```

`validation/provenance.py:trace` returns the full chain: dataset + version, task_type, license_class,
citation, converter + version, `conversion_timestamp`, `code_commit`, `transformation_steps`, the raw
source (`files` / `record_ids` / `indices`) joined with the **raw-file checksums from the source
acquisition manifest**, the `content_hash`, `episode_id`, split, and `data_quality`. That closes the
loop from a canonical record back to the exact raw bytes it came from.
