# Troubleshooting guide

Common failures and the exact fix. Most "errors" from `acquire` / `prepare-all` are **recorded
outcomes**, not crashes — read the manifest note / report first.

## No space on the working volume

Symptom: `acquire` returns `deferred_storage`, or writes fail mid-normalization.
The storage guard defers any pull that would push the volume past `SWM_DISK_STOP_FRACTION` (default
0.85). Fixes:

```bash
du -sh "$SWM_DATA_ROOT"/*                     # find the heavy dirs
rm -rf "$SWM_DATA_ROOT"/cache/*               # HF cache + tmp extraction (re-derivable)
rm -rf "$SWM_DATA_ROOT"/raw/<big-unused-id>   # raw of a set you already normalized
export SWM_DISK_STOP_FRACTION=0.90            # or raise the threshold deliberately
```

Point `SWM_DATA_ROOT` at a bigger volume for full acquisition — everything under it is re-derivable.

## HF gated / 401 / repo-not-found

Symptom: `acquire` status `blocked`, note "gated repo" or "returned 401/403". The repo needs a human
action. Accept the license/terms on the dataset (or model) page, then:

```bash
export HF_TOKEN=...                           # or HUGGING_FACE_HUB_TOKEN
python -m machine_learning.cli datasets acquire <id>
```

A `RepositoryNotFoundError` with no token present is reported as blocked-with-token — a private/gated
repo looks identical to a missing one without auth.

## git-lfs missing

Symptom: a git-cloned dataset's large files are tiny LFS **pointer** files, and the manifest note says
"git-lfs not installed: LFS files are POINTERS, not content." Either install `git-lfs` and re-acquire,
or route the dataset to the **http adapter** against its `media.githubusercontent` LFS URL (the git
adapter degrades gracefully rather than hard-failing so the orchestrator can fall back).

## `datasets>=5` dropped loader scripts

Newer `datasets` removed support for repo loader scripts. Symptom: `load_dataset(...)` errors on a
dataset that shipped a `.py` loader. Prefer the adapter's `snapshot` mode (pull the raw Parquet/JSON
directly) or `stream` mode (`acquire.hf` with `mode: stream`) instead of a script-based load. The
`data.txt` pin is `datasets>=2.18`; if a specific set needs a loader script, pin `datasets<3` in a
throwaway env for that pull.

## bitsandbytes needs CUDA

Symptom: `import bitsandbytes` fails, or `train run --launch` errors on a CPU box. bitsandbytes is
CUDA-only and used **only** by the 4-bit QLoRA path. The **smoke test uses plain CPU LoRA and never
imports bitsandbytes** — run `smoke run` to validate the whole pipeline without a GPU. For an 8B run
you need a CUDA GPU + a matching bitsandbytes build (verify with
`python -c "import torch; print(torch.cuda.is_available())"`).

## Arrow schema errors on heterogeneous streams

Symptom: `pyarrow` complains about mismatched/mixed types when streaming or writing shards. The
pipeline stores each canonical record as a **JSON blob + flat index columns** precisely to avoid this,
and the HF stream adapter coerces values to JSON-able types (bytes become `<bytes:N>` markers). If you
hit it in a custom converter, don't build a wide typed table from raw rows — emit canonical records
and let the shard writer handle serialization.

## High quarantine rate = converter bug

Symptom: `normalize` warns "HIGH quarantine rate (> 20%) — likely a converter bug." Records are being
emitted but failing schema validation. Inspect the reasons:

```bash
head "$SWM_DATA_ROOT"/state/<id>/quarantine.jsonl      # each entry has the record_id + schema errors + locator
```

Fix the converter so `payload.target` has the task's required primary key, all 12 sections are
well-formed, and no fabricated fields sneak in. Never "fix" it by loosening validation — a high
quarantine rate means the mapping is wrong.

## `provenance show` says record not found

The `record_id` prefix before the first `:` is the `dataset_id`; the record is looked up in that
dataset's normalized shards. Confirm the dataset was normalized and the id is copied exactly
(deterministic ids contain colons). If the shards were regenerated with a different converter version,
ids change — re-fetch the id from the current shards.
