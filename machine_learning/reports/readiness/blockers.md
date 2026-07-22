# Blockers

Each blocked dataset is documented with the exact reason + the human action needed.

## acl_online_shopping — ACCESS_BLOCKED
- role: ACCESS_BLOCKED  |  license: None stated; proprietary internal data
- blocker: Dataset is proprietary Amazon-internal data; never publicly released. Human action: none available without Amazon permission. Use OPeRA (same lab, released) as the multi-turn shopping substitute.

## agentsociety — INFRASTRUCTURE_ONLY
- role: INFRASTRUCTURE_ONLY  |  license: MIT (simulator). Underlying datasets: Yelp/Amazon/Goodreads own terms (Yelp non-commercial/academic).
- blocker: The challenge itself ships a SIMULATOR, not human data; we do NOT train on simulator output. To use the underlying Yelp/Amazon/Goodreads human data, register each dataset separately under its own license (Yelp is non-commercial/academic).

## behaviorbench — LICENSE_BLOCKED
- role: LICENSE_RESTRICTED_EVAL_ONLY  |  license: CC-BY-NC-ND-4.0
- blocker: CC-BY-NC-ND: No-Derivatives forbids training (which produces a derivative). Normalize for EVALUATION only; never place in a training manifest. Converter implemented + fixture-tested. Resume (eval): acquire behaviorbench --allow-large then normalize (eval-only).

## darpa_socialsim — ACCESS_BLOCKED
- role: ACCESS_BLOCKED  |  license: No public license; program/DUA terms
- blocker: No public/archival download. Human action: contact DARPA/Leidos or partner with a former SocialSim performer (USC/ISI, UCF, UVA Biocomplexity, PNNL) under a data-use agreement. Twitter/X ToS separately bars public redistribution of raw data.

## debate — ACCESS_BLOCKED
- role: ACCESS_BLOCKED  |  license: Not stated (data not yet publicly released)
- blocker: Dataset not publicly released as of last_verified_at. Human action: watch arxiv/OpenReview page rMnZbCOhSS for the promised code+data release, or email the authors (UW-Madison / Google DeepMind / Stanford) for pre-release access.

## kuairand — CONVERTER_READY_STORAGE_BLOCKED
- role: TRAIN_CANDIDATE  |  license: CC-BY-SA-4.0
- blocker: KuaiRand-Pure (~194MB) is acquirable but tar.gz extraction + CSV normalization is disk-heavy; keep randomized-exposure (is_rand=1) separate from organic. Converter implemented + fixture-tested. Resume: acquire kuairand --allow-large (Pure) then normalize. Do NOT let its size dominate the sampler (per-dataset cap set).

## mirobench — ACCESS_BLOCKED
- role: ACCESS_BLOCKED  |  license: Not stated; underlying Reddit content under Reddit ToS
- blocker: No public release located. Human action: watch arxiv 2606.14715 for a code/data link; if released, verify Reddit-ToS compliance before redistribution (local conversion only).

## omnibehavior — CONVERTER_READY_STORAGE_BLOCKED
- role: TRAIN_CANDIDATE  |  license: CC-BY-NC-SA-4.0 (non-commercial)
- blocker: 6.37GB > safe working disk here. Streaming converter implemented + fixture-tested; sample normalized. Full run resume: python -m machine_learning.cli datasets acquire omnibehavior --allow-large (on a >=20GB volume) then normalize. NON-COMMERCIAL license: excluded from any commercial training view.

## opera — CONVERTER_READY_STORAGE_BLOCKED
- role: TRAIN_CANDIDATE  |  license: CC-BY-4.0
- blocker: 8.58GB > safe working disk. Streaming converter implemented + fixture-tested; sample normalized. Rationales are self-reports, NOT infallible private states (recorded with confidence markers). Full run resume: acquire --allow-large on a >=20GB volume.

## psych101 — NORMALIZED_AND_VALIDATED
- role: TRAIN_CANDIDATE  |  license: Apache-2.0
- blocker: 859MB fits, but the converter must parse choices out of NL transcripts (response markers) per-experiment — non-trivial. Streaming converter implemented + fixture-tested on the documented transcript format; sample normalized. Full run resume: acquire --allow-large then normalize psych101.

## simbench — CONVERTER_READY_STORAGE_BLOCKED
- role: CROSS_DATASET_EVAL_ONLY  |  license: CC-BY-NC-SA-4.0 (non-commercial). Underlying 20 sources keep their own terms.
- blocker: Reserved as the held-out population-response cross-dataset transfer test; excluded from training manifests by design. NC-SA (non-commercial). Converter implemented + fixture-tested. Resume (eval): acquire simbench --allow-large then normalize (eval-only).

## socsci210 — NORMALIZED_EVAL_ONLY
- role: LICENSE_RESTRICTED_EVAL_ONLY  |  license: No license declared on the HF card
- blocker: License unspecified on the official card -> keep OUT of training (eval-only) until authors clarify. Streaming converter implemented + fixture-tested; sample normalized as eval-only. Resume: acquire --allow-large then normalize socsci210.

## some — CONVERTER_READY_STORAGE_BLOCKED
- role: CROSS_DATASET_EVAL_ONLY  |  license: Apache-2.0 (repo header) — but underlying crawled content under platform ToS
- blocker: 50GB+ exceeds working disk; content under platform ToS (eval-only, no redistribution). Converter streams + samples only empirically-grounded human posts (drops agent-benchmark task scaffolding). Resume: acquire with mode=stream larger --limit on a big volume.

## surge — CONVERTER_READY_STORAGE_BLOCKED
- role: TRAIN_CANDIDATE  |  license: MIT (code) + CC-BY-4.0 (author-created data under data/)
- blocker: Converter implemented + fixture-tested. Full acquisition deferred (needs git clone + moderate disk). Resume: python -m machine_learning.cli datasets acquire surge --allow-large && ... normalize surge.

## upworthy — CONVERTER_READY_STORAGE_BLOCKED
- role: TRAIN_CANDIDATE  |  license: CC-BY-4.0
- blocker: Converter implemented + fixture-tested. OSF direct-file ids vary by release; set acquire.http.urls to the current OSF file download link, or place the CSV in raw/upworthy/ and run normalize. Randomization-problem window (2013-06-25..2014-01-10) is flagged + excluded from confirmatory splits.

