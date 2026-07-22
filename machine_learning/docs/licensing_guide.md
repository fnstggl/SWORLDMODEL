# Licensing guide

A dataset can be *read* far more freely than it can be *trained on*. This subsystem encodes that
distinction so no dataset ever enters a training manifest unless three gates all pass. These are
engineering classifications to drive the pipeline, **not legal advice**.

## License classes

`registry/licenses.yaml` defines coarse classes; each dataset's `license_class` points at one. The
decisive flag is `training_allowed`:

| class | commercial | derivatives | `training_allowed` | notes |
|---|---|---|---|---|
| `permissive_commercial` (MIT/Apache/BSD) | yes | yes | **true** | training, derivatives, commercial all OK with attribution |
| `cc_by` (CC-BY-4.0) | yes | yes | **true** | attribution only |
| `cc_by_sa` (CC-BY-SA-4.0) | yes | yes | **true** | ShareAlike: redistributed derivatives must carry CC-BY-SA |
| `cc_by_nc` (CC-BY-NC / NC-SA) | no | yes | **true** (non-commercial) | excluded from any *commercial* view |
| `cc_by_nc_nd` (CC-BY-NC-ND) | no | **no** | **false** | No-Derivatives → training makes a derivative → eval-only |
| `research_only` | unknown | unknown | **false** | citation-required, no explicit derivative grant → eval-only |
| `unknown_unstated` | unknown | unknown | **false** | source declares no license → assume nothing |
| `upstream_tos_restricted` | no | unknown | **false** | platform ToS governs the content → eval/infra |
| `restricted_program` | no | unknown | **false** | DUA / performer-only → blocked |
| `proprietary_unreleased` | no | no | **false** | never publicly released → blocked |

Rules of thumb: **unknown / unstated → not training-eligible**; **CC-BY-NC → non-commercial only**;
**CC-BY-NC-ND (No-Derivatives) → cannot train, eval-only**.

## The training-eligibility gate (role + license + approval)

`registry_io.training_eligibility(dataset_id)` returns `(eligible, reason)`. All three must hold:

1. **role** — `dataset_role` is `TRAIN_CANDIDATE` or `VALIDATION_CANDIDATE`;
2. **license** — the dataset's license class has `training_allowed: true`;
3. **approval** — a human approved it in `registry/training_approvals.yaml` (`approved: true`).

Nothing is approved by default, so a fresh clone cannot accidentally train on anything.
`sampling/manifests.py` calls this per dataset when building a training view and lists every
exclusion with its reason. `validation/licensing.py:verify_view_licenses` re-checks that no
training-forbidden dataset slipped into a non-eval view.

## Approving a dataset

Only after reviewing the dataset's audit report (`reports/audit/<id>.md`) and its human-review sample,
edit `registry/training_approvals.yaml`:

```yaml
reviewer: "your-name"
reviewed_at: "2026-07-22"
approvals:
  casino: {approved: true, note: "Reviewed audit + sample; CC-BY-4.0, clean.", reviewed_commit: "<git-sha>"}
```

The file is versioned, so every training run's `run_manifest.json` records exactly which approvals
were in effect. Editing it is a deliberate, reviewable act. Datasets deliberately absent from this
file (eval-only, blocked, infrastructure, license-restricted) can never be approved.

## Non-commercial + No-Derivatives handling

- **CC-BY-NC** datasets are training-eligible for **non-commercial** use only and are excluded from
  any commercial view (e.g. OmniBehavior, Deal-or-No-Deal, Criteo). A commercial adapter must not
  include them.
- **CC-BY-NC-ND** is No-Derivatives: training on it produces a derivative and is not permitted. Such
  datasets are `LICENSE_RESTRICTED_EVAL_ONLY` — you may read and measure, not derive/redistribute
  modified data. The readiness report tags them `LICENSE_BLOCKED`.

## The license matrix

`readiness check` writes `reports/readiness/license_matrix.csv` from
`validation/licensing.py:license_matrix()`: per dataset — license, class, role, commercial/derivative/
redistribution flags, `training_allowed_by_license`, `training_eligible_now`, `eligible_if_approved`,
and the exact reason. `validation/licensing.py:check_consistency()` flags any registry flag that
contradicts its class (e.g. `commercial_use_allowed=yes` under a non-commercial class).
