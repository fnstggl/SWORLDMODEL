"""Reproducible Phase-7 CLI — Part 28.

Every subcommand performs a REAL operation or fails honestly (no placeholder verbs). Run from the repo root:

    PYTHONPATH=. python -m swm.world_model_v2.nonlinear <command> [options]

Commands:
  audit                 write the Part-0 nonlinearity audit artifact
  register-form         serialize the structural-form registry artifact
  fit                   fit a structural form to a committed dataset → fit artifact
  compare               compare candidate forms under identical splits on a dataset
  validate              held-out validation of a form vs baselines (validation-only selection)
  test-transfer         cross-domain transfer test of a fitted pack
  instantiate           build a scenario-specific nonlinear_spec and print it
  trace                 run a mechanism through WorldState and dump the forensic trace
  ablate                run the ablation ladder on a dataset
  register-failure      append a preserved failure record to the ledger
  promote / quarantine  change a nonlinear extension's lifecycle status (gated)
  verify-registry       recompute + verify the sidecar registry integrity hash

Determinism: seeds are explicit; artifacts carry dataset hashes + software versions + exact command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

RESULTS = "experiments/results"


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"wrote {path}")


def _load_dataset(name):
    """Resolve a committed dataset by short name → list of rows with {features, y} where applicable."""
    paths = {
        "telco": f"{RESULTS}/harvest_extra/telco_churn.json",
        "stackexchange": f"{RESULTS}/harvest_extra/stackexchange.json",
        "cmv": f"{RESULTS}/harvest_extra/cmv.json",
        "upworthy": f"{RESULTS}/exp054_upworthy/upworthy_parsed.json",
        "baby_names": f"{RESULTS}/exp072/baby_names.json",
    }
    if name not in paths:
        raise SystemExit(f"unknown dataset {name!r} (known: {sorted(paths)})")
    with open(paths[name]) as f:
        return json.load(f), paths[name]


def cmd_audit(args):
    from swm.world_model_v2.nonlinear.audit import run_audit
    _write(f"{RESULTS}/wmv2_phase7_audit.json", run_audit())


def cmd_register_form(args):
    from swm.world_model_v2.nonlinear.forms import registry_snapshot, list_forms
    snap = registry_snapshot()
    _write(f"{RESULTS}/wmv2_phase7_form_registry.json",
           {"_meta": {"n_forms": len(snap), "note": "Phase-7 structural-form registry (Part 1)"},
            "forms": snap})
    print(f"{len(list_forms())} forms registered")


def cmd_fit(args):
    from swm.world_model_v2.nonlinear import fit
    rows, path = _load_dataset(args.dataset)
    if not (rows and isinstance(rows[0], dict) and "features" in rows[0]):
        raise SystemExit(f"dataset {args.dataset} is not in {{features,y}} row form — use a runner")
    feat_keys = sorted(rows[0]["features"].keys())
    tr, va, te = fit.random_split(rows, seed=args.seed)
    fitter = fit.FITTERS.get(args.form)
    if fitter is None:
        raise SystemExit(f"no fitter for form {args.form!r} (have {sorted(fit.FITTERS)})")
    if args.form in ("logistic", "survival_hazard"):
        fr = fitter(tr, feat_keys, dataset=path, split=f"random seed={args.seed}", seed=args.seed)
    else:
        fr = fitter(tr, args.x_key or feat_keys[0], dataset=path, split=f"random seed={args.seed}",
                    seed=args.seed)
    fr.provenance["dataset_hash"] = fit.dataset_hash(rows)
    _write(f"{RESULTS}/wmv2_phase7_fit_{args.dataset}_{args.form}.json", fr.as_dict())


def cmd_compare(args):
    from experiments.wmv2_phase7_forms_validation import compare_dataset
    _write(f"{RESULTS}/wmv2_phase7_compare_{args.dataset}.json", compare_dataset(args.dataset, seed=args.seed))


def cmd_validate(args):
    from experiments.wmv2_phase7_forms_validation import validate_all
    _write(f"{RESULTS}/wmv2_phase7_validation.json", validate_all(seed=args.seed))


def cmd_ablate(args):
    from experiments.wmv2_phase7_forms_validation import ablate_dataset
    _write(f"{RESULTS}/wmv2_phase7_ablations.json", ablate_dataset(args.dataset, seed=args.seed))


def cmd_test_transfer(args):
    from experiments.wmv2_phase7_forms_validation import transfer_test
    _write(f"{RESULTS}/wmv2_phase7_transfer.json", transfer_test(seed=args.seed))


def cmd_instantiate(args):
    from swm.world_model_v2.nonlinear.forms import get_form
    form = get_form(args.form)
    spec = {"form_id": args.form, "params": json.loads(args.params) if args.params else {},
            "outcome_var": args.outcome or "outcome", "actor": args.actor or "actor_0",
            "output": "prob" if form.output_domain in ("unit_interval", "prob_window") else "rate",
            "options": ["True", "False"]}
    print(json.dumps({"nonlinear_spec": spec, "form_metadata": form.as_dict()}, indent=1, default=str))


def cmd_trace(args):
    from experiments.wmv2_phase7_traces import run_traces
    _write(f"{RESULTS}/wmv2_phase7_forensic_traces.json", run_traces())


def cmd_register_failure(args):
    from swm.world_model_v2.nonlinear.safety import FailureRecord, FailureLedger
    path = f"{RESULTS}/wmv2_phase7_failures.json"
    ledger = FailureLedger()
    if os.path.exists(path):
        doc = json.load(open(path))
        from swm.world_model_v2.nonlinear.safety import FailureRecord as FR
        for r in doc.get("failures", []):
            ledger.records.append(FR(**{k: v for k, v in r.items()}))
    ledger.add(FailureRecord(failure_id=args.id, mechanism_family=args.family, structural_form=args.form,
                             failure_type=args.type, dataset=args.dataset or "", suspected_cause=args.cause or "",
                             disposition=args.disposition or "preserved"))
    _write(path, ledger.as_dict())


def cmd_promote(args):
    from swm.world_model_v2.nonlinear.registry_ext import NonlinearExtensionStore
    store = NonlinearExtensionStore.load()
    store.set_status(args.extension, args.status, reason=args.reason or "")
    store.save()
    print(f"{args.extension} -> {args.status}")


def cmd_quarantine(args):
    args.status = "quarantined"
    cmd_promote(args)


def cmd_verify_registry(args):
    from swm.world_model_v2.nonlinear.registry_ext import verify_registry
    print(json.dumps(verify_registry(), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="nonlinear", description="Phase-7 nonlinear-mechanism CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(fn=cmd_audit)
    sub.add_parser("register-form").set_defaults(fn=cmd_register_form)
    f = sub.add_parser("fit"); f.add_argument("dataset"); f.add_argument("form")
    f.add_argument("--x-key", default=""); f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=cmd_fit)
    c = sub.add_parser("compare"); c.add_argument("dataset"); c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_compare)
    v = sub.add_parser("validate"); v.add_argument("--seed", type=int, default=0); v.set_defaults(fn=cmd_validate)
    a = sub.add_parser("ablate"); a.add_argument("dataset"); a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_ablate)
    t = sub.add_parser("test-transfer"); t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_test_transfer)
    i = sub.add_parser("instantiate"); i.add_argument("form"); i.add_argument("--params", default="")
    i.add_argument("--outcome", default=""); i.add_argument("--actor", default="")
    i.set_defaults(fn=cmd_instantiate)
    sub.add_parser("trace").set_defaults(fn=cmd_trace)
    rf = sub.add_parser("register-failure")
    for arg in ("id", "family", "form", "type"):
        rf.add_argument(f"--{arg}", required=True)
    rf.add_argument("--dataset", default=""); rf.add_argument("--cause", default="")
    rf.add_argument("--disposition", default="preserved"); rf.set_defaults(fn=cmd_register_failure)
    pr = sub.add_parser("promote"); pr.add_argument("extension"); pr.add_argument("status")
    pr.add_argument("--reason", default=""); pr.set_defaults(fn=cmd_promote)
    q = sub.add_parser("quarantine"); q.add_argument("extension"); q.add_argument("--reason", default="")
    q.set_defaults(fn=cmd_quarantine)
    sub.add_parser("verify-registry").set_defaults(fn=cmd_verify_registry)
    args = p.parse_args(argv)
    t0 = time.time()
    args.fn(args)
    print(f"[nonlinear {args.cmd}] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
