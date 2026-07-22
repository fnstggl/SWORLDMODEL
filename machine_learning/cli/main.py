"""Command-line interface for the SWORLDMODEL behaviour-ML system.

    python -m machine_learning.cli registry verify
    python -m machine_learning.cli datasets list
    python -m machine_learning.cli datasets inspect <id>
    python -m machine_learning.cli datasets acquire <id> [--allow-large]
    python -m machine_learning.cli datasets normalize <id> [--limit N]
    python -m machine_learning.cli datasets validate <id>
    python -m machine_learning.cli datasets audit <id>
    python -m machine_learning.cli datasets split <id>
    python -m machine_learning.cli datasets prepare-all [--allow-large]
    python -m machine_learning.cli manifests build <view> [--preview]
    python -m machine_learning.cli readiness check
    python -m machine_learning.cli provenance show <record_id>
    python -m machine_learning.cli eval baselines
    python -m machine_learning.cli smoke run
    python -m machine_learning.cli train run <config> [--launch]

Design: imports are lazy per-command so the CLI starts instantly and never pulls torch for
data commands. `prepare-all` is resumable and never hides a failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

TS = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ registry
def cmd_registry_verify(args):
    from ..registry_io import verify_registry, summarize
    issues = verify_registry()
    errs = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warning"]
    print(f"registry: {summarize()}")
    for i in errs:
        print(f"  ERROR  {i.dataset_id}: {i.message}")
    for i in warns:
        print(f"  warn   {i.dataset_id}: {i.message}")
    print(f"{len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0


# ------------------------------------------------------------------ datasets
def cmd_datasets_list(args):
    from ..registry_io import load_datasets
    rows = load_datasets()
    print(f"{'dataset':22} {'role':28} {'license_class':22} {'status':30}")
    for did, e in sorted(rows.items()):
        print(f"{did:22} {e['dataset_role']:28} {e.get('license_class',''):22} {e.get('conversion_status',''):30}")
    print(f"\n{len(rows)} datasets")
    return 0


def cmd_datasets_inspect(args):
    from ..registry_io import get_dataset
    from ..acquisition.download import plan
    e = get_dataset(args.dataset)
    _print({k: e.get(k) for k in ("dataset_id", "official_name", "license", "license_class",
            "dataset_role", "supported_tasks", "split_unit", "conversion_status",
            "access_requirements", "official_data_source", "blockers")})
    print("--- acquisition plan ---")
    _print(plan(args.dataset, allow_large=args.allow_large))
    return 0


def cmd_datasets_acquire(args):
    from ..acquisition.download import acquire
    man = acquire(args.dataset, allow_large=args.allow_large, limit=args.limit, timestamp=TS())
    print(f"[{args.dataset}] status={man['status']} files={len(man.get('files',[]))} "
          f"bytes={man.get('total_bytes',0)}")
    for n in man.get("notes", [])[-3:]:
        print(f"  note: {n}")
    return 0 if man["status"] in ("acquired", "partial", "blocked", "deferred_storage", "skipped") else 1


def cmd_datasets_normalize(args):
    from ..normalization.pipeline import normalize
    rep = normalize(args.dataset, timestamp=TS(), limit=args.limit)
    _print(rep.as_dict())
    return 0


def cmd_datasets_split(args):
    from ..splitting.policies import split_dataset
    from ..splitting.leakage_checks import check_dataset
    rep = split_dataset(args.dataset)
    lk = check_dataset(args.dataset)
    print(f"[{args.dataset}] splits={rep.counts} leakage_ok={lk.ok}")
    return 0 if lk.ok else 1


def cmd_datasets_validate(args):
    from ..validation import validate_dataset
    res = validate_dataset(args.dataset, limit=args.limit)
    print(f"[{args.dataset}] critical_ok={res['critical_ok']} critical={res['critical']}")
    if res["warnings"]:
        print("  warnings:", res["warnings"])
    return 0 if res["critical_ok"] else 1


def cmd_datasets_audit(args):
    from ..report_builder import build_dataset_audit
    path = build_dataset_audit(args.dataset)
    print(f"audit report -> {path}")
    return 0


def cmd_datasets_prepare_all(args):
    from ..pipeline_runner import prepare_all
    table = prepare_all(allow_large=args.allow_large, limit=args.limit,
                        only=args.only.split(",") if args.only else None)
    _print_prepare_table(table)
    return 0


def _print_prepare_table(table):
    cols = ["dataset", "access", "license", "acquired", "normalized", "validated",
            "training_eligible", "evaluation_eligible", "examples", "tokens", "blockers"]
    print("\n" + " | ".join(cols))
    print("-" * 120)
    for row in table:
        print(" | ".join(str(row.get(c, ""))[:24] for c in cols))


# ------------------------------------------------------------------ manifests
def cmd_manifests_build(args):
    from ..sampling.manifests import build_view
    s = build_view(args.view, preview=args.preview)
    print(f"view={s['view']} preview={s['preview']} included={s['datasets_included']}")
    print(f"  raw={s['n_raw_examples']} manifest_records={s['n_manifest_records']} "
          f"est_tokens={s['estimated_tokens']} hash={s['manifest_hash']}")
    for x in s["datasets_excluded"][:12]:
        print(f"  excluded {x['dataset']}: {x['reason']}")
    return 0


# ------------------------------------------------------------------ readiness / provenance / eval
def cmd_readiness_check(args):
    from ..readiness import build_readiness
    out = build_readiness()
    print(f"readiness report -> {out['report']}")
    print(f"  ready_for_gpu_training: {out['ready']}")
    print(f"  datasets: {out['n_datasets']} | normalized+validated: {out['n_validated']} | "
          f"blocked: {out['n_blocked']} | eval-only: {out['n_eval_only']}")
    return 0


def cmd_provenance_show(args):
    from ..validation.provenance import trace
    t = trace(args.record_id)
    if t is None:
        print(f"record not found: {args.record_id}")
        return 1
    _print(t)
    return 0


def cmd_eval_baselines(args):
    from ..evaluation.reports import run_baselines
    out = run_baselines()
    print(f"baselines computed for {out['n_datasets']} datasets; "
          f"{len(out['baselines'])} (dataset,task) rows -> reports/readiness/baselines.md")
    return 0


# ------------------------------------------------------------------ smoke / train
def cmd_smoke_run(args):
    from ..training.smoke import run_smoke
    res = run_smoke(dataset_id=args.dataset)
    for k, v in res.checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"SMOKE {'PASSED' if res.passed else 'FAILED'}")
    return 0 if res.passed else 1


def cmd_train_run(args):
    from ..training.train_qlora import load_train_config
    cfg = load_train_config(args.config)
    if not args.launch:
        print(f"[dry-run] training config '{args.config}' loaded and valid.")
        print(f"  base_model={cfg['model']['base_model']} quantization={cfg['model'].get('quantization')}")
        print(f"  data={cfg.get('data')}")
        print("  Refusing to launch a full fine-tune without --launch (and a GPU).")
        print(f"  To launch:  python -m machine_learning.cli train run {args.config} --launch")
        return 0
    from ..training.train_qlora import train
    res = train(cfg, run_name=args.config)
    _print(res.as_dict())
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="machine_learning.cli", description="SWORLDMODEL behaviour-ML CLI")
    sub = p.add_subparsers(dest="group", required=True)

    reg = sub.add_parser("registry").add_subparsers(dest="cmd", required=True)
    reg.add_parser("verify").set_defaults(func=cmd_registry_verify)

    ds = sub.add_parser("datasets").add_subparsers(dest="cmd", required=True)
    ds.add_parser("list").set_defaults(func=cmd_datasets_list)
    for name, func in [("inspect", cmd_datasets_inspect), ("acquire", cmd_datasets_acquire),
                       ("normalize", cmd_datasets_normalize), ("validate", cmd_datasets_validate),
                       ("audit", cmd_datasets_audit), ("split", cmd_datasets_split)]:
        sp = ds.add_parser(name)
        sp.add_argument("dataset")
        sp.add_argument("--allow-large", action="store_true")
        sp.add_argument("--limit", type=int, default=None)
        sp.set_defaults(func=func)
    pa = ds.add_parser("prepare-all")
    pa.add_argument("--allow-large", action="store_true")
    pa.add_argument("--limit", type=int, default=None)
    pa.add_argument("--only", default=None, help="comma-separated dataset ids")
    pa.set_defaults(func=cmd_datasets_prepare_all)

    mf = sub.add_parser("manifests").add_subparsers(dest="cmd", required=True)
    mb = mf.add_parser("build")
    mb.add_argument("view")
    mb.add_argument("--preview", action="store_true")
    mb.set_defaults(func=cmd_manifests_build)

    rd = sub.add_parser("readiness").add_subparsers(dest="cmd", required=True)
    rd.add_parser("check").set_defaults(func=cmd_readiness_check)

    pr = sub.add_parser("provenance").add_subparsers(dest="cmd", required=True)
    ps = pr.add_parser("show")
    ps.add_argument("record_id")
    ps.set_defaults(func=cmd_provenance_show)

    ev = sub.add_parser("eval").add_subparsers(dest="cmd", required=True)
    ev.add_parser("baselines").set_defaults(func=cmd_eval_baselines)

    sm = sub.add_parser("smoke").add_subparsers(dest="cmd", required=True)
    smr = sm.add_parser("run")
    smr.add_argument("--dataset", default="casino")
    smr.set_defaults(func=cmd_smoke_run)

    tr = sub.add_parser("train").add_subparsers(dest="cmd", required=True)
    trr = tr.add_parser("run")
    trr.add_argument("config")
    trr.add_argument("--launch", action="store_true", help="actually launch (needs a GPU)")
    trr.set_defaults(func=cmd_train_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
