"""Generate a static human-review site under reports/human_review/index.html.

A human can inspect, per dataset: up to 50 examples with inputs + targets side by side,
raw-source lineage, chronology, split assignment, missing fields, warnings, and suspected
leakage — then approve/reject each dataset by editing registry/training_approvals.yaml
(the versioned approval file). No server needed; open the HTML in a browser.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from machine_learning.config import HUMAN_REVIEW_DIR, AUDIT_DIR  # noqa: E402
from machine_learning.examples.formatters.sft import format_record  # noqa: E402
from machine_learning.registry_io import get_dataset, load_approvals, load_datasets  # noqa: E402

CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{padding:16px 24px;background:#161a22;border-bottom:1px solid #2a2f3a;position:sticky;top:0}
main{padding:24px;max-width:1100px;margin:auto}
a{color:#6cb6ff} .pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:6px}
.train{background:#1d3b1d;color:#8f8} .eval{background:#3b331d;color:#fd8} .blocked{background:#3b1d1d;color:#f88}
.card{background:#161a22;border:1px solid #2a2f3a;border-radius:8px;padding:14px 16px;margin:14px 0}
.ex{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;border-top:1px solid #232833;padding-top:10px}
pre{white-space:pre-wrap;word-break:break-word;background:#0b0d11;border:1px solid #232833;border-radius:6px;padding:8px;margin:0;font-size:12px;max-height:320px;overflow:auto}
.tgt pre{border-color:#3a5}
.meta{font-size:12px;color:#9aa4b2} .warn{color:#fd8}
h2{margin-top:28px} table{border-collapse:collapse} td,th{border:1px solid #2a2f3a;padding:4px 8px;font-size:12px}
"""


def _role_pill(role):
    cls = "train" if role in ("TRAIN_CANDIDATE", "VALIDATION_CANDIDATE") else (
        "blocked" if "BLOCKED" in role or role == "INFRASTRUCTURE_ONLY" else "eval")
    return f'<span class="pill {cls}">{role}</span>'


def _dataset_page(did: str) -> str:
    e = get_dataset(did)
    sample_p = HUMAN_REVIEW_DIR / f"{did}.sample.jsonl"
    susp_p = HUMAN_REVIEW_DIR / f"{did}.suspicious.jsonl"
    audit_p = AUDIT_DIR / f"{did}.json"
    approvals = load_approvals().get(did, {})
    audit = json.loads(audit_p.read_text()) if audit_p.exists() else {}

    parts = [f'<div class="card" id="{did}"><h2>{html.escape(e.get("official_name", did))} '
             f'{_role_pill(e["dataset_role"])}</h2>',
             f'<div class="meta">id <code>{did}</code> · license {html.escape(str(e.get("license")))} '
             f'(<code>{e.get("license_class")}</code>) · status {e.get("conversion_status")} · '
             f'approved: <b>{approvals.get("approved", False)}</b></div>',
             f'<div class="meta">source: {html.escape(str(e.get("official_data_source")))}</div>']
    if audit:
        n = audit.get("normalized", {})
        parts.append(f'<div class="meta">examples {n.get("n_valid")} · quarantined {n.get("n_quarantined")} · '
                     f'episodes {n.get("n_episodes")} · splits {audit.get("split_counts")} · '
                     f'leakage_ok {audit.get("leakage",{}).get("ok")}</div>')
        parts.append(f'<div class="meta">train rec: {html.escape(str(audit.get("training_recommendation")))}</div>')
    if not sample_p.exists():
        parts.append('<p class="meta">No normalized sample (storage-blocked or eval-only). '
                     'See the audit report + registry blockers.</p></div>')
        return "\n".join(parts)

    recs = [json.loads(l) for l in sample_p.read_text().splitlines() if l.strip()][:50]
    for i, r in enumerate(recs):
        fx = format_record(r, max_history_events=6)
        dq = r["data_quality"]
        prov = r["provenance"]["raw_record_locator"]
        parts.append(
            f'<div class="ex"><div><div class="meta">#{i+1} · {r["task_type"]} · '
            f'<code>{r["record_id"][-18:]}</code> · split={r["split_metadata"].get("split")}</div>'
            f'<pre>{html.escape(fx.prompt[-1100:])}</pre>'
            f'<div class="meta">lineage: files={prov.get("files")} ids={prov.get("ids")} '
            f'idx={prov.get("indices")}<br>missing={dq.get("missing_fields")} '
            f'<span class="warn">warnings={dq.get("warnings")}</span> '
            f'chronology_verified={dq.get("chronology_verified")} possible_leakage={dq.get("possible_leakage")}</div>'
            f'</div><div class="tgt"><div class="meta">TARGET</div>'
            f'<pre>{html.escape(fx.completion[:900])}</pre></div></div>')
    if susp_p.exists():
        susp = [json.loads(l) for l in susp_p.read_text().splitlines() if l.strip()][:25]
        if susp:
            parts.append('<h3>Most-suspicious examples</h3><ul>')
            for r in susp:
                parts.append(f'<li class="meta"><code>{r["record_id"][-18:]}</code> {r["task_type"]}: '
                             f'warnings={r["data_quality"].get("warnings")}</li>')
            parts.append('</ul>')
    parts.append("</div>")
    return "\n".join(parts)


def main():
    datasets = load_datasets()
    approvals = load_approvals()
    toc = ['<table><tr><th>dataset</th><th>role</th><th>approved</th></tr>']
    for did, e in sorted(datasets.items()):
        toc.append(f'<tr><td><a href="#{did}">{did}</a></td><td>{e["dataset_role"]}</td>'
                   f'<td>{approvals.get(did, {}).get("approved", False)}</td></tr>')
    toc.append('</table>')

    body = "\n".join(_dataset_page(did) for did in sorted(datasets))
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Behaviour-ML human review</title>
<style>{CSS}</style></head><body>
<header><b>SWORLDMODEL behaviour-ML — human review</b> ·
approve datasets by editing <code>registry/training_approvals.yaml</code> (nothing approved by default)</header>
<main><p>Review the examples below (inputs left, targets right, lineage + warnings inline).
Then set <code>approved: true</code> for each dataset you clear for training.</p>
{''.join(toc)}
{body}</main></body></html>"""
    HUMAN_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = HUMAN_REVIEW_DIR / "index.html"
    out.write_text(doc)
    print(f"human-review site -> {out}")


if __name__ == "__main__":
    main()
