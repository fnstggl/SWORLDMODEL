"""GitHub issue-response dataset loader for the general response backtest.

entity = repo, segment = repo owner/org (for multilevel pooling), outcome = responded within window.
Reuses data/gh_issues.json produced by experiments.github_individual_harness fetch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MFN = ["title_len", "body_len_log", "n_labels", "is_bug", "is_question", "is_member"]


def load_samples():
    recs = json.loads(Path("data/gh_issues.json").read_text())["records"]
    samples = []
    for r in recs:
        t = r["title"].lower()
        mf = {
            "title_len": min(1.0, len(r["title"]) / 80),
            "body_len_log": math.log1p(r["body_len"]) / 10.0,
            "n_labels": min(1.0, r["n_labels"] / 5.0),
            "is_bug": 1.0 if any(k in t for k in ("bug", "error", "crash", "fail", "broken")) else 0.0,
            "is_question": 1.0 if ("?" in r["title"] or t.startswith("how ")) else 0.0,
            "is_member": 1.0 if r["author_assoc"] in ("MEMBER", "OWNER", "COLLABORATOR") else 0.0,
        }
        seg = r["repo"].split("/")[0]                # owner/org = segment
        samples.append((r["repo"], seg, mf, r["responded"]))
    return samples, MFN
