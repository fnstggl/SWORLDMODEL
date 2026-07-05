"""GlobalOpinionQA (Anthropic/llm_global_opinions) loader — population opinion prediction.

Durmus et al. 2023: survey questions (World Values Survey + Pew Global Attitudes) with the response
distribution per country. The world-model task: predict a country's distribution over answer options
for a question. This is the AGGREGATE / population-opinion regime and a standard benchmark comparable
to the best social-simulation work.

The VariableMap framing: a country is an ENTITY whose value-variables (religiosity, individualism,
traditional↔secular, trust-in-institutions, …) are inferred from world knowledge (NOT from this
survey — non-circular), and the question is the ACTION that activates some of those values. Here we
provide the parsed records; the model + inference live in experiments/exp023_global_opinion.py.

Data (gitignored under data/): download the GlobalOpinionQA table once with an HF token, e.g.
  HF_TOKEN=... python -c "from huggingface_hub import hf_hub_download; \
    import shutil; p=hf_hub_download('Anthropic/llm_global_opinions','data/train-00000-of-00001.parquet',repo_type='dataset'); ..."
then export it to data/global_opinions.csv (columns: question, selections, options, source). The
committed inference artifact experiments/results/exp023_country_values.json holds the inferred country
value-profiles so the model is reproducible even without re-running the LLM inference.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import re
from pathlib import Path

CSV = "data/global_opinions.csv"


def _parse_selections(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return ast.literal_eval(m.group(0))
    except Exception:
        return {}


def load():
    rows = list(csv.DictReader(io.StringIO(Path(CSV).read_text(encoding="utf-8", errors="ignore"))))
    out = []
    for i, r in enumerate(rows):
        try:
            options = ast.literal_eval(r["options"])
        except Exception:
            continue
        sel = _parse_selections(r["selections"])
        if not sel or not options:
            continue
        dists = {}
        for country, probs in sel.items():
            if isinstance(probs, list) and len(probs) == len(options) and abs(sum(probs) - 1) < 0.05:
                dists[country] = [max(1e-6, float(p)) for p in probs]
        if len(dists) >= 3:
            out.append({"qid": i, "question": r["question"], "options": options,
                        "source": r.get("source", ""), "dists": dists})
    return out


def country_list(records):
    from collections import Counter
    c = Counter()
    for r in records:
        for k in r["dists"]:
            c[k] += 1
    return c


if __name__ == "__main__":
    recs = load()
    cc = country_list(recs)
    print(f"{len(recs)} questions with >=3 country distributions; {len(cc)} countries")
    print("top countries:", cc.most_common(12))
    print("example:", recs[0]["question"][:80], "| options", recs[0]["options"], "| n countries", len(recs[0]["dists"]))
