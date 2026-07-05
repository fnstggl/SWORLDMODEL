"""OpinionQA (Pew American Trends Panel) — per-INDIVIDUAL survey-response data.

This is the benchmark the best social-simulation work is measured on: predict a specific person's answer
to an opinion question. Each row is one (respondent, question): a persona of the respondent's
demographics, the question + answer choices, and the ANSWER THAT PERSON ACTUALLY GAVE. Grouping by
respondent uid gives per-person data; grouping by question gives the population.

Source: HF dataset RiverDong/OpinionQA (mirror of Santurkar et al. OpinionQA / Pew ATP), ~295k
(respondent, question) rows, ~76k respondents, 370 questions, 15 survey waves. Download once:
  HF_TOKEN=... curl -L -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/datasets/RiverDong/OpinionQA/resolve/main/data/test-00000-of-00001.parquet \
    -o data/oqa_test.parquet
A parsed, respondent-subsampled cache is committed under experiments/results/exp028_oqa/ so the
experiment is reproducible without the parquet.

We extract each respondent's structured demographics from the persona text; the value-mapping +
prediction live in experiments/exp028_individual_opinion.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PARQUET = "data/oqa_test.parquet"
CACHE = "experiments/results/exp028_oqa/oqa_parsed.json"

_PATTERNS = {
    "race": r"Racially, the person is ([^.]+)\.",
    "region": r"lives in the ([^.]+?) region",
    "age": r"in the ([^.]+?) age group",
    "sex": r"The person is (male|female)\.",
    "education": r"highest level of education is ([^.]+)\.",
    "marital": r"The person is (never been married|married|divorced|widowed|separated|living with a partner)",
    "religion": r"follows the ([^.]+?) religion",
    "attendance": r"attends religious services ([^.]+?)\.",
    "party": r"aligns with the ([^.]+?) party",
    "ideology": r"considers themselves ([^.]+?)\.",
    "income": r"earns ([^.]+?) per year",
}
_CHOICE = re.compile(r"\(([A-Z])\):\s*(.*)")


def _parse_prompt(prompt: str):
    demo = {}
    for k, pat in _PATTERNS.items():
        m = re.search(pat, prompt)
        demo[k] = m.group(1).strip().lower() if m else "unknown"
    demo["citizen"] = "no" if "is not a citizen" in prompt else ("yes" if "is a citizen" in prompt else "unknown")
    choices = _CHOICE.findall(prompt)
    return demo, [c[0] for c in choices]


def load_from_parquet(max_rows: int | None = None):
    import pyarrow.parquet as pq
    tbl = pq.read_table(PARQUET)
    rows = tbl.to_pylist()
    if max_rows:
        rows = rows[:max_rows]
    out = []
    for r in rows:
        demo, letters = _parse_prompt(r["prompt"])
        if not letters or r["answer"] not in letters:
            continue
        out.append({"uid": r["uid"], "qid": r["question_id"], "wave": r.get("folder", ""),
                    "demo": demo, "n_opt": len(letters), "answer_idx": letters.index(r["answer"])})
    return out


def load(subsample_respondents: int | None = 6000, seed: int = 0):
    """Prefer the committed parsed cache; else parse the parquet and (optionally) subsample respondents."""
    if Path(CACHE).exists():
        return json.loads(Path(CACHE).read_text())
    recs = load_from_parquet()
    if subsample_respondents:
        import random
        uids = sorted({r["uid"] for r in recs})
        rng = random.Random(seed); rng.shuffle(uids)
        keep = set(uids[:subsample_respondents])
        recs = [r for r in recs if r["uid"] in keep]
    return recs


if __name__ == "__main__":
    recs = load_from_parquet(max_rows=20000)
    from collections import Counter
    print(f"{len(recs)} parsed rows")
    ex = recs[0]
    print("example:", {k: ex[k] for k in ("uid", "qid", "n_opt", "answer_idx")})
    print("demographics:", ex["demo"])
    for f in ("religion", "ideology", "party", "attendance", "age"):
        print(f"  {f}:", Counter(r["demo"][f] for r in recs).most_common(6))
