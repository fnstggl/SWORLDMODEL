"""Per-person WRITING-HISTORY corpora from the full CMV challenge corpus (for deep-inference tests).

The full ConvoKit winning-args corpus has 19,714 challenges; ~429 challengers wrote >=8 arguments,
all timestamped. That gives us people with a real, dated writing history AND future outcomes — exactly
what the interview-gap analog needs: infer a deep persona from someone's past arguments, then predict
whether their FUTURE arguments persuade (earn a delta), with the persona computed strictly as-of.

Each returned instance is one challenge by a recurring author:
  {id, author, ts, arg_text, op_text, success}    (success = earned a delta)
Authors are those with >= min_args timestamped challenges (so a history exists). Ordered globally by ts
so a temporal train/test split is leakage-free.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

FULL = "data/cmv_challenges.json"


def load(min_args: int = 8, cap_authors: int | None = None):
    rows = json.loads(Path(FULL).read_text())
    rows = [r for r in rows if r.get("challenger") and r["challenger"] != "[deleted]" and r.get("ts")
            and r.get("arg_text")]
    cnt = Counter(r["challenger"] for r in rows)
    authors = [a for a, c in cnt.items() if c >= min_args]
    authors.sort(key=lambda a: (-cnt[a], a))
    if cap_authors is not None:
        authors = authors[:cap_authors]
    keep = set(authors)
    inst = [{"id": r["id"], "author": r["challenger"], "ts": int(r["ts"]),
             "arg_text": r["arg_text"], "op_text": r.get("op_text", ""), "success": int(r["success"])}
            for r in rows if r["challenger"] in keep]
    inst.sort(key=lambda r: r["ts"])
    return inst, authors


if __name__ == "__main__":
    inst, authors = load()
    base = sum(r["success"] for r in inst) / len(inst)
    from collections import Counter as C
    per = C(r["author"] for r in inst)
    print(f"{len(inst)} challenges by {len(authors)} recurring authors (>=8 args); base rate {base:.3f}")
    print(f"docs/author: min {min(per.values())} max {max(per.values())} "
          f"median {sorted(per.values())[len(per)//2]}")
    print("example arg:", inst[0]["arg_text"][:120].replace(chr(10), " "))
