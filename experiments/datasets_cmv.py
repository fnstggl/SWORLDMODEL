"""ChangeMyView persuasion dataset (ConvoKit winning-args corpus) — the on-thesis test.

Outcome: did an argument PERSUADE the original poster (earn a delta)? This is a stance-shift outcome
driven by exactly the latent variables the VariableMap infers — the OP's openness/skepticism/stance
and the argument's fit/evidence/tone — and the OP is one-off (NO entity history), so LLM inference is
the only way to estimate those variables. Timestamped (99% of utterances) → a real no-cheat temporal
split.

`fetch()` downloads + caches labeled challenges (OP post text + argument text + timestamp + success).
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/winning-args-corpus.zip"
CACHE = "data/cmv_challenges.json"


def fetch(op_chars=900, arg_chars=1200):
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "swm"}),
                                 timeout=180).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    utts = [json.loads(l) for l in
            z.open("winning-args-corpus/utterances.jsonl").read().decode().splitlines() if l.strip()]
    by_id = {u["id"]: u for u in utts}
    rows = []
    for u in utts:
        s = str(u["meta"].get("success"))
        if s not in ("0", "1"):
            continue
        ts = u.get("timestamp")
        if ts in (None, "None"):
            continue
        root = by_id.get(u.get("root"))
        if not root:
            continue
        rows.append({
            "id": u["id"], "op_id": u.get("root"), "op_user": root.get("user"),
            "challenger": u.get("user"), "ts": float(ts),
            "op_text": (root.get("text") or "")[:op_chars],
            "arg_text": (u.get("text") or "")[:arg_chars],
            "success": int(s),
        })
    rows.sort(key=lambda r: r["ts"])
    Path("data").mkdir(exist_ok=True)
    Path(CACHE).write_text(json.dumps(rows))
    print(f"wrote {len(rows)} labeled CMV challenges -> {CACHE} "
          f"(delta rate {sum(r['success'] for r in rows)/len(rows):.3f})")
    return rows


def load():
    if not Path(CACHE).exists():
        fetch()
    return json.loads(Path(CACHE).read_text())


if __name__ == "__main__":
    fetch()
