"""OmniBehavior harness — longitudinal individual-behavior realism on REAL user traces (now released).

Dataset: jiawei-ucas/OmniBehavior (CC-BY-NC-SA 4.0 — benchmark/research use only; 200 users × 90 days of real
Kuaishou traces, EN+ZH). This is the benchmark whose paper documented the failure modes that matter for our
individual/engagement class: hyper-activity (LLMs predict 40-60% action rates vs ~10% real), positivity bias,
persona homogenization. Best published model scored 44.55 — hard and unsolved.

Task built here (a faithful, compact slice of the paper's behavior-prediction family): for a user, given the
PROFILE + a chronological HISTORY PREFIX (compact event summaries) + the NEXT exposure's context, predict
whether the user takes a positive action (conversion/like/follow/comment/share/collect/click) on it — binary,
time-ordered (no future leaks), scored with accuracy/F1/Brier + the realism deltas (predicted vs real action
rate). Sampling: the SMALLEST English users (the sandbox can't hold the 2.8GB tail), fixed deterministic list.

Caveats stated plainly: Kuaishou short-video/e-commerce domain ≠ our core deliberation class; this measures
the INDIVIDUAL/engagement path's realism, not election forecasting. Same-arm interface as the other pilots:
any `sample(prompt)->text` (DeepSeek here, OSim on a GPU pod).
"""
from __future__ import annotations

import json
import os
import urllib.request

_API = "https://huggingface.co/api/datasets/jiawei-ucas/OmniBehavior/tree/main/raw_user_data/en"
_RES = "https://huggingface.co/datasets/jiawei-ucas/OmniBehavior/resolve/main/"
_POSITIVE = ("conversion", "like", "follow", "comment", "share", "collect", "click", "purchase", "order")

_PROMPT = """You are simulating ONE specific real user of a short-video/e-commerce platform.
WHO THEY ARE: {profile}
THEIR RECENT BEHAVIOR (chronological, real):
{history}

NOW they are shown this:
{exposure}

Real users act on only a small fraction of what they see. As THIS user, given THIS history, do they take any
positive action (click/like/follow/share/purchase/convert) on this exposure — or scroll past?
Return ONLY JSON: {{"acts": true|false, "p": <0..1 probability they act>}}"""


def _get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}),
                                  timeout=timeout).read()


def download_users(n_users=10, max_bytes=1_500_000, cache_dir="data/omnibehavior"):
    """Fetch the n smallest EN users (deterministic: sorted by size then name). Returns list of file paths."""
    os.makedirs(cache_dir, exist_ok=True)
    files = json.loads(_get(_API))
    files = sorted([f for f in files if f.get("type") == "file" and (f.get("size") or 0) < max_bytes],
                   key=lambda f: (f["size"], f["path"]))[:n_users]
    paths = []
    for f in files:
        dst = os.path.join(cache_dir, os.path.basename(f["path"]))
        if not os.path.exists(dst):
            with open(dst, "wb") as out:
                out.write(_get(_RES + f["path"]))
        paths.append(dst)
    return paths


def _acted(event) -> bool:
    acts = event.get("action") or []
    return any(str(a.get("type", "")).lower() in _POSITIVE for a in acts if isinstance(a, dict))


def _summ(event, max_len=110) -> str:
    c = event.get("context") or {}
    core = c.get("caption") or c.get("item_title") or c.get("product") or c.get("live_title") or ""
    acted = "ACTED" if _acted(event) else "passed"
    return f"[{event.get('timestamp', '')[:10]} {event.get('type', '?')}] {str(core)[:max_len]} -> {acted}"


def _exposure(event, max_len=400) -> str:
    c = {k: v for k, v in (event.get("context") or {}).items() if v not in (None, "", [])}
    return f"type={event.get('type', '?')} " + json.dumps(c, ensure_ascii=False)[:max_len]


def build_items(paths, *, prefix_len=12, per_user=8, min_history=15):
    """Time-ordered items: predict action on each of the LAST `per_user` events from the prefix before it.
    Never uses future events. Returns [{profile, history, exposure, y}]."""
    items = []
    for path in paths:
        data = json.load(open(path))
        for uid, u in data.items():
            hist = sorted(u.get("action_history") or [], key=lambda e: str(e.get("timestamp", "")))
            if len(hist) < min_history:
                continue
            for i in range(max(prefix_len, len(hist) - per_user), len(hist)):
                items.append({"user": uid, "profile": u.get("user_profile", ""),
                              "history": "\n".join(_summ(e) for e in hist[max(0, i - prefix_len):i]),
                              "exposure": _exposure(hist[i]), "y": int(_acted(hist[i]))})
    return items


def eval_arm(sample_fn, items, *, verbose=False):
    """Score one arm. Returns accuracy/F1/Brier + hyperactivity delta (pred action rate − real)."""
    from swm.engine.grounding import parse_json
    tp = fp = fn = tn = 0
    briers, n_bad = [], 0
    for it in items:
        r = parse_json(sample_fn(_PROMPT.format(profile=it["profile"], history=it["history"],
                                                exposure=it["exposure"]))) or {}
        if not isinstance(r.get("acts"), bool):
            n_bad += 1
            continue
        pred = int(r["acts"])
        p = r.get("p")
        if isinstance(p, (int, float)):
            briers.append((min(1.0, max(0.0, float(p))) - it["y"]) ** 2)
        tp += (pred and it["y"]); fp += (pred and not it["y"])
        fn += ((not pred) and it["y"]); tn += ((not pred) and not it["y"])
        if verbose:
            print(f"  y={it['y']} pred={pred} p={p}  {it['exposure'][:60]}")
    n = tp + fp + fn + tn
    if not n:
        return {"n": 0, "n_unparsed": n_bad}
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n": n, "n_unparsed": n_bad,
            "accuracy": round((tp + tn) / n, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if (prec + rec) else 0.0,
            "brier": round(sum(briers) / len(briers), 4) if briers else None,
            "real_action_rate": round((tp + fn) / n, 3),
            "pred_action_rate": round((tp + fp) / n, 3),
            "hyperactivity_delta": round((tp + fp) / n - (tp + fn) / n, 3)}
