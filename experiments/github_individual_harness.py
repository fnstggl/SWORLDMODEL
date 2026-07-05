"""EXP-014: REAL individual/entity-response backtest on public GitHub data (GH Archive).

Unblocks the individual model's biggest gap — it was validated only on synthetic data. Here the
outcome is real behavior: an issue is opened on a repo; does it get a response (a comment by a
DIFFERENT user) within a fixed window? The responding ENTITY is the repo/maintainer team — a genuine
repeat entity with evolving responsiveness state, which is exactly the regime the world-model
hypothesis says should beat a raw prior.

  recipient(repo) state_t + action(issue) + context -> P(response) -> repo state_t+1

No cheat: events are processed in time order; a repo's responsiveness state for an issue is built
ONLY from its earlier issues (as-of); the outcome window must be fully elapsed. Data is historical
(settled). Compares segment vs +entity(repo) vs +message via hierarchical partial pooling, sliced by
repo-history depth (cold vs repeat vs deep) — the direct test of "does entity state help?".

Usage:
  python -m experiments.github_individual_harness fetch --date 2024-06-03 --h0 8 --hours 12 --window 6
  python -m experiments.github_individual_harness run
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import urllib.request
from pathlib import Path

from swm.eval.individual_response_eval import evaluate
from swm.eval.metrics import brier_score, expected_calibration_error, log_loss

DATA = "data/gh_issues.json"
RESULT = "experiments/results/exp014_github_individual.json"


def _hour_url(date: str, hour: int) -> str:
    return f"https://data.gharchive.org/{date}-{hour}.json.gz"


def fetch(date: str, h0: int, hours: int, window_h: int, issue_hours: int) -> None:
    """Stream GH Archive hours; collect opened issues + their comments; label response-within-window."""
    issues = {}                      # (repo, number) -> {ts, repo, actor, title, body_len, labels}
    comments = {}                    # (repo, number) -> list of (ts, commenter)
    import time as _t
    from datetime import datetime, timezone

    def to_ts(iso):
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()

    for h in range(h0, h0 + hours):
        url = _hour_url(date, h)
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "swm"}),
                                         timeout=90).read()
            lines = gzip.decompress(raw).split(b"\n")
        except Exception as e:
            print(f"  hour {h}: FAIL {e}"); continue
        n_i = n_c = 0
        for ln in lines:
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            typ = e.get("type")
            if typ == "IssuesEvent" and e["payload"].get("action") == "opened":
                iss = e["payload"]["issue"]
                repo = e["repo"]["name"]
                key = (repo, iss["number"])
                hour_offset = h - h0
                if hour_offset < issue_hours:                 # only issues with a full response window ahead
                    issues[key] = {
                        "ts": to_ts(e["created_at"]), "repo": repo,
                        "actor": e["actor"]["login"],
                        "title": iss.get("title", "") or "",
                        "body_len": len(iss.get("body") or ""),
                        "n_labels": len(iss.get("labels") or []),
                        "author_assoc": iss.get("author_association", "NONE"),
                    }
                    n_i += 1
            elif typ == "IssueCommentEvent":
                iss = e["payload"]["issue"]
                repo = e["repo"]["name"]
                key = (repo, iss["number"])
                comments.setdefault(key, []).append((to_ts(e["created_at"]), e["actor"]["login"]))
                n_c += 1
        print(f"  hour {h}: +{n_i} issues, +{n_c} comments (issues={len(issues)})", flush=True)

    # label: responded if a comment by a DIFFERENT user within window_h hours
    recs = []
    W = window_h * 3600
    for key, iss in issues.items():
        cs = comments.get(key, [])
        responded = any(ct - iss["ts"] <= W and ct >= iss["ts"] and who != iss["actor"]
                        for ct, who in cs)
        iss["responded"] = 1 if responded else 0
        recs.append(iss)
    recs.sort(key=lambda r: r["ts"])
    Path("data").mkdir(exist_ok=True)
    Path(DATA).write_text(json.dumps({"window_h": window_h, "records": recs}))
    rate = sum(r["responded"] for r in recs) / max(1, len(recs))
    print(f"\nwrote {len(recs)} labeled issues -> {DATA}  (response<= {window_h}h rate = {rate:.3f})")


# ------------------------------------------------------------------ features + as-of entity state
def _msg_features(r, repo_state, author_state):
    t = r["title"].lower()
    return {
        "title_len": min(1.0, len(r["title"]) / 80),
        "body_len_log": math.log1p(r["body_len"]) / 10.0,
        "n_labels": min(1.0, r["n_labels"] / 5.0),
        "is_bug": 1.0 if any(k in t for k in ("bug", "error", "crash", "fail", "broken")) else 0.0,
        "is_question": 1.0 if ("?" in r["title"] or t.startswith("how ")) else 0.0,
        "is_member": 1.0 if r["author_assoc"] in ("MEMBER", "OWNER", "COLLABORATOR") else 0.0,
        "repo_past_rate": repo_state,       # as-of repo responsiveness (the entity-state feature)
    }


def run():
    blob = json.loads(Path(DATA).read_text())
    recs = blob["records"]
    window_h = blob["window_h"]
    # build as-of samples for the hierarchical evaluator: (entity=repo, message_features, outcome)
    repo_hist = {}      # repo -> [resp...]
    author_hist = {}
    samples, depths = [], []
    seg = 0.5
    for r in recs:
        rh = repo_hist.get(r["repo"], [])
        repo_rate = (sum(rh) + seg * 3) / (len(rh) + 6) if rh else seg
        mf = _msg_features(r, repo_rate, author_hist.get(r["actor"], []))
        samples.append((r["repo"], mf, r["responded"]))
        depths.append(len(rh))
        repo_hist.setdefault(r["repo"], []).append(r["responded"])
    n = len(samples)
    base = sum(s[2] for s in samples) / n
    mfn = ["title_len", "body_len_log", "n_labels", "is_bug", "is_question", "is_member", "repo_past_rate"]

    # overall evaluation: segment vs +person(repo) vs +message vs full
    res = evaluate(samples, mfn, split=0.7)
    print(f"REAL GitHub issue-response backtest: n={n}, response<= {window_h}h base rate {base:.3f}")
    print(f"  {'regime':<18}{'log_loss':>9}{'brier':>8}{'ece':>7}{'uplift@20':>11}")
    for k, v in res["regimes"].items():
        if "log_loss" in v:
            print(f"  {k:<18}{v['log_loss']:>9.4f}{v['brier']:>8.4f}{v['ece']:>7.4f}{v['uplift@20']:>11.4f}")
    print(f"  individual_beats_segment_logloss: {res['comparison']}")

    # slice by repo-history depth at prediction time (test slice)
    cut = int(0.7 * n)
    test = samples[cut:]
    dtest = depths[cut:]
    from swm.transition.individual_transition import IndividualTransition
    seg_rate = (sum(s[2] for s in samples[:cut]) + 1) / (cut + 2)
    def fit_predict(sources):
        m = IndividualTransition(message_feature_names=mfn, segment_rate=seg_rate,
                                 sources=frozenset(sources))
        m.fit_stream(samples[:cut], segment_rate=seg_rate)
        ps, ys = [], []
        for (eid, mf, o) in test:
            ps.append(min(1 - 1e-6, max(1e-6, m.predict(eid, mf)["p_mean"]))); ys.append(o)
            m.transition(eid, o)
        return ps, ys
    seg_p, y = fit_predict({"segment"})
    full_p, _ = fit_predict({"segment", "person", "message"})
    slices = {"cold_repo(0)": lambda d: d == 0, "repeat_repo(1-4)": lambda d: 1 <= d <= 4,
              "deep_repo(5+)": lambda d: d >= 5}
    print("\n  by repo-history depth (does entity state help more with depth?):")
    print(f"  {'slice':<16}{'n':>5}{'seg_ll':>9}{'full_ll':>9}{'delta':>8}")
    slice_out = {}
    for name, fn in slices.items():
        idx = [i for i, d in enumerate(dtest) if fn(d)]
        if len(idx) < 15 or sum(y[i] for i in idx) < 5:
            continue
        ys = [y[i] for i in idx]
        sl = log_loss(ys, [seg_p[i] for i in idx]); fl = log_loss(ys, [full_p[i] for i in idx])
        slice_out[name] = {"n": len(idx), "seg_ll": round(sl, 4), "full_ll": round(fl, 4),
                           "delta": round(sl - fl, 4)}
        print(f"  {name:<16}{len(idx):>5}{sl:>9.4f}{fl:>9.4f}{sl-fl:>+8.4f}")

    out = {"source": "GitHub Archive (public)", "n": n, "window_h": window_h,
           "base_rate": round(base, 4), "overall": res, "by_repo_depth": slice_out,
           "note": "real entity-response outcomes; individual (repo) state via hierarchical pooling"}
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULT}")
    return out


def score_llm():
    """Compare the individual WORLD MODEL vs raw-LLM tiers (agent predictions) on the same
    state-rich test issues — the direct 'model the moving parts vs individual guessing' test."""
    import glob
    from swm.transition.individual_transition import IndividualTransition
    blob = json.loads(Path(DATA).read_text())
    recs = blob["records"]; window_h = blob["window_h"]
    n = len(recs); cut = int(0.7 * n)
    sub = json.loads(Path("data/gh_llm_common.json").read_text())
    key = {(s["repo"], s["ts"], s["title"]): i for i, s in enumerate(sub)}
    mfn = ["title_len", "body_len_log", "n_labels", "is_bug", "is_question", "is_member", "repo_past_rate"]
    seg = 0.5
    # stream: fit world model on train, predict test, capture the sub issues
    repo_hist = {}
    samples = []
    for r in recs:
        rh = repo_hist.get(r["repo"], [])
        rate = (sum(rh) + seg * 3) / (len(rh) + 6) if rh else seg
        samples.append((r["repo"], _msg_features(r, rate, None), r["responded"], r))
        repo_hist.setdefault(r["repo"], []).append(r["responded"])
    seg_rate = (sum(s[2] for s in samples[:cut]) + 1) / (cut + 2)
    m = IndividualTransition(message_feature_names=mfn, segment_rate=seg_rate,
                             sources=frozenset({"segment", "person", "message"}))
    m.fit_stream([(e, f, o) for e, f, o, _ in samples[:cut]], segment_rate=seg_rate)
    wm = {}
    for eid, f, o, r in samples[cut:]:
        p = m.predict(eid, f)["p_mean"]
        kk = (r["repo"], r["ts"], r["title"])
        if kk in key:
            wm[key[kk]] = p
        m.transition(eid, o)
    # agent preds
    msg, ctx = {}, {}
    for fp in glob.glob("data/gh_llm_pred_*.json"):
        for p in json.loads(Path(fp).read_text()):
            i = int(p["id"].rsplit("#", 1)[1])
            msg[i] = min(0.97, max(0.01, p["p_msg"])); ctx[i] = min(0.97, max(0.01, p["p_ctx"]))
    idx = [i for i in range(len(sub)) if i in wm and i in msg and i in ctx]
    y = [sub[i]["responded"] for i in idx]

    def sc(pick):
        p = [min(1 - 1e-6, max(1e-6, pick(i))) for i in idx]
        return {"n": len(idx), "log_loss": round(log_loss(y, p), 4), "brier": round(brier_score(y, p), 4),
                "ece": round(expected_calibration_error(y, p), 4)}
    tiers = {"raw_llm_message_only": sc(lambda i: msg[i]),
             "raw_llm_with_repo_context": sc(lambda i: ctx[i]),
             "world_model(full_individual)": sc(lambda i: wm[i])}
    print(f"\nWorld model vs raw LLM on {len(idx)} STATE-RICH GitHub issues (repo depth>=1), "
          f"response<= {window_h}h, base {sum(y)/len(y):.3f}:")
    for k, v in tiers.items():
        print(f"  {k:<30} log loss {v['log_loss']:.4f}  brier {v['brier']:.4f}  ece {v['ece']:.4f}")
    # by depth
    depth = {i: sub[i]["repo_depth"] for i in idx}
    for lo, hi, name in [(1, 4, "repeat(1-4)"), (5, 999, "deep(5+)")]:
        js = [k for k in range(len(idx)) if lo <= depth[idx[k]] <= hi]
        if len(js) < 8:
            continue
        yj = [y[k] for k in js]
        def sj(pick):
            return round(log_loss(yj, [min(1 - 1e-6, max(1e-6, pick(idx[k]))) for k in js]), 4)
        print(f"  [{name} n={len(js)}] llm_msg {sj(lambda i: msg[i])}  llm_ctx {sj(lambda i: ctx[i])}"
              f"  world_model {sj(lambda i: wm[i])}")
    out = json.loads(Path(RESULT).read_text()) if Path(RESULT).exists() else {}
    out["llm_comparison_state_rich"] = {"n": len(idx), "base_rate": round(sum(y) / len(y), 4),
                                        "tiers": tiers}
    Path(RESULT).write_text(json.dumps(out, indent=1))
    print(f"  updated {RESULT}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("score-llm")
    f = sub.add_parser("fetch")
    f.add_argument("--date", default="2024-06-03"); f.add_argument("--h0", type=int, default=8)
    f.add_argument("--hours", type=int, default=12); f.add_argument("--window", type=int, default=6)
    f.add_argument("--issue-hours", type=int, default=6)
    sub.add_parser("run")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.date, a.h0, a.hours, a.window, a.issue_hours)
    elif a.cmd == "score-llm":
        score_llm()
    else:
        run()


if __name__ == "__main__":
    main()
