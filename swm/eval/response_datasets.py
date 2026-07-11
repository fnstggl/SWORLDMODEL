"""Loaders for LABELED response/engagement datasets — the ones that hit OSim's intended advantage.

  * Upworthy Research Archive (CC-BY, randomized headline A/B → real clicks): the causal headline-choice set.
    Variants grouped by clickability_test_id; empirical CTR + winner per distinct headline. Supports a
    precision@1 / pairwise-accuracy "which headline wins" eval — OSim/engagement's on-point benchmark.
  * Enron reply + delay (public, leak-free): reconstruct threads from Message-ID / In-Reply-To / References;
    a sent message is "replied-to" iff a later message references its Message-ID; delay = time gap. Split
    TIME-FORWARD (train on earlier weeks, test on later) so no future leaks. Individual reply-occurrence +
    response-delay — OSim's core claimed strength (#1).

These map to `docs/DATASET_REGISTRY.md`. Nothing is trained; loaders only. Downloads are documented so they run
identically on a GPU pod.
"""
from __future__ import annotations

import csv
import email
import os
import time
from collections import defaultdict

# Upworthy exploratory CSV (14 MB) — OSF direct download (see docs/DATASET_REGISTRY.md)
UPWORTHY_URL = "https://osf.io/download/3vqmp/"


def download_upworthy(path="data/upworthy_exploratory.csv"):
    import urllib.request
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(UPWORTHY_URL, path)
    return path


def load_upworthy_tests(path="data/upworthy_exploratory.csv", *, min_impressions=1000, min_variants=2,
                        max_variants=4, limit=None):
    """Group variants by clickability_test_id → [{test_id, variants:[{headline, ctr, impressions, clicks}],
    winner_headline}]. Winner = highest empirical CTR among variants with enough impressions (randomized, so
    CTR difference is causal). Only tests with 2-4 distinct headlines are returned."""
    csv.field_size_limit(10_000_000)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # test_id -> headline -> [impr, clicks]
    with open(path) as f:
        for r in csv.DictReader(f):
            tid, h = r.get("clickability_test_id"), (r.get("headline") or "").strip()
            try:
                impr, clk = int(r.get("impressions") or 0), int(r.get("clicks") or 0)
            except ValueError:
                continue
            if tid and h:
                agg[tid][h][0] += impr
                agg[tid][h][1] += clk
    tests = []
    for tid, heads in agg.items():
        variants = [{"headline": h, "impressions": im, "clicks": ck, "ctr": (ck / im if im else 0.0)}
                    for h, (im, ck) in heads.items() if im >= min_impressions]
        if not (min_variants <= len(variants) <= max_variants):
            continue
        variants.sort(key=lambda v: -v["ctr"])
        tests.append({"test_id": tid, "variants": variants, "winner_headline": variants[0]["headline"]})
    return tests[:limit] if limit else tests


def score_headline_ranking(tests, rank_fn):
    """rank_fn(list_of_headlines) -> ordered headlines (best first). Scores precision@1 (picked the empirical
    winner) + pairwise accuracy (fraction of headline pairs ordered correctly by CTR)."""
    p1 = pairs_ok = pairs_tot = n = 0
    for t in tests:
        heads = [v["headline"] for v in t["variants"]]
        ctr = {v["headline"]: v["ctr"] for v in t["variants"]}
        order = rank_fn(heads)
        if not order:
            continue
        n += 1
        p1 += (order[0] == t["winner_headline"])
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if ctr.get(order[i], 0) == ctr.get(order[j], 0):
                    continue
                pairs_tot += 1
                pairs_ok += (ctr.get(order[i], 0) > ctr.get(order[j], 0))
    return {"n_tests": n, "precision_at_1": (round(p1 / n, 3) if n else None),
            "pairwise_accuracy": (round(pairs_ok / pairs_tot, 3) if pairs_tot else None),
            "random_p1": (round(sum(1 / len(t["variants"]) for t in tests[:n]) / n, 3) if n else None)}


# ---------------------------------------------------------------- Enron reply + delay (leak-free)
def load_enron_reply_delay(maildir, *, limit_messages=None):
    """Reconstruct reply-occurrence + delay from an Enron maildir (download: cs.cmu.edu/~enron/, ~1.7GB).
    Returns [{msg_id, from, to, subject, body, date_ts, replied, delay_hours}]. `replied` = a later message
    references this Message-ID (In-Reply-To/References). LEAK-FREE downstream: split by date_ts (train<cut<test).
    """
    by_id, refs_to = {}, defaultdict(list)
    n = 0
    for root, _, files in os.walk(maildir):
        for fn in files:
            try:
                with open(os.path.join(root, fn), errors="ignore") as f:
                    msg = email.message_from_file(f)
            except Exception:
                continue
            mid = (msg.get("Message-ID") or "").strip()
            if not mid:
                continue
            try:
                ts = time.mktime(email.utils.parsedate(msg.get("Date")))
            except (TypeError, ValueError):
                continue
            payload = msg.get_payload()
            body = payload if isinstance(payload, str) else ""
            rec = {"msg_id": mid, "from": msg.get("From", ""), "to": msg.get("To", ""),
                   "subject": msg.get("Subject", ""), "body": body[:4000], "date_ts": ts,
                   "replied": False, "delay_hours": None}
            by_id[mid] = rec
            for ref in (msg.get("In-Reply-To", "") + " " + msg.get("References", "")).split():
                refs_to[ref.strip()].append((mid, ts))
            n += 1
            if limit_messages and n >= limit_messages:
                break
    # signal 1: a LATER message references this Message-ID (headers — rare in Enron: most 2000-era clients
    # never set In-Reply-To, which is why header-only reconstruction finds ~0 replies)
    for mid, rec in by_id.items():
        later = sorted([(ts, rid) for rid, ts in refs_to.get(mid, []) if ts > rec["date_ts"]])
        if later:
            rec["replied"] = True
            rec["delay_hours"] = round((later[0][0] - rec["date_ts"]) / 3600.0, 2)
    # signal 2 (the workhorse): SUBJECT+DIRECTION matching — a later message from B→A whose normalized
    # subject (Re:/Fw: stripped) equals an earlier A→B message's, within 30 days, is a reply to the LATEST
    # such message. Standard Enron reconstruction; conservative window; first reply wins.
    import re as _re
    def _norm(s):
        return _re.sub(r"^\s*((re|fw|fwd)\s*:\s*)+", "", str(s or "").strip().lower())[:120]
    def _addr(s):
        return str(s or "").split(",")[0].strip().lower()
    by_key = defaultdict(list)                 # (sender, recipient, norm_subject) -> [(ts, rec)]
    for rec in by_id.values():
        s, r_ = _addr(rec["from"]), _addr(rec["to"])
        if s and r_ and _norm(rec["subject"]):
            by_key[(s, r_, _norm(rec["subject"]))].append((rec["date_ts"], rec))
    for lst in by_key.values():
        lst.sort(key=lambda x: x[0])
    for rec in by_id.values():
        s, r_ = _addr(rec["from"]), _addr(rec["to"])
        subj = _norm(rec["subject"])
        if not (s and r_ and subj):
            continue
        # this message replies to the latest earlier r_→s message with the same normalized subject
        cands = [orig for ts, orig in by_key.get((r_, s, subj), [])
                 if ts < rec["date_ts"] <= ts + 30 * 86400.0]
        if cands:
            orig = cands[-1]
            delay = round((rec["date_ts"] - orig["date_ts"]) / 3600.0, 2)
            if not orig["replied"] or (orig["delay_hours"] is not None and delay < orig["delay_hours"]):
                orig["replied"] = True
                orig["delay_hours"] = delay
    return list(by_id.values())


def time_forward_split(records, *, test_frac=0.3):
    """Leak-free split: sort by date, earliest (1-test_frac) = train, latest test_frac = test."""
    rs = sorted([r for r in records if r.get("date_ts")], key=lambda r: r["date_ts"])
    cut = int(len(rs) * (1 - test_frac))
    return rs[:cut], rs[cut:]
