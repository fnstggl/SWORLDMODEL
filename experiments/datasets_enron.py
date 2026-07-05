"""Enron email reply-prediction dataset for the response backtest (streamed from the CMU tarball).

The canonical individual-response case, matching the product wedge exactly: sender -> recipient
message; does the recipient REPLY? entity = recipient (email address — a heavy repeat entity),
segment = recipient domain, outcome = a reply from the recipient in the same subject thread within a
window. As-of correct: recipient reply-rate state is built only from earlier messages.

Streamed and capped so we don't extract 500k files. Reply labels via normalized-subject threading
(recipient -> sender, "Re:" or same subject, after the send, within the window) — the same
anti-inflation logic as swm/ingestion/store.
"""
from __future__ import annotations

import email
import gzip
import io
import json
import re
import tarfile
import time
import urllib.request
from email.utils import parsedate_to_datetime, getaddresses
from pathlib import Path

CACHE = "data/enron_messages.json"
URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
MFN = ["subj_len", "n_recipients", "is_reply", "has_question", "body_len_log"]
_WS = re.compile(r"\s+")


def _norm_subject(s):
    s = (s or "").lower()
    s = re.sub(r"^(re|fw|fwd)\s*:\s*", "", s).strip()
    while re.match(r"^(re|fw|fwd)\s*:", s):
        s = re.sub(r"^(re|fw|fwd)\s*:\s*", "", s).strip()
    return _WS.sub(" ", s)[:120]


def fetch(cap_messages=120000, timeout=600):
    """Stream the tarball, parse up to cap_messages, cache compact per-message records."""
    t0 = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "swm"})
    resp = urllib.request.urlopen(req, timeout=120)
    msgs = []
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or len(msgs) >= cap_messages:
                if len(msgs) >= cap_messages:
                    break
                continue
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                msg = email.message_from_bytes(f.read())    # read fully (stream isn't seekable)
            except Exception:
                continue
            frm = getaddresses([msg.get("From", "")])
            to = getaddresses(msg.get_all("To", []) or [])
            if not frm or not frm[0][1] or not to:
                continue
            try:
                dt = parsedate_to_datetime(msg.get("Date", ""))
                ts = dt.timestamp()
            except Exception:
                continue
            sender = frm[0][1].lower()
            recips = [a[1].lower() for a in to if a[1] and "@" in a[1]][:5]
            if not recips:
                continue
            subj = msg.get("Subject", "") or ""
            # crude body length
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode("utf-8", "ignore")[:2000]
                        except Exception:
                            body = ""
                        break
            else:
                try:
                    body = (msg.get_payload(decode=True) or b"").decode("utf-8", "ignore")[:2000]
                except Exception:
                    body = ""
            msgs.append({"ts": ts, "from": sender, "to": recips, "subj": subj,
                         "nsubj": _norm_subject(subj), "is_reply": 1 if re.match(r"^\s*re\s*:", subj.lower()) else 0,
                         "body_len": len(body), "has_q": 1 if "?" in subj or "?" in body[:500] else 0})
            if len(msgs) % 20000 == 0:
                print(f"  parsed {len(msgs)} messages ({time.time()-t0:.0f}s)", flush=True)
            if time.time() - t0 > timeout:
                break
    Path("data").mkdir(exist_ok=True)
    Path(CACHE).write_text(json.dumps(msgs))
    print(f"wrote {len(msgs)} messages -> {CACHE} ({time.time()-t0:.0f}s)")
    return msgs


def load_samples(window_days=14, cap=42000):
    if not Path(CACHE).exists():
        fetch()
    msgs = json.loads(Path(CACHE).read_text())
    # keep only messages with sane timestamps (Enron has some garbage dates), then cap to a
    # contiguous recent slice so the pure-python GBDT rung is feasible.
    msgs = [m for m in msgs if 9e8 < m["ts"] < 1.1e9]     # ~1998-2005 sanity window
    msgs.sort(key=lambda m: m["ts"])
    if cap and len(msgs) > cap:
        msgs = msgs[-cap:]
    # index inbound messages by (from, nsubj) -> sorted times, to detect replies
    from collections import defaultdict
    by_pair = defaultdict(list)
    for m in msgs:
        for r in m["to"]:
            by_pair[(m["from"], r, m["nsubj"])].append(m["ts"])   # (A->B thread) sends
    # for reply detection we need B->A messages: index sender->recipient->nsubj times
    reply_idx = defaultdict(list)
    for m in msgs:
        for r in m["to"]:
            reply_idx[(m["from"], r, m["nsubj"])].append(m["ts"])
    W = window_days * 86400
    samples = []
    for m in msgs:
        # treat each (sender -> first recipient) as one outbound send; label reply within W
        recipient = m["to"][0]
        # replied if recipient sent a message back to sender in same thread after this, within W
        cand = reply_idx.get((recipient, m["from"], m["nsubj"]), [])
        replied = 1 if any(m["ts"] < ct <= m["ts"] + W for ct in cand) else 0
        seg = recipient.split("@")[-1] if "@" in recipient else "other"
        import math
        mf = {"subj_len": min(1.0, len(m["subj"]) / 60), "n_recipients": min(1.0, len(m["to"]) / 5),
              "is_reply": float(m["is_reply"]), "has_question": float(m["has_q"]),
              "body_len_log": math.log1p(m["body_len"]) / 10.0}
        samples.append((recipient, seg, mf, replied))
    return samples, MFN


if __name__ == "__main__":
    fetch()
