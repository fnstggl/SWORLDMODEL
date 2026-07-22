"""Localhost server for the Social World Model — Lean V2 replay viewer.

Pure Python stdlib (no dependencies). Serves the static frontend and the recorded runs, and
optionally triggers a fresh live recording.

    python -m swm_viz.server               # serve on http://localhost:8756
    python -m swm_viz.server --port 9000

Routes:
    GET  /                      -> frontend/index.html
    GET  /app.js, /styles.css   -> frontend assets
    GET  /recordings/<file>     -> a recording JSON (or index.json)
    POST /api/run  {case_id}    -> launch a real lean_v2 recording run in the background
    GET  /api/run/status        -> status of the last launched run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
RECORDINGS = ROOT / "recordings"

_MIME = {".html": "text/html", ".js": "application/javascript", ".css": "text/css",
         ".json": "application/json", ".svg": "image/svg+xml"}

_run_state = {"running": False, "case_id": None, "returncode": None, "log_tail": ""}
_run_lock = threading.Lock()


def _launch_run(case_id: str) -> None:
    with _run_lock:
        if _run_state["running"]:
            return
        _run_state.update(running=True, case_id=case_id, returncode=None, log_tail="")

    def _worker():
        log = RECORDINGS / "_run.log"
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "swm_viz.record_run", case_id],
                cwd=str(ROOT.parent), stdout=log.open("w"), stderr=subprocess.STDOUT)
            proc.wait()
            rc = proc.returncode
        except Exception as e:  # noqa: BLE001
            rc = -1
            _run_state["log_tail"] = f"{type(e).__name__}: {e}"
        with _run_lock:
            _run_state.update(running=False, returncode=rc)
            try:
                _run_state["log_tail"] = (RECORDINGS / "_run.log").read_text()[-1200:]
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_worker, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, base: Path, rel: str) -> Path | None:
        p = (base / rel).resolve()
        try:
            p.relative_to(base.resolve())
        except ValueError:
            return None
        return p

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file(FRONTEND / "index.html")
        if path == "/api/run/status":
            with _run_lock:
                return self._send(200, dict(_run_state))
        if path.startswith("/recordings/"):
            p = self._safe(RECORDINGS, path[len("/recordings/"):])
            if p and p.is_file():
                return self._file(p)
            return self._send(404, {"error": "not found"})
        # frontend assets
        p = self._safe(FRONTEND, path.lstrip("/"))
        if p and p.is_file():
            return self._file(p)
        return self._send(404, {"error": "not found", "path": path})

    def do_POST(self):
        if self.path.split("?")[0] == "/api/run":
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b"{}"
            try:
                case_id = (json.loads(body or b"{}") or {}).get("case_id", "banxico")
            except Exception:  # noqa: BLE001
                case_id = "banxico"
            with _run_lock:
                if _run_state["running"]:
                    return self._send(409, {"error": "a run is already in progress",
                                            "case_id": _run_state["case_id"]})
            _launch_run(str(case_id))
            return self._send(202, {"launched": True, "case_id": case_id})
        return self._send(404, {"error": "not found"})

    def _file(self, p: Path):
        ctype = _MIME.get(p.suffix, "application/octet-stream")
        self._send(200, p.read_bytes(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    idx = RECORDINGS / "index.json"
    n = len(json.loads(idx.read_text()).get("recordings", [])) if idx.exists() else 0
    print(f"\n  Social World Model — Lean V2 replay")
    print(f"  ▶  http://{args.host}:{args.port}/     ({n} recording(s) available)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
