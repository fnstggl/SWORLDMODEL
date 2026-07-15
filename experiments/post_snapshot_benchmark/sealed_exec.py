"""OS-isolated forecast launcher with an allowlisted local API relay.

The child process is constrained by macOS Sandbox.  It can read code and the
blinded forecast packet, but explicit deny rules cover every resolution store,
pseudonym mapping, canonical source archive, credential source, and unblinded
selection artifact.  Its only network destination is a loopback CONNECT proxy;
that parent proxy permits exactly ``api.deepseek.com:443``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from experiments.post_snapshot_benchmark.credentials import read_deepseek_key


ALLOWED_CONNECT = "api.deepseek.com:443"
ROOT = Path(__file__).resolve().parents[2]
UNBLINDED_FILES = (
    ROOT / "experiments/results/post_snapshot_benchmark/representative_vault.json",
    ROOT / "experiments/results/post_snapshot_benchmark/eligible_candidate_pool.json",
    ROOT / "experiments/results/post_snapshot_benchmark/event_world_clustering_map.json",
    ROOT / "experiments/results/post_snapshot_benchmark/frozen_selection_manifest.json",
    ROOT / "experiments/results/post_snapshot_benchmark/market_snapshots.json",
    ROOT / "experiments/replay_vault/SEALED_resolutions.json",
    ROOT / "experiments/replay_vault/blinding_mappings.json",
    ROOT / "experiments/replay_vault_v2/SEALED_resolutions_v2.json",
    ROOT / "experiments/replay_vault_v2/blinding_mappings.json",
)


class _RelayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.allowed = 0
        self.rejected = 0
        self.destinations = set()

    def record(self, destination, allowed):
        with self.lock:
            self.destinations.add(destination)
            if allowed:
                self.allowed += 1
            else:
                self.rejected += 1


class _ConnectHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline(8192).decode("latin1", "replace").strip()
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            self.server.state.record("non-CONNECT", False)
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
            return
        destination = parts[1].lower()
        while True:
            header = self.rfile.readline(8192)
            if header in (b"\r\n", b"\n", b""):
                break
        if destination != ALLOWED_CONNECT:
            self.server.state.record(destination, False)
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
            return
        self.server.state.record(destination, True)
        try:
            upstream = socket.create_connection(("api.deepseek.com", 443), timeout=30)
        except OSError:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            return
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()
        sockets = [self.connection, upstream]
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], 60)
                if not readable:
                    break
                for source in readable:
                    try:
                        data = source.recv(65536)
                    except (ConnectionResetError, BrokenPipeError):
                        return
                    if not data:
                        return
                    try:
                        (upstream if source is self.connection else self.connection).sendall(data)
                    except (ConnectionResetError, BrokenPipeError):
                        return
        finally:
            upstream.close()


class _ThreadedProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _quote_profile(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _profile(*, port: int, output_parent: Path, denied_roots: list[Path], credential_source: Path) -> str:
    deny_lines = []
    for path in [*denied_roots, credential_source, *UNBLINDED_FILES]:
        kind = "subpath" if path.is_dir() else "literal"
        deny_lines.append(f'(deny file-read* ({kind} "{_quote_profile(path)}"))')
    return "\n".join([
        "(version 1)", "(deny default)", '(import "system.sb")',
        "(allow process*)", "(allow file-read*)",
        f'(allow file-write* (subpath "{_quote_profile(output_parent)}") (literal "/dev/null"))',
        "(deny network*)",
        f'(allow network-outbound (remote ip "localhost:{port}"))',
        *deny_lines,
    ]) + "\n"


def _run_sandbox(profile_path: Path, command: list[str], *, env: dict, pass_fds=()):
    return subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile_path), *command],
                          cwd=ROOT, env=env, pass_fds=pass_fds, text=True)


def _probe(profile_path: Path, env: dict, denied_roots: list[Path]) -> dict:
    python = sys.executable
    locked = denied_roots[0] / "locked_test_resolutions.json"
    file_code = ("from pathlib import Path; p=Path(" + repr(str(locked)) + "); "
                 "\ntry: p.read_bytes(); raise SystemExit(9)\nexcept PermissionError: raise SystemExit(0)")
    file_result = _run_sandbox(profile_path, [python, "-c", file_code], env=env)
    map_path = denied_roots[0] / "pseudonym_mappings.json"
    map_code = ("from pathlib import Path; p=Path(" + repr(str(map_path)) + "); "
                "\ntry: p.read_bytes(); raise SystemExit(9)\nexcept PermissionError: raise SystemExit(0)")
    map_result = _run_sandbox(profile_path, [python, "-c", map_code], env=env)
    inherited = ROOT / "experiments/replay_vault_v2/SEALED_resolutions_v2.json"
    inherited_code = ("from pathlib import Path; p=Path(" + repr(str(inherited)) + "); "
                      "\ntry: p.read_bytes(); raise SystemExit(9)\nexcept PermissionError: raise SystemExit(0)")
    inherited_result = _run_sandbox(profile_path, [python, "-c", inherited_code], env=env)
    internet_code = ("import urllib.request\n"
                     "try: urllib.request.urlopen('https://example.com', timeout=8); raise SystemExit(9)\n"
                     "except Exception: raise SystemExit(0)")
    internet_result = _run_sandbox(profile_path, [python, "-c", internet_code], env=env)
    api_code = ("import urllib.error,urllib.request\n"
                "try: urllib.request.urlopen('https://api.deepseek.com/models', timeout=15)\n"
                "except urllib.error.HTTPError as e: raise SystemExit(0 if e.code in (401,403,404) else 9)\n"
                "except Exception: raise SystemExit(9)")
    api_result = _run_sandbox(profile_path, [python, "-c", api_code], env=env)
    return {
        "locked_resolution_read_denied": file_result.returncode == 0,
        "pseudonym_mapping_read_denied": map_result.returncode == 0,
        "inherited_resolution_store_read_denied": inherited_result.returncode == 0,
        "open_internet_request_denied": internet_result.returncode == 0,
        "deepseek_allowlist_endpoint_reachable": api_result.returncode == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-source", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--forecast-input", type=Path,
                        default=ROOT / "experiments/results/post_snapshot_benchmark/blinded_forecast_input.json")
    parser.add_argument("--capsule-root", type=Path,
                        default=ROOT / "experiments/results/post_snapshot_benchmark/capsules")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/post_snapshot_benchmark/preflight_forecasts.jsonl")
    parser.add_argument("--split", choices=("calibration", "validation", "locked_test"),
                        default="calibration")
    parser.add_argument("--world-limit", type=int)
    parser.add_argument("--worker", choices=("v2", "baselines"), default="v2")
    parser.add_argument("--v2-forecasts", type=Path)
    parser.add_argument("--isolation-manifest", type=Path,
                        default=ROOT / "experiments/results/post_snapshot_benchmark/isolation_manifest.json")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    state = _RelayState()
    proxy = _ThreadedProxy(("127.0.0.1", 0), _ConnectHandler)
    proxy.state = state
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    port = proxy.server_address[1]
    output_parent = args.output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    denied = [args.scorer_root.resolve(), args.archive_root.resolve()]
    profile = _profile(port=port, output_parent=output_parent, denied_roots=denied,
                       credential_source=args.credential_source.resolve())
    profile_file = Path(tempfile.mkstemp(prefix="wmv2-forecast-", suffix=".sb")[1])
    profile_file.write_text(profile)
    env = dict(os.environ)
    env.update({"HTTPS_PROXY": f"http://127.0.0.1:{port}",
                "HTTP_PROXY": f"http://127.0.0.1:{port}", "NO_PROXY": "",
                "PYTHONDONTWRITEBYTECODE": "1", "WMV2_OS_SANDBOX": "macos-sandbox-exec-v1"})
    try:
        probes = _probe(profile_file, env, denied)
        manifest = {
            "schema_version": 1,
            "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "isolation_technology": "macOS sandbox-exec plus loopback CONNECT allowlist relay",
            "sandbox_profile_sha256": hashlib.sha256(profile.encode()).hexdigest(),
            "allowed_external_destination": ALLOWED_CONNECT,
            "denied_roots": [str(path) for path in denied],
            "denied_unblinded_files": [str(path) for path in UNBLINDED_FILES],
            "credential_transport": "parent-memory to inherited anonymous pipe",
            "resolution_store_mounted": False, "pseudonym_mapping_mounted": False,
            "open_internet_disabled": True, "probes": probes,
            "all_isolation_probes_pass": all(probes.values()),
        }
        args.isolation_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not manifest["all_isolation_probes_pass"]:
            raise SystemExit("OS isolation preflight failed; forecast worker was not started")
        if args.probe_only:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        key = read_deepseek_key(args.credential_source.resolve())
        read_fd, write_fd = os.pipe()
        os.write(write_fd, key.encode())
        os.close(write_fd)
        worker_script = (ROOT / "experiments/post_snapshot_benchmark/forecast.py" if args.worker == "v2"
                         else ROOT / "experiments/post_snapshot_benchmark/baselines.py")
        command = [sys.executable, str(worker_script),
                   "--credential-fd", str(read_fd), "--forecast-input", str(args.forecast_input.resolve()),
                   "--capsule-root", str(args.capsule_root.resolve()), "--output", str(args.output.resolve()),
                   "--split", args.split]
        if args.worker == "baselines":
            if args.v2_forecasts is None:
                raise SystemExit("--v2-forecasts is required for the baseline worker")
            command.extend(["--v2-forecasts", str(args.v2_forecasts.resolve())])
        if args.world_limit is not None:
            command.extend(["--world-limit", str(args.world_limit)])
        result = _run_sandbox(profile_file, command, env=env, pass_fds=(read_fd,))
        os.close(read_fd)
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        proxy.shutdown()
        proxy.server_close()
        profile_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
