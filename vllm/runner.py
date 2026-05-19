#!/usr/bin/env python3
"""Client runner — SSH local port-forward to the vLLM BE, then talk to localhost.

If a compatible server is already reachable on the local port (e.g. a
previously-established forward), the SSH setup is skipped and the existing
tunnel is reused (and left running on exit). Stdlib only.

Examples:
  python runner.py --ssh-target user@xeon --remote-port 9001 \
                   --prompt "Hello" --stream
  python runner.py --no-forward --remote-host 127.0.0.1 --remote-port 9001 \
                   --prompt "Hi"
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_proc: subprocess.Popen | None = None
_started_by_us = False


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except (urllib.error.URLError, ValueError, OSError):
        return None, None


def _health_ok(port: int) -> bool:
    status, body = _http_get(f"http://127.0.0.1:{port}/health")
    return status == 200 and isinstance(body, dict) and "model_status" in body


def _cleanup() -> None:
    global _proc
    if _proc is not None and _started_by_us and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _ensure_forward(args) -> None:
    global _proc, _started_by_us
    lp = args.local_port

    if _port_open(lp):
        if _health_ok(lp):
            print(f"[runner] reusing existing forward/server on :{lp}",
                  file=sys.stderr)
            return
        sys.exit(
            f"[runner] port {lp} is in use by an unrelated process; "
            f"choose another --local-port"
        )

    if args.no_forward:
        sys.exit(f"[runner] --no-forward set but nothing listening on :{lp}")
    if not args.ssh_target:
        sys.exit("[runner] --ssh-target (or RUNNER_SSH) is required to forward")

    cmd = [
        "ssh", "-N",
        "-L", f"127.0.0.1:{lp}:{args.remote_host}:{args.remote_port}",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if args.ssh_extra:
        cmd += args.ssh_extra.split()
    cmd.append(args.ssh_target)

    print(f"[runner] opening forward :{lp} -> "
          f"{args.ssh_target}:{args.remote_host}:{args.remote_port}",
          file=sys.stderr)
    _proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    _started_by_us = True

    deadline = time.monotonic() + args.connect_timeout
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            err = _proc.stderr.read().decode() if _proc.stderr else ""
            sys.exit(f"[runner] ssh exited early:\n{err}")
        if _health_ok(lp):
            print("[runner] forward ready", file=sys.stderr)
            return
        time.sleep(0.3)
    sys.exit(f"[runner] timed out waiting for :{lp} to be ready")


def _wait_until_ready(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, body = _http_get(f"http://127.0.0.1:{port}/health")
        if isinstance(body, dict) and body.get("model_status") == "ready":
            return
        print("[runner] model not ready yet, waiting...", file=sys.stderr)
        time.sleep(5)
    sys.exit("[runner] model did not become ready in time")


def _request(args) -> None:
    base = f"http://127.0.0.1:{args.local_port}"
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": args.stream,
    }
    if args.stop:
        payload["stop"] = args.stop
    if args.seed is not None:
        payload["seed"] = args.seed

    if args.wait:
        _wait_until_ready(args.local_port, args.wait)

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/generate", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[runner] HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.exit(f"[runner] request failed: {e}")

    if args.stream:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                print()
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if obj.get("delta"):
                print(obj["delta"], end="", flush=True)
    else:
        body = json.loads(resp.read().decode())
        print(body.get("text", ""))
        print(
            f"[runner] {body.get('finish_reason')} "
            f"(prompt={body.get('prompt_tokens')}, "
            f"completion={body.get('completion_tokens')})",
            file=sys.stderr,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM BE client with SSH forward")
    p.add_argument("--ssh-target", default=_env("RUNNER_SSH"),
                   help="user@host for ssh -L (or RUNNER_SSH)")
    p.add_argument("--remote-host", default="127.0.0.1",
                   help="host as seen from the ssh server (default 127.0.0.1)")
    p.add_argument("--remote-port", type=int, default=9001)
    p.add_argument("--local-port", type=int,
                   default=int(_env("RUNNER_LOCAL_PORT", "9001")))
    p.add_argument("--ssh-extra", default="",
                   help="extra ssh args, e.g. '-p 2222 -i ~/.ssh/key'")
    p.add_argument("--no-forward", action="store_true",
                   help="talk directly, skip SSH setup")
    p.add_argument("--prompt", required=True, help="'-' reads stdin")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--stop", action="append")
    p.add_argument("--seed", type=int)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--connect-timeout", type=float, default=20.0)
    p.add_argument("--wait", type=float, default=0.0,
                   help="poll /health up to N seconds until model is ready")
    args = p.parse_args()

    if args.prompt == "-":
        args.prompt = sys.stdin.read()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    _ensure_forward(args)
    _request(args)


if __name__ == "__main__":
    main()
