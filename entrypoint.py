#!/usr/bin/env python3
"""
Container entrypoint for the OpenEnv Delivery Tracker benchmark.

Lifecycle:
  1. Starts the health-check HTTP server on $PORT (default 8080) as a daemon.
  2. Waits for the health server to become responsive.
  3. Runs inference.py against all tasks (or the subset in $TASKS).
  4. Exits cleanly (daemon thread dies with main process).

Usage (inside container):
  python entrypoint.py                         # all tasks
  TASKS=easy,medium python entrypoint.py       # subset

Usage (local):
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o
  export OPENAI_API_KEY=sk-...
  python entrypoint.py
"""

from __future__ import annotations

import os
import sys
import time
import threading
import urllib.request

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _start_healthcheck() -> None:
    """Launch the healthcheck HTTP server as a daemon thread."""
    from healthcheck import main as healthcheck_main
    t = threading.Thread(target=healthcheck_main, daemon=True, name="healthcheck")
    t.start()


def _wait_for_health(port: int, timeout: float = 10.0) -> bool:
    """Block until /healthz responds or timeout expires."""
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _run_inference() -> None:
    """Import and run the inference main function."""
    from inference import main as inference_main
    inference_main()


def _should_keep_alive() -> bool:
    """Whether to keep the process alive after inference finishes."""
    value = os.environ.get("KEEP_ALIVE_AFTER_RUN", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))

    # ── 1. Start health server ───────────────────────────────────────
    print(f"[entrypoint] Starting health-check server on port {port}...", flush=True)
    _start_healthcheck()

    # ── 2. Wait until /healthz is responsive ─────────────────────────
    if _wait_for_health(port):
        print("[entrypoint] Health server ready ✓", flush=True)
    else:
        print("[entrypoint] WARNING: Health server did not respond within 10 s", flush=True)

    # ── 3. Run inference ─────────────────────────────────────────────
    print("[entrypoint] Starting inference runner...", flush=True)
    _run_inference()

    print("[entrypoint] Done.", flush=True)

    if _should_keep_alive():
        print("[entrypoint] KEEP_ALIVE_AFTER_RUN enabled; serving health endpoints.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[entrypoint] Shutting down.", flush=True)
