#!/usr/bin/env python3
"""
Lightweight HTTP health-check server for the OpenEnv Delivery Tracker.

Provides two endpoints:
  GET /healthz   → 200 {"status": "ok", ...}   (liveness probe)
  GET /readyz    → 200 {"ready": true, ...}     (readiness probe — validates env loads)
  GET /reset     → 200 {"observation": ...}     (quick env reset sanity check)

Used by Docker HEALTHCHECK and Kubernetes/HF Spaces probes.

This server is intentionally minimal: stdlib only, no framework dependency.
It runs on port 8080 by default (override with PORT env var).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Pre-load environment to validate at import time
# ---------------------------------------------------------------------------
_BOOT_ERROR: str | None = None
_BOOT_TIME: float = time.time()

try:
    from env.environment import DeliveryEnvironment
    from env.tasks import get_task, TASK_IDS
    from env.graders import TaskGrader
except Exception as exc:
    _BOOT_ERROR = f"Import failed: {exc}"


def _validate_env() -> Dict[str, Any]:
    """
    Run a quick deterministic reset on the 'easy' task.
    Returns a dict with validation results.
    """
    if _BOOT_ERROR:
        return {"valid": False, "error": _BOOT_ERROR}

    try:
        cfg = get_task("easy")
        env = DeliveryEnvironment(cfg)
        obs = env.reset()
        return {
            "valid": True,
            "task_id": obs.task_id,
            "nodes": len(obs.nodes),
            "drivers": len(obs.drivers),
            "deliveries": len(obs.deliveries),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for health probes."""

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")

        if path == "/healthz":
            self._respond_json(200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - _BOOT_TIME, 1),
                "tasks": list(TASK_IDS) if not _BOOT_ERROR else [],
                "boot_error": _BOOT_ERROR,
            })

        elif path == "/readyz":
            result = _validate_env()
            code = 200 if result.get("valid") else 503
            self._respond_json(code, {"ready": result.get("valid", False), **result})

        elif path == "/reset":
            result = self._do_reset()
            code = 200 if "error" not in result else 500
            self._respond_json(code, result)

        else:
            self._respond_json(404, {"error": f"Unknown path: {self.path}"})

    def _do_reset(self) -> Dict[str, Any]:
        """Reset the easy task and return the observation summary."""
        try:
            cfg = get_task("easy")
            env = DeliveryEnvironment(cfg)
            obs = env.reset()
            return {
                "task_id": obs.task_id,
                "step_number": obs.step_number,
                "max_steps": obs.max_steps,
                "drivers": [d.driver_id for d in obs.drivers],
                "deliveries": [d.delivery_id for d in obs.deliveries],
                "available_actions_count": len(obs.available_actions),
            }
        except Exception as exc:
            return {"error": str(exc), "trace": traceback.format_exc()}

    def _respond_json(self, code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stderr logging for clean container output."""
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[healthcheck] Listening on 0.0.0.0:{port}", flush=True)
    print(f"[healthcheck]   GET /healthz  → liveness probe", flush=True)
    print(f"[healthcheck]   GET /readyz   → readiness probe", flush=True)
    print(f"[healthcheck]   GET /reset    → env reset check", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[healthcheck] Shutting down.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
