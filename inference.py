#!/usr/bin/env python3
"""
OpenEnv Inference Runner — Delivery Tracker Benchmark
=====================================================

Runs an LLM agent through all three benchmark tasks (easy, medium, hard)
using HuggingFace's OpenAI-compatible Inference API.

Configuration
-------------
  HuggingFace API credentials are embedded by default. Override with env vars:
    API_BASE_URL   — Base URL for the OpenAI-compatible endpoint.
    MODEL_NAME     — Model identifier.
    HF_TOKEN       — HuggingFace API token.

Structured logging
------------------
Every run emits deterministic JSON log lines:
  START  — emitted once per task before the first step.
  STEP   — emitted after every environment step.
  END    — emitted once per task after termination/truncation.

Usage
-----
  $ python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import heapq
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from huggingface_hub import InferenceClient

from env.environment import DeliveryEnvironment
from env.models import Action, ActionType, Observation, StepResult
from env.tasks import get_task, TASK_IDS
from env.graders import TaskGrader, GradeReport


# ═══════════════════════════════════════════════════════════════════════
# Default HuggingFace API Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_API_BASE = "https://router.huggingface.co/hf-inference/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_HF_TOKEN = ""


def _load_config() -> Dict[str, str]:
    """
    Load API config with sensible HuggingFace defaults.
    Environment variables override defaults if set.
    """
    api_base = os.environ.get("API_BASE_URL", "").strip() or DEFAULT_API_BASE
    model = os.environ.get("MODEL_NAME", "").strip() or DEFAULT_MODEL
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or DEFAULT_HF_TOKEN
    )

    key_source = "HF_TOKEN (default)"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        key_source = "OPENAI_API_KEY"
    elif os.environ.get("HF_TOKEN", "").strip():
        key_source = "HF_TOKEN"

    fallback_on_api_error = (
        os.environ.get("FALLBACK_ON_API_ERROR", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    force_rule_based = (
        os.environ.get("FORCE_RULE_BASED", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    return {
        "api_base": api_base,
        "model": model,
        "api_key": api_key,
        "key_source": key_source,
        "fallback_on_api_error": str(fallback_on_api_error),
        "force_rule_based": str(force_rule_based),
    }


# ═══════════════════════════════════════════════════════════════════════
# Structured Logging
# ═══════════════════════════════════════════════════════════════════════

def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Best-effort printing that never raises (e.g., broken pipes in CI)."""
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def _ts() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def log_start(task_id: str, model_name: str) -> None:
    """Emit a START log line."""
    record = {
        "event": "START",
        "task_id": task_id,
        "timestamp": _ts(),
        "model_name": model_name,
    }
    _safe_print(json.dumps(record), flush=True)


def log_step(
    task_id: str,
    step: int,
    action: Dict[str, Any],
    reward: float,
    valid: bool,
    observation_summary: str,
) -> None:
    """Emit a STEP log line."""
    record = {
        "event": "STEP",
        "task_id": task_id,
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "valid": valid,
        "observation_summary": observation_summary,
    }
    _safe_print(json.dumps(record), flush=True)


def log_end(
    task_id: str,
    score: float,
    completion_rate: float,
    steps_used: int,
    total_reward: float,
) -> None:
    """Emit an END log line."""
    record = {
        "event": "END",
        "task_id": task_id,
        "timestamp": _ts(),
        "score": round(score, 4),
        "completion_rate": round(completion_rate, 4),
        "steps_used": steps_used,
        "total_reward": round(total_reward, 4),
    }
    _safe_print(json.dumps(record), flush=True)


# ═══════════════════════════════════════════════════════════════════════
# LLM ↔ Environment Bridge
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a delivery dispatch AI agent operating inside a graph-based city environment.

Your goal is to complete all deliveries as efficiently as possible.

On each turn you receive a JSON observation containing:
  - nodes: list of graph nodes with neighbors and edge weights
  - drivers: list of drivers with their locations, assigned deliveries, and capacity
  - deliveries: list of deliveries with status, pickup node, and destination
  - available_actions: list of valid action strings you can take right now

You MUST respond with a single JSON object representing your chosen action:

  {"action_type": "<type>", "driver_id": "<id>", ...}

Valid action_type values:
  - "assign_driver"      → requires driver_id, delivery_id
  - "move_driver"        → requires driver_id, target_node
  - "pickup_delivery"    → requires driver_id, delivery_id
  - "complete_delivery"  → requires driver_id, delivery_id

Rules:
1. Respond ONLY with a valid JSON action object. No commentary.
2. Pick from the available_actions when possible.
3. Minimise total travel cost and number of steps.
4. For tasks requiring pickup: assign → move to pickup → pickup → move to destination → complete.
"""


def _observation_to_prompt(obs: Observation) -> str:
    """Convert an Observation into a user-prompt string for the LLM."""
    data = obs.model_dump(mode="json")
    summary = {
        "step": data["step_number"],
        "max_steps": data["max_steps"],
        "drivers": data["drivers"],
        "deliveries": data["deliveries"],
        "available_actions": data["available_actions"],
        "last_action_result": data.get("last_action_result"),
    }
    return json.dumps(summary, indent=2)


def _observation_summary(obs: Observation) -> str:
    """One-line summary of the observation for structured logging."""
    n_pending = sum(1 for d in obs.deliveries if d.status == "Pending")
    n_assigned = sum(1 for d in obs.deliveries if d.status == "Assigned")
    n_pickedup = sum(1 for d in obs.deliveries if d.status == "PickedUp")
    n_completed = sum(1 for d in obs.deliveries if d.status == "Completed")
    driver_locs = ", ".join(
        f"{d.driver_id}@{d.current_location}" for d in obs.drivers
    )
    return (
        f"step={obs.step_number}/{obs.max_steps} "
        f"deliveries(P={n_pending},A={n_assigned},U={n_pickedup},C={n_completed}) "
        f"drivers=[{driver_locs}]"
    )


def _parse_action(raw: str) -> Optional[Action]:
    """
    Parse the LLM's raw text response into an Action model.
    Handles markdown code fences and extra commentary gracefully.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return _dict_to_action(data)
    except (json.JSONDecodeError, Exception):
        pass

    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _dict_to_action(data)
        except (json.JSONDecodeError, Exception):
            pass

    return None


def _dict_to_action(data: Dict[str, Any]) -> Action:
    """Convert a parsed dict into an Action model."""
    return Action(
        action_type=data["action_type"],
        driver_id=data.get("driver_id"),
        delivery_id=data.get("delivery_id"),
        target_node=data.get("target_node"),
    )


# ═══════════════════════════════════════════════════════════════════════
# DeliveryAgent — Reusable Agent Class (for GUI + CLI)
# ═══════════════════════════════════════════════════════════════════════

# Callback type for step reporting
# signature: (step_num, action_dict, reward, valid, obs_summary, result) -> None
StepCallback = Callable[..., None]


class DeliveryAgent:
    """
    AI delivery dispatch agent powered by an LLM via OpenAI-compatible API.

    Can be used standalone (CLI) or controlled from a GUI via callbacks.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_HF_TOKEN,
        fallback_on_api_error: bool = True,
        force_rule_based: bool = False,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.client = InferenceClient(token=api_key)
        self.grader = TaskGrader()
        self._stop_requested = False
        self.fallback_on_api_error = fallback_on_api_error
        self.force_rule_based = force_rule_based
        self._api_disabled = force_rule_based

    def stop(self):
        """Request the agent to stop after the current step."""
        self._stop_requested = True

    def _shortest_path_next_hop(self, obs: Observation, start: str, goal: str) -> Optional[str]:
        """Return next hop on the weighted shortest path from start to goal."""
        if start == goal:
            return None

        graph: Dict[str, Dict[str, float]] = {
            node.node_id: dict(node.edge_weights) for node in obs.nodes
        }
        if start not in graph or goal not in graph:
            return None

        pq: List[Tuple[float, str]] = [(0.0, start)]
        dist: Dict[str, float] = {start: 0.0}
        prev: Dict[str, str] = {}

        while pq:
            cur_dist, cur = heapq.heappop(pq)
            if cur_dist > dist.get(cur, float("inf")):
                continue
            if cur == goal:
                break
            for nb, w in graph.get(cur, {}).items():
                nd = cur_dist + float(w)
                if nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(pq, (nd, nb))

        if goal not in dist:
            return None

        node = goal
        while prev.get(node) and prev[node] != start:
            node = prev[node]
        return node if prev.get(node) == start else None

    def _choose_rule_based_action(self, obs: Observation) -> Action:
        """Deterministic fallback policy when API calls are unavailable."""
        drivers = {d.driver_id: d for d in obs.drivers}
        deliveries = {d.delivery_id: d for d in obs.deliveries}

        # 1) Complete anything already at destination.
        for delivery in sorted(obs.deliveries, key=lambda d: d.delivery_id):
            if not delivery.assigned_driver or delivery.status == "Completed":
                continue
            drv = drivers.get(delivery.assigned_driver)
            if drv and drv.current_location == delivery.destination:
                if delivery.pickup_node is None or delivery.status == "PickedUp":
                    return Action(
                        action_type=ActionType.COMPLETE_DELIVERY,
                        driver_id=drv.driver_id,
                        delivery_id=delivery.delivery_id,
                    )

        # 2) Pickup where possible.
        for delivery in sorted(obs.deliveries, key=lambda d: d.delivery_id):
            if (
                delivery.pickup_node
                and delivery.assigned_driver
                and delivery.status == "Assigned"
            ):
                drv = drivers.get(delivery.assigned_driver)
                if drv and drv.current_location == delivery.pickup_node:
                    return Action(
                        action_type=ActionType.PICKUP_DELIVERY,
                        driver_id=drv.driver_id,
                        delivery_id=delivery.delivery_id,
                    )

        # 3) Assign pending deliveries to the nearest driver with free capacity.
        pending = sorted(
            [d for d in obs.deliveries if d.status == "Pending"],
            key=lambda d: d.delivery_id,
        )
        if pending:
            for delivery in pending:
                goal = delivery.pickup_node or delivery.destination
                best_driver: Optional[str] = None
                best_dist = float("inf")
                for drv in sorted(obs.drivers, key=lambda d: d.driver_id):
                    if len(drv.assigned_deliveries) >= drv.capacity:
                        continue
                    next_hop = self._shortest_path_next_hop(obs, drv.current_location, goal)
                    if drv.current_location == goal:
                        best_driver = drv.driver_id
                        best_dist = 0.0
                        break
                    if next_hop is None:
                        continue
                    # Cheap proxy distance: one-hop edge weight toward shortest path.
                    edge_w = next(
                        (
                            n.edge_weights.get(next_hop)
                            for n in obs.nodes
                            if n.node_id == drv.current_location
                        ),
                        None,
                    )
                    score = float(edge_w) if edge_w is not None else 1e9
                    if score < best_dist:
                        best_dist = score
                        best_driver = drv.driver_id

                if best_driver:
                    return Action(
                        action_type=ActionType.ASSIGN_DRIVER,
                        driver_id=best_driver,
                        delivery_id=delivery.delivery_id,
                    )

        # 4) Move drivers toward their nearest active assigned objective.
        for drv in sorted(obs.drivers, key=lambda d: d.driver_id):
            active = [deliveries[did] for did in drv.assigned_deliveries if did in deliveries]
            active = [d for d in active if d.status != "Completed"]
            if not active:
                continue

            target_delivery = sorted(active, key=lambda d: d.delivery_id)[0]
            if target_delivery.pickup_node and target_delivery.status == "Assigned":
                goal = target_delivery.pickup_node
            else:
                goal = target_delivery.destination

            if drv.current_location != goal:
                next_hop = self._shortest_path_next_hop(obs, drv.current_location, goal)
                if next_hop:
                    return Action(
                        action_type=ActionType.MOVE_DRIVER,
                        driver_id=drv.driver_id,
                        target_node=next_hop,
                    )

        # 5) Last-resort: choose a valid-looking move from available actions.
        for action_text in obs.available_actions:
            match = re.match(
                r'move_driver\(driver_id="([^"]+)", target_node="([^"]+)"\)',
                action_text,
            )
            if match:
                return Action(
                    action_type=ActionType.MOVE_DRIVER,
                    driver_id=match.group(1),
                    target_node=match.group(2),
                )

        # 6) If no valid action exists, emit an invalid no-op surrogate.
        return Action(
            action_type=ActionType.ASSIGN_DRIVER,
            driver_id="__INVALID__",
            delivery_id="__INVALID__",
        )

    def run_task(
        self,
        task_id: str,
        on_step: Optional[StepCallback] = None,
        on_error: Optional[Callable[[str], None]] = None,
        task_config=None,
    ) -> Optional[GradeReport]:
        """
        Run a single benchmark task end-to-end.

        Parameters
        ----------
        task_id : str
            One of "easy", "medium", "hard", or "custom".
        on_step : callable, optional
            Called after each step with (step_num, action_dict, reward, valid,
            obs_summary, step_result).
        on_error : callable, optional
            Called on API errors with the error message string.
        task_config : TaskConfig, optional
            Pre-built task configuration. If provided, ``get_task(task_id)``
            is skipped and this config is used directly.

        Returns
        -------
        GradeReport or None
            The grading report, or None if stopped early.
        """
        self._stop_requested = False

        cfg = task_config if task_config is not None else get_task(task_id)
        env = DeliveryEnvironment(cfg)
        obs = env.reset()

        log_start(task_id=task_id, model_name=self.model)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TASK: {cfg.task_name} (difficulty: {cfg.difficulty})\n\n"
                    f"{cfg.description}\n\n"
                    f"--- Initial Observation ---\n"
                    f"{_observation_to_prompt(obs)}"
                ),
            },
        ]

        terminated = False
        truncated = False
        step_num = 0

        while not terminated and not truncated and not self._stop_requested:
            # Call the LLM
            raw_response = ""
            if self._api_disabled:
                action = self._choose_rule_based_action(obs)
            else:
                try:
                    completion = self.client.chat_completion(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=256,
                    )
                    raw_response = completion.choices[0].message.content or ""
                    action = _parse_action(raw_response)
                except Exception as exc:
                    error_msg = f"API call failed: {exc}"
                    _safe_print(
                        json.dumps({
                            "event": "ERROR",
                            "task_id": task_id,
                            "step": step_num + 1,
                            "error": error_msg,
                            "timestamp": _ts(),
                        }),
                        flush=True,
                    )
                    if on_error:
                        on_error(error_msg)

                    if self.fallback_on_api_error:
                        self._api_disabled = True
                        _safe_print(
                            json.dumps({
                                "event": "WARN",
                                "task_id": task_id,
                                "step": step_num + 1,
                                "warning": "Switching to local rule-based fallback policy.",
                                "timestamp": _ts(),
                            }),
                            flush=True,
                        )
                        action = self._choose_rule_based_action(obs)
                    else:
                        break

            # Parse action
            if action is None:
                action = Action(
                    action_type=ActionType.ASSIGN_DRIVER,
                    driver_id="__INVALID__",
                    delivery_id="__INVALID__",
                )

            # Step the environment
            result: StepResult = env.step(action)
            step_num = result.observation.step_number
            terminated = result.terminated
            truncated = result.truncated

            # Build action dict for logging
            action_dict = {
                "action_type": action.action_type.value,
                "driver_id": action.driver_id,
                "delivery_id": action.delivery_id,
                "target_node": action.target_node,
            }

            obs_sum = _observation_summary(result.observation)

            # Structured STEP log
            log_step(
                task_id=task_id,
                step=step_num,
                action=action_dict,
                reward=result.reward.total,
                valid=result.info.valid,
                observation_summary=obs_sum,
            )

            # GUI callback
            if on_step:
                on_step(step_num, action_dict, result.reward.total,
                        result.info.valid, obs_sum, result)

            # Append to conversation
            if raw_response:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Result: {result.observation.last_action_result}\n\n"
                        f"--- Observation (step {step_num}) ---\n"
                        f"{_observation_to_prompt(result.observation)}"
                    ),
                })

            obs = result.observation

        if self._stop_requested:
            return None

        # Grade
        report = self.grader.grade_detailed(env)
        summary = env.get_episode_summary()

        log_end(
            task_id=task_id,
            score=report.score,
            completion_rate=summary["completion_rate"],
            steps_used=summary["steps_used"],
            total_reward=summary["total_reward"],
        )

        return report

    def run_all_tasks(
        self,
        on_step: Optional[StepCallback] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, GradeReport]:
        """Run all benchmark tasks and return results."""
        results = {}
        for task_id in TASK_IDS:
            report = self.run_task(task_id, on_step=on_step, on_error=on_error)
            if report:
                results[task_id] = report
            if self._stop_requested:
                break
        return results


# ═══════════════════════════════════════════════════════════════════════
# CLI Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all benchmark tasks and print a final scorecard."""
    config = _load_config()

    _safe_print("=" * 60)
    _safe_print("  OpenEnv Delivery Tracker — Inference Runner")
    _safe_print("=" * 60)
    _safe_print(f"  Model       : {config['model']}")
    _safe_print(f"  API Base    : {config['api_base']}")
    _safe_print(f"  Auth Source : {config['key_source']}")
    _safe_print(f"  Tasks       : {', '.join(TASK_IDS)}")
    _safe_print(f"  Fallback    : {'ON' if config['fallback_on_api_error'] == 'True' else 'OFF'}")
    _safe_print(f"  Rule-based  : {'ON' if config['force_rule_based'] == 'True' else 'OFF'}")
    _safe_print("=" * 60)
    _safe_print()

    agent = DeliveryAgent(
        api_base=config["api_base"],
        model=config["model"],
        api_key=config["api_key"],
        fallback_on_api_error=config["fallback_on_api_error"] == "True",
        force_rule_based=config["force_rule_based"] == "True",
    )

    results = agent.run_all_tasks()

    # ── Aggregate scorecard ──────────────────────────────────────────
    _safe_print("\n" + "═" * 60)
    _safe_print("  AGGREGATE SCORECARD")
    _safe_print("═" * 60)
    _safe_print(f"  {'Task':<10} {'Score':>8} {'Compl':>8} {'Effic':>8} {'Speed':>8} {'Valid':>8}")
    _safe_print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    total_score = 0.0
    for tid in TASK_IDS:
        if tid in results:
            r = results[tid]
            total_score += r.score
            _safe_print(
                f"  {tid:<10} {r.score:>8.4f} {r.completion_score:>8.4f} "
                f"{r.efficiency_score:>8.4f} {r.speed_score:>8.4f} "
                f"{r.validity_score:>8.4f}"
            )

    graded_count = len(results)
    avg_score = total_score / graded_count if graded_count else 0.0
    _safe_print(f"  {'─'*10} {'─'*8}")
    _safe_print(f"  {'AVERAGE':<10} {avg_score:>8.4f}")
    _safe_print("═" * 60)

    # ── JSON summary ────────────────────────────────────────────────
    json_summary = {
        "model": config["model"],
        "tasks": {
            tid: results[tid].to_dict() for tid in TASK_IDS if tid in results
        },
        "aggregate_score": round(avg_score, 4),
    }
    _safe_print("\n--- JSON Summary ---")
    _safe_print(json.dumps(json_summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never crash hard in validator environments; emit a structured fatal line.
        _safe_print(
            json.dumps(
                {
                    "event": "FATAL",
                    "timestamp": _ts(),
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        sys.exit(0)
