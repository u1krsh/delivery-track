"""
OpenEnv-compatible Delivery Tracker environment.

Wraps the existing domain logic from src/ (Graph, Driver, Delivery, Routing,
AssignmentService) into a standardised reset/step/state API with dense
reward signals and deterministic replay.
"""

from __future__ import annotations

import copy
import random
import sys
import os
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Make sure the src package is importable
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.graph import Graph
from src.models.delivery import Delivery
from src.models.driver import Driver
from src.algorithms.routing import Routing
from src.services.assignment_service import AssignmentService

from env.models import (
    Action,
    ActionType,
    DeliveryInfo,
    DriverInfo,
    Info,
    InternalState,
    NodeInfo,
    Observation,
    RewardInfo,
    StepResult,
    TaskConfig,
)


class DeliveryEnvironment:
    """
    OpenEnv-compatible delivery dispatch environment.

    Lifecycle
    ---------
    1. ``env = DeliveryEnvironment(task_config)``
    2. ``obs = env.reset()``                    → initial observation
    3. ``result = env.step(action)``             → (obs, reward, done, truncated, info)
    4. ``state = env.state()``                   → current observation snapshot
    """

    # ----- Reward constants (tunable) -----
    REWARD_PROGRESS_PER_EDGE = 0.1
    REWARD_DELIVERY_COMPLETE = 1.0
    REWARD_EFFICIENCY_MAX = 0.5
    PENALTY_INVALID_ACTION = -0.3
    PENALTY_LOOP = -0.1
    PENALTY_IDLE = -0.05

    def __init__(self, task_config: TaskConfig) -> None:
        self._config = task_config
        self._graph: Optional[Graph] = None
        self._routing: Optional[Routing] = None
        self._assignment: Optional[AssignmentService] = None
        self._drivers: Dict[str, Driver] = {}
        self._deliveries: Dict[str, Delivery] = {}

        # Extended delivery metadata (pickup node, assigned driver, picked-up flag)
        self._delivery_meta: Dict[str, Dict[str, Any]] = {}

        # Driver capacities
        self._driver_capacity: Dict[str, int] = {}

        # Episode bookkeeping
        self._step_count: int = 0
        self._done: bool = False
        self._truncated: bool = False
        self._total_reward: float = 0.0

        # Loop detection: track (driver_location_tuple) hashes
        self._visited_states: Set[int] = set()

        # Optimal costs cache (computed once per reset)
        self._optimal_costs: Dict[str, float] = {}

        # Per-delivery distance tracking (actual vs optimal)
        self._actual_travel: Dict[str, float] = {}

    # ======================================================================
    # Public API
    # ======================================================================

    def reset(self) -> Observation:
        """
        Reset the environment to the initial state defined by the task config.

        Applies ``random.seed(config.seed)`` for full deterministic replay,
        then reconstructs the world from the TaskConfig definition.

        Returns the initial Observation.
        """
        cfg = self._config

        # ---- Deterministic seed ----
        random.seed(cfg.seed)

        # Build graph
        self._graph = Graph()
        for node_def in cfg.nodes:
            self._graph.add_node(
                node_def["id"],
                {"x": node_def["x"], "y": node_def["y"]},
            )
        for edge_def in cfg.edges:
            self._graph.add_edge(
                edge_def["start"], edge_def["end"], edge_def["weight"]
            )

        self._routing = Routing(self._graph)
        self._assignment = AssignmentService(self._graph)

        # Build drivers
        self._drivers = {}
        self._driver_capacity = {}
        for drv in cfg.drivers:
            d = Driver(drv["id"], drv.get("name", drv["id"]), drv["location"])
            self._drivers[d.driver_id] = d
            self._driver_capacity[d.driver_id] = drv.get("capacity", 1)

        # Build deliveries with extended metadata
        self._deliveries = {}
        self._delivery_meta = {}
        for dlv in cfg.deliveries:
            d = Delivery(dlv["id"], dlv["destination"])
            self._deliveries[d.delivery_id] = d
            self._delivery_meta[d.delivery_id] = {
                "pickup_node": dlv.get("pickup"),  # None means direct delivery
                "assigned_driver": None,
                "picked_up": dlv.get("pickup") is None,  # already "picked up" if no pickup
            }

        # Episode state
        self._step_count = 0
        self._done = False
        self._truncated = False
        self._total_reward = 0.0
        self._visited_states = set()
        self._actual_travel = {d_id: 0.0 for d_id in self._deliveries}

        # Compute optimal costs for grading
        self._precompute_optimal_costs()

        # Record the initial state for loop detection
        self._visited_states.add(self._compute_state_hash())

        return self._build_observation(last_action_result="Environment reset. Ready for actions.")

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return (observation, reward, terminated, truncated, info).
        """
        if self._done or self._truncated:
            return StepResult(
                observation=self._build_observation(
                    last_action_result="Episode already finished."
                ),
                reward=RewardInfo(total=0.0),
                terminated=self._done,
                truncated=self._truncated,
                info=Info(valid=False, error="Episode already finished."),
            )

        self._step_count += 1
        reward = RewardInfo()
        info = Info()
        action_result = ""

        # ---- Dispatch action ----
        info.action_type = action.action_type.value
        try:
            if action.action_type == ActionType.ASSIGN_DRIVER:
                action_result = self._handle_assign(action, reward, info)
            elif action.action_type == ActionType.MOVE_DRIVER:
                action_result = self._handle_move(action, reward, info)
            elif action.action_type == ActionType.PICKUP_DELIVERY:
                action_result = self._handle_pickup(action, reward, info)
            elif action.action_type == ActionType.COMPLETE_DELIVERY:
                action_result = self._handle_complete(action, reward, info)
            else:
                action_result = f"Unknown action type: {action.action_type}"
                reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
                info.valid = False
                info.error = action_result
        except Exception as exc:
            action_result = f"Action error: {exc}"
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = action_result
            info.exception = str(exc)

        # ---- Loop detection ----
        state_hash = self._compute_state_hash()
        if state_hash in self._visited_states:
            reward.loop_penalty = self.PENALTY_LOOP
        self._visited_states.add(state_hash)

        # ---- Idle detection ----
        progress_made = (
            reward.delivery_progress > 0
            or reward.delivery_completion > 0
            or reward.efficiency_bonus > 0
        )
        if not progress_made and info.valid:
            reward.idle_penalty = self.PENALTY_IDLE

        # ---- Compute total reward ----
        reward.total = (
            reward.delivery_progress
            + reward.delivery_completion
            + reward.efficiency_bonus
            + reward.invalid_action_penalty
            + reward.loop_penalty
            + reward.idle_penalty
        )
        self._total_reward += reward.total

        # ---- Check termination ----
        all_completed = all(
            d.status == "Completed" for d in self._deliveries.values()
        )
        if all_completed:
            self._done = True

        if self._step_count >= self._config.max_steps and not self._done:
            self._truncated = True

        # ---- Populate info summary fields ----
        completed_count = sum(
            1 for d in self._deliveries.values() if d.status == "Completed"
        )
        info.cumulative_reward = self._total_reward
        info.deliveries_completed = completed_count
        info.deliveries_remaining = len(self._deliveries) - completed_count

        obs = self._build_observation(last_action_result=action_result)

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self._done,
            truncated=self._truncated,
            info=info,
        )

    def state(self) -> Observation:
        """Return the current observation without advancing the environment."""
        return self._build_observation()

    def internal_state(self) -> InternalState:
        """
        Return a full internal-state snapshot (not agent-visible).

        Used by graders, deterministic-replay tests, and debugging.
        """
        return InternalState(
            task_id=self._config.task_id,
            seed=self._config.seed,
            step_count=self._step_count,
            done=self._done,
            truncated=self._truncated,
            total_reward=self._total_reward,
            driver_locations={
                d.driver_id: d.current_location for d in self._drivers.values()
            },
            driver_assignments={
                d.driver_id: list(d.assigned_deliveries)
                for d in self._drivers.values()
            },
            delivery_statuses={
                d.delivery_id: d.status for d in self._deliveries.values()
            },
            delivery_picked_up={
                d_id: meta["picked_up"]
                for d_id, meta in self._delivery_meta.items()
            },
            optimal_costs=dict(self._optimal_costs),
            actual_travel=dict(self._actual_travel),
            visited_state_count=len(self._visited_states),
        )

    # ======================================================================
    # Action handlers
    # ======================================================================

    def _handle_assign(
        self, action: Action, reward: RewardInfo, info: Info
    ) -> str:
        drv_id = action.driver_id
        del_id = action.delivery_id

        if not drv_id or drv_id not in self._drivers:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid driver_id: {drv_id}"
            return info.error

        if not del_id or del_id not in self._deliveries:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid delivery_id: {del_id}"
            return info.error

        delivery = self._deliveries[del_id]
        driver = self._drivers[drv_id]
        meta = self._delivery_meta[del_id]

        # Check delivery not already assigned or completed
        if delivery.status not in ("Pending",):
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} is already {delivery.status}, cannot assign."
            return info.error

        # Check driver capacity
        current_load = len(driver.assigned_deliveries)
        cap = self._driver_capacity.get(drv_id, 1)
        if current_load >= cap:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Driver {drv_id} at capacity ({current_load}/{cap})."
            return info.error

        # Perform assignment
        driver.assign_delivery(del_id)
        delivery.update_status("Assigned")
        meta["assigned_driver"] = drv_id

        # Small progress reward for a valid assignment
        reward.delivery_progress = self.REWARD_PROGRESS_PER_EDGE
        return f"Assigned delivery {del_id} to driver {drv_id}."

    def _handle_move(
        self, action: Action, reward: RewardInfo, info: Info
    ) -> str:
        drv_id = action.driver_id
        target = action.target_node

        if not drv_id or drv_id not in self._drivers:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid driver_id: {drv_id}"
            return info.error

        if not target or target not in self._graph.nodes:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid target_node: {target}"
            return info.error

        driver = self._drivers[drv_id]
        current = driver.current_location

        # Must be adjacent
        if not self._graph.has_edge(current, target):
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = (
                f"Cannot move {drv_id} from {current} to {target}: "
                f"not adjacent. Neighbors: {self._graph.get_neighbors(current)}"
            )
            return info.error

        # Move the driver
        edge_cost = self._graph.get_edge_weight(current, target)
        driver.update_location(target)

        # Track travel distance for assigned deliveries
        for del_id in driver.assigned_deliveries:
            self._actual_travel[del_id] = self._actual_travel.get(del_id, 0.0) + edge_cost

        # Progress reward: moving toward a delivery destination or pickup
        made_progress = False
        for del_id in driver.assigned_deliveries:
            delivery = self._deliveries[del_id]
            meta = self._delivery_meta[del_id]
            if delivery.status in ("Completed",):
                continue

            # Determine the current target for this delivery
            if not meta["picked_up"] and meta["pickup_node"]:
                goal = meta["pickup_node"]
            else:
                goal = delivery.destination

            # Check if we moved closer to goal
            old_dist = self._shortest_distance(current, goal)
            new_dist = self._shortest_distance(target, goal)
            if new_dist < old_dist:
                made_progress = True

        if made_progress:
            reward.delivery_progress = self.REWARD_PROGRESS_PER_EDGE

        return f"Moved driver {drv_id} from {current} to {target} (cost: {edge_cost})."

    def _handle_pickup(
        self, action: Action, reward: RewardInfo, info: Info
    ) -> str:
        drv_id = action.driver_id
        del_id = action.delivery_id

        if not drv_id or drv_id not in self._drivers:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid driver_id: {drv_id}"
            return info.error

        if not del_id or del_id not in self._deliveries:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid delivery_id: {del_id}"
            return info.error

        driver = self._drivers[drv_id]
        delivery = self._deliveries[del_id]
        meta = self._delivery_meta[del_id]

        # Must be assigned to this driver
        if meta.get("assigned_driver") != drv_id:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} is not assigned to driver {drv_id}."
            return info.error

        # Must not already be picked up
        if meta["picked_up"]:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} is already picked up."
            return info.error

        # Driver must be at pickup node
        pickup = meta.get("pickup_node")
        if driver.current_location != pickup:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = (
                f"Driver {drv_id} is at {driver.current_location}, "
                f"but pickup for {del_id} is at {pickup}."
            )
            return info.error

        # Perform pickup
        meta["picked_up"] = True
        delivery.update_status("PickedUp")
        reward.delivery_progress = self.REWARD_PROGRESS_PER_EDGE * 2  # bonus for pickup
        return f"Driver {drv_id} picked up delivery {del_id} at {pickup}."

    def _handle_complete(
        self, action: Action, reward: RewardInfo, info: Info
    ) -> str:
        drv_id = action.driver_id
        del_id = action.delivery_id

        if not drv_id or drv_id not in self._drivers:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid driver_id: {drv_id}"
            return info.error

        if not del_id or del_id not in self._deliveries:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Invalid delivery_id: {del_id}"
            return info.error

        driver = self._drivers[drv_id]
        delivery = self._deliveries[del_id]
        meta = self._delivery_meta[del_id]

        # Must be assigned to this driver
        if meta.get("assigned_driver") != drv_id:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} is not assigned to driver {drv_id}."
            return info.error

        # Must be picked up
        if not meta["picked_up"]:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} has not been picked up yet."
            return info.error

        # Driver must be at destination
        if driver.current_location != delivery.destination:
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = (
                f"Driver {drv_id} is at {driver.current_location}, "
                f"but destination for {del_id} is {delivery.destination}."
            )
            return info.error

        # Already completed?
        if delivery.status == "Completed":
            reward.invalid_action_penalty = self.PENALTY_INVALID_ACTION
            info.valid = False
            info.error = f"Delivery {del_id} is already completed."
            return info.error

        # Complete the delivery
        delivery.update_status("Completed")
        delivery.progress = 100
        driver.complete_delivery(del_id)

        # Completion reward
        reward.delivery_completion = self.REWARD_DELIVERY_COMPLETE

        # Efficiency bonus
        optimal = self._optimal_costs.get(del_id, 0.0)
        actual = self._actual_travel.get(del_id, 0.0)
        if optimal > 0 and actual > 0:
            ratio = optimal / actual  # 1.0 = perfect, < 1.0 = suboptimal
            efficiency = min(ratio, 1.0)  # cap at 1.0
            reward.efficiency_bonus = self.REWARD_EFFICIENCY_MAX * efficiency

        return (
            f"Delivery {del_id} completed by driver {drv_id} at "
            f"{delivery.destination}. Efficiency: "
            f"{reward.efficiency_bonus:.2f}/{self.REWARD_EFFICIENCY_MAX}"
        )

    # ======================================================================
    # Observation builder
    # ======================================================================

    def _build_observation(
        self, last_action_result: Optional[str] = None
    ) -> Observation:
        # Nodes
        nodes = []
        for nid, attrs in self._graph.nodes.items():
            neighbors = self._graph.get_neighbors(nid)
            edge_weights = {
                nb: self._graph.get_edge_weight(nid, nb) for nb in neighbors
            }
            nodes.append(
                NodeInfo(
                    node_id=nid,
                    x=attrs.get("x", 0),
                    y=attrs.get("y", 0),
                    neighbors=neighbors,
                    edge_weights=edge_weights,
                )
            )

        # Drivers
        drivers = []
        for drv in self._drivers.values():
            drivers.append(
                DriverInfo(
                    driver_id=drv.driver_id,
                    name=drv.name,
                    current_location=drv.current_location,
                    status=drv.status,
                    assigned_deliveries=list(drv.assigned_deliveries),
                    capacity=self._driver_capacity.get(drv.driver_id, 1),
                    rating=drv.rating,
                    efficiency_score=drv.efficiency_score,
                )
            )

        # Deliveries
        deliveries = []
        for dlv in self._deliveries.values():
            meta = self._delivery_meta[dlv.delivery_id]
            deliveries.append(
                DeliveryInfo(
                    delivery_id=dlv.delivery_id,
                    pickup_node=meta.get("pickup_node"),
                    destination=dlv.destination,
                    status=dlv.status,
                    assigned_driver=meta.get("assigned_driver"),
                    progress=dlv.progress,
                )
            )

        # Available actions (contextual hints)
        available = self._compute_available_actions()

        return Observation(
            task_id=self._config.task_id,
            task_prompt=self._config.description,
            step_number=self._step_count,
            max_steps=self._config.max_steps,
            nodes=nodes,
            drivers=drivers,
            deliveries=deliveries,
            available_actions=available,
            last_action_result=last_action_result,
        )

    # ======================================================================
    # Helpers
    # ======================================================================

    def _shortest_distance(self, start: str, end: str) -> float:
        """Return weighted shortest-path distance using Dijkstra, or inf."""
        if start == end:
            return 0.0
        path = self._routing.shortest_path(start, end)
        if path is None:
            return float("inf")
        return self._routing.calculate_route_time(path)

    def _precompute_optimal_costs(self) -> None:
        """Compute the optimal travel cost for each delivery (for grading)."""
        self._optimal_costs = {}
        for del_id, delivery in self._deliveries.items():
            meta = self._delivery_meta[del_id]
            # Find closest driver to the delivery chain
            best_cost = float("inf")
            pickup = meta.get("pickup_node")
            dest = delivery.destination

            for drv in self._drivers.values():
                cost = 0.0
                if pickup:
                    cost += self._shortest_distance(drv.current_location, pickup)
                    cost += self._shortest_distance(pickup, dest)
                else:
                    cost += self._shortest_distance(drv.current_location, dest)
                best_cost = min(best_cost, cost)

            self._optimal_costs[del_id] = best_cost if best_cost < float("inf") else 0.0

    def _compute_state_hash(self) -> int:
        """Hash of mutable state for loop detection."""
        parts = []
        for drv in sorted(self._drivers.values(), key=lambda d: d.driver_id):
            parts.append(f"{drv.driver_id}:{drv.current_location}")
        for dlv in sorted(self._deliveries.values(), key=lambda d: d.delivery_id):
            parts.append(f"{dlv.delivery_id}:{dlv.status}")
        return hash(tuple(parts))

    def _compute_available_actions(self) -> List[str]:
        """Generate a human-readable list of valid actions."""
        actions = []

        for drv in self._drivers.values():
            # Move actions
            neighbors = self._graph.get_neighbors(drv.current_location)
            for nb in neighbors:
                w = self._graph.get_edge_weight(drv.current_location, nb)
                actions.append(
                    f'move_driver(driver_id="{drv.driver_id}", target_node="{nb}") '
                    f"[cost={w}]"
                )

            # Assignment actions
            cap = self._driver_capacity.get(drv.driver_id, 1)
            if len(drv.assigned_deliveries) < cap:
                for del_id, dlv in self._deliveries.items():
                    if dlv.status == "Pending":
                        actions.append(
                            f'assign_driver(driver_id="{drv.driver_id}", '
                            f'delivery_id="{del_id}")'
                        )

            # Pickup actions
            for del_id in drv.assigned_deliveries:
                meta = self._delivery_meta[del_id]
                if not meta["picked_up"] and meta.get("pickup_node"):
                    if drv.current_location == meta["pickup_node"]:
                        actions.append(
                            f'pickup_delivery(driver_id="{drv.driver_id}", '
                            f'delivery_id="{del_id}")'
                        )

            # Complete actions
            for del_id in drv.assigned_deliveries:
                dlv = self._deliveries[del_id]
                meta = self._delivery_meta[del_id]
                if meta["picked_up"] and drv.current_location == dlv.destination:
                    if dlv.status != "Completed":
                        actions.append(
                            f'complete_delivery(driver_id="{drv.driver_id}", '
                            f'delivery_id="{del_id}")'
                        )

        return actions

    # ======================================================================
    # Grading API (used by graders.py)
    # ======================================================================

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a summary dict for the grader after the episode ends."""
        total_deliveries = len(self._deliveries)
        completed = sum(
            1 for d in self._deliveries.values() if d.status == "Completed"
        )
        return {
            "task_id": self._config.task_id,
            "difficulty": self._config.difficulty,
            "steps_used": self._step_count,
            "max_steps": self._config.max_steps,
            "total_deliveries": total_deliveries,
            "completed_deliveries": completed,
            "completion_rate": completed / total_deliveries if total_deliveries > 0 else 0.0,
            "total_reward": self._total_reward,
            "optimal_costs": dict(self._optimal_costs),
            "actual_travel": dict(self._actual_travel),
            "terminated": self._done,
            "truncated": self._truncated,
        }
