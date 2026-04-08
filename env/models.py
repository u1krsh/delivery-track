"""
Typed Pydantic models for the OpenEnv Delivery Tracker environment.

Defines the Action, Observation, Reward, Info, InternalState, and StepResult
schemas used by the environment's reset/step/state API.

All models are Pydantic BaseModel subclasses for strict typing and
serialization. The environment is fully deterministic: every TaskConfig
carries a `seed` field that is applied via `random.seed()` during `reset()`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Valid action types an agent can take."""
    ASSIGN_DRIVER = "assign_driver"
    MOVE_DRIVER = "move_driver"
    PICKUP_DELIVERY = "pickup_delivery"
    COMPLETE_DELIVERY = "complete_delivery"


class Action(BaseModel):
    """
    A single agent action submitted to the environment via step().

    Examples
    --------
    >>> Action(action_type="assign_driver", driver_id="D1", delivery_id="DEL1")
    >>> Action(action_type="move_driver", driver_id="D1", target_node="B")
    """
    action_type: ActionType = Field(
        ..., description="The type of action to perform."
    )
    driver_id: Optional[str] = Field(
        None, description="Driver to act on (required for all actions)."
    )
    delivery_id: Optional[str] = Field(
        None, description="Delivery to act on (required for assign/pickup/complete)."
    )
    target_node: Optional[str] = Field(
        None, description="Target graph node (required for move_driver)."
    )


# ---------------------------------------------------------------------------
# Observation components
# ---------------------------------------------------------------------------

class NodeInfo(BaseModel):
    """A graph node visible to the agent."""
    node_id: str
    x: float
    y: float
    neighbors: List[str] = Field(default_factory=list)
    edge_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping from neighbor node_id to edge weight (travel time)."
    )


class DriverInfo(BaseModel):
    """Observable state of a driver."""
    driver_id: str
    name: str
    current_location: str
    status: str  # "Available" | "Busy"
    assigned_deliveries: List[str] = Field(default_factory=list)
    capacity: int = Field(default=1, description="Max concurrent deliveries.")
    rating: float = 5.0
    efficiency_score: float = 100.0


class DeliveryInfo(BaseModel):
    """Observable state of a delivery order."""
    delivery_id: str
    pickup_node: Optional[str] = Field(
        None, description="Node where food must be picked up (None = already with driver or direct)."
    )
    destination: str
    status: str  # "Pending" | "Assigned" | "PickedUp" | "InTransit" | "Completed"
    assigned_driver: Optional[str] = None
    progress: float = 0.0  # 0.0 – 100.0


# ---------------------------------------------------------------------------
# Observation (full state visible to the agent)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Complete observation returned to the agent after reset() or step().

    Contains the full graph topology, all driver states, all delivery states,
    and a natural-language task prompt for the LLM.
    """
    task_id: str = Field(..., description="Identifier for the current task.")
    task_prompt: str = Field(
        ..., description="Natural-language description of the task for the LLM."
    )
    step_number: int = Field(0, description="Current step in the episode.")
    max_steps: int = Field(10, description="Maximum steps allowed.")
    nodes: List[NodeInfo] = Field(default_factory=list)
    drivers: List[DriverInfo] = Field(default_factory=list)
    deliveries: List[DeliveryInfo] = Field(default_factory=list)

    # Convenience fields
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of valid action descriptions the agent can take."
    )
    last_action_result: Optional[str] = Field(
        None, description="Human-readable result of the last action taken."
    )


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------

class RewardInfo(BaseModel):
    """Detailed breakdown of the reward for a single step."""
    total: float = Field(0.0, description="Net reward for this step.")
    delivery_progress: float = Field(
        0.0, description="Reward for making progress toward a delivery."
    )
    delivery_completion: float = Field(
        0.0, description="Bonus for completing a delivery."
    )
    efficiency_bonus: float = Field(
        0.0, description="Bonus for near-optimal routing."
    )
    invalid_action_penalty: float = Field(
        0.0, description="Penalty for attempting an invalid action."
    )
    loop_penalty: float = Field(
        0.0, description="Penalty for revisiting a state in the same episode."
    )
    idle_penalty: float = Field(
        0.0, description="Penalty for a step that makes no progress."
    )


# ---------------------------------------------------------------------------
# Info (structured metadata per step)
# ---------------------------------------------------------------------------

class Info(BaseModel):
    """
    Structured metadata returned alongside each step result.

    Replaces a raw dict so that consumers get typed access to
    action validity, error details, and performance metrics.
    """
    valid: bool = Field(
        True, description="Whether the submitted action was valid."
    )
    error: Optional[str] = Field(
        None, description="Error message if the action was invalid or the episode is over."
    )
    exception: Optional[str] = Field(
        None, description="Python exception string if an unexpected error occurred."
    )
    action_type: Optional[str] = Field(
        None, description="Echo of the action type that was attempted."
    )
    cumulative_reward: float = Field(
        0.0, description="Cumulative reward from episode start to this step."
    )
    deliveries_completed: int = Field(
        0, description="Number of deliveries completed so far."
    )
    deliveries_remaining: int = Field(
        0, description="Number of deliveries not yet completed."
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Result returned by environment.step(action).

    Follows the Gymnasium pattern: (observation, reward, terminated, truncated, info).
    """
    observation: Observation
    reward: RewardInfo
    terminated: bool = Field(
        False, description="True if all deliveries are completed."
    )
    truncated: bool = Field(
        False, description="True if max_steps reached without completion."
    )
    info: Info = Field(
        default_factory=Info,
        description="Structured metadata about the step."
    )


# ---------------------------------------------------------------------------
# Internal state (non-agent-visible, for debugging / grading)
# ---------------------------------------------------------------------------

class InternalState(BaseModel):
    """
    Full internal state snapshot of the environment.

    This is *not* exposed to the agent — it is used by graders, tests,
    and deterministic-replay checks. It captures everything needed to
    reconstruct or compare environment state at any point in time.
    """
    task_id: str
    seed: int
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    total_reward: float = 0.0

    # Driver positions and assignments
    driver_locations: Dict[str, str] = Field(
        default_factory=dict,
        description="driver_id -> current node_id"
    )
    driver_assignments: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="driver_id -> list of delivery_ids assigned"
    )

    # Delivery statuses and metadata
    delivery_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="delivery_id -> status string"
    )
    delivery_picked_up: Dict[str, bool] = Field(
        default_factory=dict,
        description="delivery_id -> whether picked up"
    )

    # Cost tracking
    optimal_costs: Dict[str, float] = Field(
        default_factory=dict,
        description="delivery_id -> optimal travel cost (computed at reset)"
    )
    actual_travel: Dict[str, float] = Field(
        default_factory=dict,
        description="delivery_id -> actual travel cost accumulated"
    )

    # Loop detection state
    visited_state_count: int = Field(
        0, description="Number of unique states visited so far."
    )


# ---------------------------------------------------------------------------
# Environment configuration (used internally by tasks)
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    """
    Configuration for a deterministic task scenario.

    Used by tasks.py and scenario_generator.py to define reproducible scenarios.
    The ``seed`` field is applied via ``random.seed(seed)`` at the start of
    every ``reset()`` call, guaranteeing deterministic replay.
    """
    task_id: str
    task_name: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    max_steps: int
    seed: int = Field(
        ...,
        description="Random seed applied at reset() for deterministic replay."
    )

    # Graph definition
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"id": str, "x": float, "y": float}.'
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"start": str, "end": str, "weight": float}.'
    )

    # Drivers
    drivers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"id": str, "name": str, "location": str, "capacity": int}.'
    )

    # Deliveries
    deliveries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description='List of {"id": str, "pickup": str|None, "destination": str}.'
    )
