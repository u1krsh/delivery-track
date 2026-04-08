"""
Deterministic task definitions for the OpenEnv Delivery Tracker benchmark.

Each task specifies:
  - Objective (natural-language description for the LLM agent)
  - Initial scenario (graph, drivers, deliveries — all deterministic)
  - Termination logic (max_steps, all-deliveries-completed)
  - Fixed seed for reproducibility

Tasks
-----
  easy   — 4-node linear chain, 1 driver, 1 delivery          (max 10 steps)
  medium — 9-node 3×3 grid, 2 drivers, 3 deliveries           (max 25 steps)
  hard   — 16-node 4×4 grid, 3 drivers, 5 deliveries + pickup (max 50 steps)

Usage
-----
>>> from env.tasks import get_task, TASK_IDS
>>> cfg = get_task("easy")
>>> from env.environment import DeliveryEnvironment
>>> env = DeliveryEnvironment(cfg)
>>> obs = env.reset()
"""

from __future__ import annotations

from typing import Dict, List

from env.models import TaskConfig


# ══════════════════════════════════════════════════════════════════════
# Task registry
# ══════════════════════════════════════════════════════════════════════

TASK_IDS = ("easy", "medium", "hard")


def get_task(task_id: str) -> TaskConfig:
    """
    Return a deterministic TaskConfig by name.

    Parameters
    ----------
    task_id : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.

    Raises
    ------
    ValueError
        If *task_id* is not a recognised task name.
    """
    task_id = task_id.lower().strip()
    if task_id not in _TASK_BUILDERS:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Valid IDs: {TASK_IDS}"
        )
    return _TASK_BUILDERS[task_id]()


def list_tasks() -> List[Dict[str, str]]:
    """Return a summary list of all available tasks."""
    return [
        {
            "task_id": tid,
            "difficulty": tid,
            "description": get_task(tid).description,
            "max_steps": str(get_task(tid).max_steps),
        }
        for tid in TASK_IDS
    ]


# ══════════════════════════════════════════════════════════════════════
# EASY — Single Delivery (linear chain)
# ══════════════════════════════════════════════════════════════════════
#
#    A ──3── B ──5── C ──2── D
#
#  Driver D1 starts at A.
#  Delivery DEL1: deliver to D (no pickup node → direct).
#  Optimal: assign D1→DEL1, then A→B→C→D, complete. Cost=10, 4 steps.
#

def _build_easy() -> TaskConfig:
    return TaskConfig(
        task_id="easy",
        task_name="Single Delivery",
        difficulty="easy",
        seed=1001,
        max_steps=10,
        description=(
            "You are a delivery dispatcher managing a simple linear route.\n\n"
            "GRAPH: 4 nodes in a chain: A ──3── B ──5── C ──2── D\n"
            "DRIVERS: 1 driver (D1) starting at node A.\n"
            "DELIVERIES: 1 delivery (DEL1) to be delivered to node D.\n\n"
            "OBJECTIVE: Assign the delivery to the driver, move the driver "
            "along the chain to node D, and complete the delivery.\n\n"
            "ACTIONS AVAILABLE:\n"
            "  - assign_driver(driver_id, delivery_id)\n"
            "  - move_driver(driver_id, target_node)  [adjacent nodes only]\n"
            "  - complete_delivery(driver_id, delivery_id)  [at destination]\n\n"
            "OPTIMAL SOLUTION: 4 actions (assign → move A→B → move B→C → "
            "move C→D → complete). Minimise total actions and travel cost."
        ),
        nodes=[
            {"id": "A", "x": 0.0,   "y": 0.0},
            {"id": "B", "x": 100.0, "y": 0.0},
            {"id": "C", "x": 200.0, "y": 0.0},
            {"id": "D", "x": 300.0, "y": 0.0},
        ],
        edges=[
            {"start": "A", "end": "B", "weight": 3.0},
            {"start": "B", "end": "C", "weight": 5.0},
            {"start": "C", "end": "D", "weight": 2.0},
        ],
        drivers=[
            {"id": "D1", "name": "Alice", "location": "A", "capacity": 1},
        ],
        deliveries=[
            {"id": "DEL1", "destination": "D"},
        ],
    )


# ══════════════════════════════════════════════════════════════════════
# MEDIUM — Multi-Drop (3×3 grid)
# ══════════════════════════════════════════════════════════════════════
#
#    A ──4── B ──6── C
#    │       │       │
#    3       5       7
#    │       │       │
#    D ──3── E ──4── F
#    │       │       │
#    5       6       3
#    │       │       │
#    G ──4── H ──5── I
#
#  Driver D1 starts at A, Driver D2 starts at I.
#  Deliveries: DEL1→C, DEL2→G, DEL3→E.
#  Challenge: efficient driver-delivery assignment + routing.
#  Greedy assignment is suboptimal; D1→C and D2→G is better than D1→G.
#

def _build_medium() -> TaskConfig:
    return TaskConfig(
        task_id="medium",
        task_name="Multi-Drop Grid",
        difficulty="medium",
        seed=2002,
        max_steps=25,
        description=(
            "You are a delivery dispatcher managing two drivers on a city grid.\n\n"
            "GRAPH: 9-node grid (3×3) with varied travel times:\n"
            "  A──4──B──6──C\n"
            "  │     │     │\n"
            "  3     5     7\n"
            "  │     │     │\n"
            "  D──3──E──4──F\n"
            "  │     │     │\n"
            "  5     6     3\n"
            "  │     │     │\n"
            "  G──4──H──5──I\n\n"
            "DRIVERS:\n"
            "  - D1 (Alice) at node A, capacity 2\n"
            "  - D2 (Bob) at node I, capacity 2\n\n"
            "DELIVERIES:\n"
            "  - DEL1: deliver to node C\n"
            "  - DEL2: deliver to node G\n"
            "  - DEL3: deliver to node E\n\n"
            "OBJECTIVE: Assign each delivery to a driver, route them "
            "efficiently, and complete all 3 deliveries.\n\n"
            "HINT: Think about which driver is closer to which delivery "
            "to minimise total travel. A greedy approach may be suboptimal.\n\n"
            "ACTIONS AVAILABLE:\n"
            "  - assign_driver(driver_id, delivery_id)\n"
            "  - move_driver(driver_id, target_node)  [adjacent only]\n"
            "  - complete_delivery(driver_id, delivery_id)  [at destination]\n"
        ),
        nodes=[
            # Row 0
            {"id": "A", "x": 0.0,   "y": 0.0},
            {"id": "B", "x": 100.0, "y": 0.0},
            {"id": "C", "x": 200.0, "y": 0.0},
            # Row 1
            {"id": "D", "x": 0.0,   "y": 100.0},
            {"id": "E", "x": 100.0, "y": 100.0},
            {"id": "F", "x": 200.0, "y": 100.0},
            # Row 2
            {"id": "G", "x": 0.0,   "y": 200.0},
            {"id": "H", "x": 100.0, "y": 200.0},
            {"id": "I", "x": 200.0, "y": 200.0},
        ],
        edges=[
            # Row 0 horizontal
            {"start": "A", "end": "B", "weight": 4.0},
            {"start": "B", "end": "C", "weight": 6.0},
            # Row 1 horizontal
            {"start": "D", "end": "E", "weight": 3.0},
            {"start": "E", "end": "F", "weight": 4.0},
            # Row 2 horizontal
            {"start": "G", "end": "H", "weight": 4.0},
            {"start": "H", "end": "I", "weight": 5.0},
            # Column 0 vertical
            {"start": "A", "end": "D", "weight": 3.0},
            {"start": "D", "end": "G", "weight": 5.0},
            # Column 1 vertical
            {"start": "B", "end": "E", "weight": 5.0},
            {"start": "E", "end": "H", "weight": 6.0},
            # Column 2 vertical
            {"start": "C", "end": "F", "weight": 7.0},
            {"start": "F", "end": "I", "weight": 3.0},
        ],
        drivers=[
            {"id": "D1", "name": "Alice", "location": "A", "capacity": 2},
            {"id": "D2", "name": "Bob",   "location": "I", "capacity": 2},
        ],
        deliveries=[
            {"id": "DEL1", "destination": "C"},
            {"id": "DEL2", "destination": "G"},
            {"id": "DEL3", "destination": "E"},
        ],
    )


# ══════════════════════════════════════════════════════════════════════
# HARD — Peak-Hour Dispatch (4×4 grid with traffic + pickup/dropoff)
# ══════════════════════════════════════════════════════════════════════
#
#  16-node grid (N01–N16), some edges have high "traffic" weights,
#  one edge is blocked entirely (missing from graph).
#  3 drivers with capacity=2 each, scattered across the grid.
#  5 deliveries requiring pickup at a restaurant node THEN dropoff
#  at a customer node.
#
#  Grid layout (4×4):
#
#    N01──4──N02──3──N03──5──N04
#     │       │       │       │
#     3       6       2      10
#     │       │       │       │
#    N05──5──N06──4──N07──3──N08
#     │       │       │       │
#     4       3       8       4
#     │       │       │       │
#    N09──6──N10──5──N11──4──N12
#     │       │       │       │
#     5       7       3       6
#     │       │       │       │
#    N13──3──N14──9──N15──4──N16
#
#  N07↔N11 has weight 8 (traffic jam)
#  N14↔N15 has weight 9 (roadworks)
#  N04↔N08 has weight 10 (near-blocked arterial)
#
#  Drivers: D1@N01, D2@N06, D3@N16
#  Deliveries (all require pickup first):
#    DEL1: pickup@N02, deliver to N15
#    DEL2: pickup@N05, deliver to N12
#    DEL3: pickup@N10, deliver to N04
#    DEL4: pickup@N09, deliver to N08
#    DEL5: pickup@N14, deliver to N03
#

def _build_hard() -> TaskConfig:
    return TaskConfig(
        task_id="hard",
        task_name="Peak-Hour Dispatch",
        difficulty="hard",
        seed=3003,
        max_steps=50,
        description=(
            "You are a delivery dispatcher during peak hours in a congested city.\n\n"
            "GRAPH: 16-node grid (4×4) with varied travel times. Some roads are "
            "heavily congested (high weights). Layout:\n"
            "  N01──4──N02──3──N03──5──N04\n"
            "   │       │       │       │\n"
            "   3       6       2      10\n"
            "   │       │       │       │\n"
            "  N05──5──N06──4──N07──3──N08\n"
            "   │       │       │       │\n"
            "   4       3       8       4\n"
            "   │       │       │       │\n"
            "  N09──6──N10──5──N11──4──N12\n"
            "   │       │       │       │\n"
            "   5       7       3       6\n"
            "   │       │       │       │\n"
            "  N13──3──N14──9──N15──4──N16\n\n"
            "DRIVERS:\n"
            "  - D1 (Alice) at N01, capacity 2\n"
            "  - D2 (Bob) at N06, capacity 2\n"
            "  - D3 (Carol) at N16, capacity 2\n\n"
            "DELIVERIES (each requires PICKUP then DELIVERY):\n"
            "  - DEL1: pickup at N02, deliver to N15\n"
            "  - DEL2: pickup at N05, deliver to N12\n"
            "  - DEL3: pickup at N10, deliver to N04\n"
            "  - DEL4: pickup at N09, deliver to N08\n"
            "  - DEL5: pickup at N14, deliver to N03\n\n"
            "IMPORTANT: For each delivery, the driver must:\n"
            "  1. Be assigned the delivery (assign_driver)\n"
            "  2. Move to the PICKUP node\n"
            "  3. Pick up the food (pickup_delivery)\n"
            "  4. Move to the DESTINATION node\n"
            "  5. Complete the delivery (complete_delivery)\n\n"
            "OBJECTIVE: Complete all 5 deliveries with minimal total travel cost. "
            "Plan assignments carefully — each driver can carry at most 2 orders "
            "simultaneously. Watch out for congested routes (high-weight edges).\n\n"
            "ACTIONS AVAILABLE:\n"
            "  - assign_driver(driver_id, delivery_id)\n"
            "  - move_driver(driver_id, target_node)  [adjacent only]\n"
            "  - pickup_delivery(driver_id, delivery_id)  [at pickup node]\n"
            "  - complete_delivery(driver_id, delivery_id)  [at destination]\n"
        ),
        nodes=[
            # Row 0
            {"id": "N01", "x": 0.0,   "y": 0.0},
            {"id": "N02", "x": 100.0, "y": 0.0},
            {"id": "N03", "x": 200.0, "y": 0.0},
            {"id": "N04", "x": 300.0, "y": 0.0},
            # Row 1
            {"id": "N05", "x": 0.0,   "y": 100.0},
            {"id": "N06", "x": 100.0, "y": 100.0},
            {"id": "N07", "x": 200.0, "y": 100.0},
            {"id": "N08", "x": 300.0, "y": 100.0},
            # Row 2
            {"id": "N09", "x": 0.0,   "y": 200.0},
            {"id": "N10", "x": 100.0, "y": 200.0},
            {"id": "N11", "x": 200.0, "y": 200.0},
            {"id": "N12", "x": 300.0, "y": 200.0},
            # Row 3
            {"id": "N13", "x": 0.0,   "y": 300.0},
            {"id": "N14", "x": 100.0, "y": 300.0},
            {"id": "N15", "x": 200.0, "y": 300.0},
            {"id": "N16", "x": 300.0, "y": 300.0},
        ],
        edges=[
            # Row 0 horizontal
            {"start": "N01", "end": "N02", "weight": 4.0},
            {"start": "N02", "end": "N03", "weight": 3.0},
            {"start": "N03", "end": "N04", "weight": 5.0},
            # Row 1 horizontal
            {"start": "N05", "end": "N06", "weight": 5.0},
            {"start": "N06", "end": "N07", "weight": 4.0},
            {"start": "N07", "end": "N08", "weight": 3.0},
            # Row 2 horizontal
            {"start": "N09", "end": "N10", "weight": 6.0},
            {"start": "N10", "end": "N11", "weight": 5.0},
            {"start": "N11", "end": "N12", "weight": 4.0},
            # Row 3 horizontal
            {"start": "N13", "end": "N14", "weight": 3.0},
            {"start": "N14", "end": "N15", "weight": 9.0},   # roadworks
            {"start": "N15", "end": "N16", "weight": 4.0},
            # Column 0 vertical
            {"start": "N01", "end": "N05", "weight": 3.0},
            {"start": "N05", "end": "N09", "weight": 4.0},
            {"start": "N09", "end": "N13", "weight": 5.0},
            # Column 1 vertical
            {"start": "N02", "end": "N06", "weight": 6.0},
            {"start": "N06", "end": "N10", "weight": 3.0},
            {"start": "N10", "end": "N14", "weight": 7.0},
            # Column 2 vertical
            {"start": "N03", "end": "N07", "weight": 2.0},
            {"start": "N07", "end": "N11", "weight": 8.0},   # traffic jam
            {"start": "N11", "end": "N15", "weight": 3.0},
            # Column 3 vertical
            {"start": "N04", "end": "N08", "weight": 10.0},  # near-blocked
            {"start": "N08", "end": "N12", "weight": 4.0},
            {"start": "N12", "end": "N16", "weight": 6.0},
        ],
        drivers=[
            {"id": "D1", "name": "Alice", "location": "N01", "capacity": 2},
            {"id": "D2", "name": "Bob",   "location": "N06", "capacity": 2},
            {"id": "D3", "name": "Carol", "location": "N16", "capacity": 2},
        ],
        deliveries=[
            {"id": "DEL1", "pickup": "N02", "destination": "N15"},
            {"id": "DEL2", "pickup": "N05", "destination": "N12"},
            {"id": "DEL3", "pickup": "N10", "destination": "N04"},
            {"id": "DEL4", "pickup": "N09", "destination": "N08"},
            {"id": "DEL5", "pickup": "N14", "destination": "N03"},
        ],
    )


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════

_TASK_BUILDERS = {
    "easy":   _build_easy,
    "medium": _build_medium,
    "hard":   _build_hard,
}
