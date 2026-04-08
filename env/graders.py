"""
Deterministic agent graders for the OpenEnv Delivery Tracker benchmark.

Each grader evaluates a completed episode and returns a float score in [0.0, 1.0].

Scoring dimensions
------------------
  completion  (40%) — fraction of deliveries completed
  efficiency  (25%) — optimal travel cost / actual travel cost (capped at 1.0)
  speed       (20%) — (max_steps - steps_used) / max_steps (fewer steps = better)
  validity    (15%) — 1.0 minus deductions for invalid actions

All weights and parameters are deterministic constants — no randomness.

Usage
-----
>>> from env.graders import grade_episode, TaskGrader
>>> grader = TaskGrader()
>>> score = grader.grade(env)          # after episode ends
>>> report = grader.grade_detailed(env) # full breakdown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env.environment import DeliveryEnvironment


# ══════════════════════════════════════════════════════════════════════
# Scoring weights (must sum to 1.0)
# ══════════════════════════════════════════════════════════════════════

W_COMPLETION = 0.40
W_EFFICIENCY = 0.25
W_SPEED      = 0.20
W_VALIDITY   = 0.15

# Penalty per invalid action (deducted from validity score)
INVALID_ACTION_DEDUCTION = 0.10


# ══════════════════════════════════════════════════════════════════════
# Grade report
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GradeReport:
    """
    Detailed breakdown of a grading result.

    Attributes
    ----------
    score : float
        Final composite score in [0.0, 1.0].
    completion_score : float
        Raw completion sub-score in [0.0, 1.0].
    efficiency_score : float
        Raw efficiency sub-score in [0.0, 1.0].
    speed_score : float
        Raw speed sub-score in [0.0, 1.0].
    validity_score : float
        Raw validity sub-score in [0.0, 1.0].
    details : dict
        Additional diagnostic information.
    """
    score: float
    completion_score: float
    efficiency_score: float
    speed_score: float
    validity_score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "completion_score": round(self.completion_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "speed_score": round(self.speed_score, 4),
            "validity_score": round(self.validity_score, 4),
            "weights": {
                "completion": W_COMPLETION,
                "efficiency": W_EFFICIENCY,
                "speed": W_SPEED,
                "validity": W_VALIDITY,
            },
            "details": self.details,
        }


# ══════════════════════════════════════════════════════════════════════
# Task Grader
# ══════════════════════════════════════════════════════════════════════

class TaskGrader:
    """
    Deterministic episode grader.

    Evaluates a DeliveryEnvironment after the episode has ended
    (terminated or truncated) and returns a score in [0.0, 1.0].

    The grader is stateless and fully deterministic — same episode
    state always produces the same score.
    """

    def grade(self, env: DeliveryEnvironment) -> float:
        """
        Return final composite score in [0.0, 1.0].

        Parameters
        ----------
        env : DeliveryEnvironment
            The environment instance after the episode has ended.
        """
        return self.grade_detailed(env).score

    def grade_detailed(self, env: DeliveryEnvironment) -> GradeReport:
        """
        Return a full GradeReport with sub-scores and diagnostics.

        Parameters
        ----------
        env : DeliveryEnvironment
            The environment instance after the episode has ended.
        """
        summary = env.get_episode_summary()
        internal = env.internal_state()

        # ── 1. Completion score (0.0 – 1.0) ─────────────────────────
        completion = summary["completion_rate"]

        # ── 2. Efficiency score (0.0 – 1.0) ─────────────────────────
        #    ratio = sum(optimal) / sum(actual) for completed deliveries
        #    Perfect routing gives 1.0, worse routing gives < 1.0
        efficiency = self._compute_efficiency(summary)

        # ── 3. Speed score (0.0 – 1.0) ──────────────────────────────
        #    Fewer steps relative to max_steps = higher score
        #    speed = 1.0 - (steps_used / max_steps)
        #    But we only count speed if at least one delivery was completed
        max_steps = summary["max_steps"]
        steps_used = summary["steps_used"]
        if completion > 0 and max_steps > 0:
            speed = max(0.0, 1.0 - (steps_used / max_steps))
        else:
            speed = 0.0

        # ── 4. Validity score (0.0 – 1.0) ──────────────────────────
        #    Start at 1.0, deduct per invalid action
        invalid_count = self._count_invalid_actions(summary)
        validity = max(0.0, 1.0 - (invalid_count * INVALID_ACTION_DEDUCTION))

        # ── Composite score ─────────────────────────────────────────
        raw_score = (
            W_COMPLETION * completion
            + W_EFFICIENCY * efficiency
            + W_SPEED * speed
            + W_VALIDITY * validity
        )

        # Clamp to [0.0, 1.0]
        final_score = max(0.0, min(1.0, raw_score))

        return GradeReport(
            score=final_score,
            completion_score=completion,
            efficiency_score=efficiency,
            speed_score=speed,
            validity_score=validity,
            details={
                "task_id": summary["task_id"],
                "difficulty": summary["difficulty"],
                "total_deliveries": summary["total_deliveries"],
                "completed_deliveries": summary["completed_deliveries"],
                "steps_used": steps_used,
                "max_steps": max_steps,
                "total_reward": round(summary["total_reward"], 4),
                "terminated": summary["terminated"],
                "truncated": summary["truncated"],
                "invalid_action_count": invalid_count,
                "optimal_costs": summary["optimal_costs"],
                "actual_travel": summary["actual_travel"],
            },
        )

    # ── Internal scoring helpers ────────────────────────────────────

    @staticmethod
    def _compute_efficiency(summary: Dict[str, Any]) -> float:
        """
        Compute routing efficiency as ratio of optimal to actual cost.

        Only counts deliveries that were completed. If none completed,
        returns 0.0. If actual travel is zero (shouldn't happen for a
        completed delivery, but defensively handled), that delivery
        gets 1.0.
        """
        optimal_costs = summary.get("optimal_costs", {})
        actual_travel = summary.get("actual_travel", {})

        # We only grade efficiency for completed deliveries
        # A delivery is completed if its actual_travel > 0 and
        # its status is "Completed" (proxied by completion_rate > 0)
        completed_ids = [
            did for did, actual in actual_travel.items()
            if actual > 0
        ]

        if not completed_ids:
            return 0.0

        total_ratio = 0.0
        for did in completed_ids:
            optimal = optimal_costs.get(did, 0.0)
            actual = actual_travel.get(did, 0.0)
            if actual > 0 and optimal > 0:
                ratio = min(optimal / actual, 1.0)  # cap at 1.0
            elif actual == 0 and optimal == 0:
                ratio = 1.0  # trivial delivery
            else:
                ratio = 0.0
            total_ratio += ratio

        return total_ratio / len(completed_ids)

    @staticmethod
    def _count_invalid_actions(summary: Dict[str, Any]) -> int:
        """
        Estimate the number of invalid actions from reward data.

        The per-step reward applies PENALTY_INVALID_ACTION = -0.3
        for each invalid action. We can infer the count from the
        total reward and step count, but a more reliable approach
        is to track it directly. For now we use the total_reward
        and known penalty magnitude.

        A negative total_reward with high step count and low
        completion strongly suggests many invalid actions.
        """
        total_reward = summary.get("total_reward", 0.0)
        completed = summary.get("completed_deliveries", 0)
        steps = summary.get("steps_used", 0)

        # Each valid action contributes at most:
        #   assign: +0.1, move_toward: +0.1, pickup: +0.2, complete: +1.5
        # Each invalid action contributes: -0.3
        # Each loop: -0.1, each idle: -0.05
        # We estimate invalid actions conservatively
        if total_reward >= 0:
            return 0

        # Very rough estimate: if reward is negative and few completions,
        # attribute the deficit to invalid actions
        penalty_per = 0.3
        estimated = max(0, int(abs(total_reward) / penalty_per))
        return min(estimated, steps)  # can't have more invalids than steps


# ══════════════════════════════════════════════════════════════════════
# Convenience function
# ══════════════════════════════════════════════════════════════════════

def grade_episode(env: DeliveryEnvironment) -> float:
    """
    One-shot grading: return final score in [0.0, 1.0].

    >>> score = grade_episode(env)
    """
    return TaskGrader().grade(env)


def grade_episode_detailed(env: DeliveryEnvironment) -> GradeReport:
    """
    One-shot grading: return full GradeReport.

    >>> report = grade_episode_detailed(env)
    >>> print(report.score, report.completion_score)
    """
    return TaskGrader().grade_detailed(env)
