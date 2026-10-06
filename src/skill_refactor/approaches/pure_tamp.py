"""Pure TAMP approach without any RL skills."""

from typing import Any

from torch import Tensor

from skill_refactor.approaches.base import ApproachStepResult, BaseApproach
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


class PureTAMPApproach(BaseApproach):
    """Pure RL approach that doesn't use TAMP structure."""

    def __init__(
        self,
        system: BaseRLTAMPSystem,
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        fall_back_action = system.env.single_action_space.sample()  # type: ignore
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            system.env, CFG.control_mode
        )

        # Create planner using environment's components
        self.planner = TaskThenMotionPlanner(
            types=system.types,
            predicates=system.predicates,
            perceiver=system.perceiver,
            operators=system.operators,
            skills=system.skills,
            fallback_action=fall_back_action,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            planner_id="pyperplan",
        )

    def reset(self, obs: Tensor, info: dict[str, Any]) -> ApproachStepResult:
        """Reset approach with initial observation."""
        self.planner.reset(obs[0:1], info)
        return self.step(obs, 0.0, False, False, info)

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "pure_tamp"

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation."""
        action, operator = self.planner.step(obs)
        if action is None:
            raise RuntimeError("Task then motion planning failed, no action returned.")
        return ApproachStepResult(_action=action, op=operator)
