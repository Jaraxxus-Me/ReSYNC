"""Pure RL baseline approach without using TAMP structure."""

from typing import Any

from torch import Tensor

from skill_refactor.approaches.base import ApproachStepResult, BaseApproach
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.benchmarks.base import BaseRLTAMPSystem


class PureRLApproach(BaseApproach):
    """Pure RL approach that doesn't use TAMP structure."""

    def __init__(
        self,
        system: BaseRLTAMPSystem,
        policy: Policy,
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy

    def reset(self, obs: Tensor, info: dict[str, Any]) -> ApproachStepResult:
        """Reset approach with initial observation."""
        return self.step(obs, 0.0, False, False, info)

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "pure_rl"

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation."""
        action = self.policy.get_action(obs)
        return ApproachStepResult(_action=action)
