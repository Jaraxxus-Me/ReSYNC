"""Base class for all approaches."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.utils.structs import ApproachStepResult


class BaseApproach(ABC):
    """Base class for all approaches."""

    def __init__(self, system: BaseRLTAMPSystem, seed: int) -> None:
        """Initialize approach.

        Args:
            system: The TAMP system to use
            seed: Random seed
        """
        self.system = system
        self._seed = seed
        self._training_mode = False

    @property
    def training_mode(self) -> bool:
        """Whether the approach is in training mode."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool) -> None:
        """Set training mode."""
        self._training_mode = value

    @abstractmethod
    def reset(self, obs: Tensor, info: dict[str, Any]) -> ApproachStepResult:
        """Reset approach with initial observation."""

    @abstractmethod
    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this approach, used as the argument to
        `--approach`."""
        raise NotImplementedError("Override me!")
