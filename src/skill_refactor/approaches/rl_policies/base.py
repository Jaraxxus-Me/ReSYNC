"""Base policy interface for RL training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import wandb
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from skill_refactor.benchmarks.wrappers import PlanningStatesVectorEnv


class Logger:
    """Logger for RL training and evaluation.

    Logs to TensorBoard and optionally to Weights & Biases.
    """

    def __init__(self, tensorboard: SummaryWriter, log_wandb=False) -> None:
        """Initialize the logger with TensorBoard and optional Weights & Biases."""
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag: str, scalar_value: float | Tensor, step: int = 0) -> None:
        """Log a scalar value to TensorBoard and optionally to Weights & Biases."""
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)  # type: ignore

    def close(self) -> None:
        """Close the logger."""
        self.writer.close()  # type: ignore


class Policy(ABC):
    """Base class for policies."""

    args: Any

    def __init__(self, seed: int, rl_config: str = "") -> None:
        """Initialize policy with environment."""
        del rl_config  # to be used by subclasses
        self._seed = seed

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""

    @abstractmethod
    def initialize(self, env: PlanningStatesVectorEnv) -> None:
        """Initialize policy with environment."""

    @abstractmethod
    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current context."""

    @abstractmethod
    def get_action(self, obs: Tensor) -> Tensor:
        """Get action from policy."""

    @abstractmethod
    def reset(self, obs: Tensor) -> None:
        """Reset the policy state."""

    @abstractmethod
    def train(
        self, env: PlanningStatesVectorEnv, eval_env: PlanningStatesVectorEnv | None
    ) -> None:
        """Train the policy if needed.

        Default implementation just initializes the policy and updates context. Policies
        that need training should override this.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load policy from disk."""
