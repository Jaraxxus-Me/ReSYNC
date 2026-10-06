"""
PPO-C Policy: Combines Proximal Policy Optimization (PPO) with Terminal Classifier learning.

Two-stage training approach:
1. Stage 1 (PPO Training): Train a standard PPO policy until it reaches a reward threshold
2. Stage 2 (Classifier Training): Train a binary classifier to decide when to call the planner:
   - Classifier output 0: Use the trained PPO policy
   - Classifier output 1: Call planner via frozen/static action

The classifier learns to identify states where the learned RL policy is insufficient
and should defer to symbolic planning.
"""

from __future__ import annotations

# === Standard library ===
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

# === Third-party ===
import torch

from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.benchmarks.wrappers import PlanningStatesVectorEnv
from skill_refactor.settings import CFG

# === Local ===


class ManualPolicy(Policy):
    """PPO-C policy combining PPO with classifier learning for deciding when to call the
    planner."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        super().__init__(seed)
        self.device = CFG.device

    @property
    def requires_training(self) -> bool:
        """Check if the policy requires training."""
        return True

    def initialize(self, env: PlanningStatesVectorEnv) -> None:
        """Initialize the PPO-C policy with the given environment.

        Sets up PPO agent, classifier network, replay buffer, and training buffers.
        """

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        del obs  # Unused

    def can_initiate(self) -> bool:
        return True

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the policy.

        If classifier is enabled, uses the classifier to decide between PPO policy and
        calling the planner (frozen action). Otherwise, uses PPO policy only.
        """
        forward_action = torch.zeros_like(obs[:, :3])
        forward_action[:, 0] = 1.0  # Full forward throttle
        forward_action[:, 1] = 0.3
        forward_action[:, 2] = 0.1
        return forward_action

    def terminate(self, obs):
        """Terminate when close enough to the target object.

        Using ReadyGrasp predicate logic.

        Immediately terminate if there is no current plan or collision is detected.
        """
        never_terminate = torch.zeros(obs.shape[0], dtype=torch.bool)
        return never_terminate

    def evaluate(
        self,
        eval_env: PlanningStatesVectorEnv,
    ) -> tuple[int, int, dict]:
        """Evaluate the current policy.

        Args:
            eval_env: Evaluation environment
            global_step: Current global step for logging

        Returns:
            Tuple of (evaluated_steps, num_episodes, eval_metrics)
        """
        logging.info("Evaluating")
        eval_obs, _ = eval_env.reset()
        eval_metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(eval_env.skill_max_steps):
            with torch.no_grad():
                eval_obs, _, _, _, eval_infos = eval_env.step(self.get_action(eval_obs))
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)

        return 0, 0, eval_metrics

    def train(
        self,
        env: PlanningStatesVectorEnv,
        eval_env: Optional[PlanningStatesVectorEnv] = None,
        train_data: Optional[object] = None,
    ) -> None:
        """Two-stage training: PPO followed by classifier.

        Stage 1: Train PPO until reward threshold or max iterations
        Stage 2: Load best PPO, train classifier to learn when to call planner
        """
        del train_data  # Unused

        if eval_env is not None:
            self.evaluate(eval_env)  # Evaluate loaded model

    def load(self, path: Path) -> None:
        """Load the policy from the given path.

        Args:
            path: Path to load the policy from
        """
        del path  # Unused
        logging.info("Manual policy has no parameters to load.")

    def save(self, path: Path) -> None:
        """Save the policy to the given path.

        Args:
            path: Path to save the policy to
        """
        del path  # Unused
        logging.info("Manual policy has no parameters to save.")
