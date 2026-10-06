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
from typing import List, Optional

# === Third-party ===
import torch
from mani_skill.utils.structs.pose import Pose

# === Local ===
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.benchmarks.wrappers import PlanningStatesVectorEnv
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.motion_planning import SpotMotion, WaypointTracker


class ManualPolicy(Policy):
    """PPO-C policy combining PPO with classifier learning for deciding when to call the
    planner."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        super().__init__(seed)
        self.device = CFG.device
        self.spot_motion_generator = SpotMotion(
            device=CFG.device,
        )
        self._current_plan: List[Tensor] | None = []
        self.spot_motion_tracker: Optional[WaypointTracker] = None

    @property
    def requires_training(self) -> bool:
        """Check if the policy requires training."""
        return True

    def initialize(self, env: PlanningStatesVectorEnv) -> None:
        """Initialize the PPO-C policy with the given environment.

        Sets up PPO agent, classifier network, replay buffer, and training buffers.
        """
        # Update the args with the environment-specific values
        self._current_plan = []
        (
            self.normalize_action,
            self.arm_action_low,
            self.arm_action_high,
        ) = get_normalize_action_range(env._env, CFG.control_mode)

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        del obs  # Unused
        self._current_plan = []

    def can_initiate(self) -> bool:
        return True

    def _get_kinematic_plan(
        self,
        observation: Tensor,
    ) -> List[Tensor]:
        # get transformation matrix of the tcp pose, is default batched and on torch
        robot_body_pose_tensor = observation[:, 27:34]
        robot_hand_pose_tensor = observation[:, 20:27]
        from_pose_p = robot_hand_pose_tensor[:, 0:3].clone()
        from_pose_q = robot_hand_pose_tensor[:, 3:7].clone()
        from_pose = Pose.create_from_pq(from_pose_p, from_pose_q)
        from_joints = observation[:, 0:10].clone()

        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        target_hand_p = from_pose_p.clone()
        target_hand_p[:, 0] -= 0.3
        target_hand_p[:, 2] -= 0.2

        target_hand_pose = Pose.create_from_pq(
            target_hand_p,
            q=from_pose_q,
        )

        # 1. Move hand to default pose
        arm_actions = self.spot_motion_generator.move_hand_from_to_pose(
            robot_body_pose,
            from_joints,
            from_pose,
            target_hand_pose,
            closing=torch.zeros_like(from_joints[:, 0:1]).to(torch.bool),
            interpolate_steps=12,
        )

        # 2. Then move body to align with target object
        # target_body_p = robot_body_pose_tensor.clone()
        # target_body_p[:, 0] -= 0.5
        # target_body_pose = Pose.create_from_pq(
        #     target_body_p[:, 0:3],
        #     target_body_p[:, 3:7],
        # )
        # body_actions = self.spot_motion_generator.move_body_from_to_pose(
        #     robot_worldF_curr=robot_body_pose,
        #     robot_worldF_tgt=target_body_pose,
        #     curr_joint_positions=from_joints,
        #     closing=torch.zeros_like(from_joints[:, 0:1]).to(torch.bool),
        #     interpolate_steps=30,
        # )

        # # Combine arm and body actions
        # for arm_action, body_action in zip(arm_actions, body_actions):
        #     conbined_action = torch.zeros_like(arm_action)
        #     conbined_action[:, :3] = body_action[:, :3]
        #     conbined_action[:, 3:] = arm_action[:, 3:]
        #     reach_actions.append(conbined_action)

        return arm_actions

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the policy.

        If classifier is enabled, uses the classifier to decide between PPO policy and
        calling the planner (frozen action). Otherwise, uses PPO policy only.
        """
        if not self._current_plan:
            # get kinematic plan (qpos) given objects and observation
            self._current_plan = self._get_kinematic_plan(obs)
            self.spot_motion_tracker = WaypointTracker(
                plan=self._current_plan,
                normalize_action=self.normalize_action,
                arm_action_low=self.arm_action_low,
                arm_action_high=self.arm_action_high,
                waypoint_threshold=100,  # large enough to avoid skipping waypoints
                angular_threshold=100,  # large enough to avoid skipping waypoints
                device=self.device,
            )
        # Type narrowing for mypy
        assert self.spot_motion_tracker is not None, "spot_motion_tracker must be set"
        curr_qpos = obs[:, :10]
        delta_qpos = self.spot_motion_tracker.compute_delta_actions(curr_qpos)
        # Update _current_plan as the tracker's remaining plan
        self._current_plan = self.spot_motion_tracker.plan
        return delta_qpos

    def terminate(self, obs):
        """Terminate when close enough to the target object.

        Using ReadyGrasp predicate logic.

        Immediately terminate if there is no current plan or collision is detected.
        """

        assert self._current_plan is not None

        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(collision, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(collision, dtype=torch.bool)
        )
        terminated = not_have_current_plan_tensor
        return terminated

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
