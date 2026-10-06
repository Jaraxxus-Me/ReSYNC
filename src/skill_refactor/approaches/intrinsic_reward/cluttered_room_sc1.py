"""Example intrinsic reward for cluttered drawer environment.

This example demonstrates how to define a task-specific intrinsic reward
for the cluttered drawer manipulation task.

Usage in config or command line:
    --planner_eval False
    --intrinsic_reward_path src/skill_refactor/approaches/intrinsic_reward/cluttered_drawer_example.py
"""

import torch

from skill_refactor.benchmarks.cluttered_room.utils import (
    extract_can_pose,
    extract_obj2_goal_pose,
    extract_obj2_grasped,
    extract_robot_hand_pose,
)


def intrinsic_rwd(obs: torch.Tensor) -> torch.Tensor:
    """Compute intrinsic reward for cluttered drawer task.

    Goal: Guide the policy to achieve desired object states (e.g., object in drawer,
    drawer closed) without running the full planner.

    Args:
        obs: Observation tensor with shape (num_envs, obs_dim)

    Returns:
        Reward tensor with shape (num_envs,) in range [-1, 1]
    """
    num_envs = obs.shape[0]
    device = obs.device

    # Default: return zero for all environments
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    can_pose_tensor = extract_can_pose(obs)
    obj2_goal_pose_tensor = extract_obj2_goal_pose(obs)
    obj2_grasped_tensor = extract_obj2_grasped(obs).squeeze(-1).to(torch.bool)
    _robot_hand_pose_tensor = extract_robot_hand_pose(obs)

    dist_can_to_goal = torch.norm(
        can_pose_tensor[:, :2] - obj2_goal_pose_tensor[:, :2], dim=-1
    )
    dist_large_engough = dist_can_to_goal > 0.15
    # hand_high = robot_hand_pose_tensor[:, 2] > 1.0

    subgoal_achieved = dist_large_engough & obj2_grasped_tensor
    reward[subgoal_achieved] += 1.0
    return reward
