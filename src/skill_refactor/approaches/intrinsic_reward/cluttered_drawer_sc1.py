"""Example intrinsic reward for cluttered drawer environment.

This example demonstrates how to define a task-specific intrinsic reward
for the cluttered drawer manipulation task.

Usage in config or command line:
    --planner_eval False
    --intrinsic_reward_path src/skill_refactor/approaches/intrinsic_reward/cluttered_drawer_example.py
"""

import torch

from skill_refactor.benchmarks.cluttered_drawer.utils import (
    extract_blocking_drawer_q,
    extract_grasp_hammer_pose,
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

    curr_drawer_q = extract_blocking_drawer_q(obs).squeeze()  # shape (num_envs,)
    curr_drawer_q_large_engough = curr_drawer_q > 0.26  # drawer is open enough
    curr_hammer_pose = extract_grasp_hammer_pose(obs)
    curr_hammer_x_ok = curr_hammer_pose[:, 0] > 0.0  # hammer is in the drawer area
    curr_hammer_z_ok = (
        curr_hammer_pose[:, 2] < 0.4
    )  # hammer is low enough in the drawer
    subgoal_achieved = curr_drawer_q_large_engough & curr_hammer_x_ok & curr_hammer_z_ok
    reward[subgoal_achieved] += 1.0
    return reward
