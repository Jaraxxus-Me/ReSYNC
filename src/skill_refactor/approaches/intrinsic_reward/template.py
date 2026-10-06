"""Template intrinsic reward function for RL training without planner evaluation.

This template provides a starting point for defining custom intrinsic rewards
that can be used for fast policy iteration.

Usage:
    1. Copy this file and modify the intrinsic_rwd function
    2. Set CFG.planner_eval = False
    3. Set CFG.intrinsic_reward_path to point to your custom reward file
"""

import torch


def intrinsic_rwd(obs: torch.Tensor) -> torch.Tensor:
    """Compute intrinsic reward from observation.

    This function is called at the end of each RL episode (after skill_max_steps)
    to provide a reward signal that guides learning without running the planner.

    Args:
        obs: Observation tensor from the environment with shape (num_envs, obs_dim).
            This is the full observation from the base environment, NOT the clipped
            observation used by the RL policy.

    Returns:
        Reward tensor with shape (num_envs,). Values are typically in range [-1, 1]
        to match the planner-based reward scale.

    Example implementations:
        1. Distance-based reward (encourage getting closer to goal):
            ```python
            # Extract object position from observation
            obj_pos = obs[:, 10:13]  # Example indices
            goal_pos = torch.tensor([0.5, 0.0, 0.1], device=obs.device)
            distance = torch.norm(obj_pos - goal_pos, dim=1)
            # Negative distance as reward (closer = higher reward)
            return -distance
            ```

        2. Binary success reward (check if goal condition is met):
            ```python
            # Check if object is in target region
            obj_pos = obs[:, 10:13]
            target_low = torch.tensor([0.4, -0.1, 0.0], device=obs.device)
            target_high = torch.tensor([0.6, 0.1, 0.2], device=obs.device)
            in_target = ((obj_pos >= target_low) & (obj_pos <= target_high)).all(dim=1)
            # Return 1.0 for success, -1.0 for failure
            return in_target.float() * 2.0 - 1.0
            ```

        3. Multi-objective reward (combine multiple factors):
            ```python
            # Distance to goal
            obj_pos = obs[:, 10:13]
            goal_pos = torch.tensor([0.5, 0.0, 0.1], device=obs.device)
            dist_reward = -torch.norm(obj_pos - goal_pos, dim=1) * 0.5

            # Gripper alignment (encourage proper orientation)
            gripper_quat = obs[:, 7:11]
            alignment_reward = gripper_quat[:, 3].abs() * 0.3  # w component

            # Stability (penalize high velocity)
            obj_vel = obs[:, 13:16]
            stability_penalty = -torch.norm(obj_vel, dim=1) * 0.2

            return dist_reward + alignment_reward + stability_penalty
            ```
    """
    # Default implementation: uniform zero reward
    # Replace this with your custom reward logic
    num_envs = obs.shape[0]
    device = obs.device

    # Example: Simple distance-based reward
    # Modify the indices and target position based on your environment
    # obj_pos = obs[:, 10:13]  # Replace with actual indices for object position
    # goal_pos = torch.tensor([0.5, 0.0, 0.1], device=device)
    # distance = torch.norm(obj_pos - goal_pos, dim=1)
    # reward = torch.clamp(-distance, min=-1.0, max=1.0)

    # Placeholder: return zero reward
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)

    return reward
