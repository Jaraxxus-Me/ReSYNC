"""Tests for CarPDController."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from prbench.envs.geom2d.structs import SE2Pose

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo, MultiEnvWrapper
from skill_refactor.settings import CFG
from skill_refactor.utils.motion_planning import CarPDController, rotate_vectors


def test_car_pd_controller_single_waypoint_trans():
    """Test CarPDController produces correct output shapes and works with batched
    environments.

    Note: This test verifies the API and batching behavior, not controller performance.
    Controller tuning is left for actual usage scenarios.
    """
    test_config = {
        "num_envs": 4,
        "device": "cpu",
        "normalize_action": False,
        "kp_pos": 100.0,
        "kv_pos": 20.0,
        "kp_ang": 50.0,
        "kv_ang": 10.0,
    }
    reset_config(test_config)
    register_all_environments()

    # Create batched environments
    def make_env():
        return gym.make("skill_ref/IcyTransport2D-v0")

    envs = MultiEnvWrapper(
        make_env,
        num_envs=CFG.num_envs,
        auto_reset=False,
        to_tensor=True,
        device=CFG.device,
    )

    # envs = MultiEnvRecordVideo(
    #     envs,
    #     f"videos/control_actions_trans/",
    #     episode_trigger=lambda episode_id: True,
    # )

    # Reset environments
    obs, _ = envs.reset(seed=42)

    controller = CarPDController(
        kp_pos=CFG.kp_pos,
        kv_pos=CFG.kv_pos,
        kp_ang=CFG.kp_ang,
        kv_ang=CFG.kv_ang,
        device=CFG.device,
    )

    # Extract robot state from observation
    # For IcyTransport2D, robot features should be first in observation
    # We need to identify where x, y, theta, vx, vy, omega are
    # Based on the state dict, robot has: x, y, theta, vx_base, vy_base, omega_base

    # Define target positions for each environment (same target for simplicity)
    robot_pos = obs[:, 0:3]  # Assuming x, y are first two features]
    target_positions = robot_pos.clone()
    desired_forward = torch.tensor(
        [[0, 0.5]] * CFG.num_envs, dtype=torch.float32, device=CFG.device
    )
    rotated_forward = rotate_vectors(desired_forward, robot_pos[:, 2])
    target_positions[:, 0:2] += rotated_forward  # Move 0.5

    # Test controller for a few steps to verify it works
    max_steps = 50

    # Get action space bounds for clipping
    action_low = torch.tensor(
        envs.action_space.low[0], dtype=torch.float32, device=CFG.device
    )
    action_high = torch.tensor(
        envs.action_space.high[0], dtype=torch.float32, device=CFG.device
    )

    for _ in range(max_steps):
        # Extract robot state from observation
        # Assuming obs contains robot state at beginning: [x, y, theta, vx, vy, omega, ...]
        # Compute control
        control = controller.compute_control(obs, target_positions)

        # Check control shape
        assert control.shape == (
            CFG.num_envs,
            3,
        ), f"Control shape should be ({CFG.num_envs}, 3)"

        # Clip control to action space bounds
        control = torch.clamp(control, action_low, action_high)

        # Apply control
        obs, _, _, _, _ = envs.step(control)

    envs.close()


def test_car_pd_controller_single_waypoint_rot():
    """Test CarPDController produces correct output shapes and works with batched
    environments.

    Note: This test verifies the API and batching behavior, not controller performance.
    Controller tuning is left for actual usage scenarios.
    """
    test_config = {
        "num_envs": 4,
        "device": "cpu",
        "normalize_action": False,
        "kp_pos": 100.0,
        "kv_pos": 20.0,
        "kp_ang": 50.0,
        "kv_ang": 10.0,
    }
    reset_config(test_config)
    register_all_environments()

    # Create batched environments
    def make_env():
        return gym.make("skill_ref/IcyTransport2D-v0")

    envs = MultiEnvWrapper(
        make_env,
        num_envs=CFG.num_envs,
        auto_reset=False,
        to_tensor=True,
        device=CFG.device,
    )

    # envs = MultiEnvRecordVideo(
    #     envs,
    #     f"videos/control_actions_trans_rot/",
    #     episode_trigger=lambda episode_id: True,
    # )

    # Reset environments
    obs, _ = envs.reset(seed=42)

    controller = CarPDController(
        kp_pos=CFG.kp_pos,
        kv_pos=CFG.kv_pos,
        kp_ang=CFG.kp_ang,
        kv_ang=CFG.kv_ang,
        device=CFG.device,
    )

    # Extract robot state from observation
    # For IcyTransport2D, robot features should be first in observation
    # We need to identify where x, y, theta, vx, vy, omega are
    # Based on the state dict, robot has: x, y, theta, vx_base, vy_base, omega_base

    # Define target positions for each environment (same target for simplicity)
    robot_pos = obs[:, 0:3]  # Assuming x, y are first two features]
    target_positions = robot_pos.clone()
    target_positions[:, 2] += torch.pi / 4  # Rotate 45 degrees

    # Test controller for a few steps to verify it works
    max_steps = 50

    # Get action space bounds for clipping
    action_low = torch.tensor(
        envs.action_space.low[0], dtype=torch.float32, device=CFG.device
    )
    action_high = torch.tensor(
        envs.action_space.high[0], dtype=torch.float32, device=CFG.device
    )

    for _ in range(max_steps):
        # Extract robot state from observation
        # Assuming obs contains robot state at beginning: [x, y, theta, vx, vy, omega, ...]
        # Compute control
        control = controller.compute_control(obs, target_positions)

        # Check control shape
        assert control.shape == (
            CFG.num_envs,
            3,
        ), f"Control shape should be ({CFG.num_envs}, 3)"

        # Clip control to action space bounds
        control = torch.clamp(control, action_low, action_high)

        # Apply control
        obs, _, _, _, _ = envs.step(control)

    envs.close()
