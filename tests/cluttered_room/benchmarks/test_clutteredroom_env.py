"""Tests for ClutteredRoom environment."""

import gymnasium as gym
import torch

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG


# @pytest.mark.skip(reason="The script requires maniskills installation")
def test_cluttered_room_env():
    """Test basic functionality of ClutteredTable environment."""
    # x: > -0.7
    # y: < -0.24 > 0.24
    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "num_envs": 1,
        "c_room_spot_body_x": -0.5,
        "c_room_spot_body_y": -0.5,
    }
    register_all_environments()
    reset_config(test_config)

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "skill_ref/ClutteredRoom-v1",
        num_envs=CFG.num_envs,
        reconfiguration_freq=0,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-room-test",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=100,
        video_fps=30,
    )

    for s in range(5):
        obs, _ = env.reset(seed=s)

        env.action_space.seed(123)

        # 3-dim base + 6-dim arm + 1-dim gripper
        basic_action = torch.zeros((1, 10), dtype=torch.float32).to(env.device)
        # Set the first action to 1.0, arm_0
        # Set the x_vel to -0.5 to move further to the drawer
        basic_action[:, -3] = -1.0

        for _ in range(50):
            obs, _, _, _, _info = env.step(basic_action)
            print(f"Obs: {obs}")

    env.close()


def test_cluttered_room_held_env():
    """Test basic functionality of ClutteredTable environment."""
    # x: > -0.7
    # y: < -0.24 > 0.24
    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "num_envs": 1,
        "c_room_spot_body_x": -0.5,
        "c_room_spot_body_y": -0.5,
    }
    register_all_environments()
    reset_config(test_config)

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "skill_ref/ClutteredRoomForceHeld-v1",
        num_envs=CFG.num_envs,
        reconfiguration_freq=0,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-room-held-test",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=100,
        video_fps=30,
    )

    for s in range(1):
        obs, _ = env.reset(seed=s)

        env.action_space.seed(123)

        # 3-dim base + 6-dim arm + 1-dim gripper
        basic_action = torch.zeros((1, 10), dtype=torch.float32).to(env.device)
        # Set the first action to 1.0, arm_0
        # Set the x_vel to -0.5 to move further to the drawer
        basic_action[:, -3] = -1.0

        for _ in range(50):
            obs, _, _, _, _info = env.step(basic_action)
            print(f"Obs: {obs}")

    env.close()
