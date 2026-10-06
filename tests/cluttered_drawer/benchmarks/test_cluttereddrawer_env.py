"""Tests for core ClutteredTable environment."""

import gymnasium as gym
import mani_skill.envs  # type: ignore # pylint: disable=unused-import
import torch

from skill_refactor.args import reset_config
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo


# @pytest.mark.skip(reason="The script requires maniskills installation")
def test_cluttered_drawer_env():
    """Test basic functionality of ClutteredTable environment."""
    # x: > -0.7
    # y: < -0.24 > 0.24
    test_config = {
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        "delta_finger_control": False,
        "scenario": "1",
        "num_envs": 16,
    }
    reset_config(test_config)

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "ClutteredDrawer-v1",
        num_envs=4,
        reconfiguration_freq=None,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-drawer-test",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=100,
        video_fps=30,
    )

    for s in range(1):
        obs, _ = env.reset(seed=s)

        env.action_space.seed(123)

        # 6-dim arm + 1-dim gripper
        basic_action = torch.zeros((4, 10), dtype=torch.float32).to(env.device)
        # Set the first action to 1.0, arm_0
        # Set the x_vel to -0.5 to move further to the drawer
        basic_action[:, -3] = -1.0

        for _ in range(50):
            obs, _, _, _, _info = env.step(basic_action)
            print(f"Obs: {obs}")

    env.close()


def test_cluttered_drawer_env_sc123():
    """Test basic functionality of ClutteredTable environment."""
    # x: > -0.7
    # y: < -0.24 > 0.24
    test_config = {
        "seed": 0,
        "dreaming_noise_base_var": 0.0,
        "delta_finger_control": False,
        "scenario": "1,2,3",
        "debug_env": False,
        "num_envs": 16,
    }
    reset_config(test_config)

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "ClutteredDrawer-v1",
        num_envs=4,
        reconfiguration_freq=None,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-drawer-test2",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=100,
        video_fps=30,
    )

    for s in range(1):
        obs, _ = env.reset(seed=s)

        env.action_space.seed(123)

        # 6-dim arm + 1-dim gripper
        basic_action = torch.zeros((4, 10), dtype=torch.float32).to(env.device)
        # Set the first action to 1.0, arm_0
        # Set the x_vel to -0.5 to move further to the drawer
        basic_action[:, -3] = -1.0

        for _ in range(50):
            obs, _, _, _, _info = env.step(basic_action)
            print(f"Obs: {obs}")

    env.close()
