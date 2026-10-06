"""Tests for ClutteredTable environment with (pure) TAMP."""

# import imageio.v2 as iio
import pickle
import time

import pytest
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config

# NOTE: cluttered_table has been removed, these tests are skipped
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
    ClutteredDrawerRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


# @pytest.mark.skip(reason="The script generates local data")
@pytest.mark.parametrize("seeed", [0, 1, 2, 3, 4])
def test_cluttered_drawer_task_gen(seeed):
    """Test BlockedStacking environment with a pure TAMP planner."""

    sc = "1,2,3"
    seed = seeed
    task_save_path = f"config/specified_tasks/cluttered_drawer"
    test_config = {
        "seed": seed,
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        f"drawer_blocking_grasp": False,
        f"drawer_blocking_stacking": True,
        f"block_blocking_grasp": False,
        f"block_blocking_stacking": True,
        f"wall_blocking_grasp": True,
        f"wall_blocking_stacking": False,
        "delta_finger_control": False,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()  # type: ignore[attr-defined]
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir=f"videos/tasks-sc{sc}-seed{seed}",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )
    # envs = tamp_system.env

    for i in range(10):
        obs, info = envs.reset(seed=CFG.seed + i)
        task = planner.generate_task(obs[0:1], info)
        file_path = f"{task_save_path}/sc{sc}_task_seed{CFG.seed}_id{i}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(task, f)
        for _ in range(5):
            _, _, _, _, _ = envs.step(
                envs.action_space.sample()  # type: ignore[attr-defined]
            )
    envs.close()


# @pytest.mark.skip(reason="The script is used to run experiments locally")
def test_clutted_drawer_base_tamp():
    """Test ClutteredDrawer environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        f"drawer_blocking_grasp": False,
        f"drawer_blocking_stacking": True,
        f"block_blocking_grasp": False,
        f"block_blocking_stacking": True,
        f"wall_blocking_grasp": True,
        f"wall_blocking_stacking": False,
        "num_envs": 16,
        "c_drawer_close_frac": 0.8,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-base-tamp-non-debug1",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )

    for s in range(10):
        obs, info = envs.reset(seed=s)

        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(600):
            action, _ = planner.step(obs)
            obs, reward, _, _, infos = envs.step(action)
            total_reward += reward
            print(f"Step {step + 1}: Action: {action}, Obs: {obs}, Reward: {reward}")
            if all(infos["success"]):
                print("Episode finished successfully")
                break
        else:
            print("Episode didn't finish within 300 steps")

    envs.close()


def test_clutted_drawer_sc1_grasp_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        f"drawer_blocking_grasp": True,
        f"drawer_blocking_stacking": False,
        "num_envs": 16,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "1",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-grasping-failure-sc" + str(1),
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(300):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    assert step < 299
    envs.close()


def test_clutted_drawer_sc1_place_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        f"drawer_blocking_grasp": False,
        f"drawer_blocking_stacking": True,
        "num_envs": 16,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "1",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-grasping-failure-sc" + str(1),
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(700):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    assert step < 699
    envs.close()


def test_clutted_drawer_sc12_grasp_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        f"drawer_blocking_grasp": False,
        f"drawer_blocking_stacking": True,
        f"block_blocking_grasp": True,
        f"block_blocking_stacking": False,
        "dreaming_noise_base_var": 0.0,
        "num_envs": 16,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "1,2",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-grasping-failure-sc" + "1,2",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(800):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    assert step < 799
    envs.close()


def test_clutted_drawer_sc12_place_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        f"drawer_blocking_grasp": True,
        f"drawer_blocking_stacking": False,
        "block_blocking_grasp": False,
        "block_blocking_stacking": True,
        "num_envs": 16,
        "c_drawer_close_frac": 1.0,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "1,2",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-placing-failure-sc" + "1,2",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=400,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(400):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    # assert step < 299
    envs.close()


def test_clutted_drawer_sc123_grasp_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        f"drawer_blocking_grasp": False,
        f"drawer_blocking_stacking": True,
        f"block_blocking_grasp": False,
        f"block_blocking_stacking": True,
        f"wall_blocking_grasp": True,
        f"wall_blocking_stacking": False,
        "num_envs": 16,
        "c_drawer_close_frac": 1.0,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "1,2,3",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-grasping-failure-sc" + "1,2,3",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(300):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    assert step < 299
    envs.close()


def test_clutted_drawer_sc3_place_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        f"drawer_blocking_grasp": True,
        f"drawer_blocking_stacking": False,
        f"block_blocking_grasp": True,
        f"block_blocking_stacking": False,
        f"wall_blocking_grasp": False,
        f"wall_blocking_stacking": True,
        "num_envs": 16,
        "c_drawer_close_frac": 1.0,
        "device": "cuda:0",
        "delta_finger_control": False,
        "scenario": "3",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env.unwrapped, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-placing-failure-sc" + "3",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=400,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(400):
            action, _ = planner.step(obs)
            s = time.time()
            obs, reward, _, _, _ = envs.step(action)
            if torch.any(obs[:, -1]):
                print(f"Collide at step {step}")
                break
            # print("Step time:", time.time() - s)
            # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
            total_reward += reward
    # Should collide in less than 49 steps
    assert step < 399
    envs.close()


def test_clutted_drawer_sc1_full_to_relative_consistency():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "drawer_blocking_grasp": True,
        "drawer_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    obs, _ = tamp_system.env.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_hammer")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"


def test_clutted_drawer_sc1_full_to_relative_consistency_change_node():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "drawer_blocking_grasp": True,
        "drawer_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-rel2full-reset",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_hammer")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)
    # 6-dim arm + 1-dim gripper
    basic_action = torch.zeros((CFG.num_envs, 10), dtype=torch.float32).to(envs.device)
    # Set the first action to 1.0, arm_0
    # Set the x_vel to -0.5 to move further to the drawer
    basic_action[:, -3] = -1.0

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"

    # Move the obstruction to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] += 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    assert torch.all(
        recon_error2 < 1e-4
    ), f"Reconstruction error too high: {recon_error2}"

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # For video saving
    _, _ = envs.reset()

    envs.close()


def test_clutted_drawer_sc12_full_to_relative_consistency():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    obs, _ = tamp_system.env.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_hammer")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"


def test_clutted_drawer_sc12_full_to_relative_consistency_change_node():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        "delta_finger_control": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-rel2full-reset12",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_hammer")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)
    # 6-dim arm + 1-dim gripper
    basic_action = torch.zeros((CFG.num_envs, 10), dtype=torch.float32).to(envs.device)
    basic_action[:, -3] = -1.0

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 0.03
    ), f"Reconstruction error too high: {recon_error}"

    # Move the obstruction and drawer to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] += 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    assert torch.all(
        recon_error2 < 0.03
    ), f"Reconstruction error too high: {recon_error2}"

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # For video saving
    _, _ = envs.reset()

    envs.close()


def test_clutted_drawer_sc123_full_to_relative_consistency_change_node():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "dreaming_noise_base_var": 0.0,
        "delta_finger_control": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2,3",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-drawer-rel2full-reset123",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )

    obs, _ = envs.reset(seed=0)
    relative_obs_1 = tamp_system.full_state_to_relative_state(obs, "grasp_hammer")
    relative_obs_2 = tamp_system.full_state_to_relative_state(obs, "target_hammer")
    relative_obs_3 = tamp_system.full_state_to_relative_state(obs, "target_hammer")
    relative_obs = torch.stack(
        [relative_obs_1[:, 0, :], relative_obs_2[:, 1, :], relative_obs_3[:, 2, :]],
        dim=1,
    )
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)
    # 6-dim arm + 1-dim gripper
    basic_action = torch.zeros((CFG.num_envs, 10), dtype=torch.float32).to(envs.device)
    basic_action[:, -3] = -1.0

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 0.03
    ), f"Reconstruction error too high: {recon_error}"

    # Move the obstruction and drawer to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, 0, 0] += 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    _recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    # assert torch.all(
    #     _recon_error2 < 0.03
    # ), f"Reconstruction error too high: {_recon_error2}"

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # For video saving
    _, _ = envs.reset()

    envs.close()
