"""Tests for ClutteredRoom environment with (pure) TAMP."""

import pickle
import time

import pytest
import torch

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.parametrize("seeed", [3, 4])
def test_cluttered_room_task_gen(seeed):
    """Test ClutteredRoom environment with a pure TAMP planner."""

    sc = "1"
    seed = seeed
    task_save_path = f"config/specified_tasks/cluttered_room"
    test_config = {
        "seed": seed,
        "debug_env": False,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        f"can_blocking_target1": False,
        f"can_blocking_target2": True,
        "num_envs": 1,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
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

    for i in range(0, 10):
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


def test_cluttered_room_base_tamp():
    """Test ClutteredRoom environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "num_envs": 1,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
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
        output_dir="videos/c-room-base-tamp",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=4000,
        video_fps=30,
    )

    for s in range(1):
        obs, info = envs.reset(seed=s)

        start_time = time.time()
        planner.reset(obs, info)
        print(f"Planner reset time: {time.time() - start_time:.3f}s")

        total_reward = 0
        for step in range(4000):
            action, _ = planner.step(obs)
            obs, reward, _, _, infos = envs.step(action)
            total_reward += reward
            print(
                f"Step {step + 1}: Reward: {reward.item():.3f}, Total: {total_reward.item():.3f}"
            )
            if all(infos["success"]):
                print(f"Episode {s} finished successfully at step {step + 1}")
                break
        else:
            print(f"Episode {s} didn't finish within 1000 steps")

    envs.close()


def test_cluttered_room_sc1_placing_failure():
    """Test ClutteredRoom environment with a pure TAMP planner."""

    test_config = {
        "seed": 0,
        "debug_env": False,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        f"can_blocking_target1": False,
        f"can_blocking_target2": True,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": f"1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
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
        output_dir="videos/c-room-placing-failure-sc" + str(1),
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=3500,
        video_fps=30,
    )
    # envs = tamp_system.env
    for s in range(1):
        obs, info = envs.reset(seed=s)
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        total_reward = 0
        for step in range(3500):
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
    assert step < 3500
    envs.close()


def test_cluttered_room_sc1_full_to_relative_consistency():
    """Test ClutteredRoom environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        f"can_blocking_target1": False,
        f"can_blocking_target2": True,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    obs, _ = tamp_system.env.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "box_goal")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-1
    ), f"Reconstruction error too high: {recon_error}"


def test_cluttered_room_sc1_full_to_relative_consistency_change_node():
    """Test ClutteredRoom environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        f"can_blocking_target1": False,
        f"can_blocking_target2": True,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    envs = ManiSkillsRecordVideo(
        tamp_system.env,
        output_dir="videos/c-room-rel2full-reset",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=600,
        video_fps=30,
    )

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "box_goal")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)
    # 6-dim arm + 1-dim gripper
    basic_action = torch.zeros((CFG.num_envs, 10), dtype=torch.float32).to(envs.device)
    # Set the first action to 1.0, arm_0
    # Set the x_vel to -0.5 to move further to the drawer
    basic_action[:, -1] = -1.0

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-1
    ), f"Reconstruction error too high: {recon_error}"

    # Move the obstruction to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] -= 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs[:, 22:] - reset_obs2[:, 22:], dim=1)
    assert torch.all(
        recon_error2 < 1e-1
    ), f"Reconstruction error too high: {recon_error2}"

    for _ in range(10):
        _, _, _, _, _ = envs.step(basic_action)

    # For video saving
    _, _ = envs.reset()

    envs.close()
