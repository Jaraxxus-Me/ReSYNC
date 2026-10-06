"""Tests for ClutteredTable environment with (pure) TAMP."""

import logging
import pickle
import time
from pathlib import Path
from typing import List

import pytest

# import imageio.v2 as iio
import torch

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import PlannerDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


# @pytest.mark.skip(reason="The script generates local data")
@pytest.mark.parametrize("seeed", [0, 1, 2, 3, 4])
def test_icy_transport_task_gen(seeed):
    """Test IcyTransport environment with a pure TAMP planner."""

    sc = "1,2"
    seed = seeed
    task_save_path = f"config/specified_tasks/icy_transport"
    test_config = {
        "seed": seed,
        "debug_env": False,
        f"icy_infront_of_transport1": False,
        f"icy_infront_of_transport2": True,
        f"icy_infront_of_target": False,
        f"muddy_infront_of_transport1": True,
        f"muddy_infront_of_transport2": False,
        f"muddy_infront_of_target": False,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
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

    envs = MultiEnvRecordVideo(
        tamp_system.env,
        f"videos/tasks-sc{sc}-seed{seed}",
        episode_trigger=lambda episode_id: True,
    )
    # envs = tamp_system.env

    for i in range(10):
        obs, info = envs.reset(seed=CFG.seed + i)
        task = planner.generate_task(obs[0:1], info)
        file_path = f"{task_save_path}/sc{sc}_task_seed{CFG.seed}_id{i}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(task, f)
    envs.close()


def test_icy_transport_base_tamp():
    """Test IcyTransport environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "device": "cpu",
        "num_envs": 2,
        "scenario": "1",
        "icy_infront_of_transport1": False,
        "icy_infront_of_transport2": False,
        "icy_infront_of_target": False,
        "control_mode": "force_torque",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
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

    # envs = tamp_system.env
    envs = MultiEnvRecordVideo(
        tamp_system.env,
        "videos/icy-transport-base-tamp",
        episode_trigger=lambda episode_id: True,
    )

    for s in range(1):
        obs, info = envs.reset(seed=s)

        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        for step in range(2400):
            action, _ = planner.step(obs)
            obs, reward, terminated, _, _infos = envs.step(action)
            print(f"Step {step + 1}: Reward: {reward.sum().item()}")
            if terminated.all():
                print(f"All episodes done at step {step}")
                break

    envs.close()


def test_icy_transport_base_tamp_same_room():
    """Test IcyTransport environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "device": "cpu",
        "num_envs": 2,
        "scenario": "1",
        "icy_infront_of_transport": False,
        "icy_infront_of_target": False,
        "control_mode": "force_torque",
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
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

    envs = tamp_system.env
    # envs = MultiEnvRecordVideo(
    #     tamp_system.env,
    #     "videos/icy-transport-base-tamp-same-room",
    #     episode_trigger=lambda episode_id: True,
    # )

    for s in range(1):
        obs, info = envs.reset(seed=s)
        new_obs = obs.clone()
        new_obs[:, -42] = 0.8
        new_obs[:, -41] = 0.8
        reset_obs, _ = envs.reset(options={"init_state": new_obs})
        s = time.time()
        planner.reset(obs, info)
        print("Planner reset time:", time.time() - s)

        for step in range(600):
            action, _ = planner.step(reset_obs)
            reset_obs, reward, terminated, _, _infos = envs.step(action)
            print(f"Step {step + 1}: Reward: {reward.sum().item()}")
            if terminated.all():
                print(f"All episodes done at step {step}")
                break

    envs.close()


def test_icy_transport_sc1_or_2_or_3_transport_failure():
    """Test IcyTransport environment with a pure TAMP planner."""

    sc = "1,2"
    test_config = {
        "debug_env": False,
        f"icy_infront_of_transport1": False,
        f"icy_infront_of_transport2": True,
        f"icy_infront_of_target": False,
        f"muddy_infront_of_transport1": True,
        f"muddy_infront_of_transport2": False,
        f"muddy_infront_of_target": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
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

    envs = MultiEnvRecordVideo(tamp_system.env, f"videos/transport-test-sc{sc}")
    # envs = tamp_system.env

    obs, info = envs.reset(seed=0)
    s = time.time()
    planner.reset(obs, info)
    print("Planner reset time:", time.time() - s)

    total_reward = 0
    s = time.time()
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
    # Should collide in less than 400 steps
    # assert step < 399
    print("Total time for 400 steps:", time.time() - s)
    envs.close()


def test_icy_transport_sc1_or_2_or_3_target_failure():
    """Test IcyTransport environment with a pure TAMP planner."""

    sc = 1
    test_config = {
        "debug_env": False,
        f"icy_infront_of_transport": False,
        f"icy_infront_of_target": True,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = tamp_system.env.single_action_space.sample()
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

    envs = MultiEnvRecordVideo(tamp_system.env, f"videos/target-test-sc{sc}")
    # envs = tamp_system.env

    obs, info = envs.reset(seed=0)
    s = time.time()
    planner.reset(obs, info)
    print("Planner reset time:", time.time() - s)

    total_reward = 0
    s = time.time()
    for _ in range(800):
        action, _ = planner.step(obs)
        s = time.time()
        obs, reward, _, _, _ = envs.step(action)
        # if torch.any(obs[:, -1]):
        #     print(f"Collide at step {step}")
        #     break
        # print("Step time:", time.time() - s)
        # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
        total_reward += reward
    # Should collide in less than 400 steps
    # assert step < 399
    print("Total time for 400 steps:", time.time() - s)
    envs.close()


def test_icy_transport_sc1_full_to_relative_consistency():
    """Test IcyTransport environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "icy_infront_of_transport": True,
        "icy_infront_of_target": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/transporting-test-sc1-rel")

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "transport_obj")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"
    envs.close()


def test_icy_transport_sc1_full_to_relative_reset():
    """Test IcyTransport environment with a pure TAMP planner."""

    test_config = {
        "seed": 1,
        "debug_env": False,
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "specified_task_path": "config/specified_tasks/icy_transport",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(
        tamp_system.env,
        "videos/transporting-test-sc1-rel-reset",
        episode_trigger=lambda episode_id: True,
    )

    # Episode 1: Initial reset with random actions
    obs, _ = envs.reset(seed=0)
    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    relative_obs = tamp_system.full_state_to_relative_state(obs, "transport_obj")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Episode 2: Reset to valid state 1 with random actions
    reset_obs, _ = envs.reset(options={"init_state": reconstructed_full_obs})
    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    # Check reconstruction error
    recon_error = torch.norm(obs - reset_obs, dim=1)
    assert torch.all(
        recon_error < 1e-4
    ), f"Reconstruction error too high: {recon_error}"

    # Episode 3: Move the obstruction to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] += 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    assert torch.all(
        recon_error2 < 1e-4
    ), f"Reconstruction error too high: {recon_error2}"

    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    envs.close()


def test_icy_transport_sc12_full_to_relative_consistency():
    """Test IcyTransport environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "icy_infront_of_transport1": True,
        "icy_infront_of_transport2": False,
        "icy_infront_of_target": False,
        "muddy_infront_of_transport1": False,
        "muddy_infront_of_transport2": True,
        "muddy_infront_of_target": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/transporting-test-sc1-rel")

    obs, _ = envs.reset(seed=0)
    relative_obs_1 = tamp_system.full_state_to_relative_state(obs, "transport_obj1")
    relative_obs_2 = tamp_system.full_state_to_relative_state(obs, "transport_obj2")
    relative_obs = torch.cat(
        [
            relative_obs_1[:, 0:1, :],
            relative_obs_2[:, 1:2, :],
            relative_obs_2[:, 2:3, :],
        ],
        dim=1,
    )
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"
    envs.close()


def test_icy_transport_sc12_full_to_relative_reset():
    """Test IcyTransport environment with a pure TAMP planner for scenarios 1 and 2."""

    test_config = {
        "seed": 1,
        "debug_env": False,
        "icy_infront_of_transport1": True,
        "icy_infront_of_transport2": False,
        "icy_infront_of_target": False,
        "muddy_infront_of_transport1": False,
        "muddy_infront_of_transport2": True,
        "muddy_infront_of_target": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2",
        "specified_task_path": "config/specified_tasks/icy_transport",
        "control_mode": "force_torque",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(
        tamp_system.env,
        "videos/transporting-test-sc12-rel-reset",
        episode_trigger=lambda episode_id: True,
    )

    # Episode 1: Initial reset with random actions
    obs, _ = envs.reset(seed=0)
    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    relative_obs_1 = tamp_system.full_state_to_relative_state(obs, "transport_obj1")
    relative_obs_2 = tamp_system.full_state_to_relative_state(obs, "transport_obj2")
    relative_obs = torch.cat(
        [
            relative_obs_1[:, 0:1, :],
            relative_obs_2[:, 1:2, :],
            relative_obs_2[:, 2:3, :],
        ],
        dim=1,
    )
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Episode 2: Reset to valid state 1 with random actions
    reset_obs, _ = envs.reset(options={"init_state": reconstructed_full_obs})
    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    # Check reconstruction error
    recon_error = torch.norm(obs - reset_obs, dim=1)
    assert torch.all(
        recon_error < 1e-4
    ), f"Reconstruction error too high: {recon_error}"

    # Episode 3: Move the obstruction to base block, should also be valid
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] += 1
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    assert torch.all(
        recon_error2 < 1e-4
    ), f"Reconstruction error too high: {recon_error2}"

    for _ in range(10):
        action = envs.action_space.sample()
        _, _, terminated, truncated, _ = envs.step(action)
        if terminated.any() or truncated.any():
            break

    envs.close()
