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
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import PlannerDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skip(reason="The script generates local data")
@pytest.mark.parametrize("seeed", [4])
def test_blocked_stacking_task_gen(seeed):
    """Test BlockedStacking environment with a pure TAMP planner."""

    sc = "1,2"
    seed = seeed
    task_save_path = f"config/specified_tasks/blocked_stacking"
    test_config = {
        "seed": seed,
        "debug_env": False,
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "obstruction2_blocking_grasp": True,
        "obstruction2_blocking_stacking": False,
        "obstruction3_blocking_grasp": True,
        "obstruction3_blocking_stacking": False,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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


def test_blocked_stacking_sc1_or_2_or_3_grasp_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    sc = 2
    test_config = {
        "debug_env": False,
        f"obstruction{sc}_blocking_grasp": True,
        f"obstruction{sc}_blocking_stacking": False,
        "num_envs": 2,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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

    envs = MultiEnvRecordVideo(tamp_system.env, f"videos/stacking-test-sc{sc}")
    # envs = tamp_system.env

    obs, info = envs.reset(seed=0)
    s = time.time()
    planner.reset(obs, info)
    print("Planner reset time:", time.time() - s)

    total_reward = 0
    for step in range(200):
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
    # assert step < 59
    envs.close()


def test_blocked_stacking_sc1_or_2_or_3_place_failure():
    """Test BlockedStacking environment with a pure TAMP planner."""

    sc = 3
    test_config = {
        "debug_env": False,
        f"obstruction{sc}_blocking_grasp": False,
        f"obstruction{sc}_blocking_stacking": True,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": f"{sc}",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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

    envs = MultiEnvRecordVideo(tamp_system.env, f"videos/stacking-test-sc{sc}")
    # envs = tamp_system.env

    obs, info = envs.reset(seed=0)
    s = time.time()
    planner.reset(obs, info)
    print("Planner reset time:", time.time() - s)

    total_reward = 0
    for step in range(150):
        action, _ = planner.step(obs)
        s = time.time()
        obs, reward, _, _, _ = envs.step(action)
        if torch.any(obs[:, -1]):
            print(f"Collide at step {step}")
            break
        # print("Step time:", time.time() - s)
        # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
        total_reward += reward
    # Should collide in after 49 steps
    assert step > 50
    envs.close()


def test_blocked_stacking_sc1_full_to_relative_consistency():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/stacking-test-sc2")

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_block")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # Check reconstruction error
    recon_error = torch.norm(obs - reconstructed_full_obs, dim=1)
    assert torch.all(
        recon_error < 1e-5
    ), f"Reconstruction error too high: {recon_error}"
    envs.close()


def test_blocked_stacking_sc1_full_to_relative_reset():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "seed": 1,
        "debug_env": False,
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1",
        "specified_task_path": "config/specified_tasks/blocked_stacking",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/stacking-test-sc2")

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_block")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # reset to valid state 1
    reset_obs, _ = envs.reset(options={"init_state": reconstructed_full_obs})

    # Check reconstruction error
    recon_error = torch.norm(obs - reset_obs, dim=1)
    assert torch.all(
        recon_error < 1e-4
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

    # Move the obstruction to below the table, should be invalid
    # will be loading from the provided task.
    new_relative_obs = relative_obs.clone()
    new_relative_obs[:, :, 0] += 1
    new_relative_obs[:, :, 2] -= 0.4
    new_full_obs = tamp_system.relative_state_to_full_state(obs, new_relative_obs)
    reset_obs2, _ = envs.reset(options={"init_state": new_full_obs})
    recon_error2 = torch.norm(new_full_obs - reset_obs2, dim=1)
    assert torch.all(
        recon_error2 > 1e-2
    ), f"Should be invalid state but got low error: {recon_error2}"
    envs.close()


def test_blocked_stacking_sc12_full_to_relative_reset():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "obstruction1_blocking_grasp": True,
        "obstruction2_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/stacking-test-sc12")

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_block")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # reset to valid state 1
    reset_obs, _ = envs.reset(options={"init_state": reconstructed_full_obs})

    # Check reconstruction error
    recon_error = torch.norm(obs - reset_obs, dim=1)
    assert torch.all(
        recon_error < 5e-4
    ), f"Reconstruction error too high: {recon_error}"
    envs.close()


def test_blocked_stacking_sc123_full_to_relative_reset():
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "debug_env": False,
        "obstruction1_blocking_grasp": False,
        "obstruction1_blocking_stacking": True,
        "obstruction2_blocking_grasp": False,
        "obstruction2_blocking_stacking": True,
        "obstruction3_blocking_grasp": True,
        "obstruction3_blocking_stacking": False,
        "num_envs": 4,
        "device": "cuda:0",
        "scenario": "1,2,3",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/stacking-test-sc12")

    obs, _ = envs.reset(seed=0)
    relative_obs = tamp_system.full_state_to_relative_state(obs, "grasp_block")
    reconstructed_full_obs = tamp_system.relative_state_to_full_state(obs, relative_obs)

    # reset to valid state 1
    reset_obs, _ = envs.reset(options={"init_state": reconstructed_full_obs})

    # Check reconstruction error
    recon_error = torch.norm(obs - reset_obs, dim=1)
    assert torch.all(
        recon_error < 5e-4
    ), f"Reconstruction error too high: {recon_error}"
    envs.close()


@pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("scenario", ["scenario1"])
@pytest.mark.parametrize("seed", [0, 2, 3])
def test_load_and_save_lifted_operator_plans(scenario: str, seed: int) -> None:
    """Test loading planner dataset and extracting lifted operator plans."""
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/lifted_plan_generation_sc12.log",
        "max_env_steps": 300,
    }
    reset_config(test_config)
    register_all_environments()

    # Set up logging
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
    logging.basicConfig(
        level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")

    # Path to the dataset
    dataset_path = Path(
        f"training_data/cluttered_room/Planner_data/{scenario}/seed_{seed}"
    )

    # Check if the dataset path exists
    if not dataset_path.exists():
        pytest.skip(f"Dataset path {dataset_path} does not exist")

    # Load planner dataset with all trajectories
    logging.info(f"\nLoading dataset from {dataset_path}...")
    dataset = PlannerDataset.load(
        dataset_path, num_traj=-1, filter_incomplete_operators=True
    )

    logging.info(
        f"Loaded {len(dataset)} trajectories (with complete operator annotations)"
    )

    # Filter to only successful trajectories
    successful_dataset = dataset.success_subset()
    logging.info(
        f"Filtered to {len(successful_dataset)} successful trajectories "
        f"({len(successful_dataset)/len(dataset)*100:.1f}% success rate)"
    )

    # Extract lifted operator plans from all successful trajectories
    lifted_plans = successful_dataset.get_all_operator_plans()

    logging.info(f"Found {len(lifted_plans)} unique lifted operator plans:")
    for i, plan in enumerate(lifted_plans):
        plan_names = [op.name for op in plan]
        logging.info(f"  Plan {i+1}: {' -> '.join(plan_names)}")

    # Save the lifted operator plans to a pickle file
    output_path = dataset_path / "lifted_operator_plans.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(lifted_plans, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Saved lifted operator plans to {output_path}")

    # Verify the file can be loaded
    with open(output_path, "rb") as f:
        loaded_plans = pickle.load(f)

    assert len(loaded_plans) == len(
        lifted_plans
    ), "Loaded plans don't match saved plans"
    logging.info(f"Successfully verified: {len(loaded_plans)} plans can be loaded")
