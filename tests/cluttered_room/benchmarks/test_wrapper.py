"""Tests for PlanningStatesVectorEnv with a TAMP system."""

import logging
from pathlib import Path
from typing import List

import gymnasium as gym
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.rl_policies.manual import ManualPolicy
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import RLDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


def test_rl_planning_wrapper_cluttered_room_sc1_rnd_actions() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    sc = 1
    data_path = Path(f"training_data/cluttered_room/RL_data/scenario_{sc}")

    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "lll_config": f"config/lifelong_learning/cluttered_room_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
    }
    reset_config(test_config)
    register_all_environments()

    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][sc]
    # Update world with skill learning cfg
    cfg_settings_sc1_or_2 = scenario_info.get("skill_learning_cfg_settings", {})
    update_config(cfg_settings_sc1_or_2)
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

    pure_rl_env_train = tamp_system.env
    eval_output_dir = Path(f"videos/c-room-rnd-actions")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    pure_rl_env_eval = gym.make(
        tamp_system.env_name,
        num_envs=CFG.num_eval_envs,
        reconfiguration_freq=None,
        human_render_camera_configs={"shader_pack": "default"},
        **tamp_system.env_kwargs,
    )
    eval_envs = ManiSkillsRecordVideo(
        pure_rl_env_eval,
        output_dir=eval_output_dir,
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=CFG.max_env_steps,
        video_fps=30,
    )

    envs_mani = PlanningStatesVectorEnv(
        pure_rl_env_train,
        tamp_system,
        scenario_info,
        planner,
        CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )
    eval_envs_mani = PlanningStatesVectorEnv(
        eval_envs,
        tamp_system,
        scenario_info,
        planner,
        num_envs=CFG.num_eval_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    train_data = RLDataset.load(data_path, num_states=64)
    envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    eval_envs_mani.configure_training(train_data)  # pylint: disable=protected-access

    # try to generate videos with random actions
    _, _ = eval_envs_mani.reset()
    for _ in range(30):
        random_actions = eval_envs_mani.action_space.sample()
        _, _, _, _, _ = eval_envs_mani.step(random_actions)

    envs_mani.close()  # type: ignore[no-untyped-call]


def test_rl_planning_wrapper_cluttered_room_sc1() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    sc = 1
    data_path = Path(f"training_data/cluttered_room/RL_data/scenario_{sc}")

    # Get all RL config files
    rl_config_files = sorted(Path("config/pure_rl").glob("cluttered_room_*.yaml"))

    for rl_config_file in rl_config_files:
        # Extract config name from file path (without .yaml extension)
        config_name = rl_config_file.stem

        test_config = {
            "debug_env": False,
            "delta_finger_control": False,
            "lll_config": f"config/lifelong_learning/cluttered_room_sc{sc}.yaml",
            "log_file": f"logs/c_learning_0116_n8_{config_name}_intrinsic.log",
            "exp_name": f"c_learning_0116_n8_{config_name}_intrinsic",
            "control_mode": "pd_joint_delta_pos",
            "rl_config": str(rl_config_file),
        }
        reset_config(test_config)
        register_all_environments()
        # Set up logging
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if CFG.log_file:
            handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
        logging.basicConfig(
            level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
        )
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        if CFG.log_file:
            logging.info(f"Logging to {CFG.log_file}")

        logging.info(f"Training with config: {config_name}")

        with open(CFG.lll_config, "r", encoding="utf-8") as f:
            scenario_info = yaml.safe_load(f)["scenarios"][sc]
        # Update world with skill learning cfg
        cfg_settings_sc1_or_2 = scenario_info.get("skill_learning_cfg_settings", {})
        update_config(cfg_settings_sc1_or_2)
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

        policy = PPOCPolicy(seed=CFG.seed, rl_config=CFG.rl_config)

        pure_rl_env_train = tamp_system.env
        eval_output_dir = Path(f"videos/{tamp_system.name}_eval/{CFG.exp_name}")
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        pure_rl_env_eval = gym.make(
            tamp_system.env_name,
            num_envs=CFG.num_eval_envs,
            reconfiguration_freq=None,
            human_render_camera_configs={"shader_pack": "default"},
            **tamp_system.env_kwargs,
        )
        eval_envs = ManiSkillsRecordVideo(
            pure_rl_env_eval,
            output_dir=eval_output_dir,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=CFG.max_env_steps,
            video_fps=30,
        )

        envs_mani = PlanningStatesVectorEnv(
            pure_rl_env_train,
            tamp_system,
            scenario_info,
            planner,
            CFG.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        eval_envs_mani = PlanningStatesVectorEnv(
            eval_envs,
            tamp_system,
            scenario_info,
            planner,
            num_envs=CFG.num_eval_envs,
            ignore_terminations=True,
            record_metrics=True,
        )

        policy.initialize(envs_mani)
        train_data = RLDataset.load(data_path)
        envs_mani.configure_training(train_data)  # pylint: disable=protected-access
        eval_envs_mani.configure_training(
            train_data
        )  # pylint: disable=protected-access
        pre_trained_policy_path = Path(
            "trained_policies/c_learning_0107_cluttered_room_ppoc_sc1_intrinsic/best_ppo_ckpt.pt"
        )
        policy.load(pre_trained_policy_path)
        try:
            policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
        except ValueError as e:
            logging.error(f"Training failed for config {config_name}: {e}")
            continue
