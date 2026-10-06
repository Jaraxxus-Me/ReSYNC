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
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
    ClutteredDrawerRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import RLDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


def test_rl_planning_wrapper_cluttered_drawer_sc1() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    sc = 1
    seed = 0
    data_path = Path(
        f"training_data/cluttered_drawer/RL_data/scenario_{sc}/seed_{seed}"
    )

    # Get all RL config files
    rl_config_files = sorted(
        Path("config/pure_rl").glob("cluttered_drawer_ppoc_sc1_*.yaml")
    )

    for rl_config_file in rl_config_files:
        config_name = rl_config_file.stem

        test_config = {
            "debug_env": False,
            "dreaming_noise_base_var": 0.0,
            "delta_finger_control": False,
            "seed": seed,
            "lll_config": f"config/lifelong_learning/cluttered_drawer_sc{sc}.yaml",
            "log_file": f"logs/c_drawer_skill_learning_1231_sc{sc}_seed{seed}_{config_name}.log",
            "exp_name": f"c_drawer_skill_learning_1231_sc{sc}_seed{seed}_{config_name}",
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

        with open(CFG.lll_config, "r", encoding="utf-8") as f:
            scenario_info = yaml.safe_load(f)["scenarios"][sc]
        # Update world with skill learning cfg
        cfg_settings_sc1_or_2 = scenario_info.get("skill_learning_cfg_settings", {})
        update_config(cfg_settings_sc1_or_2)
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
        # pre_trained_policy_path = Path(
        #     "trained_policies/runs/c_learning_1207_arm3_cluttered_drawer_ppoc_sc1_n4/best_ppo_ckpt.pt"
        # )
        # policy.load(pre_trained_policy_path)
        try:
            policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
        except ValueError as e:
            logging.error(f"Training failed for run {config_name}: {e}")
            continue


def test_rl_planning_wrapper_cluttered_drawer_sc12_2() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    data_path = Path(f"training_data/cluttered_drawer/RL_data/scenario12_2_seed0")

    # Get all RL config files
    rl_config_files = sorted(
        Path("config/pure_rl").glob("cluttered_drawer_ppoc_sc12_*.yaml")
    )

    for rl_config_file in rl_config_files:
        # Extract config name from file path (without .yaml extension)
        config_name = rl_config_file.stem

        test_config = {
            "seed": 0,
            "dreaming_noise_base_var": 0.0,
            "delta_finger_control": False,
            "lll_config": "config/lifelong_learning/cluttered_drawer_sc12_2_seed0.yaml",
            "log_file": f"logs/c_learning_0102_{config_name}.log",
            "exp_name": f"c_learning_0102_{config_name}",
            "control_mode": "pd_joint_delta_pos",
            "rl_config": str(rl_config_file),
        }
        reset_config(test_config)
        register_all_environments()
        # Set up logging
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

        with open(CFG.lll_config, "rb") as f:
            lll_config_data = yaml.safe_load(f)

        scenario_info1 = lll_config_data["scenarios"][1]
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

        # Update learning info before domain knowledge update
        approach.update_learning_info(
            1,
            scenario_info1,
        )
        world_setting = scenario_info1.get("planner_learning_cfg_settings", {})
        update_config(world_setting)

        # Use the new update_domain_knowledge method (creates and loads policy internally)
        approach.update_domain_knowledge(scenario_info1)

        # Create TAMP system
        scenario_info2 = lll_config_data["scenarios"][2]
        world_setting = scenario_info2.get("skill_learning_cfg_settings", {})
        update_config(world_setting)
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        fall_back_action = tamp_system.env.single_action_space.sample()  # type: ignore[attr-defined]
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            tamp_system.env, CFG.control_mode
        )

        # Create planner using approaches components
        planner2 = TaskThenMotionPlanner(
            types=tamp_system.types,
            predicates=approach.perceiver.predicates_container.as_set(),
            perceiver=approach.perceiver,
            operators=approach.operators,
            skills=approach.skills,
            fallback_action=fall_back_action,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            planner_id="pyperplan",
        )

        # Use current RL config
        logging.info(f"Training with config: {config_name}")
        policy2 = PPOCPolicy(seed=CFG.seed, rl_config=str(rl_config_file))
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
            scenario_info2,
            planner2,
            CFG.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        eval_envs_mani = PlanningStatesVectorEnv(
            eval_envs,
            tamp_system,
            scenario_info2,
            planner2,
            num_envs=CFG.num_eval_envs,
            ignore_terminations=True,
            record_metrics=True,
        )

        policy2.initialize(envs_mani)
        train_data = RLDataset.load(data_path)
        envs_mani.configure_training(train_data)  # pylint: disable=protected-access
        eval_envs_mani.configure_training(
            train_data
        )  # pylint: disable=protected-access
        # pre_trained_policy_path = Path(
        #     "trained_policies/runs/c_learning_1215_cluttered_drawer_ppoc_sc12_2/best_ppo_ckpt.pt"
        # )
        # policy2.load(pre_trained_policy_path)
        try:
            policy2.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
        except ValueError as e:
            logging.error(f"Training failed for config {config_name}: {e}")
            continue


def test_rl_planning_wrapper_cluttered_drawer_sc123_3() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    data_path = Path(f"training_data/cluttered_drawer/RL_data/scenario123_3/seed_0")

    # Get all RL config files
    rl_config_files = sorted(
        Path("config/pure_rl").glob("cluttered_drawer_ppoc_sc123_*.yaml")
    )

    for rl_config_file in rl_config_files:
        # Extract config name from file path (without .yaml extension)
        config_name = rl_config_file.stem

        test_config = {
            "seed": 0,
            "dreaming_noise_base_var": 0.0,
            "delta_finger_control": False,
            "scenario": "1,2,3",
            "lll_config": "config/lifelong_learning/cluttered_drawer_sc123_3_seed0.yaml",
            "log_file": f"logs/c_learning_0103_{config_name}.log",
            "exp_name": f"c_learning_0103_{config_name}",
            "control_mode": "pd_joint_delta_pos",
            "rl_config": str(rl_config_file),
        }
        reset_config(test_config)
        register_all_environments()
        # Set up logging
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

        with open(CFG.lll_config, "rb") as f:
            lll_config_data = yaml.safe_load(f)

        scenario_info1 = lll_config_data["scenarios"][1]
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

        # Update learning info before domain knowledge update
        approach.update_learning_info(
            1,
            scenario_info1,
        )
        world_setting = scenario_info1.get("planner_learning_cfg_settings", {})
        update_config(world_setting)

        # Use the new update_domain_knowledge method (creates and loads policy internally)
        approach.update_domain_knowledge(scenario_info1)

        # Update learning info before domain knowledge update
        scenario_info2 = lll_config_data["scenarios"][2]
        approach.update_learning_info(
            2,
            scenario_info2,
        )
        world_setting = scenario_info2.get("planner_learning_cfg_settings", {})
        update_config(world_setting)

        # Use the new update_domain_knowledge method (creates and loads policy internally)
        approach.update_domain_knowledge(scenario_info2)

        # Create TAMP system
        scenario_info3 = lll_config_data["scenarios"][3]
        world_setting = scenario_info3.get("skill_learning_cfg_settings", {})
        update_config(world_setting)
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        fall_back_action = tamp_system.env.single_action_space.sample()  # type: ignore[attr-defined]
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            tamp_system.env, CFG.control_mode
        )

        # Create planner using approaches components
        planner3 = TaskThenMotionPlanner(
            types=tamp_system.types,
            predicates=approach.perceiver.predicates_container.as_set(),
            perceiver=approach.perceiver,
            operators=approach.operators,
            skills=approach.skills,
            fallback_action=fall_back_action,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            planner_id="pyperplan",
        )

        # Use current RL config
        logging.info(f"Training with config: {config_name}")
        policy3 = PPOCPolicy(seed=CFG.seed, rl_config=str(rl_config_file))
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
            scenario_info3,
            planner3,
            CFG.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        eval_envs_mani = PlanningStatesVectorEnv(
            eval_envs,
            tamp_system,
            scenario_info3,
            planner3,
            num_envs=CFG.num_eval_envs,
            ignore_terminations=True,
            record_metrics=True,
        )

        policy3.initialize(envs_mani)
        train_data = RLDataset.load(data_path)
        envs_mani.configure_training(train_data)  # pylint: disable=protected-access
        eval_envs_mani.configure_training(
            train_data
        )  # pylint: disable=protected-access
        # pre_trained_policy_path = Path(
        #     "trained_policies/runs/c_learning_1215_cluttered_drawer_ppoc_sc12_2/best_ppo_ckpt.pt"
        # )
        # policy2.load(pre_trained_policy_path)
        try:
            policy3.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
        except ValueError as e:
            logging.error(f"Training failed for config {config_name}: {e}")
            continue


def test_rl_planning_wrapper_cluttered_drawer_manual_policy() -> None:
    """Test RL Planning Wrapper with ClutteredDrawer environment."""
    sc = 1
    data_path = Path(f"training_data/cluttered_drawer/RL_data/scenario_{sc}")

    # Extract config name from file path (without .yaml extension)
    test_config = {
        "debug_env": True,
        "delta_finger_control": False,
        "lll_config": f"config/lifelong_learning/cluttered_drawer_sc{sc}.yaml",
        "log_file": f"logs/c_learning_manual_1207.log",
        "exp_name": f"c_learning_manual_1207",
        "control_mode": "pd_joint_delta_pos",
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

    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][sc]
    # Update world with skill learning cfg
    cfg_settings_sc1_or_2 = scenario_info.get("skill_learning_cfg_settings", {})
    update_config(cfg_settings_sc1_or_2)
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

    policy = ManualPolicy(seed=CFG.seed)

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
    train_data = RLDataset.load(data_path, num_states=32)
    envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    eval_envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    # pre_trained_policy_path = Path(
    #     "trained_policies/runs/skill_pred_0822_s1_debug_15/ckpt_1228800.pt"
    # )
    # policy.load(pre_trained_policy_path)
    policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
