"""Tests for PlanningStatesVectorEnv with a TAMP system."""

import logging
from pathlib import Path
from typing import List

import gymnasium as gym
import pytest
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
    MultiEnvWrapper,
    NormalizeActionMultiEnvWrapper,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import RLDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skip(reason="Requires local data to run")
def test_rl_planning_wrapper_blocked_stacking_sc1_or_2_or_3() -> None:
    """Test RL Planning Wrapper with ClutteredTable environment."""
    sc = 1
    seed = 0
    data_path = Path(
        f"training_data/blocked_stacking/RL_data/scenario_{sc}/seed_{seed}"
    )
    test_config = {
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        "log_file": f"logs/c_learning_debug_{sc}.log",
        "exp_name": f"c_learning_debug_{sc}",
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
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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

    # envs = tamp_system.env
    policy = PPOCPolicy(seed=CFG.seed, rl_config=cfg_settings_sc1_or_2["rl_config"])

    pure_rl_env_train = tamp_system.env

    # For PRBench envs
    def make_env_fn():
        return gym.make(
            tamp_system.env_name,
            **tamp_system.env_kwargs,
        )

    if CFG.normalize_action:
        pure_rl_env_eval = NormalizeActionMultiEnvWrapper(  # type: ignore
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    else:
        pure_rl_env_eval = MultiEnvWrapper(  # type: ignore[assignment]
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    # Record
    eval_output_dir = Path(f"videos/{tamp_system.name}_eval/{CFG.exp_name}")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_envs = MultiEnvRecordVideo(
        pure_rl_env_eval,
        video_folder=eval_output_dir.as_posix(),
        episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
    )

    envs_mani = PlanningStatesVectorEnv(
        pure_rl_env_train,
        tamp_system,
        scenario_info,
        planner,
        num_envs=CFG.num_envs,
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
    eval_envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    # pre_trained_policy_path = Path(
    #     "trained_policies/runs/skill_1111_sc3/ckpt_208000.pt"
    # )
    # policy.load(pre_trained_policy_path)
    policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore


@pytest.mark.skip(reason="Requires local data to run")
def test_rl_planning_wrapper_blocked_stacking_sc12_2() -> None:
    """Test RL Planning Wrapper with the BlockedStacking environment."""
    data_path = Path("training_data/blocked_stacking/RL_data/scenario12_2")
    test_config = {
        "lll_config": "config/lifelong_learning/blocked_stacking_sc12_2.yaml",
        "log_file": "logs/1105_sc12_2_rl_planning.log",
        "exp_name": "1105_sc12_2_rl_planning",
        "control_mode": "pd_joint_delta_pos",
    }
    register_all_environments()
    reset_config(test_config)

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
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
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

    policy2 = PPOCPolicy(seed=CFG.seed, rl_config=world_setting["rl_config"])
    pure_rl_env_train = tamp_system.env

    # For PRBench envs
    def make_env_fn():
        return gym.make(
            tamp_system.env_name,
            **tamp_system.env_kwargs,
        )

    if CFG.normalize_action:
        pure_rl_env_eval = NormalizeActionMultiEnvWrapper(  # type: ignore
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    else:
        pure_rl_env_eval = MultiEnvWrapper(  # type: ignore[assignment]
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    # Record
    eval_output_dir = Path(f"videos/{tamp_system.name}_eval/{CFG.exp_name}")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_envs = MultiEnvRecordVideo(
        pure_rl_env_eval,
        video_folder=eval_output_dir.as_posix(),
        episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
    )

    envs_mani = PlanningStatesVectorEnv(
        pure_rl_env_train,
        tamp_system,
        scenario_info2,
        planner2,
        num_envs=CFG.num_envs,
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
    eval_envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    # pre_trained_policy_path = Path(
    #     "trained_policies/runs/1105_sc12_2_rl_planning/ckpt_6400.pt"
    # )
    # policy2.load(pre_trained_policy_path)
    policy2.train(envs_mani, eval_envs_mani, train_data)  # type: ignore


@pytest.mark.skip(reason="Requires local data to run")
def test_rl_planning_wrapper_blocked_stacking_sc123_3() -> None:
    """Test RL Planning Wrapper with the BlockedStacking environment."""
    sc = "123_3"
    data_path = Path(f"training_data/blocked_stacking/RL_data/scenario{sc}")
    test_config = {
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        "log_file": f"logs/1118_sc{sc}_rl_planning.log",
        "exp_name": f"1118_sc{sc}_rl_planning",
        "control_mode": "pd_joint_delta_pos",
    }
    register_all_environments()
    reset_config(test_config)

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
        scenario_infos = yaml.safe_load(f)["scenarios"]

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    sc_ids = []
    for scenario_id, scenario_info in scenario_infos.items():
        sc_ids.append(int(scenario_id))
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        if scenario_info.get("trained"):
            approach.update_domain_knowledge(scenario_info)

    # Now, rl for the latest scenario
    latest_sc = int(sc.rsplit("_", maxsplit=1)[-1])
    scenario_info_latest = scenario_infos[latest_sc]
    # Update CFG with skill_learning_cfg_settings
    cfg_settings_after_sc2 = scenario_info_latest.get("skill_learning_cfg_settings", {})
    update_config(cfg_settings_after_sc2)
    latest_tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    fall_back_action = latest_tamp_system.env.single_action_space.sample()  # type: ignore[attr-defined]
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        latest_tamp_system.env, CFG.control_mode
    )

    # Create planner using approaches components
    planner2 = TaskThenMotionPlanner(
        types=latest_tamp_system.types,
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

    policy2 = PPOCPolicy(seed=CFG.seed, rl_config=cfg_settings_after_sc2["rl_config"])
    pure_rl_env_train = latest_tamp_system.env

    # For PRBench envs
    def make_env_fn():
        return gym.make(
            latest_tamp_system.env_name,
            **latest_tamp_system.env_kwargs,
        )

    if CFG.normalize_action:
        pure_rl_env_eval = NormalizeActionMultiEnvWrapper(  # type: ignore
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    else:
        pure_rl_env_eval = MultiEnvWrapper(  # type: ignore[assignment]
            make_env_fn,
            num_envs=CFG.num_eval_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    # Record
    eval_output_dir = Path(f"videos/{latest_tamp_system.name}_eval/{CFG.exp_name}")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_envs = MultiEnvRecordVideo(
        pure_rl_env_eval,
        video_folder=eval_output_dir.as_posix(),
        episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
    )

    envs_mani = PlanningStatesVectorEnv(
        pure_rl_env_train,
        latest_tamp_system,
        scenario_info_latest,
        planner2,
        num_envs=CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )
    eval_envs_mani = PlanningStatesVectorEnv(
        eval_envs,
        latest_tamp_system,
        scenario_info_latest,
        planner2,
        num_envs=CFG.num_eval_envs,
        ignore_terminations=True,
        record_metrics=True,
    )
    policy2.initialize(envs_mani)
    train_data = RLDataset.load(data_path)
    envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    eval_envs_mani.configure_training(train_data)  # pylint: disable=protected-access
    # pre_trained_policy_path = Path(
    #     "trained_policies/runs/1105_sc12_2_rl_planning/ckpt_6400.pt"
    # )
    # policy2.load(pre_trained_policy_path)
    policy2.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
