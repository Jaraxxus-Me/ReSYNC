"""Tests for PlanningStatesVectorEnv with a TAMP system."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, List

import gymnasium as gym
import pytest
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.rl_policies.manual_car import ManualPolicy
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
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


# @pytest.mark.skip(reason="Requires local data to run")
def test_rl_planning_wrapper_icy_transport_sc1() -> None:
    """Test RL Planning Wrapper with IcyTransport environment."""
    sc = 1
    seed = 0
    data_path = Path(f"training_data/icy_transport/RL_data/scenario_{sc}/seed_{seed}")

    # Get all RL config files
    rl_config_files = sorted(Path("config/pure_rl").glob("icy_transport_*.yaml"))

    for rl_config_file in rl_config_files:
        # Extract config name from file path (without .yaml extension)
        config_name = rl_config_file.stem

        test_config = {
            "debug_env": False,
            "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
            "log_file": f"logs/c_learning_1231_{config_name}.log",
            "exp_name": f"c_learning_1231_{config_name}",
            "rl_config": str(rl_config_file),
            "control_mode": "force_torque",
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
        tamp_system = IcyTransportRLTAMPSystem.create_default(
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

        # For PRBench envs
        # Factory function to avoid cell-var-from-loop issue
        def _create_make_env_fn(
            env_name: str, env_kwargs: dict[str, Any]
        ) -> Callable[[], gym.Env]:
            def make_env_fn() -> gym.Env:
                return gym.make(env_name, **env_kwargs)

            return make_env_fn

        make_env_fn = _create_make_env_fn(tamp_system.env_name, tamp_system.env_kwargs)

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
        eval_envs_mani.configure_training(
            train_data
        )  # pylint: disable=protected-access
        # pre_trained_policy_path = Path(
        #     "trained_policies/runs/c_learning_1228_icy_transport_ppoc_sc1_n16/best_ppo_ckpt.pt"
        # )
        # policy.load(pre_trained_policy_path)
        try:
            policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
        except ValueError as e:
            logging.error(f"Training failed for config {config_name}: {e}")
            continue


def test_rl_planning_wrapper_blocked_stacking_sc12_2() -> None:
    """Test RL Planning Wrapper with the IcyTransport environment."""
    sc = "12_2"
    seed = 0
    data_path = Path(f"training_data/icy_transport/RL_data/scenario_{sc}/seed_{seed}")

    # Get all RL config files
    rl_config_files = sorted(
        Path("config/pure_rl").glob("icy_transport_ppoc_sc12_2_*.yaml")
    )

    for rl_config_file in rl_config_files:
        config_name = rl_config_file.stem

        test_config = {
            "debug_env": False,
            "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
            "log_file": f"logs/c_learning_0101_{config_name}.log",
            "exp_name": f"c_learning_0101_{config_name}",
            "rl_config": str(rl_config_file),
            "control_mode": "force_torque",
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
        tamp_system = IcyTransportRLTAMPSystem.create_default(
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
        tamp_system = IcyTransportRLTAMPSystem.create_default(
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
        # Use a factory to capture loop variables correctly
        def _make_env_fn_factory(
            env_name: str, env_kwargs: dict[str, Any]
        ) -> Callable[[], gym.Env]:
            def make_env_fn() -> gym.Env:
                return gym.make(env_name, **env_kwargs)

            return make_env_fn

        make_env_fn = _make_env_fn_factory(
            tamp_system.env_name, dict(tamp_system.env_kwargs)
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
        eval_envs_mani.configure_training(
            train_data
        )  # pylint: disable=protected-access
        # pre_trained_policy_path = Path(
        #     "trained_policies/runs/1105_sc12_2_rl_planning/ckpt_6400.pt"
        # )
        # policy2.load(pre_trained_policy_path)
        policy2.train(envs_mani, eval_envs_mani, train_data)  # type: ignore


# @pytest.mark.skip(reason="Requires local data to run")
def test_rl_planning_wrapper_icy_transport_manual_policy() -> None:
    """Test RL Planning Wrapper with IcyTransport environment."""
    sc = "12_2"
    data_path = Path(f"training_data/icy_transport/RL_data/scenario_{sc}/seed_0")

    # Extract config name from file path (without .yaml extension)
    test_config = {
        "debug_env": False,
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}_seed0.yaml",
        "log_file": f"logs/c_learning_manual_0101.log",
        "exp_name": f"c_learning_manual_0101",
        "control_mode": "force_torque",
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
        scenario_info = yaml.safe_load(f)["scenarios"][2]
    # Update world with skill learning cfg
    cfg_settings_sc1_or_2 = scenario_info.get("skill_learning_cfg_settings", {})
    update_config(cfg_settings_sc1_or_2)
    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
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
