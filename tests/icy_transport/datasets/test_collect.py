"""Unit Tests for the data collection, in IcyTransport environment."""

import glob
import logging
import pickle
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref_icy import LifelongRefIcyApproach
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import Task
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


# @pytest.mark.skip(reason="The script takes too long")
def test_icy_transport_rl_data_collection_sc1() -> None:
    """Test Data collection in IcyTransport environment."""
    sc = 1
    seed = 0
    test_config = {
        "seed": seed,
        "debug_env": False,
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
        "control_mode": "force_torque",
        "normalize_action": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/icy_transport_1231_sc{sc}_rl_datac.log",
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

    # sc1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][sc]

    # Update world with skill learning cfg
    update_config(scenario_info.get("skill_learning_cfg_settings", {}))
    task_files = glob.glob(f"{CFG.specified_task_path}/sc{sc}_task_seed{seed}_*.pkl")
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefIcyApproach(tamp_system, seed=CFG.seed)
    approach.update_learning_info(
        sc,
        scenario_info,
    )
    train_data = approach.collect_rl_data(tamp_system.env, init_states)
    # Check that collision status is 0 (no collisions)
    assert torch.all(torch.stack(train_data.states, dim=0)[:, -1] == 0)
    train_data.save(
        Path(f"training_data/icy_transport/RL_data/scenario_{CFG.scenario}/seed_{seed}")
    )


# @pytest.mark.skip(reason="The script takes too long")
def test_icy_transport_rl_data_collection_sc12_2() -> None:
    """Test Data collection in IcyTransport environment for scenario 12_2."""
    sc = "12_2"
    seed = 0
    test_config = {
        "seed": seed,
        "debug_env": False,
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}_seed{seed}.yaml",
        "control_mode": "force_torque",
        "normalize_action": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/icy_transport_0101_sc{sc}_rl_datac_seed{seed}.log",
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

    # Start by updating the system to after scenario 1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_infos = yaml.safe_load(f)["scenarios"]

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefIcyApproach(tamp_system, seed=CFG.seed)

    sc_ids = []
    for scenario_id, scenario_info in scenario_infos.items():
        sc_ids.append(int(scenario_id))
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        if scenario_info.get("trained"):
            approach.update_domain_knowledge(scenario_info)

    # Now, collect data for scenario 2
    second_sc = int(sc.rsplit("_", maxsplit=1)[-1])
    scenario_info2 = scenario_infos[second_sc]
    # Update CFG with skill_learning_cfg_settings
    cfg_settings_after_sc2 = scenario_info2.get("skill_learning_cfg_settings", {})
    update_config(cfg_settings_after_sc2)
    latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach.update_learning_info(
        second_sc,
        scenario_info2,
        latest_tamp_system=latest_tamp_system,
    )
    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{CFG.scenario}_task_seed{seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    # Re-create tamp-system since we have different settings now
    # envs = latest_tamp_system.env
    envs = MultiEnvRecordVideo(
        latest_tamp_system.env,
        "videos/icy_transport-rl-c-sc12",
        episode_trigger=lambda x: True,
    )
    train_data = approach.collect_rl_data(envs, init_states)
    assert torch.all(torch.stack(train_data.states, dim=0)[:, -1] == 0)
    train_data.save(
        Path(f"training_data/icy_transport/RL_data/scenario_{CFG.scenario}/seed_{seed}")
    )


# @pytest.mark.skip(reason="The script takes too long")
def test_icy_transport_planner_data_collection_sc1() -> None:
    """Test planner data collection in IcyTransport environment."""
    sc = "1"
    seed = 0
    test_config = {
        "seed": seed,
        "log_file": f"logs/icy_transport_0101_planner_collect_sc{sc}.log",
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
        "control_mode": "force_torque",
        "normalize_action": True,
        "debug_env": False,
        "loglevel": logging.INFO,
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
    # sc1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][int(sc)]

    cfg_settings = scenario_info.get("planner_learning_cfg_settings", {})
    update_config(cfg_settings)
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefIcyApproach(tamp_system, seed=CFG.seed)
    approach.update_learning_info(
        int(sc),
        scenario_info,
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

    policy = PPOCPolicy(seed=CFG.seed, rl_config=cfg_settings["rl_config"])

    envs = tamp_system.env
    # envs = MultiEnvRecordVideo(
    #     tamp_system.env,
    #     "videos/icy_transport-planner-dreaming",
    #     episode_trigger=lambda x: True,
    # )

    envs_mani = PlanningStatesVectorEnv(
        envs,
        tamp_system,
        scenario_info,
        planner,
        num_envs=CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    policy.initialize(envs_mani)
    assert "pre_trained_policy_path" in cfg_settings
    pre_trained_policy_path = Path(cfg_settings["pre_trained_policy_path"])
    policy.load(pre_trained_policy_path)

    # Load initial states from task files
    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{sc}_task_seed{CFG.seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    train_data = approach.collect_planner_data(
        envs, policy, init_states, real_env_scenario_name=f"sc{sc}_pre_n1"
    )
    envs.close()  # type: ignore[no-untyped-call]
    train_data.save(Path(cfg_settings["planner_dataset_path"]))


# @pytest.mark.skip(reason="The script takes too long")
def test_icy_transport_planner_data_collection_sc12_2() -> None:
    """Test planner data collection in IcyTransport environment for scenario 12_2."""
    sc = "12_2"
    seed = 0
    test_config = {
        "seed": seed,
        "debug_env": False,
        "log_file": f"logs/0102_icy_transport_planner_collect_sc{sc}.log",
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}_seed{seed}.yaml",
        "control_mode": "force_torque",
        "normalize_action": True,
        "loglevel": logging.INFO,
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
    # Start by updating the system to after scenario 1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_infos = yaml.safe_load(f)["scenarios"]

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    approach = LifelongRefIcyApproach(tamp_system, seed=CFG.seed)
    sc_ids = []
    for scenario_id, scenario_info in scenario_infos.items():
        sc_ids.append(int(scenario_id))
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        if scenario_info.get("trained"):
            approach.update_domain_knowledge(scenario_info)

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

    policy = PPOCPolicy(seed=CFG.seed, rl_config=cfg_settings_after_sc["rl_config"])

    envs = tamp_system.env
    # envs = MultiEnvRecordVideo(
    #     tamp_system.env,
    #     f"videos/icy_transport-planner-dreaming{sc}",
    #     episode_trigger=lambda x: True,
    # )

    envs_mani = PlanningStatesVectorEnv(
        envs,
        tamp_system,
        scenario_info,  # pylint: disable=undefined-loop-variable
        planner2,
        num_envs=CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    policy.initialize(envs_mani)
    assert "pre_trained_policy_path" in cfg_settings_after_sc
    pre_trained_policy_path = Path(cfg_settings_after_sc["pre_trained_policy_path"])
    policy.load(pre_trained_policy_path)

    # Load initial states from task files for all scenarios
    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{CFG.scenario}_task_seed{CFG.seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    real_env_scenario_name = CFG.real_scenario_name
    train_data = approach.collect_planner_data(
        envs, policy, init_states, real_env_scenario_name=real_env_scenario_name
    )
    envs.close()  # type: ignore[no-untyped-call]
    train_data.save(Path(cfg_settings_after_sc["planner_dataset_path"]))
