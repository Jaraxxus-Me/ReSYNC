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
from skill_refactor.approaches.recovery_chain import RecoveryChainApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import Task


@pytest.mark.skip(reason="The script takes too long")
@pytest.mark.parametrize("seeed", [1, 2, 3, 4])
def test_icy_transport_failure_data_collection_sc1(seeed) -> None:
    """Test Data collection in IcyTransport environment."""
    sc = 1
    seed = seeed
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": str(sc),
        "lll_config": f"config/recovery_chaining/icy_transport_sc{sc}.yaml",
        "control_mode": "force_torque",
        "force_skip_pred_learning": True,
        "pre_trained_policy_path": f"trained_policies/icy_transport/skill_0101_i_transport_sc1_seed{seed}/best_ppo_ckpt.pt",
        "loglevel": logging.INFO,
        "log_file": f"logs/icy_transport_0111_sc{sc}_datac_seed{seed}.log",
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

    # Update world with failure learning cfg
    update_config(scenario_info.get("failure_learning_cfg_settings", {}))
    task_files = glob.glob(f"{CFG.specified_task_path}/sc{sc}_task_seed{seed}_*.pkl")
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = RecoveryChainApproach(tamp_system, seed=CFG.seed)

    # Update learning info before domain knowledge update
    approach.update_learning_info(
        int(CFG.scenario),
        scenario_info,
    )

    # Load recovery policy and create recovery skill (NO predicate/operator learning)
    approach.update_domain_knowledge(scenario_info)

    train_data = approach.collect_planner_data(
        tamp_system.env,
        init_states,
        real_env_scenario_name=scenario_info.get("real_scenario_name", None),
    )
    # Check that we have both positive and negative examples
    assert len(train_data.states) > 0, "No training data collected"
    assert torch.any(train_data.labels == 0), "No negative examples collected"
    assert torch.any(train_data.labels == 1), "No positive examples collected"
    logging.info(
        f"Collected {len(train_data.states)} states: "
        f"{torch.sum(train_data.labels == 1)} positive, "
        f"{torch.sum(train_data.labels == 0)} negative"
    )
    train_data.save(
        Path(
            f"training_data/icy_transport/Failure_data/scenario_{CFG.scenario}/seed_{seed}/state_labels.pt"
        )
    )


@pytest.mark.skip(reason="The script takes too long")
def test_icy_transport_failure_data_collection_sc12_2() -> None:
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
