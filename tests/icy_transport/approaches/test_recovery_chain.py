"""Unit Tests for the Recovery Chain Approach, in Icy Transport environment."""

import logging
from pathlib import Path
from typing import Any, List

import pytest
import torch
import yaml
from gymnasium import Env

from skill_refactor import register_all_environments
from skill_refactor.approaches.recovery_chain import RecoveryChainApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanningFailure,
)


# @pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recovery_chain_icy_transport_sc1(seed: int) -> None:
    """Test Recovery Chain Approach with IcyTransport environment (scenario 1).

    This test evaluates the reactive recovery baseline that:
    - Detects failures during execution (e.g., collisions)
    - Triggers learned recovery skills
    - Does NOT update operators/predicates proactively
    """
    sc = "1"
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": sc,
        "lll_config": f"config/recovery_chaining/icy_transport_sc{sc}.yaml",
        "control_mode": "force_torque",
        "force_skip_pred_learning": True,
        "pre_trained_policy_path": f"trained_policies/icy_transport/skill_0101_i_transport_sc1_seed{seed}/best_ppo_ckpt.pt",
        "failured_det_nn_path": f"skill_0101_i_transport_sc1_pred_nets_seed{seed}/terminal_gotopickobject_icydrive_0_model.pth",
        "loglevel": logging.INFO,
        "log_file": f"logs/recovery_chain_sc{sc}_eva_seed{seed}.log",
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

    with open(CFG.lll_config, "rb") as f:
        lll_config_data = yaml.safe_load(f)
    scenario_info = lll_config_data["scenarios"][int(CFG.scenario)]
    world_setting = scenario_info.get("failure_learning_cfg_settings", {})
    update_config(world_setting)

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

    # Now test the approach in new situations - evaluate both configurations
    eval_configs = [
        {
            "name": "1_g1",
            "icy_infront_of_transport1": True,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": False,
        },
        {
            "name": "1_b",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": True,
        },
        {
            "name": "1_g2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": True,
            "icy_infront_of_target": False,
        },
    ]

    for eval_config in eval_configs:
        eval_name = eval_config["name"]
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting evaluation for configuration: {eval_name}")
        logging.info(f"{'='*80}\n")

        test_config = {
            "num_envs": 1,
            "num_eval_episodes": 50,
            f"icy_infront_of_transport1": eval_config["icy_infront_of_transport1"],
            f"icy_infront_of_transport2": eval_config["icy_infront_of_transport2"],
            f"icy_infront_of_target": eval_config["icy_infront_of_target"],
        }
        update_config(test_config)
        tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        envs: MultiEnvRecordVideo | Env[Any, Any]
        if seed == 0:
            video_folder = Path(
                f"videos/recovery_chain_sc{sc}_seed{seed}_eva_{eval_name}"
            )
            envs = MultiEnvRecordVideo(
                tamp_system.env,
                video_folder=video_folder.as_posix(),
                episode_trigger=lambda _: True,
            )
        else:
            # This env rendering is too slow for multiple seeds, skip video recording
            envs = tamp_system.env
        success = []
        recovery_triggered_count = 0
        rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))
        for epi in range(0, CFG.num_eval_episodes):
            reset_options: dict = {}
            obs, info = envs.reset(
                seed=rnd_seed[epi] + seed, options=reset_options
            )  # type: ignore[no-untyped-call]
            try:
                step_result = approach.reset(obs, info)
            except TaskThenMotionPlanningFailure as e:
                logging.info(
                    f"Episode {epi} failed during reset with TaskThenMotionPlanningFailure: {e}"
                )
                success.append(False)
                continue
            total_reward = torch.tensor(
                [0.0] * CFG.num_envs, dtype=torch.float32, device=CFG.device
            )
            epi_success = torch.zeros(CFG.num_envs, dtype=torch.bool, device=CFG.device)
            episode_triggered_recovery = False
            for step in range(CFG.max_env_steps + 1):
                stepping_action = step_result.action.to(torch.float64)
                obs, _, _, _, info = envs.step(stepping_action)
                bool_suceess = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_suceess
                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    break
                if approach.exhausted.all():
                    logging.info(f"Episode {epi} all exhausted skills at step {step}.")
                    break

                # Track if recovery mode was triggered in this episode
                if approach._in_recovery_mode.any():
                    episode_triggered_recovery = True

                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]

            if episode_triggered_recovery:
                recovery_triggered_count += 1

            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")

        logging.info(f"\n{'='*80}")
        logging.info(
            f"Configuration {eval_name} - Success rate: {sum(success) / len(success)}"
        )
        logging.info(
            f"Configuration {eval_name} - Recovery triggered: {recovery_triggered_count}/{CFG.num_eval_episodes} episodes"
        )
        logging.info(f"{'='*80}\n")
        envs.close()  # type: ignore[no-untyped-call]


@pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seed", [2])
def test_recovery_chain_icy_transport_sc12_2(seed: int) -> None:
    """Test Recovery Chain Approach with IcyTransport environment (scenario 1->2).

    This test evaluates the recovery chain approach across multiple learning phases,
    where each phase may introduce a new recovery skill.
    """
    sc = "12_2"
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "lll_config": f"config/recovery_chaining/icy_transport_sc{sc}_seed{seed}.yaml",
        "control_mode": "force_torque",
        "force_skip_pred_learning": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/recovery_chain_sc{sc}_eva_seed{seed}.log",
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

    with open(CFG.lll_config, "rb") as f:
        lll_config_data = yaml.safe_load(f)

    # Load initial config from first scenario to get failured_det_operator_objects
    first_scenario_id = list(lll_config_data["scenarios"].keys())[0]
    first_scenario_info = lll_config_data["scenarios"][first_scenario_id]
    initial_cfg = first_scenario_info.get("failure_learning_cfg_settings", {})
    update_config(initial_cfg)

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = RecoveryChainApproach(tamp_system, seed=CFG.seed)

    # Iterate through all scenarios to build up recovery skills
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update CFG with planner_learning_cfg_settings
        cfg_settings_after_sc = scenario_info.get("failure_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)

        latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        approach.update_domain_knowledge(scenario_info)

    # Now test the approach in new situations - evaluate one configuration for testing
    eval_configs = [
        # 1 region
        {
            "name": "1_g1",
            "scenario": "1",
            "icy_infront_of_transport1": True,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "1_b",
            "scenario": "1",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": True,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "1_g2",
            "scenario": "1",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": True,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "2_g1",
            "scenario": "2",
            "icy_infront_of_transport1": True,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "2_b",
            "scenario": "2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": True,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": True,
        },
        {
            "name": "2_g2",
            "scenario": "2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": True,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": True,
            "muddy_infront_of_target": False,
        },
        # # 2 regions
        {
            "name": "1_g1_2_b",
            "scenario": "1,2",
            "icy_infront_of_transport1": True,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": True,
        },
        {
            "name": "1_b_2_g1",
            "scenario": "1,2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": True,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "1_b_2_g2",
            "scenario": "1,2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": True,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": True,
            "muddy_infront_of_target": False,
        },
        {
            "name": "1_g2_2_g1",
            "scenario": "1,2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": True,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": True,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": False,
        },
        {
            "name": "1_g2_2_b",
            "scenario": "1,2",
            "icy_infront_of_transport1": False,
            "icy_infront_of_transport2": True,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": False,
            "muddy_infront_of_target": True,
        },
        {
            "name": "1_g1_2_g2",
            "scenario": "1,2",
            "icy_infront_of_transport1": True,
            "icy_infront_of_transport2": False,
            "icy_infront_of_target": False,
            "muddy_infront_of_transport1": False,
            "muddy_infront_of_transport2": True,
            "muddy_infront_of_target": False,
        },
    ]

    for eval_config in eval_configs:
        eval_name = eval_config["name"]
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting evaluation for configuration: {eval_name}")
        logging.info(f"{'='*80}\n")

        test_config = {
            "num_envs": 1,
            "scenario": eval_config["scenario"],
            "num_eval_episodes": 50,  # Reduced from 50 for faster testing
            "icy_infront_of_transport1": eval_config["icy_infront_of_transport1"],
            "icy_infront_of_transport2": eval_config["icy_infront_of_transport2"],
            "icy_infront_of_target": eval_config["icy_infront_of_target"],
            "muddy_infront_of_transport1": eval_config["muddy_infront_of_transport1"],
            "muddy_infront_of_transport2": eval_config["muddy_infront_of_transport2"],
            "muddy_infront_of_target": eval_config["muddy_infront_of_target"],
            "max_env_steps": 5000,  # Reduced from 5000 for faster testing
        }
        update_config(test_config)
        tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        video_folder = Path(
            f"videos/{tamp_system.name}_{approach.get_name()}_sc{sc}_seed{CFG.seed}_{eval_name}"
        )
        envs: MultiEnvRecordVideo | Env[Any, Any]
        if seed == 0:
            envs = MultiEnvRecordVideo(
                tamp_system.env,
                video_folder=video_folder.as_posix(),
                episode_trigger=lambda _: True,
            )
        else:
            # This env rendering is too slow for multiple seeds, skip video recording
            envs = tamp_system.env
        success = []
        recovery_triggered_count = 0
        rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))
        for epi in range(0, CFG.num_eval_episodes):
            reset_options: dict = {}
            obs, info = envs.reset(
                options=reset_options, seed=rnd_seed[epi] + seed
            )  # type: ignore[no-untyped-call]
            try:
                step_result = approach.reset(obs, info)
            except TaskThenMotionPlanningFailure as e:
                logging.info(
                    f"Episode {epi} failed during reset with TaskThenMotionPlanningFailure: {e}"
                )
                success.append(False)
                continue
            total_reward = torch.tensor(
                [0.0] * CFG.num_envs, dtype=torch.float32, device=CFG.device
            )
            epi_success = torch.zeros(CFG.num_envs, dtype=torch.bool, device=CFG.device)
            episode_triggered_recovery = False
            for step in range(CFG.max_env_steps + 1):
                obs, _, _, _, info = envs.step(step_result.action)
                bool_suceess = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_suceess
                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    break
                if approach.exhausted.all():
                    logging.info(f"Episode {epi} all exhausted skills at step {step}.")
                    break

                # Track if recovery mode was triggered in this episode
                if approach._in_recovery_mode.any():
                    episode_triggered_recovery = True

                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]

            if episode_triggered_recovery:
                recovery_triggered_count += 1

            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")

        logging.info(f"\n{'='*80}")
        logging.info(
            f"Configuration {eval_name} - Success rate: {sum(success) / len(success)}"
        )
        logging.info(
            f"Configuration {eval_name} - Recovery triggered: {recovery_triggered_count}/{CFG.num_eval_episodes} episodes"
        )
        logging.info(f"{'='*80}\n")
        envs.close()  # type: ignore[no-untyped-call]
