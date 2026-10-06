"""Unit Tests for the Lifelong Refactoring Approach, in Cluttered Table environment."""

import logging
from pathlib import Path
from typing import List

# import imageio.v2 as iio
import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanningFailure,
)


@pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [0])
def test_loading_learned_skill_predicate_blocked_stacking_sc1_or_sc2_or_sc3(
    seeed,
) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "1"  # "1" or "2"
    seed = seeed
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": sc,
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "pred_net_save_dir": f"skill_1127_sc{sc}_pred_nets_seed{seed}",
        "pre_trained_policy_path": f"trained_policies/runs/skill_1127_sc1_seed{seed}/best_ppo_ckpt.pt",
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_1209_sc{sc}_eva_seed{seed}.log",
        "max_env_steps": 250,
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
    world_setting = scenario_info.get("planner_learning_cfg_settings", {})
    update_config(world_setting)

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    # Update learning info before domain knowledge update
    approach.update_learning_info(
        int(CFG.scenario),
        scenario_info,
    )

    # Use the new update_domain_knowledge method (creates and loads policy internally)
    approach.update_domain_knowledge(scenario_info)

    # Now test the approach in new situations - evaluate both configurations
    eval_configs = [
        {
            "name": "1_b",
            "blocking_grasp": False,
            "blocking_stacking": True,
        },
        {
            "name": "1_g",
            "blocking_grasp": True,
            "blocking_stacking": False,
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
            f"obstruction{sc}_blocking_grasp": eval_config["blocking_grasp"],
            f"obstruction{sc}_blocking_stacking": eval_config["blocking_stacking"],
        }
        update_config(test_config)
        tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        # Use Training states if necessary
        # planner_dataset = PlannerDataset.load(
        #     Path(world_setting["planner_dataset_path"]),
        #     num_traj=CFG.num_eval_episodes,
        # )
        video_folder = Path(f"videos/skill_pred_1127_sc{sc}_seed{seed}_eva_{eval_name}")
        envs = MultiEnvRecordVideo(
            tamp_system.env,
            video_folder=video_folder.as_posix(),
            episode_trigger=lambda _: True,
        )
        # envs = tamp_system.env
        success = []
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
                [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
            )
            epi_success = torch.zeros(
                CFG.num_envs, dtype=torch.bool, device=envs.device
            )
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
                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")
        logging.info(f"\n{'='*80}")
        logging.info(
            f"Configuration {eval_name} - Success rate: {sum(success) / len(success)}"
        )
        logging.info(f"{'='*80}\n")
        envs.close()  # type: ignore[no-untyped-call]


@pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [0])
def test_loading_learned_skill_predicate_blocked_stacking_sc12_2(seeed) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "12_2"  # "1" or "2"
    seed = seeed
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}_seed{seed}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_1207_sc{sc}_eva_seed{seed}_continued.log",
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

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
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
        approach.update_domain_knowledge(scenario_info)

    # Now test the approach in new situations - evaluate all 4 configurations
    eval_configs = [
        # {
        #     "name": "1_b_2_b",
        #     "scenario": "1,2",
        #     "obstruction1_blocking_grasp": False,
        #     "obstruction1_blocking_stacking": True,
        #     "obstruction2_blocking_grasp": False,
        #     "obstruction2_blocking_stacking": True,
        # },
        {
            "name": "1_b_2_g",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
        },
        {
            "name": "1_g_2_b",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
        },
        # {
        #     "name": "1_g_2_g",
        #     "scenario": "1,2",
        #     "obstruction1_blocking_grasp": True,
        #     "obstruction1_blocking_stacking": False,
        #     "obstruction2_blocking_grasp": True,
        #     "obstruction2_blocking_stacking": False,
        # },
        # {
        #     "name": "1_g",
        #     "scenario": "1",
        #     "obstruction1_blocking_grasp": True,
        #     "obstruction1_blocking_stacking": False,
        #     "obstruction2_blocking_grasp": False,
        #     "obstruction2_blocking_stacking": True,
        # },
        # {
        #     "name": "1_b",
        #     "scenario": "1",
        #     "obstruction1_blocking_grasp": False,
        #     "obstruction1_blocking_stacking": True,
        #     "obstruction2_blocking_grasp": False,
        #     "obstruction2_blocking_stacking": True,
        # },
        # {
        #     "name": "2_g",
        #     "scenario": "2",
        #     "obstruction1_blocking_grasp": True,
        #     "obstruction1_blocking_stacking": False,
        #     "obstruction2_blocking_grasp": True,
        #     "obstruction2_blocking_stacking": False,
        # },
        # {
        #     "name": "2_b",
        #     "scenario": "2",
        #     "obstruction1_blocking_grasp": False,
        #     "obstruction1_blocking_stacking": True,
        #     "obstruction2_blocking_grasp": False,
        #     "obstruction2_blocking_stacking": True,
        # },
    ]

    for eval_config in eval_configs:
        eval_name = eval_config["name"]
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting evaluation for configuration: {eval_name}")
        logging.info(f"{'='*80}\n")

        test_config = {
            "num_envs": 1,
            "scenario": eval_config["scenario"],
            "num_eval_episodes": 50,
            "obstruction1_blocking_grasp": eval_config["obstruction1_blocking_grasp"],
            "obstruction1_blocking_stacking": eval_config[
                "obstruction1_blocking_stacking"
            ],
            "obstruction2_blocking_grasp": eval_config["obstruction2_blocking_grasp"],
            "obstruction2_blocking_stacking": eval_config[
                "obstruction2_blocking_stacking"
            ],
            "max_env_steps": 350,
        }
        update_config(test_config)
        tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        video_folder = Path(
            f"videos/{tamp_system.name}_{approach.get_name()}_1204sc{sc}_seed{CFG.seed}_{eval_name}"
        )
        envs = MultiEnvRecordVideo(
            tamp_system.env,
            video_folder=video_folder.as_posix(),
            episode_trigger=lambda _: True,
        )
        success = []
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
                [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
            )
            epi_success = torch.zeros(
                CFG.num_envs, dtype=torch.bool, device=envs.device
            )
            for step in range(CFG.max_env_steps + 1):
                obs, _, _, _, info = envs.step(step_result.action)
                bool_suceess = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_suceess
                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    break
                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")
        logging.info(f"\n{'='*80}")
        logging.info(
            f"Configuration {eval_name} - Success rate: {sum(success) / len(success)}"
        )
        logging.info(f"{'='*80}\n")
        envs.close()  # type: ignore[no-untyped-call]


@pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [0])
def test_loading_learned_skill_predicate_blocked_stacking_sc123_3(seeed) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "123_3"  # "1" or "2"
    seed = seeed
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}_seed{seed}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_1209_sc{sc}_eva_seed{seed}.log",
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

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
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
        approach.update_domain_knowledge(scenario_info)

    # Now test the approach in new situations - evaluate all 4 configurations
    eval_configs = [
        # 1/3 obstructions blocking each
        {
            "name": "1_g_2_g_3_b",
            "scenario": "1,2,3",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
        },
        {
            "name": "1_b_2_b_3_g",
            "scenario": "1,2,3",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
        },
        {
            "name": "1_g",
            "scenario": "1",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
        },
        {
            "name": "1_b",
            "scenario": "1",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
        },
        {
            "name": "2_g",
            "scenario": "2",
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
        },
        {
            "name": "2_b",
            "scenario": "2",
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
        },
        {
            "name": "3_g",
            "scenario": "3",
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
        },
        {
            "name": "3_b",
            "scenario": "3",
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
        },
        # 2 of 3 obstructions blocking each
        {
            "name": "1_b_2_b",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
        },
        {
            "name": "1_b_2_g",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
        },
        {
            "name": "1_g_2_b",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
        },
        {
            "name": "1_g_2_g",
            "scenario": "1,2",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
        },
        {
            "name": "1_b_3_g",
            "scenario": "1,3",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
        },
        {
            "name": "1_g_3_b",
            "scenario": "1,3",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
        },
        {
            "name": "2_b_3_g",
            "scenario": "2,3",
            "obstruction2_blocking_grasp": False,
            "obstruction2_blocking_stacking": True,
            "obstruction3_blocking_grasp": True,
            "obstruction3_blocking_stacking": False,
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
        },
        {
            "name": "2_g_3_b",
            "scenario": "2,3",
            "obstruction2_blocking_grasp": True,
            "obstruction2_blocking_stacking": False,
            "obstruction3_blocking_grasp": False,
            "obstruction3_blocking_stacking": True,
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
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
            "num_eval_episodes": 50,
            "obstruction1_blocking_grasp": eval_config["obstruction1_blocking_grasp"],
            "obstruction1_blocking_stacking": eval_config[
                "obstruction1_blocking_stacking"
            ],
            "obstruction2_blocking_grasp": eval_config["obstruction2_blocking_grasp"],
            "obstruction2_blocking_stacking": eval_config[
                "obstruction2_blocking_stacking"
            ],
            "obstruction3_blocking_grasp": eval_config["obstruction3_blocking_grasp"],
            "obstruction3_blocking_stacking": eval_config[
                "obstruction3_blocking_stacking"
            ],
            "max_env_steps": 500,
        }
        update_config(test_config)
        print("ok here")
        tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        print("ok here")
        video_folder = Path(
            f"videos/{tamp_system.name}_{approach.get_name()}_1206sc{sc}_seed{CFG.seed}_{eval_name}"
        )
        envs = MultiEnvRecordVideo(
            tamp_system.env,
            video_folder=video_folder.as_posix(),
            episode_trigger=lambda _: True,
        )
        success = []
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
                [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
            )
            epi_success = torch.zeros(
                CFG.num_envs, dtype=torch.bool, device=envs.device
            )
            for step in range(CFG.max_env_steps + 1):
                obs, _, _, _, info = envs.step(step_result.action)
                bool_suceess = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_suceess
                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    break
                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")
        logging.info(f"\n{'='*80}")
        logging.info(
            f"Configuration {eval_name} - Success rate: {sum(success) / len(success)}"
        )
        logging.info(f"{'='*80}\n")
        envs.close()  # type: ignore[no-untyped-call]
