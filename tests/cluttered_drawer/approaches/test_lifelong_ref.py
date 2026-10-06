"""Unit Tests for the Lifelong Refactoring Approach, in Cluttered Table environment."""

import logging
from pathlib import Path
from typing import Any, List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
    ClutteredDrawerRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanningFailure,
)


# @pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [0, 1, 2, 3, 4])
def test_loading_learned_skill_predicate_c_drawer_sc1_or_sc2_or_sc3(
    seeed,
) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "1"  # "1" or "2"
    seed = seeed
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": sc,
        "debug_env": False,
        "delta_finger_control": False,
        "lll_config": f"config/lifelong_learning/cluttered_drawer_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "pred_net_save_dir": f"c_drawer_sc{sc}_pred_nets_seed{seed}",
        "pre_trained_policy_path": f"trained_policies/runs/skill_1231_sc1_seed0/best_ppo_ckpt.pt",
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_c_drawer_0105_sc{sc}_eva_seed{seed}.log",
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

    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
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
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
        },
        {
            "name": "1_g",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
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
            f"drawer_blocking_grasp": eval_config["drawer_blocking_grasp"],
            f"drawer_blocking_stacking": eval_config["drawer_blocking_stacking"],
        }
        update_config(test_config)
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        # Use Training states if necessary
        # planner_dataset = PlannerDataset.load(
        #     Path(world_setting["planner_dataset_path"]),
        #     num_traj=CFG.num_eval_episodes,
        # )
        video_folder = Path(
            f"videos/skill_pred_c_drawer_0105_sc{sc}_seed{seed}_eva_{eval_name}"
        )
        if seed == 0:
            envs: ManiSkillsRecordVideo | Any = ManiSkillsRecordVideo(
                tamp_system.env,
                output_dir=video_folder,
                save_trajectory=False,
                save_video=True,
                trajectory_name="trajectory",
                max_steps_per_video=CFG.max_env_steps,
                video_fps=30,
            )
        else:
            envs = tamp_system.env
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
                obs, _, _, _, info = envs.step(step_result.action)
                bool_suceess = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_suceess
                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    break
                if approach.exhausted.all():
                    logging.info(
                        f"Episode {epi} all exhausted skill attempts at step {step}."
                    )
                    break
                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            success.extend(epi_success.cpu().numpy().tolist())
        logging.info(f"Episode {epi} final success: {epi_success}.")
    logging.info(f"Success rate: {sum(success) / len(success)}")
    envs.close()  # type: ignore[no-untyped-call]


# @pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [2])
def test_loading_learned_skill_predicate_cluttered_drawer_sc12_2(seeed) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "12_2"  # "1" or "2"
    seed = seeed
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "lll_config": f"config/lifelong_learning/cluttered_drawer_sc{sc}_seed{seed}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_0107_sc{sc}_eva_seed{seed}.log",
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

    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
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
        {
            "name": "1_b_2_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
        },
        {
            "name": "1_b_2_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
        },
        {
            "name": "1_g_2_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
        },
        {
            "name": "1_g_2_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
        },
        {
            "name": "1_g",
            "scenario": "1",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
        },
        {
            "name": "1_b",
            "scenario": "1",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
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
            "drawer_blocking_grasp": eval_config["drawer_blocking_grasp"],
            "drawer_blocking_stacking": eval_config["drawer_blocking_stacking"],
            "block_blocking_grasp": eval_config["block_blocking_grasp"],
            "block_blocking_stacking": eval_config["block_blocking_stacking"],
            "max_env_steps": 800,
        }
        update_config(test_config)
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        video_folder = Path(
            f"videos/skill_pred_c_drawer_0105_sc{sc}_seed{CFG.seed}_eva_{eval_name}"
        )
        # if seed == 0:
        envs = ManiSkillsRecordVideo(
            tamp_system.env,
            output_dir=video_folder,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=CFG.max_env_steps,
            video_fps=30,
        )
        # else:
        #     envs = tamp_system.env
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


@pytest.mark.parametrize("seeed", [0])
def test_loading_learned_skill_predicate_cluttered_drawer_sc123_3(seeed) -> None:
    """Test RL Planning Wrapper with BlockedStacking environment."""
    sc = "123_3"  # "1" or "2"
    seed = seeed
    test_config = {
        "num_envs": 1,
        "seed": seed,
        "lll_config": f"config/lifelong_learning/cluttered_drawer_sc{sc}_seed{seed}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_0108_sc{sc}_eva_seed{seed}_sele02.log",
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

    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)

    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
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
        # Three
        {
            "name": "1_g_2_g_3_b",
            "scenario": "1,2,3",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
            "wall_blocking_grasp": False,
            "wall_blocking_stacking": True,
        },
        {
            "name": "1_b_2_b_3_g",
            "scenario": "1,2,3",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        # Two
        {
            "name": "1_b_2_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
            "wall_blocking_grasp": False,
            "wall_blocking_stacking": True,
        },
        {
            "name": "1_g_2_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        {
            "name": "1_g_2_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        {
            "name": "1_b_2_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        {
            "name": "1_b_3_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        {
            "name": "1_g_3_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
            "wall_blocking_grasp": False,
            "wall_blocking_stacking": True,
        },
        {
            "name": "1_g_3_g",
            "scenario": "1,2",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": True,
            "block_blocking_stacking": False,
            "wall_blocking_grasp": True,
            "wall_blocking_stacking": False,
        },
        {
            "name": "1_b_3_b",
            "scenario": "1,2",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
            "wall_blocking_grasp": False,
            "wall_blocking_stacking": True,
        },
        {
            "name": "1_g",
            "scenario": "1",
            "drawer_blocking_grasp": True,
            "drawer_blocking_stacking": False,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
        },
        {
            "name": "1_b",
            "scenario": "1",
            "drawer_blocking_grasp": False,
            "drawer_blocking_stacking": True,
            "block_blocking_grasp": False,
            "block_blocking_stacking": True,
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
            "drawer_blocking_grasp": eval_config["drawer_blocking_grasp"],
            "drawer_blocking_stacking": eval_config["drawer_blocking_stacking"],
            "block_blocking_grasp": eval_config["block_blocking_grasp"],
            "block_blocking_stacking": eval_config["block_blocking_stacking"],
            "wall_blocking_grasp": eval_config["wall_blocking_grasp"],
            "wall_blocking_stacking": eval_config["wall_blocking_stacking"],
            "max_env_steps": 1000,
        }
        update_config(test_config)
        tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        video_folder = Path(
            f"videos/{tamp_system.name}_{approach.get_name()}_0108sc{sc}_seed{CFG.seed}_{eval_name}_debug"
        )
        envs = ManiSkillsRecordVideo(
            tamp_system.env,
            output_dir=video_folder,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=CFG.max_env_steps,
            video_fps=30,
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
                if approach.exhausted.all():
                    logging.info(
                        f"Episode {epi} all exhausted skill attempts at step {step}."
                    )
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
