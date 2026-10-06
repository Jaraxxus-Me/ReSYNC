"""Unit Tests for the Lifelong Refactoring Approach, in Cluttered Room environment."""

import logging
import os
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import PlannerDataset
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanningFailure,
)


# @pytest.mark.skip(reason="The script requires local data")
@pytest.mark.parametrize("seeed", [0, 2, 3])
def test_loading_learned_skill_predicate_c_room_sc1(seeed) -> None:
    """Test RL Planning Wrapper with ClutteredRoom environment."""
    sc = "1"
    seed = seeed
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": sc,
        "debug_env": False,
        "delta_finger_control": False,
        "lll_config": f"config/lifelong_learning/cluttered_room_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "force_skip_pred_learning": True,
        "clamp_xy_to_radius": 0.1,
        "pred_net_save_dir": f"trained_pred_nets_sc1_debug3_seed{seed}",
        "pre_trained_policy_path": f"trained_policies/c_learning_0116_n8_cluttered_room_ppoc_sc1_intrinsic/best_ppo_ckpt.pt",
        "loglevel": logging.INFO,
        "log_file": f"logs/skill_pred_c_room_0118_sc{sc}_eva_seed{seed}.log",
        "max_env_steps": 7000,
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

    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
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
            "name": "1_t1",
            "can_blocking_target1": True,
            "can_blocking_target2": False,
        },
        {
            "name": "1_t2",
            "can_blocking_target1": False,
            "can_blocking_target2": True,
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
            "can_blocking_target1": eval_config["can_blocking_target1"],
            "can_blocking_target2": eval_config["can_blocking_target2"],
        }
        update_config(test_config)
        tamp_system = ClutteredRoomRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        # Use Training states if necessary
        dataset_path = Path(
            os.path.join(world_setting["planner_dataset_path"], f"seed_{CFG.seed}")
        )
        _planner_dataset = PlannerDataset.load(
            dataset_path,
            num_traj=2,
        )
        _video_folder = Path(
            f"videos/skill_pred_c_room_0118_sc{sc}_seed{seed}_eva_{eval_name}"
        )
        # if seed == 0:
        #     envs = ManiSkillsRecordVideo(
        #         tamp_system.env,
        #         output_dir=video_folder,
        #         save_trajectory=False,
        #         save_video=True,
        #         trajectory_name="trajectory",
        #         max_steps_per_video=CFG.max_env_steps,
        #         video_fps=30,
        #     )
        # else:
        envs = tamp_system.env
        success = []
        rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))
        for epi in range(0, CFG.num_eval_episodes):
            # train_traj = planner_dataset.trajectories[epi]
            # reset_options = {
            #     "init_state": train_traj.states[0].unsqueeze(0),
            # }
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
                [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device  # type: ignore[attr-defined]
            )
            epi_success = torch.zeros(
                CFG.num_envs, dtype=torch.bool, device=envs.device  # type: ignore[attr-defined]
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
                    logging.info(f"Episode {epi} all exhausted early at step {step}.")
                    break
                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")
        logging.info(f"Success rate: {sum(success) / len(success)}")
    envs.close()  # type: ignore[no-untyped-call]
