"""Unit Tests for the data collection, in Cluttered Room environment."""

import glob
import logging
import os
import pickle
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref_room import LifelongRefRoomApproach
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
    PlanningStatesVectorRoomEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import Task
from skill_refactor.utils.ttmp import TaskThenMotionPlanner

# Test Data collection in ClutteredRoom environment.
sc = "1"
seed = 4
test_config = {
    "debug_env": False,
    "seed": seed,
    "delta_finger_control": False,
    "log_file": f"logs/cluttered_room_collect{sc}_seed{seed}.log",
    "lll_config": f"config/lifelong_learning/cluttered_room_sc{sc}.yaml",
    "control_mode": "pd_joint_delta_pos",
    "exp_name": f"cluttered_room_collect{sc}",
    "render_planner_collection_img": f"logs/cluttered_room_planner_collect{sc}_imgs_seed{seed}",
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
# sc1/2
with open(CFG.lll_config, "r", encoding="utf-8") as f:
    scenario_info = yaml.safe_load(f)["scenarios"][int(sc)]

cfg_settings = scenario_info.get("planner_learning_cfg_settings", {})
update_config(cfg_settings)
task_files = glob.glob(f"{CFG.specified_task_path}/sc{sc}_task_seed{CFG.seed}_*.pkl")
init_states = []
for task_file in task_files:
    with open(task_file, "rb") as f:
        task_data: Task = pickle.load(f)
        init_states.append(task_data.init)

tamp_system = ClutteredRoomRLTAMPSystem.create_default(render_mode="rgb_array", seed=42)
approach = LifelongRefRoomApproach(tamp_system, seed=CFG.seed)
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
eval_output_dir = Path(f"videos/croom-planner-data-collection-sc{sc}")
eval_output_dir.mkdir(parents=True, exist_ok=True)
# envs = ManiSkillsRecordVideo(
#     envs,
#     output_dir=eval_output_dir,
#     save_trajectory=False,
#     save_video=True,
#     trajectory_name="trajectory",
#     max_steps_per_video=CFG.max_env_steps * 2,
#     video_fps=30,
# )

envs_mani = PlanningStatesVectorRoomEnv(
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

# Set up dataset save path for incremental saving
dataset_save_path = (
    Path(cfg_settings["planner_dataset_path"]).parent
    / f"planner_incremental_sc{sc}_s{CFG.seed}"
)
dataset_save_path.mkdir(parents=True, exist_ok=True)

logging.info(f"Dataset will be saved incrementally to: {dataset_save_path}")
logging.info(f"Save frequency: every 5 episodes")

train_data = approach.collect_planner_data(
    envs,
    policy,
    init_states,
    real_env_scenario_name=f"sc{sc}_pre_n4",
    dataset_save_path=dataset_save_path,
    save_frequency=5,
)
envs.close()  # type: ignore[no-untyped-call]

# Save final consolidated dataset
final_dataset_path = Path(
    os.path.join(cfg_settings["planner_dataset_path"], f"seed_{CFG.seed}")
)
logging.info(f"Saving final consolidated dataset to: {final_dataset_path}")
train_data.save(final_dataset_path)
