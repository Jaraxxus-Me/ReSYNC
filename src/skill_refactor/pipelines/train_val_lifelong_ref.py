"""Train and evaluate SLAP policies for shortcut learning."""

import copy
import logging

# import imageio.v2 as iio
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from mani_skill.envs.sapien_env import BaseEnv as ManiskillBaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode  # type: ignore

from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.lifelong_ref_icy import LifelongRefIcyApproach
from skill_refactor.approaches.lifelong_ref_mani import LifelongRefManiApproach
from skill_refactor.approaches.pred_learner.sequential_finetuner import (
    SequentialPredicateFinetuner,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.approaches.rl_policies.ppo import PPOPolicy
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.approaches.rl_policies.ppo_q import PPOQPolicy
from skill_refactor.approaches.rl_policies.sac import SACPolicy
from skill_refactor.args import update_config
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.benchmarks.wrappers import (
    ManiSkillsRecordVideo,
    MultiEnvRecordVideo,
    MultiEnvWrapper,
    NormalizeActionMultiEnvWrapper,
    PlanningStatesVectorEnv,
)
from skill_refactor.datasets import get_or_collect_planner_data, get_or_collect_rl_data
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.gpu_utils import set_torch_seed
from skill_refactor.utils.structs import PlannerDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


def create_rl_policy(policy_name: str, seed: int, rl_config: str) -> Policy:
    """Create customized SAC policy."""
    if policy_name == "SAC":
        return SACPolicy(seed=seed, rl_config=rl_config)
    if policy_name == "PPOQ":
        return PPOQPolicy(seed=seed, rl_config=rl_config)
    if policy_name == "PPOC":
        return PPOCPolicy(seed=seed, rl_config=rl_config)
    assert policy_name == "PPO", f"Unsupported policy: {policy_name}"
    return PPOPolicy(seed=seed, rl_config=rl_config)


def train_and_evaluate_ll_ref(
    system_cls: type[BaseRLTAMPSystem],
    approach_name: str,
) -> Metrics:
    """Train and evaluate a policy on a system."""
    # Set all random seeds at the entry point
    seed = CFG.seed
    set_torch_seed(seed)
    learning_info: dict = {}
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        learning_info = yaml.safe_load(f)["scenarios"]

    logging.info(
        f"************SEED {seed} Lifelong learning starts, Total {len(learning_info)} Scenarios**************"
    )
    # Create approach
    base_system = system_cls.create_default(
        render_mode="rgb_array" if CFG.render else None,
        seed=CFG.seed,
    )

    # NOTE: The difference between the three LifelongRef approaches is mainly
    # in planner data collection labeling, since the environments are using different physical
    # actions (e.g., maniskill is stochastic, icy_transport has low-level force control).
    approach: LifelongRefIcyApproach | LifelongRefManiApproach | LifelongRefApproach
    if CFG.env == "icy_transport":
        approach = LifelongRefIcyApproach(base_system, seed=CFG.seed)
    elif CFG.env in ["cluttered_drawer", "cluttered_room"]:
        approach = LifelongRefManiApproach(base_system, seed=CFG.seed)
    else:
        assert CFG.env == "blocked_stacking"
        approach = LifelongRefApproach(base_system, seed=CFG.seed)

    for scenario_id, scenario_info in learning_info.items():
        if scenario_info.get("trained", False):
            logging.info(
                f"---Skipping scenario {scenario_id} as no learning is required---"
            )

            world_setting = scenario_info.get("planner_learning_cfg_settings", {})
            update_config(world_setting)

            curr_tamp_system = system_cls.create_default(
                render_mode="rgb_array" if CFG.render else None,
                seed=CFG.seed,
            )

            # Update learning info before domain knowledge update
            approach.update_learning_info(
                int(scenario_id),
                scenario_info,
                latest_tamp_system=curr_tamp_system,
            )

            # Use the new update_domain_knowledge method (creates and loads policy internally)
            approach.update_domain_knowledge(scenario_info)
        else:
            # Learning is required here.
            logging.info(
                f"---Learning in scenario {scenario_id} with info: {scenario_info}"
            )
            # 0). Reset config for learning skills in the current scenario
            cfg_settings_skill_learning = scenario_info.get(
                "skill_learning_cfg_settings", {}
            )
            update_config(cfg_settings_skill_learning)

            # 1). Create pure RL envs for data collection
            # Base env
            curr_tamp_system = system_cls.create_default(
                render_mode="rgb_array" if CFG.render else None,
                seed=CFG.seed,
            )
            pure_rl_env_train = curr_tamp_system.env
            if isinstance(pure_rl_env_train.unwrapped, ManiskillBaseEnv):
                # For ManiSkill3 envs
                pure_rl_env_eval = gym.make(
                    curr_tamp_system.env_name,
                    num_envs=CFG.num_eval_envs,
                    reconfiguration_freq=None,
                    human_render_camera_configs={"shader_pack": "default"},
                    **curr_tamp_system.env_kwargs,
                )
                # Record
                eval_output_dir = Path(
                    f"videos/{curr_tamp_system.name}_{approach_name}_sc{scenario_id}_eval/{CFG.exp_name}"
                )
                eval_output_dir.mkdir(parents=True, exist_ok=True)
                eval_envs = ManiSkillsRecordVideo(
                    pure_rl_env_eval,
                    output_dir=eval_output_dir.as_posix(),
                    save_trajectory=False,
                    save_video=True,
                    trajectory_name="trajectory",
                    max_steps_per_video=CFG.max_env_steps,
                    video_fps=30,
                )
            else:
                assert isinstance(pure_rl_env_train, MultiEnvWrapper)

                # For PRBench envs
                def make_env_fn(tamp_sys=curr_tamp_system):
                    return gym.make(
                        tamp_sys.env_name,
                        **tamp_sys.env_kwargs,
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
                    pure_rl_env_eval = MultiEnvWrapper(
                        make_env_fn,
                        num_envs=CFG.num_eval_envs,
                        auto_reset=False,
                        to_tensor=True,
                        device=CFG.device,
                        max_episode_steps=CFG.max_env_steps,
                    )
                # Record
                eval_output_dir = Path(
                    f"videos/{curr_tamp_system.name}_{approach_name}_sc{scenario_id}_eval/{CFG.exp_name}"
                )
                eval_output_dir.mkdir(parents=True, exist_ok=True)
                eval_envs = MultiEnvRecordVideo(  # type: ignore[assignment]
                    pure_rl_env_eval,
                    video_folder=eval_output_dir.as_posix(),
                    episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
                )

            fall_back_action = pure_rl_env_train.single_action_space.sample()  # type: ignore
            normalize_action, arm_action_low, arm_action_high = (
                get_normalize_action_range(curr_tamp_system.env, CFG.control_mode)
            )

            # 2). Learning new policy with the current planner
            latest_planner = TaskThenMotionPlanner(
                types=curr_tamp_system.types,
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
            policy = create_rl_policy(
                CFG.rl_algo, seed, cfg_settings_skill_learning.get("rl_config", "")
            )
            envs_mani = PlanningStatesVectorEnv(
                pure_rl_env_train,
                curr_tamp_system,
                scenario_info,
                latest_planner,
                num_envs=CFG.num_envs,
                ignore_terminations=True,
                record_metrics=True,
            )
            eval_envs_mani = PlanningStatesVectorEnv(
                eval_envs,
                curr_tamp_system,
                scenario_info,
                copy.deepcopy(latest_planner),
                num_envs=CFG.num_eval_envs,
                ignore_terminations=True,
                record_metrics=True,
            )

            policy.initialize(envs_mani)
            approach.update_learning_info(
                scenario_id,
                scenario_info,
            )

            # Collect training data for the new skill
            # Use the original environment since we will be using all the skills.
            train_data = get_or_collect_rl_data(pure_rl_env_train, approach)

            logging.info("\nTraining policy...")
            envs_mani.configure_training(train_data)  # pylint: disable=protected-access
            eval_envs_mani.configure_training(  # pylint: disable=protected-access
                train_data
            )
            start_time = time.time()
            pre_trained_policy_path = Path(CFG.pre_trained_policy_path)
            if pre_trained_policy_path.exists():
                logging.info(
                    f"Loading pre-trained policy from {pre_trained_policy_path}"
                )
                policy.load(pre_trained_policy_path)
            else:
                policy.train(envs_mani, eval_envs_mani, train_data)  # type: ignore
                policy_training_time = time.time() - start_time
                logging.info(
                    f"Training completed in {policy_training_time:.2f} seconds"
                )
                pre_trained_policy_path = (
                    Path(CFG.rl_policy_save_dir) / f"runs/{CFG.exp_name}/final_ckpt.pt"
                )
                assert (
                    pre_trained_policy_path.exists()
                ), f"Pre-trained policy not found at {pre_trained_policy_path}"
                logging.info(f"Using pre-trained policy from {pre_trained_policy_path}")
                policy.load(pre_trained_policy_path)

            # 2). Collect planner data for the new skill in the current scenario
            cfg_settings_planner_learning = scenario_info.get(
                "planner_learning_cfg_settings", {}
            )
            update_config(cfg_settings_planner_learning)
            curr_tamp_system = system_cls.create_default(
                render_mode="rgb_array" if CFG.render else None,
                seed=CFG.seed,
            )
            pure_rl_env_planner = curr_tamp_system.env
            new_planner_data = get_or_collect_planner_data(
                pure_rl_env_planner,
                approach,
                policy=policy,
                real_scenario_name=CFG.real_scenario_name,
            )
            pure_rl_env_planner.close()  # type: ignore[no-untyped-call]

            # 3). Invent new predicates and refactor the planner part of the approach

            # 3.1). Use latest planner dataset to fine-tune predicates from all previous scenarios
            # for prev_scenario_id, prev_scenario_info in learning_info.items():
            #     if prev_scenario_info.get("trained"):
            #         # For all the trained scenarios, use the current data to fine-tine the predicates
            #         prev_predicate_config = prev_scenario_info[
            #             "planner_learning_cfg_settings"
            #         ].get("predicate_config", None)
            #         # Load trajectory dataset
            #         prev_dataset_path_str = prev_scenario_info[
            #             "planner_learning_cfg_settings"
            #         ].get("planner_dataset_path")
            #         if prev_dataset_path_str is None:
            #             raise ValueError(
            #                 "dataset_path must be specified in learning_info"
            #             )
            #         prev_dataset_path = Path(
            #             os.path.join(prev_dataset_path_str, f"seed_{CFG.seed}")
            #         )
            #         num_traj = prev_scenario_info["planner_learning_cfg_settings"].get(
            #             "planner_num_traj", -1
            #         )
            #         prev_planner_dataset = PlannerDataset.load(
            #             prev_dataset_path, num_traj=num_traj
            #         )
            #         assert prev_predicate_config is not None
            #         with open(prev_predicate_config, "r", encoding="utf-8") as f:
            #             prev_config_data = yaml.safe_load(f)
            #         prev_predicate_configures = prev_config_data["predicates"]
            #         sequential_finetuner = SequentialPredicateFinetuner(
            #             old_planner_dataset=prev_planner_dataset,
            #             scenario=str(prev_scenario_id),
            #             dataset=new_planner_data,
            #             tamp_system=curr_tamp_system,
            #             predicate_configures=prev_predicate_configures,
            #             quantify_basic_predicates=False,
            #             verbose=True,
            #         )
            #         sequential_finetuner.finetune()

            # Create a new approach instance here
            # Reset approach to avoid predicate conflicting after finetuning
            if CFG.env == "icy_transport":
                approach = LifelongRefIcyApproach(curr_tamp_system, seed=CFG.seed)
            elif CFG.env in ["cluttered_drawer", "cluttered_room"]:
                approach = LifelongRefManiApproach(curr_tamp_system, seed=CFG.seed)
            else:
                approach = LifelongRefApproach(curr_tamp_system, seed=CFG.seed)
            # 3.2). Instantiate approach to load these fine-tuned predicates
            for prev_scenario_id, prev_scenario_info in learning_info.items():
                # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
                cfg_settings_after_sc = prev_scenario_info.get(
                    "planner_learning_cfg_settings", {}
                )
                update_config(cfg_settings_after_sc)
                latest_tamp_system = system_cls.create_default(
                    render_mode="rgb_array", seed=42
                )
                approach.update_learning_info(
                    int(prev_scenario_id),
                    prev_scenario_info,
                    latest_tamp_system=latest_tamp_system,
                )
                if prev_scenario_info.get("trained"):
                    approach.update_domain_knowledge(prev_scenario_info)

            # 3.3). Invent new predicates and operators with the new data
            with open(CFG.predicate_config, "r", encoding="utf-8") as f:
                predicate_configures = yaml.safe_load(f)
            predicate_learner = TopDownPredicateLearner(
                dataset=new_planner_data,
                tamp_system=curr_tamp_system,
                latest_perceiver=approach.perceiver,
                predicate_configures=predicate_configures["predicates"],
                quantify_basic_predicates=CFG.quantify_basic,
                verbose=True,
                scenario=str(scenario_id),
            )
            start_time = time.time()
            invented_pred_interpr, op_set = predicate_learner.invent()
            pred_learning_time = time.time() - start_time
            logging.info(f"Predicate learning took {pred_learning_time:.2f} seconds")

            # 3.4). Update approach with the new predicates and operators
            if CFG.filter_preconditions:
                invented_pred_interpr_sofar = (
                    approach.get_invented_predicate_interpretr_so_far()
                )
                invented_pred_interpr_sofar.update(invented_pred_interpr)
                filtered_op_set = predicate_learner.filter_preconditions(
                    op_set,
                    invented_pred_interpr_sofar,
                    dataset=new_planner_data,
                    num_traj_per_scenario=5,
                    unsolvable_threshold=CFG.pred_search_precond_unsolvable_threshold,
                )
                json_path = Path(
                    os.path.join(
                        CFG.pred_net_save_dir,
                        CFG.invented_pred_op_json + f"_sc{scenario_id}_filtered.json",
                    )
                )
                predicate_learner.save_invented_predicates_and_operators(
                    invented_pred_interpr, filtered_op_set, json_path
                )

    return Metrics(
        success_rate=0.0,
        avg_episode_length=0.0,
        avg_reward=0.0,
        training_time=policy_training_time + pred_learning_time,
    )
