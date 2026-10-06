import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pytest
import torch
import wandb
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.bc import BehaviorCloningApproach
from skill_refactor.approaches.behavior_cloning.gnn_bc import GNNBCPolicy
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.neural_dataset import (
    create_train_val_dataloaders,
    train_predicate_model,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    EncodeDecodeMLP,
    PoseMLP,
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.approaches.pred_learner.sequential_finetuner import (
    SequentialPredicateFinetuner,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    OperatorTransition,
    TopDownPredicateLearner,
)
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.base import GraphData
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import (
    PlannerDataset,
    Segment,
)
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanner,
    TaskThenMotionPlanningFailure,
)


@pytest.mark.skip(reason="The script requires local data and trained model")
def test_creating_transition_dataset():
    """
    Smoke test for BC-GNN Step A:
    - Load PlannerDataset
    - Run TopDownPredicateLearner
    - Verify operator_transition_data exists and has graphs

    This test does NOT train a GNN.
    """

    sc = "12_2"
    seed = "0"
    reset_config(
        {
            "traj_segmenter": "operator_changes",
            "predicate_config": f"config/predicates/blocked_stacking_confined_enu_sc{sc}.yaml",
            "loglevel": logging.INFO,
            "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        }
    )
    # important to register for gym envs
    register_all_environments()

    # -----------------------------
    # Load predicate config
    # -----------------------------
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    # -----------------------------
    # Load planner dataset (SC12, seed 2)
    # -----------------------------
    dataset_path = Path(
        f"training_data/blocked_stacking/Planner_data/scenario{sc}/seed_{seed}"
    )
    assert dataset_path.exists(), dataset_path

    planner_dataset = PlannerDataset.load(dataset_path, num_traj=5)

    # -----------------------------
    # Create TAMP system (for graphification only)
    # -----------------------------
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # -----------------------------
    # Initialize TopDown (NO predicate invention)
    # -----------------------------
    topdown = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        quantify_basic_predicates=False,
        verbose=False,
        scenario=sc.rsplit("_", maxsplit=1)[-1],
    )

    # -----------------------------
    # Build transition dataset
    # -----------------------------
    segment_data = []
    for low_level_traj, ground_atoms in topdown._ground_atom_dataset:
        segments = segment_trajectory(low_level_traj, ground_atoms)
        segment_data.append(segments)

    op_transition_data = topdown._create_transition_dataset(segment_data)
    # -----------------------------
    # Sanity checks / prints
    # -----------------------------
    assert len(op_transition_data) > 0

    assert isinstance(op_transition_data, list)
    assert len(op_transition_data) > 0

    first = op_transition_data[0]
    example = first[0] if isinstance(first, list) else first

    print("Example type:", type(example))
    print("Example:", example)
    print("\n[BC-GNN]")
    print("Num transitions:", len(op_transition_data))

    # Access first transition from op_transition_data
    first_transition = op_transition_data[0]
    pre_graph = first_transition.pre_state_graph
    print("Node features:", pre_graph.node_features.shape)
    print("Edge index:", pre_graph.edge_indices.shape)


@pytest.mark.skip(reason="The script requires local data and trained model")
@pytest.mark.parametrize("seeed", [0])
def test_loading_bc_gnn_blocked_stacking_sc12_2(seeed) -> None:
    """Test BC-GNN policy evaluation on BlockedStacking environment.

    This test:
    - Loads a trained BC-GNN model from checkpoint
    - Evaluates it on multiple obstruction configurations
    - Records videos and logs success rates

    Model path: trained_policies/runs/bc_gnn_BlockedStackingTAMPSystem_sc12_2_seed_0/model.pt
    """
    model_location = "trained_policies/runs/bc_gnn_pred_BlockedStackingTAMPSystem_sc12_2_seed_0_20260109_0003"
    name_of_the_model = model_location.rsplit("/", maxsplit=1)[-1]
    sc = "12_2"
    seed = seeed

    # Read bc_use_predicate_augmentation from environment variable
    # Default to True if not set
    bc_use_pred_aug = os.environ.get(
        "BC_USE_PREDICATE_AUGMENTATION", "True"
    ).lower() in ("true", "1", "yes")

    CFG.use_wandb = os.environ.get("USE_WANDB", "True").lower() in ("true", "1", "yes")

    test_config = {
        # Note: Do NOT set scenario here - it will be set in eval_configs loop
        # Setting it here causes "12_2" to be parsed as obstruction IDs
        "num_envs": 1,
        "seed": seed,
        "control_mode": "pd_joint_delta_pos",
        "loglevel": logging.INFO,
        "log_file": f"logs/{name_of_the_model}.log",
        "max_env_steps": 300,
        "planner_dataset_path": f"training_data/blocked_stacking/Planner_data/scenario{sc}/seed_{seed}",
        "traj_segmenter": "operator_changes",
        "bc_model": "gnn",
        "bc_use_predicate_augmentation": bc_use_pred_aug,
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}_seed{seed}.yaml",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pre_trained_policy_path": f"trained_policies/runs/skill_1127_sc1_seed{seed}/best_ppo_ckpt.pt",
    }
    reset_config(test_config)
    logging.info(
        f"BC predicate augmentation: {'ENABLED' if bc_use_pred_aug else 'DISABLED'}"
    )
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

    logging.info("=" * 80)
    logging.info(f"Testing BC-GNN for Scenario {sc}")
    logging.info(
        f"Predicate augmentation: {'ENABLED' if CFG.bc_use_predicate_augmentation else 'DISABLED'}"
    )
    logging.info("=" * 80)

    with open(CFG.lll_config, "rb") as f:
        lll_config_data = yaml.safe_load(f)
    # -----------------------------
    # Create TAMP system for graphification
    # -----------------------------
    base_tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    tamp_system = base_tamp_system

    # extracting goal atoms from the tamp system
    obs, info = tamp_system.env.reset()
    _, _, goal_atoms = tamp_system.perceiver.reset(obs[:1], info)
    logging.info(f"Extracted goal atoms: {goal_atoms}")

    # -----------------------------
    # Create BC-GNN policy
    # -----------------------------
    bc_policy = GNNBCPolicy(
        dataset=[],
        seed=seed,
        device=CFG.device,
        tamp_system=tamp_system,
        goal_atoms=goal_atoms,
    )

    # -----------------------------
    # Load trained model from checkpoint
    # -----------------------------
    # Check for both predicate-augmented and non-augmented models
    pred_suffix = "_pred" if CFG.bc_use_predicate_augmentation else "_nopred"
    model_path = Path(
        f"trained_policies/runs/bc_gnn{pred_suffix}_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
    )

    if not model_path.exists():
        logging.warning(f"Model checkpoint not found at {model_path}")
        logging.warning("Attempting to find in alternative locations...")

        # Try alternative paths (both with and without predicate suffix)
        alt_paths = [
            Path(
                f"trained_policies/runs/bc_gnn{pred_suffix}_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            Path(
                f"trained_policies/bc_gnn{pred_suffix}_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            # Legacy paths without suffix
            Path(
                f"trained_policies/runs/bc_gnn_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            Path(
                f"trained_policies/bc_gnn_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            Path(
                "trained_policies/runs/bc_gnn_BlockedStackingTAMPSystem_sc12_2_seed_0/model.pt"
            ),
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                logging.info(f"Found model at alternative path: {model_path}")
                break
        else:
            pytest.skip(f"No trained model found. Expected at: {model_path}")

    logging.info(f"Loading BC-GNN model from: {model_path}")
    bc_policy.load(model_path)
    logging.info("BC-GNN model loaded successfully")
    logging.info(
        f"Model expects node predicates: {list(bc_policy._node_pred_to_index.keys())}"
    )
    logging.info(
        f"Model expects edge predicates: {list(bc_policy._edge_pred_to_index.keys())}"
    )

    # -----------------------------
    # Load lifelong operators and skills
    # -----------------------------
    # Create lifelong approach and load operators/skills from ALL scenarios
    # For sc12_2, this means loading from scenario 1, then scenario 2
    lifelong_approach = LifelongRefApproach(
        system=base_tamp_system,
        seed=seed,
    )

    # Loop through all scenarios in the lifelong config (just like lifelong_ref test)
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update config with settings from this scenario
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)

        # Create fresh TAMP system for this scenario
        latest_tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )

        # Update approach with this scenario's operators/skills
        lifelong_approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        lifelong_approach.update_domain_knowledge(scenario_info)

    # After loading all scenarios, get the final system with all operators/skills/predicates
    tamp_system = lifelong_approach.system  # type: ignore[assignment]
    tamp_system.components.operators = lifelong_approach.operators
    tamp_system.components.skills = lifelong_approach.skills

    bc_policy.tamp_system = tamp_system  # Update BC-GNN's reference for consistency

    approach = BehaviorCloningApproach(
        system=tamp_system,
        bc_policy=bc_policy,
        seed=seed,
    )

    # -----------------------------
    # Evaluation configurations
    # -----------------------------
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
    ]

    # Store overall results
    all_results = {}

    # -----------------------------
    # Initialize Weights & Biases for evaluation (if enabled)
    # -----------------------------
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    if CFG.use_wandb:
        wandb_project = getattr(CFG, "wandb_project", "bc-gnn-evaluation")
        wandb_entity = getattr(CFG, "wandb_entity", None)
        pred_tag = "using_pred_aug" if CFG.bc_use_predicate_augmentation else "nopred"

        # Add timestamp to run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"eval_bc_gnn_{pred_tag}_sc{sc}_seed{seed}_{timestamp}"

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "scenario": sc,
                "seed": seed,
                "bc_use_predicate_augmentation": CFG.bc_use_predicate_augmentation,
                "num_eval_episodes": CFG.num_eval_episodes,
                "max_env_steps": CFG.max_env_steps,
                "model_path": str(model_path),
            },
            tags=["evaluation", f"sc{sc}", pred_tag],
        )
        logging.info(f"Initialized W&B evaluation run: {wandb_run.name}")

    # -----------------------------
    # Run evaluation for each configuration
    # -----------------------------
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

        # Create new TAMP system with updated config
        # tamp_system = BlockedStackingRLTAMPSystem.create_default(
        #    render_mode="rgb_array", seed=42
        # )

        ## Update approach with new system
        # approach.system = tamp_system

        # Set up video recording
        # Include predicate augmentation status in video folder name
        pred_tag = "using_pred_aug" if CFG.bc_use_predicate_augmentation else "nopred"
        video_folder = Path(
            f"videos/bc_gnn_{pred_tag}_{tamp_system.name}_sc{sc}_seed{seed}_{eval_name}"
        )
        envs = MultiEnvRecordVideo(
            tamp_system.env,
            video_folder=video_folder.as_posix(),
            episode_trigger=lambda _: True,
        )

        success = []
        episode_lengths = []
        rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))

        for epi in range(CFG.num_eval_episodes):
            reset_options: dict = {}
            obs, info = envs.reset(
                options=reset_options, seed=rnd_seed[epi] + seed
            )  # type: ignore[no-untyped-call]

            # Reset approach for new episode
            step_result = approach.reset(obs, info)

            total_reward = torch.tensor(
                [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
            )
            epi_success = torch.zeros(
                CFG.num_envs, dtype=torch.bool, device=envs.device
            )

            for step in range(CFG.max_env_steps + 1):
                obs, _, _, _, info = envs.step(step_result.action)
                bool_success = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_success

                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    episode_lengths.append(step)
                    break

                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            else:
                # Episode finished without success
                episode_lengths.append(CFG.max_env_steps)

            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")

        success_rate = sum(success) / len(success)
        avg_length = sum(episode_lengths) / len(episode_lengths)

        logging.info(f"\n{'='*80}")
        logging.info(f"Configuration {eval_name} Results:")
        logging.info(f"  Success rate: {success_rate:.3f}")
        logging.info(f"  Avg episode length: {avg_length:.1f}")
        logging.info(f"{'='*80}\n")

        all_results[eval_name] = {
            "success_rate": success_rate,
            "avg_length": avg_length,
            "num_episodes": len(success),
        }

        # Log to Weights & Biases
        if wandb_run is not None:
            wandb.log(
                {
                    f"eval/{eval_name}/success_rate": success_rate,
                    f"eval/{eval_name}/avg_episode_length": avg_length,
                    f"eval/{eval_name}/num_episodes": len(success),
                }
            )

        envs.close()  # type: ignore[no-untyped-call]

    # -----------------------------
    # Summary of all configurations
    # -----------------------------
    logging.info(f"\n{'='*80}")
    logging.info("OVERALL EVALUATION SUMMARY")
    logging.info(f"{'='*80}")
    for config_name, results in all_results.items():
        logging.info(
            f"{config_name}: Success={results['success_rate']:.3f}, "
            f"Avg Length={results['avg_length']:.1f}"
        )
    logging.info(f"{'='*80}\n")

    # Log overall summary to Weights & Biases
    if wandb_run is not None:
        # Calculate overall metrics
        overall_success = sum(r["success_rate"] for r in all_results.values()) / len(
            all_results
        )
        overall_avg_length = sum(r["avg_length"] for r in all_results.values()) / len(
            all_results
        )

        wandb.log(
            {
                "eval/overall_success_rate": overall_success,
                "eval/overall_avg_episode_length": overall_avg_length,
            }
        )
        wandb.finish()
        logging.info("W&B evaluation run finished")


@pytest.mark.skip(reason="The script requires local trained model")
@pytest.mark.parametrize("seed", [0])
def test_loading_bc_gnn_blocked_stacking_sc1(seed) -> None:
    """Test BC-GNN with BlockedStacking environment - Scenario 1."""

    model_location = "trained_policies/runs/bc_gnn_pred_BlockedStackingTAMPSystem_sc1_seed_0_20260105_1033"
    name_of_the_model = model_location.rsplit("/", maxsplit=1)[-1]
    sc = "1"

    # Set up configuration
    bc_use_pred_aug = os.environ.get(
        "BC_USE_PREDICATE_AUGMENTATION", "True"
    ).lower() in ("true", "1", "yes")
    CFG.use_wandb = os.environ.get("USE_WANDB", "True").lower() in ("true", "1", "yes")
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": sc,
        "control_mode": "pd_joint_delta_pos",
        "bc_use_predicate_augmentation": bc_use_pred_aug,
        "loglevel": logging.INFO,
        "log_file": f"logs/{name_of_the_model}_test.log",
        "max_env_steps": 350,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        "pred_net_save_dir": f"skill_1127_sc{sc}_pred_nets_seed{seed}",
        "pre_trained_policy_path": f"trained_policies/runs/skill_1127_sc1_seed{seed}/best_ppo_ckpt.pt",
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

    logging.info("=" * 80)
    logging.info(f"Testing BC-GNN for Scenario {sc}")
    logging.info(
        f"Predicate augmentation: {'ENABLED' if CFG.bc_use_predicate_augmentation else 'DISABLED'}"
    )
    logging.info("=" * 80)

    # -----------------------------
    # Load planner dataset
    # -----------------------------
    planner_dataset_path = Path(
        f"training_data/blocked_stacking/Planner_data/scenario{sc}/seed_{seed}"
    )
    logging.info(f"Loading planner dataset from: {planner_dataset_path}")

    if not planner_dataset_path.exists():
        pytest.skip(f"Planner dataset not found at {planner_dataset_path}")

    planner_dataset = PlannerDataset.load(
        planner_dataset_path,
        num_traj=-1,  # Load all trajectories
    )
    logging.info(f"Loaded {len(planner_dataset.trajectories)} trajectories")

    # -----------------------------
    # Create TAMP system with lifelong operators/skills
    # -----------------------------
    with open(CFG.lll_config, "rb") as f:
        lll_config_data = yaml.safe_load(f)

    # create base TAMP system
    base_tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=seed
    )

    tamp_system = base_tamp_system

    # -----------------------------
    # Build operator transition dataset
    # -----------------------------
    logging.info("Building operator transition dataset...")
    topdown = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=base_tamp_system,  # Use lifelong system for consistency
        predicate_configures=[],
        quantify_basic_predicates=False,
        verbose=False,
        scenario=sc,
    )

    segment_data = []
    for low_traj, ground_atoms in topdown._ground_atom_dataset:
        segments = segment_trajectory(low_traj, ground_atoms)
        segment_data.append(segments)

    operator_transition_data = topdown._create_transition_dataset(segment_data)
    logging.info(f"Built {len(operator_transition_data)} operator transitions")

    # Extract goal atoms for predicate augmentation (fairness with other approaches)
    # All trajectories in the same scenario share the same goal
    goal_atoms = topdown._train_tasks[0].goal if topdown._train_tasks else set()
    logging.info(
        f"Using goal with {len(goal_atoms)} atoms for BC-GNN predicate augmentation"
    )

    # -----------------------------
    # Create BC-GNN policy
    # -----------------------------
    bc_policy = GNNBCPolicy(
        dataset=operator_transition_data,
        seed=seed,
        device=CFG.device,
        tamp_system=tamp_system,  # Now tamp_system is defined
        goal_atoms=goal_atoms,
    )

    # -----------------------------
    # Load trained model from checkpoint
    # -----------------------------
    # Check for both predicate-augmented and non-augmented models
    pred_suffix = "_pred" if CFG.bc_use_predicate_augmentation else "_nopred"
    model_path = Path(f"{model_location}/model.pt")

    if not model_path.exists():
        logging.warning(f"Model checkpoint not found at {model_path}")
        logging.warning("Attempting to find in alternative locations...")

        # Try alternative paths
        alt_paths = [
            Path(
                f"trained_policies/runs/bc_gnn{pred_suffix}_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            Path(
                f"trained_policies/bc_gnn{pred_suffix}_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            # Legacy paths without suffix
            Path(
                f"trained_policies/runs/bc_gnn_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
            Path(
                f"trained_policies/bc_gnn_BlockedStackingTAMPSystem_sc{sc}_seed_{seed}/model.pt"
            ),
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                logging.info(f"Found model at alternative path: {model_path}")
                break
        else:
            pytest.skip(f"No trained model found. Expected at: {model_path}")

    logging.info(f"Loading BC-GNN model from: {model_path}")
    bc_policy.load(model_path)
    logging.info("BC-GNN model loaded successfully")

    # -----------------------------
    # Wrap in BehaviorCloningApproach
    # -----------------------------

    lifelong_approach = LifelongRefApproach(
        system=base_tamp_system,
        seed=seed,
    )
    scenario_info = lll_config_data["scenarios"][int(sc)]

    world_setting = scenario_info.get("planner_learning_cfg_settings", {})
    update_config(world_setting)
    lifelong_approach.update_learning_info(int(sc), scenario_info)
    lifelong_approach.update_domain_knowledge(scenario_info)

    tamp_system = lifelong_approach.system  # type: ignore[assignment]
    tamp_system.components.operators = lifelong_approach.operators  # ← Add this!
    tamp_system.components.skills = lifelong_approach.skills

    bc_policy.tamp_system = tamp_system

    logging.info(
        f"TAMP system has {len(tamp_system.operators)} operators and {len(tamp_system.skills)} skills after copying the lifelong_operators and skills"
    )

    approach = BehaviorCloningApproach(
        system=tamp_system,
        bc_policy=bc_policy,
        seed=seed,
    )

    # -----------------------------
    # Evaluation configurations for scenario 1
    # -----------------------------
    eval_configs = [
        {
            "name": "1_b",
            "scenario": "1",
            "obstruction1_blocking_grasp": False,
            "obstruction1_blocking_stacking": True,
        },
        {
            "name": "1_g",
            "scenario": "1",
            "obstruction1_blocking_grasp": True,
            "obstruction1_blocking_stacking": False,
        },
    ]

    # Store overall results
    all_results = {}

    # -----------------------------
    # Initialize Weights & Biases for evaluation
    # -----------------------------
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
    CFG.use_wandb = True
    if CFG.use_wandb:
        wandb_project = getattr(CFG, "wandb_project", "bc-gnn-evaluation")
        wandb_entity = getattr(CFG, "wandb_entity", None)
        pred_tag = "pred" if CFG.bc_use_predicate_augmentation else "nopred"

        # Add timestamp to run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"eval_bc_gnn_{pred_tag}_sc{sc}_seed{seed}_{timestamp}"

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "scenario": sc,
                "seed": seed,
                "bc_use_predicate_augmentation": CFG.bc_use_predicate_augmentation,
                "num_eval_episodes": CFG.num_eval_episodes,
                "max_env_steps": CFG.max_env_steps,
                "model_path": str(model_path),
            },
            tags=["evaluation", f"sc{sc}", pred_tag],
        )
        logging.info(f"Initialized W&B evaluation run: {wandb_run.name}")

    # -----------------------------
    # Run evaluation for each configuration
    # -----------------------------
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
            "max_env_steps": 350,
        }
        update_config(test_config)

        # Create new TAMP system with updated config
        # tamp_system = BlockedStackingRLTAMPSystem.create_default(
        #    render_mode="rgb_array", seed=42
        # )

        # Update approach with new system
        # approach.system = tamp_system

        # Set up video recording
        # Include predicate augmentation status in video folder name
        pred_tag = "pred" if CFG.bc_use_predicate_augmentation else "nopred"
        video_folder = Path(
            f"videos/bc_gnn_{pred_tag}_{tamp_system.name}_sc{sc}_seed{seed}_{eval_name}"
        )
        envs = MultiEnvRecordVideo(
            tamp_system.env,
            video_folder=video_folder.as_posix(),
            episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
        )

        success = []
        episode_lengths = []
        rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))

        for epi in range(CFG.num_eval_episodes):
            reset_options: dict = {}
            obs, info = envs.reset(seed=rnd_seed[epi] + seed, options=reset_options)  # type: ignore[no-untyped-call]

            try:
                step_result = approach.reset(obs, info)
            except Exception as e:  # pylint: disable=broad-except
                import traceback

                logging.error(f"Episode {epi} failed during reset with error: {e}")
                logging.error(f"Traceback:\n{traceback.format_exc()}")
                success.append(False)
                episode_lengths.append(0)
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
                bool_success = torch.tensor(
                    info["success"], dtype=torch.bool, device=epi_success.device
                )
                epi_success |= bool_success

                if epi_success.all():
                    logging.info(f"Episode {epi} all succeeded early at step {step}.")
                    episode_lengths.append(step + 1)
                    break

                step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]
            else:
                # Episode finished without success
                episode_lengths.append(CFG.max_env_steps)

            success.extend(epi_success.cpu().numpy().tolist())
            logging.info(f"Episode {epi} final success: {epi_success}.")

        success_rate = sum(success) / len(success)
        avg_length = sum(episode_lengths) / len(episode_lengths)

        logging.info(f"\n{'='*80}")
        logging.info(f"Configuration {eval_name} Results:")
        logging.info(f"  Success rate: {success_rate:.3f}")
        logging.info(f"  Avg episode length: {avg_length:.1f}")
        logging.info(f"{'='*80}\n")

        all_results[eval_name] = {
            "success_rate": success_rate,
            "avg_length": avg_length,
            "num_episodes": len(success),
        }

        # Log to Weights & Biases
        if wandb_run is not None:
            wandb.log(
                {
                    f"eval/{eval_name}/success_rate": success_rate,
                    f"eval/{eval_name}/avg_episode_length": avg_length,
                    f"eval/{eval_name}/num_episodes": len(success),
                }
            )

        envs.close()  # type: ignore[no-untyped-call]

    # -----------------------------
    # Summary of all configurations
    # -----------------------------
    logging.info(f"\n{'='*80}")
    logging.info("OVERALL EVALUATION SUMMARY - SCENARIO 1")
    logging.info(f"{'='*80}")
    for config_name, results in all_results.items():
        logging.info(
            f"{config_name}: Success={results['success_rate']:.3f}, "
            f"Avg Length={results['avg_length']:.1f}"
        )
    logging.info(f"{'='*80}\n")

    # Log overall summary to Weights & Biases
    if wandb_run is not None:
        # Calculate overall metrics
        overall_success = sum(r["success_rate"] for r in all_results.values()) / len(
            all_results
        )
        overall_avg_length = sum(r["avg_length"] for r in all_results.values()) / len(
            all_results
        )

        wandb.log(
            {
                "eval/overall_success_rate": overall_success,
                "eval/overall_avg_episode_length": overall_avg_length,
            }
        )
        wandb.finish()
        logging.info("W&B evaluation run finished")
