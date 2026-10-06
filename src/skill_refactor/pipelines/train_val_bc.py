"""Train and evaluate a Behavior Cloning (BC) baseline using offline planner data."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import wandb

from skill_refactor.approaches.behavior_cloning.gnn_bc import GNNBCPolicy
from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.topdown_learner import (
    OperatorTransition,
    TopDownPredicateLearner,
)
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import PlannerDataset


def extract_dataset_identifier(dataset_path: str) -> str:
    """Extract dataset identifier from planner dataset path.

    Example: 'training_data/blocked_stacking/Planner_data/scenario12_2/seed_0'
             -> 'sc12_2_seed_0'
    """
    path = Path(dataset_path)
    parts = path.parts

    # Find scenario and seed directories
    scenario = None
    seed = None
    for part in parts:
        if part.startswith("scenario"):
            # Extract scenario number: scenario12_2 -> sc12_2
            scenario = "sc" + part.replace("scenario", "")
        elif part.startswith("seed_"):
            seed = part  # Keep as seed_0, seed_1, etc.

    if scenario and seed:
        return f"{scenario}_{seed}"
    # Fallback: use last two directory names
    return "_".join(parts[-2:]) if len(parts) >= 2 else "unknown"


def train_and_evaluate_bc(
    system_cls: type[BaseRLTAMPSystem],
    approach_name: str,  # pylint: disable=unused-argument
) -> Metrics:
    """Offline behavior cloning baseline."""

    seed = CFG.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------
    # 1. Load planner dataset (silently for now)
    # --------------------------------------------------
    assert CFG.planner_dataset_path, "CFG.planner_dataset_path must be set"

    dataset_path = Path(CFG.planner_dataset_path)

    planner_dataset = PlannerDataset.load(
        dataset_path,
        num_traj=CFG.bc_dataset_num_traj,
    )

    num_trajectories = len(planner_dataset.trajectories)

    # --------------------------------------------------
    # 2. Create TAMP system (graphification only)
    # --------------------------------------------------
    tamp_system = system_cls.create_default(
        render_mode="rgb_array" if CFG.render else None,
        seed=seed,
    )

    # --------------------------------------------------
    # 3. Build operator transition dataset (graphs)
    # --------------------------------------------------
    topdown = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=[],  # not used for BC
        quantify_basic_predicates=False,
        verbose=False,
        scenario=str(CFG.scenario),
    )

    segment_data = []
    for low_traj, ground_atoms in topdown._ground_atom_dataset:
        segments = segment_trajectory(low_traj, ground_atoms)
        segment_data.append(segments)

    operator_transition_data = topdown._create_transition_dataset(segment_data)
    # this is a list.
    # each element shows (pre_state_graph, operator) -> post_state_graph.

    num_transitions = len(operator_transition_data)

    # Extract goal atoms for predicate augmentation (fairness with other approaches)
    # All trajectories in the same scenario share the same goal
    goal_atoms = topdown._train_tasks[0].goal if topdown._train_tasks else set()
    num_goal_atoms = len(goal_atoms)

    # --------------------------------------------------
    # 4. Initialize Weights & Biases (if enabled)
    # --------------------------------------------------
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None

    if CFG.use_wandb:
        # Extract dataset identifier for run naming
        dataset_id = extract_dataset_identifier(CFG.planner_dataset_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        pred_suffix = "pred" if CFG.bc_use_predicate_augmentation else "nopred"

        wandb_run_name = f"bc_{CFG.bc_model}_{pred_suffix}_{tamp_system.name}_{dataset_id}_{timestamp}"

        wandb_run = wandb.init(
            project=CFG.wandb_project,
            entity=CFG.wandb_entity,
            name=wandb_run_name,
            config={
                # Model configuration
                "bc_model": CFG.bc_model,
                "bc_use_predicate_augmentation": CFG.bc_use_predicate_augmentation,
                # Dataset information
                "num_trajectories": num_trajectories,
                "num_transitions": num_transitions,
                "num_goal_atoms": num_goal_atoms,
                "planner_dataset_path": str(dataset_path),
                "dataset_identifier": dataset_id,
                # Environment configuration
                "env": CFG.env,
                "scenario": CFG.scenario,
                "seed": seed,
                # Training configuration (will be updated by GNN training)
                "device": str(CFG.device),
            },
            tags=["training", f"bc_{CFG.bc_model}", pred_suffix, f"sc{CFG.scenario}"],
        )

        # Log code for reproducibility
        if wandb_run is not None:
            wandb_run.log_code("src/skill_refactor")
            logging.info(f"Initialized W&B run: {wandb_run.name}")

            # Log dataset statistics
            wandb.log(
                {
                    "dataset/num_trajectories": num_trajectories,
                    "dataset/num_transitions": num_transitions,
                    "dataset/num_goal_atoms": num_goal_atoms,
                }
            )

    # --------------------------------------------------
    # Now log all the information (after wandb is initialized)
    # --------------------------------------------------
    logging.info("=" * 80)
    logging.info(f"Running BC pipeline with model: {CFG.bc_model}")
    logging.info("=" * 80)
    logging.info(f"Environment: {CFG.env}")
    logging.info(f"Scenario: {CFG.scenario}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Device: {CFG.device}")
    logging.info(f"Planner dataset path: {dataset_path}")
    logging.info(f"Loaded {num_trajectories} trajectories")
    logging.info(f"Built {num_transitions} operator transitions")
    logging.info(
        f"Using goal with {num_goal_atoms} atoms for BC-GNN predicate augmentation"
    )
    logging.info(
        f"Predicate augmentation: {'ENABLED' if CFG.bc_use_predicate_augmentation else 'DISABLED'}"
    )
    logging.info("=" * 80)

    # --------------------------------------------------
    # 5. Create BC policy
    # --------------------------------------------------
    if CFG.bc_model == "gnn":
        bc_policy = GNNBCPolicy(
            dataset=operator_transition_data,
            seed=seed,
            device=CFG.device,
            tamp_system=tamp_system,
            goal_atoms=goal_atoms,
        )
    elif CFG.bc_model == "transformer":
        raise NotImplementedError("Transformer BC not implemented yet")
    else:
        raise ValueError(f"Unknown bc_model: {CFG.bc_model}")

    # --------------------------------------------------
    # 7. Train BC model (offline)
    # --------------------------------------------------
    start_train = time.time()

    if CFG.bc_load_path and Path(CFG.bc_load_path).exists():
        logging.info(f"Loading BC model from {CFG.bc_load_path}")
        bc_policy.load(Path(CFG.bc_load_path))
    else:
        logging.info("Training BC model offline...")

        # Extract dataset identifier from planner dataset path
        dataset_id = extract_dataset_identifier(CFG.planner_dataset_path)

        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Create run directory for this training run
        # Pattern: runs/bc_{model}_{pred/nopred}_{env}_{dataset_id}_{timestamp}/
        pred_suffix = "_pred" if CFG.bc_use_predicate_augmentation else "_nopred"
        run_dir = (
            Path(CFG.rl_policy_save_dir)
            / "runs"
            / f"bc_{CFG.bc_model}{pred_suffix}_{tamp_system.name}_{dataset_id}_{timestamp}"
        )
        logging.info(f"Training artifacts will be saved to {run_dir}")

        # Train with checkpointing (saves to run_dir)
        # The train_offline will handle its own wandb logging internally
        bc_policy.train_offline(checkpoint_dir=run_dir)

        # Auto-generate final model save path if not provided
        # Save final model with metadata in the same run directory
        save_path_str = CFG.bc_save_path
        if not save_path_str:
            # Save in run directory: runs/bc_{model}_{env}_{dataset_id}/model.pt
            save_path_str = str(run_dir / "model.pt")
            logging.info(f"Auto-generated final model save path: {save_path_str}")

        save_path = Path(save_path_str)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bc_policy.save(save_path)
        logging.info(f"Saved final BC model with metadata to {save_path}")

    training_time = time.time() - start_train
    logging.info(f"BC training finished in {training_time:.2f}s")
    logging.info(
        "Training complete. Use test scripts to evaluate the trained model on different scenarios."
    )

    # Log training completion to wandb
    if wandb_run is not None:
        wandb.log(
            {
                "training_time_seconds": training_time,
                "training_time_minutes": training_time / 60,
            }
        )
        wandb.finish()
        logging.info("W&B run finished")

    return Metrics(
        success_rate=0.0,
        avg_episode_length=0.0,
        avg_reward=0.0,
        training_time=training_time,
        total_time=training_time,
    )
