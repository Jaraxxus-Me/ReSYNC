"""Create offline datasets by collecting demonstrations."""

import glob
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    PlannerDataset,
    RLDataset,
    Task,
)


def get_or_collect_rl_data(
    envs: ManiSkillVectorEnv,
    approach: LifelongRefApproach,
) -> RLDataset:
    """Get existing or collect new training data for RL."""
    # Check if saved data exists
    data_path = Path(os.path.join(CFG.rl_dataset_path, f"seed_{CFG.seed}"))
    data_path.mkdir(parents=True, exist_ok=True)
    if not CFG.force_collect and data_path.exists():
        logging.info(f"\nLoading existing training data from {data_path}")
        try:
            train_data = RLDataset.load(data_path)
            # Verify config matches
            logging.info(f"Loaded {len(train_data)} training states.")
            return train_data

        except Exception as e:
            logging.info(f"Error loading training data: {e}")
            logging.info("Collecting new data instead...")

    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{CFG.scenario}_task_seed{CFG.seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    # Collect new data
    train_data = approach.collect_rl_data(envs, init_states)

    # Save the collected data
    logging.info(f"\nSaving training data to {data_path}")
    train_data.save(data_path)

    return train_data


def get_or_collect_planner_data(
    envs: ManiSkillVectorEnv,
    approach: LifelongRefApproach,
    policy: Policy,
    real_scenario_name: str = "",
    num_trajectories: Optional[int] = -1,
) -> PlannerDataset:
    """Get existing or collect new training data for Planner."""
    # Check if saved data exists
    data_path = Path(os.path.join(CFG.planner_dataset_path, f"seed_{CFG.seed}"))
    data_path.mkdir(parents=True, exist_ok=True)
    if not CFG.force_collect and data_path.exists():
        logging.info(f"\nLoading existing training data from {data_path}")
        try:
            train_data = PlannerDataset.load(data_path, num_traj=num_trajectories)
            # Verify config matches
            logging.info(f"Loaded {len(train_data)} training trajectories")
            return train_data

        except Exception as e:
            logging.info(f"Error loading training data: {e}")
            logging.info("Collecting new data instead...")

    # Load initial states from task files
    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{CFG.scenario}_task_seed{CFG.seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    # Collect new data
    train_data = approach.collect_planner_data(
        envs, policy, init_states, real_env_scenario_name=real_scenario_name
    )

    successful_dataset = train_data.success_subset()
    logging.info(
        f"Filtered to {len(successful_dataset)} successful trajectories "
        f"({len(successful_dataset)/len(train_data)*100:.1f}% success rate)"
    )

    # Extract lifted operator plans from all successful trajectories
    lifted_plans = successful_dataset.get_all_operator_plans()

    logging.info(f"Found {len(lifted_plans)} unique lifted operator plans:")
    for i, plan in enumerate(lifted_plans):
        plan_names = [op.name for op in plan]
        logging.info(f"  Plan {i+1}: {' -> '.join(plan_names)}")

    # Save the lifted operator plans to a pickle file
    output_path = data_path / "lifted_operator_plans.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(lifted_plans, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the collected data
    logging.info(f"\nSaving training data to {data_path}")
    train_data.save(data_path)

    return train_data
