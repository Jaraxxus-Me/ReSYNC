"""Downsample a PlannerDataset by keeping operator boundaries and sampling intermediate
states.

For each trajectory:
- Keep the initial state and end state where operators change (operator boundaries)
- Between operator boundaries, uniformly sample a specified number of intermediate states
- This reduces dataset size while preserving operator structure for predicate learning

Usage:
    python scripts/downsample_planner_dataset.py \
        --input_path <input_dataset_path> \
        --output_path <output_dataset_path> \
        --num_intermediate_samples <num_samples>
"""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

from relational_structs import (
    Object,
    Variable,
)

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.utils.structs import (
    ApproachStepResult,
    LiftedOperator,
    LowLevelTrajectory,
    PlannerDataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def downsample_trajectory(
    traj: LowLevelTrajectory,
    num_intermediate_samples: int,
    name_to_operator: dict[str, "LiftedOperator"],
    name_to_objects: Optional[dict[str, "Object"]] = None,
) -> LowLevelTrajectory:
    """Downsample a trajectory by keeping operator boundaries and sampling intermediate
    states.

    Args:
        traj: Original trajectory
        num_intermediate_samples: Number of intermediate states to sample within each operator segment

    Returns:
        Downsampled trajectory with operator boundaries preserved

    Note:
        The downsampled trajectory maintains operator annotations and the structure
        len(states) = len(actions) + 1, but consecutive states may not be directly
        connected by their associated actions in the original environment dynamics.
        This is acceptable for predicate learning where we primarily care about
        the sequence of symbolic states.
    """
    if not traj.has_complete_operator_annotations():
        raise ValueError(
            "Trajectory must have complete operator annotations for downsampling"
        )

    # Find operator change points (action indices where operator changes)
    operator_change_indices = [0]  # Always start at first action

    if len(traj.actions) > 0:
        current_op = traj.actions[0].get_op().parent
        for i in range(1, len(traj.actions)):
            if traj.actions[i].get_op().parent != current_op:
                operator_change_indices.append(i)
                current_op = traj.actions[i].get_op().parent

    # Build list of action indices to keep
    action_indices_to_keep: List[int] = []

    for seg_idx in range(len(operator_change_indices)):
        seg_start = operator_change_indices[seg_idx]
        seg_end = (
            operator_change_indices[seg_idx + 1]
            if seg_idx + 1 < len(operator_change_indices)
            else len(traj.actions)
        )

        segment_length = seg_end - seg_start

        # Determine which indices to keep in this segment
        if segment_length <= num_intermediate_samples + 2:
            # Segment is short enough, keep all actions
            indices = list(range(seg_start, seg_end))
        else:
            # Sample: keep first, last, and uniformly sample middle
            if num_intermediate_samples == 0:
                # Only keep first and last actions of the segment
                indices = [seg_start, seg_end - 1]
            else:
                # Sample intermediate actions uniformly
                middle_start = seg_start + 1
                middle_end = seg_end - 1
                middle_length = middle_end - middle_start

                if middle_length <= num_intermediate_samples:
                    # Not enough middle actions, keep all
                    indices = list(range(seg_start, seg_end))
                else:
                    # Uniformly sample from middle
                    step = middle_length / num_intermediate_samples
                    middle_indices = [
                        middle_start + int(i * step)
                        for i in range(num_intermediate_samples)
                    ]
                    indices = [seg_start] + middle_indices + [seg_end - 1]

        action_indices_to_keep.extend(indices)

    # Build downsampled trajectory
    new_states = [traj.states[0]]  # Always include initial state
    new_actions = []

    for action_idx in action_indices_to_keep:
        old_op = traj.actions[action_idx].op
        assert old_op is not None, "Action must have an associated operator"
        old_objects = old_op.parameters
        assert name_to_objects is not None, "name_to_objects must be provided"
        if old_op.name in name_to_operator:
            new_lifted_op = name_to_operator[old_op.name]
            new_objects = [name_to_objects[obj.name] for obj in old_objects]
            new_op = new_lifted_op.ground(tuple(new_objects))
        else:
            new_objects = [name_to_objects[obj.name] for obj in old_objects]
            new_variables = [
                Variable(name=f"?{obj.type.name}", type=obj.type) for obj in new_objects
            ]
            new_lifted_op = LiftedOperator(
                name=old_op.name,
                parameters=new_variables,
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
            )
            new_op = new_lifted_op.ground(tuple(new_objects))

        new_actions.append(
            ApproachStepResult(
                traj.actions[action_idx].action,
                op=new_op,
            )
        )
        new_states.append(traj.states[action_idx + 1])

    return LowLevelTrajectory(
        _states=new_states,
        _actions=new_actions,
        _train_scenario_idx=traj.train_scenario_idx,
        _success=traj.success,
        _train_task_idx=traj.train_task_idx,
    )


def downsample_dataset(
    dataset: PlannerDataset,
    num_intermediate_samples: int,
) -> PlannerDataset:
    """Downsample all trajectories in a PlannerDataset.

    Args:
        dataset: Original dataset
        num_intermediate_samples: Number of intermediate states to sample per operator segment

    Returns:
        New PlannerDataset with downsampled trajectories
    """
    logging.info(
        "Downsampling dataset with %d trajectories, "
        "keeping %d intermediate samples per operator segment",
        len(dataset),
        num_intermediate_samples,
    )

    downsampled_trajectories = []
    downsampled_annotations: Optional[List[Any]] = (
        [] if dataset.has_annotations else None
    )
    downsampled_scenario_indices = []
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    latest_op_set = tamp_system.operators
    name_to_operator = {op.name: op for op in latest_op_set}
    name_to_objects = tamp_system.components.perceiver.objects.as_dict()

    original_total_states = 0
    downsampled_total_states = 0

    for idx, traj in enumerate(dataset.trajectories):
        original_total_states += len(traj.states)

        try:
            traj_plan = traj.get_lifted_sequence()
            op_next_push = None
            curr_is_push = False
            for lifted_op in traj_plan:
                if lifted_op.name == "Push":
                    curr_is_push = True
                if curr_is_push and lifted_op.name != "Push":
                    op_next_push = lifted_op
                    curr_is_push = False
            if op_next_push is not None:
                if op_next_push.name != "GoToPlaceBox":
                    logging.info(
                        "Skipping trajectory %d: next op after Push is not GoToPlaceBox",
                        idx,
                    )
                    continue

            downsampled_traj = downsample_trajectory(
                traj, num_intermediate_samples, name_to_operator, name_to_objects
            )
            downsampled_trajectories.append(downsampled_traj)
            downsampled_scenario_indices.append(dataset.training_scenario_indices[idx])

            if downsampled_annotations is not None:
                downsampled_annotations.append(dataset.annotations[idx])

            downsampled_total_states += len(downsampled_traj.states)

        except ValueError as e:
            logging.warning("Skipping trajectory %d: %s", idx, str(e))

    compression_ratio = (
        (1 - downsampled_total_states / original_total_states) * 100
        if original_total_states > 0
        else 0
    )

    logging.info(
        "Downsampling complete: %d -> %d trajectories",
        len(dataset),
        len(downsampled_trajectories),
    )
    logging.info(
        "Total states: %d -> %d (%.1f%% reduction)",
        original_total_states,
        downsampled_total_states,
        compression_ratio,
    )

    return PlannerDataset(
        _trajectories=downsampled_trajectories,
        _annotations=downsampled_annotations,
        _training_scenario_idx=downsampled_scenario_indices,
    )


def main() -> None:
    """Main function to downsample a planner dataset."""
    parser = argparse.ArgumentParser(
        description="Downsample PlannerDataset by keeping operator boundaries and "
        "sampling intermediate states"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input PlannerDataset directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output downsampled PlannerDataset directory",
    )
    parser.add_argument(
        "--num_intermediate_samples",
        type=int,
        default=5,
        help="Number of intermediate states to sample between operator boundaries (default: 5)",
    )
    parser.add_argument(
        "--num_traj",
        type=int,
        default=-1,
        help="Number of trajectories to load from input (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--no_filter_incomplete",
        action="store_true",
        help="Do not filter out trajectories with incomplete operator annotations",
    )

    args = parser.parse_args()
    register_all_environments()
    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "dreaming_noise_base_var": 0.0,
        f"can_blocking_target1": False,
        f"can_blocking_target2": True,
        "num_envs": 1,
        "device": "cuda:0",
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Validate input path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Load dataset
    logging.info("Loading dataset from %s", input_path)
    dataset = PlannerDataset.load(
        input_path,
        num_traj=args.num_traj,
        filter_incomplete_operators=not args.no_filter_incomplete,
    )
    logging.info("Loaded %d trajectories", len(dataset))

    # Downsample dataset
    downsampled_dataset = downsample_dataset(dataset, args.num_intermediate_samples)

    # Save downsampled dataset
    logging.info("Saving downsampled dataset to %s", output_path)
    downsampled_dataset.save(output_path)
    logging.info("Done!")


if __name__ == "__main__":
    main()
