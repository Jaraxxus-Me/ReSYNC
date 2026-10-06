"""Predicate model evaluation utilities for PlannerDataset."""

import logging
from typing import Dict, List, Set

from relational_structs import GroundAtom, GroundOperator

from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.utils.structs import (
    LiftedOperator,
    LowLevelTrajectory,
    Perceiver,
    PlannerDataset,
    Predicate,
    Task,
)
from skill_refactor.utils.task_planning import (
    apply_operator,
    create_task_planning_heuristic,
    get_object_combinations,
    task_plan_grounding,
)


def evaluate_predicate_model(
    dataset: PlannerDataset,
    predicate: Predicate,
    operators: Set[LiftedOperator],
    perceiver: Perceiver,
    tamp_system: BaseRLTAMPSystem,
    evaluate_middle_states: bool = False,
) -> Dict[str, float]:
    """Evaluate a neural predicate model on a PlannerDataset.

    Args:
        dataset: PlannerDataset containing trajectories and ground atom data
        predicate: The predicate being evaluated
        operators: Set of lifted operators for task planning
        perceiver: Perceiver for converting states to atoms
        tamp_system: TAMP system for state-to-graph conversion
        evaluate_middle_states: Whether to include middle trajectory states

    Returns:
        Dictionary containing evaluation metrics including accuracy
    """
    logging.info(
        f"Evaluating predicate {predicate.pddl_str} on {len(dataset.trajectories)} trajectories"
    )

    # Initialize ground truth and predicted value dictionaries
    gt_values: Dict[str, Dict[str, Dict[str, bool]]] = {}
    predicted_values: Dict[str, Dict[str, Dict[str, bool]]] = {}

    # Create GroundAtom dataset (predicted ground atoms)
    ground_atom_dataset, train_tasks = dataset.get_ground_atoms_and_tasks(
        tamp_system.perceiver
    )

    # Process each trajectory
    for ll_traj, ground_atoms in ground_atom_dataset:
        traj_key = f"traj_{ll_traj.train_task_idx}"
        gt_values[traj_key] = {}
        predicted_values[traj_key] = {}

        # Extract ground truth values using task planning
        _extract_ground_truth_values(
            trajectory=ll_traj,
            task=train_tasks[ll_traj.train_task_idx],
            init_atoms=ground_atoms[0],
            predicate=predicate,
            operators=operators,
            perceiver=perceiver,
            traj_key=traj_key,
            gt_values=gt_values,
            evaluate_middle_states=evaluate_middle_states,
        )

        # Extract predicted values using neural model
        _extract_predicted_values(
            trajectory=ll_traj,
            task=train_tasks[ll_traj.train_task_idx],
            ground_atom_data=ground_atoms,
            predicate=predicate,
            traj_key=traj_key,
            predicted_values=predicted_values,
            evaluate_middle_states=evaluate_middle_states,
        )

    # Calculate and report accuracy
    metrics = _calculate_accuracy_metrics(gt_values, predicted_values)

    logging.info(f"Predicate {predicate.name} evaluation results:")
    for metric_name, value in metrics.items():
        logging.info(f"  {metric_name}: {value:.4f}")

    return metrics


def _extract_ground_truth_values(
    trajectory: LowLevelTrajectory,
    task: Task,
    init_atoms: Set[GroundAtom],
    predicate: Predicate,
    operators: Set[LiftedOperator],
    perceiver: Perceiver,
    traj_key: str,
    gt_values: Dict[str, Dict[str, Dict[str, bool]]],
    evaluate_middle_states: bool,
) -> None:
    """Extract ground truth predicate values using task planning."""
    # Get objects from perceiver
    objects = task.objects
    goal = task.goal

    # Ground operators and create heuristic
    try:
        ground_operators, _ = task_plan_grounding(
            init_atoms=init_atoms,
            objects=objects,
            operators=operators,
            allow_noops=False,
        )

        _ = create_task_planning_heuristic(
            heuristic_name="hff",
            init_atoms=init_atoms,
            goal=goal,
            ground_ops=ground_operators,
            predicates=perceiver.predicates_container.as_set(),
            objects=objects,
        )
    except Exception as e:
        logging.warning(f"Failed to create task planning components: {e}")
        return

    # Execute planning to get state sequence
    current_atoms = init_atoms.copy()
    # The operator in the planner dataset. with limited content
    current_operator = trajectory.actions[0].op
    assert current_operator is not None, "First action must have an operator"
    # The learned operator set with full content
    current_operator_learned = None
    for op in ground_operators:
        assert isinstance(op, GroundOperator)
        if (op.name == current_operator.name) and (
            op.parameters == current_operator.parameters
        ):
            current_operator_learned = op
            break
    assert (
        current_operator_learned is not None
    ), f"Operator {current_operator} not found in learned operators"
    gt_values[traj_key] = {"step_0": {}}
    for args in get_object_combinations(
        objects,
        predicate.types,
    ):
        ground_atom = GroundAtom(predicate, args)
        if ground_atom in current_atoms:
            gt_values[traj_key]["step_0"][ground_atom.pddl_str] = True
        else:
            gt_values[traj_key]["step_0"][ground_atom.pddl_str] = False

    for step in range(len(trajectory.states) - 1):
        operator = trajectory.actions[step].op
        if operator == current_operator:
            # Same operator should have the same atoms
            step_atoms = current_atoms.copy()
            if evaluate_middle_states:
                gt_values[traj_key][f"step_{step}"] = {}
                for args in get_object_combinations(
                    objects,
                    predicate.types,
                ):
                    ground_atom = GroundAtom(predicate, args)
                    if ground_atom in step_atoms:
                        gt_values[traj_key][f"step_{step}"][ground_atom.pddl_str] = True
                    else:
                        gt_values[traj_key][f"step_{step}"][
                            ground_atom.pddl_str
                        ] = False
        else:
            # Apply operator effects to get new atoms
            current_atoms_new = apply_operator(
                operator=current_operator_learned,
                current_atoms=current_atoms,
            )
            current_atoms = current_atoms_new
            #
            gt_values[traj_key][f"step_{step}"] = {}
            for args in get_object_combinations(
                objects,
                predicate.types,
            ):
                ground_atom = GroundAtom(predicate, args)
                if ground_atom in current_atoms:
                    gt_values[traj_key][f"step_{step}"][ground_atom.pddl_str] = True
                else:
                    gt_values[traj_key][f"step_{step}"][ground_atom.pddl_str] = False
            # Then update to the new operator
            current_operator = operator
            assert (
                current_operator is not None
            ), f"Action at step {step} must have an operator"
            for op in ground_operators:
                assert isinstance(op, GroundOperator)
                if (op.name == current_operator.name) and (
                    op.parameters == current_operator.parameters
                ):
                    current_operator_learned = op
                    break
            assert (
                current_operator_learned is not None
            ), f"Operator {current_operator} not found in learned operators"

    # Handle the last step
    current_atoms_new = apply_operator(
        operator=current_operator_learned,
        current_atoms=current_atoms,
    )
    current_atoms = current_atoms_new
    gt_values[traj_key][f"step_{step+1}"] = {}
    for args in get_object_combinations(
        objects,
        predicate.types,
    ):
        ground_atom = GroundAtom(predicate, args)
        if ground_atom in current_atoms:
            gt_values[traj_key][f"step_{step+1}"][ground_atom.pddl_str] = True
        else:
            gt_values[traj_key][f"step_{step+1}"][ground_atom.pddl_str] = False


def _extract_predicted_values(
    trajectory: LowLevelTrajectory,
    task: Task,
    ground_atom_data: List[Set[GroundAtom]],
    predicate: Predicate,
    traj_key: str,
    predicted_values: Dict[str, Dict[str, Dict[str, bool]]],
    evaluate_middle_states: bool,
) -> None:
    """Extract predicted predicate values using task planning."""
    objects = task.objects
    current_atoms = ground_atom_data[0]
    current_operator = trajectory.actions[0].op
    predicted_values[traj_key] = {"step_0": {}}
    for args in get_object_combinations(
        objects,
        predicate.types,
    ):
        ground_atom = GroundAtom(predicate, args)
        if ground_atom in current_atoms:
            predicted_values[traj_key]["step_0"][ground_atom.pddl_str] = True
        else:
            predicted_values[traj_key]["step_0"][ground_atom.pddl_str] = False

    for step in range(len(trajectory.states) - 1):
        operator = trajectory.actions[step].op
        current_atoms = ground_atom_data[step]
        if (operator == current_operator) and evaluate_middle_states and step > 0:
            predicted_values[traj_key][f"step_{step}"] = {}
            for args in get_object_combinations(
                objects,
                predicate.types,
            ):
                ground_atom = GroundAtom(predicate, args)
                if ground_atom in current_atoms:
                    predicted_values[traj_key][f"step_{step}"][
                        ground_atom.pddl_str
                    ] = True
                else:
                    predicted_values[traj_key][f"step_{step}"][
                        ground_atom.pddl_str
                    ] = False
        else:
            if operator != current_operator:
                current_operator = operator
                predicted_values[traj_key][f"step_{step}"] = {}
                for args in get_object_combinations(
                    objects,
                    predicate.types,
                ):
                    ground_atom = GroundAtom(predicate, args)
                    if ground_atom in current_atoms:
                        predicted_values[traj_key][f"step_{step}"][
                            ground_atom.pddl_str
                        ] = True
                    else:
                        predicted_values[traj_key][f"step_{step}"][
                            ground_atom.pddl_str
                        ] = False

    # Handle the last step
    current_atoms = ground_atom_data[-1]
    predicted_values[traj_key][f"step_{step+1}"] = {}
    for args in get_object_combinations(
        objects,
        predicate.types,
    ):
        ground_atom = GroundAtom(predicate, args)
        if ground_atom in current_atoms:
            predicted_values[traj_key][f"step_{step+1}"][ground_atom.pddl_str] = True
        else:
            predicted_values[traj_key][f"step_{step+1}"][ground_atom.pddl_str] = False


def _calculate_accuracy_metrics(
    gt_values: Dict[str, Dict[str, Dict[str, bool]]],
    predicted_values: Dict[str, Dict[str, Dict[str, bool]]],
) -> Dict[str, float]:
    """Calculate accuracy metrics comparing ground truth and predicted values."""
    total_predictions = 0
    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for traj_key in gt_values:
        if traj_key not in predicted_values:
            continue

        for step_key in gt_values[traj_key]:
            if step_key not in predicted_values[traj_key]:
                continue

            for grounding_key in gt_values[traj_key][step_key]:
                if grounding_key not in predicted_values[traj_key][step_key]:
                    continue

                gt_value = gt_values[traj_key][step_key][grounding_key]
                pred_value = predicted_values[traj_key][step_key][grounding_key]

                total_predictions += 1

                if gt_value == pred_value:
                    correct_predictions += 1

                if gt_value and pred_value:
                    true_positives += 1
                elif not gt_value and pred_value:
                    false_positives += 1
                elif not gt_value and not pred_value:
                    true_negatives += 1
                elif gt_value and not pred_value:
                    false_negatives += 1

    if total_predictions == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    accuracy = correct_predictions / total_predictions

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
    }
