"""A recomposer that first loads pre-trained predicate models, then selects a subset of
them based on compositional planning performance.

This module implements the topdown optimization approach that:
1. Load pre-trained predicate models (neural networks) from different sceanrios.
2.
"""

from __future__ import annotations

import copy
import glob
import logging
import time
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

import torch
from relational_structs import Object
from relational_structs.pddl import GroundAtom
from torch import Tensor

from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.neural_dataset import (
    OperatorTransition,
    distill_quantified_ae_vector,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    create_neural_predicate_interpreter,
    create_quantified_predicate,
    setup_predicate_net,
)
from skill_refactor.approaches.pred_learner.symbolic_search import (
    HillClimbingSearch,
    OperatorBeliefScoreFunction,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.approaches.pred_learner.utils import two2one
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    GroundAtomTrajectory,
    LiftedOperator,
    LowLevelTrajectory,
    PlannerDataset,
    Predicate,
    Segment,
    Task,
)


class IndividualPredicateRecomposer(TopDownPredicateLearner):
    """A predicate learner that first generates candidate predicates using pre-trained
    neural models, then selects a subset of them using top-down compositional planning
    performance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert (
            CFG.force_skip_pred_learning
        ), "This recomposer requires pre-trained predicate models."
        assert (
            len(CFG.scenario.split(",")) >= 2
        ), "Must have multiple scenarios for pre-trained models."
        self.all_scenarios = self._dataset.get_all_sceanrio()
        assert self.all_scenarios == CFG.scenario.split(
            ","
        ), "The scenarios in the dataset must match those specified in CFG.scenario."
        # For each scenario, how to map the effect vectors to the current composed ones.
        self.scenario_action_mapper: Dict[str, List[int]] = {}
        for scenario_str in self.all_scenarios:
            scenario_operators = self._dataset.get_appearing_operators()  # type: ignore[call-arg]
            scenario_action_names = sorted(
                [op.name for op in scenario_operators], key=lambda x: x
            )
            scenario_action_mapper = []
            for scenario_action_name in scenario_action_names:
                assert (
                    scenario_action_name in self.action_names
                ), f"Action {scenario_action_name} not found in current dataset."
                mapped_idx = self.action_names.index(scenario_action_name)
                scenario_action_mapper.append(mapped_idx)
            self.scenario_action_mapper[scenario_str] = scenario_action_mapper

    def _extract_action_names(self) -> None:
        """Extract all unique action names from trajectories in sorted order."""
        action_names_set = set()
        lifted_operator_set = set()
        for traj in self._trajectories:
            for action_result in traj.actions:
                if action_result.has_op():
                    ground_op = action_result.get_op()
                    action_names_set.add(ground_op.parent.name)
                    lifted_operator_set.add(ground_op.parent)

        # Convert to sorted list for consistent ordering
        self.action_names = sorted(list(action_names_set))
        self.operators = sorted(list(lifted_operator_set), key=lambda x: x.name)

        # Check if action names align with operators
        for i, op in enumerate(self.operators):
            if op.name != self.action_names[i]:
                raise ValueError(
                    f"Action name mismatch: {op.name} != {self.action_names[i]}"
                )

    def _generate_candidates(
        self,
    ) -> Tuple[
        List[Predicate],
        List[Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        List[Tensor],
        List[Tensor],
    ]:
        """Generate candidate predicates from different scenario pre-trained models."""
        logging.info("Starting topdown predicate learning...")
        candidate_predicates: List[Predicate] = []
        candidate_interpretations: List[
            Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ] = []
        candidate_ae_vec: List[Tensor] = []
        candidate_ae_var: List[Tensor] = []

        # Step 0: Add quantified base predicates to candidates
        for pred, data in self.init_predicate_ae_vectors.items():
            if (data["ae_vector"].sum() > 0) and (
                pred in self.perceiver.predicate_interpreters
            ):
                # This is an effect predicate
                if pred not in self.basic_predicates:
                    logging.info(
                        f"Added candidate predicate {pred.pddl_str} with AE vector {two2one(data['ae_vector'])} "
                        f"and var binding index {data['var_bind_idx']}"
                    )
                    # Quantified predicate that is not in the existing planner.
                    candidate_predicates.append(pred)
                    candidate_ae_vec.append(data["ae_vector"])
                    candidate_ae_var.append(data["var_bind_idx"])
                    candidate_interpretations.append(
                        self.perceiver.predicate_interpreters[pred]
                    )

        # Step 1: Convert ll-traj dataset to operator transition dataset
        # NOTE: We will split the dataset into scenarios here
        operator_transition_datas: List[List[OperatorTransition]] = []
        for scenario_str in self.all_scenarios:
            segment_data: List[List[Segment]] = []
            for low_level_traj, ground_atoms in self._ground_atom_dataset:
                traj_scenario = low_level_traj.train_scenario_idx
                if int(scenario_str) != traj_scenario:  # type: ignore[comparison-overlap]
                    # Skip trajectories from other scenarios
                    continue
                segments = segment_trajectory(low_level_traj, ground_atoms)
                segment_data.append(segments)
            operator_transition_data = self._create_transition_dataset(segment_data)
            operator_transition_datas.append(operator_transition_data)
            logging.info(
                f"Created {len(operator_transition_data)} training examples for scenario {scenario_str}"
            )

        # Step 2: For each predicate type, load neural model and remap ae vector
        for predicate_config in self.predicate_configures:
            var_list = [
                self._tamp_system.components.type_container[type_name]
                for type_name in predicate_config["types"]
            ]
            target_predicate = Predicate(name=predicate_config["name"], types=var_list)
            logging.info(
                f"Generating candidates for predicate {target_predicate.name} with arguments {var_list}"
            )
            num_ae_vec = (
                len(
                    glob.glob(
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_*_ae_vector.pth"
                    )
                )
                // 2
            )

            # Step 7: For each saved neural model, create a interpretation function
            for i in range(num_ae_vec):
                if "terminal" in predicate_config["name"]:
                    # Terminal predicates do not participate in predicate selection
                    continue

                # Create a fresh neural network for each interpreter to avoid closure issues
                archi = predicate_config["nn_archi"]
                individual_predicate_net = setup_predicate_net(archi=archi)

                # NOTE: The inference of this predicate is only on scenarios where it was trained
                assert (
                    "sc" in target_predicate.name
                ), "Scenario id must be in the predicate name."
                scenario_id = target_predicate.name.split("sc")[1][0]
                effect_mapper = self.scenario_action_mapper[scenario_id]
                assert (
                    scenario_id in self.all_scenarios
                ), "Scenario id in predicate name must be in the list of all scenarios."
                operator_transition_data = operator_transition_datas[
                    self.all_scenarios.index(scenario_id)
                ]

                model_path = (
                    f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_model.pth"
                )
                if not torch.cuda.is_available():
                    individual_predicate_net.load_state_dict(
                        torch.load(model_path, map_location="cpu")
                    )
                else:
                    individual_predicate_net.load_state_dict(torch.load(model_path))

                # Create a callable interpretation function
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                predicate_ins = Predicate(
                    name=target_predicate.name + f"_{i}",
                    types=target_predicate.types,
                )
                interpretation_fn = create_neural_predicate_interpreter(
                    model=individual_predicate_net,
                    predicate=predicate_ins,
                    tamp_system=self._tamp_system,
                    cls_threshold=predicate_config.get("cls_threshold", 0.5),
                    device=device,
                )

                ae_vector = torch.zeros(
                    (len(self.action_names), 2), dtype=torch.float32
                )
                ae_vector_path = (
                    f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_ae_vector.pth"
                )
                old_ae_vector = torch.load(ae_vector_path)
                assert len(old_ae_vector) == len(
                    effect_mapper
                ), "Effect mapper length does not match old AE vector length."
                for idx, mapped_idx in enumerate(effect_mapper):
                    ae_vector[mapped_idx] = old_ae_vector[idx]

                var_bind_idx_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_var_bind_idx.pth"
                var_bind_idx = torch.load(var_bind_idx_path)

                # Store the candidate interpretation function
                candidate_predicates.append(predicate_ins)
                candidate_interpretations.append(interpretation_fn)
                candidate_ae_vec.append(ae_vector)
                candidate_ae_var.append(var_bind_idx)

                # Compute Quantified Action-Effect (AE) vector
                for quantifier_str in ["ForAll", "Exist", ""]:
                    for quantified_var_id in range(predicate_ins.arity):
                        for negation in [False, True]:
                            if (quantifier_str == "") and (
                                (not negation) or (negation and quantified_var_id > 0)
                            ):
                                continue
                            quantified_ae_vec, quantified_var_idx = (
                                distill_quantified_ae_vector(
                                    individual_predicate_net,
                                    operator_transition_data,
                                    predicate_ins,
                                    var_bind_idx,
                                    self.action_to_index,
                                    cls_threshold=predicate_config.get(
                                        "cls_threshold", 0.5
                                    ),
                                    tamp_system=self._tamp_system,
                                    quantifier=quantifier_str,
                                    quantified_variable_id=quantified_var_id,
                                    negation=negation,
                                )
                            )
                            if quantified_ae_vec.sum() == 0:
                                # No effect, skip
                                continue
                            # For now just store the quantified AE vector without checking
                            quantified_pred, quantified_interp_fn = (
                                create_quantified_predicate(
                                    base_predicate=predicate_ins,
                                    base_interpreter=interpretation_fn,
                                    tamp_system=self._tamp_system,
                                    quantifier=quantifier_str,
                                    variable_id=quantified_var_id,
                                    negation=negation,
                                )
                            )
                            candidate_predicates.append(quantified_pred)
                            candidate_interpretations.append(quantified_interp_fn)
                            candidate_ae_vec.append(quantified_ae_vec)
                            candidate_ae_var.append(quantified_var_idx)

        return (
            candidate_predicates,
            candidate_interpretations,
            candidate_ae_vec,
            candidate_ae_var,
        )

    def _select_candidates(
        self,
        candidate_predicates: List[Predicate],
        candidate_interpretations: List[
            Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ],
        candidate_ae_vec: List[Tensor],
        candidate_ae_var: List[Tensor],
    ) -> Tuple[
        Dict[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        Set[LiftedOperator],
    ]:
        """Select candidates from generated candidates.

        Current implimentation is a hill-climbing search over the discrete action-effect
        matrix configurations. Note that candidate predicates appear in all possible
        trajectories to capture full pre-conditions.
        """
        # 0. Use a subset of the dataset for faster selection
        pred_selection_task_idx = []
        tasks_per_scenario = CFG.pred_search_num_trajectories // len(self.all_scenarios)
        for scenario_str in self.all_scenarios:
            scenario_task_idx = [
                t.train_task_idx
                for t in self._dataset.trajectories
                if t.train_scenario_idx == int(scenario_str)  # type: ignore[comparison-overlap]
            ]
            selected_idx = list(
                self._rng.choice(
                    scenario_task_idx,
                    size=tasks_per_scenario,
                )
            )
            pred_selection_task_idx.extend(selected_idx)
        pred_selection_trajectories: List[LowLevelTrajectory] = []
        pred_selection_ground_atom_dataset: List[GroundAtomTrajectory] = []
        pred_selection_train_tasks: List[Task] = []
        for traj, atoms in self._ground_atom_dataset:
            if traj.train_task_idx in pred_selection_task_idx:
                pred_selection_trajectories.append(traj)
                pred_selection_ground_atom_dataset.append((traj, atoms))
                pred_selection_train_tasks.append(
                    self._train_tasks[traj.train_task_idx]
                )
        pred_selection_dataset = PlannerDataset(  # type: ignore[call-arg] # pylint: disable=no-value-for-parameter
            _trajectories=pred_selection_trajectories
        )

        # 1. Add candidates to perceiver and create ground atom dataset
        # Check if any new predicates are equivalent to existing ones.
        new_pred_atoms: List[List[GroundAtomTrajectory]] = []
        filtered_candidate_predicates: List[Predicate] = []
        filtered_candidate_ae_vec: List[Tensor] = []
        filtered_candidate_ae_var: List[Tensor] = []
        for i, pred in enumerate(candidate_predicates):
            if pred in self.init_predicate_ae_vectors:
                # This predicate is already in the initial set
                # Directly add without checking equivalence
                filtered_candidate_predicates.append(pred)
                filtered_candidate_ae_vec.append(candidate_ae_vec[i])
                filtered_candidate_ae_var.append(candidate_ae_var[i])
                continue
            interp_fn = candidate_interpretations[i]
            self.perceiver.add_predicate_interpreter(
                name=pred.name,
                types=pred.types,
                interpreter=interp_fn,
            )
            predicate_ident, part_ground_atoms = (
                pred_selection_dataset.get_predicate_identifier(self.perceiver, pred)
            )
            if predicate_ident not in self.predicate_identifiers:
                new_pred_atoms.append(part_ground_atoms)
                self.predicate_identifiers.append(predicate_ident)
                logging.info(
                    f"Added candidate predicate {pred.pddl_str} \n with AE vector {two2one(candidate_ae_vec[i])} "
                    f"and var binding index {candidate_ae_var[i]}"
                )
                filtered_candidate_predicates.append(pred)
                filtered_candidate_ae_vec.append(candidate_ae_vec[i])
                filtered_candidate_ae_var.append(candidate_ae_var[i])
            else:
                # This candidate is equivalent to an existing predicate, remove it
                self.perceiver.delete_predicate_interpreter(pred)

        all_pred_augmented_atom_dataset = copy.deepcopy(
            pred_selection_ground_atom_dataset
        )
        for part_ground_atoms in new_pred_atoms:
            for i, (traj, atom_list) in enumerate(all_pred_augmented_atom_dataset):
                assert traj.train_task_idx == part_ground_atoms[i][0].train_task_idx
                assert len(atom_list) == len(part_ground_atoms[i][1])
                for t, atoms in enumerate(atom_list):  # type: ignore[assignment]
                    atom_set: Set[GroundAtom] = atoms  # type: ignore[assignment]
                    atom_set.update(part_ground_atoms[i][1][t])

        # 2. Create the score function class
        score_function = OperatorBeliefScoreFunction(
            _atom_dataset=all_pred_augmented_atom_dataset,
            _train_tasks=self._train_tasks,  # All training tasks
            _row_names=self.operators,
            metric_name="num_nodes_created",
        )

        # 3. Get the provided static predicates and effect predicate ae vectors
        provided_effect_predicates: List[Predicate] = []
        provided_prec_predicates: List[Predicate] = []
        basic_matrix: List[Tensor] = []
        basic_pred_var_idx: List[Tensor] = []

        for pred, data in self.init_predicate_ae_vectors.items():
            if data["ae_vector"].sum() > 0:
                # This is an effect predicate
                if pred not in self.basic_predicates:
                    if pred not in candidate_predicates:
                        logging.info(f"Predicate {pred.pddl_str} is abandoned")
                    continue
                logging.info(
                    f"Adding provided effect predicate {pred.pddl_str} with AE vector {two2one(data['ae_vector'])} "
                    f"and var binding index {data['var_bind_idx']}"
                )
                provided_effect_predicates.append(pred)
                basic_matrix.append(data["ae_vector"])
                basic_pred_var_idx.append(data["var_bind_idx"])
            else:
                # This is a pre-condition only predicate
                if pred in self.basic_predicates:
                    logging.info(
                        f"Adding provided pre-cond predicate {pred.pddl_str} with AE vector {two2one(data['ae_vector'])} "
                        f"and var binding index {data['var_bind_idx']}"
                    )
                    provided_prec_predicates.append(pred)

        # 4. Start hill-climbing search
        hill_climber = HillClimbingSearch(
            score_function=score_function,
            provided_effect_predicates=provided_effect_predicates,
            provided_prec_predicates=provided_prec_predicates,
            basic_matrix=basic_matrix,
            basic_pred_var_idx=basic_pred_var_idx,
            verbose=self._verbose,
        )

        s = time.time()
        final_eff_predicates, _, op_set, score = hill_climber.search(
            candidate_predicates=filtered_candidate_predicates,
            candidate_ae_vectors=filtered_candidate_ae_vec,
            candidate_var_indices=filtered_candidate_ae_var,
        )

        logging.info(
            f"Hill-climbing completed: predicate set {[p.name for p in final_eff_predicates]} "
            f"has score {score} in {time.time()-s:.2f} seconds."
        )
        # Print the operators
        for op in op_set:
            logging.info(f"Operator: \n{op}")
        final_invented_predicates = {}
        for pred, interp_fn in self.perceiver.predicate_interpreters.items():
            if (pred in final_eff_predicates) and (pred not in self.basic_predicates):
                final_invented_predicates[pred] = interp_fn

        return final_invented_predicates, op_set
