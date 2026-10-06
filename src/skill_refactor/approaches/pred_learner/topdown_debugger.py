"""Top-down predicate learning implementation for neural predicate invention.

This module implements the topdown optimization approach that:
1. Generate some AE vectors (by enumeration, LLMs, random, or exhaustive).
2. For each AE vector, create a neural learning dataset and train a neural
   predicate model to fit the data.
3. Use a search procedure (e.g., hill-climbing) to select the best subset
   of invented predicates based on their AE vectors.
"""

from __future__ import annotations

import copy
import glob
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from relational_structs import Object
from relational_structs.pddl import GroundAtom
from torch import Tensor

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
    Task,
)


class TopDownPredicateDebugger(TopDownPredicateLearner):
    """Top-down predicate learner, where the effect vectors are generated without any
    low-level feedback (fixed or exhaustive generation)."""

    # DEBUG: Invent with loaded json
    def invent(
        self,
    ) -> Tuple[
        Dict[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        Set[LiftedOperator],
    ]:
        """Main learning pipeline.

        Args:
            dataset: Training dataset
        """
        logging.info("Starting predicate learning...")

        # Step 1: Generate candidates and their interpretations
        (
            candidate_predicates,
            candidate_interpr,
            candidate_ae_vec,
            candidate_ae_var,
        ) = self._generate_candidates()

        # Step 1.5: Check if already existing JSON file exists, if so, load and return
        json_path = Path(CFG.pred_net_save_dir) / (
            CFG.invented_pred_op_json + f"_sc{self.scenario}.json"
        )
        if not json_path.exists():
            raise FileNotFoundError(
                f"TopDownPredicateDebugger requires existing JSON file: {json_path}"
            )

        logging.info(f"Found existing invented predicates/operators file: {json_path}")
        invented_pred_interpr, _ = self._load_invented_predicates_and_operators(
            json_path, candidate_predicates, candidate_interpr
        )
        filtered_candidate_predicates = []
        filtered_candidate_ae_vec = []
        filtered_candidate_ae_var = []
        filtered_candidate_interpr = []
        for i, pred in enumerate(candidate_predicates):
            if pred in invented_pred_interpr:
                filtered_candidate_predicates.append(pred)
                filtered_candidate_ae_vec.append(candidate_ae_vec[i])
                filtered_candidate_ae_var.append(candidate_ae_var[i])
                filtered_candidate_interpr.append(candidate_interpr[i])

        # Step 2: Select the candidates based on loaded predicates (debug)
        invented_pred_interpr, op_set = self._select_candidates(
            filtered_candidate_predicates,
            filtered_candidate_interpr,
            filtered_candidate_ae_vec,
            filtered_candidate_ae_var,
        )

        # Step 3: Save the invented predicates and operators to JSON
        new_json_path = (
            Path(CFG.pred_net_save_dir)
            / "op_refinement_invented_predicates_operators.json"
        )
        self.save_invented_predicates_and_operators(
            invented_pred_interpr, op_set, new_json_path
        )

        return invented_pred_interpr, op_set

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
        matrix configurations.
        """
        # 0. Use a subset of the dataset for faster selection
        # pred_selection_task_idx = [t.train_task_idx for t in self._dataset.trajectories]
        pred_selection_task_idx = list(
            self._rng.choice(
                [t.train_task_idx for t in self._dataset.trajectories],
                size=CFG.pred_search_num_trajectories,
            )
        )
        pred_selection_trajectories: List[LowLevelTrajectory] = []
        pred_selection_ground_atom_dataset: List[GroundAtomTrajectory] = []
        pred_selection_train_tasks: List[Task] = []
        for traj, atoms in self._ground_atom_dataset:
            if traj.train_task_idx in pred_selection_task_idx:
                pred_selection_trajectories.append(traj)
                pred_selection_ground_atom_dataset.append((traj, atoms))
                pred_selection_train_tasks.append(
                    [
                        t
                        for t in self._train_tasks
                        if t.train_task_idx == traj.train_task_idx
                    ][0]
                )
        train_traj_scenarios = [
            traj.train_scenario_idx for traj in pred_selection_trajectories
        ]
        pred_selection_dataset = PlannerDataset(
            pred_selection_trajectories, train_traj_scenarios
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
                base_types=pred.base_types,
            )
            predicate_ident, part_ground_atoms = (
                pred_selection_dataset.get_predicate_identifier(self.perceiver, pred)
            )
            if predicate_ident not in self.predicate_identifiers:
                # Only add pred_selection part of the ground atoms
                new_pred_atoms.append(
                    [
                        ground_atom_traj
                        for ground_atom_traj in part_ground_atoms
                        if ground_atom_traj[0].train_task_idx in pred_selection_task_idx
                    ]
                )
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
            basic_operators=self.basic_operators,
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
