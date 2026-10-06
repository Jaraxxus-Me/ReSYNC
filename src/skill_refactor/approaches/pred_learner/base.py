"""Base class for predicate learning algorithms.

This module provides a base class for predicate learning approaches, following the
structure used in operator learning.
"""

from __future__ import annotations

import abc
import copy
import glob
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from relational_structs import LiftedAtom, Variable
from torch import Tensor

from skill_refactor.approaches.pred_learner.neural_models import (
    create_terminal_neural_predicate_interpreter,
    setup_predicate_net,
)
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    LiftedOperator,
    Object,
    Perceiver,
    PlannerDataset,
    Predicate,
)


class BasePredicateLearner(abc.ABC):
    """Base class for predicate learning approaches, following operator learner
    pattern."""

    def __init__(
        self,
        dataset: PlannerDataset,
        tamp_system: BaseRLTAMPSystem,
        predicate_configures: List[Dict],
        scenario: str,
        latest_perceiver: Optional[Perceiver] = None,
        success_traj_only: bool = True,
        verbose: bool = True,
        basic_operators: Optional[Set[LiftedOperator]] = None,
    ):
        # Store predicate configurations for each type
        self.predicate_configures = predicate_configures
        self._rng = np.random.default_rng(CFG.seed)
        if success_traj_only:
            use_dataset = dataset.success_subset()
        else:
            use_dataset = dataset
        self._dataset = use_dataset
        self._trajectories = use_dataset.trajectories
        self._tamp_system = tamp_system  # Initial TAMP system
        self.scenario = scenario
        if latest_perceiver is not None:
            # Might be from a previous learning iteration
            # with new predicates.
            self.perceiver = copy.deepcopy(latest_perceiver)
        else:
            self.perceiver = copy.deepcopy(self._tamp_system.perceiver)
        (
            self._ground_atom_dataset,
            self._train_tasks,
        ) = use_dataset.get_ground_atoms_and_tasks(self.perceiver)
        self._verbose = verbose

        # Learned state
        self.learned_predicates: Set[Predicate] = set()
        self.learning_metrics: Dict[str, Any] = {}

        # Check if new op exists
        data_op = dataset.get_appearing_operators()
        tamp_op = self._tamp_system.operators
        if len(list(tamp_op)) == len(list(data_op)):
            logging.info("The dataset does not contain any new operators")

        if basic_operators is not None:
            self.basic_operators = basic_operators
        else:
            self.basic_operators = tamp_op

        # Make directories for saving trained models
        pred_net_dir = Path(CFG.pred_net_save_dir)
        pred_net_dir.mkdir(parents=True, exist_ok=True)

    def get_current_perceiver(self) -> Perceiver:
        """Get the current perceiver from the TAMP system."""
        return self.perceiver

    def _load_invented_predicates_and_operators(
        self,
        json_path: Path,
        candidate_predicates: List[Predicate],
        candidate_interpretations: List[
            Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ],
    ) -> Tuple[
        Dict[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        Set[LiftedOperator],
    ]:
        """Load invented predicates and operators from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct predicates from candidates by matching pddl_str
        invented_pred_interpr = {}
        pred_name_to_candidate = {
            pred.pddl_str: (pred, interpr)
            for pred, interpr in zip(candidate_predicates, candidate_interpretations)
        }

        for pred_pddl_str in data["invented_predicates"]:
            if pred_pddl_str in pred_name_to_candidate:
                pred, interpr = pred_name_to_candidate[pred_pddl_str]
                invented_pred_interpr[pred] = interpr
                logging.info(f"Loaded invented predicate: {pred_pddl_str}")

        # Reconstruct operators
        op_set = set()
        for op_data in data["operators"]:
            # Create variables for operator parameters
            variables = [
                Variable(
                    f"{param_name}",
                    self._tamp_system.components.type_container[param_type],
                )
                for param_name, param_type in op_data["parameters"]
            ]

            # Reconstruct preconditions, add_effects, and delete_effects
            preconditions = set()
            add_effects = set()
            delete_effects = set()

            for pred_pddl_str, param_indices in op_data["preconditions"].items():
                found_pred: Optional[Predicate] = self._find_predicate_by_pddl_str(
                    candidate_predicates, pred_pddl_str
                )
                if found_pred is not None:
                    for var_ind_tuple in param_indices:
                        atom_args = [variables[i] for i in var_ind_tuple]
                        preconditions.add(LiftedAtom(found_pred, atom_args))

            for pred_pddl_str, param_indices in op_data["add_effects"].items():
                add_pred: Optional[Predicate] = self._find_predicate_by_pddl_str(
                    candidate_predicates, pred_pddl_str
                )
                if add_pred is not None:
                    for var_ind_tuple in param_indices:
                        atom_args = [variables[i] for i in var_ind_tuple]
                        add_effects.add(LiftedAtom(add_pred, atom_args))

            for pred_pddl_str, param_indices in op_data["delete_effects"].items():
                del_pred: Optional[Predicate] = self._find_predicate_by_pddl_str(
                    candidate_predicates, pred_pddl_str
                )
                if del_pred is not None:
                    for var_ind_tuple in param_indices:
                        atom_args = [variables[i] for i in var_ind_tuple]
                        delete_effects.add(LiftedAtom(del_pred, atom_args))

            # Create the lifted operator
            operator = LiftedOperator(
                name=op_data["name"],
                parameters=variables,
                preconditions=preconditions,
                add_effects=add_effects,
                delete_effects=delete_effects,
            )
            op_set.add(operator)
            logging.info(f"Loaded operator: {operator.pddl_str}")

        return invented_pred_interpr, op_set

    def _find_predicate_by_pddl_str(
        self, candidate_predicates: List[Predicate], pddl_str: str
    ) -> Optional[Predicate]:
        """Find predicate by its PDDL string representation."""
        # Check in basic predicates
        for pred in candidate_predicates:
            if pred.pddl_str == pddl_str:
                return pred

        for pred in self.perceiver.predicates_container.as_set():
            if pred.pddl_str == pddl_str:
                return pred
        return None

    def save_invented_predicates_and_operators(
        self,
        invented_pred_interpr: Dict[
            Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ],
        op_set: Set[LiftedOperator],
        specificed_path: Optional[Path] = None,
    ) -> None:
        """Save invented predicates and operators to JSON file."""
        # Serialize invented predicates as PDDL strings
        invented_predicates = [pred.pddl_str for pred in invented_pred_interpr.keys()]

        # Serialize operators
        operators = []
        for operator in op_set:
            # Extract parameter names and types
            parameters = [
                (param.name, param.type.name) for param in operator.parameters
            ]

            # Extract preconditions, add_effects, and delete_effects
            preconditions: Dict[str, List[List[int]]] = {}
            for atom in operator.preconditions:
                pred_pddl_str = atom.predicate.pddl_str
                param_indices = list(
                    operator.parameters.index(arg) for arg in atom.variables
                )
                if pred_pddl_str in preconditions:
                    # Cases where predicate with same name is binded to different args
                    preconditions[pred_pddl_str].append(param_indices)
                else:
                    preconditions[pred_pddl_str] = [param_indices]

            add_effects: Dict[str, List[List[int]]] = {}
            for atom in operator.add_effects:
                pred_pddl_str = atom.predicate.pddl_str
                param_indices = list(
                    operator.parameters.index(arg) for arg in atom.variables
                )
                if pred_pddl_str in add_effects:
                    add_effects[pred_pddl_str].append(param_indices)
                else:
                    add_effects[pred_pddl_str] = [param_indices]

            delete_effects: Dict[str, List[List[int]]] = {}
            for atom in operator.delete_effects:
                pred_pddl_str = atom.predicate.pddl_str
                param_indices = list(
                    operator.parameters.index(arg) for arg in atom.variables
                )
                if pred_pddl_str in delete_effects:
                    delete_effects[pred_pddl_str].append(param_indices)
                else:
                    delete_effects[pred_pddl_str] = [param_indices]

            operators.append(
                {
                    "name": operator.name,
                    "parameters": parameters,
                    "preconditions": preconditions,
                    "add_effects": add_effects,
                    "delete_effects": delete_effects,
                }
            )

        # Create the data structure to save
        data = {"invented_predicates": invented_predicates, "operators": operators}

        # Save to JSON file
        if specificed_path is not None:
            json_path = specificed_path
        else:
            json_path = Path(CFG.pred_net_save_dir) / (
                CFG.invented_pred_op_json + f"_sc{self.scenario}.json"
            )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging.info(
            f"Saved {len(invented_predicates)} invented predicates and {len(operators)} operators to {json_path}"
        )

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
        if json_path.exists():
            logging.info(
                f"Found existing invented predicates/operators file: {json_path}"
            )
            invented_pred_interpr, op_set = (
                self._load_invented_predicates_and_operators(
                    json_path, candidate_predicates, candidate_interpr
                )
            )
            if invented_pred_interpr or op_set:
                logging.info(
                    "Successfully loaded invented predicates and operators from JSON"
                )
                return invented_pred_interpr, op_set
            logging.warning(
                "Failed to load from JSON, proceeding with candidate selection"
            )

        # Step 2: Select the candidates based on some criteria
        invented_pred_interpr, op_set = self._select_candidates(
            candidate_predicates, candidate_interpr, candidate_ae_vec, candidate_ae_var
        )

        # Step 3: Save the invented predicates and operators to JSON
        self.save_invented_predicates_and_operators(invented_pred_interpr, op_set)

        return invented_pred_interpr, op_set

    @abc.abstractmethod
    def _generate_candidates(
        self,
    ) -> Tuple[
        List[Predicate],
        List[Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        List[Tensor],
        List[Tensor],
    ]:
        """Generate candidate predicates and their interpretations."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
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
        """Generate candidate predicates and their interpretations."""
        raise NotImplementedError("Override me!")

    def create_terminal_pred_interpretrs(
        self,
    ) -> Dict[str, Callable[[Tensor, List[Sequence[Object]]], Tensor]]:
        """Create terminal predicate interpreters based on configurations."""
        pred_interpreters = {}
        for predicate_config in self.predicate_configures:
            pred_name = predicate_config["name"]
            if "terminal" not in pred_name:
                continue
            var_list = [
                self._tamp_system.components.type_container[type_name]
                for type_name in predicate_config["types"]
            ]
            target_predicate = Predicate(name=predicate_config["name"], types=var_list)
            archi = predicate_config["nn_archi"]
            predicate_net = setup_predicate_net(archi=archi)
            model_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_0_model.pth"
            loaded_model = torch.load(model_path)
            if "model_state_dict" in loaded_model:
                # Loaded from a checkpoint
                model_state_dict = loaded_model["model_state_dict"]
            else:
                model_state_dict = loaded_model
            if not torch.cuda.is_available():
                predicate_net.load_state_dict(model_state_dict)
            else:
                predicate_net.load_state_dict(model_state_dict)
            # Create a callable interpretation function
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            interpretation_fn = create_terminal_neural_predicate_interpreter(
                model=predicate_net,
                predicate=target_predicate,
                tamp_system=self._tamp_system,
                cls_threshold=predicate_config.get("cls_threshold", 0.5),
                device=device,
            )
            pred_interpreters[target_predicate.name] = interpretation_fn
        return pred_interpreters
