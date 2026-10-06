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

import torch
from relational_structs import Object
from relational_structs.pddl import GroundAtom
from torch import Tensor

from skill_refactor.approaches.operator_learner import learn_operator_from_data
from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.base import BasePredicateLearner
from skill_refactor.approaches.pred_learner.neural_dataset import (
    OperatorTransition,
    create_train_val_dataloaders,
    distill_ae_vector,
    distill_quantified_ae_vector,
    generate_in_out_predicate_data,
    generate_in_out_predicate_data_terminal,
    sample_middle_state_graphs,
    train_predicate_model,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    create_neural_predicate_interpreter,
    create_quantified_predicate,
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.approaches.pred_learner.symbolic_search import (
    HillClimbingSearch,
    OperatorBeliefScoreFunction,
    filter_operator_preconditions,
    get_ae_generator_by_name,
)
from skill_refactor.approaches.pred_learner.utils import operator_to_ae_vector, two2one
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    FrozenSet,
    GroundAtomTrajectory,
    LiftedOperator,
    LowLevelTrajectory,
    PlannerDataset,
    Predicate,
    Segment,
    Task,
)


class TopDownPredicateLearner(BasePredicateLearner):
    """Top-down predicate learner, where the effect vectors are generated without any
    low-level feedback (fixed or exhaustive generation)."""

    def __init__(self, quantify_basic_predicates: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Track all action names that appear in trajectories (maintain consistent order)
        self.action_names: List[str] = []
        self.operators: List[LiftedOperator] = []
        self._extract_action_names()
        self.action_to_index = self.get_action_to_index_mapping()
        self.basic_predicates: set[Predicate] = (
            self.perceiver.predicates_container.as_set()
        )
        self.predicate_identifiers: List[FrozenSet] = []
        for pred in self.basic_predicates:
            pred_id, _ = self._dataset.get_predicate_identifier(self.perceiver, pred)
            self.predicate_identifiers.append(pred_id)

        # Optionally generate quantified predicates and their AE vectors
        if quantify_basic_predicates:
            desired_predicate_atoms: List[List[GroundAtomTrajectory]] = []
            new_predicates: List[Predicate] = []
            inital_predicate_interp = copy.deepcopy(
                self.perceiver.predicate_interpreters
            )
            for (
                pred,
                interpr_fn,
            ) in inital_predicate_interp.items():
                if pred.arity not in [1, 2]:
                    continue
                for quantified_var_id in range(pred.arity):
                    for quantifier in ["Exist", "ForAll", ""]:
                        for negation in [False, True]:
                            if (quantifier == "") and (
                                (not negation) or (negation and quantified_var_id > 0)
                            ):
                                continue
                            quantified_pred, quantified_interpr = (
                                create_quantified_predicate(
                                    base_predicate=pred,
                                    base_interpreter=interpr_fn,
                                    tamp_system=self._tamp_system,
                                    quantifier=quantifier,
                                    variable_id=quantified_var_id,
                                    negation=negation,
                                )
                            )
                            self.perceiver.add_predicate_interpreter(
                                name=quantified_pred.name,
                                types=quantified_pred.types,
                                interpreter=quantified_interpr,
                                base_types=pred.types,
                            )
                            # Check if this quantified predicate is equivalent to any existing predicate
                            identifier, part_ground_atoms = (
                                self._dataset.get_predicate_identifier(
                                    self.perceiver, quantified_pred
                                )
                            )
                            if identifier not in self.predicate_identifiers:
                                desired_predicate_atoms.append(part_ground_atoms)
                                self.predicate_identifiers.append(identifier)
                                new_predicates.append(quantified_pred)
                            else:
                                self.perceiver.delete_predicate_interpreter(
                                    quantified_pred
                                )

            # Get the provided static predicates and effect predicate ae vectors
            augmented_ground_atom_dataset = copy.deepcopy(self._ground_atom_dataset)
            for part_ground_atoms in desired_predicate_atoms:
                for i, (traj, atom_list) in enumerate(augmented_ground_atom_dataset):
                    assert traj.train_task_idx == part_ground_atoms[i][0].train_task_idx
                    assert len(atom_list) == len(part_ground_atoms[i][1])
                    for t, atoms in enumerate(atom_list):  # type: ignore[assignment]
                        atom_set: Set[GroundAtom] = atoms  # type: ignore[assignment]
                        atom_set.update(part_ground_atoms[i][1][t])

            # Augment the ground atom dataset with the new quantified predicates
            self._ground_atom_dataset = augmented_ground_atom_dataset

        init_op, _, _, _ = learn_operator_from_data(
            "clustering",
            self._trajectories,
            self._train_tasks,
            self.perceiver.predicates_container.as_set(),
            self._dataset.get_appearing_operators(),
            self._ground_atom_dataset,
        )
        self.init_predicate_ae_vectors = operator_to_ae_vector(
            init_op, self.action_names
        )

        if self._verbose:
            logging.info(
                f"Initialized TopDownPredicateLearner with {len(self.action_names)} actions"
            )
            logging.info(f"Action names: {self.action_names}")
            for pred, data in self.init_predicate_ae_vectors.items():
                logging.info(
                    f"Predicate {pred.pddl_str} has AE vector {two2one(data['ae_vector'])} "
                    f"and var binding index {data['var_bind_idx']}"
                )

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

    def get_action_to_index_mapping(self) -> Dict[str, int]:
        """Get mapping from action names to indices."""
        return {name: i for i, name in enumerate(self.action_names)}

    def _generate_candidates(
        self,
    ) -> Tuple[
        List[Predicate],
        List[Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        List[Tensor],
        List[Tensor],
    ]:
        """Generate neural learning datasets for all predicate candidates using top-down
        supervision-based optimization."""
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
        segment_data: List[List[Segment]] = []
        for low_level_traj, ground_atoms in self._ground_atom_dataset:
            segments = segment_trajectory(low_level_traj, ground_atoms)
            segment_data.append(segments)
        operator_transition_data = self._create_transition_dataset(segment_data)
        logging.info(f"Created {len(operator_transition_data)} training examples")

        # Step 2: For each predicate type, generate ae vectors and learn neural model.
        for predicate_config in self.predicate_configures:
            var_list = [
                self._tamp_system.components.type_container[type_name]
                for type_name in predicate_config["types"]
            ]
            target_predicate = Predicate(name=predicate_config["name"], types=var_list)
            logging.info(
                f"Generating candidates for predicate {target_predicate.name} with arguments {var_list}"
            )
            # Step 3.1: Create the neural model based on the specified architecture
            archi = predicate_config["nn_archi"]
            predicate_net = setup_predicate_net(archi=archi)
            # Step 3.2: Create the optimizer and learning rate scheduler
            optimizer_config = predicate_config["optimizer"]
            scheduler_config = predicate_config.get("lr_scheduler", None)
            optimizer, scheduler = setup_predicate_optimizer(
                model=predicate_net,
                opti_config=optimizer_config,
                lr_scheduler_config=scheduler_config,
            )

            # Step 4: For each generated AE vector, create a neural learning dataset and targets
            # Initialize AE vector generator using factory function
            ae_generator_kwargs = predicate_config.get("ae_generator", {})
            ae_generator_name = ae_generator_kwargs.get("name", "base")
            assert ae_generator_name in [
                "fixed",
                "exhaustive",
            ]
            ae_generator_class = get_ae_generator_by_name(ae_generator_name)
            generator_kwargs = ae_generator_kwargs.get("kwargs", {})
            ae_vector_generator = ae_generator_class(
                num_operators=len(self.action_names),
                arity=target_predicate.arity,
                **generator_kwargs,
            )
            # Step 5: Use configured AE vector generator
            logging.info(
                f"Will generate {ae_vector_generator.count_total_vectors()} AE vector candidates"
            )
            num_ae_vec = 0
            for ae_vector, var_bind_idx in ae_vector_generator.generate_all_vectors():
                if predicate_config.get("skip", False) or CFG.force_skip_pred_learning:
                    num_ae_vec = (
                        len(
                            glob.glob(
                                f"{CFG.pred_net_save_dir}/{target_predicate.name}_*_ae_vector.pth"
                            )
                        )
                        // 2
                    )
                    logging.info(
                        f"Skipping predicate net learning {predicate_config['name']}, already have {num_ae_vec} AE vectors"
                    )
                    break
                logging.info(
                    f"Processing AE vector with var binding: {ae_vector} | {var_bind_idx}"
                )

                # Step 5.1: Fresh the neural model based on the specified architecture
                archi = predicate_config["nn_archi"]
                predicate_net = setup_predicate_net(archi=archi)
                # Step 5.2: Create the optimizer and learning rate scheduler
                optimizer_config = predicate_config["optimizer"]
                scheduler_config = predicate_config.get("lr_scheduler", None)
                optimizer, scheduler = setup_predicate_optimizer(
                    model=predicate_net,
                    opti_config=optimizer_config,
                    lr_scheduler_config=scheduler_config,
                )

                if "terminal" in predicate_config["name"]:
                    (
                        input_data_list,
                        input_data_list_,
                        input_middle_data_list,
                        target_data_list,
                        target_data_list_,
                    ) = generate_in_out_predicate_data_terminal(
                        operator_transition_data,
                        target_predicate,
                        ae_vector,
                        self.action_to_index.copy(),
                    )
                else:
                    (
                        input_data_list,
                        input_data_list_,
                        input_middle_data_list,
                        target_data_list,
                        target_data_list_,
                    ) = generate_in_out_predicate_data(
                        operator_transition_data,
                        target_predicate,
                        ae_vector,
                        var_bind_idx,
                        self.action_to_index.copy(),
                    )

                # Create train/val data loaders
                batch_size = predicate_config.get("batch_size", 16)
                train_ratio = predicate_config.get("train_ratio", 0.8)

                train_loader, val_loader = create_train_val_dataloaders(
                    input_data_list=input_data_list,
                    target_data_list=target_data_list,
                    input_data_list_=input_data_list_,
                    target_data_list_=target_data_list_,
                    input_middle_data_list=input_middle_data_list,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    shuffle_train=True,
                )

                # Step 6: Train the neural model on this dataset
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                num_epochs = predicate_config.get("num_epochs", 50)
                val_freq = predicate_config.get("val_freq", 5)
                pre_trained_dict_path = predicate_config.get(
                    "pre_trained_dict_path", None
                )
                if pre_trained_dict_path is not None:
                    logging.info(
                        f"Fine-tuning pre-trained model weights from {pre_trained_dict_path}"
                    )
                    predicate_net.load_state_dict(torch.load(pre_trained_dict_path))

                logging.info(
                    f"Training neural model for {num_epochs} epochs on device {device}"
                )

                best_weights, best_val_loss, best_train_loss = train_predicate_model(
                    model=predicate_net,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    optimizer=optimizer,
                    super_label=CFG.super_label,
                    num_epochs=num_epochs,
                    device=device,
                    scheduler=scheduler,
                    val_freq=val_freq,  # Validate every 5 epochs
                )

                if pre_trained_dict_path is not None:
                    logging.info(
                        f"Fine-tuning completed. Best validation loss: {best_val_loss:.5f}"
                    )
                    logging.info(
                        f"Saving fine-tuned model weights to {pre_trained_dict_path}"
                    )
                    # Save the trained model
                    torch.save(
                        best_weights,
                        pre_trained_dict_path,
                    )
                    continue

                # Load best weights into the model
                if "terminal" in predicate_config["name"]:
                    # Terminal predicates do not participate in AE distillation
                    learned_ae_vector = torch.zeros_like(ae_vector)
                else:
                    learned_ae_vector, _ = distill_ae_vector(
                        predicate_net,
                        operator_transition_data,
                        target_predicate,
                        var_bind_idx,
                        self.action_to_index,
                        cls_threshold=predicate_config.get("cls_threshold", 0.5),
                    )
                if best_val_loss < predicate_config.get(
                    "val_loss_thresh", 0.05
                ) and best_train_loss < predicate_config.get("train_loss_thresh", 0.05):
                    logging.info(
                        f"Best validation loss {best_val_loss:.5f} and Best training loss {best_train_loss:.5f} is below threshold, saving model"
                    )
                    predicate_net.load_state_dict(best_weights)
                    # Save the trained model
                    torch.save(
                        predicate_net.state_dict(),
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{num_ae_vec}_model.pth",
                    )
                    torch.save(
                        ae_vector,
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{num_ae_vec}_ae_vector.pth",
                    )
                    torch.save(
                        var_bind_idx,
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{num_ae_vec}_var_bind_idx.pth",
                    )
                    # Store the learned predicate and its interpretation function
                    torch.save(
                        learned_ae_vector,
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{num_ae_vec}_learned_ae_vector.pth",
                    )
                    num_ae_vec += 1
                else:
                    logging.info(
                        f"Best validation loss {best_val_loss:.5f} did not meet threshold"
                    )

            # Step 7: For each saved neural model, create a interpretation function
            for i in range(num_ae_vec):
                if "terminal" in predicate_config["name"]:
                    # Terminal predicates do not participate in predicate selection
                    continue
                # Create a fresh neural network for each interpreter to avoid closure issues
                archi = predicate_config["nn_archi"]
                individual_predicate_net = setup_predicate_net(archi=archi)

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
                ae_vector_path = (
                    f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_ae_vector.pth"
                )
                ae_vector = torch.load(ae_vector_path)
                var_bind_idx_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_var_bind_idx.pth"
                var_bind_idx = torch.load(var_bind_idx_path)

                dim_ae_vector = len(self.action_names)
                if ae_vector.shape[0] != dim_ae_vector:
                    # The predicate will be used in a different domain with different
                    # number of actions, need to re-distill the AE vector
                    logging.info(
                        f"AE vector dimension {ae_vector.shape[0]} does not match current action space \
                        {dim_ae_vector}, re-distilling AE vector"
                    )
                    learned_ae_vector, _ = distill_ae_vector(
                        individual_predicate_net,
                        operator_transition_data,
                        target_predicate,
                        var_bind_idx,
                        self.action_to_index,
                        cls_threshold=predicate_config.get("cls_threshold", 0.5),
                    )
                    ae_vector = learned_ae_vector
                    logging.info(
                        f"Re-distilled AE vector: {two2one(learned_ae_vector)}"
                    )

                # Store the candidate interpretation function
                candidate_predicates.append(predicate_ins)
                candidate_interpretations.append(interpretation_fn)
                candidate_ae_vec.append(ae_vector)
                candidate_ae_var.append(var_bind_idx)

                # Compute Quantified Action-Effect (AE) vector
                for quantifier_str in CFG.pred_quantifiers:
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

    # DEBUG: Invent with loaded json
    # def invent(
    #     self,
    # ) -> Tuple[
    #     Dict[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]],
    #     Set[LiftedOperator],
    # ]:
    #     """Main learning pipeline.

    #     Args:
    #         dataset: Training dataset
    #     """
    #     logging.info("Starting predicate learning...")

    #     # Step 1: Generate candidates and their interpretations
    #     (
    #         candidate_predicates,
    #         candidate_interpr,
    #         candidate_ae_vec,
    #         candidate_ae_var,
    #     ) = self._generate_candidates()

    #     # Step 1.5: Check if already existing JSON file exists, if so, load and return
    #     json_path = Path(CFG.pred_net_save_dir) / (
    #         CFG.invented_pred_op_json + f"_sc{self.scenario}.json"
    #     )
    #     if json_path.exists():
    #         logging.info(
    #             f"Found existing invented predicates/operators file: {json_path}"
    #         )
    #         invented_pred_interpr, op_set = (
    #             self._load_invented_predicates_and_operators(
    #                 json_path, candidate_predicates, candidate_interpr
    #             )
    #         )
    #         filtered_candidate_predicates = []
    #         filtered_candidate_ae_vec = []
    #         filtered_candidate_ae_var = []
    #         filtered_candidate_interpr = []
    #         for i, pred in enumerate(candidate_predicates):
    #             if pred in invented_pred_interpr:
    #                 filtered_candidate_predicates.append(pred)
    #                 filtered_candidate_ae_vec.append(candidate_ae_vec[i])
    #                 filtered_candidate_ae_var.append(candidate_ae_var[i])
    #                 filtered_candidate_interpr.append(candidate_interpr[i])

    #         # Step 2: Select the candidates based on loaded predicates (debug)
    #         invented_pred_interpr, op_set = self._select_candidates(
    #             filtered_candidate_predicates, filtered_candidate_interpr, filtered_candidate_ae_vec,
    #             filtered_candidate_ae_var
    #         )

    #     # Step 3: Save the invented predicates and operators to JSON
    #     # new_json_path = Path(CFG.pred_net_save_dir) / 'op_refinement_invented_predicates_operators.json'
    #     # self._save_invented_predicates_and_operators(invented_pred_interpr, op_set, new_json_path)

    #     return invented_pred_interpr, op_set

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

        # This is a pre-condition only predicate
        # pylint: disable=undefined-loop-variable
        for pred in list(self.basic_predicates):
            if pred not in self.init_predicate_ae_vectors:
                logging.info(
                    f"Adding provided pre-cond predicate {pred.pddl_str} with AE vector {two2one(data['ae_vector'])} "
                    f"and var binding index {data['var_bind_idx']}"
                )
                provided_prec_predicates.append(pred)
        # pylint: enable=undefined-loop-variable

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

    def _create_transition_dataset(
        self, segment_data: List[List[Segment]]
    ) -> List[OperatorTransition]:
        """Convert segmented trajectory data into operator transition dataset."""
        transition_data = []

        for segment_traj in segment_data:
            for segment in segment_traj:
                state_tensor = segment.states[0]  # the segment's initial state
                state_graph = self._tamp_system.state_to_graph(
                    state_tensor.unsqueeze(0), training=True
                )[0]
                atoms = segment.init_atoms  # the segment's initial atoms
                state_tensor_ = segment.states[-1]  # the segment's final state
                state_graph_ = self._tamp_system.state_to_graph(
                    state_tensor_.unsqueeze(0), training=True
                )[0]
                atoms_ = segment.final_atoms
                operator = segment.actions[
                    0
                ].get_op()  # the operator applied in this segment
                middle_state_graphs = sample_middle_state_graphs(
                    segment, self._tamp_system, CFG.num_middle_states
                )

                transition = OperatorTransition(
                    pre_state_graph=state_graph,
                    pre_atoms=atoms,
                    post_state_graph=state_graph_,
                    post_atoms=atoms_,
                    operator=operator,
                    middle_state_graphs=middle_state_graphs,
                )
                transition_data.append(transition)

        return transition_data

    def filter_preconditions(
        self,
        op_set: Set[LiftedOperator],
        pred_interpr: Dict[
            Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ],
        dataset: PlannerDataset,
        num_traj_per_scenario: int = 5,
        unsolvable_threshold: float = 0.3,
    ) -> Set[LiftedOperator]:
        """Filter preconditions of the given operators based on the provided predicates
        and dataset.

        Args:
            op_set: Set of LiftedOperators to filter preconditions for.
            pred_interpr: Dictionary mapping Predicates to their interpretation functions.
            dataset: PlannerDataset to use for filtering, including all scenarios that we want
                the operators to work in.
        Returns:
            A new set of LiftedOperators with filtered preconditions.
        """

        sampled_dataset = dataset.sample_sub_dataset(num_traj_per_scenario)
        tasks: List[Task] = []
        for traj in sampled_dataset.trajectories:
            # Reset the perceiver to get initial state and goal
            objects, _, goal = self.perceiver.reset(traj.states[0].unsqueeze(0), {})
            task = Task(
                init=traj.states[0],
                objects=objects,
                goal=goal,
                train_task_idx=traj.train_task_idx,
            )
            tasks.append(task)

        filtered_op_set = filter_operator_preconditions(
            operators=op_set,
            predicate_interpreters=pred_interpr,
            train_tasks=tasks,
            unsolvable_threshold=unsolvable_threshold,
        )
        return filtered_op_set
