"""A recomposer that first loads pre-trained predicate models, then selects a subset of
them based on compositional planning performance.

This module implements the topdown optimization approach that:
1. Load pre-trained predicate models (neural networks) from different sceanrios.
2.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from relational_structs import Object
from torch import Tensor

from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.neural_dataset import (
    OperatorTransition,
    create_train_val_dataloaders,
    generate_in_out_predicate_data,
    train_predicate_model,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    PlannerDataset,
    Predicate,
    Segment,
)


class SequentialPredicateFinetuner(TopDownPredicateLearner):
    """A predicate learner that first generates candidate predicates using pre-trained
    neural models, then selects a subset of them using top-down compositional planning
    performance."""

    def __init__(self, old_planner_dataset: PlannerDataset, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert (
            len(CFG.scenario.split(",")) >= 2
        ), "Must have multiple scenarios for pre-trained models."
        # For each scenario, how to map the effect vectors to the current composed ones.
        self.scenario_action_mapper: Dict[str, List[int]] = {}

        old_operators = old_planner_dataset.get_appearing_operators()
        scenario_action_names = sorted(
            [op.name for op in old_operators], key=lambda x: x
        )
        scenario_action_mapper = []
        for scenario_action_name in scenario_action_names:
            assert (
                scenario_action_name in self.action_names
            ), f"Action {scenario_action_name} not found in current dataset."
            mapped_idx = self.action_names.index(scenario_action_name)
            scenario_action_mapper.append(mapped_idx)
        self.scenario_action_mapper[self.scenario] = scenario_action_mapper
        self.selected_predicate_names: List[str] = []

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

    def finetune(
        self,
    ) -> None:
        """Main learning pipeline.

        Args:
            dataset: Training dataset
        """
        logging.info("Starting predicate learning...")

        # Existing JSON file exists
        json_path = Path(CFG.pred_net_save_dir) / (
            CFG.invented_pred_op_json + f"_sc{self.scenario}.json"
        )
        assert json_path.exists()
        logging.info(f"Found existing invented predicates/operators file: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract base predicate names from invented predicates
        base_predicate_names = set()
        for pred_str in data["invented_predicates"]:
            # Extract predicate name from PDDL string: "(PredName ?x0 - type)" -> "PredName"
            pred_name = pred_str.strip("()").split()[0]
            # Find base predicate name matching pattern b_sc<num>p<num>_<num>
            match = re.search(r"b_sc\d+p\d+_\d+", pred_name)
            if match:
                base_predicate_names.add(match.group(0))

        self.selected_predicate_names = sorted(list(base_predicate_names))
        logging.info(
            f"Extracted {len(self.selected_predicate_names)} base predicate names: {self.selected_predicate_names}"
        )

        self._finetune_candidates()

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

                # Fine-tuning does not need middle states
                transition = OperatorTransition(
                    pre_state_graph=state_graph,
                    pre_atoms=atoms,
                    post_state_graph=state_graph_,
                    post_atoms=atoms_,
                    operator=operator,
                    middle_state_graphs=[],
                )
                transition_data.append(transition)

        return transition_data

    def _finetune_candidates(
        self,
    ) -> None:
        """Generate candidate predicates from different scenario pre-trained models."""
        logging.info("Starting fine-tuning...")

        # Step 1: Convert ll-traj dataset to operator transition dataset
        effect_mapper = self.scenario_action_mapper[self.scenario]
        segment_data: List[List[Segment]] = []
        for low_level_traj, ground_atoms in self._ground_atom_dataset:
            segments = segment_trajectory(low_level_traj, ground_atoms)
            segment_data.append(segments)
        operator_transition_data = self._create_transition_dataset(segment_data)
        logging.info(f"Created {len(operator_transition_data)} training examples")

        # Step 2: For each predicate type, load existing, remapped ae vectors and fine-tune neural model.
        # NOTE: If the neural model is part of the selected predicates.
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
                pred_instance_name = f"{target_predicate.name}_{i}"
                if pred_instance_name in self.selected_predicate_names:
                    logging.info(
                        f"Starting fine-tuning for existing selected predicate {pred_instance_name}"
                    )
                    model_path = (
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_model.pth"
                    )
                    ae_vector_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_ae_vector.pth"
                    old_ae_vector = torch.load(ae_vector_path)
                    var_bind_idx_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_var_bind_idx.pth"
                    var_bind_idx = torch.load(var_bind_idx_path)

                    dim_ae_vector = len(self.action_names)
                    assert old_ae_vector.shape[0] != dim_ae_vector
                    # The predicate will be used in a different domain with different
                    # number of actions, need to re-distill the AE vector
                    logging.info(
                        f"Old AE vector: {old_ae_vector.shape}, New AE vector dim: {dim_ae_vector}"
                    )
                    ae_vector = torch.zeros(
                        (len(self.action_names), 2), dtype=torch.float32
                    )
                    assert len(old_ae_vector) == len(
                        effect_mapper
                    ), "Effect mapper length does not match old AE vector length."
                    for idx, mapped_idx in enumerate(effect_mapper):
                        ae_vector[mapped_idx] = old_ae_vector[idx]

                    logging.info(
                        f"Remapped AE vector for predicate {pred_instance_name}: {ae_vector}"
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
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    num_epochs = predicate_config.get("num_epochs", 50)
                    val_freq = predicate_config.get("val_freq", 5)
                    predicate_net.load_state_dict(torch.load(model_path))

                    logging.info(
                        f"Training neural model for {num_epochs} epochs on device {device}"
                    )

                    best_weights, best_val_loss, _ = train_predicate_model(
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

                    logging.info(
                        f"Fine-tuning completed. Best validation loss: {best_val_loss:.5f}"
                    )
                    logging.info(f"Saving fine-tuned model weights to {model_path}")
                    # Save the trained model
                    torch.save(
                        best_weights,
                        model_path,
                    )
                else:
                    logging.info(
                        f"Skipping fine-tuning for predicate {pred_instance_name} as it is not selected."
                    )
