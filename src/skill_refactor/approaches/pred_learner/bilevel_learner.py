"""Bilevel predicate learning implementation for neural predicate invention.

This module implements a bilevel optimization approach that combines top-down AE vector
generation with feedback from neural learning to iteratively improve the generation
strategy.
"""

from __future__ import annotations

import glob
import logging
from typing import Callable, List, Sequence, Tuple

import torch
from relational_structs import Object
from torch import Tensor

from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.neural_dataset import (
    compute_guidance_vector,
    create_train_val_dataloaders,
    distill_ae_vector,
    distill_quantified_ae_vector,
    generate_in_out_predicate_data,
    train_predicate_model,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    create_neural_predicate_interpreter,
    create_quantified_predicate,
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.approaches.pred_learner.symbolic_search import (
    get_ae_generator_by_name,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.approaches.pred_learner.utils import two2one
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import Predicate


class BilevelPredicateLearner(TopDownPredicateLearner):
    """Bilevel predicate learner that uses feedback from neural learning to guide AE
    vector generation."""

    def _generate_candidates(
        self,
    ) -> Tuple[
        List[Predicate],
        List[Callable[[Tensor, List[Sequence[Object]]], Tensor]],
        List[Tensor],
        List[Tensor],
    ]:
        """Generate neural learning datasets for all predicate candidates using bilevel
        optimization with feedback from neural learning."""
        logging.info("Starting bilevel predicate learning...")
        candidate_predicates: List[Predicate] = []
        candidate_interpretations: List[
            Callable[[Tensor, List[Sequence[Object]]], Tensor]
        ] = []
        candidate_ae_vec: List[Tensor] = []
        candidate_ae_var: List[Tensor] = []

        # Step 0: Add quantified base predicates to candidates
        for pred, data in self.init_predicate_ae_vectors.items():
            if data["ae_vector"].sum() > 0:
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
        segment_data: List[List] = []
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

            # Step 3: Initialize AE vector generator using factory function
            ae_generator_kwargs = predicate_config.get("ae_generator", {})
            ae_generator_name = ae_generator_kwargs.get("name", "base")
            assert (
                ae_generator_name == "mct_expansion"
            ), f"Bilevel learner currently only supports 'mct_expansion' generator, got {ae_generator_name}"
            # Bilevel learner supports all generator types including adaptive ones
            ae_generator_class = get_ae_generator_by_name(ae_generator_name)
            generator_kwargs = ae_generator_kwargs.get("kwargs", {})
            ae_vector_generator = ae_generator_class(
                num_operators=len(self.action_names),
                arity=target_predicate.arity,
                **generator_kwargs,
            )

            logging.info(
                f"Will generate up to {ae_vector_generator.count_total_vectors()} AE vector candidates"
            )

            num_ae_vec = 0
            num_iterations = 0
            # Bilevel learning: iterative generation with feedback
            for ae_vector, var_bind_idx in ae_vector_generator.generate_all_vectors():
                # Generate next AE vector from the adaptive generator
                logging.info(f"AE vector generation iteration {num_iterations + 1}")
                num_iterations += 1

                if predicate_config.get("skip", False):
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

                # Step 4: Create fresh neural model for this iteration
                archi = predicate_config["nn_archi"]
                predicate_net = setup_predicate_net(archi=archi)

                # Step 5: Create the optimizer and learning rate scheduler
                optimizer_config = predicate_config["optimizer"]
                scheduler_config = predicate_config.get("lr_scheduler", None)
                optimizer, scheduler = setup_predicate_optimizer(
                    model=predicate_net,
                    opti_config=optimizer_config,
                    lr_scheduler_config=scheduler_config,
                )

                # Step 6: Generate training data for this AE vector
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

                # Step 7: Train the neural model on this dataset
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                num_epochs = predicate_config.get("num_epochs", 50)
                val_freq = predicate_config.get("val_freq", 5)

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
                    val_freq=val_freq,
                )

                # Step 8: Extract learned AE vector and provide feedback to generator
                learned_ae_vector, ae_prob = distill_ae_vector(
                    predicate_net,
                    operator_transition_data,
                    target_predicate,
                    var_bind_idx,
                    self.action_to_index,
                    cls_threshold=predicate_config.get("cls_threshold", 0.5),
                )

                feed_back = compute_guidance_vector(
                    prob_vector=ae_prob,
                    tgt_vector=ae_vector,
                )

                logging.info(
                    f"Learned AE vector for {target_predicate.name}: {learned_ae_vector}"
                )

                # KEY DIFFERENCE: Provide feedback to the AE generator
                ae_vector_generator.update_belief(feed_back)

                # Step 9: Save good models based on validation loss threshold
                val_loss_thresh = predicate_config.get("val_loss_thresh", 0.05)
                if best_val_loss < val_loss_thresh:
                    logging.info(
                        f"Best validation loss {best_val_loss:.5f} is below threshold, saving model"
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
                    torch.save(
                        learned_ae_vector,
                        f"{CFG.pred_net_save_dir}/{target_predicate.name}_{num_ae_vec}_learned_ae_vector.pth",
                    )
                    num_ae_vec += 1
                else:
                    logging.info(
                        f"Best validation loss {best_val_loss:.5f} did not meet threshold"
                    )

                # Early stopping if we reach maximum saved models
                max_saved_models = predicate_config.get("max_saved_models", 5)
                if num_ae_vec >= max_saved_models:
                    logging.info(f"Reached maximum of {max_saved_models} saved models")
                    break

            # Step 10: Load saved models and create interpretation functions
            for i in range(num_ae_vec):
                model_path = (
                    f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_model.pth"
                )

                # Create fresh model instance for loading
                archi = predicate_config["nn_archi"]
                predicate_net = setup_predicate_net(archi=archi)

                if not torch.cuda.is_available():
                    predicate_net.load_state_dict(
                        torch.load(model_path, map_location="cpu")
                    )
                else:
                    predicate_net.load_state_dict(torch.load(model_path))

                # Create a callable interpretation function
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                predicate_ins = Predicate(
                    name=target_predicate.name + f"_{i}",
                    types=target_predicate.types,
                )
                interpretation_fn = create_neural_predicate_interpreter(
                    model=predicate_net,
                    predicate=predicate_ins,
                    tamp_system=self._tamp_system,
                    cls_threshold=predicate_config.get("cls_threshold", 0.5),
                    device=device,
                )

                # Load corresponding AE vector and variable binding
                ae_vector_path = (
                    f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_ae_vector.pth"
                )
                ae_vector = torch.load(ae_vector_path)

                var_bind_idx_path = f"{CFG.pred_net_save_dir}/{target_predicate.name}_{i}_var_bind_idx.pth"
                var_bind_idx = torch.load(var_bind_idx_path)

                # Store the candidate interpretation function
                candidate_predicates.append(predicate_ins)
                candidate_interpretations.append(interpretation_fn)
                candidate_ae_vec.append(ae_vector)
                candidate_ae_var.append(var_bind_idx)

                # Step 11: Generate quantified predicates (same as TopDown)
                for quantifier_str in ["ForAll", "Exist", ""]:
                    for quantified_var_id in range(target_predicate.arity):
                        for negation in [False, True]:
                            if (quantifier_str == "") and (
                                (not negation) or (negation and quantified_var_id > 0)
                            ):
                                continue

                            try:
                                quantified_ae_vec, quantified_var_idx = (
                                    distill_quantified_ae_vector(
                                        model=predicate_net,
                                        operator_transition_data=operator_transition_data,
                                        base_predicate=target_predicate,
                                        var_bind_idx=var_bind_idx,
                                        action_to_index=self.action_to_index,
                                        cls_threshold=predicate_config.get(
                                            "cls_threshold", 0.5
                                        ),
                                        tamp_system=self._tamp_system,
                                        quantifier=quantifier_str,
                                        quantified_variable_id=quantified_var_id,
                                        negation=negation,
                                    )
                                )

                                quantified_predicate, quantified_interpreter = (
                                    create_quantified_predicate(
                                        base_predicate=target_predicate,
                                        base_interpreter=interpretation_fn,
                                        tamp_system=self._tamp_system,
                                        quantifier=quantifier_str,
                                        variable_id=quantified_var_id,
                                        negation=negation,
                                    )
                                )

                                quantified_predicate_instance = Predicate(
                                    name=quantified_predicate.name + f"_{i}",
                                    types=quantified_predicate.types,
                                )

                                # Store quantified predicates
                                candidate_predicates.append(
                                    quantified_predicate_instance
                                )
                                candidate_interpretations.append(quantified_interpreter)
                                candidate_ae_vec.append(quantified_ae_vec)
                                candidate_ae_var.append(quantified_var_idx)

                            except Exception as e:
                                logging.warning(
                                    f"Failed to create quantified predicate "
                                    f"{quantifier_str}_{quantified_var_id}_{negation}_{target_predicate.name}: {e}"
                                )

        return (
            candidate_predicates,
            candidate_interpretations,
            candidate_ae_vec,
            candidate_ae_var,
        )
