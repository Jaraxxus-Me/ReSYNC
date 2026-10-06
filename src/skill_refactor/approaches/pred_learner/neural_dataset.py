"""Dataset utility for predicate learning in neural networks."""

from __future__ import division

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from skill_refactor.approaches.pred_learner.utils import two2one
from skill_refactor.benchmarks.base import BaseRLTAMPSystem, GraphData
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    GroundAtom,
    GroundOperator,
    Predicate,
    Segment,
)


class OperatorTransitionDataset(Dataset):
    """A Dataset that batch the operator transition data."""

    def __init__(
        self,
        input_data: List[Tensor],
        target: List[Tensor],
        input_data_: List[Tensor],
        target_: List[Tensor],
        middle_input_data: List[Tensor],
    ):
        """Initialize the dataset with input and target data.

        Args:
            input_data: List of input tensors of current states
            target: List of target tensors of current states
            input_data_: List of input tensors of next states
            target_: List of target tensors of next states
        """
        self.input_list = input_data
        self.target_list = target
        self.input_list_ = input_data_
        self.target_list_ = target_
        self.middle_input_list = middle_input_data
        logging.info(
            f"Created transition dataset with {len(self.input_list)} transition pairs"
        )

    def __post_init__(self):
        assert len(self.input_list) == len(
            self.target_list
        ), "Input and target lists must have the same length."
        assert len(self.input_list_) == len(
            self.target_list_
        ), "Input_ and target_ lists must have the same length."
        assert len(self.input_list) == len(
            self.input_list_
        ), "Input and input_ lists must have the same length."
        if len(self.middle_input_list):
            assert len(self.input_list) == len(
                self.middle_input_list
            ), "Input and middle_input lists must have the same length."

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if len(self.middle_input_list):
            all_middle_inputs = self.middle_input_list[idx]
            num_middle = all_middle_inputs.shape[0]
            rnd_ind = torch.randperm(num_middle)[0]
            return {
                "input": self.input_list[idx],
                "target": self.target_list[idx],
                "input_": self.input_list_[idx],
                "target_": self.target_list_[idx],
                "middle_input": all_middle_inputs[rnd_ind],
            }
        return {
            "input": self.input_list[idx],
            "target": self.target_list[idx],
            "input_": self.input_list_[idx],
            "target_": self.target_list_[idx],
        }


def create_train_val_dataloaders(
    input_data_list: List[Tensor],
    target_data_list: List[Tensor],
    input_data_list_: List[Tensor],
    target_data_list_: List[Tensor],
    input_middle_data_list: List[Tensor],
    train_ratio: float = 0.8,
    batch_size: int = 16,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val data loaders from input/target lists.

    Args:
        input_data_list: List of input tensors for current states
        target_data_list: List of target tensors for current states
        input_data_list_: List of input tensors for next states
        target_data_list_: List of target tensors for next states
        train_ratio: Fraction of data to use for training (default: 0.8)
        batch_size: Batch size for data loaders (default: 16)
        shuffle_train: Whether to shuffle training data (default: True)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data into train/val sets
    total_samples = len(input_data_list)
    train_size = int(train_ratio * total_samples)

    # Training data
    train_input_data = input_data_list[:train_size]
    train_input_data_ = input_data_list_[:train_size]
    train_target_data = target_data_list[:train_size]
    train_target_data_ = target_data_list_[:train_size]
    if len(input_middle_data_list):
        train_input_middle_data = input_middle_data_list[:train_size]
    else:
        train_input_middle_data = []

    # Validation data
    val_input_data = input_data_list[train_size:]
    val_input_data_ = input_data_list_[train_size:]
    val_target_data = target_data_list[train_size:]
    val_target_data_ = target_data_list_[train_size:]
    if len(input_middle_data_list):
        val_input_middle_data = input_middle_data_list[train_size:]
    else:
        val_input_middle_data = []

    # Create datasets
    train_dataset = OperatorTransitionDataset(
        input_data=train_input_data,
        target=train_target_data,
        input_data_=train_input_data_,
        target_=train_target_data_,
        middle_input_data=train_input_middle_data,
    )
    val_dataset = OperatorTransitionDataset(
        input_data=val_input_data,
        target=val_target_data,
        input_data_=val_input_data_,
        target_=val_target_data_,
        middle_input_data=val_input_middle_data,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Created training dataset with {len(train_dataset)} samples")
    logging.info(f"Created validation dataset with {len(val_dataset)} samples")

    return train_loader, val_loader


def js_divergence_sigmoid(
    p: torch.Tensor, q: torch.Tensor, sigmoid: bool = True
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two distributions.

    Args:
        p: Tensor representing logits/probabilities of distribution p
        q: Tensor representing logits/probabilities of distribution q
        sigmoid: Whether to apply sigmoid to convert logits to probabilities

    Returns:
        JS divergence tensor
    """
    if sigmoid:
        p = torch.sigmoid(p)
        q = torch.sigmoid(q)

    # Clamp probabilities to avoid log(0)
    p = torch.clamp(p, min=1e-10, max=1 - 1e-10)
    q = torch.clamp(q, min=1e-10, max=1 - 1e-10)

    # Calculate midpoint distribution
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_pm = F.kl_div(m.log(), p, reduction="batchmean")
    kl_qm = F.kl_div(m.log(), q, reduction="batchmean")

    # JS divergence is average of KL divergences
    return 0.5 * (kl_pm + kl_qm)


def simple_predicate_criterion(
    outputs_s: torch.Tensor,
    outputs_s_: torch.Tensor,
    targets_s: torch.Tensor,
    targets_s_: torch.Tensor,
    super_label: Dict[str, float],
    device: torch.device,
    middle_outputs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simplified supervision criterion for flat tensor inputs.

    Args:
        outputs_s: Model predictions for state s (B*N, 1)
        outputs_s_: Model predictions for state s' (B*N, 1)
        targets_s: Target labels for state s (B*N, 1)
        targets_s_: Target labels for state s' (B*N, 1)
        super_label: Dictionary with supervision label values
        device: PyTorch device
        middle_outputs: Model predictions for middle states (B*NxM, 1)

    Returns:
        Tuple of (non_change_loss, change_loss)
    """
    bce_loss = torch.nn.BCEWithLogitsLoss()
    non_change_loss = torch.tensor(0.0, device=device)
    change_loss = torch.tensor(0.0, device=device)
    num_middle = 0

    # Non-change loss: JS divergence for predicates that shouldn't change
    for label_type in ["non_change_1", "non_change_2", "non_change_3"]:
        mask_s = (targets_s == super_label[label_type]).squeeze()
        mask_s_ = (targets_s_ == super_label[label_type]).squeeze()

        if mask_s.any() and torch.equal(mask_s, mask_s_):
            # For initial and end states
            masked_outputs_s = outputs_s[mask_s]  # Kx1
            masked_outputs_s_ = outputs_s_[mask_s_]  # Kx1
            if masked_outputs_s.numel() > 0:
                non_change_loss += js_divergence_sigmoid(
                    masked_outputs_s, masked_outputs_s_
                )
                # logging.info(f"Non-change loss init-end states for {label_type}, "
                #              f"{mask_s.sum()} pairs: {non_change_loss.item():.5f}")

            # For middle states
            if middle_outputs is not None:
                masked_middle_outputs = middle_outputs[mask_s]  # KxMx1
                masked_outputs_s_r = masked_outputs_s_.unsqueeze(1).repeat(
                    1, num_middle
                )  # KxMx1
                middle_non_change_loss = js_divergence_sigmoid(
                    masked_middle_outputs.flatten(0, 1),
                    masked_outputs_s_r.flatten(0, 1),
                )
                non_change_loss += middle_non_change_loss
                # logging.info(f"Non-change loss middle states for {label_type}, "
                #              f"{masked_outputs_s_r.flatten(0, 1).shape[0]} pairs: "
                #              f"{middle_non_change_loss.item():.5f}")

    # Change loss: BCE for predicates that should change
    all_outputs = torch.cat([outputs_s, outputs_s_], dim=0)
    all_targets = torch.cat([targets_s, targets_s_], dim=0)
    if middle_outputs is not None:
        all_outputs = torch.cat([all_outputs, middle_outputs], dim=0)
        all_targets = torch.cat([all_targets, targets_s], dim=0)

    change_mask = (
        (all_targets == super_label["change_pos"])
        | (all_targets == super_label["change_neg"])
    ).squeeze()

    if change_mask.any():

        # pos_mask = (all_targets == super_label["change_pos"]).squeeze()
        # neg_mask = (all_targets == super_label["change_neg"]).squeeze()

        masked_outputs = all_outputs[change_mask]
        masked_targets = all_targets[change_mask]
        change_loss += bce_loss(masked_outputs, masked_targets)

    # Balance positive and negative samples
    # pos_mask = (masked_targets == super_label["change_pos"]).squeeze()
    # neg_mask = (masked_targets == super_label["change_neg"]).squeeze()

    # pos_indices = torch.where(pos_mask)[0]
    # neg_indices = torch.where(neg_mask)[0]

    # if len(pos_indices) > 0 and len(neg_indices) > 0:
    #     # Sample equal number of positive and negative examples
    #     min_samples = min(len(pos_indices), len(neg_indices))

    #     # Random sampling without replacement
    #     pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:min_samples]]
    #     neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:min_samples]]

    #     balanced_indices = torch.cat([pos_sample_indices, neg_sample_indices])
    #     balanced_outputs = masked_outputs[balanced_indices]
    #     balanced_targets = masked_targets[balanced_indices]

    #     change_loss = bce_loss(balanced_outputs, balanced_targets)
    # else:
    #     # Fallback to original behavior if only one class present
    #     change_loss = bce_loss(masked_outputs, masked_targets)

    return non_change_loss, change_loss


def train_predicate_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    super_label: Dict[str, float],
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_freq: int = 1,
) -> Tuple[OrderedDict, float, float]:
    """Simplified training loop for predicate models.

    Args:
        model: Neural network model to train
        train_dataloader: Training data loader with OperatorTransitionDataset
        val_dataloader: Validation data loader
        optimizer: PyTorch optimizer
        super_label: Dictionary with supervision label values
        num_epochs: Number of training epochs
        device: PyTorch device
        scheduler: Optional learning rate scheduler
        val_freq: Validation frequency (every N epochs)

    Returns:
        Tuple of (best_model_weights, best_val_loss)
    """
    since = time.perf_counter()
    best_model_weights: Dict[str, Any] = OrderedDict()
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    # M
    # n_middle = CFG.num_middle_states
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        # logging.info(f"***** Training Epoch {epoch}/{num_epochs - 1} *****")
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move data to device
            input_s = batch["input"].to(device)
            input_s_ = batch["input_"].to(device)
            target_s = batch["target"].to(device)
            target_s_ = batch["target_"].to(device)
            B = input_s.shape[0]

            # Forward pass
            if "middle_input" in batch:
                # for middle states
                input_middle_s = batch["middle_input"].to(device)  # BxNxC
                all_input = torch.cat([input_s, input_s_, input_middle_s], dim=0)
                outputs = model(all_input)  # Bx(M*N)x1
                outputs_s = outputs[:B]  # BxNx1
                outputs_s_ = outputs[B : B * 2]  # BxNx1
                outputs_middles = outputs[B * 2 :].flatten()  # (B*M)xNx1
            else:
                all_input = torch.cat([input_s, input_s_], dim=0)
                outputs = model(all_input)  # BxNx1
                outputs_s = outputs[: input_s.size(0)]  # BxNx1
                outputs_s_ = outputs[input_s.size(0) :]  # BxNx1
                outputs_middles = None

            # Compute losses, use flattened tensors
            non_change_loss, change_loss = simple_predicate_criterion(
                outputs_s.flatten(),
                outputs_s_.flatten(),
                target_s.flatten(),
                target_s_.flatten(),
                super_label,
                device,
                outputs_middles,
            )

            total_loss = non_change_loss + change_loss
            train_losses.append(total_loss.item())

            # Backward pass
            if not total_loss.any():
                # Skip if loss is zero to avoid issues in backward()
                continue
            total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        # Validation phase
        if (epoch + 1) % val_freq == 0:
            model.eval()
            val_losses = []
            logging.info(f"***** Evaluation Epoch {epoch}/{num_epochs - 1} *****")
            with torch.no_grad():
                for batch in val_dataloader:
                    input_s = batch["input"].to(device)
                    input_s_ = batch["input_"].to(device)
                    target_s = batch["target"].to(device)
                    target_s_ = batch["target_"].to(device)

                    # for middle states
                    if "middle_input" in batch:
                        # for middle states
                        input_middle_s = batch["middle_input"].to(device)  # BxNxC
                        all_input = torch.cat(
                            [input_s, input_s_, input_middle_s], dim=0
                        )
                        outputs = model(all_input)  # Bx(M*N)x1
                        outputs_s = outputs[:B]  # BxNx1
                        outputs_s_ = outputs[B : B * 2]  # BxNx1
                        outputs_middles = outputs[B * 2 :].flatten()
                    else:
                        all_input = torch.cat([input_s, input_s_], dim=0)
                        outputs = model(all_input)  # BxNx1
                        outputs_s = outputs[: input_s.size(0)]  # BxNx1
                        outputs_s_ = outputs[input_s.size(0) :]  # BxNx1
                        outputs_middles = None

                    non_change_loss, change_loss = simple_predicate_criterion(
                        outputs_s.flatten(),
                        outputs_s_.flatten(),
                        target_s.flatten(),
                        target_s_.flatten(),
                        super_label,
                        device,
                        outputs_middles,
                    )

                    val_losses.append((non_change_loss + change_loss).item())

            # Update best model
            avg_val_loss = (
                sum(val_losses) / len(val_losses) if val_losses else float("inf")
            )
            logging.info(f"Val loss: {avg_val_loss:.5f} *****")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = dict(model.state_dict())
                logging.info(
                    f"New best model at epoch {epoch} with val loss {best_val_loss:.5f}"
                )
                if avg_val_loss < 1e-3 and best_train_loss < 1e-3:
                    # Some times too low loss causes trouble in loss.backward()
                    logging.info("Early stopping as val loss is already low enough.")
                    break

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Log progress
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            logging.info(
                f"New model at epoch {epoch} with train loss {best_train_loss:.5f}"
            )
        if epoch % 10 == 0:
            logging.info(
                f"***** Training Epoch {epoch}/{num_epochs - 1} train loss: {avg_train_loss:.5f} *****"
            )

    # Training complete
    time_elapsed = time.perf_counter() - since
    logging.info(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    logging.info(f"Best training loss: {best_train_loss:.5f}")
    logging.info(f"Best validation loss: {best_val_loss:.5f}")

    return OrderedDict(best_model_weights), best_val_loss, best_train_loss


@dataclass
class OperatorTransition:
    """Single training example for neural predicate learning."""

    pre_state_graph: GraphData
    pre_atoms: Set[GroundAtom]  # Atoms in pre-state
    post_state_graph: GraphData
    post_atoms: Set[GroundAtom]  # Atoms in post-state
    operator: GroundOperator
    middle_state_graphs: List[GraphData]


@dataclass
class TransitionOperatorInfo:
    """Information about the operator in a transition."""

    id: int
    arity: int
    param_node_idxs: np.ndarray


def sample_middle_state_graphs(
    segment: Segment,
    tamp_system: BaseRLTAMPSystem,
    max_num_middle_states: int,
) -> List[GraphData]:
    """Sample middle state graphs from a segment based on num_middle_states.

    Args:
        segment: Trajectory segment containing states
        tamp_system: TAMP system for state-to-graph conversion
        max_num_middle_states: Maximum number of middle states to sample

    Returns:
        List of GraphData representing sampled middle states
    """
    middle_state_graphs: List[GraphData] = []

    # Extract middle states (excluding first and last)
    middle_states = (
        segment.states[1:-1] if len(segment.states) > 2 else [segment.states[0]]
    )
    # end_state = segment.states[-1]
    filtered_middle_states = middle_states
    # for s in middle_states:
    #     if (s - end_state).abs().max() > 1e-4:
    #         filtered_middle_states.append(s)

    if len(filtered_middle_states) > max_num_middle_states:
        # Uniformly sample from middle states
        indices = torch.linspace(
            0, len(filtered_middle_states) - 1, max_num_middle_states
        ).long()
        selected_states = [filtered_middle_states[i] for i in indices]
    else:
        # Use all middle states
        selected_states = filtered_middle_states

    # Convert selected states to graph representations
    stacked_middle_states = torch.stack(selected_states, dim=0)
    middle_state_graphs.extend(tamp_system.state_to_graph(stacked_middle_states))

    return middle_state_graphs


def generate_in_out_predicate_data(
    operator_transition_data: List[OperatorTransition],
    target_predicate: Predicate,
    ae_vector: Tensor,
    var_bind_idx: Tensor,
    action_to_index: Dict[str, int],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Generate input/target data for predicate learning.

    Args:
        operator_transition_data: List of operator transitions
        target_predicate: Target predicate to generate data for
        ae_vector: Action-effect vector [num_actions, 2]
        var_bind_idx: Variable binding indices for the predicate
        action_to_index: Mapping from action names to indices

    Returns:
        Tuple of (input_data_list, input_data_list_, input_middle_data_list, target_data_list, target_data_list_)
    """
    input_data_list: list[Any] = []
    input_data_list_: list[Any] = []
    input_middle_data_list: list[Any] = []
    target_data_list: list[Any] = []
    target_data_list_: list[Any] = []

    for transition in operator_transition_data:
        # Generate input/target pairs for each transition
        target_s, _ = compute_target_graph(
            transition.pre_state_graph,
            target_predicate,
            var_bind_idx,
            ae_vector,
            False,
            transition.operator,
            action_to_index,
        )
        target_s_, _ = compute_target_graph(
            transition.post_state_graph,
            target_predicate,
            var_bind_idx,
            ae_vector,
            True,
            transition.operator,
            action_to_index,
        )
        if target_predicate.arity == 1:
            input_data_list.append(transition.pre_state_graph.node_features)
            input_data_list_.append(transition.post_state_graph.node_features)
            target_data_list.append(target_s.node_features)
            target_data_list_.append(target_s_.node_features)
        else:
            assert (
                target_predicate.arity == 2
            ), "Only unary and binary predicates supported"
            input_data_list.append(transition.pre_state_graph.edge_features)
            input_data_list_.append(transition.post_state_graph.edge_features)
            target_data_list.append(target_s.edge_features)
            target_data_list_.append(target_s_.edge_features)

    return (
        input_data_list,
        input_data_list_,
        input_middle_data_list,
        target_data_list,
        target_data_list_,
    )


def generate_in_out_predicate_data_terminal(
    operator_transition_data: List[OperatorTransition],
    target_predicate: Predicate,
    ae_vector: Tensor,
    action_to_index: Dict[str, int],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Generate input/target data for terminal predicate learning.

    Args:
        operator_transition_data: List of operator transitions
        target_predicate: Target predicate to generate data for
        ae_vector: Action-effect vector [num_actions, 2]
        action_to_index: Mapping from action names to indices

    Returns:
        Tuple of (input_data_list, input_data_list_, input_middle_data_list, target_data_list, target_data_list_)
    """
    input_data_list = []
    input_data_list_ = []
    input_middle_data_list = []
    target_data_list = []
    target_data_list_ = []

    assert (
        "terminal" in target_predicate.name
    ), "This function is only for terminal predicates"
    assert ae_vector.sum() == 1, "This function is only for terminal predicates"
    target_operator_name = None
    for action_name, idx in action_to_index.items():
        if ae_vector[idx][0] == 1:
            target_operator_name = action_name
            break
    assert target_operator_name is not None

    for transition in operator_transition_data:
        # Skip transitions from skills that do not match this terminal predicate
        if transition.operator.name != target_operator_name:
            continue
        # Generate input/target pairs for each transition
        # Extract the input object feat (concate nodes)
        object_to_node = transition.pre_state_graph.object_to_node
        assert object_to_node is not None, "object_to_node must not be None"
        input_feat_idx = [object_to_node[obj] for obj in transition.operator.parameters]
        # Init states have False label
        input_feat = torch.concatenate(
            [transition.pre_state_graph.node_features[i] for i in input_feat_idx],
            dim=-1,
        )
        input_data_list.append(input_feat.unsqueeze(0))
        target_data_list.append(
            torch.zeros_like(input_feat[0]) + CFG.super_label["change_neg"]
        )
        # End states have True label
        input_feat_ = torch.concatenate(
            [transition.post_state_graph.node_features[i] for i in input_feat_idx],
            dim=-1,
        )
        target_data_list_.append(
            torch.zeros_like(input_feat[0]) + CFG.super_label["change_pos"]
        )
        input_data_list_.append(input_feat_.unsqueeze(0))

        # Middle states
        if len(transition.middle_state_graphs):
            middle_feat_list = []
            for g in transition.middle_state_graphs:
                individual_feat = torch.concatenate(
                    [g.node_features[i] for i in input_feat_idx], dim=-1
                )
                middle_feat_list.append(individual_feat.unsqueeze(0))
            input_middle_data_list.append(torch.stack(middle_feat_list))

    return (
        input_data_list,
        input_data_list_,
        input_middle_data_list,
        target_data_list,
        target_data_list_,
    )


def compute_target_graph(
    input_graph: GraphData,
    curr_pred: Predicate,
    var_bind_idx: Tensor,
    ae_vector: Tensor,
    is_next_state: bool,
    operator: GroundOperator,
    action_to_index: Dict[str, int],
) -> Tuple[GraphData, TransitionOperatorInfo]:
    """Compute target graph for predicate learning supervision.
    This is the most tricky part of the bilevel learning algorithm:
    Given a lifted effect vector,
    how to compute the supervision signal for the ground predicate.
    This includes:
        - Direct positive/negative supervision for operatored objects if effect exists.
        - Implicit non-change supervision for non-operatored objects / no effects.
        - Ignore labels for wrong types.

    Args:
        input_graph: Input graph data
        curr_pred: Current predicate to compute target for
        var_bind_idx: Predicate variable binding indices in operator
        ae_vector: Action-effect vector [num_actions, 2] for all actions
        is_next_state: Whether this is the next state (True) or current state (False)
        operator: The ground operator being executed

    Returns:
        target_graph: Target graph data with supervision labels for the predicate
        operator_info: Information about the operator in this transition
    """
    num_nodes = input_graph.num_nodes
    num_edges = input_graph.num_edges

    # Initialize all features with 'ignore' label
    target_node_features = torch.full((num_nodes, 1), float(CFG.super_label["ignore"]))
    target_edge_features = torch.full((num_edges, 1), float(CFG.super_label["ignore"]))

    # Get object to node mapping
    assert (
        input_graph.object_to_node is not None
    ), "Input graph must have object_to_node mapping"
    object_to_node = input_graph.object_to_node

    # Get operator index and corresponding AE vector entry
    operator_name = operator.parent.name
    if operator_name in action_to_index:
        action_index = action_to_index[operator_name]
        if action_index < ae_vector.shape[0]:
            ae_vector_value = ae_vector[action_index]  # [add_effect, delete_effect]
            add_effect = ae_vector_value[0].item()
            delete_effect = (
                ae_vector_value[1].item() if ae_vector_value.numel() > 1 else 0
            )
        else:
            # Action not in AE vector, assume no effect
            add_effect = 0
            delete_effect = 0
    else:
        # Unknown action, assume no effect
        add_effect = 0
        delete_effect = 0

    # Get action objects
    action_objects = operator.parameters

    # Create operator info for tracking
    param_node_idxs = []
    for obj in action_objects:
        assert obj in object_to_node, f"Object {obj} not found in input graph"
        param_node_idxs.append(object_to_node[obj])

    operator_info = TransitionOperatorInfo(
        id=action_to_index.get(operator_name, -1),
        arity=len(action_objects),
        param_node_idxs=np.array(param_node_idxs),
    )

    # Handle unary predicates (node features)
    if curr_pred.arity == 1:
        pred_type = curr_pred.types[0]
        ent = int(var_bind_idx[0].item())

        # Find action objects of the predicate type
        action_objects_of_type = [
            obj for obj in action_objects if obj.type == pred_type
        ]

        # Get the specific object for this predicate argument
        pred_type_obj = None
        if len(action_objects_of_type) > ent:
            pred_type_obj = action_objects_of_type[ent]

        # Apply supervision labels based on action effects
        if add_effect == 1:
            # Add effect: predicate becomes true after action
            if pred_type_obj in object_to_node:
                obj_node_idx = object_to_node[pred_type_obj]
                if is_next_state:
                    target_node_features[obj_node_idx] = CFG.super_label["change_pos"]
                else:
                    target_node_features[obj_node_idx] = CFG.super_label["change_neg"]

                # Non-operated objects get non_change_2 label
                for obj in action_objects_of_type:
                    if obj != pred_type_obj:
                        node_idx = object_to_node[obj]
                        target_node_features[node_idx] = CFG.super_label["non_change_2"]

        elif delete_effect == 1:
            # Delete effect: predicate becomes false after action
            if pred_type_obj in object_to_node:
                obj_node_idx = object_to_node[pred_type_obj]
                if is_next_state:
                    target_node_features[obj_node_idx] = CFG.super_label["change_neg"]
                else:
                    target_node_features[obj_node_idx] = CFG.super_label["change_pos"]

                # Non-operated objects get non_change_2 label
                for obj in action_objects_of_type:
                    if obj != pred_type_obj:
                        node_idx = object_to_node[obj]
                        target_node_features[node_idx] = CFG.super_label["non_change_2"]

        else:
            # No effect: predicate doesn't change
            for obj, node_idx in object_to_node.items():
                if obj not in action_objects:
                    # Non-operated objects
                    target_node_features[node_idx] = CFG.super_label["non_change_1"]
                else:
                    # Operated objects but different from target type
                    target_node_features[node_idx] = CFG.super_label["non_change_3"]

        # Set ignore label for wrong types
        for obj, node_idx in object_to_node.items():
            if obj.type != pred_type:
                target_node_features[node_idx] = CFG.super_label["ignore"]

    # Handle binary predicates (edge features)
    elif curr_pred.arity == 2:
        type0, type1 = curr_pred.types
        ent0 = int(var_bind_idx[0].item()) if var_bind_idx.numel() > 0 else 0
        ent1 = int(var_bind_idx[1].item()) if var_bind_idx.numel() > 1 else 0

        # Find action objects of each predicate type
        action_objects_type0 = [obj for obj in action_objects if obj.type == type0]
        action_objects_type1 = [obj for obj in action_objects if obj.type == type1]

        # Get the specific objects for this predicate arguments
        pred_obj0 = None
        pred_obj1 = None
        if len(action_objects_type0) > ent0:
            pred_obj0 = action_objects_type0[ent0]
        if len(action_objects_type1) > ent1:
            pred_obj1 = action_objects_type1[ent1]

        # Find the corresponding edge in the graph
        target_edge_idx = None
        if (pred_obj0 in object_to_node) and (pred_obj1 in object_to_node):
            obj0_node_idx = object_to_node[pred_obj0]
            obj1_node_idx = object_to_node[pred_obj1]

            # Find edge index for this object pair
            for i in range(num_edges):
                src, tgt = input_graph.edge_indices[:, i]
                if src == obj0_node_idx and tgt == obj1_node_idx:
                    target_edge_idx = i
                    break

        # Apply supervision labels based on action effects
        if add_effect == 1:
            # Add effect: relation becomes true after action
            if target_edge_idx is not None:
                if is_next_state:
                    target_edge_features[target_edge_idx] = CFG.super_label[
                        "change_pos"
                    ]
                else:
                    target_edge_features[target_edge_idx] = CFG.super_label[
                        "change_neg"
                    ]

                # All other edges get non_change_2 label
                for i in range(num_edges):
                    if i != target_edge_idx:
                        target_edge_features[i] = CFG.super_label["non_change_2"]

        elif delete_effect == 1:
            # Delete effect: relation becomes false after action
            if target_edge_idx is not None:
                if is_next_state:
                    target_edge_features[target_edge_idx] = CFG.super_label[
                        "change_neg"
                    ]
                else:
                    target_edge_features[target_edge_idx] = CFG.super_label[
                        "change_pos"
                    ]

                # All other edges get non_change_2 label
                for i in range(num_edges):
                    if i != target_edge_idx:
                        target_edge_features[i] = CFG.super_label["non_change_2"]

        else:
            # No effect: relation doesn't change
            for i in range(num_edges):
                src, tgt = input_graph.edge_indices[:, i]

                # Find corresponding objects for this edge
                src_obj = None
                tgt_obj = None
                for obj, node_idx in object_to_node.items():
                    if node_idx == src:
                        src_obj = obj
                    if node_idx == tgt:
                        tgt_obj = obj

                if (src_obj not in action_objects) or (tgt_obj not in action_objects):
                    # Non-operated object pairs
                    target_edge_features[i] = CFG.super_label["non_change_1"]
                else:
                    # Operated object pairs but not the target
                    target_edge_features[i] = CFG.super_label["non_change_3"]

        # Set ignore label for wrong type combinations
        for i in range(num_edges):
            src, tgt = input_graph.edge_indices[:, i]

            # Find corresponding objects for this edge
            src_obj = None
            tgt_obj = None
            for obj, node_idx in object_to_node.items():
                if node_idx == src:
                    src_obj = obj
                if node_idx == tgt:
                    tgt_obj = obj

            assert (src_obj is not None) and (
                tgt_obj is not None
            ), f"Objects for edge {i} not found in object_to_node mapping"
            if (src_obj.type != type0) or (tgt_obj.type != type1):
                target_edge_features[i] = CFG.super_label["ignore"]

    # Create target graph with supervision labels
    target_graph = GraphData(
        node_features=target_node_features,
        edge_features=target_edge_features,
        edge_indices=input_graph.edge_indices.clone(),
        global_features=input_graph.global_features,
        object_to_node=input_graph.object_to_node,
    )

    return target_graph, operator_info


def distill_ae_vector(
    model: torch.nn.Module,
    operator_transition_data: List[OperatorTransition],
    target_predicate: Predicate,
    var_bind_idx: Tensor,
    action_to_index: Dict[str, int],
    cls_threshold: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Extract the actual AE vector learned by the model.

    Args:
        model: Trained predicate model
        operator_transition_data: List of operator transitions
        target_predicate: Target predicate to distill AE vector for
        var_bind_idx: Variable binding indices for the predicate
        action_to_index: Mapping from action names to indices

    Returns:
        Tensor: Distilled AE vector [num_actions, 2] (add_effect, delete_effect)
        Tensor: Probability scores for each action [num_actions, 2]
    """
    model.eval()
    device = next(model.parameters()).device

    num_actions = len(action_to_index)
    # Count [no_change, add_effect, delete_effect] for each action
    effect_counts = torch.zeros(num_actions, 3)

    # Step 1: Extract features based on predicate arity
    if target_predicate.arity == 1:
        # Extract node features
        pre_features = []
        post_features = []
        for transition in operator_transition_data:
            pre_features.append(transition.pre_state_graph.node_features)
            post_features.append(transition.post_state_graph.node_features)
    elif target_predicate.arity == 2:
        # Extract edge features
        pre_features = []
        post_features = []
        for transition in operator_transition_data:
            pre_features.append(transition.pre_state_graph.edge_features)
            post_features.append(transition.post_state_graph.edge_features)
    else:
        raise ValueError(
            f"Only unary and binary predicates supported, got arity {target_predicate.arity}"
        )

    # Step 2: Batched classification
    pre_binary = None
    post_binary = None
    if pre_features:
        # Stack all features for batch processing
        pre_features_batched = torch.stack(pre_features, dim=0).to(device)
        post_features_batched = torch.stack(post_features, dim=0).to(device)

        with torch.no_grad():
            # Get model predictions (logits)
            pre_logits = model(
                pre_features_batched
            )  # [num_transitions, num_nodes/edges, 1]
            post_logits = model(
                post_features_batched
            )  # [num_transitions, num_nodes/edges, 1]

            # Convert to probabilities
            pre_probs = torch.sigmoid(pre_logits)
            post_probs = torch.sigmoid(post_logits)

            # Convert to binary predictions (threshold at 0.5)
            pre_binary = (pre_probs > cls_threshold).float()
            post_binary = (post_probs > cls_threshold).float()

    # Step 3: For each transition, compute effect type
    for i, transition in enumerate(operator_transition_data):
        operator = transition.operator
        operator_name = operator.parent.name

        if operator_name not in action_to_index:
            continue

        action_idx = action_to_index[operator_name]

        # Get object to node mapping
        object_to_node = transition.pre_state_graph.object_to_node
        assert (
            object_to_node is not None
        ), "Input graph must have object_to_node mapping"

        # Find the specific element (node/edge) that this predicate operates on
        target_element_idx = None

        if target_predicate.arity == 1:
            # Handle unary predicates (similar to compute_target_graph L536-594)
            pred_type = target_predicate.types[0]
            ent = int(var_bind_idx[0].item())

            # Find action objects of the predicate type
            action_objects = operator.parameters
            action_objects_of_type = [
                obj for obj in action_objects if obj.type == pred_type
            ]

            # Get the specific object for this predicate argument
            if len(action_objects_of_type) > ent:
                pred_type_obj = action_objects_of_type[ent]
                if pred_type_obj in object_to_node:
                    target_element_idx = object_to_node[pred_type_obj]

        elif target_predicate.arity == 2:
            # Handle binary predicates (similar to compute_target_graph L597-625)
            type0, type1 = target_predicate.types
            ent0 = int(var_bind_idx[0].item()) if var_bind_idx.numel() > 0 else 0
            ent1 = int(var_bind_idx[1].item()) if var_bind_idx.numel() > 1 else 0

            # Find action objects of each predicate type
            action_objects = operator.parameters
            action_objects_type0 = [obj for obj in action_objects if obj.type == type0]
            action_objects_type1 = [obj for obj in action_objects if obj.type == type1]

            # Get the specific objects for this predicate arguments
            pred_obj0 = None
            pred_obj1 = None
            if len(action_objects_type0) > ent0:
                pred_obj0 = action_objects_type0[ent0]
            if len(action_objects_type1) > ent1:
                pred_obj1 = action_objects_type1[ent1]

            # Find the corresponding edge in the graph
            if (pred_obj0 in object_to_node) and (pred_obj1 in object_to_node):
                obj0_node_idx = object_to_node[pred_obj0]
                obj1_node_idx = object_to_node[pred_obj1]

                # Find edge index for this object pair
                edge_indices = transition.pre_state_graph.edge_indices
                num_edges = edge_indices.shape[1]
                for edge_idx in range(num_edges):
                    src, tgt = edge_indices[:, edge_idx]
                    if src == obj0_node_idx and tgt == obj1_node_idx:
                        target_element_idx = edge_idx
                        break

        # If we found the target element, determine the effect type
        if (
            target_element_idx is not None
            and pre_binary is not None
            and post_binary is not None
        ):
            pre_pred = pre_binary[i, target_element_idx, 0].item()
            post_pred = post_binary[i, target_element_idx, 0].item()

            if pre_pred == post_pred:
                # No change
                effect_counts[action_idx, 0] += 1
            elif pre_pred == 0 and post_pred == 1:
                # Add effect (predicate becomes true)
                effect_counts[action_idx, 1] += 1
            elif pre_pred == 1 and post_pred == 0:
                # Delete effect (predicate becomes false)
                effect_counts[action_idx, 2] += 1

    # Step 4: Compute averaged probabilities for AE vector
    ae_vector = torch.zeros(num_actions, 2)
    for action_idx in range(num_actions):
        total_count = effect_counts[action_idx].sum()
        if total_count > 0:
            # Convert counts to probabilities
            probs = effect_counts[action_idx] / total_count
            ae_vector[action_idx, 0] = probs[1]  # add effect probability
            ae_vector[action_idx, 1] = probs[2]  # delete effect probability
        # If no samples for this action, probabilities remain 0
    one_mask = ae_vector >= 0.5
    zero_mask = ae_vector < 0.5
    ae_vector[one_mask] = 1.0
    ae_vector[zero_mask] = 0.0
    return ae_vector, effect_counts / total_count


def calculate_entropy(prob_vector: torch.Tensor) -> torch.Tensor:
    """Calculate the entropy of a probability distribution."""
    # Ensure the probabilities are in the correct range
    prob_vector = torch.clamp(prob_vector, min=1e-10, max=1.0)
    entropy = -torch.sum(prob_vector * torch.log(prob_vector), dim=-1)
    return entropy


def compute_guidance_vector(
    prob_vector: torch.Tensor,
    tgt_vector: Optional[torch.Tensor] = None,
    entropy_w: float = 0.5,
    loss_w: float = 0.5,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
):
    """Compute guidance vector from probability and target vectors.

    Args:
        prob_vector: Probability vector from neural model predictions
        tgt_vector: Optional target vector for loss computation
        entropy_w: Weight for entropy component (default: 0.5)
        loss_w: Weight for loss component (default: 0.5)
        min_prob: Minimum probability value for clamping (default: 0.0)
        max_prob: Maximum probability value for clamping (default: 1.0)

    Returns:
        Guidance vector combining entropy and loss information
    """
    # Replace NaNs with zeros
    nan_mask = torch.isnan(prob_vector)
    prob_vector = torch.where(nan_mask, torch.tensor(0.0), prob_vector)

    # Calculate entropy for each WxH position
    entropy_vector = calculate_entropy(prob_vector)

    # Set entropy to zero for positions where the original values were NaN
    nan_mask = torch.any(nan_mask, dim=-1)
    entropy_vector = torch.where(nan_mask, torch.tensor(0.0), entropy_vector)

    # Set other places (>0.0) to between neupi_entropy_entry_min and neupi_entropy_entry_max
    entropy_vector[~nan_mask] = torch.clamp(
        entropy_vector[~nan_mask], min=min_prob, max=max_prob
    )

    # Convert probability matrix to logits
    if tgt_vector is not None:
        logits_vector = torch.log(prob_vector + 1e-10)

        # Flatten the matrices for loss computation
        tgt_vector = two2one(tgt_vector)
        label_vector = tgt_vector.clone()
        label_vector[tgt_vector == -1] = 2
        logits_vec_flat = logits_vector.view(-1, logits_vector.shape[-1])
        tgt_vec_flat = label_vector.view(-1).long()

        # Calculate cross-entropy loss for each entry
        loss_vec = F.cross_entropy(logits_vec_flat, tgt_vec_flat, reduction="none")

        # Reshape the loss matrix back to WxH
        loss_vec = loss_vec.view(prob_vector.shape[0])

        # Set loss to zero for positions where the original values were NaN
        loss_vec = torch.where(
            nan_mask, torch.tensor(0.0, dtype=loss_vec.dtype), loss_vec
        )
        loss_vec[~nan_mask] = torch.clamp(
            loss_vec[~nan_mask], min=min_prob, max=max_prob
        )
    else:
        entropy_w = 1.0
        loss_w = 0.0
        loss_vec = torch.zeros_like(entropy_vector)

    guidance_vec = entropy_w * entropy_vector + loss_w * loss_vec
    return guidance_vec


def distill_quantified_ae_vector(
    model: torch.nn.Module,
    operator_transition_data: List[OperatorTransition],
    base_predicate: Predicate,
    var_bind_idx: Tensor,
    action_to_index: Dict[str, int],
    cls_threshold: float,
    tamp_system: BaseRLTAMPSystem,
    quantifier: str,
    quantified_variable_id: int,
    negation: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Extract AE vector for a quantified predicate learned by the model.

    Args:
        model: Trained predicate model for base predicate
        operator_transition_data: List of operator transitions
        base_predicate: Base predicate being quantified
        var_bind_idx: Variable binding indices for base predicate
        action_to_index: Mapping from action names to indices
        cls_threshold: Classification threshold for binary predictions
        tamp_system: TAMP system for accessing objects and types
        quantifier: Either "ForAll", "Exist", or "" (empty for negation only)
        quantified_variable_id: ID of variable to quantify (0-indexed)
        negation: Whether to apply negation to the base predicate

    Returns:
        Tuple of (ae_vector, reduced_var_bind_idx):
        - ae_vector: Tensor [num_actions, 2] for quantified predicate
        - reduced_var_bind_idx: Variable binding indices with quantified variable removed (same as input for negation only)
    """
    if quantifier not in ["ForAll", "Exist", ""]:
        raise ValueError(
            f"Quantifier must be 'ForAll', 'Exist', or '', got '{quantifier}'"
        )

    model.eval()
    device = next(model.parameters()).device
    num_actions = len(action_to_index)

    # Handle empty quantifier case (negation only)
    if quantifier == "":
        if not negation:
            raise ValueError("Empty quantifier requires negation=True")
        # For negation only, we don't reduce arity or check variable_id bounds
        reduced_var_bind_idx = var_bind_idx
        quantified_type = None
        quantified_objects = []
    else:
        if quantified_variable_id >= base_predicate.arity:
            raise IndexError(
                f"Variable ID {quantified_variable_id} out of bounds for predicate arity {base_predicate.arity}"
            )

        # Step 1: Identify quantified variable type and create reduced var_bind_idx
        quantified_type = base_predicate.types[quantified_variable_id]

        # Remove quantified variable from var_bind_idx
        reduced_var_bind_idx = (
            torch.cat(
                [
                    var_bind_idx[:quantified_variable_id],
                    var_bind_idx[quantified_variable_id + 1 :],
                ]
            )
            if var_bind_idx.numel() > 1
            else torch.tensor([], dtype=var_bind_idx.dtype)
        )

        # Step 2: Get all objects of quantified type
        all_objects = tamp_system.perceiver.objects.as_set()
        quantified_objects = [obj for obj in all_objects if obj.type == quantified_type]

    if quantifier != "" and not quantified_objects:
        # No objects of quantified type - return zero effects
        ae_vector = torch.zeros(num_actions, 2)
        return ae_vector, reduced_var_bind_idx

    # Count [no_change, add_effect, delete_effect] for each action
    effect_counts = torch.zeros(num_actions, 3)

    # Step 3: Extract features and run batched inference
    if base_predicate.arity == 1:
        # Extract node features for unary predicates
        pre_features = []
        post_features = []
        for transition in operator_transition_data:
            pre_features.append(transition.pre_state_graph.node_features)
            post_features.append(transition.post_state_graph.node_features)
    elif base_predicate.arity == 2:
        # Extract edge features for binary predicates
        pre_features = []
        post_features = []
        for transition in operator_transition_data:
            pre_features.append(transition.pre_state_graph.edge_features)
            post_features.append(transition.post_state_graph.edge_features)
    else:
        raise ValueError(
            f"Only unary and binary base predicates supported, got arity {base_predicate.arity}"
        )

    # Run batched inference if we have features
    pre_binary = None
    post_binary = None
    if pre_features:
        # Stack all features for batch processing
        pre_features_batched = torch.stack(pre_features, dim=0).to(device)
        post_features_batched = torch.stack(post_features, dim=0).to(device)

        with torch.no_grad():
            # Get model predictions (logits)
            pre_logits = model(pre_features_batched)
            post_logits = model(post_features_batched)

            # Convert to probabilities and binary predictions
            pre_probs = torch.sigmoid(pre_logits)
            post_probs = torch.sigmoid(post_logits)
            pre_binary = (pre_probs > cls_threshold).float()
            post_binary = (post_probs > cls_threshold).float()

    # Step 4: For each transition, compute effect
    for i, transition in enumerate(operator_transition_data):
        operator = transition.operator
        operator_name = operator.parent.name

        if operator_name not in action_to_index:
            continue

        action_idx = action_to_index[operator_name]
        object_to_node = transition.pre_state_graph.object_to_node
        assert (
            object_to_node is not None
        ), "Input graph must have object_to_node mapping"

        # Handle negation-only case (empty quantifier)
        if quantifier == "":
            # Direct negation of base predicate - use original distill_ae_vector logic
            # Find the target element for base predicate grounding
            target_element_idx = None
            action_objects = operator.parameters

            if base_predicate.arity == 1:
                # Handle unary predicates
                pred_type = base_predicate.types[0]
                ent = int(var_bind_idx[0].item())

                # Find action objects of the predicate type
                action_objects_of_type = [
                    obj for obj in action_objects if obj.type == pred_type
                ]

                # Get the specific object for this predicate argument
                if len(action_objects_of_type) > ent:
                    pred_type_obj = action_objects_of_type[ent]
                    if pred_type_obj in object_to_node:
                        target_element_idx = object_to_node[pred_type_obj]

            elif base_predicate.arity == 2:
                # Handle binary predicates
                type0, type1 = base_predicate.types
                ent0 = int(var_bind_idx[0].item()) if var_bind_idx.numel() > 0 else 0
                ent1 = int(var_bind_idx[1].item()) if var_bind_idx.numel() > 1 else 0

                # Find action objects of each predicate type
                action_objects_type0 = [
                    obj for obj in action_objects if obj.type == type0
                ]
                action_objects_type1 = [
                    obj for obj in action_objects if obj.type == type1
                ]

                # Get the specific objects for this predicate arguments
                pred_obj0 = None
                pred_obj1 = None
                if len(action_objects_type0) > ent0:
                    pred_obj0 = action_objects_type0[ent0]
                if len(action_objects_type1) > ent1:
                    pred_obj1 = action_objects_type1[ent1]

                # Find the corresponding edge in the graph
                if (pred_obj0 in object_to_node) and (pred_obj1 in object_to_node):
                    obj0_node_idx = object_to_node[pred_obj0]
                    obj1_node_idx = object_to_node[pred_obj1]

                    # Find edge index for this object pair
                    edge_indices = transition.pre_state_graph.edge_indices
                    num_edges = edge_indices.shape[1]
                    for edge_idx in range(num_edges):
                        src, tgt = edge_indices[:, edge_idx]
                        if src == obj0_node_idx and tgt == obj1_node_idx:
                            target_element_idx = edge_idx
                            break

            # If we found the target element, determine the effect type
            if (
                target_element_idx is not None
                and pre_binary is not None
                and post_binary is not None
            ):
                pre_pred = pre_binary[i, target_element_idx, 0].item()
                post_pred = post_binary[i, target_element_idx, 0].item()

                # Apply negation
                if negation:
                    pre_pred = 1.0 - pre_pred
                    post_pred = 1.0 - post_pred

                if pre_pred == post_pred:
                    # No change
                    effect_counts[action_idx, 0] += 1
                elif pre_pred == 0 and post_pred == 1:
                    # Add effect (predicate becomes true)
                    effect_counts[action_idx, 1] += 1
                elif pre_pred == 1 and post_pred == 0:
                    # Delete effect (predicate becomes false)
                    effect_counts[action_idx, 2] += 1
            continue

        # Find the reduced grounding (non-quantified variables) for quantified case
        action_objects = operator.parameters

        # Get reduced objects (excluding quantified position)
        reduced_objects = []
        for var_idx, pred_type in enumerate(base_predicate.types):
            if var_idx == quantified_variable_id:
                continue  # Skip quantified variable

            # Find action objects of this type
            ent = (
                int(var_bind_idx[var_idx].item())
                if var_idx < var_bind_idx.numel()
                else 0
            )
            action_objects_of_type = [
                obj for obj in action_objects if obj.type == pred_type
            ]

            if len(action_objects_of_type) > ent:
                reduced_objects.append(action_objects_of_type[ent])

        if len(reduced_objects) != (base_predicate.arity - 1):
            # Action objects do not contain the remaining typed variables
            # it has to be no effect
            effect_counts[action_idx, 0] += 1
            continue

        # Collect base predicate results for all quantified objects
        base_results_pre = []
        base_results_post = []

        for quant_obj in quantified_objects:
            # Create full grounding by inserting quantified object
            full_objects = reduced_objects.copy()
            full_objects.insert(quantified_variable_id, quant_obj)

            # Find target element (node/edge) for this grounding
            target_element_idx = None

            if base_predicate.arity == 1:
                # Unary predicate: find node index
                target_obj = full_objects[0]
                if target_obj in object_to_node:
                    target_element_idx = object_to_node[target_obj]

            elif base_predicate.arity == 2:
                # Binary predicate: find edge index
                obj0, obj1 = full_objects[0], full_objects[1]
                if (obj0 in object_to_node) and (obj1 in object_to_node):
                    obj0_node_idx = object_to_node[obj0]
                    obj1_node_idx = object_to_node[obj1]

                    # Find edge index for this object pair
                    edge_indices = transition.pre_state_graph.edge_indices
                    num_edges = edge_indices.shape[1]
                    for edge_idx in range(num_edges):
                        src, tgt = edge_indices[:, edge_idx]
                        if src == obj0_node_idx and tgt == obj1_node_idx:
                            target_element_idx = edge_idx
                            break

            # Extract binary predictions for this grounding
            if (
                target_element_idx is not None
                and pre_binary is not None
                and post_binary is not None
            ):
                pre_pred = pre_binary[i, target_element_idx, 0].item()
                post_pred = post_binary[i, target_element_idx, 0].item()

                # Apply negation if needed
                if negation:
                    pre_pred = 1.0 - pre_pred
                    post_pred = 1.0 - post_pred
                base_results_pre.append(pre_pred)
                base_results_post.append(post_pred)

        # Step 5: Apply quantifier logic if we have results (for quantified case)
        if base_results_pre and base_results_post:
            pre_tensor = torch.tensor(base_results_pre, dtype=torch.bool)
            post_tensor = torch.tensor(base_results_post, dtype=torch.bool)

            # Apply quantifier
            if quantifier == "ForAll":
                quantified_pre = torch.all(pre_tensor).item()
                quantified_post = torch.all(post_tensor).item()
            else:  # "Exist"
                quantified_pre = torch.any(pre_tensor).item()
                quantified_post = torch.any(post_tensor).item()

            # Determine effect type (same logic as distill_ae_vector L864-872)
            if quantified_pre == quantified_post:
                # No change
                effect_counts[action_idx, 0] += 1
            elif quantified_pre == 0 and quantified_post == 1:
                # Add effect (predicate becomes true)
                effect_counts[action_idx, 1] += 1
            elif quantified_pre == 1 and quantified_post == 0:
                # Delete effect (predicate becomes false)
                effect_counts[action_idx, 2] += 1

    # Step 6: Compute averaged probabilities for AE vector
    ae_vector = torch.zeros(num_actions, 2)
    for action_idx in range(num_actions):
        total_count = effect_counts[action_idx].sum()
        if total_count > 0:
            # Convert counts to probabilities
            probs = effect_counts[action_idx] / total_count
            ae_vector[action_idx, 0] = probs[1]  # add effect probability
            ae_vector[action_idx, 1] = probs[2]  # delete effect probability
        # If no samples for this action, probabilities remain 0

    one_mask = ae_vector >= 0.5
    zero_mask = ae_vector < 0.5
    ae_vector[one_mask] = 1.0
    ae_vector[zero_mask] = 0.0
    return ae_vector, reduced_var_bind_idx
