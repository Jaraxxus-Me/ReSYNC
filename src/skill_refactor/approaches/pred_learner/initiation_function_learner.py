"""Initiation function learner for training neural failure classifiers.

This module provides a learner class for training neural networks to predict when
recovery skills should be initiated based on observed states and operators.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import torch
from relational_structs import Object
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from skill_refactor.approaches.pred_learner.neural_models import (
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import GroundOperator


class FailureDataset:
    """Dataset for failure detection training.

    Attributes:
        states: Tensor of states [N, state_dim]
        operators: List of operator names for each state
        parameters: List of parameter name lists for each state
        labels: Binary labels [N] - 1 if collision will occur, 0 otherwise
    """

    def __init__(
        self,
        states: List[Tensor],
        labels: Tensor,
    ):
        """Initialize failure dataset.

        Args:
            states: List of state tensors
            operators: List of operator names
            parameters: List of parameter name lists
            labels: Binary label tensor
        """
        self.states = states
        self.labels = labels

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.states)

    def save(self, save_path: Path) -> None:
        """Save dataset to disk.

        Args:
            save_path: Path to save the dataset
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "states": self.states,
                "labels": self.labels,
            },
            save_path,
        )
        logging.info(f"Saved FailureDataset to {save_path}")

    @classmethod
    def load(cls, load_path: Path) -> FailureDataset:
        """Load dataset from disk.

        Args:
            load_path: Path to load the dataset from

        Returns:
            Loaded FailureDataset instance
        """
        data = torch.load(load_path)
        logging.info(f"Loaded FailureDataset from {load_path}")
        return cls(
            states=data["states"],
            labels=data["labels"],
        )


class InitiationFunctionLearner:
    """Learner for training neural failure detection classifiers.

    This class handles training, validation, and deployment of neural networks that
    predict when recovery skills should be initiated based on state observations.
    """

    def __init__(
        self,
        tamp_system: BaseRLTAMPSystem,
        failed_op: str,
        failed_objects: List[str],
        config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the initiation function learner.

        Args:
            tamp_system: TAMP system for feature extraction
            failed_op: Name of the operator this classifier recovers from
            failed_objects: Object names this classifier is specific to
            config: Configuration dictionary with architecture and training settings
            device: PyTorch device for training
        """
        self.tamp_system = tamp_system
        self.failed_op = failed_op
        self.failed_objects = failed_objects
        self.config = config
        self.device = device

        # Extract sub-configs (support both "nn_archi" and "architecture" keys)
        self.arch_config = config.get("nn_archi", config.get("architecture", {}))
        self.optim_config = config.get("optimizer", {})
        self.scheduler_config = config.get("lr_scheduler", None)

        # Training config - consolidate from top-level and "training" sub-dict
        self.training_config = config.get("training", {})
        # Allow top-level training params to override
        for key in ["batch_size", "num_epochs", "val_split", "early_stopping_patience"]:
            if key in config:
                self.training_config[key] = config[key]

        # Create neural network
        self.model = self._create_model()
        self.optimizer, self.scheduler = self._create_optimizer()

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, Any]] = None

        logging.info(
            f"Initialized InitiationFunctionLearner for {failed_op} with objects {failed_objects}"
        )

    def _create_model(self) -> nn.Module:
        """Create the neural network model from config.

        Returns:
            Initialized neural network
        """
        # Input dimension is determined by state_to_vec output
        # For IcyTransport with (robot, transport_obj, icy_region): 5D feature vector
        # The config can specify input_dim, or we infer it from typical state_to_vec output
        if "input_dim" in self.arch_config:
            input_dim = self.arch_config["input_dim"]
        else:
            # Default for IcyTransport scenario with 3 objects
            # state_to_vec produces: [activated, x_rel, y_rel, cos_rel, sin_rel] = 5D
            input_dim = 5
            logging.info(
                f"No input_dim in config, using default {input_dim} "
                f"(typical for state_to_vec with 3 objects)"
            )

        # Add input_dim to architecture config
        arch_config_with_dim = self.arch_config.copy()
        arch_config_with_dim["input_dim"] = input_dim

        model = setup_predicate_net(arch_config_with_dim)
        model.to(self.device)

        logging.info(
            f"Created model with input_dim={input_dim}: {type(model).__name__}"
        )
        return model

    def _create_optimizer(
        self,
    ) -> Tuple[torch.optim.Optimizer, Optional[_LRScheduler]]:
        """Create optimizer and learning rate scheduler from config.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        return setup_predicate_optimizer(
            self.model, self.optim_config, self.scheduler_config
        )

    def train(
        self,
        train_dataset: FailureDataset,
        val_dataset: Optional[FailureDataset] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Train the failure detection classifier.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            save_dir: Optional directory to save checkpoints

        Returns:
            Dictionary of training metrics (train_loss, val_loss, train_acc, val_acc)
        """
        logging.info(
            f"Starting training with {len(train_dataset)} training samples, "
            f"{len(val_dataset) if val_dataset else 0} validation samples"
        )

        # Training hyperparameters
        num_epochs = self.training_config.get("num_epochs", 100)
        batch_size = self.training_config.get("batch_size", 32)
        patience = self.training_config.get("early_stopping_patience", 10)
        log_interval = self.training_config.get("log_interval", 2)

        # Prepare data loaders
        train_loader = self._create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = (
            self._create_dataloader(val_dataset, batch_size, shuffle=False)
            if val_dataset
            else None
        )

        # Training metrics
        metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Training loop
        patience_counter = 0
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)

            # Validate
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader)
                metrics["val_loss"].append(val_loss)
                metrics["val_acc"].append(val_acc)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    patience_counter = 0

                    if save_dir:
                        self.save(save_dir / "best_model.pt")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % log_interval == 0:
                    logging.info(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
            else:
                if (epoch + 1) % log_interval == 0:
                    logging.info(
                        f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    )

        # Load best model if validation was used
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logging.info(
                f"Loaded best model with validation loss: {self.best_val_loss:.4f}"
            )

        return metrics

    def _create_dataloader(
        self, dataset: FailureDataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Create a PyTorch DataLoader from FailureDataset.

        Args:
            dataset: FailureDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        # States in the dataset are already extracted features from state_to_vec
        # during data collection. Just use them directly.
        features_list = []
        for state in dataset.states:
            # Remove batch dimension if present [1, feature_dim] -> [feature_dim]
            if state.dim() == 2 and state.shape[0] == 1:
                features_list.append(state.squeeze(0))
            else:
                features_list.append(state)

        features_tensor = torch.stack(features_list).to(self.device)
        labels_tensor = dataset.labels.to(self.device)

        tensor_dataset = TensorDataset(features_tensor, labels_tensor)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def _balance_batch(self, features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Balance a batch by upsampling the minority class.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            labels: Label tensor [batch_size]

        Returns:
            Tuple of (balanced_features, balanced_labels)
        """
        # Separate positive and negative samples
        positive_mask = labels == 1
        negative_mask = labels == 0

        positive_features = features[positive_mask]
        negative_features = features[negative_mask]
        positive_labels = labels[positive_mask]
        negative_labels = labels[negative_mask]

        num_positive = positive_features.shape[0]
        num_negative = negative_features.shape[0]

        # If one class is missing, return original batch
        if num_positive == 0 or num_negative == 0:
            return features, labels

        # Upsample minority class to match majority class
        if num_positive < num_negative:
            # Repeat positive samples to match negative count
            repeat_factor = (num_negative + num_positive - 1) // num_positive
            positive_features = positive_features.repeat(repeat_factor, 1)[
                :num_negative
            ]
            positive_labels = positive_labels.repeat(repeat_factor)[:num_negative]
        elif num_negative < num_positive:
            # Repeat negative samples to match positive count
            repeat_factor = (num_positive + num_negative - 1) // num_negative
            negative_features = negative_features.repeat(repeat_factor, 1)[
                :num_positive
            ]
            negative_labels = negative_labels.repeat(repeat_factor)[:num_positive]

        # Combine balanced samples
        balanced_features = torch.cat([positive_features, negative_features], dim=0)
        balanced_labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Shuffle to mix positive and negative samples
        shuffle_idx = torch.randperm(balanced_features.shape[0], device=features.device)
        balanced_features = balanced_features[shuffle_idx]
        balanced_labels = balanced_labels[shuffle_idx]

        return balanced_features, balanced_labels

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch with batch balancing.

        Balances each batch by upsampling the minority class to match majority class count.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        for features, labels in train_loader:
            # Balance the batch by upsampling minority class
            features_balanced, labels_balanced = self._balance_batch(features, labels)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                features_balanced.unsqueeze(1)
            )  # Add grounding dimension
            logits = logits.squeeze(-1).squeeze(-1)  # Remove extra dims

            # Compute loss
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels_balanced.float()
            )

            # Backward pass
            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) >= 0.5).long()
            correct += (predictions == labels_balanced).sum().item()
            total += labels_balanced.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch with batch balancing.

        Balances each batch by upsampling the minority class to match majority class count.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                # Balance the batch by upsampling minority class
                features_balanced, labels_balanced = self._balance_batch(
                    features, labels
                )

                # Forward pass
                logits = self.model(
                    features_balanced.unsqueeze(1)
                )  # Add grounding dimension
                logits = logits.squeeze(-1).squeeze(-1)  # Remove extra dims

                # Compute loss
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels_balanced.float()
                )

                # Track metrics
                total_loss += loss.item()
                predictions = (torch.sigmoid(logits) >= 0.5).long()
                correct += (predictions == labels_balanced).sum().item()
                total += labels_balanced.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def save(self, save_path: Path) -> None:
        """Save the trained model to disk.

        Args:
            save_path: Path to save the model
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "failed_op": self.failed_op,
                "failed_objects": self.failed_objects,
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )
        logging.info(f"Saved model to {save_path}")

    @classmethod
    def load(
        cls,
        load_path: Path,
        failed_op: str,
        failed_objects: List[str],
        config: Dict[str, Any],
        tamp_system: BaseRLTAMPSystem,
        device: torch.device = torch.device("cpu"),
    ) -> InitiationFunctionLearner:
        """Load a trained model from disk.

        Args:
            load_path: Path to load the model from
            tamp_system: TAMP system for feature extraction
            device: PyTorch device

        Returns:
            Loaded InitiationFunctionLearner instance
        """
        checkpoint = torch.load(load_path, map_location=device)

        # Create learner instance
        learner = cls(
            tamp_system=tamp_system,
            failed_op=failed_op,
            failed_objects=failed_objects,
            config=config,
            device=device,
        )

        # Load model and optimizer state
        learner.model.load_state_dict(checkpoint)

        logging.info(f"Loaded model from {load_path}")
        return learner

    def create_interpreter(
        self,
    ) -> Callable[[Tensor, Dict[str, Any], Optional[GroundOperator]], Tensor]:
        """Create an interpreter function for the trained model.

        The interpreter takes observations, info dict, and current operator,
        and returns a boolean tensor indicating whether to trigger recovery.

        Returns:
            Callable interpreter function
        """
        self.model.eval()
        model = self.model

        def interpret_initiation(
            obj_centric_state: Tensor,
            info: Dict[str, Any],
            _current_ground_operator: Optional[GroundOperator] = None,
        ) -> Tensor:
            """Interpret whether to initiate recovery based on current state.

            Args:
                obj_centric_state: Current observation [batch_size, obs_dim]
                info: Environment info dict
                _current_ground_operator: Currently executing ground operator (unused)

            Returns:
                Boolean tensor indicating which environments should trigger recovery
            """
            # If no current operator, no recovery needed
            # Get decision threshold from info dict or use default
            decision_thresh = info.get("decision_thresh", 0.5)
            logits = model(obj_centric_state.unsqueeze(0))
            prob = torch.sigmoid(logits).squeeze()
            # Use default threshold from CFG if available, otherwise 0.5
            predictions = prob >= decision_thresh

            return predictions

        return interpret_initiation

    def evaluate(self, test_dataset: FailureDataset) -> Dict[str, float]:
        """Evaluate the model on a test dataset.

        Args:
            test_dataset: Test dataset

        Returns:
            Dictionary of evaluation metrics
        """
        test_loader = self._create_dataloader(
            test_dataset, batch_size=32, shuffle=False
        )
        test_loss, test_acc = self._validate_epoch(test_loader)

        # Compute additional metrics
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for features, labels in test_loader:
                logits = self.model(features.unsqueeze(1))
                logits = logits.squeeze(-1).squeeze(-1)
                predictions = (torch.sigmoid(logits) >= 0.5).long()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute precision, recall, F1
        true_positives = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
        false_positives = sum(
            (p == 1 and l == 0) for p, l in zip(all_preds, all_labels)
        )
        false_negatives = sum(
            (p == 0 and l == 1) for p, l in zip(all_preds, all_labels)
        )

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
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logging.info(
            f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        return metrics
