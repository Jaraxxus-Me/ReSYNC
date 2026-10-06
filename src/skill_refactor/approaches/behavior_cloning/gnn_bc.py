import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from relational_structs import GroundAtom, Object, Predicate
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from skill_refactor.approaches.behavior_cloning.base import BCPolicy
from skill_refactor.approaches.behavior_cloning.gnn import (
    EncodeProcessDecode,
    setup_graph_net,
)
from skill_refactor.approaches.behavior_cloning.gnn_utils import (
    GraphDictDataset,
    get_single_model_prediction,
    graph_batch_collate,
    train_model,
)
from skill_refactor.approaches.pred_learner.neural_dataset import (
    OperatorTransition,
)
from skill_refactor.benchmarks.base import GraphData
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import PlannerDataset


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class GNNBCPolicy(BCPolicy):
    """
    Behavior Cloning policy:
    Graph (state) -> operator logits
    """

    def __init__(
        self,
        dataset: List[OperatorTransition],
        seed: int = 0,
        device: str = "cuda",
        tamp_system: Optional[Any] = None,
        goal_atoms: Optional[Set[GroundAtom]] = None,
    ):
        super().__init__(seed)
        self.dataset = dataset
        self.device = device
        self.tamp_system = tamp_system
        # Goal atoms are the same for all transitions in a single scenario
        # we use the goal specification that other approaches have access to
        self.goal_atoms = goal_atoms if goal_atoms is not None else set()

        self._gnn: Optional[EncodeProcessDecode] = None
        self._trained = False

        self.operator_to_id = {}
        self.id_to_operator = {}

        self._bce_loss = torch.nn.BCEWithLogitsLoss()
        self._crossent_loss = torch.nn.CrossEntropyLoss()

        ops = sorted({tr.operator.parent.name for tr in dataset})

        for i, op_name in enumerate(ops):
            self.operator_to_id[op_name] = i
            self.id_to_operator[i] = op_name

        self.num_operators = len(self.operator_to_id)

        # Initialize predicate feature indices (will be populated during training or loading)
        self._unary_predicates: List[Predicate] = []
        self._binary_predicates: List[Predicate] = []
        self._node_pred_to_index: Dict[str, int] = {}
        self._edge_pred_to_index: Dict[str, int] = {}

    @property
    def requires_training(self) -> bool:
        """Offline BC always requires training."""
        return True

    def initialize(self) -> None:
        """No environment-dependent initialization needed for offline BC."""

    def can_initiate(self, obs: Tensor) -> bool:
        """BC policy can always be queried."""
        _ = obs
        return True

    def reset(self, obs: Tensor | None = None) -> None:
        """Stateless policy → nothing to reset."""
        _ = obs

    def save(self, path: Path) -> None:
        """Save model weights and metadata."""
        assert self._gnn is not None

        # Save model state dict and architecture info
        checkpoint = {
            "model_state_dict": self._gnn.state_dict(),
            "operator_to_id": self.operator_to_id,
            "id_to_operator": self.id_to_operator,
            "num_operators": self.num_operators,
            "max_option_objects": self._max_option_objects,
            # Save model architecture parameters for reconstruction
            "gnn_num_message_passing": CFG.gnn_num_message_passing,
            "gnn_layer_size": CFG.gnn_layer_size,
            "gnn_num_layers": CFG.gnn_num_layers,
            # Save predicate feature mappings for inference-time augmentation
            "node_pred_to_index": self._node_pred_to_index,
            "edge_pred_to_index": self._edge_pred_to_index,
            "unary_predicates": [p.pddl_str for p in self._unary_predicates],
            "binary_predicates": [p.pddl_str for p in self._binary_predicates],
        }

        torch.save(checkpoint, path)
        logging.info(f"Saved BC model checkpoint with {self.num_operators} operators")

    def load(self, path: Path) -> None:
        """Load model weights and metadata."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load metadata first
        self.operator_to_id = checkpoint["operator_to_id"]
        self.id_to_operator = checkpoint["id_to_operator"]
        self.num_operators = checkpoint["num_operators"]
        self._max_option_objects = checkpoint["max_option_objects"]

        # Load predicate feature mappings if they exist (for models trained with predicates)
        if "node_pred_to_index" in checkpoint:
            self._node_pred_to_index = checkpoint["node_pred_to_index"]
            self._edge_pred_to_index = checkpoint["edge_pred_to_index"]
            # Note: We save predicate PDDL strings but don't reconstruct Predicate objects
            # since we only need the string names for feature indexing
            logging.info(
                f"Loaded predicate mappings: {len(self._node_pred_to_index)} node features, "
                f"{len(self._edge_pred_to_index)} edge features"
            )
        else:
            logging.info(
                "No predicate mappings in checkpoint - model trained without predicates"
            )

        # Reconstruct model architecture from checkpoint
        # We need to create a dummy dataset to infer dimensions
        # Create a minimal graph dict to build the model
        state_dict = checkpoint["model_state_dict"]

        # Infer dimensions from saved state dict
        # The node encoder's first layer tells us the input node dimension
        node_encoder_weight = state_dict["node_encoder.0.weight"]
        node_input_dim = node_encoder_weight.shape[1]

        edge_encoder_weight = state_dict["edge_encoder.0.weight"]
        edge_input_dim = edge_encoder_weight.shape[1]

        # Global encoder (if it exists)
        if "global_encoder.0.weight" in state_dict:
            global_encoder_weight = state_dict["global_encoder.0.weight"]
            global_input_dim = global_encoder_weight.shape[1]
        else:
            global_input_dim = 0

        # Node decoder output dimension (should be max_option_objects)
        node_decoder_weight = state_dict["node_decoder.0.weight"]
        node_output_dim = node_decoder_weight.shape[0]

        # Global decoder output dimension (should be num_operators)
        global_decoder_weight = state_dict["global_decoder.0.weight"]
        global_output_dim = global_decoder_weight.shape[0]

        # Create minimal dataset for model setup
        dummy_input = {
            "nodes": np.zeros((1, node_input_dim), dtype=np.float32),
            "edges": np.zeros((1, edge_input_dim), dtype=np.float32),
            "globals": (
                np.zeros((global_input_dim,), dtype=np.float32)
                if global_input_dim > 0
                else np.zeros((1,), dtype=np.float32)
            ),
            "senders": np.array([0], dtype=np.int64),
            "receivers": np.array([0], dtype=np.int64),
            "n_node": np.array([1], dtype=np.int64),
            "n_edge": np.array([1], dtype=np.int64),
        }

        dummy_output = {
            "nodes": np.zeros((1, node_output_dim), dtype=np.float32),
            "edges": np.zeros((1, 1), dtype=np.float32),
            "globals": np.zeros((global_output_dim,), dtype=np.float32),
            "senders": np.array([0], dtype=np.int64),
            "receivers": np.array([0], dtype=np.int64),
            "n_node": np.array([1], dtype=np.int64),
            "n_edge": np.array([1], dtype=np.int64),
        }

        dummy_dataset = GraphDictDataset([dummy_input], [dummy_output])

        # Build model with saved architecture parameters
        self._gnn = setup_graph_net(
            dummy_dataset,
            num_steps=checkpoint["gnn_num_message_passing"],
            layer_size=checkpoint["gnn_layer_size"],
            num_layers=checkpoint["gnn_num_layers"],
        )

        # Load weights
        self._gnn.load_state_dict(state_dict)
        self._gnn.to(self.device)
        self._gnn.eval()

        self._trained = True
        logging.info(f"Loaded BC model with {self.num_operators} operators")

    def predict_operator_and_objects(
        self,
        graph: Dict,
        object_to_node: Dict[Object, int],
        current_atoms: Optional[Set[GroundAtom]] = None,
    ) -> Tuple[str, List[Object]]:
        """Predict operator name and object parameters from graph using ranking.

        Args:
            graph: Graph dict with nodes, edges, globals
            object_to_node: Mapping from objects to node indices
            current_atoms: Optional current state atoms for predicate augmentation

        Returns:
            Tuple of (operator_name, object_parameters) for best prediction
        """
        assert self._trained, "BC model not trained"
        assert self._gnn is not None
        self._gnn.eval()

        # Augment graph with predicates if the model was trained with them
        # We check if predicate mappings exist (loaded from checkpoint)
        if self._node_pred_to_index or self._edge_pred_to_index:
            nodes = graph["nodes"]
            edges = graph["edges"]
            senders = graph["senders"]
            receivers = graph["receivers"]

            # Use provided atoms if available, otherwise empty sets
            atoms_to_use = current_atoms if current_atoms is not None else set()

            logging.info(
                f"BC-GNN inference: Augmenting graph with {len(atoms_to_use)} current atoms, "
                f"{len(self.goal_atoms)} goal atoms, "
                f"predicate mappings: {len(self._node_pred_to_index)} node, {len(self._edge_pred_to_index)} edge"
            )

            # Augment with predicate features
            nodes, edges = self._augment_graph_with_predicates(
                nodes=nodes,
                edges=edges,
                senders=senders,
                receivers=receivers,
                object_to_node=object_to_node,
                current_atoms=atoms_to_use,
                goal_atoms=self.goal_atoms,
            )

            # Update graph dict with augmented features
            graph = {
                "nodes": nodes,
                "edges": edges,
                "globals": graph["globals"],
                "senders": senders,
                "receivers": receivers,
                "n_node": graph["n_node"],
                "n_edge": graph["n_edge"],
            }

        with torch.no_grad():
            device = (
                torch.device(self.device)
                if isinstance(self.device, str)
                else self.device
            )
            output_graph = get_single_model_prediction(self._gnn, graph, device=device)

            # Get operator predictions from globals
            # output_graph has numpy arrays
            operator_logits = output_graph["globals"]  # [num_operators]

            # Apply sigmoid to get probabilities
            operator_probs = 1 / (1 + np.exp(-operator_logits))  # sigmoid

            # Get object predictions from nodes
            # output_graph["nodes"] has shape [num_nodes, max_option_objects]
            # Each column represents one object slot, values are logits
            node_logits = output_graph["nodes"]  # [num_nodes, max_option_objects]

            # Apply sigmoid to get probabilities
            node_probs = 1 / (1 + np.exp(-node_logits))  # sigmoid

            # Create reverse mapping from node indices to objects
            node_to_object = {node_idx: obj for obj, node_idx in object_to_node.items()}

            # Rank operators by their global probability
            ranked_operators = sorted(
                [
                    (op_id, self.id_to_operator[op_id], operator_probs[op_id])
                    for op_id in range(self.num_operators)
                ],
                key=lambda x: x[2],
                reverse=True,
            )

            # For the top-ranked operator, extract best object combination
            # For now, use simple greedy approach: take argmax for each slot
            _, best_op_name, _ = ranked_operators[0]

            predicted_objects: List[Object] = []
            # Keep track of which node indices we've used to avoid duplicates
            used_nodes: set[int] = set()

            for slot_idx in range(self._max_option_objects):
                # Find node with highest probability for this slot
                slot_probs = node_probs[:, slot_idx].copy()

                # Mask out already-used nodes to avoid duplicates
                for used_idx in used_nodes:
                    slot_probs[used_idx] = -np.inf

                # Always take the argmax, even if below 0.5
                # This ensures we always predict the required number of objects
                best_node_idx = int(slot_probs.argmax())

                if (
                    best_node_idx in node_to_object
                    and slot_probs[best_node_idx] > -np.inf
                ):
                    predicted_objects.append(node_to_object[best_node_idx])
                    used_nodes.add(best_node_idx)
                else:
                    # If no valid node found for this slot, stop predicting
                    # This can happen if we've run out of objects or all nodes are masked
                    logging.warning(
                        f"Could not find object for slot {slot_idx}, stopping prediction"
                    )
                    break

            return best_op_name, predicted_objects

    def predict_operator(self, graph: Dict, object_to_node: Dict[Object, int]) -> str:
        """Predict operator name from graph.

        Args:
            graph: Graph dict with nodes, edges, globals
            object_to_node: Mapping from objects to node indices

        Returns:
            operator_name: The predicted operator name
        """
        # Use the predict_operator_and_objects method and return just the operator
        # Note: This method doesn't have access to current_atoms, so augmentation won't happen
        op_name, _ = self.predict_operator_and_objects(
            graph, object_to_node, current_atoms=None
        )
        return op_name

    def _setup_output_specific_fields(
        self, transitions: List[OperatorTransition]
    ) -> None:
        # Go through the data, identifying the maximum number of option
        # objects and parameters.
        max_option_objects = 0
        for option in transitions:
            operator = option.operator
            num_objects = len(operator.parameters)
            max_option_objects = max(max_option_objects, num_objects)
        self._max_option_objects = max_option_objects

    def _setup_predicate_features(self, transitions: List[OperatorTransition]) -> None:
        """Extract all predicates from the dataset and create feature mappings.

        This builds index mappings for:
        - Unary predicates -> node features
        - Binary predicates -> edge features
        - Goal predicates (with GOAL- prefix)
        """
        if self.tamp_system is None or not CFG.bc_use_predicate_augmentation:
            # No predicate augmentation if tamp_system not provided or disabled via config
            self._unary_predicates = []
            self._binary_predicates = []
            self._node_pred_to_index = {}
            self._edge_pred_to_index = {}
            if not CFG.bc_use_predicate_augmentation:
                logging.info(
                    "Predicate augmentation DISABLED via CFG.bc_use_predicate_augmentation=False"
                )
            return

        # Collect all unique predicates from atoms
        unary_preds_set: Set[Predicate] = set()
        binary_preds_set: Set[Predicate] = set()

        for tr in transitions:
            # Check both pre and post atoms
            for atoms in [tr.pre_atoms, tr.post_atoms]:
                for atom in atoms:
                    pred = atom.predicate
                    if pred.arity == 1:
                        unary_preds_set.add(pred)
                    elif pred.arity == 2:
                        binary_preds_set.add(pred)
                    elif pred.arity > 2:
                        logging.warning(
                            f"Predicate {pred.name} has arity {pred.arity} > 2, skipping"
                        )

        # Sort for consistent ordering
        self._unary_predicates = sorted(unary_preds_set, key=lambda p: p.name)
        self._binary_predicates = sorted(binary_preds_set, key=lambda p: p.name)

        # Build node feature indices:
        # For each unary predicate, we add 2 features:
        #  1. Current state: predicate holds
        #  2. Goal state: GOAL-predicate holds
        self._node_pred_to_index = {}
        index = 0
        for pred in self._unary_predicates:
            self._node_pred_to_index[pred.name] = index
            index += 1
        for pred in self._unary_predicates:
            self._node_pred_to_index[f"GOAL-{pred.name}"] = index
            index += 1

        # Build edge feature indices:
        # For each binary predicate, we add 4 features:
        #  1. Current: predicate(obj1, obj2)
        #  2. Current reverse: REV-predicate(obj2, obj1)
        #  3. Goal: GOAL-predicate(obj1, obj2)
        #  4. Goal reverse: GOAL-REV-predicate(obj2, obj1)
        self._edge_pred_to_index = {}
        index = 0
        for pred in self._binary_predicates:
            self._edge_pred_to_index[pred.name] = index
            index += 1
        for pred in self._binary_predicates:
            self._edge_pred_to_index[f"REV-{pred.name}"] = index
            index += 1
        for pred in self._binary_predicates:
            self._edge_pred_to_index[f"GOAL-{pred.name}"] = index
            index += 1
        for pred in self._binary_predicates:
            self._edge_pred_to_index[f"GOAL-REV-{pred.name}"] = index
            index += 1

        logging.info(
            f"Augmenting graphs with {len(self._unary_predicates)} unary "
            f"and {len(self._binary_predicates)} binary predicates"
        )
        logging.info(
            f"Node features: +{len(self._node_pred_to_index)} predicate features"
        )
        logging.info(
            f"Edge features: +{len(self._edge_pred_to_index)} predicate features"
        )

    def _augment_graph_with_predicates(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        senders: np.ndarray,
        receivers: np.ndarray,
        object_to_node: Dict[Object, int],
        current_atoms: Set[GroundAtom],
        goal_atoms: Set[GroundAtom],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment node and edge features with predicate information.

        Args:
            nodes: Original node features [num_nodes, node_dim]
            edges: Original edge features [num_edges, edge_dim]
            senders: Edge source indices [num_edges]
            receivers: Edge target indices [num_edges]
            object_to_node: Mapping from objects to node indices
            current_atoms: Ground atoms in current state
            goal_atoms: Ground atoms in goal specification

        Returns:
            Tuple of (augmented_nodes, augmented_edges)
        """
        if not self._node_pred_to_index and not self._edge_pred_to_index:
            # No predicate augmentation
            return nodes, edges

        num_nodes = nodes.shape[0]
        num_edges = edges.shape[0]

        # Augment node features with unary predicates
        if self._node_pred_to_index:
            node_pred_features = np.zeros(
                (num_nodes, len(self._node_pred_to_index)), dtype=np.float32
            )

            # Fill in current state predicates
            for atom in current_atoms:
                if atom.predicate.arity == 1:
                    pred_name = atom.predicate.name
                    if pred_name in self._node_pred_to_index:
                        # Get the object and its node index
                        obj = atom.objects[0]
                        if obj in object_to_node:
                            node_idx = object_to_node[obj]
                            feature_idx = self._node_pred_to_index[pred_name]
                            node_pred_features[node_idx, feature_idx] = 1.0

            # Fill in goal state predicates
            for atom in goal_atoms:
                if atom.predicate.arity == 1:
                    pred_name = f"GOAL-{atom.predicate.name}"
                    if pred_name in self._node_pred_to_index:
                        obj = atom.objects[0]
                        if obj in object_to_node:
                            node_idx = object_to_node[obj]
                            feature_idx = self._node_pred_to_index[pred_name]
                            node_pred_features[node_idx, feature_idx] = 1.0

            # Concatenate with original node features
            nodes = np.concatenate([nodes, node_pred_features], axis=1)

        # Augment edge features with binary predicates
        if self._edge_pred_to_index:
            edge_pred_features = np.zeros(
                (num_edges, len(self._edge_pred_to_index)), dtype=np.float32
            )

            # Create edge index lookup: (sender, receiver) -> edge_index
            edge_lookup = {}
            for edge_idx in range(num_edges):
                sender = int(senders[edge_idx])
                receiver = int(receivers[edge_idx])
                edge_lookup[(sender, receiver)] = edge_idx

            # Fill in current state binary predicates
            for atom in current_atoms:
                if atom.predicate.arity == 2:
                    pred_name = atom.predicate.name
                    obj1, obj2 = atom.objects[0], atom.objects[1]

                    # Forward direction: predicate(obj1, obj2)
                    if pred_name in self._edge_pred_to_index:
                        if obj1 in object_to_node and obj2 in object_to_node:
                            node1 = object_to_node[obj1]
                            node2 = object_to_node[obj2]
                            if (node1, node2) in edge_lookup:
                                edge_idx = edge_lookup[(node1, node2)]
                                feature_idx = self._edge_pred_to_index[pred_name]
                                edge_pred_features[edge_idx, feature_idx] = 1.0

                    # Reverse direction: REV-predicate(obj2, obj1)
                    rev_pred_name = f"REV-{pred_name}"
                    if rev_pred_name in self._edge_pred_to_index:
                        if obj1 in object_to_node and obj2 in object_to_node:
                            node1 = object_to_node[obj1]
                            node2 = object_to_node[obj2]
                            if (node2, node1) in edge_lookup:
                                edge_idx = edge_lookup[(node2, node1)]
                                feature_idx = self._edge_pred_to_index[rev_pred_name]
                                edge_pred_features[edge_idx, feature_idx] = 1.0

            # Fill in goal state binary predicates
            for atom in goal_atoms:
                if atom.predicate.arity == 2:
                    pred_name = atom.predicate.name
                    obj1, obj2 = atom.objects[0], atom.objects[1]

                    # Forward direction: GOAL-predicate(obj1, obj2)
                    goal_pred_name = f"GOAL-{pred_name}"
                    if goal_pred_name in self._edge_pred_to_index:
                        if obj1 in object_to_node and obj2 in object_to_node:
                            node1 = object_to_node[obj1]
                            node2 = object_to_node[obj2]
                            if (node1, node2) in edge_lookup:
                                edge_idx = edge_lookup[(node1, node2)]
                                feature_idx = self._edge_pred_to_index[goal_pred_name]
                                edge_pred_features[edge_idx, feature_idx] = 1.0

                    # Reverse direction: GOAL-REV-predicate(obj2, obj1)
                    goal_rev_pred_name = f"GOAL-REV-{pred_name}"
                    if goal_rev_pred_name in self._edge_pred_to_index:
                        if obj1 in object_to_node and obj2 in object_to_node:
                            node1 = object_to_node[obj1]
                            node2 = object_to_node[obj2]
                            if (node2, node1) in edge_lookup:
                                edge_idx = edge_lookup[(node2, node1)]
                                feature_idx = self._edge_pred_to_index[
                                    goal_rev_pred_name
                                ]
                                edge_pred_features[edge_idx, feature_idx] = 1.0

            # Concatenate with original edge features
            edges = np.concatenate([edges, edge_pred_features], axis=1)

        return nodes, edges

    def build_graph_dataset(
        self, transitions: List[OperatorTransition]
    ) -> GraphDictDataset:

        inputs, targets = [], []
        self._setup_output_specific_fields(transitions)
        self._setup_predicate_features(transitions)

        for tr in transitions:
            g = tr.pre_state_graph
            object_to_node = g.object_to_node
            assert object_to_node is not None, "object_to_node must be set"

            # -------- INPUT GRAPH --------

            nodes = to_numpy(g.node_features).astype(np.float32)
            edges = to_numpy(g.edge_features).astype(np.float32)
            senders = to_numpy(g.edge_indices[0]).astype(np.int64)
            receivers = to_numpy(g.edge_indices[1]).astype(np.int64)

            # Augment graph features with predicates for fairness
            # This gives BC-GNN access to the same symbolic information as other approaches
            nodes, edges = self._augment_graph_with_predicates(
                nodes=nodes,
                edges=edges,
                senders=senders,
                receivers=receivers,
                object_to_node=object_to_node,
                current_atoms=tr.pre_atoms,  # Current state atoms
                goal_atoms=self.goal_atoms,  # Goal specification
            )

            n_node = np.array([nodes.shape[0]], dtype=np.int64)
            n_edge = np.array([edges.shape[0]], dtype=np.int64)

            if g.global_features is None:
                globals_in = np.zeros((1,), dtype=np.float32)
            else:
                globals_in = to_numpy(g.global_features).astype(np.float32).reshape(-1)

            graph_in = {
                "nodes": nodes,
                "edges": edges,
                "globals": globals_in,
                "senders": senders,
                "receivers": receivers,
                "n_node": n_node,
                "n_edge": n_edge,
            }

            # -------- TARGET GRAPH --------
            ## target graph construction.
            ## the global represents the operator to be predicted as one-hot (the lifted operator)
            ## the nodes capture the objects (object mask) on which the operator acts.
            # operator must be mapped to int already
            # (1) operator class → globals
            op_name = tr.operator.parent.name
            y = self.operator_to_id[op_name]

            globals_tgt = np.zeros((self.num_operators,), dtype=np.float32)
            globals_tgt[y] = 1.0

            # (2) object-parameter mask → nodes
            object_mask = np.zeros(
                (len(object_to_node), self._max_option_objects), dtype=np.float32
            )

            slot = 0
            for obj in tr.operator.parameters:
                if obj not in object_to_node:
                    continue
                object_mask[object_to_node[obj], slot] = 1.0
                slot += 1

            graph_tgt = {
                "nodes": object_mask,
                "edges": np.zeros((edges.shape[0], 1), dtype=np.float32),
                "globals": globals_tgt,
                "senders": senders,
                "receivers": receivers,
                "n_node": n_node,
                "n_edge": n_edge,
            }

            inputs.append(graph_in)
            targets.append(graph_tgt)

        return GraphDictDataset(inputs, targets)

    def _criterion(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._max_option_objects == 0:
            return torch.tensor(0.0)
        return self._bce_loss(output, target)

    def _global_criterion(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        onehot_loss = self._crossent_loss(output, target.argmax(dim=1))
        return onehot_loss

    def train_offline(self, checkpoint_dir: Optional[Path] = None) -> None:
        """Train the BC model offline.

        Args:
            checkpoint_dir: Optional directory to save training checkpoints.
                           If provided, saves periodic, best, and final checkpoints.
        """
        dataset = self.build_graph_dataset(self.dataset)

        graph_inputs = dataset.graph_dicts_input
        graph_targets = dataset.graph_dicts_target

        # Note: Normalization is already handled in state_to_graph() when creating GraphData
        # No additional normalization needed here

        # -------- split --------
        num_val = int(0.1 * len(graph_inputs)) if CFG.gnn_use_validation_set else 0
        num_val = max(1, num_val) if num_val > 0 else 0

        train_inputs = graph_inputs[num_val:]
        train_targets = graph_targets[num_val:]
        val_inputs = graph_inputs[:num_val]
        val_targets = graph_targets[:num_val]

        train_dataset = GraphDictDataset(train_inputs, train_targets)
        val_dataset = GraphDictDataset(val_inputs, val_targets)

        # -------- model --------
        self._gnn = setup_graph_net(
            train_dataset,
            num_steps=CFG.gnn_num_message_passing,
            layer_size=CFG.gnn_layer_size,
            num_layers=CFG.gnn_num_layers,
        )

        assert self._gnn is not None
        optimizer = torch.optim.Adam(self._gnn.parameters(), lr=CFG.gnn_learning_rate)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.gnn_batch_size,
            shuffle=True,
            collate_fn=graph_batch_collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=graph_batch_collate,
        )

        logging.info(f"Training BC-GNN on {len(train_dataset)} examples")

        # Prepare wandb config
        wandb_config = None
        wandb_run_name = None
        if CFG.use_wandb:
            # Create run name from checkpoint dir
            if checkpoint_dir is not None:
                wandb_run_name = checkpoint_dir.name

            # Prepare config dictionary with training hyperparameters
            wandb_config = {
                "learning_rate": CFG.gnn_learning_rate,
                "weight_decay": CFG.gnn_weight_decay,
                "num_epochs": CFG.gnn_num_epochs,
                "batch_size": CFG.gnn_batch_size,
                "num_message_passing": CFG.gnn_num_message_passing,
                "layer_size": CFG.gnn_layer_size,
                "num_layers": CFG.gnn_num_layers,
                "use_validation": CFG.gnn_use_validation_set,
                "num_train_examples": len(train_dataset),
                "num_val_examples": len(val_dataset) if num_val > 0 else 0,
                "seed": CFG.seed,
                "env": CFG.env,
                "scenario": CFG.scenario,
                "bc_use_predicate_augmentation": CFG.bc_use_predicate_augmentation,
            }

        best_state = train_model(
            self._gnn,
            {"train": train_loader, "val": val_loader},
            optimizer=optimizer,
            criterion=self._criterion,
            global_criterion=self._global_criterion,
            num_epochs=CFG.gnn_num_epochs,
            do_validation=num_val > 0,
            device=CFG.device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=getattr(CFG, "gnn_checkpoint_interval", 100),
            use_wandb=CFG.use_wandb,
            wandb_project=CFG.wandb_project,
            wandb_entity=CFG.wandb_entity,
            wandb_config=wandb_config,
            wandb_run_name=wandb_run_name,
        )
        assert self._gnn is not None

        self._gnn.load_state_dict(best_state)
        self._trained = True
