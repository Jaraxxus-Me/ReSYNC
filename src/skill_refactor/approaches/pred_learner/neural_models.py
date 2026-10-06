"""Model utility for predicate learning in neural networks."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from relational_structs import Object
from torch import Tensor, nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.utils.structs import Predicate


def MLP(layers: List[int], input_dim: int, with_ln: bool) -> nn.Sequential:
    """Create a multi-layer perceptron with GELU activation and normalization.

    Args:
        layers: List of hidden layer dimensions
        input_dim: Input feature dimension
        with_ln: Whether to add layer normalization at the end

    Returns:
        Sequential MLP model
    """
    mlp_layers: List[nn.Module] = [nn.Linear(input_dim, layers[0])]

    for layer_num in range(len(layers) - 1):
        mlp_layers.extend(
            [
                nn.GELU(),
                nn.BatchNorm1d(layers[layer_num]),
                nn.Linear(layers[layer_num], layers[layer_num + 1]),
            ]
        )

    if with_ln:
        mlp_layers.append(nn.LayerNorm(layers[-1]))

    return nn.Sequential(*mlp_layers)


class EncodeDecodeMLP(nn.Module):
    """Simple encode-decode MLP architecture for predicate learning.

    Args:
        arity: Predicate arity (1 for unary, 2 for binary)
        encoder: Encoder neural network
        decoder: Decoder neural network
        init_fn: Parameter initialization function
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        init_fn: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._apply_init(init_fn)

    def _apply_init(self, init_fn: Callable[[torch.Tensor], None]) -> None:
        """Apply custom initialization to model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through encode-decode architecture.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        n_input = input_tensor.shape[1]
        b = input_tensor.shape[0]
        encoded = self.encoder(input_tensor.view(-1, input_tensor.shape[2]))
        decoded = self.decoder(encoded)
        return decoded.reshape(b, n_input, 1)


class EncodeDecodePosMLP(nn.Module):
    """Simple encode-decode MLP architecture for predicate learning.

    Args:
        arity: Predicate arity (1 for unary, 2 for binary)
        encoder: Encoder neural network
        decoder: Decoder neural network
        init_fn: Parameter initialization function
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        init_fn: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._apply_init(init_fn)

    def _apply_init(self, init_fn: Callable[[torch.Tensor], None]) -> None:
        """Apply custom initialization to model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through encode-decode architecture.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        n_input = input_tensor.shape[1]
        b = input_tensor.shape[0]
        input_tensor_pos = input_tensor[..., :3]
        encoded = self.encoder(input_tensor_pos.view(-1, 3))
        decoded = self.decoder(encoded)
        return decoded.reshape(b, n_input, 1)


class PoseMLP(nn.Module):
    """Encode-decode MLP using relative pose features only.

    This model extracts rotation and translation features for two entities,
    computes their relative pose difference, and uses only that relative
    information for prediction.

    Args:
        arity: Can be 1 (ego pose) or 2 (relative pose between two entities)
        encoder: Encoder for non-rotational features
        encoder_rot: Encoder for rotational features
        decoder: Decoder network
        init_fn: Parameter initialization function
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_rot: nn.Module,
        decoder: nn.Module,
        init_fn: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder

        self._apply_init(init_fn)

    def _apply_init(self, init_fn: Callable[[torch.Tensor], None]) -> None:
        """Apply custom initialization to model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                init_fn(p)
            else:
                nn.init.uniform_(p, -1.0, 1.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass using relative pose features only.

        Args:
            input_tensor: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """

        # Compute relative translation
        n_input = input_tensor.shape[1]
        b = input_tensor.shape[0]

        # Consider finger state as another translation dimension
        rel_trans = input_tensor[..., :4]  # (batch_size, 4)
        rel_trans = rel_trans.view(-1, 4)  # Flatten to (batch_size, 4)

        # Compute relative rotation matrix
        rel_rot_q = input_tensor[..., 3:7]  # (batch_size, 4)
        rel_rot_q = rel_rot_q.view(-1, 4)
        rel_rot = quaternion_to_matrix(rel_rot_q)  # (batch_size, 3, 3)
        rel_rot_flat = rel_rot.view(-1, 9)  # (batch_size, 9)

        # Encode features separately
        encoded_trans = self.encoder(rel_trans)
        encoded_rot = self.encoder_rot(rel_rot_flat)

        # Concatenate encoded features
        encoded = torch.cat([encoded_trans, encoded_rot], dim=1)

        # Decode to final output
        output = self.decoder(encoded)

        return output.reshape(b, n_input, 1)  # Reshape to (batch_size, n_input, 1)


class PoseGT(nn.Module):
    """GT Pose MLP using relative pose features only."""

    def __init__(
        self,
        encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass using relative pose features only.

        Args:
            input_tensor: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """

        # Compute relative translation
        n_input = input_tensor.shape[1]
        b = input_tensor.shape[0]

        # Consider finger state as another translation dimension
        rel_trans = input_tensor[..., :4]  # (batch_size, 4)
        rel_trans = rel_trans.view(-1, 4)  # Flatten to (batch_size, 4)

        # Compute relative rotation matrix
        bool_res = (
            (rel_trans[:, 0] == 0.0)
            & (rel_trans[:, 1] > 0.02)
            & (rel_trans[:, 1] < 0.085)
            & (abs(rel_trans[:, 3]) < 0.08)
        )
        output = torch.zeros((b * n_input, 1), dtype=torch.float32).to(
            input_tensor.device
        )
        output[bool_res] = 0.9999
        output[~bool_res] = 0.0001
        # convert to logit
        output = torch.log(output / (1 - output))

        return output.reshape(b, n_input, 1)  # Reshape to (batch_size, n_input, 1)


def get_init_fn(init_type: str) -> Callable[[torch.Tensor], None]:
    """Function to retrieve different initialization functions."""
    if init_type == "xavier":

        def xavier_init_wrapper(tensor: torch.Tensor) -> None:
            nn.init.xavier_uniform_(tensor)

        return xavier_init_wrapper
    if init_type == "kaiming":

        def kaiming_init_wrapper(tensor: torch.Tensor) -> None:
            nn.init.kaiming_uniform_(tensor)

        return kaiming_init_wrapper
    if init_type == "uniform":

        def uniform_init_wrapper(tensor: torch.Tensor) -> None:
            nn.init.uniform_(tensor)

        return uniform_init_wrapper

    raise ValueError(f"Unknown initializer: {init_type}")


class PoseGT2D(nn.Module):
    """GT Pose MLP using relative pose features only."""

    def __init__(
        self,
        encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass using relative pose features only.

        Args:
            input_tensor: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """

        # Compute relative translation
        n_input = input_tensor.shape[1]
        b = input_tensor.shape[0]

        # Consider finger state as another translation dimension
        rel_trans = input_tensor.view(-1, 5)  # Flatten to (batch_size, 2)
        # Compute relative rotation matrix
        bool_res = rel_trans[:, 0] < 0.5
        output = torch.zeros((b * n_input, 1), dtype=torch.float32).to(
            input_tensor.device
        )
        output[bool_res] = 0.9999
        output[~bool_res] = 0.0001
        # convert to logit
        output = torch.log(output / (1 - output))

        return output.reshape(b, n_input, 1)  # Reshape to (batch_size, n_input, 1)


def setup_predicate_net(
    archi: Dict[str, Any],
) -> Union[EncodeDecodeMLP, EncodeDecodePosMLP, PoseMLP, PoseGT, PoseGT2D]:
    """Create a neural network for predicate learning.

    Args:
        pred_arity: Predicate arity (1 or 2)
        archi: Architecture configuration dictionary
        node_feat2inx: Node feature name to index mapping (for arity=1)
        edge_feat2inx: Edge feature name to index mapping (for arity=2)

    Returns:
        Configured neural network model
    """

    layer_size = int(archi.get("layer_size", 64))
    init_fn = get_init_fn(str(archi.get("initializer", "kaiming")))
    assert (
        "input_dim" in archi
    ), "Please provide input dimension of this neural classifer"
    input_dim_val = archi.get("input_dim")
    assert input_dim_val is not None, "input_dim must be provided"
    input_dim = int(input_dim_val)

    if archi["type"] == "MLP":
        encoder = MLP([layer_size, layer_size * 2], input_dim, True)
        decoder = MLP([layer_size * 2, layer_size, 1], layer_size * 2, False)
        return EncodeDecodeMLP(encoder, decoder, init_fn)

    if archi["type"] == "Pos_MLP":
        encoder = MLP([layer_size, layer_size * 2], 3, True)
        decoder = MLP([layer_size * 2, layer_size, 1], layer_size * 2, False)
        return EncodeDecodePosMLP(encoder, decoder, init_fn)

    if archi["type"] == "PoseMLP":

        # Relative features: 4 for translation (w/ finger), 9 for rotation matrix
        encoder = MLP([layer_size, layer_size * 2], 4, True)
        encoder_rot = MLP([layer_size, layer_size * 2], 9, True)
        decoder = MLP([layer_size * 2, layer_size, 1], layer_size * 4, False)

        return PoseMLP(encoder, encoder_rot, decoder, init_fn)

    if archi["type"] == "PoseGT":
        encoder = MLP([layer_size, layer_size * 2], 4, True)
        return PoseGT(encoder=encoder)

    if archi["type"] == "PoseGT2D":
        encoder = MLP([layer_size, layer_size * 2], 24, True)
        return PoseGT2D(encoder=encoder)

    raise ValueError(f"Unsupported architecture type: {archi['type']}")


def setup_predicate_optimizer(
    model: nn.Module,
    opti_config: Dict[str, Any],
    lr_scheduler_config: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.optim.Optimizer, Optional[_LRScheduler]]:
    """Setup optimizer and learning rate scheduler for neural networks.

    Args:
        model: PyTorch model to optimize
        opti_config: Optimizer configuration with 'type' and 'kwargs'
        lr_scheduler_config: Optional scheduler configuration

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    if opti_config["type"] == "Adam":
        kwargs = opti_config.get("kwargs", {})
        assert isinstance(kwargs, dict)
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif opti_config["type"] == "AdamW":
        kwargs = opti_config.get("kwargs", {})
        assert isinstance(kwargs, dict)
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {opti_config['type']}")

    # Create scheduler if specified
    scheduler: Optional[_LRScheduler] = None
    if lr_scheduler_config is not None:
        scheduler_type = lr_scheduler_config["type"]
        kwargs = lr_scheduler_config.get("kwargs", {})
        assert isinstance(kwargs, dict)

        if scheduler_type == "StepLR":
            scheduler = cast(_LRScheduler, StepLR(optimizer, **kwargs))
        elif scheduler_type == "ExponentialLR":
            scheduler = cast(_LRScheduler, ExponentialLR(optimizer, **kwargs))
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = cast(_LRScheduler, CosineAnnealingLR(optimizer, **kwargs))
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = cast(_LRScheduler, ReduceLROnPlateau(optimizer, **kwargs))
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return optimizer, scheduler


def create_neural_predicate_interpreter(
    model: nn.Module,
    predicate: Predicate,
    tamp_system: BaseRLTAMPSystem,
    cls_threshold: float,
    device: torch.device,
) -> Callable[[Tensor, List[Sequence[Object]]], Tensor]:
    """Create a neural predicate interpreter function from a trained model.

    Args:
        model: Trained neural network model
        predicate: The predicate this interpreter handles
        tamp_system: TAMP system for converting observations to GraphData
        device: PyTorch device for computation

    Returns:
        Callable interpretation function with signature:
        (obs: Tensor, objects: List[Sequence[Object]]) -> Tensor
    """
    model.eval()
    model.to(device)

    def interpret_neural_predicate(
        obs: Tensor,
        objects: List[Sequence[Object]],
        local_model=model,
        local_predicate=predicate,
        local_tamp_system=tamp_system,
        local_cls_threshold=cls_threshold,
    ) -> Tensor:
        """Neural predicate interpreter function.

        Args:
            obs: Batched observations [batch_size, obs_dim]
            objects: List of object sequences for predicate grounding

        Returns:
            Boolean groundings [batch_size, num_groundings]
        """
        batch_size = obs.shape[0]
        num_groundings = len(objects)

        with torch.no_grad():
            # Convert observations to graph data
            graph_data_list = local_tamp_system.state_to_graph(obs)

            try:
                node_dim = graph_data_list[0].node_features.shape[-1]
            except IndexError:
                raise IndexError(
                    f"Graph data list {graph_data_list}" f"obs shape {obs.shape}"
                )
            edge_dim = graph_data_list[0].edge_features.shape[-1]
            feature_dim = node_dim if (len(objects[0]) == 1) else edge_dim

            # Collect features for all batch×grounding combinations
            all_features = torch.empty(
                (batch_size, num_groundings, feature_dim), dtype=torch.float32
            ).to(device)

            for grounding_idx, object_sequence in enumerate(objects):
                for batch_idx in range(batch_size):
                    graph_data = graph_data_list[batch_idx]

                    if local_predicate.arity == 1:
                        # Unary predicate: use node features
                        obj = object_sequence[0]
                        assert (
                            graph_data.object_to_node is not None
                            and obj in graph_data.object_to_node
                        ), f"Object {obj} not found in graph data"
                        node_idx = graph_data.object_to_node[obj]
                        features = graph_data.node_features[node_idx]
                        all_features[batch_idx, grounding_idx] = features

                    elif local_predicate.arity == 2:
                        # Binary predicate: use edge features
                        obj0, obj1 = object_sequence[0], object_sequence[1]

                        assert (
                            graph_data.object_to_node is not None
                            and obj0 in graph_data.object_to_node
                            and obj1 in graph_data.object_to_node
                        ), f"Objects {obj0} and {obj1} not found in graph data"

                        node_idx0 = graph_data.object_to_node[obj0]
                        node_idx1 = graph_data.object_to_node[obj1]

                        # Find the edge between these nodes
                        edge_idx = None
                        for i in range(graph_data.num_edges):
                            src, tgt = graph_data.edge_indices[:, i]
                            if src == node_idx0 and tgt == node_idx1:
                                edge_idx = i
                                break

                        assert edge_idx is not None
                        features = graph_data.edge_features[edge_idx]
                        all_features[batch_idx, grounding_idx] = features
                    else:
                        raise ValueError(
                            f"Unsupported predicate arity: {predicate.arity}"
                        )

            # Stack all features and run batched inference
            # Shape: [batch_size * num_groundings, feature_dim]

            # Run single batched forward pass
            logits = local_model(all_features)  # [batch_size * num_groundings, 1]

            # Convert to probabilities and reshape
            probs = torch.sigmoid(logits.squeeze(-1))  # [batch_size * num_groundings]
            probs = probs.view(
                batch_size, num_groundings
            )  # [batch_size, num_groundings]

            predictions = (probs >= local_cls_threshold).bool()

            return predictions

    return interpret_neural_predicate


def create_terminal_neural_predicate_interpreter(
    model: nn.Module,
    predicate: Predicate,
    tamp_system: BaseRLTAMPSystem,
    cls_threshold: float,
    device: torch.device,
) -> Callable[[Tensor, List[Sequence[Object]]], Tensor]:
    """Create a neural predicate interpreter function from a trained model. This is
    specifically for terminal predicates.

    Args:
        model: Trained neural network model
        predicate: The predicate this interpreter handles
        tamp_system: TAMP system for converting observations to GraphData
        device: PyTorch device for computation

    Returns:
        Callable interpretation function with signature:
        (obs: Tensor, objects: List[Sequence[Object]]) -> Tensor
    """
    model.eval()
    model.to(device)
    assert "terminal" in predicate.name, "This function is only for terminal predicates"

    def interpret_neural_predicate(
        obs: Tensor,
        objects: List[Sequence[Object]],
        local_model=model,
        local_tamp_system=tamp_system,
        local_cls_threshold=cls_threshold,
    ) -> Tensor:
        """Neural predicate interpreter function.

        Args:
            obs: Batched observations [batch_size, obs_dim]
            objects: List of object sequences for predicate grounding

        Returns:
            Boolean groundings [batch_size, num_groundings]
        """
        batch_size = obs.shape[0]
        num_groundings = len(objects)
        num_objects_per_grounding = len(objects[0])

        with torch.no_grad():
            # Convert observations to graph data
            graph_data_list = local_tamp_system.state_to_graph(obs)

            node_dim = graph_data_list[0].node_features.shape[-1]
            feature_dim = node_dim * num_objects_per_grounding

            # Collect features for all batch×grounding combinations
            all_features = torch.empty(
                (batch_size, num_groundings, feature_dim), dtype=torch.float32
            ).to(device)

            for grounding_idx, object_sequence in enumerate(objects):
                for batch_idx in range(batch_size):
                    graph_data = graph_data_list[batch_idx]
                    node_feature_list = [
                        graph_data.node_features[graph_data.object_to_node[obj]]
                        for obj in object_sequence
                    ]
                    features = torch.cat(node_feature_list, dim=-1)
                    all_features[batch_idx, grounding_idx] = features

            # Stack all features and run batched inference
            # Shape: [batch_size * num_groundings, feature_dim]

            # Run single batched forward pass
            logits = local_model(all_features)  # [batch_size * num_groundings, 1]

            # Convert to probabilities and reshape
            probs = torch.sigmoid(logits.squeeze(-1))  # [batch_size * num_groundings]
            probs = probs.view(
                batch_size, num_groundings
            )  # [batch_size, num_groundings]

            predictions = (probs >= local_cls_threshold).bool()
            return predictions

    return interpret_neural_predicate


def create_quantified_predicate(
    base_predicate: Predicate,
    base_interpreter: Callable[[Tensor, List[Sequence[Object]]], Tensor],
    tamp_system: BaseRLTAMPSystem,
    quantifier: str,
    variable_id: int,
    negation: bool = False,
) -> Tuple[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]]:
    """Create a quantified version of a predicate with its interpreter.

    Args:
        base_predicate: The base predicate to quantify
        base_interpreter: The interpreter function for the base predicate
        tamp_system: TAMP system for accessing all objects and types
        quantifier: Either "ForAll", "Exist", or "" (empty for negation only)
        variable_id: ID of the variable to quantify (0-indexed position in base predicate args)
        negation: Whether to apply negation to the base predicate

    Returns:
        Tuple of (quantified_predicate, quantified_interpreter)

    Raises:
        ValueError: If quantifier is not "ForAll", "Exist", or ""
        IndexError: If variable_id is out of bounds for the base predicate arity
    """
    if quantifier not in ["ForAll", "Exist", ""]:
        raise ValueError(
            f"Quantifier must be 'ForAll', 'Exist', or '', got '{quantifier}'"
        )

    # Handle empty quantifier case (negation only)
    if quantifier == "":
        if not negation:
            raise ValueError("Empty quantifier requires negation=True")
        # For negation only, we don't reduce arity or check variable_id bounds
        negated_pred_name = f"Not_{base_predicate.name}"
        quantified_predicate = Predicate(negated_pred_name, base_predicate.types)
        quantified_type = None
        reduced_arg_types = base_predicate.types
    else:
        if variable_id >= base_predicate.arity:
            raise IndexError(
                f"Variable ID {variable_id} out of bounds for predicate arity {base_predicate.arity}"
            )

        # Get the type of the variable being quantified
        quantified_type = base_predicate.types[variable_id]
        var_name = quantified_type.name

        # Create new predicate name with negation if needed
        neg_prefix = "Not_" if negation else ""
        quantified_pred_name = (
            f"{quantifier}_{var_name}_{variable_id}_{neg_prefix}{base_predicate.name}"
        )

        # Create reduced argument types (removing the quantified variable)
        reduced_arg_types = [
            arg_type
            for i, arg_type in enumerate(base_predicate.types)
            if i != variable_id
        ]

        # Create the quantified predicate
        quantified_predicate = Predicate(
            quantified_pred_name, reduced_arg_types, base_types=base_predicate.types
        )

    def quantified_interpreter(
        obs: Tensor,
        objects: List[Sequence[Object]],
        local_quantifier=quantifier,
        local_variable_id=variable_id,
        local_negation=negation,
        local_base_interpreter=base_interpreter,
        local_tamp_system=tamp_system,
        local_quantified_type=quantified_type,
    ) -> Tensor:
        """Quantified predicate interpreter function.

        Args:
            obs: Batched observations [batch_size, obs_dim]
            objects: List of object sequences for predicate grounding
            (reduced arity for quantified, same arity for negation only)

        Returns:
            Boolean groundings [batch_size, num_groundings]
        """
        batch_size = obs.shape[0]
        num_groundings = len(objects)

        # Handle negation-only case (empty quantifier)
        if local_quantifier == "":
            # Direct negation of base predicate
            all_base_results = local_base_interpreter(obs, objects)
            if local_negation:
                all_base_results = ~all_base_results
            return all_base_results

        # Handle quantified cases
        # Get all objects of the quantified type from TAMP system
        all_objects = local_tamp_system.perceiver.objects.as_set()
        quantified_objects = [
            obj for obj in all_objects if obj.type == local_quantified_type
        ]

        if not quantified_objects:
            # If no objects of the quantified type exist, handle edge cases
            if local_quantifier == "ForAll":
                # ForAll over empty set is vacuously true
                result = torch.ones(
                    (batch_size, num_groundings), dtype=torch.bool, device=obs.device
                )
            else:  # "Exist"
                # Exist over empty set is false
                result = torch.zeros(
                    (batch_size, num_groundings), dtype=torch.bool, device=obs.device
                )
            return result

        # Generate all expanded groundings in a batched manner
        # For each reduced grounding, expand it with each quantified object
        all_expanded_groundings: List[Sequence[Object]] = []
        grounding_to_expanded_indices: List[List[int]] = []

        for reduced_objects in objects:
            expanded_indices_for_this_grounding = []
            for quant_obj in quantified_objects:
                # Insert quantified object at the correct position
                full_grounding = list(reduced_objects)
                if quant_obj in full_grounding:
                    # Avoid duplicate objects in the same grounding
                    # (Do not quantify self)
                    continue
                full_grounding.insert(local_variable_id, quant_obj)
                all_expanded_groundings.append(full_grounding)
                expanded_indices_for_this_grounding.append(
                    len(all_expanded_groundings) - 1
                )
            grounding_to_expanded_indices.append(expanded_indices_for_this_grounding)

        if not all_expanded_groundings:
            # No expanded groundings to evaluate
            result = torch.zeros(
                (batch_size, num_groundings), dtype=torch.bool, device=obs.device
            )
            if local_negation:
                result = ~result
            return result

        # Single batched call to base interpreter
        all_base_results = local_base_interpreter(obs, all_expanded_groundings)
        # Apply negation to base results if needed
        if local_negation:
            all_base_results = ~all_base_results
        # all_base_results shape: [batch_size, total_expanded_groundings]

        # Prepare results tensor
        results = torch.zeros(
            (batch_size, num_groundings), dtype=torch.bool, device=obs.device
        )

        # Apply quantifier logic for each original grounding
        for grounding_idx, expanded_indices in enumerate(grounding_to_expanded_indices):
            if expanded_indices:
                # Extract results for this grounding's expanded versions
                grounding_results = all_base_results[:, expanded_indices]
                # grounding_results shape: [batch_size, num_quantified_objects]

                if local_quantifier == "ForAll":
                    # ForAll: True if all groundings are true
                    results[:, grounding_idx] = torch.all(grounding_results, dim=1)
                else:  # "Exist"
                    # Exist: True if any grounding is true
                    results[:, grounding_idx] = torch.any(grounding_results, dim=1)

        return results

    return quantified_predicate, quantified_interpreter
