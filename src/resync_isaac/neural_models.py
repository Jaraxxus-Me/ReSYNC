"""Checkpoint-compatible Isaac architectures; legacy PoseMLP is untouched."""

from typing import Any, Callable

import torch
from torch import nn

from skill_refactor.approaches.pred_learner.neural_models import (
    MLP,
    quaternion_to_matrix,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    setup_predicate_net as legacy_setup,
)


class IsaacPoseMLP(nn.Module):
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
        init_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_rot = encoder_rot
        self.decoder = decoder

        self._apply_init(init_fn)

    def _apply_init(self, init_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
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

        # Three translation coordinates followed by a wxyz quaternion.
        rel_trans = input_tensor[..., :3]  # (batch_size, 3)
        rel_trans = rel_trans.view(-1, 3)  # Flatten to (batch_size, 3)

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

        return output.reshape(b, n_input, 1)


class UpAxisMLP(nn.Module):
    """Classify object orientation from its world-frame local up axis only.

    This removes position and yaw, which are nuisance variables for predicates
    such as ``TopUp(block)``. Input quaternions use the repository's wxyz
    convention.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        init_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        for parameter in self.parameters():
            if parameter.dim() > 1:
                init_fn(parameter)
            else:
                nn.init.uniform_(parameter, -1.0, 1.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Predict from the local z axis expressed in the world frame."""
        batch_size, num_inputs = input_tensor.shape[:2]
        quaternions = input_tensor[..., 3:7].reshape(-1, 4)
        rotation = quaternion_to_matrix(quaternions)
        local_up_world = rotation[..., :, 2]
        encoded = self.encoder(local_up_world)
        output = self.decoder(encoded)
        return output.reshape(batch_size, num_inputs, 1)


def setup_predicate_net(archi: dict[str, Any]) -> nn.Module:
    size = archi["layer_size"]
    init_fn = nn.init.xavier_uniform_
    if archi["type"] == "IsaacPoseMLP":
        return IsaacPoseMLP(
            MLP([size, size * 2], 3, True),
            MLP([size, size * 2], 9, True),
            MLP([size * 2, size, 1], size * 4, False),
            init_fn,
        )
    if archi["type"] == "UpAxisMLP":
        return UpAxisMLP(
            MLP([size, size * 2], 3, True),
            MLP([size * 2, size, 1], size * 2, False),
            init_fn,
        )
    return legacy_setup(archi)
