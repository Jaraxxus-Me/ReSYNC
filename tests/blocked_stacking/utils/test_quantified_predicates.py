"""Unit tests for quantified predicate creation and interpretation."""

from typing import List, Sequence

import pytest
import torch
from relational_structs import Object
from torch import Tensor

from skill_refactor import register_all_environments
from skill_refactor.approaches.pred_learner.neural_models import (
    create_quantified_predicate,
)
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.utils.structs import Predicate


def test_create_quantified_predicate_hand_defined() -> None:
    """Test quantified predicate creation with hand-defined predicates."""
    # Configure and create TAMP system
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Get types and objects
    robot_type = tamp_system.components.type_container.as_dict()["robot"]
    block_type = tamp_system.components.type_container.as_dict()["block"]

    # Create a binary predicate Near(robot, block)
    near_predicate = Predicate("Near", [robot_type, block_type])

    def near_interpreter(obs: Tensor, objects: List[Sequence[Object]]) -> Tensor:
        """Simple hand-defined interpreter: always returns True for testing."""
        batch_size = obs.shape[0]
        num_groundings = len(objects)
        return torch.ones(
            (batch_size, num_groundings), dtype=torch.bool, device=obs.device
        )

    # Test ForAll quantification over first variable (robot)
    forall_pred, forall_interp = create_quantified_predicate(
        near_predicate, near_interpreter, tamp_system, "ForAll", 0
    )

    # Check predicate name and arity
    assert forall_pred.name == "ForAll_robot_0_Near"
    assert forall_pred.arity == 1
    assert forall_pred.types[0] == block_type

    # Test Exist quantification over second variable (block)
    exist_pred, exist_interp = create_quantified_predicate(
        near_predicate, near_interpreter, tamp_system, "Exist", 1
    )

    # Check predicate name and arity
    assert exist_pred.name == "Exist_block_1_Near"
    assert exist_pred.arity == 1
    assert exist_pred.types[0] == robot_type

    # Test interpreter execution
    obs = torch.randn(2, 100)  # batch_size=2, obs_dim=100

    # For ForAll quantification, we provide block objects
    block_objects = [
        tamp_system.perceiver.objects["grasp_block"],
        tamp_system.perceiver.objects["base_block"],
    ]
    forall_objects: List[Sequence[Object]] = [[obj] for obj in block_objects]

    forall_result = forall_interp(obs, forall_objects)
    assert forall_result.shape == (2, 2)  # batch_size=2, num_groundings=2
    assert forall_result.dtype == torch.bool
    # Since base interpreter always returns True, ForAll should be True
    assert torch.all(forall_result)

    # For Exist quantification, we provide robot objects
    robot_objects = [tamp_system.perceiver.objects["robot"]]
    exist_objects: List[Sequence[Object]] = [[obj] for obj in robot_objects]

    exist_result = exist_interp(obs, exist_objects)
    assert exist_result.shape == (2, 1)  # batch_size=2, num_groundings=1
    assert exist_result.dtype == torch.bool
    # Since base interpreter always returns True, Exist should be True
    assert torch.all(exist_result)


def test_create_quantified_predicate_neural() -> None:
    """Test quantified predicate creation with neural predicates."""
    # Configure and create TAMP system
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Get types
    robot_type = tamp_system.components.type_container.as_dict()["robot"]
    block_type = tamp_system.components.type_container.as_dict()["block"]
    obstruction_rec_type = tamp_system.components.type_container.as_dict()[
        "obstruction_rec"
    ]

    # Create a ternary predicate Close(robot, block, obstruction_rec)
    close_predicate = Predicate("Close", [robot_type, block_type, obstruction_rec_type])

    # Create a simple neural network that returns constant values
    class ConstantNet(torch.nn.Module):
        """Simple neural network that returns constant values for testing."""

        def __init__(self, constant_value: float):
            super().__init__()
            self.constant_value = constant_value

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass returning constant values."""
            batch_size, num_groundings, _ = x.shape
            # Return constant logits
            return torch.full((batch_size, num_groundings, 1), self.constant_value)

    # Create neural interpreter that returns constant values
    def create_constant_neural_interpreter(constant_value: float):
        model = ConstantNet(constant_value)

        def constant_neural_interpreter(
            obs: Tensor, objects: List[Sequence[Object]]
        ) -> Tensor:
            batch_size = obs.shape[0]
            num_groundings = len(objects)

            # Create dummy features for neural model
            features = torch.randn((batch_size, num_groundings, 10))
            logits = model(features)
            probs = torch.sigmoid(logits.squeeze(-1))
            return (probs >= 0.5).bool()

        return constant_neural_interpreter

    # Test with different constant values
    true_interpreter = create_constant_neural_interpreter(10.0)  # High logit -> True
    false_interpreter = create_constant_neural_interpreter(-10.0)  # Low logit -> False

    # Test ForAll quantification over middle variable (block) with True base
    forall_true_pred, forall_true_interp = create_quantified_predicate(
        close_predicate, true_interpreter, tamp_system, "ForAll", 1
    )

    assert forall_true_pred.name == "ForAll_block_1_Close"
    assert forall_true_pred.arity == 2
    assert forall_true_pred.types == (robot_type, obstruction_rec_type)

    # Test Exist quantification over first variable (robot) with False base
    exist_false_pred, exist_false_interp = create_quantified_predicate(
        close_predicate, false_interpreter, tamp_system, "Exist", 0
    )

    assert exist_false_pred.name == "Exist_robot_0_Close"
    assert exist_false_pred.arity == 2
    assert exist_false_pred.types == (block_type, obstruction_rec_type)

    # Test interpreter execution
    obs = torch.randn(3, 100)  # batch_size=3, obs_dim=100

    # For ForAll over block (True base), provide robot-obstruction pairs
    robot_obj = tamp_system.perceiver.objects["robot"]
    obstruction_obj = tamp_system.perceiver.objects["obstruction_rec"]
    forall_objects: List[Sequence[Object]] = [[robot_obj, obstruction_obj]]

    forall_result = forall_true_interp(obs, forall_objects)
    assert forall_result.shape == (3, 1)
    # Since base returns True for all blocks, ForAll should be True
    assert torch.all(forall_result)

    # For Exist over robot (False base), provide block-obstruction pairs
    block_obj = tamp_system.perceiver.objects["grasp_block"]
    exist_objects: List[Sequence[Object]] = [[block_obj, obstruction_obj]]

    exist_result = exist_false_interp(obs, exist_objects)
    assert exist_result.shape == (3, 1)
    # Since base returns False for all robots, Exist should be False
    assert torch.all(~exist_result)


def test_quantified_predicate_edge_cases() -> None:
    """Test edge cases for quantified predicates."""
    # Configure and create TAMP system
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Get types
    robot_type = tamp_system.components.type_container.as_dict()["robot"]

    # Create unary predicate
    unary_predicate = Predicate("IsActive", [robot_type])

    def dummy_interpreter(obs: Tensor, objects: List[Sequence[Object]]) -> Tensor:
        batch_size = obs.shape[0]
        num_groundings = len(objects)
        return torch.ones(
            (batch_size, num_groundings), dtype=torch.bool, device=obs.device
        )

    # Test invalid quantifier
    with pytest.raises(ValueError, match="Quantifier must be 'ForAll', 'Exist', or ''"):
        create_quantified_predicate(
            unary_predicate, dummy_interpreter, tamp_system, "Invalid", 0
        )

    # Test invalid variable_id
    with pytest.raises(IndexError, match="Variable ID 1 out of bounds"):
        create_quantified_predicate(
            unary_predicate, dummy_interpreter, tamp_system, "ForAll", 1
        )

    # Test quantification over unary predicate (should result in nullary predicate)
    nullary_pred, nullary_interp = create_quantified_predicate(
        unary_predicate, dummy_interpreter, tamp_system, "ForAll", 0
    )

    assert nullary_pred.name == "ForAll_robot_0_IsActive"
    assert nullary_pred.arity == 0

    # Test nullary predicate execution
    obs = torch.randn(2, 100)
    nullary_objects: List[Sequence[Object]] = [
        []
    ]  # Empty grounding for nullary predicate

    nullary_result = nullary_interp(obs, nullary_objects)
    assert nullary_result.shape == (2, 1)
    # Should be True since base interpreter returns True for the single robot
    assert torch.all(nullary_result)


def test_quantified_predicate_different_base_results() -> None:
    """Test quantifiers with mixed True/False base results."""
    # Configure and create TAMP system
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Get types and objects
    robot_type = tamp_system.components.type_container.as_dict()["robot"]
    block_type = tamp_system.components.type_container.as_dict()["block"]

    # Create binary predicate
    mixed_predicate = Predicate("Mixed", [robot_type, block_type])

    def mixed_interpreter(obs: Tensor, objects: List[Sequence[Object]]) -> Tensor:
        """Interpreter that returns alternating True/False based on object names."""
        batch_size = obs.shape[0]
        num_groundings = len(objects)
        results = torch.zeros(
            (batch_size, num_groundings), dtype=torch.bool, device=obs.device
        )

        for i, obj_seq in enumerate(objects):
            # Return True if the second object (block) is "grasp_block", False
            # otherwise
            if len(obj_seq) >= 2 and obj_seq[1].name == "grasp_block":
                results[:, i] = True
            else:
                results[:, i] = False

        return results

    # Test ForAll quantification over block
    _, forall_interp = create_quantified_predicate(
        mixed_predicate, mixed_interpreter, tamp_system, "ForAll", 1
    )

    # Test Exist quantification over block
    _, exist_interp = create_quantified_predicate(
        mixed_predicate, mixed_interpreter, tamp_system, "Exist", 1
    )

    obs = torch.randn(2, 100)
    robot_obj = tamp_system.perceiver.objects["robot"]

    # For quantification over block, provide robot objects
    robot_objects: List[Sequence[Object]] = [[robot_obj]]

    forall_result = forall_interp(obs, robot_objects)
    exist_result = exist_interp(obs, robot_objects)

    # Since base returns True for grasp_block and False for base_block:
    # - ForAll should be False (not all blocks satisfy the predicate)
    # - Exist should be True (at least one block satisfies the predicate)
    assert torch.all(~forall_result)  # ForAll is False
    assert torch.all(exist_result)  # Exist is True
