"""Unit tests for operator to AE vector conversion utilities."""

import torch
from relational_structs import (
    LiftedAtom,
    Predicate,
    Type,
    Variable,
)

from skill_refactor.approaches.pred_learner.utils import operator_to_ae_vector
from skill_refactor.utils.structs import LiftedOperator


def test_operator_to_ae_vector_basic() -> None:
    """Test basic functionality of operator_to_ae_vector."""
    # Create types
    robot_type = Type("robot")
    entity_type = Type("entity")

    # Create predicates
    holding = Predicate("Holding", [robot_type, entity_type])
    on = Predicate("On", [entity_type, entity_type])

    # Create variables
    robot = Variable("?robot", robot_type)
    obj1 = Variable("?obj1", entity_type)
    obj2 = Variable("?obj2", entity_type)

    # Create operator 1: Grasp - adds Holding(?robot, ?obj1)
    grasp_op = LiftedOperator(
        name="Grasp",
        parameters=[robot, obj1],
        preconditions=set(),
        add_effects={LiftedAtom(holding, [robot, obj1])},
        delete_effects=set(),
    )

    # Create operator 2: Place - deletes Holding(?robot, ?obj1), adds On(?obj1, ?obj2)
    place_op = LiftedOperator(
        name="Place",
        parameters=[robot, obj1, obj2],
        preconditions={LiftedAtom(holding, [robot, obj1])},
        add_effects={LiftedAtom(on, [obj1, obj2])},
        delete_effects={LiftedAtom(holding, [robot, obj1])},
    )

    # Test the function
    operator_set = {grasp_op, place_op}
    name_list = ["Grasp", "Place"]

    result = operator_to_ae_vector(operator_set, name_list)

    # Check that we have the right predicates
    assert len(result) == 2
    assert holding in result  # type: ignore[comparison-overlap]
    assert on in result  # type: ignore[comparison-overlap]

    # Check Holding predicate
    holding_data = result[holding]  # type: ignore[index]
    assert "ae_vector" in holding_data
    assert "var_bind_idx" in holding_data

    # Holding should have add effect for Grasp (index 0) and delete effect for Place (index 1)
    holding_ae = holding_data["ae_vector"]
    assert holding_ae.shape == (2, 2)  # [num_ops, 2]
    assert holding_ae[0, 0] == 1.0  # Grasp adds Holding
    assert holding_ae[0, 1] == 0.0  # Grasp doesn't delete Holding
    assert holding_ae[1, 0] == 0.0  # Place doesn't add Holding
    assert holding_ae[1, 1] == 1.0  # Place deletes Holding

    # Check variable bindings for Holding (arity 2: robot, entity)
    holding_var_bind = holding_data["var_bind_idx"]
    assert holding_var_bind.shape == (2,)  # arity of Holding
    assert holding_var_bind[0] == 0  # robot parameter (index 0 among robot types)
    assert holding_var_bind[1] == 0  # obj1 parameter (index 0 among entity types)

    # Check On predicate
    on_data = result[on]  # type: ignore[index]
    on_ae = on_data["ae_vector"]
    assert on_ae.shape == (2, 2)
    assert on_ae[0, 0] == 0.0  # Grasp doesn't add On
    assert on_ae[0, 1] == 0.0  # Grasp doesn't delete On
    assert on_ae[1, 0] == 1.0  # Place adds On
    assert on_ae[1, 1] == 0.0  # Place doesn't delete On

    # Check variable bindings for On (arity 2: entity, entity)
    on_var_bind = on_data["var_bind_idx"]
    assert on_var_bind.shape == (2,)
    assert on_var_bind[0] == 0  # obj1 parameter (index 0 among entity types)
    assert on_var_bind[1] == 1  # obj2 parameter (index 1 among entity types)


def test_operator_to_ae_vector_multiple_same_type() -> None:
    """Test with multiple parameters of the same type - should create separate predicate instances."""
    # Create types
    entity_type = Type("entity")

    # Create predicate
    on = Predicate("On", [entity_type, entity_type])

    # Create variables
    obj1 = Variable("?obj1", entity_type)
    obj2 = Variable("?obj2", entity_type)
    obj3 = Variable("?obj3", entity_type)

    # Create operator: Stack - adds On(?obj1, ?obj2), deletes On(?obj2, ?obj3)
    stack_op = LiftedOperator(
        name="Stack",
        parameters=[obj1, obj2, obj3],  # All entity type
        preconditions=set(),
        add_effects={LiftedAtom(on, [obj1, obj2])},
        delete_effects={LiftedAtom(on, [obj2, obj3])},
    )

    operator_set = {stack_op}
    name_list = ["Stack"]

    result = operator_to_ae_vector(operator_set, name_list)

    # Should now have 2 separate predicate instances
    assert len(result) == 2

    # Find the predicates by their names
    on_predicates = [pred for pred in result if pred.name.startswith("On")]
    assert len(on_predicates) == 2

    # Sort by name to get consistent ordering
    on_predicates.sort(key=lambda p: p.name)

    # First predicate (On) should have add effect with var_bind [0, 1]
    on1_data = result[on_predicates[0]]
    on1_ae = on1_data["ae_vector"]
    assert on1_ae.shape == (1, 2)
    assert on1_ae[0, 0] == 1.0  # Add effect
    assert on1_ae[0, 1] == 0.0  # No delete effect
    assert torch.equal(on1_data["var_bind_idx"], torch.tensor([0, 1]))  # obj1, obj2

    # Second predicate (On2) should have delete effect with var_bind [1, 2]
    on2_data = result[on_predicates[1]]
    on2_ae = on2_data["ae_vector"]
    assert on2_ae.shape == (1, 2)
    assert on2_ae[0, 0] == 0.0  # No add effect
    assert on2_ae[0, 1] == 1.0  # Delete effect
    assert torch.equal(on2_data["var_bind_idx"], torch.tensor([1, 2]))  # obj2, obj3


def test_operator_to_ae_vector_missing_operator() -> None:
    """Test with operator name not in the set."""
    entity_type = Type("entity")
    on = Predicate("On", [entity_type, entity_type])

    obj1 = Variable("?obj1", entity_type)
    obj2 = Variable("?obj2", entity_type)

    place_op = LiftedOperator(
        name="Place",
        parameters=[obj1, obj2],
        preconditions=set(),
        add_effects={LiftedAtom(on, [obj1, obj2])},
        delete_effects=set(),
    )

    operator_set = {place_op}
    name_list = ["Place", "NonExistentOp"]  # Include non-existent operator

    result = operator_to_ae_vector(operator_set, name_list)

    assert len(result) == 1
    assert on in result  # type: ignore[comparison-overlap]

    on_data = result[on]  # type: ignore[index]
    on_ae = on_data["ae_vector"]
    assert on_ae.shape == (2, 2)  # Still 2 operators in name_list
    assert on_ae[0, 0] == 1.0  # Place adds On
    assert on_ae[1, 0] == 0.0  # NonExistentOp doesn't add On (should remain 0)
    assert on_ae[1, 1] == 0.0  # NonExistentOp doesn't delete On


def test_operator_to_ae_vector_no_effects() -> None:
    """Test with operator that has no effects on tracked predicates."""
    robot_type = Type("robot")
    entity_type = Type("entity")

    holding = Predicate("Holding", [robot_type, entity_type])

    robot = Variable("?robot", robot_type)
    obj1 = Variable("?obj1", entity_type)

    # Operator with no effects
    noop_op = LiftedOperator(
        name="NoOp",
        parameters=[robot, obj1],
        preconditions={LiftedAtom(holding, [robot, obj1])},
        add_effects=set(),
        delete_effects=set(),
    )

    operator_set = {noop_op}
    name_list = ["NoOp"]

    result = operator_to_ae_vector(operator_set, name_list)

    # Should have empty result since operator has no effects
    # (function only processes effects, not preconditions)
    assert len(result) == 0


def test_predicate_splitting_example() -> None:
    """Test the exact example from the problem description: On1 [1,0] var[0,1], On2 [0,1] var[1,2]."""
    # Create types
    entity_type = Type("entity")

    # Create predicate
    on = Predicate("On", [entity_type, entity_type])

    # Create variables
    obj1 = Variable("?obj1", entity_type)
    obj2 = Variable("?obj2", entity_type)
    obj3 = Variable("?obj3", entity_type)

    # Create operator: Stack - adds On(?obj1, ?obj2), deletes On(?obj2, ?obj3)
    stack_op = LiftedOperator(
        name="Stack",
        parameters=[obj1, obj2, obj3],
        preconditions=set(),
        add_effects={LiftedAtom(on, [obj1, obj2])},  # On(?obj1, ?obj2)
        delete_effects={LiftedAtom(on, [obj2, obj3])},  # On(?obj2, ?obj3)
    )

    operator_set = {stack_op}
    name_list = ["Stack"]

    result = operator_to_ae_vector(operator_set, name_list)

    # Should have 2 predicates: On and On2
    assert len(result) == 2

    # Find predicates
    predicates = list(result.keys())
    predicates.sort(key=lambda p: p.name)  # Sort to ensure consistent order

    on1 = predicates[0]  # "On"
    on2 = predicates[1]  # "On2"

    assert on1.name == "On"
    assert on2.name == "On2"

    # Check On1: eff_vect [1, 0], var [0, 1]
    on1_data = result[on1]
    assert torch.equal(
        on1_data["ae_vector"], torch.tensor([[1.0, 0.0]])
    )  # [add=1, delete=0]
    assert torch.equal(
        on1_data["var_bind_idx"], torch.tensor([0, 1])
    )  # [obj1=0, obj2=1]

    # Check On2: eff_vect [0, 1], var [1, 2]
    on2_data = result[on2]
    assert torch.equal(
        on2_data["ae_vector"], torch.tensor([[0.0, 1.0]])
    )  # [add=0, delete=1]
    assert torch.equal(
        on2_data["var_bind_idx"], torch.tensor([1, 2])
    )  # [obj2=1, obj3=2]


if __name__ == "__main__":
    test_operator_to_ae_vector_basic()
    test_operator_to_ae_vector_multiple_same_type()
    test_operator_to_ae_vector_missing_operator()
    test_operator_to_ae_vector_no_effects()
    test_predicate_splitting_example()
    print("All tests passed!")
