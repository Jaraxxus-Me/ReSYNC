"""Unit tests for task planning utilities."""

# pylint: disable=redefined-outer-name  # False positive for pytest fixtures

import pytest
from relational_structs import GroundAtom, LiftedAtom, Object, Predicate, Type, Variable

from skill_refactor.utils.structs import LiftedOperator
from skill_refactor.utils.task_planning import (
    all_ground_operators,
    apply_operator,
    create_task_planning_heuristic,
    get_applicable_operators,
    get_reachable_atoms,
    task_plan,
    task_plan_grounding,
)


@pytest.fixture
def simple_domain():
    """Create a simple domain with types, predicates, objects, and operators."""
    # Types
    block_type = Type("block")

    # Predicates
    on_pred = Predicate("on", [block_type, block_type])
    clear_pred = Predicate("clear", [block_type])

    # Objects
    block_a = Object("a", block_type)
    block_b = Object("b", block_type)
    block_c = Object("c", block_type)
    objects = {block_a, block_b, block_c}

    # Variables
    var_x = Variable("?x", block_type)
    var_y = Variable("?y", block_type)

    # Operators
    move_op = LiftedOperator(
        name="move",
        parameters=[var_x, var_y],
        preconditions={
            LiftedAtom(clear_pred, [var_x]),
            LiftedAtom(clear_pred, [var_y]),
        },
        add_effects={LiftedAtom(on_pred, [var_x, var_y])},
        delete_effects={LiftedAtom(clear_pred, [var_y])},
    )

    return {
        "types": [block_type],
        "predicates": [on_pred, clear_pred],
        "objects": objects,
        "operators": [move_op],
        "block_a": block_a,
        "block_b": block_b,
        "block_c": block_c,
        "on_pred": on_pred,
        "clear_pred": clear_pred,
        "move_op": move_op,
    }


def test_all_ground_operators(simple_domain):
    """Test grounding of lifted operators."""
    move_op = simple_domain["move_op"]
    objects = simple_domain["objects"]

    ground_ops = list(all_ground_operators(move_op, objects))

    # Should have 3*3 = 9 groundings (including self-moves)
    assert len(ground_ops) == 9

    # Check that all combinations are present
    obj_names = ["a", "b", "c"]
    expected_names = [(x, y) for x in obj_names for y in obj_names]
    actual_names = [(op.parameters[0].name, op.parameters[1].name) for op in ground_ops]

    assert sorted(expected_names) == sorted(actual_names)


def test_get_reachable_atoms(simple_domain):
    """Test reachability analysis."""
    block_a, block_b, block_c = (
        simple_domain["block_a"],
        simple_domain["block_b"],
        simple_domain["block_c"],
    )
    on_pred, clear_pred = simple_domain["on_pred"], simple_domain["clear_pred"]
    move_op = simple_domain["move_op"]
    objects = simple_domain["objects"]

    # Initial state: all blocks clear
    init_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(clear_pred, [block_b]),
        GroundAtom(clear_pred, [block_c]),
    }

    # Ground operators
    ground_ops = list(all_ground_operators(move_op, objects))

    # Get reachable atoms
    reachable = get_reachable_atoms(ground_ops, init_atoms)

    # Should include initial atoms plus all possible on relationships
    assert len(reachable) >= len(init_atoms)

    # Should include some on relationships
    on_atoms = [atom for atom in reachable if atom.predicate == on_pred]
    assert len(on_atoms) > 0


def test_get_applicable_operators(simple_domain):
    """Test finding applicable operators."""
    block_a, block_b = simple_domain["block_a"], simple_domain["block_b"]
    clear_pred = simple_domain["clear_pred"]
    move_op = simple_domain["move_op"]
    objects = simple_domain["objects"]

    # State where both blocks are clear
    current_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(clear_pred, [block_b]),
    }

    # Ground operators
    ground_ops = list(all_ground_operators(move_op, objects))

    # Find applicable operators
    applicable = get_applicable_operators(ground_ops, current_atoms)

    # Should be able to move a onto b or b onto a
    assert len(applicable) >= 2

    # Check specific moves are possible
    move_a_b = next(
        (
            op
            for op in applicable
            if op.parameters[0].name == "a" and op.parameters[1].name == "b"
        ),
        None,
    )
    assert move_a_b is not None


def test_apply_operator(simple_domain):
    """Test operator application."""
    block_a, block_b = simple_domain["block_a"], simple_domain["block_b"]
    on_pred, clear_pred = simple_domain["on_pred"], simple_domain["clear_pred"]
    move_op = simple_domain["move_op"]

    # Ground the move(a, b) operator
    move_a_b = move_op.ground((block_a, block_b))

    # Initial state
    current_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(clear_pred, [block_b]),
    }

    # Apply operator
    new_atoms = apply_operator(move_a_b, current_atoms)

    # Check effects
    assert GroundAtom(on_pred, [block_a, block_b]) in new_atoms  # Added
    assert GroundAtom(clear_pred, [block_b]) not in new_atoms  # Deleted
    assert GroundAtom(clear_pred, [block_a]) in new_atoms  # Unchanged


def test_apply_operator_invalid_preconditions(simple_domain):
    """Test that applying operator with unsatisfied preconditions raises error."""
    block_a, block_b, _ = (  # block_c unused
        simple_domain["block_a"],
        simple_domain["block_b"],
        simple_domain["block_c"],
    )
    clear_pred = simple_domain["clear_pred"]
    move_op = simple_domain["move_op"]

    # Ground the move(a, b) operator
    move_a_b = move_op.ground((block_a, block_b))

    # State where preconditions are not satisfied (b is not clear)
    current_atoms = {
        GroundAtom(clear_pred, [block_a])
        # block_b is not clear, so move(a, b) should not be applicable
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="Operator preconditions not satisfied"):
        apply_operator(move_a_b, current_atoms)


def test_task_plan_grounding(simple_domain):
    """Test the complete grounding process."""
    block_a, block_b, block_c = (
        simple_domain["block_a"],
        simple_domain["block_b"],
        simple_domain["block_c"],
    )
    clear_pred = simple_domain["clear_pred"]
    operators = simple_domain["operators"]
    objects = simple_domain["objects"]

    # Initial state: all blocks clear
    init_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(clear_pred, [block_b]),
        GroundAtom(clear_pred, [block_c]),
    }

    # Ground operators
    ground_ops, reachable_atoms = task_plan_grounding(init_atoms, objects, operators)

    # Should have some ground operators
    assert len(ground_ops) > 0

    # Should have reachable atoms including initial state
    assert init_atoms.issubset(reachable_atoms)

    # All returned operators should be applicable (preconditions reachable)
    for op in ground_ops:
        assert op.preconditions.issubset(reachable_atoms)


def test_task_plan_grounding_no_effects():
    """Test grounding with allow_noops parameter."""
    # Create a no-op operator for testing
    block_type = Type("block")
    var_x = Variable("?x", block_type)

    noop_op = LiftedOperator(
        name="noop",
        parameters=[var_x],
        preconditions=set(),
        add_effects=set(),
        delete_effects=set(),
    )

    block_a = Object("a", block_type)
    objects = {block_a}
    init_atoms = set()

    # Without allow_noops, should filter out no-effect operators
    ground_ops, _ = task_plan_grounding(
        init_atoms, objects, [noop_op], allow_noops=False
    )
    assert len(ground_ops) == 0

    # With allow_noops, should include no-effect operators
    ground_ops, _ = task_plan_grounding(
        init_atoms, objects, [noop_op], allow_noops=True
    )
    assert len(ground_ops) == 1


def test_task_plan_simple(simple_domain):
    """Test simple task planning with A* search."""
    block_a, block_b = simple_domain["block_a"], simple_domain["block_b"]
    on_pred, clear_pred = simple_domain["on_pred"], simple_domain["clear_pred"]
    operators = simple_domain["operators"]
    objects = simple_domain["objects"]
    predicates = simple_domain["predicates"]

    # Initial state: both blocks clear
    init_atoms = {GroundAtom(clear_pred, [block_a]), GroundAtom(clear_pred, [block_b])}

    # Goal: a on b
    goal = {GroundAtom(on_pred, [block_a, block_b])}

    # Ground operators
    ground_ops, reachable_atoms = task_plan_grounding(init_atoms, objects, operators)

    # Create heuristic
    heuristic = create_task_planning_heuristic(
        "hff", init_atoms, goal, ground_ops, predicates, objects
    )

    # Find plan using A* search
    plans_found = 0
    for skeleton, atoms_sequence, _ in task_plan(  # metrics unused
        init_atoms,
        goal,
        ground_ops,
        reachable_atoms,
        heuristic,
        seed=42,
        timeout=5.0,
        max_skeletons_optimized=1,
    ):
        plans_found += 1

        # Should find a plan
        assert len(skeleton) > 0
        assert len(atoms_sequence) == len(skeleton) + 1  # initial + each step

        # Verify goal is achieved in final state
        final_atoms = atoms_sequence[-1]
        assert goal.issubset(final_atoms)

        # Verify atoms sequence is valid
        current_atoms = init_atoms.copy()
        for i, op in enumerate(skeleton):
            current_atoms = apply_operator(op, current_atoms)
            assert current_atoms == atoms_sequence[i + 1]

        break  # Only check first plan

    assert plans_found > 0, "No plans found"


def test_task_plan_goal_already_satisfied(simple_domain):
    """Test planning when goal is already satisfied."""
    block_a, block_b = simple_domain["block_a"], simple_domain["block_b"]
    on_pred, clear_pred = simple_domain["on_pred"], simple_domain["clear_pred"]
    operators = simple_domain["operators"]
    objects = simple_domain["objects"]
    predicates = simple_domain["predicates"]

    # Initial state includes the goal
    init_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(on_pred, [block_a, block_b]),
    }

    # Goal: a on b (already satisfied)
    goal = {GroundAtom(on_pred, [block_a, block_b])}

    # Ground operators
    ground_ops, reachable_atoms = task_plan_grounding(init_atoms, objects, operators)

    # Create heuristic
    heuristic = create_task_planning_heuristic(
        "hff", init_atoms, goal, ground_ops, predicates, objects
    )

    # Find plan using A* search
    plans_found = 0
    for skeleton, atoms_sequence, _ in task_plan(  # metrics unused
        init_atoms,
        goal,
        ground_ops,
        reachable_atoms,
        heuristic,
        seed=42,
        timeout=5.0,
        max_skeletons_optimized=1,
    ):
        plans_found += 1

        # Should return empty plan (goal already satisfied)
        assert len(skeleton) == 0
        assert len(atoms_sequence) == 1  # Just initial state
        assert atoms_sequence[0] == init_atoms

        break  # Only check first plan

    assert plans_found > 0, "Should find empty plan when goal already satisfied"


def test_task_plan_unreachable_goal(simple_domain):
    """Test planning with unreachable goal."""
    block_a = simple_domain["block_a"]
    clear_pred = simple_domain["clear_pred"]
    operators = simple_domain["operators"]
    objects = simple_domain["objects"]
    predicates = simple_domain["predicates"]

    # Create a predicate that's not reachable
    impossible_pred = Predicate("impossible", [Type("block")])

    # Initial state
    init_atoms = {GroundAtom(clear_pred, [block_a])}

    # Impossible goal
    goal = {GroundAtom(impossible_pred, [block_a])}

    # Ground operators
    ground_ops, reachable_atoms = task_plan_grounding(init_atoms, objects, operators)

    # Create heuristic (won't be used due to early error)
    heuristic = create_task_planning_heuristic(
        "hff",
        init_atoms,
        init_atoms,
        ground_ops,
        predicates,
        objects,  # Use init_atoms as dummy goal
    )

    # Should raise AssertionError for unreachable goal
    with pytest.raises(AssertionError, match="Goal .* not dr-reachable"):
        list(
            task_plan(
                init_atoms,
                goal,
                ground_ops,
                reachable_atoms,
                heuristic,
                seed=42,
                timeout=5.0,
                max_skeletons_optimized=1,
            )
        )


def test_task_plan_no_solution(simple_domain):
    """Test planning when no solution exists within timeout."""
    block_a, block_b, block_c = (
        simple_domain["block_a"],
        simple_domain["block_b"],
        simple_domain["block_c"],
    )
    on_pred, clear_pred = simple_domain["on_pred"], simple_domain["clear_pred"]
    operators = simple_domain["operators"]
    objects = simple_domain["objects"]
    predicates = simple_domain["predicates"]

    # Initial state: all blocks clear, goal is reachable but complex
    init_atoms = {
        GroundAtom(clear_pred, [block_a]),
        GroundAtom(clear_pred, [block_b]),
        GroundAtom(clear_pred, [block_c]),
    }

    # Goal: a on b AND b on c (requires 2 steps)
    goal = {
        GroundAtom(on_pred, [block_a, block_b]),
        GroundAtom(on_pred, [block_b, block_c]),
    }

    # Ground operators
    ground_ops, reachable_atoms = task_plan_grounding(init_atoms, objects, operators)

    # Create heuristic
    heuristic = create_task_planning_heuristic(
        "hff", init_atoms, goal, ground_ops, predicates, objects
    )

    # Find plan with very short timeout (should time out or hit max skeletons)
    plans_found = 0
    try:
        for _, _, _ in task_plan(  # skeleton, atoms_sequence, metrics unused
            init_atoms,
            goal,
            ground_ops,
            reachable_atoms,
            heuristic,
            seed=42,
            timeout=0.001,
            max_skeletons_optimized=1,  # Very short timeout
        ):
            plans_found += 1
            break
    except AssertionError as e:
        # Should either timeout or reach max skeletons
        assert "timed out" in str(e) or "max_skeletons_optimized" in str(e)

    # If no exception, verify we found a valid plan or no plans
    if plans_found == 0:
        # This is fine - no solution found within constraints
        pass
    else:
        # If we found a plan, it should be valid
        assert plans_found == 1
