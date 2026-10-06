"""Task planning utilities for STRIPS-style planning."""

import functools
import heapq as hq
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Collection, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from pyperplan.heuristics.heuristic_base import Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS
from relational_structs import GroundAtom, Object

from skill_refactor.utils.structs import (
    GroundAtomTrajectory,
    GroundOperator,
    LiftedOperator,
    Predicate,
    Segment,
    Task,
    Type,
    Variable,
)


@dataclass(repr=False, eq=False)
class _Node:
    """A node for the search over skeletons."""

    atoms: Set[GroundAtom]
    skeleton: List[GroundOperator]
    atoms_sequence: List[Set[GroundAtom]]  # expected state sequence
    parent: Optional["_Node"]
    cumulative_cost: float


@dataclass(frozen=True)
class _TaskPlanningHeuristic:
    """A task planning heuristic."""

    name: str
    init_atoms: Collection[GroundAtom]
    goal: Set[GroundAtom]
    ground_ops: Collection[GroundOperator]

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        raise NotImplementedError("Override me!")


# Helper functions for entity combinations, var bindings and ground atoms
# for abstract planning.
def get_object_combinations(
    objects: Collection[Object], types: Sequence[Type], skip_self: bool = False
) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
    sorted_entities = sorted(objects)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        if skip_self and len(choice) != len(set(choice)):
            continue
        yield list(choice)


def get_variable_combinations(
    variables: Collection[Variable], types: Sequence[Type]
) -> Iterator[List[Variable]]:
    """Get all combinations of variables satisfying the given types sequence."""
    sorted_entities = sorted(variables)
    choices = []
    for vt in types:
        this_choices = []
        for ent in sorted_entities:
            if ent.is_instance(vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_all_ground_atoms_for_predicate(
    predicate: Predicate, objects: Collection[Object]
) -> Set[GroundAtom]:
    """Get all groundings of the predicate given objects.

    Note: we don't want lru_cache() on this function because we might want
    to call it with stripped predicates, and we wouldn't want it to return
    cached values.
    """
    ground_atoms = set()
    for args in get_object_combinations(objects, predicate.types):
        ground_atom = GroundAtom(predicate, args)  # type: ignore[arg-type]
        ground_atoms.add(ground_atom)
    return ground_atoms


def get_static_atoms(
    ground_ops: Collection[GroundOperator], atoms: Collection[GroundAtom]
) -> Set[GroundAtom]:
    """Get the subset of atoms from the given set that are static with respect to the
    given ground operators.

    Note that this can include MORE than simply the set of atoms whose predicates are
    static, because now we have ground operators.
    """
    static_atoms = set()
    for atom in atoms:
        # This atom is not static if it appears in any op's effects.
        if any(
            any(atom == eff for eff in op.add_effects)
            or any(atom == eff for eff in op.delete_effects)
            for op in ground_ops
        ):
            continue
        static_atoms.add(atom)
    return static_atoms


def prune_ground_atom_dataset(
    ground_atom_dataset: List[GroundAtomTrajectory],
    kept_predicates: Collection[Predicate],
) -> List[GroundAtomTrajectory]:
    """Create a new ground atom dataset by keeping only some predicates."""
    new_ground_atom_dataset = []
    for traj, atoms in ground_atom_dataset:
        assert len(traj.states) == len(atoms)
        kept_atoms = [
            {a for a in sa if a.predicate in kept_predicates}  # type: ignore[comparison-overlap]
            for sa in atoms
        ]
        new_ground_atom_dataset.append((traj, kept_atoms))
    return new_ground_atom_dataset


def segment_trajectory_to_atoms_sequence(
    seg_traj: List[Segment],
) -> List[Set[GroundAtom]]:
    """Convert a trajectory of segments into a trajectory of ground atoms.

    The length of the return value will always be one greater than the length of the
    given seg_traj.
    """
    assert len(seg_traj) >= 1
    atoms_seq = []
    for i, seg in enumerate(seg_traj):
        atoms_seq.append(seg.init_atoms)
        if i < len(seg_traj) - 1:
            assert seg.final_atoms == seg_traj[i + 1].init_atoms
    atoms_seq.append(seg_traj[-1].final_atoms)
    assert len(atoms_seq) == len(seg_traj) + 1
    return atoms_seq


############################### Pyperplan Glue ###############################


def _create_pyperplan_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> "_PyperplanHeuristicWrapper":
    """Create a pyperplan heuristic that inherits from _TaskPlanningHeuristic."""
    assert heuristic_name in _PYPERPLAN_HEURISTICS
    static_atoms = get_static_atoms(ground_ops, init_atoms)
    pyperplan_heuristic_cls = _PYPERPLAN_HEURISTICS[heuristic_name]
    pyperplan_task = _create_pyperplan_task(
        init_atoms, goal, ground_ops, predicates, objects, static_atoms
    )
    pyperplan_heuristic = pyperplan_heuristic_cls(pyperplan_task)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    return _PyperplanHeuristicWrapper(
        heuristic_name,
        init_atoms,
        goal,
        ground_ops,
        static_atoms,
        pyperplan_heuristic,
        pyperplan_goal,
    )


_PyperplanFacts = FrozenSet[str]


@dataclass(frozen=True)
class _PyperplanNode:
    """Container glue for pyperplan heuristics."""

    state: _PyperplanFacts
    goal: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanOperator:
    """Container glue for pyperplan heuristics."""

    name: str
    preconditions: _PyperplanFacts
    add_effects: _PyperplanFacts
    del_effects: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanTask:
    """Container glue for pyperplan heuristics."""

    facts: _PyperplanFacts
    initial_state: _PyperplanFacts
    goals: _PyperplanFacts
    operators: Collection[_PyperplanOperator]


@dataclass(frozen=True)
class _PyperplanHeuristicWrapper(_TaskPlanningHeuristic):
    """A light wrapper around pyperplan's heuristics."""

    _static_atoms: Set[GroundAtom]
    _pyperplan_heuristic: _PyperplanBaseHeuristic
    _pyperplan_goal: _PyperplanFacts

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        # Note: filtering out static atoms.
        pyperplan_facts = _atoms_to_pyperplan_facts(set(atoms) - self._static_atoms)
        return self._evaluate(
            pyperplan_facts, self._pyperplan_goal, self._pyperplan_heuristic
        )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate(
        pyperplan_facts: _PyperplanFacts,
        pyperplan_goal: _PyperplanFacts,
        pyperplan_heuristic: _PyperplanBaseHeuristic,
    ) -> float:
        pyperplan_node = _PyperplanNode(pyperplan_facts, pyperplan_goal)
        logging.disable(logging.DEBUG)
        result = pyperplan_heuristic(pyperplan_node)
        logging.disable(logging.NOTSET)
        return result


def _create_pyperplan_task(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
    static_atoms: Set[GroundAtom],
) -> _PyperplanTask:
    """Helper glue for pyperplan heuristics."""
    all_atoms = set()
    for predicate in predicates:
        all_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, frozenset(objects))
        )
    # Note: removing static atoms.
    pyperplan_facts = _atoms_to_pyperplan_facts(all_atoms - static_atoms)
    pyperplan_state = _atoms_to_pyperplan_facts(init_atoms - static_atoms)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    pyperplan_operators = set()
    for op in ground_ops:
        # Note: the pyperplan operator must include the objects, because hFF
        # uses the operator name in constructing the relaxed plan, and the
        # relaxed plan is a set. If we instead just used op.name, there would
        # be a very nasty bug where two ground operators in the relaxed plan
        # that have different objects are counted as just one.
        name = op.name + "-".join(o.name for o in op.parameters)
        pyperplan_operator = _PyperplanOperator(
            name,
            # Note: removing static atoms from preconditions.
            _atoms_to_pyperplan_facts(op.preconditions - static_atoms),
            _atoms_to_pyperplan_facts(op.add_effects),
            _atoms_to_pyperplan_facts(op.delete_effects),
        )
        pyperplan_operators.add(pyperplan_operator)
    return _PyperplanTask(
        pyperplan_facts, pyperplan_state, pyperplan_goal, pyperplan_operators
    )


@functools.lru_cache(maxsize=None)
def _atom_to_pyperplan_fact(atom: GroundAtom) -> str:
    """Convert atom to tuple for interface with pyperplan."""
    arg_str = " ".join(o.name for o in atom.objects)
    return f"({atom.predicate.name} {arg_str})"


def _atoms_to_pyperplan_facts(atoms: Collection[GroundAtom]) -> _PyperplanFacts:
    """Light wrapper around _atom_to_pyperplan_fact() that operates on a collection of
    atoms."""
    return frozenset({_atom_to_pyperplan_fact(atom) for atom in atoms})


############################## End Pyperplan Glue ##############################


def create_task_planning_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _TaskPlanningHeuristic:
    """Create a task planning heuristic that consumes ground atoms and estimates the
    cost-to-go."""
    if heuristic_name in _PYPERPLAN_HEURISTICS:
        return _create_pyperplan_heuristic(
            heuristic_name, init_atoms, goal, ground_ops, predicates, objects
        )
    raise ValueError(f"Unrecognized heuristic name: {heuristic_name}.")


def all_ground_operators(
    lifted_operator: LiftedOperator, objects: Set[Object]
) -> Iterator[GroundOperator]:
    """Generate all possible groundings of a lifted operator with given objects."""
    object_list = list(objects)
    for obj_combo in itertools.product(
        object_list, repeat=len(lifted_operator.parameters)
    ):
        # Check type compatibility more carefully
        compatible = True
        for obj, param in zip(obj_combo, lifted_operator.parameters):
            if hasattr(param, "type"):
                if hasattr(obj, "type") and obj.type != param.type:
                    compatible = False
                    break
            elif hasattr(param, "types"):
                if hasattr(obj, "type") and obj.type not in param.types:
                    compatible = False
                    break

        if compatible:
            try:
                yield lifted_operator.ground(tuple(obj_combo))
            except (AttributeError, TypeError):
                # Skip problematic groundings
                continue


def get_reachable_atoms(
    ground_operators: List[GroundOperator], init_atoms: Set[GroundAtom]
) -> Set[GroundAtom]:
    """Compute all atoms reachable from initial atoms using given operators."""
    reachable = set(init_atoms)
    changed = True

    while changed:
        changed = False
        for operator in ground_operators:
            if operator.preconditions.issubset(reachable):
                new_atoms = operator.add_effects - reachable
                if new_atoms:
                    reachable.update(new_atoms)
                    changed = True

    return reachable


def get_applicable_operators(
    ground_operators: List[GroundOperator], current_atoms: Set[GroundAtom]
) -> List[GroundOperator]:
    """Get operators whose preconditions are satisfied by current atoms."""
    return [op for op in ground_operators if op.preconditions.issubset(current_atoms)]


def apply_operator(
    operator: GroundOperator, current_atoms: Set[GroundAtom]
) -> Set[GroundAtom]:
    """Apply operator to current atoms, returning new atom set."""
    if not operator.preconditions.issubset(current_atoms):
        raise ValueError("Operator preconditions not satisfied")

    new_atoms = current_atoms.copy()
    new_atoms.update(operator.add_effects)
    new_atoms -= operator.delete_effects
    return new_atoms


def _filter_lifted_operator_by_types(
    operator: LiftedOperator,
    available_types: Set[Type],
) -> LiftedOperator:
    """Filter a lifted operator to only include predicates with available types.

    Removes any atoms from preconditions/effects whose predicates reference
    base types that don't exist in the available_types set.

    Args:
        operator: The lifted operator to filter
        available_types: Set of types available for grounding

    Returns:
        A new LiftedOperator with filtered preconditions and effects
    """
    # Filter preconditions - keep only atoms whose predicate base_types are all available
    filtered_preconditions = {
        atom
        for atom in operator.preconditions
        if all(
            param_type in available_types
            for param_type in (
                atom.predicate.base_types
                if hasattr(atom.predicate, "base_types")
                else atom.predicate.types
            )
        )
    }

    # Filter add effects
    filtered_add_effects = {
        atom
        for atom in operator.add_effects
        if all(
            param_type in available_types
            for param_type in (
                atom.predicate.base_types
                if hasattr(atom.predicate, "base_types")
                else atom.predicate.types
            )
        )
    }

    # Filter delete effects
    filtered_delete_effects = {
        atom
        for atom in operator.delete_effects
        if all(
            param_type in available_types
            for param_type in (
                atom.predicate.base_types
                if hasattr(atom.predicate, "base_types")
                else atom.predicate.types
            )
        )
    }

    # Return a copy with filtered atoms
    return operator.copy_with(
        preconditions=filtered_preconditions,
        add_effects=filtered_add_effects,
        delete_effects=filtered_delete_effects,
    )


def task_plan_grounding(
    init_atoms: Set[GroundAtom],
    objects: Set[Object],
    operators: Collection[LiftedOperator],
    allow_noops: bool = False,
) -> Tuple[List[GroundOperator], Set[GroundAtom]]:
    """Ground all operators for task planning, filtering out ones that are unreachable
    or have empty effects.

    First filters operators to only include predicates that reference types
    available in the objects set. Also return the set of reachable atoms, which is
    used by task planning to quickly determine if a goal is unreachable.

    Args:
        init_atoms: Initial state atoms
        objects: Objects available for grounding
        operators: Lifted operators to ground
        allow_noops: Whether to allow operators with no effects

    Returns:
        Tuple of (reachable_operators, reachable_atoms)
    """
    # Get available types from objects
    available_types = {obj.type for obj in objects}

    # Filter operators to only include predicates with available types
    applicable_operators = [
        _filter_lifted_operator_by_types(op, available_types) for op in operators
    ]

    # Ground the filtered operators
    ground_operators: List[GroundOperator] = []
    for operator in sorted(applicable_operators):
        for ground_operator in all_ground_operators(operator, objects):
            if allow_noops or (
                ground_operator.add_effects | ground_operator.delete_effects
            ):
                ground_operators.append(ground_operator)

    reachable_atoms = get_reachable_atoms(ground_operators, init_atoms)
    reachable_operators = [
        operator
        for operator in ground_operators
        if operator.preconditions.issubset(reachable_atoms)
    ]
    return reachable_operators, reachable_atoms


def task_plan(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_operators: List[GroundOperator],
    reachable_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = False,
) -> Iterator[Tuple[List[GroundOperator], List[Set[GroundAtom]], dict]]:
    """Run only the task planning portion of SeSamE. A* search is run, and skeletons
    that achieve the goal symbolically are yielded. Specifically, yields a tuple of
    (skeleton, atoms sequence, metrics dictionary).

    This method is NOT used by SeSamE, but is instead provided as a convenient wrapper
    around _skeleton_generator below (which IS used by SeSamE) that takes in only the
    minimal necessary arguments.

    This method is tightly coupled with task_plan_grounding -- the reason they are
    separate methods is that it is sometimes possible to ground only once and then plan
    multiple times (e.g. from different initial states, or to different goals). To run
    task planning once, call task_plan_grounding to get ground_operators and
    reachable_atoms; then create a heuristic using utils.create_task_planning_heuristic;
    then call this method. See the tests in tests/test_planning for usage examples.
    """
    if not goal.issubset(reachable_atoms):
        # logging.info(f"Detected goal unreachable. Goal: {goal}")
        # logging.info(f"Initial atoms: {init_atoms}")
        raise AssertionError(f"Goal {goal} not dr-reachable")
    dummy_task = Task(torch.tensor([]), set(), goal, 0)  # dummy init and train_task_idx
    metrics: dict = defaultdict(float)
    generator = _skeleton_generator(
        dummy_task,
        ground_operators,
        init_atoms,
        heuristic,
        seed,
        timeout,
        metrics,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set,
    )
    # Note that we use this pattern to avoid having to catch an exception
    # when _skeleton_generator runs out of skeletons to optimize.
    for skeleton, atoms_sequence in islice(generator, max_skeletons_optimized):
        yield skeleton, atoms_sequence, metrics.copy()


def _skeleton_generator(
    task: Task,
    ground_operators: List[GroundOperator],
    init_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    metrics: dict,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = False,
) -> Iterator[Tuple[List[GroundOperator], List[Set[GroundAtom]]]]:
    """A* search over skeletons (sequences of ground operators). Iterates over pairs of
    (skeleton, atoms sequence).

    Note that we can't use utils.run_astar() here because we want to yield multiple
    skeletons, whereas that utility method returns only a single solution. Furthermore,
    it's easier to track and update our metrics dictionary if we re-implement the search
    here. If use_visited_state_set is False (which is the default), then we may revisit
    the same abstract states multiple times, unlike in typical A*. See Issue #1117 for a
    discussion on why this is False by default.
    """
    start_time = time.perf_counter()
    queue: List[Tuple[float, float, _Node]] = []
    root_node = _Node(
        atoms=init_atoms,
        skeleton=[],
        atoms_sequence=[init_atoms],
        parent=None,
        cumulative_cost=0,
    )
    metrics["num_nodes_created"] += 1
    rng_prio = np.random.default_rng(seed)
    hq.heappush(queue, (heuristic(root_node.atoms), rng_prio.uniform(), root_node))
    # Initialize with empty skeleton for root.
    # We want to keep track of the visited skeletons so that we avoid
    # repeatedly outputting the same faulty skeletons.
    visited_skeletons: Set[Tuple[GroundOperator, ...]] = set()
    visited_skeletons.add(tuple(root_node.skeleton))
    if use_visited_state_set:
        # This set will maintain (frozen) atom sets that have been fully
        # expanded already, and ensure that we never expand redundantly.
        visited_atom_sets = set()
    # Start search.
    while queue and (time.perf_counter() - start_time < timeout):
        if int(metrics["num_skeletons_optimized"]) == max_skeletons_optimized:
            raise AssertionError("Planning reached max_skeletons_optimized!")
        _, _, node = hq.heappop(queue)
        if use_visited_state_set:
            frozen_atoms = frozenset(node.atoms)
            visited_atom_sets.add(frozen_atoms)
        # Good debug point #1: print out the skeleton here to see what
        # the high-level search is doing. You can accomplish this via:
        # for act in node.skeleton:
        #     logging.info(f"{act.name} {act.objects}")
        # logging.info("")
        if task.goal.issubset(node.atoms):
            # If this skeleton satisfies the goal, yield it.
            metrics["num_skeletons_optimized"] += 1
            yield node.skeleton, node.atoms_sequence
        else:
            # Generate successors.
            metrics["num_nodes_expanded"] += 1
            # Generate primitive successors.
            for operator in get_applicable_operators(ground_operators, node.atoms):
                child_atoms = apply_operator(operator, set(node.atoms))
                if use_visited_state_set:
                    frozen_atoms = frozenset(child_atoms)
                    if frozen_atoms in visited_atom_sets:
                        continue
                child_skeleton = node.skeleton + [operator]
                child_skeleton_tup = tuple(child_skeleton)
                if child_skeleton_tup in visited_skeletons:  # pragma: no cover
                    continue
                visited_skeletons.add(child_skeleton_tup)
                # Action costs are unitary.
                child_cost = node.cumulative_cost + 1.0
                child_node = _Node(
                    atoms=child_atoms,
                    skeleton=child_skeleton,
                    atoms_sequence=node.atoms_sequence + [child_atoms],
                    parent=node,
                    cumulative_cost=child_cost,
                )
                metrics["num_nodes_created"] += 1
                # priority is g [cost] plus h [heuristic]
                priority = child_node.cumulative_cost + heuristic(child_node.atoms)
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
    if not queue:
        raise AssertionError("Planning ran out of skeletons!")
    assert time.perf_counter() - start_time >= timeout
    raise AssertionError(
        "Planning timed out before finding a solution! "
        f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds."
    )
