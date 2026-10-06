"""Data structures.

Borrowed from Tom's task_then_motion_planning. The key Difference is that here we used
Batched Tensor for skills and perceivers.
"""

import abc
import json
import logging
import pickle
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

import torch
from multimethod import multimethod
from relational_structs.pddl import (
    GroundAtom,
)
from relational_structs.pddl import GroundOperator as BaseGroundOperator
from relational_structs.pddl import (
    LiftedAtom,
)
from relational_structs.pddl import LiftedOperator as BaseLiftedOperator
from relational_structs.pddl import (
    Object,
)
from relational_structs.pddl import Predicate as BasePredicate
from relational_structs.pddl import (
    Type,
    TypedEntity,
    Variable,
    consistent_hash,
)
from torch import Tensor

# Planning Structs


@dataclass
class ApproachStepResult(abc.ABC):
    """Result from an approach step."""

    _action: Tensor
    info: dict[str, Any] = field(default_factory=dict)
    op: Optional["GroundOperator"] = None

    @property
    def action(self) -> Tensor:
        """The array representation of this action."""
        return self._action.clone()

    def has_op(self) -> bool:
        """Whether this action has a non-default option attached."""
        return self.op is not None

    def get_op(self) -> "GroundOperator":
        """Get the option that produced this action."""
        assert self.has_op()
        assert self.op is not None  # for mypy
        return self.op

    def set_op(self, op: "GroundOperator") -> None:
        """Set the option that produced this action."""
        self.op = op

    def unset_op(self) -> None:
        """Unset the option that produced this action."""
        self.op = None
        assert not self.has_op()


class Skill(abc.ABC):
    """In the context of this repository, a skill is responsible for executing
    operators, that is, taking actions to achieve the effects when the preconditions
    hold. Control flow is handled externally to the skill. For example, checking whether
    the operator effects have been satisfied, or checking if the skill has exceeded a
    max number of actions, happens outside the skill itself.

    The skill can internally maintain memory and so needs to be reset after each
    execution.
    """

    def __init__(self) -> None:
        self.current_ground_operator: GroundOperator | None = None

    @abc.abstractmethod
    def can_execute(self, ground_operator: "GroundOperator") -> bool:
        """Determine whether the skill knows how to execute this operator.

        A typical implementation would have one skill per LiftedOperator and would check
        here if the ground_operator's parent matches.
        """

    @abc.abstractmethod
    def get_action(self, obs: Tensor) -> Tensor:
        """Assuming that ground_operator can be executed, return an action to execute
        given the current observation.

        The internal memory may be updated assuming that the action will be executed.
        """

    def reset(self, ground_operator: "GroundOperator", obs: Tensor) -> None:
        """Reset any internal memory given a ground operator that can be executed."""
        del obs  # unused
        assert self.can_execute(ground_operator)
        self.current_ground_operator = ground_operator


class LiftedOperatorSkill(Skill):
    """A skill that is one-to-one with a specific LiftedOperator."""

    _current_plan: list[Tensor] | None = []

    @property
    def current_plan(self) -> list[Tensor] | None:
        """Return a copy of the current plan for external access."""
        if self._current_plan is None:
            return None
        return list(self._current_plan)

    @abc.abstractmethod
    def get_lifted_operator(self) -> "LiftedOperator":
        """Return the lifted operator for this skill."""

    @abc.abstractmethod
    def get_operator_name(self) -> str:
        """Return the lifted operator name for this skill."""

    @abc.abstractmethod
    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Defines an object-parameterized policy."""

    def terminate_with_objects(self, objects: Sequence[Object], obs: Tensor) -> Tensor:
        """Defines termination condition for the skill given objects.

        Note that obs is batched, and the output should be batched as well.
        """
        raise NotImplementedError

    def can_execute(self, ground_operator: "GroundOperator") -> bool:
        # NOTE: Use name comparison, since during real execution we might use
        # a different, filtered lifted operator.
        return ground_operator.parent.name == self.get_lifted_operator().name

    def get_action(self, obs: Tensor) -> Tensor:
        assert self.current_ground_operator is not None
        objects = self.current_ground_operator.parameters
        return self.get_action_given_objects(objects, obs)

    def terminate(self, obs: Tensor) -> Tensor:
        """Check if the skill should terminate given the current observation.

        Args:
            obs: Current observation tensor

        Returns:
            Boolean tensor indicating whether to terminate
        """
        assert self.current_ground_operator is not None
        objects = self.current_ground_operator.parameters
        return self.terminate_with_objects(objects, obs)


class Predicate(BasePredicate):  # pylint: disable=abstract-method
    """Struct defining a predicate (a lifted classifier over states).

    This extends the base relational_structs.pddl.Predicate with support for quantified
    predicates via the base_types field.
    """

    __slots__ = ("base_types",)

    # Type annotation for mypy (actual value set in __init__)
    base_types: tuple[Type, ...]

    def __init__(  # pylint: disable=super-init-not-called
        self, name: str, types: Sequence[Type], base_types: Sequence[Type] | None = None
    ):
        """Initialize Predicate with name, types, and optional base_types."""
        # Convert to tuples for hashing (required for frozen dataclass parent)
        types_tuple = tuple(types) if not isinstance(types, tuple) else types
        base_types_tuple = (
            tuple(base_types)
            if base_types is not None and not isinstance(base_types, tuple)
            else base_types
        )

        # Initialize parent using object.__setattr__ since parent is frozen
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "types", types_tuple)
        object.__setattr__(
            self,
            "base_types",
            base_types_tuple if base_types_tuple is not None else types_tuple,
        )

    @property
    def quantified(self) -> bool:
        """Whether this predicate is a quantified predicate (i.e., has a parent)."""
        return self.base_types != self.types


class PredicateContainer(Protocol):
    """Protocol for predicate containers."""

    predicates: Dict[str, Predicate]

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return self.predicates[key]

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return set(self.predicates.values())

    def add_predicate(
        self,
        name: str,
        arg_types: List[Type],
        base_types: Optional[Sequence[Type]] = None,
    ) -> Predicate:
        """Create and register a new predicate."""
        assert (
            name not in self.predicates
        ), f"Predicate {name} already exists in the container."
        pred = Predicate(name, arg_types, base_types)
        self.predicates[name] = pred
        return pred


class TypeContainer(Protocol):
    """Protocol for type containers."""

    def __getitem__(self, key: str) -> Type:
        """Get type by name."""
        return self.as_dict()[key]

    def as_set(self) -> set[Type]:
        """Convert to set of types."""

    def as_dict(self) -> dict[str, Type]:
        """Convert to dictionary of types."""


class ObjectContainer(Protocol):
    """Protocol for object containers."""

    def __getitem__(self, key: str) -> Object:
        """Get object by name."""
        return self.as_dict()[key]

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""


class Perceiver(abc.ABC):
    """Turns observations into objects, ground atoms, and goals.

    A perceiver may use internal memory to produce predicates, so it needs to be reset
    after every "episode".

    Assumes that object sets and goals do not change within an episode.
    """

    predicates_container: PredicateContainer
    predicate_interpreters: Dict[
        Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
    ]

    @abc.abstractmethod
    def reset(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Called at the beginning of each new episode.

        Resets internal memory and returns known objects, ground atoms in the initial
        state, and goal.
        """

    @abc.abstractmethod
    def step(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Step perceiver with observation."""

    @property
    def objects(self) -> ObjectContainer:
        """Get the object container."""
        raise NotImplementedError

    def add_predicate_interpreter(
        self,
        name: str,
        types: Sequence[Type],
        interpreter: Callable[[Tensor, List[Sequence[Object]]], Tensor],
        base_types: Optional[Sequence[Type]] = None,
    ) -> None:
        """Add a new predicate interpreter."""
        if base_types is None:
            base_types = types
        new_predicate = self.predicates_container.add_predicate(
            name, list(types), base_types
        )
        self.predicate_interpreters[new_predicate] = interpreter

    def delete_predicate_interpreter(self, predicate: Predicate) -> None:
        """Delete a predicate interpreter."""
        if predicate in self.predicate_interpreters:
            del self.predicate_interpreters[predicate]
        if predicate.name in self.predicates_container.predicates:
            del self.predicates_container.predicates[predicate.name]


@dataclass(frozen=True, repr=False, eq=False)
class LiftedOperator(BaseLiftedOperator):
    """Struct defining a lifted symbolic operator (as in STRIPS)."""

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> "GroundOperator":  # type: ignore[override]
        """Ground into a _GroundSTRIPSOperator, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        return GroundOperator(
            self.name,
            list(objects),
            preconditions,
            add_effects,
            delete_effects,
            self.cost,
            self,
        )

    def copy_with(self, **kwargs: Any) -> "LiftedOperator":
        """Create a copy of the operator, optionally while replacing any of the
        arguments."""
        default_kwargs = dict(
            name=self.name,
            parameters=self.parameters,
            preconditions=self.preconditions,
            add_effects=self.add_effects,
            delete_effects=self.delete_effects,
        )
        assert set(kwargs.keys()).issubset(default_kwargs.keys())
        default_kwargs.update(kwargs)
        # mypy is known to have issues with this pattern:
        # https://github.com/python/mypy/issues/5382
        return LiftedOperator(**default_kwargs)  # type: ignore

    def get_complexity(self) -> float:
        """Get the complexity of this operator.

        We only care about the arity of the operator, since that is what affects
        grounding. We'll use 2^arity as a measure of grounding effort.
        """
        return float(2 ** len(self.parameters))


@dataclass(frozen=True, repr=False, eq=False)
class GroundOperator(BaseGroundOperator):
    """A STRIPSOperator + objects."""

    parent: LiftedOperator = field(default=...)  # type: ignore


# Data Structs
@dataclass(frozen=False, repr=False, eq=False)
class LowLevelTrajectory:
    """A structure representing a low-level trajectory, containing a state sequence,
    action sequence, and optional train task id.

    This trajectory may or may not be a demonstration.
    """

    _states: List[Tensor]
    _actions: List[ApproachStepResult]
    _train_scenario_idx: str
    _success: bool
    _train_task_idx: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        assert len(self._states) == len(self._actions) + 1
        assert self._train_task_idx is not None

    @property
    def states(self) -> List[Tensor]:
        """States in the trajectory."""
        return self._states

    @property
    def actions(self) -> List[ApproachStepResult]:
        """Actions in the trajectory."""
        return self._actions

    @property
    def train_task_idx(self) -> int:
        """The index of the train task."""
        assert (
            self._train_task_idx is not None
        ), "This trajectory doesn't contain a train task idx!"
        return self._train_task_idx

    @property
    def train_scenario_idx(self) -> str:
        """The index of the train scenario, if available."""
        return self._train_scenario_idx

    @property
    def success(self) -> bool:
        """Whether the trajectory was successful."""
        return self._success

    def has_complete_operator_annotations(self) -> bool:
        """Check if all actions in the trajectory have operator annotations.

        Returns:
            True if all actions have operators, False otherwise
        """
        return all(action.has_op() for action in self._actions)

    def get_lifted_sequence(self) -> List[LiftedOperator]:
        """Get the sequence of lifted operators for the trajectory."""
        assert self.has_complete_operator_annotations()
        curr_lifted_op = self._actions[0].get_op().parent
        lifted_plan = [curr_lifted_op]
        for action in self._actions:
            action_op = action.get_op()
            if action_op.parent != curr_lifted_op:
                curr_lifted_op = action_op.parent
                lifted_plan.append(curr_lifted_op)
        return lifted_plan


@dataclass(eq=False)
class Segment:
    """A segment represents a low-level trajectory that is the result of executing one
    option-operator. The segment stores the abstract state (ground atoms) that held
    immediately before the option started executing, and the abstract state (ground
    atoms) that held immediately after.

    Segments are used during learning.
    """

    trajectory: LowLevelTrajectory
    init_atoms: Set[GroundAtom]
    final_atoms: Set[GroundAtom]
    op: GroundOperator

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1

    @property
    def states(self) -> List[Tensor]:
        """States in the trajectory."""
        return self.trajectory.states

    @property
    def actions(self) -> List[ApproachStepResult]:
        """Actions in the trajectory."""
        return self.trajectory.actions

    @property
    def add_effects(self) -> Set[GroundAtom]:
        """Atoms in the final atoms but not the init atoms.

        Do not cache; init and final atoms can change.
        """
        return self.final_atoms - self.init_atoms

    @property
    def delete_effects(self) -> Set[GroundAtom]:
        """Atoms in the init atoms but not the final atoms.

        Do not cache; init and final atoms can change.
        """
        return self.init_atoms - self.final_atoms

    def get_op(self) -> GroundOperator:
        """Get the option that produced this segment."""
        return self.op

    def set_op(self, op: GroundOperator) -> None:
        """Set the option that produced this segment."""
        self.op = op


@dataclass(repr=False, eq=False)
class PlannerDataset:
    """A collection of LowLevelTrajectory objects from RL rollouts, and optionally,
    lists of annotations, one per trajectory.

    For planner refactorization.
    """

    _trajectories: List[LowLevelTrajectory]
    _training_scenario_idx: List[str]
    _annotations: Optional[List[Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self._annotations is not None:
            assert len(self._trajectories) == len(self._annotations)
        assert len(self._trajectories) == len(self._training_scenario_idx)
        for _, traj in enumerate(self._trajectories):
            assert traj.train_scenario_idx in self._training_scenario_idx

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self._trajectories)

    @property
    def trajectories(self) -> List[LowLevelTrajectory]:
        """The trajectories in the dataset."""
        return self._trajectories

    @property
    def has_annotations(self) -> bool:
        """Whether this dataset has annotations in it."""
        return self._annotations is not None

    @property
    def training_scenario_indices(self) -> List[str]:
        """The training scenario indices in the dataset."""
        return self._training_scenario_idx

    def get_all_sceanrio(self) -> List[str]:
        """Get all unique training scenario indices in the dataset."""
        list_int = list(set(self._training_scenario_idx))
        return list_int

    def sample_sub_dataset(self, traj_per_scenario: int) -> "PlannerDataset":
        """Sample a subset of trajectories with traj_per_scenario trajectories per
        unique scenario.

        Args:
            traj_per_scenario: Number of trajectories to sample per scenario

        Returns:
            A new PlannerDataset containing sampled trajectories
        """
        import random

        # Group trajectories by scenario
        scenario_to_trajectories: Dict[str, List[int]] = {}
        for idx, scenario_idx in enumerate(self._training_scenario_idx):
            if scenario_idx not in scenario_to_trajectories:
                scenario_to_trajectories[scenario_idx] = []
            scenario_to_trajectories[scenario_idx].append(idx)

        # Sample trajectories from each scenario
        sampled_trajectories: List[LowLevelTrajectory] = []
        sampled_annotations: Optional[List[Any]] = (
            [] if self._annotations is not None else None
        )
        sampled_scenario_indices: List[str] = []

        for scenario_idx, traj_indices in scenario_to_trajectories.items():
            # Sample min(traj_per_scenario, available_trajectories)
            num_to_sample = min(traj_per_scenario, len(traj_indices))
            sampled_indices = random.sample(traj_indices, num_to_sample)

            for idx in sampled_indices:
                sampled_trajectories.append(self._trajectories[idx])
                sampled_scenario_indices.append(scenario_idx)
                if sampled_annotations is not None:
                    assert self._annotations is not None
                    sampled_annotations.append(self._annotations[idx])

        return PlannerDataset(
            _trajectories=sampled_trajectories,
            _annotations=sampled_annotations,
            _training_scenario_idx=sampled_scenario_indices,
        )

    def success_subset(self) -> "PlannerDataset":
        """Return a subset containing only successful trajectories.

        Returns:
            A new PlannerDataset containing only trajectories where success=True
        """
        successful_trajectories: List[LowLevelTrajectory] = []
        successful_annotations: Optional[List[Any]] = (
            [] if self._annotations is not None else None
        )
        successful_scenario_indices: List[str] = []

        for idx, traj in enumerate(self._trajectories):
            if traj.success:
                successful_trajectories.append(traj)
                successful_scenario_indices.append(self._training_scenario_idx[idx])
                if successful_annotations is not None:
                    assert self._annotations is not None
                    successful_annotations.append(self._annotations[idx])

        return PlannerDataset(
            _trajectories=successful_trajectories,
            _annotations=successful_annotations,
            _training_scenario_idx=successful_scenario_indices,
        )

    def filter_complete_operators(self) -> "PlannerDataset":
        """Return a subset containing only trajectories with complete operator
        annotations.

        Returns:
            A new PlannerDataset containing only trajectories where all actions
            have operator annotations
        """
        filtered_trajectories: List[LowLevelTrajectory] = []
        filtered_annotations: Optional[List[Any]] = (
            [] if self._annotations is not None else None
        )
        filtered_scenario_indices: List[str] = []

        original_count = len(self._trajectories)
        for idx, traj in enumerate(self._trajectories):
            if traj.has_complete_operator_annotations():
                filtered_trajectories.append(traj)
                filtered_scenario_indices.append(self._training_scenario_idx[idx])
                if filtered_annotations is not None:
                    assert self._annotations is not None
                    filtered_annotations.append(self._annotations[idx])

        filtered_count = original_count - len(filtered_trajectories)
        if filtered_count > 0:
            logging.info(
                "Filtered out %d/%d trajectories with incomplete operator annotations",
                filtered_count,
                original_count,
            )

        return PlannerDataset(
            _trajectories=filtered_trajectories,
            _annotations=filtered_annotations,
            _training_scenario_idx=filtered_scenario_indices,
        )

    @property
    def annotations(self) -> List[Any]:
        """The annotations in the dataset."""
        assert self._annotations is not None
        return self._annotations

    def append(
        self,
        trajectory: LowLevelTrajectory,
        annotation: Optional[Any] = None,
    ) -> None:
        """Append one more trajectory and annotation to the dataset."""
        if annotation is None:
            assert self._annotations is None
        else:
            assert self._annotations is not None
            self._annotations.append(annotation)
        self._training_scenario_idx.append(trajectory.train_scenario_idx)
        self._trajectories.append(trajectory)

    def save(self, path: Path) -> None:
        """Save planner data to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # 1. trajectories - move tensors to CPU before saving to reduce memory usage
        cpu_trajectories = self._move_trajectories_to_device(self._trajectories, "cpu")
        traj_path = path / "trajectories.pkl"
        with open(traj_path, "wb") as f:
            pickle.dump(cpu_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. annotations (optional)
        ann_path = path / "annotations.pkl"
        if self._annotations is not None:
            with open(ann_path, "wb") as f:
                pickle.dump(self._annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # could choose not to create the file; if you prefer an empty file:
            if ann_path.exists():
                ann_path.unlink(missing_ok=True)

        # 3. training scenario indices (optional)
        train_scenario_idx_path = path / "training_scenario_indices.pkl"
        with open(train_scenario_idx_path, "wb") as f:
            pickle.dump(
                self._training_scenario_idx,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Optional: tiny meta file to record presence/absence
        meta = {
            "has_annotations": self._annotations is not None,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def get_appearing_operators(self) -> Set[LiftedOperator]:
        """Get the set of appearing operators in the dataset."""
        appearing_ops = set()
        for traj in self._trajectories:
            for action in traj.actions:
                if action.has_op():
                    appearing_ops.add(action.get_op().parent)
        return appearing_ops

    def get_all_operator_plans(self) -> List[List[LiftedOperator]]:
        """Get all unique operator plans in the dataset."""
        all_plans = set()
        for traj in self._trajectories:
            lifted_plan = traj.get_lifted_sequence()
            all_plans.add(tuple(lifted_plan))
        # Convert back to list of lists
        return [list(plan) for plan in all_plans]

    def get_ground_atoms_and_tasks(
        self, perceiver: Perceiver
    ) -> Tuple[List["GroundAtomTrajectory"], List["Task"]]:
        """Get the ground atoms and tasks from the dataset using the perceiver."""
        ground_atom_dataset: List[GroundAtomTrajectory] = []
        tasks: List[Task] = []
        for traj in self._trajectories:
            atom_list = []
            # Reset the perceiver to get initial state and goal
            objects, atoms, goal = perceiver.reset(traj.states[0].unsqueeze(0), {})
            task = Task(
                init=traj.states[0],
                objects=objects,
                goal=goal,
                train_task_idx=traj.train_task_idx,
            )
            tasks.append(task)
            atom_list.append(atoms)
            state_stack = torch.stack(traj.states[1:], dim=0)
            atom_stack = perceiver.step(state_stack)
            atom_list.extend(atom_stack)
            ground_atom_dataset.append((traj, atom_list))

        return ground_atom_dataset, tasks

    def get_predicate_identifier(
        self,
        perceiver: Perceiver,
        predicate: Predicate,
    ) -> Tuple[
        FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]],
        List["GroundAtomTrajectory"],
    ]:
        """Get a unique identifier for a predicate based on its interpretations across
        the dataset.

        This is useful for checking if two predicates are equivalent in terms of their
        interpretations over the dataset.
        """
        raw_identifiers = set()
        ground_atom_dataset: List[GroundAtomTrajectory] = []
        interpreter = perceiver.predicate_interpreters.get(predicate, None)
        if interpreter is None:
            raise ValueError(f"No interpreter found for predicate {predicate}")

        for traj_idx, traj in enumerate(self._trajectories):
            atom_list = []
            _, atoms, _ = perceiver.reset(
                traj.states[0].unsqueeze(0), {"desired_predicates": [predicate.name]}
            )
            atom_list.append(atoms)
            atom_args = frozenset(tuple(a.objects) for a in atoms)
            raw_identifiers.add((traj_idx, 0, atom_args))
            state_stack = torch.stack(traj.states[1:], dim=0)
            atom_stack = perceiver.step(
                state_stack, {"desired_predicates": [predicate.name]}
            )
            atom_list.extend(atom_stack)
            ground_atom_dataset.append((traj, atom_list))
            for t_idx, atom_set in enumerate(atom_stack):
                atom_args = frozenset(tuple(a.objects) for a in atom_set)
                raw_identifiers.add((traj_idx, t_idx + 1, atom_args))

        return frozenset(raw_identifiers), ground_atom_dataset

    @classmethod
    def load(
        cls,
        path: Path,
        num_traj: Optional[int] = -1,
        filter_incomplete_operators: bool = True,
    ) -> "PlannerDataset":
        """Load planner data from disk.

        Args:
            path: Path to the directory containing saved dataset files
            num_traj: Number of trajectories to load (-1 for all)
            filter_incomplete_operators: If True, filter out trajectories where
                any action lacks an operator annotation

        Returns:
            PlannerDataset with loaded trajectories
        """
        traj_path = path / "trajectories.pkl"
        if not traj_path.exists():
            raise FileNotFoundError(f"{traj_path} not found")

        with open(traj_path, "rb") as f:
            trajectories: List[LowLevelTrajectory] = pickle.load(f)

        # Filter out trajectories with incomplete operator annotations if requested
        # Track which indices are kept for filtering annotations/scenario indices
        kept_indices: Optional[List[int]] = None
        if filter_incomplete_operators:
            original_count = len(trajectories)
            kept_indices = [
                i
                for i, traj in enumerate(trajectories)
                if traj.has_complete_operator_annotations()
            ]
            trajectories = [trajectories[i] for i in kept_indices]
            filtered_count = original_count - len(trajectories)
            if filtered_count > 0:
                logging.info(
                    "Filtered out %d/%d trajectories with incomplete operator annotations",
                    filtered_count,
                    original_count,
                )

        # Move tensors to appropriate device and ensure memory efficiency
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if num_traj is None or num_traj < 0:
            trajectories = cls._move_trajectories_to_device(trajectories, device)
        else:
            trajectories = cls._move_trajectories_to_device(
                trajectories[:num_traj], device
            )

        # meta tells us if annotations exist (optional)
        has_annotations_flag = False

        ann: Optional[List[Any]] = None
        ann_path = path / "annotations.pkl"
        if has_annotations_flag and ann_path.exists():
            with open(ann_path, "rb") as f:
                ann = pickle.load(f)

            # Filter annotations if we filtered trajectories
            if kept_indices is not None:
                ann = [ann[i] for i in kept_indices]

            partial_ann = (
                ann[:num_traj] if num_traj is not None and num_traj > 0 else ann
            )
            if len(partial_ann) != len(trajectories):
                logging.warning(
                    "Annotation/trajectory length mismatch (%d vs %d). "
                    "Setting annotations to None.",
                    len(partial_ann),
                    len(trajectories),
                )
                ann = None

        training_scenario_indices: Optional[List[str]] = None
        train_scenario_idx_path = path / "training_scenario_indices.pkl"
        with open(train_scenario_idx_path, "rb") as f:
            training_scenario_indices = pickle.load(f)

        # Filter scenario indices if we filtered trajectories
        if kept_indices is not None:
            training_scenario_indices = [
                training_scenario_indices[i] for i in kept_indices
            ]

        partial_indices = (
            training_scenario_indices[:num_traj]
            if num_traj is not None and num_traj > 0
            else training_scenario_indices
        )
        assert len(partial_indices) == len(trajectories)
        training_scenario_indices = partial_indices

        return cls(
            _trajectories=trajectories,
            _annotations=ann,
            _training_scenario_idx=training_scenario_indices,
        )

    def merge(self, other: "PlannerDataset") -> "PlannerDataset":
        """Merge this dataset with another PlannerDataset to create a new combined
        dataset.

        Args:
            other: Another PlannerDataset to merge with this one

        Returns:
            A new PlannerDataset containing trajectories and annotations from both datasets

        Raises:
            AssertionError: If annotation consistency is violated (one has annotations, other doesn't)
        """
        # Check annotation consistency
        if self.has_annotations != other.has_annotations:
            raise ValueError(
                "Cannot merge datasets with inconsistent annotation status. "
                f"Self has_annotations={self.has_annotations}, "
                f"other has_annotations={other.has_annotations}"
            )

        # Merge trajectories
        merged_trajectories = self.trajectories.copy()
        last_idx = len(merged_trajectories)
        for traj in other.trajectories:
            # Adjust train_task_idx to avoid conflicts
            if traj._train_task_idx is not None:  # pylint: disable=protected-access
                new_train_task_idx = (
                    traj._train_task_idx + last_idx  # pylint: disable=protected-access
                )
            else:
                new_train_task_idx = None
            new_traj = LowLevelTrajectory(
                _states=traj._states,  # pylint: disable=protected-access
                _actions=traj._actions,  # pylint: disable=protected-access
                _train_scenario_idx=traj._train_scenario_idx,  # pylint: disable=protected-access
                _success=traj._success,  # pylint: disable=protected-access
                _train_task_idx=new_train_task_idx,
            )
            merged_trajectories.append(new_traj)

        # Merge annotations if they exist
        merged_annotations = None
        if self.has_annotations and other.has_annotations:
            assert self.annotations is not None
            assert other.annotations is not None
            merged_annotations = self.annotations + other.annotations

        # Merge training sceanrio if they exist
        merged_training_scenarios = (
            self.training_scenario_indices + other.training_scenario_indices
        )

        return PlannerDataset(
            _trajectories=merged_trajectories,
            _annotations=merged_annotations,
            _training_scenario_idx=merged_training_scenarios,
        )

    @staticmethod
    def _move_trajectories_to_device(
        trajectories: List[LowLevelTrajectory], device: str
    ) -> List[LowLevelTrajectory]:
        """Move all tensors in trajectories to the specified device.

        This helps manage memory usage and ensures consistent device placement.
        """
        moved_trajectories = []
        for traj in trajectories:
            # Move states to device
            moved_states = [state.to(device) for state in traj.states]

            # Move action tensors to device
            moved_actions = []
            for action in traj.actions:
                moved_action = ApproachStepResult(
                    _action=action.action.to(device), info=action.info, op=action.op
                )
                moved_actions.append(moved_action)

            # Create new trajectory with moved tensors
            moved_traj = LowLevelTrajectory(
                _states=moved_states,
                _actions=moved_actions,
                _train_scenario_idx=traj.train_scenario_idx,
                _success=traj.success,
                _train_task_idx=traj.train_task_idx,
            )
            moved_trajectories.append(moved_traj)

        return moved_trajectories


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is an initial state, objects, and goal."""

    init: Tensor
    objects: Set[Object]
    goal: Set[GroundAtom]
    train_task_idx: int

    def __post_init__(self) -> None:
        # Verify types.
        for atom in self.goal:
            assert isinstance(atom, GroundAtom)


@dataclass
class RLDataset:
    """A collection of failed states, current atoms, and goal atoms, from planner
    executions.

    For RL policy learning.
    """

    states: list[Tensor]  # List of states where intervention needed
    current_atoms: list[set[GroundAtom]]
    goal_atoms: list[frozenset[GroundAtom]]
    config: dict[str, Any]

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: Path) -> None:
        """Save training data to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save states
        # convert to a big tensor if they are not already
        states = torch.stack(self.states, dim=0).cpu()
        states_path = path / "states.pt"
        torch.save(states, states_path)

        # Save current atoms and goal atoms as pickle
        data_paths = {
            "current_atoms": self.current_atoms,
            "goal_atoms": self.goal_atoms,
        }
        for name, obj in data_paths.items():
            file_path = path / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        # Save config as JSON
        serializable_config = dict(self.config)

        if "atom_to_index" in serializable_config:
            # Convert keys to strings if they aren't already
            atom_to_index = {
                str(k): v for k, v in serializable_config["atom_to_index"].items()
            }
            serializable_config["atom_to_index"] = atom_to_index

        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2)

    @classmethod
    def load(cls, path: Path, num_states: Optional[int] = -1) -> "RLDataset":
        """Load training data from disk."""
        # Load states
        states_path = path / "states.pt"
        states = torch.load(
            states_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        state_list = list(states)[:num_states]

        # Load current atoms and goal atoms
        data = {}
        data_names = [
            "current_atoms",
            "goal_atoms",
        ]
        for name in data_names:
            file_path = path / f"{name}.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    data[name] = pickle.load(f)[:num_states]
            else:
                logging.info(f"Warning: {file_path} not found, using empty list.")
                data[name] = []

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "atom_to_index" in config:
            config["atom_to_index"] = {
                k: int(v) if isinstance(v, str) else v
                for k, v in config["atom_to_index"].items()
            }

        return cls(
            states=state_list,
            current_atoms=data["current_atoms"],
            goal_atoms=data["goal_atoms"],
            config=config,
        )


GroundAtomTrajectory = Tuple[LowLevelTrajectory, List[Set[GroundAtom]]]
ObjToVarSub = Dict[Object, Variable]
ObjToObjSub = Dict[Object, Object]
VarToObjSub = Dict[Variable, Object]
Datastore = List[Tuple[Segment, VarToObjSub]]


@dataclass(eq=False, repr=False)
class OpData:
    """Data store for operator learning.

    A helper class for Operator learning that contains information useful to maintain
    throughout the learning procedure. Each object of this class corresponds to a
    learned Operator. We use this class because we don't want to clutter the Operator
    class with a datastore, since data is only used for learning and is not part of the
    representation itself.
    """

    op: LiftedOperator
    # The datastore, a list of segments that are covered by the
    # STRIPSOperator self.op. For each such segment, the datastore also
    # maintains a substitution dictionary of type VarToObjSub,
    # under which the operator and effects for all
    # segments in the datastore are equivalent.
    datastore: Datastore

    def add_to_datastore(
        self, member: Tuple[Segment, VarToObjSub], check_effect_equality: bool = True
    ) -> None:
        """Add a new member to self.datastore."""
        seg, var_obj_sub = member
        if len(self.datastore) > 0:
            # All variables should have a corresponding object.
            assert set(var_obj_sub) == set(self.op.parameters)
            # The effects should match.
            if check_effect_equality:
                obj_var_sub = {o: v for (v, o) in var_obj_sub.items()}
                lifted_add_effects = {a.lift(obj_var_sub) for a in seg.add_effects}
                lifted_del_effects = {a.lift(obj_var_sub) for a in seg.delete_effects}
                assert lifted_add_effects == self.op.add_effects
                assert lifted_del_effects == self.op.delete_effects
            # The operator's name should match.
            operator = seg.get_op()
            assert operator.parent.name == self.op.name
        # Add to datastore.
        self.datastore.append(member)

    def copy(self) -> "OpData":
        """Make a copy of this PNAD object, taking care to ensure that modifying the
        original will not affect the copy."""
        new_op = self.op
        new_opdata = OpData(new_op, self.datastore)
        return new_opdata

    def __repr__(self) -> str:
        return f"{self.op}"

    def __str__(self) -> str:
        return repr(self)

    def __lt__(self, other: "OpData") -> bool:
        return repr(self) < repr(other)
