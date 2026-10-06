"""Base class for a STRIPS operator learning algorithm."""

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from relational_structs import LiftedAtom
from torch import Tensor

from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    LiftedOperator,
    LowLevelTrajectory,
    OpData,
    Predicate,
    Segment,
    Task,
)


@dataclass
class OpEffBelief:
    """A belief about the effects of an operator."""

    row_names: List[LiftedOperator]
    col_names: List[Predicate]
    col_var_idx: List[Tensor]
    ae_matrix: Tensor


class BaseSTRIPSLearner(abc.ABC):
    """Base class definition."""

    def __init__(
        self,
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task],
        predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        verify_harmlessness: bool,
        verbose: bool = True,
        operator_belief: Optional[OpEffBelief] = None,
    ) -> None:
        self._trajectories = trajectories
        self._train_tasks = train_tasks
        self._predicates = predicates
        self._segmented_trajs = segmented_trajs
        self._verify_harmlessness = verify_harmlessness
        self._verbose = verbose
        self._num_segments = sum(len(t) for t in segmented_trajs)
        self._belief = operator_belief
        assert len(self._trajectories) == len(self._segmented_trajs)

    def learn(self, given_operators: Set[LiftedOperator]) -> Tuple[List[OpData], int]:
        """The public method for a STRIPS operator learning strategy.

        A wrapper around self._learn() to sanity check that harmlessness holds on the
        training data, and then filter out operators without enough data. We check
        harmlessness first because filtering may break it.
        """

        learned_opdata, num_samples_out = self._learn(given_operators)
        # For now, assume we don't do harmless checks/pruning.
        return learned_opdata, num_samples_out

    @abc.abstractmethod
    def _learn(self, given_operators: Set[LiftedOperator]) -> Tuple[List[OpData], int]:
        """The key method that a STRIPS operator learning strategy must implement.

        Returns a new list of opdata learned from the data, with op (STRIPSOperator),
        datastore, and option_spec fields filled in (but not sampler).
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _induce_preconditions_via_intersection(opdata: OpData) -> Set[LiftedAtom]:
        """Given a opdata with a nonempty datastore, compute the preconditions for the
        opdata's operator using threshold-based frequency analysis."""
        assert len(opdata.datastore) > 0

        # Count frequency of each lifted atom across all segments
        atom_counts: Dict[LiftedAtom, int] = {}
        total_segments = len(opdata.datastore)
        n_bp4 = 0
        for segment, var_to_obj in opdata.datastore:
            for atom in segment.init_atoms:
                if atom.predicate.name == "b_p4_0":
                    n_bp4 += 1
            objects = set(var_to_obj.values())
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.init_atoms
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}

            for lifted_atom in lifted_atoms:
                atom_counts[lifted_atom] = atom_counts.get(lifted_atom, 0) + 1

        # Select atoms that appear in at least threshold fraction of segments
        threshold = CFG.precondition_threshold
        min_count = int(threshold * total_segments)
        preconditions = {
            atom for atom, count in atom_counts.items() if count >= min_count
        }

        return preconditions

    @staticmethod
    def _compute_opdata_delete_effects(opdata: OpData) -> None:
        """Update the given opdata to change the delete effects to ones obtained by
        unioning all lifted images in the datastore.

        IMPORTANT NOTE: We want to do a union here because the most
        general delete effects are the ones that capture _any possible_
        deletion that occurred in a training transition. (This is
        contrast to preconditions, where we want to take an intersection
        over our training transitions.) However, we do not allow
        creating new variables when we create these delete effects.
        Instead, we filter out delete effects that include new
        variables. Therefore, even though it may seem on the surface
        like this procedure will cause all delete effects in the data to
        be modeled accurately, this is not actually true.
        """
        delete_effects = set()
        # Count frequency of each lifted atom across all segments
        atom_counts: Dict[LiftedAtom, int] = {}
        total_segments = len(opdata.datastore)
        for segment, var_to_obj in opdata.datastore:
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.delete_effects
                if all(o in obj_to_var for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
            for lifted_atom in lifted_atoms:
                atom_counts[lifted_atom] = atom_counts.get(lifted_atom, 0) + 1
            delete_effects |= lifted_atoms
        # Select atoms that appear in at least threshold fraction of segments
        threshold = CFG.delete_effect_threshold
        min_count = int(threshold * total_segments)
        delete_effects = {
            atom for atom, count in atom_counts.items() if count >= min_count
        }
        opdata.op = opdata.op.copy_with(delete_effects=delete_effects)
