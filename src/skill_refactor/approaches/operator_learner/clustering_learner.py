"""STRIPS learner that leverages access to oracle operators used to generate
demonstrations via bilevel planning."""

from typing import Dict, List, Set, Tuple

from relational_structs import LiftedAtom

from skill_refactor.approaches.operator_learner.base import BaseSTRIPSLearner
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import Datastore, LiftedOperator, OpData


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators but re-learns all the
    components via currently-implemented methods in the base class.

    This is different from the oracle learner because here, we assume that our demo data
    is annotated with the ground-truth operators used to produce it. We thus know
    exactly how to associate (i.e, cluster) demos into sets corresponding to each
    operator.
    """

    def _induce_add_effects_by_intersection(
        self, opdata: OpData
    ) -> Tuple[Set[LiftedAtom], int]:
        """Given a PNAD with a nonempty datastore, compute the add effects for the
        PNAD's operator using threshold-based frequency analysis."""
        assert len(opdata.datastore) > 0

        # Count frequency of each lifted atom across all segments
        atom_counts: Dict[LiftedAtom, int] = {}
        total_segments = len(opdata.datastore)

        for segment, var_to_obj in opdata.datastore:
            objects = set(var_to_obj.values())
            obj_to_var = {o: v for v, o in var_to_obj.items()}
            atoms = {
                atom
                for atom in segment.add_effects
                if all(o in objects for o in atom.objects)
            }
            lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}

            for lifted_atom in lifted_atoms:
                atom_counts[lifted_atom] = atom_counts.get(lifted_atom, 0) + 1

        # Select atoms that appear in at least threshold fraction of segments
        threshold = CFG.add_effect_threshold
        min_count = int(threshold * total_segments)
        add_effects = {
            atom for atom, count in atom_counts.items() if count >= min_count
        }
        num_sample_out = 0
        for _, count in atom_counts.items():
            if min_count <= count < total_segments:
                num_sample_out += total_segments - count

        return add_effects, num_sample_out

    def _compute_datastores_given_operator(self, op: LiftedOperator) -> Datastore:
        datastore = []
        for seg_traj in self._segmented_trajs:
            for segment in seg_traj:
                segment_operator = segment.actions[0].op
                if segment_operator is not None and segment_operator.parent == op:
                    for i in range(len(segment.actions)):
                        # all action should have the same operator in a segment
                        assert segment.actions[i].op == segment_operator
                    op_vars = op.parameters
                    assert segment_operator is not None  # for mypy
                    obj_sub = segment_operator.parameters
                    var_to_obj_sub = dict(zip(op_vars, obj_sub))
                    datastore.append((segment, var_to_obj_sub))
        return datastore

    def _learn(self, given_operators: Set[LiftedOperator]) -> Tuple[List[OpData], int]:
        """Re-learn operator components from the given operators and the segmented
        trajectories."""
        opdatas: List[OpData] = []
        given_operator_list = list(given_operators)
        total_num_samples_out = 0
        for operator in given_operator_list:
            datastore = self._compute_datastores_given_operator(operator)
            opdata = OpData(operator, datastore)
            add_effects, num_sample_out = self._induce_add_effects_by_intersection(
                opdata
            )
            total_num_samples_out += num_sample_out
            preconditions = self._induce_preconditions_via_intersection(opdata)
            complete_opdata = OpData(
                opdata.op.copy_with(
                    preconditions=preconditions, add_effects=add_effects
                ),
                datastore,
            )
            self._compute_opdata_delete_effects(complete_opdata)
            opdatas.append(complete_opdata)

        return opdatas, total_num_samples_out
