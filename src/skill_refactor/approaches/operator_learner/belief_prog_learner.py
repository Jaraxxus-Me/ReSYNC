"""STRIPS learner that leverages access to oracle operators used to generate
demonstrations via bilevel planning."""

import logging
from itertools import permutations, product
from typing import Dict, List, Set, Tuple

from relational_structs import LiftedAtom
from relational_structs.pddl import Predicate as BasePredicate

from skill_refactor.approaches.operator_learner.belief_learner import (
    BeliefSTRIPSLearner,
)
from skill_refactor.utils.structs import (
    Datastore,
    LiftedOperator,
    Object,
    OpData,
    Predicate,
    Segment,
    Variable,
)


class BeliefProgSTRIPSLearner(BeliefSTRIPSLearner):
    """Progressive STRIPS learner that uses belief matrices and respects previously
    learned predicates.

    This learner extends BeliefSTRIPSLearner with progressive learning capabilities:
    - For operators matching names in `given_operators`, only predicates present in the
      given operator are retained in the learned preconditions and effects
    - For new operators not in `given_operators`, all learned predicates are used
    - This ensures that previously learned operators maintain consistent predicate usage
      while allowing the system to learn new operator structures

    This is useful for lifelong learning scenarios where we want to refine operator
    positions/arguments while keeping the predicate vocabulary consistent with prior knowledge.
    """

    def _learn(self, given_operators: Set[LiftedOperator]) -> Tuple[List[OpData], int]:
        """Re-learn operator components while respecting predicates from given
        operators.

        Args:
            given_operators: Previously learned operators whose predicate vocabulary
                should be preserved. Operators with matching names will only use
                predicates present in the given operator.

        Returns:
            Tuple of (learned operator data, number of segments that didn't fit any operator)
        """
        num_sample_in = 0
        num_sample_out = 0
        # Create a mapping from operator name to given operator for quick lookup
        given_op_map: Dict[str, LiftedOperator] = {}
        given_predicates: Set[BasePredicate] = set()
        for op in given_operators:
            given_op_map[op.name] = op
            for atom in op.preconditions | op.add_effects | op.delete_effects:
                given_predicates.add(atom.predicate)

        segments: List[Segment] = []
        for segs in self._segmented_trajs:
            for seg in segs:
                if seg.op.parent.name in given_op_map:
                    # For given operators, we only consider new predicates
                    # if they are the pre-conditions
                    new_init_atoms = {
                        atom
                        for atom in seg.init_atoms
                        if atom.predicate not in given_predicates
                    }
                    new_final_atoms = {
                        atom
                        for atom in seg.final_atoms
                        if atom.predicate not in given_predicates
                    }
                    segments.append(
                        Segment(
                            trajectory=seg.trajectory,
                            init_atoms=new_init_atoms,
                            final_atoms=new_final_atoms,
                            op=seg.op,
                        )
                    )
                else:
                    # For new operators, keep all predicates
                    # since old predicates may also be useful
                    segments.append(seg)

        # Cluster the segments according to common option and effects.
        opdatas, added_segment_idxs = self._belief2opdatas_init(
            segments, given_op_map, _given_predicates=given_predicates
        )
        filtered_opdatas: List[OpData] = []

        for opdata in opdatas:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            if len(opdata.datastore) > 0:
                filtered_opdatas.append(opdata)
                continue
            logging.info(
                f"Operator data {opdata.op.name} has no samples. It will have empty effects."
            )

        for ids, segment in enumerate(segments):
            if ids in added_segment_idxs:
                # this segment has been added to a PNAD
                continue
            segment_operator = segment.actions[0].get_op()
            segment_objects = segment_operator.parameters
            segment_effect_objects: List[Object] = sorted(
                {
                    o
                    for atom in segment.add_effects | segment.delete_effects
                    for o in atom.objects
                }
                | set(segment_objects)
            )
            suc = False
            for opdata in filtered_opdatas:
                if opdata.op.name != segment_operator.parent.name:
                    # Since we are already segmenting by operator,
                    # the segment should match the operator name.
                    # If it does not, we skip this operator.
                    continue
                if segment_operator.parent.name in given_op_map:
                    # For given operators, we add it as long as it has empty effects
                    if len(segment.add_effects) > 0 or len(segment.delete_effects) > 0:
                        # this segment has effects, skip it
                        num_sample_out += 1
                        continue
                    var_to_obj = dict(
                        zip(
                            segment_operator.parent.parameters,
                            segment_operator.parameters,
                        )
                    )
                    opdata.add_to_datastore((segment, var_to_obj))
                    num_sample_in += 1
                    suc = True
                    break
                if len(opdata.op.parameters) != len(segment_effect_objects):
                    # The number of objects in the segment does not match
                    # the number of parameters in the operator.
                    continue
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                suc, _, var_to_obj = self._match_objs2vars(
                    list(segment_objects),
                    list(opdata.op.parameters),
                    opdata.op.add_effects,
                    opdata.op.delete_effects,
                    segment,
                )
                if suc:
                    # Add to this PNAD.
                    assert set(var_to_obj.keys()) == set(opdata.op.parameters)
                    opdata.add_to_datastore((segment, var_to_obj))
                    num_sample_in += 1
                    break
            if not suc:
                # the sample does not fit any existing PNAD
                num_sample_out += 1

        logging.info(f"Number of samples in: {num_sample_in}, Specifically:")

        complete_opdatas = []
        for opdata in filtered_opdatas:
            logging.info(
                f"Number of samples in for {opdata.op.name}: {len(opdata.datastore)}"
            )
            preconditions = self._induce_preconditions_via_intersection(opdata)

            # Check if this operator matches a given operator
            if opdata.op.name in given_op_map:
                given_op = given_op_map[opdata.op.name]
                # Keep atoms from given operator (for known predicates)
                given_precond_atoms = set(given_op.preconditions)
                given_add_atoms = set(given_op.add_effects)
                given_delete_atoms = set(given_op.delete_effects)

                # Keep learned atoms with NEW predicates (not in given operator)
                new_precond_atoms = {
                    atom
                    for atom in preconditions
                    if atom.predicate not in given_predicates
                }
                new_add_atoms = {
                    atom
                    for atom in opdata.op.add_effects
                    if atom.predicate not in given_predicates
                }
                new_delete_atoms = {
                    atom
                    for atom in opdata.op.delete_effects
                    if atom.predicate not in given_predicates
                }

                # Combine: preserve given atoms + add newly learned atoms
                final_preconditions = given_precond_atoms | new_precond_atoms
                final_add_effects = given_add_atoms | new_add_atoms
                final_delete_effects = given_delete_atoms | new_delete_atoms

                logging.info(
                    f"Progressive learning for {opdata.op.name}: "
                    f"kept {len(given_precond_atoms)} given precond atoms + {len(new_precond_atoms)} new, "
                    f"kept {len(given_add_atoms)} given add atoms + {len(new_add_atoms)} new, "
                    f"kept {len(given_delete_atoms)} given delete atoms + {len(new_delete_atoms)} new"
                )

                complete_opdata = OpData(
                    opdata.op.copy_with(
                        preconditions=final_preconditions,
                        add_effects=final_add_effects,
                        delete_effects=final_delete_effects,
                    ),
                    opdata.datastore,
                )
            else:
                # New operator not in given_operators, use learned predicates as-is
                complete_opdata = OpData(
                    opdata.op.copy_with(
                        preconditions=preconditions,
                        add_effects=opdata.op.add_effects,
                        delete_effects=opdata.op.delete_effects,
                    ),
                    opdata.datastore,
                )

            complete_opdatas.append(complete_opdata)
        logging.info(f"Number of samples out: {num_sample_out}")

        return complete_opdatas, num_sample_out

    def _belief2opdatas_init(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        segments: List[Segment],
        given_op_map: Dict[str, LiftedOperator],
        _given_predicates: Set[BasePredicate],
    ) -> Tuple[List[OpData], List[int]]:
        """Initialize the opdatas with the belief that each segment is a new
        operator."""
        opdatas: List[OpData] = []
        if self._belief is None:
            return opdatas, []
        row_names = self._belief.row_names
        col_names = self._belief.col_names
        col_var_idx = self._belief.col_var_idx
        ae_matrix = self._belief.ae_matrix
        added_segment_idxs = []
        for i, operator in enumerate(row_names):
            params = operator.parameters
            preconds: Set[LiftedAtom] = set()  # will be learned later
            add_effects: Set[LiftedAtom] = set()
            delete_effects: Set[LiftedAtom] = set()
            if operator.name not in given_op_map:
                # For new operators, effects should not be learned again.
                # For both new and old predicates.
                for j, effect_p in enumerate(col_names):
                    if ae_matrix[i, j].sum() == 0:
                        # No effects for this predicate, skip it.
                        continue
                    assert ae_matrix[i, j].sum() == 1
                    if effect_p.arity == 0:
                        if ae_matrix[i, j, 0] == 1:
                            add_effects.add(LiftedAtom(effect_p, []))
                        else:
                            delete_effects.add(LiftedAtom(effect_p, []))
                        continue
                    var_idx = col_var_idx[j]
                    # Convert to list of ints regardless of input type
                    var_idx_list: List[int] = var_idx.tolist()
                    input_vars = self._get_pred_input_vars(
                        effect_p, var_idx_list, list(params)
                    )
                    lifted_atom = effect_p(input_vars)
                    if ae_matrix[i, j, 0] == 1:
                        add_effects.add(lifted_atom)
                    else:
                        delete_effects.add(lifted_atom)
            # For old operators, their effects will be added later from given_op_map
            # and old operator segment should not have effect with new predicates
            op = LiftedOperator(
                operator.name, params, preconds, add_effects, delete_effects
            )
            # Find a segment that has the same operator and effect
            datastore: Datastore = []
            opdata = OpData(op, datastore)
            for ids, segment in enumerate(segments):
                assert segment.actions[
                    0
                ].has_op(), f"Segment {ids} has no operator: {segment.actions[0]}"
                segment_op = segment.actions[0].op
                if segment_op is None:
                    continue
                segment_param_op = segment_op.parent
                segment_op_objs = tuple(segment_op.parameters)
                if segment_param_op.name == op.name:
                    if operator.name in given_op_map:
                        # For old operators, we will directly add segments
                        # that match the given operator names
                        # and the effects should be empty
                        if (
                            len(segment.add_effects) > 0
                            or len(segment.delete_effects) > 0
                        ):
                            # this segment has effects, skip it
                            continue
                        # we got the right segment
                        var_to_obj = dict(
                            zip(segment_param_op.parameters, segment_op_objs)
                        )
                        opdata.add_to_datastore((segment, var_to_obj))
                        added_segment_idxs.append(ids)
                        break
                    # For new operators, we need to match effects
                    # Try to see if the effects match
                    effect_objects = sorted(
                        {
                            o
                            for atom in segment.add_effects | segment.delete_effects
                            for o in atom.objects
                        }
                        | set(segment_op_objs)
                    )
                    objects_lst = segment_op.parameters
                    if len(objects_lst) != len(params):
                        # this can't be the right segment
                        continue
                    if len(effect_objects) != len(params):
                        # this can't be the right segment
                        # as the effects involves more objects not operated
                        continue
                    succ, _, var_to_obj = self._match_objs2vars(
                        list(objects_lst),
                        list(params),
                        add_effects,
                        delete_effects,
                        segment,
                    )
                    if succ:
                        opdata.add_to_datastore((segment, var_to_obj))
                        # we got the right segment
                        added_segment_idxs.append(ids)
                        break
            opdatas.append(opdata)
        return opdatas, added_segment_idxs
