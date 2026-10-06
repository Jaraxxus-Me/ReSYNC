"""STRIPS learner that leverages access to oracle operators used to generate
demonstrations via bilevel planning."""

import logging
from itertools import permutations, product
from typing import Dict, List, Set, Tuple

from relational_structs import LiftedAtom

from skill_refactor.approaches.operator_learner.base import BaseSTRIPSLearner
from skill_refactor.utils.structs import (
    Datastore,
    LiftedOperator,
    Object,
    OpData,
    Predicate,
    Segment,
    Variable,
)


class BeliefSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators but re-learns all the
    components via currently-implemented methods in the base class.

    This is different from the oracle learner because here, we assume that our demo data
    is annotated with the ground-truth operators used to produce it. We thus know
    exactly how to associate (i.e, cluster) demos into sets corresponding to each
    operator.
    """

    def __post_init__(self) -> None:
        """Post-initialization to ensure the belief is set."""
        assert self._belief, "Belief must be set before calling this method."
        assert len(self._belief.row_names) == self._belief.ae_matrix.shape[0]
        assert len(self._belief.col_names) == self._belief.ae_matrix.shape[1]
        assert len(self._belief.col_names) == len(self._belief.col_var_idx)

    def _learn(self, given_operators: Set[LiftedOperator]) -> Tuple[List[OpData], int]:
        """Re-learn operator components from the given operators and the segmented
        trajectories."""
        num_sample_in = 0
        num_sample_out = 0
        segments: List[Segment] = [
            seg for segs in self._segmented_trajs for seg in segs
        ]
        # Cluster the segments according to common option and effects.
        opdatas, added_segment_idxs = self._belief2opdatas_init(segments)
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

    def _get_pred_input_vars(
        self, pred: Predicate, col_var_idx: List[int], op_param: List[Variable]
    ) -> List:
        """Get the input variables of a predicate."""
        # Update this function for same type inputs.
        opt_param_dict: Dict[str, List] = {}
        for var in op_param:
            if var.type.name not in opt_param_dict:
                opt_param_dict[var.type.name] = [var]
            else:
                opt_param_dict[var.type.name].append(var)
        input_vars = []
        for i, pt in enumerate(pred.types):
            n = col_var_idx[i]
            if n >= len(opt_param_dict[pt.name]):
                input_vars.append(opt_param_dict[pt.name][0])
            else:
                input_vars.append(opt_param_dict[pt.name][n])
        return input_vars

    def _match_objs2vars(
        self,
        objects: List[Object],
        params: List[Variable],
        add_effects: Set[LiftedAtom],
        delete_effects: Set[LiftedAtom],
        segment: Segment,
    ) -> Tuple[bool, Dict, Dict]:
        """Match objects to variables by checking equivalent effects."""
        # Important: we assume objects are ordered in the same way as params
        # itertools imported at module level

        # Step 1: Group variables and objects by their types.
        type_to_vars: Dict[str, List[Variable]] = {}
        type_to_objs: Dict[str, List[Object]] = {}

        for var in params:
            type_to_vars.setdefault(var.type.name, []).append(var)

        for obj in objects:
            type_to_objs.setdefault(obj.type.name, []).append(obj)

        # Step 2: Check for type mismatches or insufficient objects.
        for type_, vars_list in type_to_vars.items():
            if type_ not in type_to_objs or len(vars_list) > len(type_to_objs[type_]):
                # Not enough objects to match variables of this type.
                return False, {}, {}
        # Step 2.5: Check if the objects are already matched with current lists.
        full_var_to_obj_mapping = {}
        for var, obj in zip(params, objects):
            full_var_to_obj_mapping[var] = obj

        full_obj_to_var_mapping = {
            obj: var for var, obj in full_var_to_obj_mapping.items()
        }
        add_effects_seg = set()
        delete_effects_seg = set()

        for atom in segment.add_effects:
            part_object_to_var_mapping = {
                o: full_obj_to_var_mapping[o] for o in atom.objects
            }
            lifted_atom = atom.lift(part_object_to_var_mapping)
            add_effects_seg.add(lifted_atom)
        for atom in segment.delete_effects:
            part_object_to_var_mapping = {
                o: full_obj_to_var_mapping[o] for o in atom.objects
            }
            lifted_atom = atom.lift(part_object_to_var_mapping)
            delete_effects_seg.add(lifted_atom)

        if add_effects_seg == add_effects and delete_effects_seg == delete_effects:
            # Successful mapping found.
            succ = True
            var_to_obj = full_var_to_obj_mapping
            obj_to_var = full_obj_to_var_mapping
            return succ, obj_to_var, var_to_obj

        # Step 3: Generate all possible mappings for each type.
        mappings_per_type = []

        for type_, vars_list in type_to_vars.items():
            objs_list = type_to_objs[type_]

            # Generate all permutations of objects assigned to variables.
            type_mappings = [
                dict(zip(vars_list, objs_perm))
                for objs_perm in permutations(objs_list, len(vars_list))
            ]
            mappings_per_type.append(type_mappings)

        # Step 4: Generate all combinations of type-specific mappings.
        all_mappings = product(*mappings_per_type)

        # Step 5: Test each combined mapping.
        for mapping_combination in all_mappings:
            full_var_to_obj_mapping = {}
            used_objs = set()
            valid_mapping = True

            # Combine mappings from all types.
            for mapping in mapping_combination:
                for var, obj in mapping.items():
                    if obj in used_objs:
                        # Object already assigned; skip this mapping.
                        valid_mapping = False
                        break
                    full_var_to_obj_mapping[var] = obj
                    used_objs.add(obj)
                if not valid_mapping:
                    break

            if not valid_mapping:
                continue  # Try the next mapping combination.

            # Step 6: Create object-to-variable mapping.
            obj_to_var = {obj: var for var, obj in full_var_to_obj_mapping.items()}

            # Step 7: Lift the segment effects using the current mapping.
            add_effects_seg = {atom.lift(obj_to_var) for atom in segment.add_effects}
            delete_effects_seg = {
                atom.lift(obj_to_var) for atom in segment.delete_effects
            }

            # Step 8: Check if the lifted effects match the target effects.
            if add_effects_seg == add_effects and delete_effects_seg == delete_effects:
                # Successful mapping found.
                succ = True
                var_to_obj = full_var_to_obj_mapping
                return succ, obj_to_var, var_to_obj

        # No valid mapping found after exhausting all possibilities.
        return False, {}, {}

    def _belief2opdatas_init(
        self, segments: List[Segment]
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
            # every operator has non-empty effects
            # assert add_effects or delete_effects
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
