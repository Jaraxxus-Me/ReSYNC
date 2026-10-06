"""The core algorithm for learning a collection of operator data structures."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Type

from skill_refactor.approaches.operator_learner.base import (
    BaseSTRIPSLearner,
    OpEffBelief,
)
from skill_refactor.approaches.operator_learner.belief_learner import (
    BeliefSTRIPSLearner,
)
from skill_refactor.approaches.operator_learner.belief_prog_learner import (
    BeliefProgSTRIPSLearner,
)
from skill_refactor.approaches.operator_learner.clustering_learner import (
    ClusteringSTRIPSLearner,
)
from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.utils.structs import (
    GroundAtomTrajectory,
    LiftedOperator,
    LowLevelTrajectory,
    Predicate,
    Segment,
    Task,
)

OPERATOR_LEARNERS: Dict[str, Type[BaseSTRIPSLearner]] = {
    "clustering": ClusteringSTRIPSLearner,
    "belief": BeliefSTRIPSLearner,
    "belief_prog": BeliefProgSTRIPSLearner,
}


def learn_operator_from_data(
    op_learner: str,
    trajectories: List[LowLevelTrajectory],
    train_tasks: List[Task],
    predicates: Set[Predicate],
    given_operators: Set[LiftedOperator],
    ground_atom_dataset: List[GroundAtomTrajectory],
    operator_belief: Optional[OpEffBelief] = None,
) -> Tuple[
    Set[LiftedOperator], List[List[Segment]], Dict[Segment, LiftedOperator], int
]:
    """Learn Operators from the given dataset of low-level transitions, using the given
    set of predicates.

    There are three return values: (1) The final set of Operators. (2) The
    segmented trajectories. These are returned because any operator that
    were learned will be contained properly in these segments. (3) A
    mapping from segment to operators. This is returned because not all
    segments in return value (2) are necessarily covered by an operators, in
    the case that we are enforcing a min_data (see
    base_strips_learner.py).
    """
    logging.info(f"\nLearning Operators on {len(trajectories)} trajectories...")

    # Search over data orderings to find least complex opdata set.
    # If the strips learner is not Backchaining then it will
    # only do one iteration, because all other approaches are
    # data order invariant.
    # smallest_opdatas = None
    # smallest_opdata_complexity = float('inf')

    # STEP 1: Segment each trajectory in the dataset based on changes in
    #         either predicates or options. If we are doing option learning,
    #         then the data will not contain options, so this segmenting
    #         procedure only uses the predicates.
    segmented_trajs = [
        segment_trajectory(traj, atom_seq=atom_seq)
        for traj, atom_seq in ground_atom_dataset
    ]

    # STEP 2: Learn STRIPS operators from the data, and use them to
    #         produce opdata objects. Each opdata
    #         contains a STRIPSOperator, Datastore, and OptionSpec. The
    #         samplers will be filled in on a later step.
    op_learner_cls = OPERATOR_LEARNERS[op_learner]
    learner_instance = op_learner_cls(
        trajectories=trajectories,
        train_tasks=train_tasks,
        predicates=predicates,
        segmented_trajs=segmented_trajs,
        verify_harmlessness=True,
        verbose=True,
        operator_belief=operator_belief,
    )
    opdatas, num_samples_out = learner_instance.learn(given_operators)

    # Save least complex learned opdata set across data orderings.
    # opdatas_complexity = sum(opdata.op.get_complexity() for opdata in opdatas)
    # if opdatas_complexity < smallest_opdata_complexity:
    #     smallest_opdata_complexity = opdatas_complexity
    #     smallest_opdatas = opdatas
    # assert smallest_opdatas is not None  # smallest opdatas should be set here

    # assert smallest_opdatas is not None
    # opdatas = smallest_opdatas

    # We delete ground_atom_dataset because it's prone to causing bugs --
    # we should rarely care about the low-level ground atoms sequence after
    # segmentation.
    del ground_atom_dataset

    # STEP 5: Make, log, and return the Operators.
    Operators = []
    seg_to_operator = {}
    for opdata in opdatas:
        operator = opdata.op
        Operators.append(operator)
        for seg, _ in opdata.datastore:
            assert seg not in seg_to_operator
            seg_to_operator[seg] = operator
    logging.info("\nLearned Operators:")
    for operator in Operators:
        logging.info(operator)
    logging.info("")

    return set(Operators), segmented_trajs, seg_to_operator, num_samples_out
