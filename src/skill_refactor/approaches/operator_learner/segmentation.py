"""Methods for segmenting low-level trajectories into segments."""

from typing import Callable, List, Set

from torch import Tensor

from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    ApproachStepResult,
    GroundAtom,
    LowLevelTrajectory,
    Segment,
)


def segment_trajectory(
    ll_traj: LowLevelTrajectory, atom_seq: List[Set[GroundAtom]]
) -> List[Segment]:
    """Segment a ground atom trajectory."""
    # Start with the segmenters that don't need atom_seq. Still pass it in
    # because if it was provided, it can be used to avoid calling abstract.
    if CFG.traj_segmenter == "operator_changes":
        return _segment_with_operator_changes(ll_traj, atom_seq)
    raise NotImplementedError(f"Unrecognized segmenter: {CFG.traj_segmenter}.")


def _segment_with_operator_changes(
    ll_traj: LowLevelTrajectory, atom_seq: List[Set[GroundAtom]]
) -> List[Segment]:
    """Segment a trajectory whenever the (assumed known) option changes."""

    def _switch_fn(t: int) -> bool:
        # Segment by checking whether the option changes on the next step.
        op_t = ll_traj.actions[t].op
        # As a special case, if this is the last timestep, return True
        if t == len(ll_traj.actions) - 1:
            # Calculate the number of steps since the option changed.
            backward_t = t
            while backward_t > 0:
                if ll_traj.actions[backward_t - 1].op is not op_t:
                    break
                backward_t -= 1
            return True
        return op_t is not ll_traj.actions[t + 1].op

    return _segment_with_switch_function(ll_traj, atom_seq, _switch_fn)


def _segment_with_switch_function(
    ll_traj: LowLevelTrajectory,
    atom_seq: List[Set[GroundAtom]],
    switch_fn: Callable[[int], bool],
) -> List[Segment]:
    """Helper for other segmentation methods.

    The switch_fn takes in a timestep and returns True if the trajectory should be
    segmented at the end of that timestep.
    """
    segments = []
    assert len(ll_traj.states) > 0
    current_segment_states: List[Tensor] = []
    current_segment_actions: List[ApproachStepResult] = []
    assert len(ll_traj.states) == len(atom_seq)
    current_segment_init_atoms = atom_seq[0]
    for t in range(len(ll_traj.actions)):
        current_segment_states.append(ll_traj.states[t])
        current_segment_actions.append(ll_traj.actions[t])
        if switch_fn(t):
            # Include the final state as the end of this segment.
            current_segment_states.append(ll_traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(
                current_segment_states,
                current_segment_actions,
                ll_traj.train_scenario_idx,
                ll_traj.success,
                ll_traj.train_task_idx,
            )
            current_segment_final_atoms = atom_seq[t + 1]
            action_op = ll_traj.actions[t].op
            assert (
                action_op is not None
            ), "Action must have an operator to create segment"
            segment = Segment(
                current_segment_traj,
                current_segment_init_atoms,
                current_segment_final_atoms,
                action_op,
            )
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
            current_segment_init_atoms = current_segment_final_atoms
    return segments
