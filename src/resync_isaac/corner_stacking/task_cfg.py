"""Custom CornerStacking task config.

Inherits the upstream UWLab OmniReset cube eval task (dataset-driven reset of
the robot + objects) and *appends* per-cube reset events that overwrite the
two cube root poses with a uniform distribution in x/y/yaw. The default
robot-joint reset from :class:`TrainEvalEventCfg` (dataset-backed) is kept.

IMPORTANT: Isaac Lab's ``AppLauncher`` must be started before this module is
imported, since it transitively pulls in isaaclab configclasses.
"""

from __future__ import annotations

import gymnasium as gym
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    TrainEvalEventCfg,
    Ur5eRobotiq2f140RelCartesianOSCEvalCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import (  # re-exports isaaclab.envs.mdp
    reset_root_state_uniform,
)

from skill_refactor.settings import CFG

# ---------------------------------------------------------------------------
# Gym task id
# ---------------------------------------------------------------------------

CORNER_STACKING_CUSTOM_TASK_ID = "CornerStacking-Ur5eRobotiq2f140-CubeReset-v0"


# ---------------------------------------------------------------------------
# Event cfg
# ---------------------------------------------------------------------------


def _cube_pose_range() -> dict[str, tuple[float, float]]:
    """Build the uniform sampling range for both cubes.

    Cubes sit on the table, so z/roll/pitch stay at their default. Only
    x, y (table plane) and yaw get randomized.
    """
    return {
        "x": tuple(CFG.c_stack_reset_cube_x_range),
        "y": tuple(CFG.c_stack_reset_cube_y_range),
        "yaw": tuple(CFG.c_stack_reset_cube_yaw_range),
    }


_ZERO_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


@configclass
class CornerStackingEventCfg(TrainEvalEventCfg):
    """Inherit dataset-based robot+object reset, then overwrite cube poses.

    Order matters: because these fields are declared in the subclass, Isaac
    Lab's EventManager applies them *after* ``reset_from_reset_states``, so
    the uniform cube sampling overrides the dataset's cube pose while the
    dataset's robot-joint state is preserved.

    Cube ranges are resolved from :data:`CFG` at ``__post_init__`` time so
    that callers can override ``c_stack_reset_cube_*`` before constructing
    the env cfg.
    """

    reset_grasp_cube_pose: EventTerm = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": _ZERO_VELOCITY_RANGE,
            "asset_cfg": SceneEntityCfg("insertive_object"),
        },
    )

    reset_target_cube_pose: EventTerm = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": _ZERO_VELOCITY_RANGE,
            "asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        pose_range = _cube_pose_range()
        self.reset_grasp_cube_pose.params["pose_range"] = pose_range
        self.reset_target_cube_pose.params["pose_range"] = pose_range


# ---------------------------------------------------------------------------
# Task cfg
# ---------------------------------------------------------------------------


@configclass
class CornerStackingCustomCfg(Ur5eRobotiq2f140RelCartesianOSCEvalCfg):
    """Eval task with uniform cube-pose reset on top of the default robot reset."""

    events: CornerStackingEventCfg = CornerStackingEventCfg()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


gym.register(
    id=CORNER_STACKING_CUSTOM_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CornerStackingCustomCfg",
    },
)
