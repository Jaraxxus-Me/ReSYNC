"""Cornered-stacking task config.

Three scene objects (cornered_block / cornered_box assets):
    * ``receptive_object``  — the box (kinematic), reset within the existing
      ``c_stack_reset_cube_*`` range, just like the 2-object task.
    * ``insertive_object``  — the in-box block, placed at the USD
      ``assembled_offset`` relative pose each reset. Uses
      ``cornered_block/block/block.usd``.
    * ``outside_block``     — a second block of the same USD, placed outside
      the box at ``box_pos_w + (dx, dy, dz)`` (world frame) with a random yaw.
      The ranges live in ``CFG.cstack_outside_*``.

Goal (sparse reward only): the two blocks form a small tower — one stacked
on the other with yaw aligned and both upright — and the tower is grasped
off the table by the gripper. :data:`CFG.cstack_in_box_role` selects
which block plays which role:
    * ``"grasp"`` → the in-box block is the base (gripped at success).
    * ``"stack"`` → the in-box block is the top (stacked on the outside one).

Only safety signals (time-out, abnormal-robot termination) are kept from the
parent env — no dense shaping.

IMPORTANT: Isaac Lab's ``AppLauncher`` must be started before this module is
imported, since it transitively pulls in isaaclab configclasses.
"""

from __future__ import annotations

import math
import os
from typing import Any

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    subtract_frame_transforms,
    wrap_to_pi,
)
from uwlab_assets import custom_cloud_path
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    ObservationsCfg as BlockBoxObservationsCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    RlStateSceneCfg,
    TerminationsCfg,
    TrainEvalEventCfg,
    Ur5eRobotiq2f140RelCartesianOSCEvalCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import (
    MultiResetManager,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_root_states_uniform,
    target_asset_pose_in_root_asset_frame,
)

from resync_isaac.paths import ASSET_ROOT
from skill_refactor.settings import CFG

# ---------------------------------------------------------------------------
# Gym task id & constants
# ---------------------------------------------------------------------------

CORNERED_STACKING_TASK_ID = "CornerStacking-Ur5eRobotiq2f140-CorneredStacking-v0"

_BLOCK_USD = str(
    ASSET_ROOT / "simulation/objects/objects/single_ur_scene/block/block.usd"
)
_LOCAL_OMNIRESET_DIR = str(ASSET_ROOT / "corner_stacking")
_POST_RL_RESETS_DIR = str(ASSET_ROOT / "corner_stacking")

_ZERO_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Scene: add outside_block on top of the parent scene
# ---------------------------------------------------------------------------


@configclass
class CorneredStackingSceneCfg(RlStateSceneCfg):
    """Parent scene (robot, insertive_object, receptive_object, table, …) plus
    a second block sitting outside the box.

    The USD defaults to ``cornered_block/block/block.usd`` so the randomization
    ranges inherited from ``BaseEventCfg.insertive_object_material`` /
    ``randomize_insertive_object_mass`` are appropriate. The env factory may
    swap USDs via ``omnireset_variants`` before the scene is built.
    """

    outside_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OutsideBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BLOCK_USD,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


# ---------------------------------------------------------------------------
# Events: dataset-driven robot/insertive/receptive reset + outside-block reset
# ---------------------------------------------------------------------------
#
# The dataset at ``third-party/UWLab/Datasets/OmniReset/Resets/block__box/
# resets_ObjectAnywhereEEAnywhere.pt`` (loaded by ``MultiResetManager``
# below) provides the full robot-joint + insertive-block + box-pose reset
# distribution — that is the intended ground truth for this env. The
# cornered-stacking task adds ONE scene entity the dataset does not know
# about (``outside_block``), so the only custom reset event is sampling
# that entity's pose relative to the box's dataset-sampled pose.


def _outside_range() -> dict[str, tuple[float, float]]:
    return {
        "x": tuple(CFG.cstack_outside_dx_range),
        "y": tuple(CFG.cstack_outside_dy_range),
        "z": tuple(CFG.cstack_outside_dz_range),
        "yaw": tuple(CFG.cstack_outside_yaw_range),
    }


@configclass
class CorneredStackingEventCfg(TrainEvalEventCfg):
    """Dataset-based reset of robot + insertive + receptive (inherited), plus
    a custom reset for ``outside_block`` relative to the box's sampled pose.

    No pose overrides on insertive_object or receptive_object — those come
    straight from ``resets_ObjectAnywhereEEAnywhere.pt``. The RL policy is
    expected to drive insertive to the assembled-in-box pose itself.
    """

    # Point MultiResetManager at the local block__box dataset (the HuggingFace
    # mirror does not host this pair). Re-declaring the term in the subclass
    # replaces the parent's URL-based one.
    reset_from_reset_states: EventTerm = EventTerm(
        func=MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": _LOCAL_OMNIRESET_DIR,
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

    # Startup domain randomization for the new asset, mirroring the
    # insertive object's ranges (same USD → same expected dynamics).
    outside_block_material: EventTerm = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (1.0, 2.0),
            "dynamic_friction_range": (0.9, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("outside_block"),
            "make_consistent": True,
        },
    )

    randomize_outside_block_mass: EventTerm = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("outside_block"),
            "mass_distribution_params": (0.02, 0.2),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # Sample the outside block at box_pos_w + (dx, dy, dz) with random yaw.
    # Runs after MultiResetManager because it is declared in the subclass.
    reset_outside_pose: EventTerm = EventTerm(
        func=reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": _ZERO_VELOCITY_RANGE,
            "asset_cfgs": {"outside_block": SceneEntityCfg("outside_block")},
            "offset_asset_cfg": SceneEntityCfg("receptive_object"),
            "offset_use_current_pose": True,
        },
    )

    def __post_init__(self):  # type: ignore[override]
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.reset_outside_pose.params["pose_range"] = _outside_range()

        # Switch the dataset_dir / reset_types pair on the inherited
        # ``reset_from_reset_states`` term based on CFG. ``MultiResetManager``
        # reads its params at __init__ (via cfg.params), so updating them
        # here — before the env is constructed — is the supported override
        # point.
        if getattr(CFG, "c_stack_reset_from_post_rl", False):
            self.reset_from_reset_states.params["dataset_dir"] = _POST_RL_RESETS_DIR
            self.reset_from_reset_states.params["reset_types"] = ["RLTerminated"]
            self.reset_from_reset_states.params["probs"] = [1.0]
        else:
            self.reset_from_reset_states.params["dataset_dir"] = _LOCAL_OMNIRESET_DIR
            self.reset_from_reset_states.params["reset_types"] = [
                "ObjectAnywhereEEAnywhere"
            ]
            self.reset_from_reset_states.params["probs"] = [1.0]


# ---------------------------------------------------------------------------
# Sparse reward: tower-formed-and-grasped
# ---------------------------------------------------------------------------


class TowerGraspedSuccess(ManagerTermBase):
    """Single sparse success term.

    Returns 1.0 iff all of the following hold, else 0.0:

    * Base block upright:  |roll| + |pitch| < ``upright_tol``.
    * Top block upright:   same.
    * Top stacked on base: base-frame ``‖(dx, dy)‖ < stack_xy_tol``,
      ``|dz − stack_z_offset| < stack_z_tol`` (yaw-symmetric; see below).
    * Top yaw aligned to base modulo the cube's 4-fold symmetry (π/2).
    * EE near base block:  base-frame xy distance < ``grasp_ee_xy_tol`` and
      z distance < ``grasp_ee_z_tol``.
    * Base lifted:         base-block world z > ``lift_world_z_min``.
    * Gripper closed:      ``finger_joint > finger_closed_min``.

    Which scene entity plays "base" vs "top" is chosen by
    :data:`CFG.cstack_in_box_role`:
        ``"grasp"`` → insertive_object is base, outside_block is top.
        ``"stack"`` → outside_block is base, insertive_object is top.

    ``.success`` is exposed as a (num_envs,) bool tensor so helpers like
    ``resync_isaac.corner_stacking.utils.get_success`` keep
    working against a reward term named ``progress_context``.
    """

    def __init__(self, cfg: RewTerm, env: Any) -> None:
        super().__init__(cfg, env)

        in_box_role: str = cfg.params["in_box_role"]
        if in_box_role == "grasp":
            base_name, top_name = "insertive_object", "outside_block"
        elif in_box_role == "stack":
            base_name, top_name = "outside_block", "insertive_object"
        else:
            raise ValueError(
                f"cstack_in_box_role must be 'grasp' or 'stack', got {in_box_role!r}"
            )
        self._base_name = base_name
        self._top_name = top_name

        robot: Articulation = env.scene["robot"]
        grip_body_name: str = cfg.params["grip_body_name"]
        finger_joint_name: str = cfg.params["finger_joint_name"]
        body_ids, _ = robot.find_bodies(grip_body_name)
        joint_ids, _ = robot.find_joints(finger_joint_name)
        self._grip_body_id = int(body_ids[0])
        self._finger_joint_id = int(joint_ids[0])

        # Per-env outputs
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: Any,
        in_box_role: str,
        grip_body_name: str,
        finger_joint_name: str,
        upright_tol: float,
        stack_xy_tol: float,
        stack_z_offset: float,
        stack_z_tol: float,
        stack_yaw_tol: float,
        grasp_ee_xy_tol: float,
        grasp_ee_z_tol: float,
        lift_world_z_min: float,
        finger_closed_min: float,
    ) -> torch.Tensor:
        del in_box_role, grip_body_name, finger_joint_name  # captured at init

        base: RigidObject = env.scene[self._base_name]
        top: RigidObject = env.scene[self._top_name]
        robot: Articulation = env.scene["robot"]

        base_pos = base.data.root_pos_w
        base_quat = base.data.root_quat_w
        top_pos = top.data.root_pos_w
        top_quat = top.data.root_quat_w

        # Upright: roll/pitch ≈ 0 in world frame for both blocks.
        eb_x, eb_y, _ = euler_xyz_from_quat(base_quat)
        et_x, et_y, _ = euler_xyz_from_quat(top_quat)
        base_upright = (wrap_to_pi(eb_x).abs() + wrap_to_pi(eb_y).abs()) < upright_tol
        top_upright = (wrap_to_pi(et_x).abs() + wrap_to_pi(et_y).abs()) < upright_tol

        # Top pose in base frame.
        rel_pos, rel_quat = subtract_frame_transforms(
            base_pos, base_quat, top_pos, top_quat
        )
        xy_dist = rel_pos[:, :2].norm(dim=-1)
        z_err = (rel_pos[:, 2] - stack_z_offset).abs()
        _, _, yaw_rel = euler_xyz_from_quat(rel_quat)
        # Cubes are 4-fold symmetric about z → wrap to [-π/4, π/4].
        quarter = math.pi / 2.0
        yaw_sym = (
            torch.remainder(wrap_to_pi(yaw_rel) + quarter / 2.0, quarter)
            - quarter / 2.0
        ).abs()

        stacked = (
            (xy_dist < stack_xy_tol) & (z_err < stack_z_tol) & (yaw_sym < stack_yaw_tol)
        )

        # Gripper near base, base lifted, finger closed.
        ee_pos_w = robot.data.body_pos_w[:, self._grip_body_id]
        ee_to_base = base_pos - ee_pos_w
        ee_near = (ee_to_base[:, :2].norm(dim=-1) < grasp_ee_xy_tol) & (
            ee_to_base[:, 2].abs() < grasp_ee_z_tol
        )
        lifted = base_pos[:, 2] > lift_world_z_min
        finger_pos = robot.data.joint_pos[:, self._finger_joint_id]
        closed = finger_pos > finger_closed_min
        grasped = ee_near & lifted & closed

        self.success[:] = base_upright & top_upright & stacked & grasped
        return self.success.float()


@configclass
class CorneredStackingRewardsCfg:
    """Sparse reward only: 1.0 on tower-grasped success, 0.0 otherwise.

    Named ``progress_context`` so ``get_success(env)`` (which looks up that
    term) continues to work.
    """

    progress_context: RewTerm = RewTerm(
        func=TowerGraspedSuccess,
        weight=1.0,
        params={
            "in_box_role": "grasp",
            "grip_body_name": "robotiq_base_link",
            "finger_joint_name": "finger_joint",
            "upright_tol": 0.25,
            "stack_xy_tol": 0.02,
            "stack_z_offset": 0.04,
            "stack_z_tol": 0.02,
            "stack_yaw_tol": 0.3,
            "grasp_ee_xy_tol": 0.03,
            "grasp_ee_z_tol": 0.05,
            "lift_world_z_min": 0.04,
            "finger_closed_min": 0.4,
        },
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        p = self.progress_context.params
        p["in_box_role"] = CFG.cstack_in_box_role
        p["upright_tol"] = float(CFG.cstack_upright_tol)
        p["stack_xy_tol"] = float(CFG.cstack_stack_xy_tol)
        p["stack_z_offset"] = float(CFG.cstack_stack_z_offset)
        p["stack_z_tol"] = float(CFG.cstack_stack_z_tol)
        p["stack_yaw_tol"] = float(CFG.cstack_stack_yaw_tol)
        p["grasp_ee_xy_tol"] = float(CFG.cstack_grasp_ee_xy_tol)
        p["grasp_ee_z_tol"] = float(CFG.cstack_grasp_ee_z_tol)
        p["lift_world_z_min"] = float(CFG.cstack_lift_world_z_min)
        p["finger_closed_min"] = float(CFG.cstack_finger_closed_min)


# ---------------------------------------------------------------------------
# Observations: canonical PolicyCfg (superset) — block_box layout + outside_block
# ---------------------------------------------------------------------------
#
# Contract with downstream skill adapters:
#   * ``outside_block_pose`` is intentionally UNANNOTATED and appended in the
#     subclass body. Isaac Lab's ``ObservationManager`` promotes annotated
#     fields ahead of unannotated ones in the concatenated tensor, so the
#     parent's fields keep their original order and the new term lands at
#     the end.
#   * ``history_length``, ``enable_corruption``, ``concatenate_terms`` and the
#     ``rotation_repr="axis_angle"`` on pose terms all match the block_box
#     training cfg — so the first 215 columns of the canonical policy obs are
#     byte-identical to the block_box training obs. A skill whose policy was
#     trained on block_box can run on this env by slicing ``obs[..., :215]``.
#   * ``CriticCfg`` is inherited unchanged, so the pretrained critic's 204-dim
#     input still matches and no critic adapter is needed.


_POLICY_HISTORY: int = 5
_ACTION_DIM: int = 7  # 6 OSC + 1 finger (Ur5eRobotiq2f140RelativeOSCAction)
_JOINT_DIM: int = 12  # UR5e (6) + Robotiq 2f140 (6)
_POSE_DIM: int = 6  # 3 pos + 3 axis-angle

# Dimension of the block_box-trained policy's flat obs tensor. Enumerated
# explicitly so the skill adapter (``prefix_slice_adapter(BLOCK_BOX_POLICY_OBS_DIM)``)
# is self-documenting and this file is the single source of truth.
# Layout (after annotated-first promotion): ``[insertive_in_receptive |
# prev_actions | joint_pos | ee_pose_b | insertive_in_ee | receptive_in_ee]``
BLOCK_BOX_POLICY_OBS_DIM: int = _POLICY_HISTORY * (
    _POSE_DIM + _ACTION_DIM + _JOINT_DIM + _POSE_DIM + _POSE_DIM + _POSE_DIM
)  # 5 * 43 = 215


@configclass
class CorneredStackingObservationsCfg(BlockBoxObservationsCfg):
    """Canonical observations: block_box PolicyCfg + outside_block_pose term."""

    @configclass
    class PolicyCfg(BlockBoxObservationsCfg.PolicyCfg):
        """Extend the policy observations with the outside block pose."""

        outside_block_pose = ObsTerm(
            func=target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("outside_block"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Full env cfg
# ---------------------------------------------------------------------------


@configclass
class CorneredStackingCfg(Ur5eRobotiq2f140RelCartesianOSCEvalCfg):
    """Three-block sparse-reward corner-stacking task."""

    scene: CorneredStackingSceneCfg = CorneredStackingSceneCfg(
        num_envs=1, env_spacing=1.5
    )
    events: CorneredStackingEventCfg = CorneredStackingEventCfg()
    observations: CorneredStackingObservationsCfg = CorneredStackingObservationsCfg()
    rewards: CorneredStackingRewardsCfg = CorneredStackingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()


# ---------------------------------------------------------------------------
# Gym registration
# ---------------------------------------------------------------------------


gym.register(
    id=CORNERED_STACKING_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CorneredStackingCfg",
    },
)
