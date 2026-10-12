"""Pose- and state-extraction helpers for the CornerStacking benchmark.

Two families of helpers live here:

1. **Env-based** (``get_ee_pose_in_world`` etc.): read directly from the live
   Isaac Lab scene. Used by tests/visualizers that need world-frame data,
   and at construction time to resolve articulation layout.
2. **Obs-based** (:class:`CornerStackingObsLayout`): operate on the policy
   observation tensor only, with no env access. Perceiver and scripted
   skills use this path at runtime so they are decoupled from the env.
"""

from __future__ import annotations

import inspect
import os
from importlib import metadata
from typing import Any

import torch
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate,
    subtract_frame_transforms,
    wrap_to_pi,
)
from relational_structs import Object
from torch import Tensor

from skill_refactor.settings import CFG

# ---------------------------------------------------------------------------
# Test / collection wiring helpers
# ---------------------------------------------------------------------------


def make_corner_stacking_frame_markers() -> tuple[Any, Any]:
    """Create EE goal/current frame markers for corner-stacking visualizers."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG

    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ee_goal"))
    ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ee_current"))
    return goal_marker, ee_marker


def resolve_block_box_rl_checkpoint() -> str | None:
    """Return the block_box RL checkpoint path, or ``None`` if unavailable."""
    checkpoint = os.path.abspath(os.path.expanduser(str(CFG.c_stack_checkpoint)))
    return checkpoint if os.path.isfile(checkpoint) else None


def load_block_box_rl_policy(
    raw_env: Any,
    agent_cfg: Any,
    checkpoint: str,
) -> tuple[Any, Any]:
    """Load the pretrained block_box RSL-RL policy for the c_stack RLSkill.

    The actor was trained on the 215-column block_box observation layout. The
    cornered-stacking env emits a canonical superset, so this constructs the
    runner against a temporary prefix-sliced wrapper and returns the inference
    policy plus that wrapper.
    """
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from resync_isaac.corner_stacking.cornered_stacking_task_cfg import (
        BLOCK_BOX_POLICY_OBS_DIM,
    )
    from resync_isaac.wrappers import (
        SkillVecEnvWrapper,
        prefix_slice_adapter,
    )

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    alg_cfg = agent_cfg.algorithm
    class_name = getattr(alg_cfg, "class_name", None)
    if class_name is not None:
        from rsl_rl import algorithms

        alg_class = getattr(algorithms, class_name, None)
        if alg_class is not None:
            accepted = set(inspect.signature(alg_class.__init__).parameters.keys())
            for key in list(vars(alg_cfg)):
                if key != "class_name" and key not in accepted:
                    delattr(alg_cfg, key)

    loader_env = SkillVecEnvWrapper(
        raw_env,
        obs_adapter=prefix_slice_adapter(BLOCK_BOX_POLICY_OBS_DIM),
        clip_actions=agent_cfg.clip_actions,
    )
    runner = OnPolicyRunner(
        loader_env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    resume_path = retrieve_file_path(checkpoint)
    print(f"[c_stack] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=loader_env.device)
    return policy, loader_env


def make_corner_stacking_task_planner(
    tamp_system: Any,
    wrapped_env: Any,
    planner_id: str = "pyperplan",
    domain_name: str = "CornerStacking-domain",
) -> Any:
    """Construct the standard c_stack task-then-motion planner."""
    from skill_refactor.utils.ttmp import TaskThenMotionPlanner

    num_actions = wrapped_env.num_actions
    device = wrapped_env.device
    fallback_action = torch.zeros(num_actions, device=device).cpu().numpy()
    dummy_low = torch.full((num_actions,), -1.0, device=device)
    dummy_high = torch.full((num_actions,), 1.0, device=device)
    return TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fallback_action,
        normalize_action=False,
        arm_action_low=dummy_low,
        arm_action_high=dummy_high,
        planner_id=planner_id,
        domain_name=domain_name,
    )


def pad_or_truncate_action(action: Tensor, num_actions: int) -> Tensor:
    """Return an action tensor with exactly ``num_actions`` columns."""
    if action.shape[-1] == num_actions:
        return action
    if action.shape[-1] > num_actions:
        return action[:, :num_actions]
    padded = torch.zeros(action.shape[0], num_actions, device=action.device)
    padded[:, : action.shape[-1]] = action
    return padded


def corner_stacking_goal_atom(perceiver: Any, goal_0on1: bool | None = None) -> Any:
    """Return the configured c_stack ``On`` goal atom."""
    if goal_0on1 is None:
        goal_0on1 = bool(CFG.c_stack_goal_0on1)
    On = perceiver.predicates_container["On"]
    objs = perceiver.objects.as_dict()
    if goal_0on1:
        return On([objs["block0"], objs["block1"]])
    return On([objs["block1"], objs["block0"]])


# ---------------------------------------------------------------------------
# Robot / EE state
# ---------------------------------------------------------------------------


def get_robot_body_pose_in_base(env: Any, body_name: str) -> tuple[Tensor, Tensor]:
    """Get a robot body pose in robot base frame.

    Returns (pos_b, quat_b) each of shape (N, 3) and (N, 4).
    """
    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.find_bodies(body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    return ee_pos_b, ee_quat_b


def get_ee_pose_in_base(env: Any) -> tuple[Tensor, Tensor]:
    """Get EE (wrist_3_link) pose in robot base frame.

    Returns (pos_b, quat_b) each of shape (N, 3) and (N, 4).
    """
    return get_robot_body_pose_in_base(env, "wrist_3_link")


def get_ee_pose_in_world(env: Any) -> tuple[Tensor, Tensor]:
    """Get EE (wrist_3_link) pose in world frame.

    Returns (pos_w, quat_w) each of shape (N, 3) and (N, 4).
    """
    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    return (
        robot.data.body_pos_w[:, ee_idx],
        robot.data.body_quat_w[:, ee_idx],
    )


def get_finger_joint_pos(env: Any) -> Tensor:
    """Get Robotiq 2F-85 drive joint position, shape (N,)."""
    robot = env.unwrapped.scene["robot"]
    j_ids, _ = robot.find_joints(CFG.c_stack_finger_joint_name)
    return robot.data.joint_pos[:, j_ids[0]]


# ---------------------------------------------------------------------------
# Cube state
# ---------------------------------------------------------------------------


def get_insertive_object_pose_in_base(env: Any) -> tuple[Tensor, Tensor]:
    """Get the graspable cube pose in robot base frame.

    Returns (pos_b, quat_b) each of shape (N, 3) and (N, 4).
    """
    obj = env.unwrapped.scene["insertive_object"]
    robot = env.unwrapped.scene["robot"]
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )


def get_receptive_object_pose_in_base(env: Any) -> tuple[Tensor, Tensor]:
    """Get the target (receptive) cube pose in robot base frame."""
    obj = env.unwrapped.scene["receptive_object"]
    robot = env.unwrapped.scene["robot"]
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )


def get_cube_pos_in_base(env: Any, cube_name: str) -> Tensor:
    """Return (N, 3) position of a cube in robot base frame.

    ``cube_name`` is the semantic name from our object container
    (``grasp_cube`` or ``target_cube``), which maps to the Isaac Lab scene
    entities ``insertive_object`` / ``receptive_object``.
    """
    if cube_name == "grasp_cube":
        pos_b, _ = get_insertive_object_pose_in_base(env)
    elif cube_name == "target_cube":
        pos_b, _ = get_receptive_object_pose_in_base(env)
    else:
        raise ValueError(f"Unknown cube name: {cube_name}")
    return pos_b


def extract_object_pose_in_base(env: Any, obj: Object) -> tuple[Tensor, Tensor]:
    """Dispatch an ``Object`` to its (pos_b, quat_b) pose in the base frame."""
    if obj.name == "robot":
        return get_ee_pose_in_base(env)
    if obj.name == "grasp_cube":
        return get_insertive_object_pose_in_base(env)
    if obj.name == "target_cube":
        return get_receptive_object_pose_in_base(env)
    raise ValueError(f"Unknown object: {obj.name}")


def extract_object_pos_in_base(env: Any, obj: Object) -> Tensor:
    """Position-only variant of :func:`extract_object_pose_in_base`."""
    pos_b, _ = extract_object_pose_in_base(env, obj)
    return pos_b


# ---------------------------------------------------------------------------
# Goal construction
# ---------------------------------------------------------------------------


def compute_goal_from_insertive_object(
    env: Any,
    z_offset: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Compute a goal pose above the insertive object.

    Goal position = insertive object position + ``z_offset`` above.
    Goal orientation = current EE orientation (so no rotation is needed
    during the approach).
    """
    obj_pos_b, _ = get_insertive_object_pose_in_base(env)
    _, ee_quat_b = get_ee_pose_in_base(env)

    goal_pos_b = obj_pos_b.clone()
    goal_pos_b[:, 2] += z_offset
    goal_quat_b = ee_quat_b.clone()
    return goal_pos_b, goal_quat_b


def goal_pose_base_to_world(
    env: Any,
    goal_pos_b: Tensor,
    goal_quat_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert a goal pose from robot base frame to world frame."""
    robot = env.unwrapped.scene["robot"]
    return combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        goal_pos_b,
        goal_quat_b,
    )


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------


def get_success(env: Any) -> Tensor:
    """Query per-env success from the ProgressContext reward term.

    Returns a bool tensor of shape (N,).
    """
    progress_ctx = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    return progress_ctx.success.clone()


# Robotiq 2F-85 finger-joint ↔ gap (metres) linear approximation.
# ``finger_joint = 0`` rad → 85 mm gap; ``finger_joint = 0.785398`` rad → 0 mm
# gap. The 4-bar linkage is not strictly linear but near the open end this
# approximation is within a few mm and is what we use to translate a
# user-friendly gap threshold into a joint-angle upper bound.
_ROBOTIQ_2F140_MAX_GAP_M: float = 0.085
_ROBOTIQ_2F140_CLOSE_RAD: float = 0.785398


def gripper_gap_to_finger_joint(min_gap_m: float) -> float:
    """Return the ``finger_joint`` upper bound that guarantees at least
    ``min_gap_m`` metres of opening on a Robotiq 2F-85 (linear model)."""
    min_gap_m = max(0.0, min(float(min_gap_m), _ROBOTIQ_2F140_MAX_GAP_M))
    return _ROBOTIQ_2F140_CLOSE_RAD * (1.0 - min_gap_m / _ROBOTIQ_2F140_MAX_GAP_M)


class InsertiveAssembledChecker:
    """Per-env trigger: insertive assembled in box AND gripper opened.

    Three conditions must all hold for ``__call__`` to fire:

    1. Alignment points of insertive / receptive coincide within
       ``pos_tol`` metres (same math as OmniReset's ``ProgressContext``,
       using each asset's ``assembled_offset`` from metadata).
    2. Insertive block is upright: ``|roll| + |pitch| < upright_tol``
       (world frame).
    3. Gripper is open with gap ≥ ``min_gripper_gap`` metres — i.e. the
       robot has released the block. Implemented as a linear cutoff on
       ``finger_joint`` position (see :func:`gripper_gap_to_finger_joint`).

    The two halves are also exposed individually as :meth:`assembled_only`
    (1+2) and :meth:`gripper_open` (3). The two-phase ``RLSkill`` uses
    them as separate transition triggers — phase 1 ends on
    ``assembled_only``; phase 2 (scripted gripper-open + arm-lift) ends
    on ``gripper_open`` plus a lift threshold.

    Decoupled from the reward graph so it works even though the env's
    ``progress_context`` reward term is the 3-block ``TowerGraspedSuccess``.
    """

    def __init__(
        self,
        env: Any,
        pos_tol: float = 0.04,
        upright_tol: float = 0.2,
        min_gripper_gap: float = 0.06,
        finger_joint_name: str = "finger_joint",
    ) -> None:
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import (
            read_metadata_from_usd_directory,
        )

        self._pos_tol = float(pos_tol)
        self._upright_tol = float(upright_tol)
        self._finger_joint_max = gripper_gap_to_finger_joint(min_gripper_gap)
        scene = env.unwrapped.scene
        ins_meta = read_metadata_from_usd_directory(
            scene["insertive_object"].cfg.spawn.usd_path
        )
        rec_meta = read_metadata_from_usd_directory(
            scene["receptive_object"].cfg.spawn.usd_path
        )
        device = env.unwrapped.device
        self._ins_off = torch.tensor(
            ins_meta["assembled_offset"]["pos"], device=device, dtype=torch.float32
        )
        self._rec_off = torch.tensor(
            rec_meta["assembled_offset"]["pos"], device=device, dtype=torch.float32
        )
        # Resolve finger-joint index once.
        robot = scene["robot"]
        joint_ids, _ = robot.find_joints(finger_joint_name)
        if not joint_ids:
            raise ValueError(
                f"Robot has no joint named {finger_joint_name!r} — "
                "cannot build InsertiveAssembledChecker."
            )
        self._finger_joint_id = int(joint_ids[0])

    @property
    def finger_joint_max(self) -> float:
        """``finger_joint`` upper bound (rad) that satisfies the gap requirement."""
        return self._finger_joint_max

    def assembled_only(self, env: Any) -> Tensor:
        """Per-env bool: insertive aligned in box AND upright. Gripper-state agnostic."""
        scene = env.unwrapped.scene
        ins = scene["insertive_object"]
        rec = scene["receptive_object"]
        ins_pos_w = ins.data.root_pos_w
        ins_quat_w = ins.data.root_quat_w
        rec_pos_w = rec.data.root_pos_w
        rec_quat_w = rec.data.root_quat_w
        n = ins_pos_w.shape[0]

        # Alignment points in world: asset_pos + R_asset @ assembled_offset.
        ins_align = ins_pos_w + quat_rotate(ins_quat_w, self._ins_off.expand(n, -1))
        rec_align = rec_pos_w + quat_rotate(rec_quat_w, self._rec_off.expand(n, -1))
        xyz_dist = (ins_align - rec_align).norm(dim=-1)

        # Upright insertive: roll + pitch in world frame near zero.
        ex, ey, _ = euler_xyz_from_quat(ins_quat_w)
        roll_pitch = wrap_to_pi(ex).abs() + wrap_to_pi(ey).abs()

        return (xyz_dist < self._pos_tol) & (roll_pitch < self._upright_tol)

    def gripper_open(self, env: Any) -> Tensor:
        """Per-env bool: ``finger_joint`` open past the configured min gap."""
        robot = env.unwrapped.scene["robot"]
        finger_pos = robot.data.joint_pos[:, self._finger_joint_id]
        return finger_pos < self._finger_joint_max

    def __call__(self, env: Any) -> Tensor:
        return self.assembled_only(env) & self.gripper_open(env)


# ---------------------------------------------------------------------------
# Observation layout (obs-only helpers)
# ---------------------------------------------------------------------------


def _axis_angle_to_quat(aa: Tensor) -> Tensor:
    """Convert an axis-angle tensor (…, 3) to a wxyz quaternion (…, 4)."""
    angle = aa.norm(dim=-1)
    axis = aa / angle.clamp(min=1e-8).unsqueeze(-1)
    return quat_from_angle_axis(angle, axis)


def _yaw_from_quat(q: Tensor) -> Tensor:
    """Extract the world-z yaw angle from a wxyz quat tensor (…, 4)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_quarter_turn(angle: Tensor) -> Tensor:
    """Wrap an angle difference to [-pi/4, pi/4] assuming 4-fold symmetry.

    Square-cross-section cubes are invariant under 90-degree yaw rotations,
    so we only need the gripper to align up to the nearest quarter turn.
    """
    import math

    period = math.pi / 2.0
    return torch.remainder(angle + period / 2.0, period) - period / 2.0


def yaw_aligned_target_quat_from_yaw(
    current_ee_quat_b: Tensor,
    target_yaw: Tensor,
) -> Tensor:
    """Rotate ``current_ee_quat_b`` about world-z to reach ``target_yaw``.

    Only the world-z yaw is adjusted — the EE's current roll/pitch are
    preserved. Because the returned target *inherits* the current rx/ry
    components, :func:`compute_pose_error` between current EE and this
    target yields an axis-angle error with pure rz, so the OSC action's
    slow rx/ry axes never activate.

    Cube rotational symmetry is quarter-turn (square cross-section), so
    the applied yaw delta is wrapped to ``[-pi/4, pi/4]`` for the shortest
    alignment.
    """
    ee_yaw = _yaw_from_quat(current_ee_quat_b)
    delta = _wrap_to_quarter_turn(target_yaw - ee_yaw)
    axis_z = torch.zeros(
        (current_ee_quat_b.shape[0], 3), device=current_ee_quat_b.device
    )
    axis_z[:, 2] = 1.0
    q_delta = quat_from_angle_axis(delta, axis_z)
    return quat_mul(q_delta, current_ee_quat_b)


def object_pitch_180_target_quat(object_quat_b: Tensor) -> Tensor:
    """Return ``object_quat_b`` followed by a local-frame 180 deg pitch."""
    import math

    zero = torch.zeros(
        object_quat_b.shape[:-1],
        dtype=object_quat_b.dtype,
        device=object_quat_b.device,
    )
    pitch = torch.full_like(zero, math.pi)
    q_delta = quat_from_euler_xyz(zero, pitch, zero)
    return quat_mul(object_quat_b, q_delta)


def yaw_aligned_target_quat(
    current_ee_quat_b: Tensor,
    cube_quat_b: Tensor,
) -> Tensor:
    """Convenience wrapper: align yaw to the cube quat's yaw component."""
    return yaw_aligned_target_quat_from_yaw(
        current_ee_quat_b, _yaw_from_quat(cube_quat_b)
    )


def yaw_from_quat(q: Tensor) -> Tensor:
    """Public alias for the world-z yaw of a wxyz quat."""
    return _yaw_from_quat(q)


def top_down_target_quat(gripper_quat_b: Tensor) -> Tensor:
    """Return a target quat with the frame's local ``+Z`` vertical down.

    Robotiq metadata defines the gripper approach direction as local ``+Z``.
    A top-down gripper pose therefore maps local ``+Z`` to base ``-Z``; yaw
    is copied from the current pose so the retreat does not spin about the
    vertical axis unnecessarily.
    """
    import math

    yaw = _yaw_from_quat(gripper_quat_b)
    yaw_use = torch.full_like(yaw, math.pi / 2)
    roll = torch.zeros_like(yaw)
    pitch = torch.full_like(yaw, -math.pi)
    return quat_from_euler_xyz(roll, pitch, yaw_use)


def top_down_wrist_target_quat(env: Any) -> Tensor:
    """Return the wrist target quat that makes the Robotiq gripper top-down.

    The OSC controls ``wrist_3_link``, while the physical gripper approach
    axis belongs to ``robotiq_base_link``. The fixed relative transform is
    recovered from the current articulation state so the wrist target can
    enforce top-down orientation on the gripper frame itself.
    """
    _, wrist_quat_b = get_ee_pose_in_base(env)
    _, gripper_quat_b = get_robot_body_pose_in_base(env, "robotiq_base_link")
    wrist_to_gripper = quat_mul(quat_inv(wrist_quat_b), gripper_quat_b)
    gripper_goal_quat_b = top_down_target_quat(gripper_quat_b)
    return quat_mul(gripper_goal_quat_b, quat_inv(wrist_to_gripper))


class CornerStackingObsLayout:
    """Column layout of the concatenated OmniReset policy observation.

    The UWLab PolicyCfg (``rl_state_cfg.ObservationsCfg.PolicyCfg``) stacks
    six terms with ``history_length=H`` and ``flatten_history_dim=True``;
    within each term block the history is laid out oldest→newest, so the
    most-recent sample is the last ``D`` columns of the block.

    NOTE on ordering: Isaac Lab's ``ObservationManager`` iterates terms in
    the order emitted by ``configclass`` introspection, which promotes
    *annotated* fields ahead of unannotated ones. In PolicyCfg only
    ``insertive_asset_in_receptive_asset_frame`` carries an explicit type
    annotation, so it is processed first. The cornered-stacking subclass
    appends an unannotated ``outside_block_pose`` term at the end. The
    runtime layout is::

        [ insertive_in_receptive (H*6) | prev_actions (H*7) |
          joint_pos (H*J) | ee_pose_b (H*6) | insertive_in_ee (H*6) |
          receptive_in_ee (H*6) | outside_in_ee (H*6)? ]

    The trailing ``outside_in_ee`` block is present only when
    ``has_outside=True`` (the 3-object cornered-stacking task); the 2-object
    tasks use ``has_outside=False`` and the layout stops at ``receptive_in_ee``.

    All helpers return tensors built from the *latest* history slice only;
    the older entries are ignored since predicates and scripted skills are
    memoryless.
    """

    # UWLab OmniReset constants (rl_state_cfg.py)
    ACTION_DIM: int = 7
    POSE_DIM: int = 6  # 3 pos + 3 axis-angle
    HISTORY: int = 5

    def __init__(
        self,
        joint_dim: int,
        finger_joint_idx: int,
        has_outside: bool = False,
    ) -> None:
        self.joint_dim = joint_dim
        self.finger_joint_idx = finger_joint_idx
        self.has_outside = has_outside

        H = self.HISTORY
        A = self.ACTION_DIM
        J = joint_dim
        P = self.POSE_DIM

        # Offsets to the start of each term block (annotated field first).
        ins_in_rec_start = 0
        act_start = ins_in_rec_start + H * P
        joint_start = act_start + H * A
        ee_start = joint_start + H * J
        ins_in_ee_start = ee_start + H * P
        rec_in_ee_start = ins_in_ee_start + H * P
        out_in_ee_start = rec_in_ee_start + H * P

        self._ins_in_rec_latest = (
            ins_in_rec_start + (H - 1) * P,
            ins_in_rec_start + H * P,
        )
        self._act_latest = (act_start + (H - 1) * A, act_start + H * A)
        self._joint_latest = (joint_start + (H - 1) * J, joint_start + H * J)
        self._ee_latest = (ee_start + (H - 1) * P, ee_start + H * P)
        self._ins_in_ee_latest = (
            ins_in_ee_start + (H - 1) * P,
            ins_in_ee_start + H * P,
        )
        self._rec_in_ee_latest = (
            rec_in_ee_start + (H - 1) * P,
            rec_in_ee_start + H * P,
        )
        if has_outside:
            self._out_in_ee_latest: tuple[int, int] | None = (
                out_in_ee_start + (H - 1) * P,
                out_in_ee_start + H * P,
            )
            total = out_in_ee_start + H * P
        else:
            self._out_in_ee_latest = None
            total = rec_in_ee_start + H * P
        self.expected_obs_dim = total

    @classmethod
    def from_env(
        cls, env: Any, has_outside: bool | None = None
    ) -> "CornerStackingObsLayout":
        """Resolve joint layout once from a live articulation.

        The finger joint index is discovered via ``find_joints`` on the
        Robotiq drive joint configured in :data:`CFG.c_stack_finger_joint_name`.
        When ``has_outside`` is left as ``None``, the presence of an
        ``outside_block`` scene entity decides whether to reserve the
        trailing ``outside_in_ee`` obs slice.
        """
        robot = env.unwrapped.scene["robot"]
        joint_dim = int(robot.data.joint_pos.shape[1])
        j_ids, _ = robot.find_joints(CFG.c_stack_finger_joint_name)
        if has_outside is None:
            has_outside = "outside_block" in env.unwrapped.scene.keys()
        return cls(
            joint_dim=joint_dim,
            finger_joint_idx=int(j_ids[0]),
            has_outside=bool(has_outside),
        )

    # ------------------------------------------------------------------
    # Raw slices
    # ------------------------------------------------------------------

    def latest_prev_action(self, obs: Tensor) -> Tensor:
        lo, hi = self._act_latest
        return obs[:, lo:hi]

    def latest_joint_pos(self, obs: Tensor) -> Tensor:
        lo, hi = self._joint_latest
        return obs[:, lo:hi]

    def latest_finger_pos(self, obs: Tensor) -> Tensor:
        """Robotiq 2F-85 drive-joint position, shape (N,)."""
        joints = self.latest_joint_pos(obs)
        return joints[:, self.finger_joint_idx]

    def latest_ee_pose_in_base(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """EE pose in robot base frame as (pos, axis_angle), each (N, 3)."""
        lo, hi = self._ee_latest
        block = obs[:, lo:hi]
        return block[:, :3], block[:, 3:]

    def latest_insertive_pose_in_ee(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Grasp (insertive) cube pose in wrist_3 frame."""
        lo, hi = self._ins_in_ee_latest
        block = obs[:, lo:hi]
        return block[:, :3], block[:, 3:]

    def latest_receptive_pose_in_ee(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Target (receptive) cube pose in wrist_3 frame."""
        lo, hi = self._rec_in_ee_latest
        block = obs[:, lo:hi]
        return block[:, :3], block[:, 3:]

    def latest_insertive_pose_in_receptive(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Grasp cube pose in receptive cube's frame."""
        lo, hi = self._ins_in_rec_latest
        block = obs[:, lo:hi]
        return block[:, :3], block[:, 3:]

    def latest_outside_pose_in_ee(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Outside-block pose in wrist_3 frame (only for 3-object tasks)."""
        if self._out_in_ee_latest is None:
            raise RuntimeError(
                "Obs layout has no outside_block slice — built with has_outside=False."
            )
        lo, hi = self._out_in_ee_latest
        block = obs[:, lo:hi]
        return block[:, :3], block[:, 3:]

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def latest_ee_quat_in_base(self, obs: Tensor) -> Tensor:
        """EE quaternion (wxyz) in robot base frame, shape (N, 4)."""
        _, aa = self.latest_ee_pose_in_base(obs)
        return _axis_angle_to_quat(aa)

    def _pose_in_ee_to_base(self, obs: Tensor, pos_in_ee: Tensor) -> Tensor:
        """Compose an EE-frame position back to base frame via EE pose."""
        ee_pos_b, _ = self.latest_ee_pose_in_base(obs)
        ee_quat_b = self.latest_ee_quat_in_base(obs)
        R_b_ee = matrix_from_quat(ee_quat_b)  # (N, 3, 3)
        rotated = torch.bmm(R_b_ee, pos_in_ee.unsqueeze(-1)).squeeze(-1)
        return ee_pos_b + rotated

    def latest_insertive_pos_in_base(self, obs: Tensor) -> Tensor:
        """Grasp cube position in robot base frame."""
        pos_in_ee, _ = self.latest_insertive_pose_in_ee(obs)
        return self._pose_in_ee_to_base(obs, pos_in_ee)

    def latest_receptive_pos_in_base(self, obs: Tensor) -> Tensor:
        """Target cube position in robot base frame."""
        pos_in_ee, _ = self.latest_receptive_pose_in_ee(obs)
        return self._pose_in_ee_to_base(obs, pos_in_ee)

    def latest_outside_pos_in_base(self, obs: Tensor) -> Tensor:
        """Outside-block position in robot base frame (3-object tasks only)."""
        pos_in_ee, _ = self.latest_outside_pose_in_ee(obs)
        return self._pose_in_ee_to_base(obs, pos_in_ee)

    def latest_block_pos_in_base(self, obs: Tensor, block_name: str) -> Tensor:
        """Dispatch block position by semantic name (``block0`` / ``block1``).

        ``block0`` is the insertive block (the one that starts in the box);
        ``block1`` is the outside block. The receptive_object (the box) is
        addressed via :meth:`latest_receptive_pos_in_base` instead.
        """
        if block_name == "block0":
            return self.latest_insertive_pos_in_base(obs)
        if block_name == "block1":
            return self.latest_outside_pos_in_base(obs)
        if block_name == "box":
            return self.latest_receptive_pos_in_base(obs)
        raise ValueError(f"Unknown block/box name: {block_name}")

    def latest_insertive_quat_in_base(self, obs: Tensor) -> Tensor:
        """Grasp-cube orientation (wxyz quat) in robot base frame, (N, 4)."""
        _, ins_aa_ee = self.latest_insertive_pose_in_ee(obs)
        cube_quat_ee = _axis_angle_to_quat(ins_aa_ee)
        ee_quat_b = self.latest_ee_quat_in_base(obs)
        return quat_mul(ee_quat_b, cube_quat_ee)

    def latest_receptive_quat_in_base(self, obs: Tensor) -> Tensor:
        """Target-cube orientation (wxyz quat) in robot base frame, (N, 4)."""
        _, rec_aa_ee = self.latest_receptive_pose_in_ee(obs)
        cube_quat_ee = _axis_angle_to_quat(rec_aa_ee)
        ee_quat_b = self.latest_ee_quat_in_base(obs)
        return quat_mul(ee_quat_b, cube_quat_ee)

    def latest_outside_quat_in_base(self, obs: Tensor) -> Tensor:
        """Outside-block orientation (wxyz quat) in robot base frame, (N, 4)."""
        _, out_aa_ee = self.latest_outside_pose_in_ee(obs)
        cube_quat_ee = _axis_angle_to_quat(out_aa_ee)
        ee_quat_b = self.latest_ee_quat_in_base(obs)
        return quat_mul(ee_quat_b, cube_quat_ee)

    def latest_block_quat_in_base(self, obs: Tensor, block_name: str) -> Tensor:
        """Dispatch block orientation by semantic name (``block0`` / ``block1``)."""
        if block_name == "block0":
            return self.latest_insertive_quat_in_base(obs)
        if block_name == "block1":
            return self.latest_outside_quat_in_base(obs)
        if block_name == "box":
            return self.latest_receptive_quat_in_base(obs)
        raise ValueError(f"Unknown block/box name: {block_name}")
