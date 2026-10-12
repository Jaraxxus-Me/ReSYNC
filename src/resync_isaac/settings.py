"""Demo-only defaults, installed into CFG only by the demo process."""

from typing import Any

import numpy as np

from resync_isaac.paths import ASSET_ROOT


class IsaacDefaults:
    """Configuration defaults for the optional Isaac simulation demos."""

    # ******* Corner Stacking (Isaac Sim) Settings ********
    c_stack_num_envs: int = 1
    c_stack_max_steps: int = 400
    c_stack_goal_z_offset: float = (
        0.1  # legacy default (still used by the scripted Place skill)
    )
    # PickSkill heights (base-frame z above the grasp cube top)
    c_stack_pre_grasp_z_offset: float = 0.4  # approach/lift height
    c_stack_grasp_z_offset: float = 0.25  # descent height before closing
    # Max controller steps allowed in one scripted Pick/Place phase before
    # forcing the skill to advance to the next phase. Set <=0 to disable.
    c_stack_scripted_max_phase_steps: int = 80
    # Per-skill convergence thresholds (used by scripted Pick/Place in
    # ``corner_stacking.py`` to decide when a scripted skill has reached its
    # goal pose). Not used to gate RL → scripted handoff in the hybrid test
    # — that uses ``InsertiveAssembledChecker`` instead.
    c_stack_pos_threshold: float = 0.011
    c_stack_rot_threshold: float = 0.08
    c_stack_checkpoint: str = str(ASSET_ROOT / "corner_stacking/policy.pt")

    # Corner Stacking predicate thresholds (in robot base frame)
    c_stack_on_xy_thresh: float = 0.03  # On: xy distance between cubes
    c_stack_place_dz: float = 0.26  # Place: hand z above target.z for placing
    c_stack_on_z_min: float = 0.03  # On: grasp_cube.z above target.z min
    c_stack_on_z_max: float = 0.06  # On: grasp_cube.z above target.z max
    c_stack_gripper_closed_min: float = 0.15  # Holding: finger_joint lower bound
    c_stack_gripper_closed_max: float = 0.75  # Holding: finger_joint upper bound
    c_stack_holding_ee_dist_xy: float = 0.01  # Holding: ee-to-cube distance tol
    c_stack_holding_ee_dist_z: float = 0.3  # Holding: ee above cube z tol
    c_stack_finger_joint_name: str = "finger_joint"  # Robotiq 2F-85 drive joint

    # Corner Stacking block-reset ranges (applied on top of default joint reset)
    # Ranges are additive deltas on each cube's default root pose (base frame).
    c_stack_reset_cube_x_range: tuple = (0.3, 0.6)
    c_stack_reset_cube_y_range: tuple = (-0.3, 0.3)
    c_stack_reset_cube_yaw_range: tuple = (-1.57, 1.57)

    # ******* Cornered Stacking (box + grasp block + outside block) *******
    # At reset: box is sampled by the original c_stack_reset_cube_* range;
    # one block starts "in" the box at the USD assembled_offset; the other
    # starts outside at box.pos + (dx, dy, dz) (world frame) with random yaw.
    # CFG only decides which role each plays in the goal:
    #   "grasp" → in-box block is the base (bottom, gripped at success)
    #   "stack" → in-box block is the top (stacked on the other block)
    cstack_in_box_role: str = "grasp"
    # Outside-block distribution, relative to the box's sampled world pose.
    cstack_outside_dx_range: tuple = (-0.1, 0.1)
    cstack_outside_dy_range: tuple = (0.3, 0.4)
    cstack_outside_dz_range: tuple = (0.0, 0.0)
    cstack_outside_yaw_range: tuple = (-1.57, 1.57)
    # Sparse-success thresholds (all in world / base-frame metres or radians)
    cstack_stack_xy_tol: float = 0.02
    cstack_stack_z_offset: float = 0.04  # expected (top.z − base.z) when stacked
    cstack_stack_z_tol: float = 0.02
    cstack_stack_yaw_tol: float = 0.3
    cstack_upright_tol: float = 0.25  # |roll| + |pitch| (rad)
    cstack_grasp_ee_xy_tol: float = 0.03
    cstack_grasp_ee_z_tol: float = 0.05
    cstack_lift_world_z_min: float = 0.04  # base-block world z ≥ this ⇒ lifted
    cstack_finger_closed_min: float = 0.4
    # Trigger for handing off RL → scripted retreat: insertive is assembled
    # in the box (position + upright) AND the gripper has opened to at least
    # ``cstack_assembled_min_gripper_gap`` metres. The checker converts the
    # gap threshold to a ``finger_joint`` upper bound using the 2F-85 linear
    # approximation (finger_joint=0 → 85 mm, finger_joint=0.785 rad → 0 mm).
    cstack_assembled_pos_tol: float = 0.04
    cstack_assembled_upright_tol: float = 0.2
    cstack_assembled_min_gripper_gap: float = 0.06  # ≈ 1.5 × block size
    # Two-phase ``RLSkill``: how far above the phase-2 entry EE-z (world
    # frame) the gripper must lift before the skill terminates (m).
    cstack_retreat_lift_z_min: float = 0.10
    cstack_retreat_lift_z_tol: float = 0.002
    # Max arm-joint speed (rad/s) before RL retreat can terminate. LIFT keeps
    # tracking its static top-down goal until this is satisfied.
    cstack_retreat_lift_joint_vel_max: float = 0.05
    # Task-planning goal selector for the cornered-stacking TAMP system.
    # True  → goal = {On(block0, block1)}  (block0 is the in-box block)
    # False → goal = {On(block1, block0)}
    c_stack_goal_0on1: bool = False

    # Demo reset states always start before recovery; no training-data path.
    c_stack_reset_from_post_rl: bool = False

    # Two, three, or four 6 cm ReSYNC cubes, ordered top to bottom.
    num_of_blocks: int = 4
    # A comma-separated pair or random_pair selects exactly two inverted cubes.
    fstack_flipped_cube: str = "random_pair"
    fstack_num_envs: int = 1
    fstack_max_steps: int = 3000
    fstack_checkpoint: str = str(ASSET_ROOT / "flipped_stacking/policy.pt")
    fstack_cube_x_range: tuple = (-0.65, -0.35)
    fstack_cube_y_range: tuple = (-0.20, 0.20)
    fstack_cube_center_z: float = 0.017
    fstack_cube_min_xy_dist: float = 0.10
    fstack_flipped_roll_range: tuple = (np.pi, np.pi)
    fstack_flipped_pitch_range: tuple = (0.0, 0.0)
    fstack_cube_yaw_range: tuple = (-np.pi, np.pi)
    fstack_home_joints: tuple = (
        -0.129152,
        -1.689962,
        -1.674141,
        -1.362756,
        1.558274,
        -1.728060,
    )
    fstack_home_joint_noise: float = 0.10
    # ReSYNC ObjectAnywhereEEAnywhere initiation set.
    fstack_move_pregrasp_height: float = 0.30
    fstack_move_pregrasp_xy_tol: float = 0.05
    fstack_move_pregrasp_height_range: tuple = (0.25, 0.35)
    fstack_move_pregrasp_angle_tol: float = np.pi / 12
    # Virtual ReSYNC region sampled once when Flip starts and held fixed.
    fstack_flip_region_radius_range: tuple = (0.08, 0.12)
    fstack_flip_region_other_cube_min_xy_dist: float = 0.10
    fstack_flip_region_yaw_range: tuple = (11 * np.pi / 12, 13 * np.pi / 12)
    fstack_flip_region_sample_attempts: int = 256
    fstack_flip_position_tol: float = 0.005
    fstack_flip_upright_tol: float = 0.025
    fstack_flip_open_finger_max: float = 0.25
    fstack_flip_retreat_height: float = 0.30
    fstack_flip_max_rl_steps: int = 400
    fstack_scripted_max_phase_steps: int = 100
    # Scripted Pick/Place wrist heights relative to cube center. These include
    # the fixed wrist-to-gripper offset of the UR5e + Robotiq assembly.
    fstack_pick_pregrasp_dz: float = 0.30
    fstack_pick_grasp_dz: float = 0.245
    fstack_pick_lift_dz: float = 0.37
    fstack_pick_lift_clearance: float = 0.09
    fstack_pick_grasp_xy_tol: float = 0.008
    fstack_pick_grasp_z_tol: float = 0.010
    fstack_pick_max_retries: int = 4
    fstack_pick_retry_lift_steps: int = 60
    # (dx, dy, dz) search offsets around the live cube center.
    fstack_pick_retry_offsets: tuple = (
        (0.0, 0.0, 0.0),
        (-0.01, 0.0, -0.01),
        (0.01, 0.0, -0.01),
        (0.0, -0.01, -0.01),
        (0.0, 0.01, -0.01),
    )
    # Absolute base-frame wrist height used for the vertical lift, lateral
    # transit, and post-place retract.  Relative tower clearance can raise it
    # further, but Place never transits below this height.
    fstack_place_fixed_safe_ee_z: float = 0.55
    fstack_place_pre_dz: float = 0.36
    # Required free space between the support's top face and the held cube's
    # bottom face during the lateral pre-place approach.
    fstack_place_pre_clearance: float = 0.025
    fstack_place_dz: float = 0.305
    fstack_place_release_clearance: float = 0.020
    # Slow the scripted Place controller after it enters the tower's local
    # neighborhood.  This reduces lateral/vertical impulses on partial stacks
    # without slowing the long free-space approach.
    fstack_place_near_tower_xy_dist: float = 0.10
    fstack_place_near_tower_motion_scale: float = 0.50
    fstack_place_xy_tol: float = 0.004
    fstack_place_z_tol: float = 0.003
    fstack_place_pre_yaw_tol: float = 0.10
    fstack_place_yaw_tol: float = 0.08
    fstack_place_open_dwell_steps: int = 20
    fstack_place_max_phase_steps: int = 400
    fstack_scripted_pos_tol: float = 0.020
    fstack_scripted_rot_tol: float = 0.08
    fstack_gripper_dwell_steps: int = 40
    # Predicates/final task success.
    fstack_upright_tol: float = 0.10
    fstack_on_xy_tol: float = 0.025
    fstack_on_z_offset: float = 0.06
    fstack_on_z_tol: float = 0.012
    fstack_on_yaw_tol: float = 0.20
    fstack_bottom_contact_min_force: float = 0.003
    fstack_other_contact_max_force: float = 0.002
    fstack_holding_xy_tol: float = 0.025
    fstack_holding_z_tol: float = 0.30
    fstack_closed_finger_min: float = 0.15
    fstack_closed_finger_max: float = 0.78


def configure(**overrides: Any) -> None:
    from skill_refactor.args import reset_config
    from skill_refactor.settings import CFG

    for name, value in vars(IsaacDefaults).items():
        if not name.startswith("_"):
            setattr(CFG, name, value)
    reset_config(overrides)
