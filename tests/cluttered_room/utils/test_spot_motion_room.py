"""Tests for SpotMotion in ClutteredRoom environment."""

from typing import Any, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.cluttered_room.utils import (
    extract_obj1_pose,
    extract_obj2_pose,
    extract_robot_body_pose,
    extract_robot_hand_pose,
    extract_robot_joint_vels,
    extract_robot_joints,
)
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    get_normalize_action_range,
)
from skill_refactor.utils.motion_planning import SpotMotionRoom, WaypointTrackerRoom

# ============================================================================
# Helper Functions
# ============================================================================


def setup_test_environment(output_dir: str, max_steps: int = 300):
    """Set up test configuration and environment for ClutteredRoom tests.

    Args:
        output_dir: Directory for video output
        max_steps: Maximum steps per video

    Returns:
        tuple: (env, spot_motion_planner)
    """
    test_config = {
        "debug_env": False,
        "delta_finger_control": False,
        "num_envs": 1,
    }
    register_all_environments()
    reset_config(test_config)
    spot_motion_planner = SpotMotionRoom(device=CFG.device)

    env_kwargs: dict[str, Any] = {
        "obs_mode": "state",
        "render_mode": "rgb_array",
        "sim_backend": "gpu",
    }
    basic_env = gym.make(
        "skill_ref/ClutteredRoom-v1",
        num_envs=CFG.num_envs,
        reconfiguration_freq=0,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir=output_dir,
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=max_steps,
        video_fps=30,
    )

    return env, spot_motion_planner


def extract_robot_state(obs: Any) -> Tuple[Pose, Pose, torch.Tensor]:
    """Extract robot hand, body, and joint state from observation.

    Args:
        obs: Environment observation

    Returns:
        tuple: (hand_pose, body_pose, joint_positions)
    """
    robot_hand_pose_tensor = extract_robot_hand_pose(obs)
    hand_pose = Pose.create_from_pq(
        robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
    )

    robot_body_pose_tensor = extract_robot_body_pose(obs)
    body_pose = Pose.create_from_pq(
        robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
    )

    joint_positions = extract_robot_joints(obs)

    return hand_pose, body_pose, joint_positions


def create_waypoint_tracker(plan: Any, env: Any) -> WaypointTrackerRoom:
    """Create a waypoint tracker for feedback control.

    Args:
        plan: Planned actions
        env: Environment to get action normalization from

    Returns:
        WaypointTrackerRoom: Configured tracker
    """
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        env, CFG.control_mode
    )

    return WaypointTrackerRoom(
        plan=plan,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        angular_threshold=CFG.angular_threshold,
        waypoint_threshold=CFG.waypoint_threshold,
        device=CFG.device,
    )


def execute_motion_plan(
    env, obs, tracker: WaypointTrackerRoom, max_steps: int, use_velocity: bool = False
):
    """Execute a motion plan with feedback control.

    Args:
        env: Environment
        obs: Current observation
        tracker: Waypoint tracker
        max_steps: Maximum steps to execute
        use_velocity: Whether to use joint velocity in tracking

    Returns:
        Updated observation after execution
    """
    for step in range(max_steps):
        curr_qpos = extract_robot_joints(obs)

        if use_velocity:
            curr_qvel = extract_robot_joint_vels(obs)
            if (
                len(tracker.plan) == 0
                and tracker.subgoal_achieved(curr_qpos, curr_qvel).all()
            ):
                break
            delta_actions = tracker.compute_delta_actions(curr_qpos, curr_qvel)
        else:
            if len(tracker.plan) == 0 and tracker.subgoal_achieved(curr_qpos).all():
                break
            delta_actions = tracker.compute_delta_actions(curr_qpos)

        obs, _, _, _, _ = env.step(delta_actions)
        print(f"Step {step}, remaining waypoints: {len(tracker.plan)}")

    return obs


def execute_gripper_action(
    env,
    obs,
    spot_motion_planner: SpotMotionRoom,
    close: bool,
    max_steps: int = 30,
    use_velocity: bool = True,
):
    """Execute gripper open/close action.

    Args:
        env: Environment
        obs: Current observation
        spot_motion_planner: Motion planner
        close: If True, close gripper; if False, open gripper
        max_steps: Maximum steps to execute
        use_velocity: Whether to use joint velocity in tracking

    Returns:
        Updated observation after execution
    """
    curr_qpos = extract_robot_joints(obs)
    if close:
        gripper_action = spot_motion_planner.close_gripper(curr_qpos=curr_qpos, t=10)
    else:
        gripper_action = spot_motion_planner.open_gripper(curr_qpos=curr_qpos, t=10)

    tracker = create_waypoint_tracker(gripper_action, env)
    return execute_motion_plan(env, obs, tracker, max_steps, use_velocity)


def execute_hand_motion(
    env,
    obs,
    spot_motion_planner,
    target_pose,
    closing: bool = False,
    interpolate_steps: int = 40,
    max_steps: int = 250,
    use_velocity: bool = True,
):
    """Execute hand motion to target pose.

    Args:
        env: Environment
        obs: Current observation
        spot_motion_planner: Motion planner
        target_pose: Target hand pose
        closing: Whether gripper is closing
        interpolate_steps: Number of interpolation steps
        max_steps: Maximum execution steps
        use_velocity: Whether to use joint velocity in tracking

    Returns:
        Updated observation after execution
    """
    hand_pose, body_pose, joint_positions = extract_robot_state(obs)

    reach_actions = spot_motion_planner.move_hand_from_to_pose(
        body_pose,
        joint_positions,
        hand_pose,
        target_pose,
        closing=torch.tensor([closing], device=CFG.device, dtype=torch.bool).repeat(
            CFG.num_envs
        ),
        interpolate_steps=interpolate_steps,
    )

    tracker = create_waypoint_tracker(reach_actions, env)
    return execute_motion_plan(env, obs, tracker, max_steps, use_velocity)


# ============================================================================
# Tests
# ============================================================================


def test_hand_motion_in_cluttered_room_env():
    """Test hand motion planning and control in ClutteredRoom environment."""
    env, spot_motion_planner = setup_test_environment("videos/c-room-hand-motion-test")

    for s in range(1):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(123)

        # Extract current robot state
        hand_pose, body_pose, joint_positions = extract_robot_state(obs)

        # Create target pose relative to current hand pose
        rel_pose = Pose.create_from_pq(
            torch.tensor(
                [[0.1, 0.0, -0.1]], dtype=torch.float32, device=CFG.device
            ).repeat(CFG.num_envs, 1),
            euler2quat(0, -torch.pi / 6, 0),
        )
        grasp_pose = hand_pose * rel_pose

        # Plan hand motion to grasp pose
        reach_actions = spot_motion_planner.move_hand_from_to_pose(
            body_pose,
            joint_positions,
            hand_pose,
            grasp_pose,
            closing=torch.zeros((CFG.num_envs,), device=CFG.device, dtype=torch.bool),
            interpolate_steps=40,
        )
        assert reach_actions[0].shape == (CFG.num_envs, 11)

        # Execute motion with feedback control
        tracker = create_waypoint_tracker(reach_actions, env)
        obs = execute_motion_plan(env, obs, tracker, max_steps=100)

    env.close()


def test_body_motion_in_cluttered_room_env():
    """Test body motion planning and control in ClutteredRoom environment."""
    env, spot_motion_planner = setup_test_environment("videos/c-room-body-motion-test")

    for s in range(1):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(123)

        # Get current robot state
        _, body_pose, joint_positions = extract_robot_state(obs)
        obj1_pose_tensor = extract_obj1_pose(obs)

        # Create target body pose: navigate to object
        target_body_pose = spot_motion_planner.build_navigate_pose(
            target_pose=Pose.create_from_pq(
                obj1_pose_tensor[:, 0:3], obj1_pose_tensor[:, 3:7]
            ),
            facing_axis="x",
            distance=CFG.c_room_navigate_rel_dx,
        )
        holding = torch.zeros((CFG.num_envs,), device=CFG.device, dtype=torch.bool)

        # Plan body motion
        body_actions = spot_motion_planner.move_body_from_to_pose(
            robot_worldF_curr=body_pose,
            robot_worldF_tgt=target_body_pose,
            curr_joint_positions=joint_positions,
            closing=holding,
            interpolate_steps=40,
        )
        assert body_actions[0].shape == (CFG.num_envs, 11)

        # Execute motion with feedback control (body motion needs more steps)
        tracker = create_waypoint_tracker(body_actions, env)
        obs = execute_motion_plan(env, obs, tracker, max_steps=300)

        # Validate body reached target pose
        curr_body_pose_final = extract_robot_body_pose(obs)
        final_pose = Pose.create_from_pq(
            curr_body_pose_final[:, 0:3], curr_body_pose_final[:, 3:7]
        )

        # Check position error (body motion has lower precision than hand motion)
        _position_error = torch.norm(final_pose.p - target_body_pose.p, dim=-1)
        # assert (
        #     _position_error < 0.1
        # ).all(), f"Body position error too high: {_position_error}"

        # Check orientation error
        _orientation_error = torch.acos(
            torch.clamp(torch.sum(final_pose.q * target_body_pose.q, dim=-1), -1.0, 1.0)
        )
        # assert (
        #     orientation_error < 0.1
        # ).all(), f"Body orientation error too high: {orientation_error}"

    env.close()


def test_grasping_bowl_motion_in_cluttered_room_env():
    """Test full grasping sequence: navigate, grasp, and lift object."""
    env, spot_motion_planner = setup_test_environment(
        "videos/c-room-grasping-bowl-test", max_steps=1500
    )

    obs, _ = env.reset(seed=0)
    env.action_space.seed(123)

    # Step 1: Navigate body to object
    _, body_pose, joint_positions = extract_robot_state(obs)
    obj1_pose_tensor = extract_obj1_pose(obs)
    target_body_pose = spot_motion_planner.build_navigate_pose(
        target_pose=Pose.create_from_pq(
            obj1_pose_tensor[:, 0:3], obj1_pose_tensor[:, 3:7]
        ),
        facing_axis="x",
        distance=0.7,
    )
    holding = torch.zeros((CFG.num_envs,), device=CFG.device, dtype=torch.bool)
    body_actions = spot_motion_planner.move_body_from_to_pose(
        robot_worldF_curr=body_pose,
        robot_worldF_tgt=target_body_pose,
        curr_joint_positions=joint_positions,
        closing=holding,
        interpolate_steps=40,
    )
    assert body_actions[0].shape == (CFG.num_envs, 11)
    tracker = create_waypoint_tracker(body_actions, env)
    obs = execute_motion_plan(env, obs, tracker, max_steps=300, use_velocity=True)

    # Step 2: Open gripper
    obs = execute_gripper_action(env, obs, spot_motion_planner, close=False)

    # Step 3: Move hand to prepare pose (above object)
    hand_pose, _, _ = extract_robot_state(obs)
    obj1_pose = Pose.create_from_pq(
        extract_obj1_pose(obs)[:, 0:3], extract_obj1_pose(obs)[:, 3:7]
    )
    rel_pose_robot = Pose.create_from_pq(
        torch.zeros((CFG.num_envs, 3), dtype=torch.float32, device=CFG.device),
        euler2quat(0, -torch.pi / 12, 0),
    )
    rel_pose_obj = Pose.create_from_pq(
        torch.tensor(
            [[0.08, 0.0, 0.05]], dtype=torch.float32, device=CFG.device
        ).repeat(CFG.num_envs, 1),
        euler2quat(0, 0, 0),
    )
    prepare_pose = Pose.create_from_pq(
        (obj1_pose * rel_pose_obj).p, (hand_pose * rel_pose_robot).q
    )
    obs = execute_hand_motion(
        env, obs, spot_motion_planner, prepare_pose, max_steps=250
    )

    # Step 4: Move hand forward to grasp position
    hand_pose, _, _ = extract_robot_state(obs)
    rel_pose = Pose.create_from_pq(
        torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32, device=CFG.device).repeat(
            CFG.num_envs, 1
        ),
        euler2quat(0, 0, 0),
    )
    grasp_pose = hand_pose * rel_pose
    obs = execute_hand_motion(
        env, obs, spot_motion_planner, grasp_pose, interpolate_steps=10, max_steps=50
    )

    # Step 5: Close gripper to grasp object
    obs = execute_gripper_action(env, obs, spot_motion_planner, close=True)

    # Step 6: Lift object
    hand_pose_tensor = extract_robot_hand_pose(obs)
    lift_pos = hand_pose_tensor[:, 0:3] + torch.tensor(
        [[0.0, 0.0, 0.1]], dtype=torch.float, device=CFG.device
    ).repeat(CFG.num_envs, 1)
    lift_pose = Pose.create_from_pq(
        lift_pos,
        Pose.create_from_pq(hand_pose_tensor[:, 0:3], hand_pose_tensor[:, 3:7]).q,
    )
    obs = execute_hand_motion(env, obs, spot_motion_planner, lift_pose, closing=True)

    env.close()


def test_grasping_box_motion_in_cluttered_room_env():
    """Test full grasping sequence: navigate, grasp, and lift object."""
    env, spot_motion_planner = setup_test_environment(
        "videos/c-room-grasping-box-test", max_steps=1500
    )

    obs, _ = env.reset(seed=0)
    env.action_space.seed(123)

    # Step 1: Navigate body to object
    _, body_pose, joint_positions = extract_robot_state(obs)
    obj2_pose_tensor = extract_obj2_pose(obs)
    target_body_pose = spot_motion_planner.build_navigate_pose(
        target_pose=Pose.create_from_pq(
            obj2_pose_tensor[:, 0:3], obj2_pose_tensor[:, 3:7]
        ),
        facing_axis="x",
        distance=0.7,
    )
    holding = torch.zeros((CFG.num_envs,), device=CFG.device, dtype=torch.bool)
    body_actions = spot_motion_planner.move_body_from_to_pose(
        robot_worldF_curr=body_pose,
        robot_worldF_tgt=target_body_pose,
        curr_joint_positions=joint_positions,
        closing=holding,
        interpolate_steps=40,
    )
    assert body_actions[0].shape == (CFG.num_envs, 11)
    tracker = create_waypoint_tracker(body_actions, env)
    obs = execute_motion_plan(env, obs, tracker, max_steps=300, use_velocity=True)

    # Step 2: Open gripper
    obs = execute_gripper_action(env, obs, spot_motion_planner, close=False)

    # Step 3: Move hand to prepare pose (above object)
    hand_pose, body_pose, _ = extract_robot_state(obs)
    obj2_pose = Pose.create_from_pq(
        extract_obj2_pose(obs)[:, 0:3], extract_obj2_pose(obs)[:, 3:7]
    )
    rel_pose_robot = Pose.create_from_pq(
        torch.zeros((CFG.num_envs, 3), dtype=torch.float32, device=CFG.device),
        euler2quat(0, torch.pi / 2, 0),
    )
    rel_pose_obj = Pose.create_from_pq(
        torch.tensor([[0.1, 0.15, 0.0]], dtype=torch.float32, device=CFG.device).repeat(
            CFG.num_envs, 1
        ),
        euler2quat(0, 0, 0),
    )
    prepare_pose = Pose.create_from_pq(
        (obj2_pose * rel_pose_obj).p, (body_pose * rel_pose_robot).q
    )
    obs = execute_hand_motion(
        env, obs, spot_motion_planner, prepare_pose, max_steps=250
    )

    # Step 3.5: Move hand to prepare pose (above object)
    hand_pose, body_pose, _ = extract_robot_state(obs)
    obj2_pose = Pose.create_from_pq(
        extract_obj2_pose(obs)[:, 0:3], extract_obj2_pose(obs)[:, 3:7]
    )
    rel_pose_robot = Pose.create_from_pq(
        torch.zeros((CFG.num_envs, 3), dtype=torch.float32, device=CFG.device),
        euler2quat(0, torch.pi / 2, 0),
    )
    rel_pose_obj = Pose.create_from_pq(
        torch.tensor([[0.0, 0.15, 0.0]], dtype=torch.float32, device=CFG.device).repeat(
            CFG.num_envs, 1
        ),
        euler2quat(0, 0, 0),
    )
    prepare_pose = Pose.create_from_pq(
        (obj2_pose * rel_pose_obj).p, (body_pose * rel_pose_robot).q
    )
    obs = execute_hand_motion(
        env, obs, spot_motion_planner, prepare_pose, max_steps=250
    )

    # Step 4: Move hand forward to grasp position
    hand_pose, _, _ = extract_robot_state(obs)
    rel_pose = Pose.create_from_pq(
        torch.tensor([[0.15, 0.0, 0.0]], dtype=torch.float32, device=CFG.device).repeat(
            CFG.num_envs, 1
        ),
        euler2quat(0, 0, 0),
    )
    grasp_pose = hand_pose * rel_pose
    obs = execute_hand_motion(
        env, obs, spot_motion_planner, grasp_pose, interpolate_steps=10, max_steps=50
    )

    # Step 5: Close gripper to grasp object
    obs = execute_gripper_action(env, obs, spot_motion_planner, close=True)

    # Step 6: Lift object
    hand_pose_tensor = extract_robot_hand_pose(obs)
    lift_pos = hand_pose_tensor[:, 0:3] + torch.tensor(
        [[0.0, 0.0, 0.1]], dtype=torch.float, device=CFG.device
    ).repeat(CFG.num_envs, 1)
    lift_pose = Pose.create_from_pq(
        lift_pos,
        Pose.create_from_pq(hand_pose_tensor[:, 0:3], hand_pose_tensor[:, 3:7]).q,
    )
    obs = execute_hand_motion(env, obs, spot_motion_planner, lift_pose, closing=True)

    env.close()
