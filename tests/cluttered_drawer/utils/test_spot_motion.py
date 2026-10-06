"""Tests for SpotMotion in ClutteredTable environment."""

import gymnasium as gym
import mani_skill.envs  # type: ignore # pylint: disable=unused-import
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from skill_refactor.args import reset_config
from skill_refactor.benchmarks.cluttered_drawer.utils import (
    extract_robot_body_pose,
    extract_robot_hand_pose,
    extract_robot_joints,
)
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    get_normalize_action_range,
)
from skill_refactor.utils.motion_planning import SpotMotion, WaypointTracker


# @pytest.mark.skip(reason="The script requires maniskills installation")
def test_hand_motion_in_cluttered_drawer_env():
    """Test basic functionality of ClutteredTable environment."""
    # x: > -0.7
    # y: < -0.24 > 0.24
    test_config = {
        "debug_env": True,
        "delta_finger_control": False,
        "num_envs": 16,
        "c_drawer_spot_body_x": -0.8,
        "c_drawer_spot_body_z": 0.0,
    }
    reset_config(test_config)
    spot_motion_planner = SpotMotion(
        device=CFG.device,
    )

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "ClutteredDrawer-v1",
        num_envs=CFG.num_envs,
        reconfiguration_freq=None,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-drawer-hand-motion-test",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )

    for s in range(1):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(123)

        robot_hand_pose = extract_robot_hand_pose(obs)
        from_pose = Pose.create_from_pq(
            robot_hand_pose[:, 0:3], robot_hand_pose[:, 3:7]
        )
        robot_body_pose = extract_robot_body_pose(obs)
        body_pose = Pose.create_from_pq(
            robot_body_pose[:, 0:3], robot_body_pose[:, 3:7]
        )
        from_joints = extract_robot_joints(obs)

        object_p = robot_body_pose[:, 0:3].clone()
        object_p[:, 0] += 0.6
        object_p[:, 2] += 0.3
        object_q = robot_body_pose[:, 3:7].clone()
        object_pose = Pose.create_from_pq(object_p, object_q)
        object_pose_mat = object_pose.to_transformation_matrix()
        object_y_axis = object_pose_mat[:, :3, 1]
        object_y_axis_proj = object_y_axis.clone()
        object_y_axis_proj[..., 2] = 0
        object_y_axis_proj = F.normalize(object_y_axis_proj, dim=-1)
        # project to xoy plane
        object_center = object_p
        approaching = torch.tensor([0, 0, -1], dtype=torch.float32).to(CFG.device)
        approaching = approaching.unsqueeze(0).repeat(CFG.num_envs, 1)
        grasp_pose = spot_motion_planner.build_grasp_pose(
            approaching, object_y_axis_proj, object_center
        )

        reach_actions = []
        # First lift hand
        lift_pos = grasp_pose.p.clone()
        # lift_pos[:, 2] += CFG.c_drawer_reachtograsp_lift_hand_z  # lift hand up by 30cm
        lift_pose = Pose.create_from_pq(lift_pos, grasp_pose.q)
        # kinematic plan
        reach_actions.extend(
            spot_motion_planner.move_hand_from_to_pose(
                body_pose,
                from_joints,
                from_pose,
                lift_pose,
                closing=torch.zeros(
                    (CFG.num_envs,), device=CFG.device, dtype=torch.bool
                ),
                interpolate_steps=40,
            )
        )
        assert reach_actions[0].shape == (CFG.num_envs, 10)
        # Then move to grasp pose by calculating delta actions
        # and feedback control

        # Get action normalization parameters from environment
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            env, CFG.control_mode
        )

        # Create waypoint tracker for feedback control
        tracker = WaypointTracker(
            plan=reach_actions,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            angular_threshold=CFG.angular_threshold,
            waypoint_threshold=CFG.waypoint_threshold,
            device=CFG.device,
        )

        # Execute motion with feedback control
        for _ in range(60):
            curr_qpos = extract_robot_joints(obs)
            if len(tracker.plan) == 0 and tracker.subgoal_achieved(curr_qpos).all():
                break
            delta_actions = tracker.compute_delta_actions(curr_qpos)
            obs, _, _, _, _ = env.step(delta_actions)

        curr_hand_pose = extract_robot_hand_pose(obs)
        position_error = torch.norm(curr_hand_pose[:, 0:3] - grasp_pose.p, dim=-1)
        orientation_error = torch.acos(
            torch.clamp(
                torch.sum(curr_hand_pose[:, 3:7] * grasp_pose.q, dim=-1), -1.0, 1.0
            )
        )
        assert (
            position_error < 0.02
        ).all(), f"Position error too high: {position_error}"
        assert (
            orientation_error < 0.05
        ).all(), f"Orientation error too high: {orientation_error}"

    env.close()


def test_body_motion_in_cluttered_drawer_env():
    """Test body motion planning and control in ClutteredDrawer environment."""
    test_config = {
        "debug_env": True,
        "delta_finger_control": False,
        "num_envs": 16,
        "c_drawer_spot_body_x": -0.8,
        "c_drawer_spot_body_z": 0.0,
    }
    reset_config(test_config)
    spot_motion_planner = SpotMotion(
        device=CFG.device,
    )

    env_kwargs = {"obs_mode": "state", "render_mode": "rgb_array", "sim_backend": "gpu"}
    basic_env = gym.make(
        "ClutteredDrawer-v1",
        num_envs=CFG.num_envs,
        reconfiguration_freq=None,
        **env_kwargs,
    )

    env = ManiSkillsRecordVideo(
        basic_env,
        output_dir="videos/c-drawer-body-motion-test",
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=300,
        video_fps=30,
    )

    for s in range(1):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(123)

        # Get current robot state
        robot_body_pose = extract_robot_body_pose(obs)
        curr_body_pose = Pose.create_from_pq(
            robot_body_pose[:, 0:3], robot_body_pose[:, 3:7]
        )
        from_joints = extract_robot_joints(obs)

        # Create target body pose: move forward 0.3m and rotate 30 degrees
        target_p = robot_body_pose[:, 0:3].clone()
        target_p[:, 0] += 0.5  # Move forward 0.3m
        target_q = euler2quat(0.0, 0.0, np.pi / 3)
        target_q = (
            torch.tensor(target_q, dtype=torch.float32)
            .to(CFG.device)
            .unsqueeze(0)
            .repeat(CFG.num_envs, 1)
        )
        target_body_pose = Pose.create_from_pq(target_p, q=target_q)
        holding = torch.zeros((CFG.num_envs,), device=CFG.device, dtype=torch.bool)

        # Plan body motion
        body_actions = spot_motion_planner.move_body_from_to_pose(
            robot_worldF_curr=curr_body_pose,
            robot_worldF_tgt=target_body_pose,
            curr_joint_positions=from_joints,
            closing=holding,
            interpolate_steps=40,
        )
        assert body_actions[0].shape == (CFG.num_envs, 10)

        # Get action normalization parameters from environment
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            env, CFG.control_mode
        )

        # Create waypoint tracker for feedback control
        tracker = WaypointTracker(
            plan=body_actions,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            angular_threshold=CFG.angular_threshold,
            waypoint_threshold=CFG.waypoint_threshold,
            device=CFG.device,
        )

        # Execute motion with feedback control (body motion needs more steps)
        for step in range(300):
            curr_qpos = extract_robot_joints(obs)
            if len(tracker.plan) == 0 and tracker.subgoal_achieved(curr_qpos).all():
                break
            delta_actions = tracker.compute_delta_actions(curr_qpos)
            obs, _, _, _, _ = env.step(delta_actions)
            print(f"Step {step}, remaining waypoints: {len(tracker.plan)}")

        # Validate body reached target pose
        curr_body_pose_final = extract_robot_body_pose(obs)
        final_pose = Pose.create_from_pq(
            curr_body_pose_final[:, 0:3], curr_body_pose_final[:, 3:7]
        )

        # Check position error (body motion has lower precision than hand motion)
        position_error = torch.norm(final_pose.p - target_body_pose.p, dim=-1)
        assert (
            position_error < 0.1
        ).all(), f"Body position error too high: {position_error}"

        # Check orientation error
        orientation_error = torch.acos(
            torch.clamp(torch.sum(final_pose.q * target_body_pose.q, dim=-1), -1.0, 1.0)
        )
        assert (
            orientation_error < 0.1
        ).all(), f"Body orientation error too high: {orientation_error}"

    env.close()
