"""ClutteredRoom environment planning components."""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose
from relational_structs import (
    GroundAtom,
    Object,
    PDDLDomain,
    Type,
    Variable,
)
from torch import Tensor
from transforms3d.euler import euler2quat

from skill_refactor.benchmarks.base import (
    BaseRLTAMPSystem,
    GraphData,
    TensorPlanningComponents,
)
from skill_refactor.benchmarks.cluttered_room.cluttered_room_env import (
    ClutteredRoomEnv,
)
from skill_refactor.benchmarks.cluttered_room.cluttered_room_held_env import (
    ClutteredRoomHeldEnv,
)
from skill_refactor.benchmarks.cluttered_room.utils import (
    extract_goal_achieved,
    extract_hand_empty,
    extract_obj1_grasped,
    extract_obj2_goal_achieved,
    extract_obj2_grasped,
    extract_object_held,
    extract_object_pose,
    extract_robot_body_pose,
    extract_robot_hand_pose,
    extract_robot_joint_vels,
    extract_robot_joints,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    get_normalize_action_range,
)
from skill_refactor.utils.motion_planning import SpotMotionRoom, WaypointTrackerRoom
from skill_refactor.utils.structs import (
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    ObjectContainer,
    Perceiver,
    Predicate,
    PredicateContainer,
    TypeContainer,
)
from skill_refactor.utils.task_planning import (
    get_object_combinations,
)

# Scenario obstructions mapping
SCENARIO_OBSTRUCTIONS = ["can", "chair"]


def clamp_xy_to_radius(
    rel_pos: torch.Tensor, R: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    rel_pos: (..., 3) or (..., >=2). Clamps x,y to be within a disk of radius R.
    Returns a new tensor (does not modify input in-place).
    """
    xy = rel_pos[..., :2]  # (..., 2)
    r = torch.linalg.norm(xy, dim=-1, keepdim=True)  # (..., 1)
    s = torch.clamp(R / (r + eps), max=1.0)  # (..., 1)
    xy_clamped = xy * s
    out = rel_pos.clone()
    out[..., :2] = xy_clamped
    return out


def quat_to_yaw_only(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B, 4) quaternion in (qw, qx, qy, qz)
    returns: (B, 4) yaw-only quaternion
    """
    qw, qx, qy, qz = q.unbind(dim=-1)

    # Extract yaw
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    half_yaw = 0.5 * yaw

    q_yaw = torch.stack(
        (
            torch.cos(half_yaw),  # qw
            torch.zeros_like(half_yaw),  # qx
            torch.zeros_like(half_yaw),  # qy
            torch.sin(half_yaw),  # qz
        ),
        dim=-1,
    )

    return q_yaw


def _replace_obstruction_pose(obs: Tensor, new_obs_poses: Tensor) -> Tensor:
    """Replace obstruction position in observation.

    idx is 0-indexed obstruction number.
    """
    new_obs = obs.clone()
    num_obstructions = len(CFG.scenario.split(","))
    new_obs_poses = new_obs_poses.to(obs.device)
    assert new_obs_poses.shape[1] == num_obstructions
    assert new_obs_poses.shape[0] == obs.shape[0]

    for idx in range(num_obstructions):
        if idx == 0:
            # can
            blocking_s, blocking_e = 68, 75
            new_obs[:, blocking_s:blocking_e] = new_obs_poses[:, idx]
        else:
            raise ValueError(f"Unknown obstruction index: {idx}")

    return new_obs


class ClutteredRoomTypes(TypeContainer):
    """Container for ClutteredRoom types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = Type("robot")
        self.pp_obj = Type("pp_obj")  # Pick-place objects (bowl, box)
        self.goal = Type("goal")  # Pick-place goals
        self.can = Type("can")  # Obstruction
        self.chair = Type("chair")  # Obstruction

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {self.robot, self.pp_obj, self.can, self.chair, self.goal}

    def as_dict(self) -> dict[str, Type]:
        """Convert to dictionary of types."""
        return {
            "robot": self.robot,
            "pp_obj": self.pp_obj,
            "goal": self.goal,
            "can": self.can,
            "chair": self.chair,
        }


class ClutteredRoomPredicates(PredicateContainer):
    """Container for ClutteredRoom predicates."""

    def __init__(self, types: ClutteredRoomTypes) -> None:
        """Initialize predicates."""
        isbowl = Predicate("IsBowl", [types.pp_obj])
        isbox = Predicate("IsBox", [types.pp_obj])
        bodyreadypick = Predicate("BodyReadyPick", [types.robot, types.pp_obj])
        bodyreadyplace = Predicate("BodyReadyPlace", [types.robot, types.goal])
        holdingbox = Predicate("HoldingBox", [types.robot])
        holdingbowl = Predicate("HoldingBowl", [types.robot])
        handempty = Predicate("HandEmpty", [types.robot])
        at_goal = Predicate("AtGoal", [types.pp_obj, types.goal])  # obj at goal
        box_at_goal = Predicate("BoxAtGoal", [types.robot])  # for enforced box first
        no_bodyreadyplace = Predicate("NoBodyReadyPlace", [types.robot])

        self.predicates = {
            "IsBowl": isbowl,
            "IsBox": isbox,
            "BodyReadyPick": bodyreadypick,
            "BodyReadyPlace": bodyreadyplace,
            "HoldingBox": holdingbox,
            "HoldingBowl": holdingbowl,
            "HandEmpty": handempty,
            "AtGoal": at_goal,
            "BoxAtGoal": box_at_goal,
            "NoBodyReadyPlace": no_bodyreadyplace,
        }


class ClutteredRoomObjectsSC1(ObjectContainer):
    """Container for ClutteredRoom objects."""

    def __init__(self, types: ClutteredRoomTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.bowl = Object("bowl", types.pp_obj)
        self.box = Object("box", types.pp_obj)
        self.bowl_goal = Object("bowl_goal", types.goal)
        self.box_goal = Object("box_goal", types.goal)
        self.can = Object("can", types.can)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.bowl,
            self.box,
            self.bowl_goal,
            self.box_goal,
            self.can,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        return {
            "robot": self.robot,
            "bowl": self.bowl,
            "box": self.box,
            "bowl_goal": self.bowl_goal,
            "box_goal": self.box_goal,
            "can": self.can,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.bowl: 1,
            self.box: 2,
            self.bowl_goal: 3,
            self.box_goal: 4,
            self.can: 5,
        }


def _extract_current_objects() -> ClutteredRoomObjectsSC1:
    if CFG.scenario == "1":
        return ClutteredRoomObjectsSC1(ClutteredRoomTypes())
    raise ValueError(f"Unknown scenario: {CFG.scenario}")


class BaseClutteredRoomEnvSkill(LiftedOperatorSkill):
    """Base class for ClutteredRoom environment skills."""

    def __init__(self, env: ClutteredRoomEnv, operators: set[LiftedOperator]) -> None:
        """Initialize skill."""
        super().__init__()
        self._all_operators = operators
        self._control_mode = env.control_mode
        self._lifted_operator = self.get_lifted_operator()
        self.device = env.device
        self.spot_motion_generator = SpotMotionRoom(
            device=env.device,
        )
        (
            self.normalize_action,
            self.arm_action_low,
            self.arm_action_high,
        ) = get_normalize_action_range(env, self._control_mode)
        self._current_plan: List[Tensor] | None = []
        self.spot_motion_tracker: Optional[WaypointTrackerRoom] = None
        self.use_qvel = False

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        """Reset skill with ground operator."""
        self._current_plan = []
        return super().reset(ground_operator, obs)

    @abc.abstractmethod
    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> List[Tensor]:
        """Get kinematic plan given objects and observation."""
        raise NotImplementedError

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Get action given objects and observation."""
        if not self._current_plan:
            # get kinematic plan (qpos) given objects and observation
            self._current_plan = self._get_kinematic_plan_given_objects(objects, obs)
            self.spot_motion_tracker = WaypointTrackerRoom(
                plan=self._current_plan,
                normalize_action=self.normalize_action,
                arm_action_low=self.arm_action_low,
                arm_action_high=self.arm_action_high,
                waypoint_threshold=CFG.waypoint_threshold,
                angular_threshold=CFG.angular_threshold,
                device=self.device,
            )
        assert (
            self._control_mode == "pd_joint_delta_pos"
        ), f"Unsupported control mode: {self._control_mode}"
        # Type narrowing for mypy
        assert self.spot_motion_tracker is not None, "spot_motion_tracker must be set"
        curr_qpos = extract_robot_joints(obs)
        if self.use_qvel:
            curr_qvel = extract_robot_joint_vels(obs)
            delta_qpos = self.spot_motion_tracker.compute_delta_actions(
                curr_qpos, curr_qvel
            )
        else:
            delta_qpos = self.spot_motion_tracker.compute_delta_actions(curr_qpos)
        # Update _current_plan as the tracker's remaining plan
        self._current_plan = self.spot_motion_tracker.plan
        return delta_qpos

    def get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op for op in self._all_operators if op.name == self.get_operator_name()
        )

    def get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class GoToPickBowlSkill(BaseClutteredRoomEnvSkill):
    """Skill for navigating to pick bowl."""

    def __init__(self, env, operators):
        super().__init__(env, operators)
        self.use_qvel = True

    def get_operator_name(self) -> str:
        return "GoToPickBowl"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Navigate body to bowl for picking."""
        bowl_obj = objects[1]
        body_actions = []
        assert bowl_obj.name == "bowl", f"Expected bowl, got {bowl_obj.name}"

        # Check if holding any object
        holding = extract_object_held(obs, bowl_obj).to(torch.bool)

        # Get current state
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get bowl pose
        bowl_pose_tensor = extract_object_pose(obs, bowl_obj)
        bowl_pose = Pose.create_from_pq(
            bowl_pose_tensor[:, 0:3], bowl_pose_tensor[:, 3:7]
        )

        # Build target body pose
        robot_body_pos_1 = robot_body_pose_tensor[:, 0:3].clone()
        robot_body_pos_1[:, 0] = CFG.c_room_navigate_home_x
        target_body_pose1 = Pose.create_from_pq(
            robot_body_pos_1, robot_body_pose_tensor[:, 3:7]
        )

        body_actions1 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=robot_body_pose,
            robot_worldF_tgt=target_body_pose1,
            curr_joint_positions=joint_positions,
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions1)

        robot_body_pos_2 = bowl_pose_tensor[:, 0:3].clone()
        robot_body_pos_2[:, 0] = CFG.c_room_navigate_home_x
        robot_body_pos_2[:, 2] = robot_body_pos_1[:, 2]
        target_body_pose2 = Pose.create_from_pq(
            robot_body_pos_2, robot_body_pose_tensor[:, 3:7]
        )
        body_actions2 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose1,
            robot_worldF_tgt=target_body_pose2,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=30,
        )
        body_actions.extend(body_actions2)

        target_body_pose3 = self.spot_motion_generator.build_navigate_pose(
            target_pose=bowl_pose,
            facing_axis="x",
            distance=CFG.c_room_navigate_bowl_rel_dx,
        )

        # Plan body motion
        body_actions3 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose2,
            robot_worldF_tgt=target_body_pose3,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions3)
        return body_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when navigation complete or collision detected."""
        assert self._current_plan is not None
        collision = obs[:, -1].to(torch.bool)
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(collision, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(collision, dtype=torch.bool)
        )
        force_terminated = not_have_current_plan_tensor | collision
        pick_obj = objects[1]

        robot_body_pose = extract_robot_body_pose(obs)[:, 0:3]
        obj_pose = extract_object_pose(obs, pick_obj)[:, 0:3]
        dx_navigate = (
            CFG.c_room_navigate_bowl_rel_dx
            if ("bowl" in pick_obj.name)
            else CFG.c_room_navigate_box_rel_dx
        )

        # Check if body is near object
        dx = torch.abs(torch.abs(robot_body_pose[:, 0] - obj_pose[:, 0]) - dx_navigate)
        dy = torch.abs(robot_body_pose[:, 1] - obj_pose[:, 1])

        ready = (dx < CFG.c_room_body_ready_pick_dist) & (
            dy < CFG.c_room_body_ready_pick_dist
        )
        return force_terminated | ready


class PickBowlSkill(BaseClutteredRoomEnvSkill):
    """Skill for picking bowl (hand motion + grasp)."""

    def __init__(self, env, operators):
        super().__init__(env, operators)
        self.use_qvel = True

    def get_operator_name(self) -> str:
        return "PickBowl"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Execute bowl picking sequence: approach from top, grasp, lift."""
        bowl_obj = objects[1]
        assert bowl_obj.name == "bowl", f"Expected bowl, got {bowl_obj.name}"

        B = obs.shape[0]
        all_actions: List[Tensor] = []

        # Get current state
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        hand_pose = Pose.create_from_pq(
            robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
        )
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get bowl pose
        bowl_pose_tensor = extract_object_pose(obs, bowl_obj)
        bowl_pose = Pose.create_from_pq(
            bowl_pose_tensor[:, 0:3], bowl_pose_tensor[:, 3:7]
        )

        # Step 1: Open gripper
        all_actions.extend(
            self.spot_motion_generator.open_gripper(curr_qpos=joint_positions, t=10)
        )

        # Step 2: Move hand to prepare pose (above bowl)
        rel_pose_robot = Pose.create_from_pq(
            torch.zeros((B, 3), dtype=torch.float32, device=self.device),
            euler2quat(0, CFG.c_room_pick_bowl_hand_dpitch, 0),
        )
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [[CFG.c_room_pick_bowl_hand_dx, 0.0, CFG.c_room_pick_bowl_hand_dz]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        prepare_pose1 = Pose.create_from_pq(hand_pose.p, (body_pose * rel_pose_robot).q)

        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                hand_pose,
                prepare_pose1,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=50,
            )
        )

        # Step 2: Move hand to prepare pose (above bowl)
        prepare_pose2 = Pose.create_from_pq(
            (bowl_pose * rel_pose_obj).p, prepare_pose1.q
        )

        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                hand_pose,
                prepare_pose2,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=50,
            )
        )

        # Step 3: Move hand forward to grasp position
        from_joints = all_actions[-1].clone()

        rel_pose = Pose.create_from_pq(
            torch.tensor(
                [[CFG.c_room_pick_bowl_hand_forward, 0.0, 0.0]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        grasp_pose = prepare_pose2 * rel_pose
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                prepare_pose2,
                grasp_pose,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        # Step 4: Close gripper
        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.close_gripper(curr_qpos=from_joints, t=10)
        )

        # Step 5: Lift object
        from_joints = all_actions[-1].clone()
        lift_pos = grasp_pose.p + torch.tensor(
            [[CFG.c_room_pick_obj_lifting_dx, 0.0, CFG.c_room_pick_obj_lifting_dz]],
            dtype=torch.float,
            device=self.device,
        ).repeat(B, 1)
        lift_pose = Pose.create_from_pq(lift_pos, grasp_pose.q)
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                grasp_pose,
                lift_pose,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        return all_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when bowl is grasped and lifted."""
        holding = extract_object_held(obs, objects[1]).to(torch.bool).squeeze(-1)
        handempty = extract_hand_empty(obs).to(torch.bool).squeeze(-1)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(holding, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(holding, dtype=torch.bool)
        )
        # Pick terminates when waypoints are done AND object is held
        terminated = holding & not_have_current_plan_tensor & (~handempty)
        collision = obs[:, -1].to(torch.bool)
        terminated = terminated | collision
        return terminated


class GoToPlaceBowlSkill(BaseClutteredRoomEnvSkill):
    """Skill for navigating to place bowl."""

    def get_operator_name(self) -> str:
        return "GoToPlaceBowl"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Navigate body to bowl goal for placing."""
        bowl_goal_obj = objects[2]
        body_actions = []
        assert (
            bowl_goal_obj.name == "bowl_goal"
        ), f"Expected bowl_goal, got {bowl_goal_obj.name}"

        # Should be holding bowl
        holding = extract_object_held(obs, objects[1]).to(torch.bool)

        # Get current state
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get bowl goal pose (from obj1 goal)
        bowl_goal_pose_tensor = extract_object_pose(obs, bowl_goal_obj)
        bowl_goal_pose = Pose.create_from_pq(
            bowl_goal_pose_tensor[:, 0:3], bowl_goal_pose_tensor[:, 3:7]
        )

        # Build target body pose
        robot_body_pos_1 = robot_body_pose_tensor[:, 0:3].clone()
        robot_body_pos_1[:, 0] = CFG.c_room_navigate_home_x
        target_body_pose1 = Pose.create_from_pq(
            robot_body_pos_1, robot_body_pose_tensor[:, 3:7]
        )

        body_actions1 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=robot_body_pose,
            robot_worldF_tgt=target_body_pose1,
            curr_joint_positions=joint_positions,
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions1)

        robot_body_pos_2 = bowl_goal_pose_tensor[:, 0:3].clone()
        robot_body_pos_2[:, 0] = CFG.c_room_navigate_home_x
        robot_body_pos_2[:, 2] = robot_body_pos_1[:, 2]
        target_body_pose2 = Pose.create_from_pq(
            robot_body_pos_2, robot_body_pose_tensor[:, 3:7]
        )
        body_actions2 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose1,
            robot_worldF_tgt=target_body_pose2,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=30,
        )
        body_actions.extend(body_actions2)

        target_body_pose3 = self.spot_motion_generator.build_navigate_pose(
            target_pose=bowl_goal_pose,
            facing_axis="x",
            distance=CFG.c_room_navigate_bowl_rel_dx,
        )

        # Plan body motion
        body_actions3 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose2,
            robot_worldF_tgt=target_body_pose3,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions3)
        return body_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when navigation complete or collision detected."""
        assert self._current_plan is not None
        collision = obs[:, -1].to(torch.bool)
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(collision, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(collision, dtype=torch.bool)
        )
        force_terminated = not_have_current_plan_tensor | collision

        obj_held = objects[1]
        goal = objects[2]
        robot_body_pose = extract_robot_body_pose(obs)[:, 0:3]
        goal_pose = extract_object_pose(obs, goal)[:, 0:3]
        holding = extract_object_held(obs, obj_held).to(torch.bool).squeeze(-1)

        hand_pose_q = extract_robot_hand_pose(obs)[:, 3:7]
        hand_rot_mat = quaternion_to_matrix(hand_pose_q)
        hand_y_axis = hand_rot_mat[:, :, 1]
        negative_y_axis = torch.zeros_like(hand_y_axis)
        negative_y_axis[:, 1] = -1.0
        cos_angle = torch.clamp(
            torch.sum(hand_y_axis * negative_y_axis, dim=1), -1.0, 1.0
        )
        angle = torch.abs(torch.acos(cos_angle))  # in radians

        angle_ok = angle < CFG.c_room_place_hand_facing_goal_thresh_rad

        # Check if body is near goal and holding the object
        dx_navigate = (
            CFG.c_room_navigate_bowl_rel_dx
            if ("bowl" in obj_held.name)
            else CFG.c_room_navigate_box_rel_dx
        )

        # Check if body is near object
        dx = torch.abs(torch.abs(robot_body_pose[:, 0] - goal_pose[:, 0]) - dx_navigate)
        dy = torch.abs(robot_body_pose[:, 1] - goal_pose[:, 1])

        ready = (
            (dx < CFG.c_room_body_ready_pick_dist)
            & (dy < CFG.c_room_body_ready_pick_dist)
            & holding
            & angle_ok
        )
        return force_terminated | ready


class PlaceBowlSkill(BaseClutteredRoomEnvSkill):
    """Skill for placing bowl (hand motion + release)."""

    def get_operator_name(self) -> str:
        return "PlaceBowl"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Execute bowl placing sequence: lower, release, lift."""
        goal_obj = objects[2]
        B = obs.shape[0]
        all_actions: List[Tensor] = []

        # Get current state
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        hand_pose = Pose.create_from_pq(
            robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
        )
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get bowl goal pose
        bowl_goal_pose_tensor = extract_object_pose(obs, goal_obj)
        bowl_goal_pose = Pose.create_from_pq(
            bowl_goal_pose_tensor[:, 0:3], bowl_goal_pose_tensor[:, 3:7]
        )

        # Step 1: Move hand to place pose (above goal)
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [
                    [
                        CFG.c_room_pre_place_obj_lifting_dx,
                        0.0,
                        CFG.c_room_pre_place_obj_lifting_dz,
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        place_pose1 = Pose.create_from_pq(
            (bowl_goal_pose * rel_pose_obj).p, hand_pose.q
        )

        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                joint_positions,
                hand_pose,
                place_pose1,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 2: Move hand to place pose (overlapping goal)
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [
                    [
                        CFG.c_room_pre_place_obj_lifting_dx,
                        0.0,
                        CFG.c_room_place_obj_lifting_dz,
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        place_pose2 = Pose.create_from_pq(
            (bowl_goal_pose * rel_pose_obj).p, hand_pose.q
        )

        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                all_actions[-1],
                place_pose1,
                place_pose2,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 3: Open gripper
        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.open_gripper(curr_qpos=from_joints, t=10)
        )

        # Step 4: Lift hand
        from_joints = all_actions[-1].clone()
        lift_pos = place_pose2.p + torch.tensor(
            [[0.0, 0.0, CFG.c_room_post_place_obj_lifting_dz]],
            dtype=torch.float,
            device=self.device,
        ).repeat(B, 1)
        lift_pose = Pose.create_from_pq(lift_pos, place_pose2.q)
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                place_pose2,
                lift_pose,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        return all_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when bowl is released and at goal."""
        goal_obj = objects[1]
        at_goal = extract_goal_achieved(obs, goal_obj).to(torch.bool).squeeze(-1)
        holding = extract_object_held(obs, goal_obj).to(torch.bool).squeeze(-1)
        hand_empty = extract_hand_empty(obs).to(torch.bool).squeeze(-1)

        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(at_goal, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(at_goal, dtype=torch.bool)
        )
        terminated = at_goal & ~holding & not_have_current_plan_tensor & hand_empty
        collision = obs[:, -1].to(torch.bool)
        terminated = terminated | collision
        return terminated


class GoToPickBoxSkill(GoToPickBowlSkill):
    """Skill for navigating to pick box."""

    def get_operator_name(self) -> str:
        return "GoToPickBox"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Navigate body to box for picking."""
        box_obj = objects[1]
        body_actions = []
        assert box_obj.name == "box", f"Expected box, got {box_obj.name}"

        # Check if holding any object
        holding = extract_object_held(obs, box_obj).to(torch.bool)

        # Get current state
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get box pose
        box_pose_tensor = extract_object_pose(obs, box_obj)
        box_pose = Pose.create_from_pq(box_pose_tensor[:, 0:3], box_pose_tensor[:, 3:7])

        # Build target body pose
        robot_body_pos_1 = robot_body_pose_tensor[:, 0:3].clone()
        robot_body_pos_1[:, 0] = CFG.c_room_navigate_home_x
        target_body_pose1 = Pose.create_from_pq(
            robot_body_pos_1, robot_body_pose_tensor[:, 3:7]
        )

        body_actions1 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=robot_body_pose,
            robot_worldF_tgt=target_body_pose1,
            curr_joint_positions=joint_positions,
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions1)

        robot_body_pos_2 = box_pose_tensor[:, 0:3].clone()
        robot_body_pos_2[:, 0] = CFG.c_room_navigate_home_x
        robot_body_pos_2[:, 2] = robot_body_pos_1[:, 2]
        target_body_pose2 = Pose.create_from_pq(
            robot_body_pos_2, robot_body_pose_tensor[:, 3:7]
        )
        body_actions2 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose1,
            robot_worldF_tgt=target_body_pose2,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=30,
        )
        body_actions.extend(body_actions2)

        target_body_pose3 = self.spot_motion_generator.build_navigate_pose(
            target_pose=box_pose,
            facing_axis="x",
            distance=CFG.c_room_navigate_box_rel_dx,
        )

        # Plan body motion
        body_actions3 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose2,
            robot_worldF_tgt=target_body_pose3,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=40,
        )
        body_actions.extend(body_actions3)
        return body_actions


class PickBoxSkill(PickBowlSkill):
    """Skill for picking box (hand motion + grasp from side)."""

    def get_operator_name(self) -> str:
        return "PickBox"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Execute box picking sequence: approach from side, grasp, lift."""
        box_obj = objects[1]
        assert box_obj.name == "box", f"Expected box, got {box_obj.name}"

        B = obs.shape[0]
        all_actions: List[Tensor] = []

        # Get current state
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        hand_pose = Pose.create_from_pq(
            robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
        )
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get box pose
        box_pose_tensor = extract_object_pose(obs, box_obj)
        box_pose = Pose.create_from_pq(box_pose_tensor[:, 0:3], box_pose_tensor[:, 3:7])

        # Step 1: Open gripper
        all_actions.extend(
            self.spot_motion_generator.open_gripper(curr_qpos=joint_positions, t=10)
        )

        # Step 2: Move hand to prepare pose (side approach)
        rel_pose_robot = Pose.create_from_pq(
            torch.zeros((B, 3), dtype=torch.float32, device=self.device),
            euler2quat(0, CFG.c_room_pick_box_hand_dpitch, 0),
        )
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [[CFG.c_room_pick_box_hand_dx1, CFG.c_room_pick_box_hand_dy1, 0.0]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        prepare_pose = Pose.create_from_pq(
            (box_pose * rel_pose_obj).p, (body_pose * rel_pose_robot).q
        )

        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                hand_pose,
                prepare_pose,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 3: Move closer to box
        from_joints = all_actions[-1].clone()
        rel_pose_obj2 = Pose.create_from_pq(
            torch.tensor(
                [[CFG.c_room_pick_box_hand_dx2, CFG.c_room_pick_box_hand_dy2, 0.0]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        prepare_pose2 = Pose.create_from_pq(
            (box_pose * rel_pose_obj2).p, (body_pose * rel_pose_robot).q
        )
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                prepare_pose,
                prepare_pose2,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        # Step 4: Move hand to grasp position
        from_joints = all_actions[-1].clone()
        hand_pose_p = prepare_pose2.p
        hand_pose_q = prepare_pose2.q
        hand_pose_updated = Pose.create_from_pq(hand_pose_p, hand_pose_q)

        rel_pose = Pose.create_from_pq(
            torch.tensor(
                [[CFG.c_room_pick_box_hand_forward, 0.0, 0.0]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        grasp_pose = hand_pose_updated * rel_pose
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                hand_pose_updated,
                grasp_pose,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 5: Close gripper
        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.close_gripper(curr_qpos=from_joints, t=10)
        )

        # Step 6: Lift object
        from_joints = all_actions[-1].clone()
        lift_pos = grasp_pose.p + torch.tensor(
            [[CFG.c_room_pick_obj_lifting_dx, 0.0, CFG.c_room_pick_obj_lifting_dz]],
            dtype=torch.float,
            device=self.device,
        ).repeat(B, 1)
        lift_pose = Pose.create_from_pq(lift_pos, grasp_pose.q)
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                grasp_pose,
                lift_pose,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        return all_actions


class GoToPlaceBoxSkill(GoToPlaceBowlSkill):
    """Skill for navigating to place box."""

    def get_operator_name(self) -> str:
        return "GoToPlaceBox"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Navigate body to box goal for placing."""
        box_goal_obj = objects[2]
        body_actions = []

        # Should be holding box
        holding = extract_object_held(obs, objects[1]).to(torch.bool)

        # Get current state
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        hand_pose = Pose.create_from_pq(
            robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
        )
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get box goal pose
        box_goal_pose_tensor = extract_object_pose(obs, box_goal_obj)
        box_goal_pose = Pose.create_from_pq(
            box_goal_pose_tensor[:, 0:3], box_goal_pose_tensor[:, 3:7]
        )

        # First get hand to pre-place pose
        B = obs.shape[0]
        body2hand_rel_pos = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        body2hand_rel_pos[:, 0] = CFG.c_room_pre_goto_place_hand_body_dx
        body2hand_rel_pos[:, 2] = CFG.c_room_pre_goto_place_hand_body_dz
        rel_pose_robot = Pose.create_from_pq(
            body2hand_rel_pos,
            euler2quat(0, CFG.c_room_pick_box_hand_dpitch, 0),
        )
        prepare_pose = robot_body_pose * rel_pose_robot

        from_joints = joint_positions.clone()
        body_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                robot_body_pose,
                from_joints,
                hand_pose,
                prepare_pose,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Build target body pose
        robot_body_pos_1 = robot_body_pose_tensor[:, 0:3].clone()
        robot_body_pos_1[:, 0] = CFG.c_room_navigate_home_x
        target_body_pose1 = Pose.create_from_pq(
            robot_body_pos_1, robot_body_pose_tensor[:, 3:7]
        )

        body_actions1 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=robot_body_pose,
            robot_worldF_tgt=target_body_pose1,
            curr_joint_positions=body_actions[-1].clone(),
            closing=holding,
            interpolate_steps=20,
        )
        body_actions.extend(body_actions1)

        robot_body_pos_2 = box_goal_pose_tensor[:, 0:3].clone()
        robot_body_pos_2[:, 0] = CFG.c_room_navigate_home_x
        robot_body_pos_2[:, 2] = robot_body_pos_1[:, 2]
        target_body_pose2 = Pose.create_from_pq(
            robot_body_pos_2, robot_body_pose_tensor[:, 3:7]
        )
        body_actions2 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose1,
            robot_worldF_tgt=target_body_pose2,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=30,
        )
        body_actions.extend(body_actions2)

        target_body_pose3 = self.spot_motion_generator.build_navigate_pose(
            target_pose=box_goal_pose,
            facing_axis="x",
            distance=CFG.c_room_navigate_box_rel_dx,
        )

        # Plan body motion
        body_actions3 = self.spot_motion_generator.move_body_from_to_pose(
            robot_worldF_curr=target_body_pose2,
            robot_worldF_tgt=target_body_pose3,
            curr_joint_positions=body_actions[-1],
            closing=holding,
            interpolate_steps=40,
        )
        body_actions.extend(body_actions3)
        return body_actions


class PlaceBoxSkill(PlaceBowlSkill):
    """Skill for placing box (hand motion + release)."""

    def get_operator_name(self) -> str:
        return "PlaceBox"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Execute bowl placing sequence: lower, release, lift."""
        goal_obj = objects[2]
        B = obs.shape[0]
        all_actions: List[Tensor] = []

        # Get current state
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        hand_pose = Pose.create_from_pq(
            robot_hand_pose_tensor[:, 0:3], robot_hand_pose_tensor[:, 3:7]
        )
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        joint_positions = extract_robot_joints(obs)

        # Get bowl goal pose
        bowl_goal_pose_tensor = extract_object_pose(obs, goal_obj)
        bowl_goal_pose = Pose.create_from_pq(
            bowl_goal_pose_tensor[:, 0:3], bowl_goal_pose_tensor[:, 3:7]
        )

        # Step 1: Move hand to place pose (above goal)
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [[0.0, 0.0, CFG.c_room_pre_place_obj_lifting_dz]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        place_pose1 = Pose.create_from_pq(
            (bowl_goal_pose * rel_pose_obj).p, hand_pose.q
        )

        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                joint_positions,
                hand_pose,
                place_pose1,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 2: Move hand to place pose (overlapping goal)
        rel_pose_obj = Pose.create_from_pq(
            torch.tensor(
                [[0.0, 0.0, CFG.c_room_place_obj_lifting_dz]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(B, 1),
            euler2quat(0, 0, 0),
        )
        place_pose2 = Pose.create_from_pq(
            (bowl_goal_pose * rel_pose_obj).p, hand_pose.q
        )

        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                all_actions[-1],
                place_pose1,
                place_pose2,
                closing=torch.ones((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=30,
            )
        )

        # Step 3: Open gripper
        from_joints = all_actions[-1].clone()
        all_actions.extend(
            self.spot_motion_generator.open_gripper(curr_qpos=from_joints, t=10)
        )

        # Step 4: Lift hand
        from_joints = all_actions[-1].clone()
        lift_pos = place_pose2.p + torch.tensor(
            [[0.0, 0.0, CFG.c_room_post_place_obj_lifting_dz]],
            dtype=torch.float,
            device=self.device,
        ).repeat(B, 1)
        lift_pose = Pose.create_from_pq(lift_pos, place_pose2.q)
        all_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                place_pose2,
                lift_pose,
                closing=torch.zeros((B,), device=self.device, dtype=torch.bool),
                interpolate_steps=20,
            )
        )

        return all_actions


class ClutteredRoomPerceiver(Perceiver):
    """Perceiver for ClutteredRoom environment."""

    def __init__(
        self, predicates: ClutteredRoomPredicates, types: ClutteredRoomTypes
    ) -> None:
        """Initialize with required types."""
        self.predicates_container = predicates
        self._types = types
        self.predicate_interpreters = {
            predicates["IsBowl"]: self._interpret_isbowl,
            predicates["IsBox"]: self._interpret_isbox,
            predicates["BodyReadyPick"]: self._interpret_bodyreadypick,
            predicates["BodyReadyPlace"]: self._interpret_bodyreadyplace,
            predicates["HoldingBox"]: self._interpret_holding_box,
            predicates["HoldingBowl"]: self._interpret_holding_bowl,
            predicates["HandEmpty"]: self._interpret_handempty,
            predicates["AtGoal"]: self._interpret_atgoal,
            predicates["BoxAtGoal"]: self._interpret_boxatgoal,
            predicates["NoBodyReadyPlace"]: self._interpret_no_bodyreadyplace,
        }

    def reset(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        assert (
            obs.shape[0] == 1
        ), "Expected batch size of 1 for observation for perceiver reset."

        atoms = self._get_atoms(obs, info)

        # Goal: both objects at their goals
        latest_objects = _extract_current_objects()
        bowl = latest_objects.bowl
        box = latest_objects.box
        bowl_goal = latest_objects.bowl_goal
        box_goal = latest_objects.box_goal

        goal = {
            self.predicates_container["AtGoal"]([bowl, bowl_goal]),
            self.predicates_container["AtGoal"]([box, box_goal]),
        }
        return latest_objects.as_set(), atoms[0], goal

    def step(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Step perceiver with observation."""
        return self._get_atoms(obs, info)

    @property
    def objects(self) -> ObjectContainer:
        """Get the object container."""
        return _extract_current_objects()

    def _get_atoms(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Convert a batch of observations into a List of GroundAtom-sets."""
        B = obs.shape[0]
        atoms_List: List[set[GroundAtom]] = [set() for _ in range(B)]
        latest_objects = _extract_current_objects()

        desired_predicates = list(self.predicates_container.predicates.keys())
        if info is not None and "desired_predicates" in info:
            desired_predicates = [
                p
                for p in info["desired_predicates"]
                if p in self.predicates_container.predicates
            ]

        for predicate, interpret_fn in self.predicate_interpreters.items():
            if predicate.name not in desired_predicates:
                continue

            input_args: List[List[Object]] = []
            for args in get_object_combinations(
                latest_objects.as_set(),
                predicate.types,
                skip_self=True,
            ):
                input_args.append(args)

            stacked_res = interpret_fn(obs, input_args)  # type: ignore[arg-type]
            assert stacked_res.shape[0] == B
            assert stacked_res.shape[1] == len(input_args)

            for b in range(B):
                for n in range(len(input_args)):
                    if stacked_res[b, n]:
                        if predicate.arity > 0:
                            atoms_List[b].add(predicate(input_args[n]))
                        else:
                            atoms_List[b].add(GroundAtom(predicate, []))
        return atoms_List

    def _interpret_isbowl(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret IsBowl predicate."""
        result = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for i, obj_seq in enumerate(objects):
            if obj_seq[0].name == "bowl":
                result[:, i] = True
        return result

    def _interpret_isbox(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret IsBox predicate."""
        result = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for i, obj_seq in enumerate(objects):
            if obj_seq[0].name == "box":
                result[:, i] = True
        return result

    def _interpret_bodyreadypick(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret BodyReadyPick predicate."""
        result_list = []
        for obj_pair in objects:
            assert obj_pair[0].name == "robot"
            pick_obj = obj_pair[1]

            is_goal = pick_obj.name in ["bowl_goal", "box_goal"]
            is_goal_tensor = (
                torch.ones((obs.shape[0],), dtype=torch.bool, device=obs.device)
                if is_goal
                else torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
            )
            robot_body_pose = extract_robot_body_pose(obs)[:, 0:3]
            obj_pose = extract_object_pose(obs, pick_obj)[:, 0:3]
            object_held = extract_object_held(obs, pick_obj).to(torch.bool).squeeze(-1)
            dx_navigate = (
                CFG.c_room_navigate_bowl_rel_dx
                if ("bowl" in pick_obj.name)
                else CFG.c_room_navigate_box_rel_dx
            )

            # Check if body is near object
            dx = torch.abs(
                torch.abs(robot_body_pose[:, 0] - obj_pose[:, 0]) - dx_navigate
            )
            dy = torch.abs(robot_body_pose[:, 1] - obj_pose[:, 1])

            ready = (
                (dx < CFG.c_room_body_ready_pick_dist)
                & (dy < CFG.c_room_body_ready_pick_dist)
                & ~object_held
                & ~is_goal_tensor
            )
            result_list.append(ready)

        return torch.stack(result_list, dim=1)

    def _interpret_bodyreadyplace(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret BodyReadyPlace predicate."""
        result_list = []
        for obj_triplet in objects:
            assert obj_triplet[0].name == "robot"
            goal = obj_triplet[1]

            robot_body_pose = extract_robot_body_pose(obs)[:, 0:3]
            goal_pose = extract_object_pose(obs, goal)[:, 0:3]
            holding = ~extract_hand_empty(obs).to(torch.bool)
            is_goal = goal.name in ["bowl_goal", "box_goal"]
            is_goal_tensor = (
                torch.ones((obs.shape[0],), dtype=torch.bool, device=obs.device)
                if is_goal
                else torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
            )

            hand_pose_q = extract_robot_hand_pose(obs)[:, 3:7]
            hand_rot_mat = quaternion_to_matrix(hand_pose_q)
            hand_y_axis = hand_rot_mat[:, :, 1]
            negative_y_axis = torch.zeros_like(hand_y_axis)
            negative_y_axis[:, 1] = -1.0
            cos_angle = torch.clamp(
                torch.sum(hand_y_axis * negative_y_axis, dim=1), -1.0, 1.0
            )
            angle = torch.abs(torch.acos(cos_angle))  # in radians

            angle_ok = angle < CFG.c_room_place_hand_facing_goal_thresh_rad

            # Check if body is near goal and holding the object

            # Check if body is near object
            dx1 = torch.abs(
                torch.abs(robot_body_pose[:, 0] - goal_pose[:, 0])
                - CFG.c_room_navigate_bowl_rel_dx
            )
            dx2 = torch.abs(
                torch.abs(robot_body_pose[:, 0] - goal_pose[:, 0])
                - CFG.c_room_navigate_box_rel_dx
            )
            dx_ok = (dx1 < CFG.c_room_body_ready_pick_dist) | (
                dx2 < CFG.c_room_body_ready_pick_dist
            )

            dy = torch.abs(robot_body_pose[:, 1] - goal_pose[:, 1])

            ready = (
                dx_ok
                & (dy < CFG.c_room_body_ready_pick_dist)
                & holding
                & is_goal_tensor
                & angle_ok
            )
            result_list.append(ready)

        return torch.stack(result_list, dim=1)

    def _interpret_holding_box(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Holding predicate."""
        result_list = []
        for obj_pair in objects:
            assert obj_pair[0].name == "robot"
            holding = extract_obj2_grasped(obs).to(torch.bool).squeeze(-1)
            result_list.append(holding)

        return torch.stack(result_list, dim=1)

    def _interpret_holding_bowl(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Holding predicate."""
        result_list = []
        for obj_pair in objects:
            assert obj_pair[0].name == "robot"
            holding = extract_obj1_grasped(obs).to(torch.bool).squeeze(-1)
            result_list.append(holding)

        return torch.stack(result_list, dim=1)

    def _interpret_handempty(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret HandEmpty predicate."""
        hand_empty = extract_hand_empty(obs).to(torch.bool)
        result = hand_empty.unsqueeze(1).repeat(1, len(objects))
        return result

    def _interpret_atgoal(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret AtGoal predicate."""
        result_list = []
        for obj_pair in objects:
            obj = obj_pair[0]
            goal = obj_pair[1]

            # Check if this is a valid object-goal pairing
            valid_pairings = {("bowl", "bowl_goal"), ("box", "box_goal")}

            if (obj.name, goal.name) in valid_pairings:
                at_goal = extract_goal_achieved(obs, obj).to(torch.bool).squeeze(-1)
            else:
                at_goal = torch.zeros(
                    (obs.shape[0],), dtype=torch.bool, device=obs.device
                )

            result_list.append(at_goal)

        return torch.stack(result_list, dim=1)

    def _interpret_boxatgoal(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret BoxAtGoal predicate."""
        result_list = []
        for _ in objects:
            box_at_goal = extract_obj2_goal_achieved(obs).to(torch.bool).squeeze(-1)
            result_list.append(box_at_goal)

        return torch.stack(result_list, dim=1)

    def _interpret_no_bodyreadyplace(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret FarFromGoals predicate."""
        objects_box_goal: List[Sequence[Object]] = []
        objects_bowl_goal: List[Sequence[Object]] = []
        for obj_triplet in objects:
            assert obj_triplet[0].name == "robot"
            box_goal_obj = self.objects.as_dict()["box_goal"]
            bowl_goal_obj = self.objects.as_dict()["bowl_goal"]
            objects_box_goal.append([obj_triplet[0], box_goal_obj])
            objects_bowl_goal.append([obj_triplet[0], bowl_goal_obj])

        bodyready_place_box_goal = self._interpret_bodyreadyplace(obs, objects_box_goal)
        bodyready_place_bowl_goal = self._interpret_bodyreadyplace(
            obs, objects_bowl_goal
        )
        no_bodyready_place = ~(bodyready_place_box_goal | bodyready_place_bowl_goal)
        return no_bodyready_place


class ClutteredRoomRLTAMPSystem(BaseRLTAMPSystem):
    """TAMP system for ClutteredRoom environment."""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ClutteredRoom TAMP system."""
        self._render_mode = render_mode
        self.env_kwargs = {
            "obs_mode": "state",
            "render_mode": render_mode,
            "sim_backend": "gpu",
            "control_mode": CFG.control_mode,
            "reward_mode": "sparse",
        }
        if CFG.c_room_force_held:
            self.env_name = "skill_ref/ClutteredRoomForceHeld-v1"
        else:
            self.env_name = "skill_ref/ClutteredRoom-v1"
        super().__init__(
            planning_components, name="ClutteredRoomRLTAMPSystem", seed=seed  # type: ignore
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        envs = gym.make(
            self.env_name,
            num_envs=CFG.num_envs,
            reconfiguration_freq=0,  # NOTE: No reconfiguration during episodes to avoid memory leaks
            **self.env_kwargs,
        )
        return envs

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "ClutteredRoom-domain"

    def get_domain(self) -> PDDLDomain:
        """Get domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.type_container.as_set(),
        )

    @classmethod
    def _create_planning_components(cls) -> TensorPlanningComponents:
        """Create planning components for ClutteredRoom system."""
        types_container = ClutteredRoomTypes()
        predicates = ClutteredRoomPredicates(types_container)

        IsBowl = predicates["IsBowl"]
        IsBox = predicates["IsBox"]
        BodyReadyPick = predicates["BodyReadyPick"]
        BodyReadyPlace = predicates["BodyReadyPlace"]
        HoldingBox = predicates["HoldingBox"]
        HoldingBowl = predicates["HoldingBowl"]
        HandEmpty = predicates["HandEmpty"]
        AtGoal = predicates["AtGoal"]
        BoxAtGoal = predicates["BoxAtGoal"]
        NoBodyReadyPlace = predicates["NoBodyReadyPlace"]

        perceiver = ClutteredRoomPerceiver(predicates, types_container)

        robot = Variable("?robot", types_container.robot)
        pick_obj = Variable("?pick_obj", types_container.pp_obj)
        goal = Variable("?goal", types_container.goal)

        operators = {
            LiftedOperator(
                "GoToPickBowl",
                [robot, pick_obj],
                preconditions={
                    BoxAtGoal([robot]),
                    IsBowl([pick_obj]),
                    HandEmpty([robot]),
                },
                add_effects={
                    BodyReadyPick([robot, pick_obj]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "PickBowl",
                [robot, pick_obj],
                preconditions={
                    BoxAtGoal([robot]),
                    IsBowl([pick_obj]),
                    BodyReadyPick([robot, pick_obj]),
                    HandEmpty([robot]),
                },
                add_effects={
                    HoldingBowl([robot]),
                },
                delete_effects={
                    HandEmpty([robot]),
                    BodyReadyPick([robot, pick_obj]),
                },
            ),
            LiftedOperator(
                "GoToPlaceBowl",
                [robot, pick_obj, goal],
                preconditions={
                    BoxAtGoal([robot]),
                    IsBowl([pick_obj]),
                    HoldingBowl([robot]),
                    NoBodyReadyPlace([robot]),
                },
                add_effects={
                    BodyReadyPlace([robot, goal]),
                },
                delete_effects={
                    NoBodyReadyPlace([robot]),
                },
            ),
            LiftedOperator(
                "PlaceBowl",
                [robot, pick_obj, goal],
                preconditions={
                    BoxAtGoal([robot]),
                    IsBowl([pick_obj]),
                    HoldingBowl([robot]),
                    BodyReadyPlace([robot, goal]),
                },
                add_effects={
                    AtGoal([pick_obj, goal]),
                    HandEmpty([robot]),
                    NoBodyReadyPlace([robot]),
                },
                delete_effects={
                    HoldingBowl([robot]),
                    BodyReadyPlace([robot, goal]),
                },
            ),
            LiftedOperator(
                "GoToPickBox",
                [robot, pick_obj],
                preconditions={
                    IsBox([pick_obj]),
                    HandEmpty([robot]),
                },
                add_effects={
                    BodyReadyPick([robot, pick_obj]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "PickBox",
                [robot, pick_obj],
                preconditions={
                    IsBox([pick_obj]),
                    BodyReadyPick([robot, pick_obj]),
                    HandEmpty([robot]),
                },
                add_effects={
                    HoldingBox([robot]),
                },
                delete_effects={
                    HandEmpty([robot]),
                    BodyReadyPick([robot, pick_obj]),
                },
            ),
            LiftedOperator(
                "GoToPlaceBox",
                [robot, pick_obj, goal],
                preconditions={
                    IsBox([pick_obj]),
                    HoldingBox([robot]),
                    NoBodyReadyPlace([robot]),
                },
                add_effects={
                    BodyReadyPlace([robot, goal]),
                },
                delete_effects={
                    NoBodyReadyPlace([robot]),
                },
            ),
            LiftedOperator(
                "PlaceBox",
                [robot, pick_obj, goal],
                preconditions={
                    IsBox([pick_obj]),
                    HoldingBox([robot]),
                    BodyReadyPlace([robot, goal]),
                },
                add_effects={
                    AtGoal([pick_obj, goal]),
                    HandEmpty([robot]),
                    BoxAtGoal([robot]),
                    NoBodyReadyPlace([robot]),
                },
                delete_effects={
                    HoldingBox([robot]),
                    BodyReadyPlace([robot, goal]),
                },
            ),
        }

        return TensorPlanningComponents(
            type_container=types_container,
            predicate_container=predicates,
            operators=operators,
            skills=set(),
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> ClutteredRoomRLTAMPSystem:
        """Factory method for creating system with default components."""
        planning_components = cls._create_planning_components()
        system = cls(
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        assert isinstance(
            system.env.unwrapped, (ClutteredRoomEnv, ClutteredRoomHeldEnv)
        )
        skills = {
            GoToPickBowlSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type,no-untyped-call]
            PickBowlSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type,no-untyped-call]
            GoToPlaceBowlSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            PlaceBowlSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            GoToPickBoxSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type,no-untyped-call]
            PickBoxSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type,no-untyped-call]
            GoToPlaceBoxSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            PlaceBoxSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)  # type: ignore
        return system

    def state_to_graph(self, state: Tensor, training: bool = False) -> List[GraphData]:
        """Convert batched ClutteredRoom environment state tensor to graph
        representation.

        Args:
            state: State tensor from ClutteredRoom environment, batched
            training: Whether in training mode (unused in this implementation)

        Returns:
            List of GraphData with nodes representing objects and edges representing relationships
        """
        # Get objects from the perceiver's object container
        b = state.shape[0]
        num_nodes = len(self.perceiver.objects.object_to_node.keys())

        # Node features: position (3) + quaternion rotation (4) = 7 dimensions
        node_feature_dim = 7
        node_features = torch.zeros(b, num_nodes, node_feature_dim)

        # Extract features for each object using existing helper functions
        # NOTE: In this env, we represent object poses relative to the robot body frame
        robot_body_pose_tensor = extract_robot_body_pose(state)
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        can_idx = None
        for obj, i in self.perceiver.objects.object_to_node.items():
            # Extract object pose (position + rotation)
            if obj.name == "robot":
                obj_pose_tensor = extract_robot_hand_pose(state)
                obj_pose = Pose.create_from_pq(
                    obj_pose_tensor[:, 0:3], obj_pose_tensor[:, 3:7]
                )
                obj_pose_rel = robot_body_pose.inv() * obj_pose
                obj_pose_rel_tensor = torch.cat(
                    [obj_pose_rel.p, obj_pose_rel.q], dim=-1
                )  # [batch_size, 7]
            else:
                if obj.name == "can":
                    can_idx = i
                obj_pose_tensor = extract_object_pose(state, obj)
                obj_pose = Pose.create_from_pq(
                    obj_pose_tensor[:, 0:3], obj_pose_tensor[:, 3:7]
                )
                obj_pose_rel = robot_body_pose.inv() * obj_pose
                obj_pose_rel_tensor = torch.cat(
                    [obj_pose_rel.p, obj_pose_rel.q], dim=-1
                )  # [batch_size, 7]

            if training:
                # Add noise to object pose during training for NN robustness
                noise_xyz = (
                    torch.randn_like(obj_pose_rel_tensor[:, :3])
                    * CFG.c_room_reach_drift_trans
                    if hasattr(CFG, "c_room_reach_drift_trans")
                    else torch.zeros_like(obj_pose_rel_tensor[:, :3])
                )
                noise_quaternion = torch.zeros_like(obj_pose_rel_tensor[:, 3:7])
                noise = torch.cat([noise_xyz, noise_quaternion], dim=-1)
                obj_pose_rel_tensor = obj_pose_rel_tensor + noise

            node_features[:, i, :] = obj_pose_rel_tensor

        # Create edges: fully connected graph between all objects
        edge_list = []
        edge_features_list = []
        edge_feature_dim = 7  # relative pose: position (3) + quaternion (4)
        assert can_idx is not None, "Can object not found in object_to_node mapping."

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_list.append([i, j])

                    # Compute proper relative pose between objects using SE(3) transformation
                    pos_i = node_features[:, i, 0:3]  # position of object i
                    quat_i = node_features[:, i, 3:7]  # quaternion of object i
                    pos_j = node_features[:, j, 0:3]  # position of object j
                    quat_j = node_features[:, j, 3:7]  # quaternion of object j

                    # Create poses and compute relative transformation
                    pose_i = Pose.create_from_pq(pos_i, quat_i)
                    pose_j = Pose.create_from_pq(pos_j, quat_j)
                    relative_pose = pose_i.inv() * pose_j

                    # Extract relative pose features (position + quaternion)
                    rel_pos = relative_pose.p  # [3]
                    if can_idx in (i, j):
                        # Normalize can rel pos for easier predicate learning.
                        if CFG.clamp_xy_to_radius > 0.0:
                            # Clamp relative position in XY plane to a max radius
                            clamped_rel_pos = clamp_xy_to_radius(
                                rel_pos,
                                CFG.clamp_xy_to_radius,
                            )
                            rel_pos = torch.abs(clamped_rel_pos)
                    rel_quat = relative_pose.q  # [4]

                    edge_feat = torch.cat(
                        [rel_pos, rel_quat], dim=-1
                    )  # 7-dim edge features
                    edge_features_list.append(edge_feat)

        # Convert edge list to tensor format
        if edge_list:
            edge_indices = torch.tensor(edge_list, dtype=torch.long).T  # [2, num_edges]
            edge_features = torch.stack(
                edge_features_list, dim=1
            )  # [batch_size, num_edges, edge_feature_dim]
        else:
            edge_indices = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((b, 0, edge_feature_dim))

        # Create GraphData objects for each batch element
        graph_data_list = []
        for batch_idx in range(b):
            graph_data = GraphData(
                node_features=node_features[batch_idx],  # [num_nodes, node_feature_dim]
                edge_features=edge_features[batch_idx],  # [num_edges, edge_feature_dim]
                edge_indices=edge_indices,  # [2, num_edges] - same for all batch elements
                global_features=None,
                object_to_node=self.perceiver.objects.object_to_node.copy(),
            )
            graph_data_list.append(graph_data)

        return graph_data_list

    def state_to_vec(self, state: Tensor, objects: Sequence[Object]) -> Tensor:
        """Convert batched ClutteredRoom environment state tensor to sub vector
        representation. Mainly used for skill learning.

        Args:
            state: State tensor from ClutteredRoom environment, batched
            objects: Sequence of objects to include in the state vector
        Returns:
            State vector tensor, batched
        """
        # Get objects from the perceiver's object container
        num_nodes = len(objects)
        node_features: List[Tensor] = []
        has_obj = torch.zeros(num_nodes, dtype=torch.bool)
        body_pose_tensor = extract_robot_body_pose(state)
        body_pose = Pose.create_from_pq(
            body_pose_tensor[:, 0:3], body_pose_tensor[:, 3:7]
        )

        # Extract features for each object using existing helper functions
        for obj, _ in self.perceiver.objects.object_to_node.items():
            if obj in objects:
                obj_idx = objects.index(obj)
                if obj.name == "robot":
                    node_features.append(
                        state[:, :22].clone()
                    )  # robot joints + joint vel
                    hand_pose_tensor = extract_robot_hand_pose(state)
                    hand_pose = Pose.create_from_pq(
                        hand_pose_tensor[:, 0:3], hand_pose_tensor[:, 3:7]
                    )
                    rel_pose = body_pose.inv() * hand_pose
                    rel_pose_tensor = torch.cat(
                        [rel_pose.p, rel_pose.q], dim=-1
                    )  # [batch_size, 7]
                    node_features.append(rel_pose_tensor.clone())  # robot hand pose
                else:
                    # For all other objects (bowl, box, goals, chair), extract pose
                    obj_pose_tensor = extract_object_pose(state, obj)
                    obj_pose = Pose.create_from_pq(
                        p=obj_pose_tensor[:, 0:3], q=obj_pose_tensor[:, 3:7]
                    )
                    rel_pose = body_pose.inv() * obj_pose
                    rel_pose_tensor = torch.cat(
                        [rel_pose.p, rel_pose.q], dim=-1
                    )  # [batch_size, 7]
                    node_features.append(rel_pose_tensor.clone())
                has_obj[obj_idx] = True
        # Flatten node features
        assert has_obj.all(), "Some objects not found in the state."
        state_vec = torch.cat(
            node_features, dim=-1
        )  # shape (B, num_nodes * feature_dim)
        return state_vec

    def skill_obs_space(self) -> gym.Space:
        """Get observation and action spaces for skill learning.

        Returns:
            Observation space and action space for skill learning
        """
        # Total observation dimension: robot (22) + hand_pose (7) + 2 objects (14)
        obs_dim = 43
        obs_low = np.array([-np.inf] * obs_dim, dtype=np.float32)
        obs_high = np.array([np.inf] * obs_dim, dtype=np.float32)

        # Set bounds for robot joint positions (first 11 dims)
        # These are typical bounds for robot joints
        obs_low[:11] = np.array(
            [
                -5.0,  # x
                -5.0,  # y
                0.0,  # z (height)
                -2.6179938,  # joint limits
                -3.1415927,
                0.0,
                -2.7925267,
                -1.8325957,
                -2.8797932,
                -0.05,
                -0.05,  # gripper
            ],
            dtype=np.float32,
        )
        obs_high[:11] = np.array(
            [
                0.0,
                0.0,
                0.0,
                3.1415927,
                0.5235988,
                3.1415927,
                2.7925267,
                1.8325957,
                2.8797932,
                0.05,
                0.05,  # gripper
            ],
            dtype=np.float32,
        )

        # Set bounds for joint velocities (dims 11-22)
        obs_low[11:22] = np.array([-2.0] * 11, dtype=np.float32)
        obs_high[11:22] = np.array([2.0] * 11, dtype=np.float32)

        # Set bounds for joint velocities (dims 11-22)
        # All objects should be within 1.5m region around the robot
        obs_low[22:] = np.array([-1.5] * 21, dtype=np.float32)
        obs_high[22:] = np.array([1.5] * 21, dtype=np.float32)

        obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        return obs_space

    def full_state_to_relative_state(
        self, full_state: Tensor, relative_name: str = ""
    ) -> Tensor:
        """Convert full state to relative state.

        Here, relative state means the relative pose between
        the base object (bowl or box) and the obstruction (chair).

        Args:
            full_state: Full state tensor from ClutteredRoom environment, batched
            relative_name: Name of the base object (e.g., "bowl" or "box")

        Returns:
            Relative state tensor, batched.
            Bx1x8, where 1 is the single obstruction (chair),
            8 represents (base_node_id, rel_pose_7d: position(3) + quaternion(4))
        """
        state_graphs = self.state_to_graph(full_state)
        b = full_state.shape[0]
        obstruction_node_ids = []
        obstruction_node_index = []
        obj_node_id = None

        # NOTE: Force obstruction order to be consistent with
        # how observation is created.
        for name in SCENARIO_OBSTRUCTIONS:
            for key_name, obj in (self.perceiver.objects.as_dict()).items():
                if key_name == name:
                    obstruction_node_ids.append(
                        self.perceiver.objects.object_to_node[obj]
                    )
                    obstruction_node_index.append(SCENARIO_OBSTRUCTIONS.index(name))
                if key_name == relative_name:
                    obj_node_id = self.perceiver.objects.object_to_node[obj]

        assert len(obstruction_node_ids) == len(
            CFG.scenario.split(",")
        ), "Obstruction node ids length mismatch."
        assert obj_node_id is not None, f"Base node id not found for {relative_name}."

        # relative states: 2 obstruction (chair)
        # 8-dim = (base_node_id, rel_pose_7d)
        relative_states = torch.zeros((b, 2, 8))

        for batch_idx in range(b):
            graph = state_graphs[batch_idx]
            # Find the edge index from base object to chair
            for idx, node_obstruction in enumerate(obstruction_node_ids):
                edge_idx = None
                # Find the edge index from block to obstruction
                for ei in range(graph.edge_indices.shape[1]):
                    if (
                        graph.edge_indices[0, ei].item() == obj_node_id
                        and graph.edge_indices[1, ei].item() == node_obstruction
                    ):
                        edge_idx = ei
                        break
                if edge_idx is not None:
                    edge_feat = graph.edge_features[edge_idx]
                    obs_idx = obstruction_node_index[idx]
                    relative_states[batch_idx, obs_idx, 0] = obj_node_id
                    relative_states[batch_idx, obs_idx, 1:] = edge_feat.clone()

        return relative_states

    def relative_state_to_full_state(
        self, inital_full_state: Tensor, relative_state: Tensor
    ) -> Tensor:
        """Convert relative state back to full state.

        Here, relative state means the relative pose between
        the base object and the obstruction (chair).

        Args:
            initial_full_state: Initial full state tensor from ClutteredRoom environment, batched
            relative_state: Relative state tensor, batched
                Bx1x8, where 1 is the single obstruction (chair),
                8 represents (base_node_id, rel_pose_7d: position(3) + quaternion(4))
        """
        init_state_graphs = self.state_to_graph(inital_full_state)
        B = inital_full_state.shape[0]
        obstruction_node_ids = []
        obstruction_idx = []

        # NOTE: Force obstruction order to be consistent with
        # how observation is created.
        for name in SCENARIO_OBSTRUCTIONS:
            for key_name, obj in (self.perceiver.objects.as_dict()).items():
                if key_name == name:
                    obstruction_node_ids.append(
                        self.perceiver.objects.object_to_node[obj]
                    )
                    obstruction_idx.append(SCENARIO_OBSTRUCTIONS.index(name))

        extracted_relative_states = torch.zeros((B, len(obstruction_node_ids), 8))
        for num, obstruction_id in enumerate(obstruction_idx):
            assert relative_state[
                :, obstruction_id, 1:
            ].any(), "Relative state for obstruction is empty."
            extracted_relative_states[:, num, :] = relative_state[:, obstruction_id, :]
        obj_node_ids = extracted_relative_states[:, :, 0].long()

        base_obj_poses = torch.zeros((B, len(obstruction_node_ids), 7))
        for b in range(B):
            graph = init_state_graphs[b]
            for obs_idx, _ in enumerate(obstruction_node_ids):
                base_node_id = int(obj_node_ids[b, obs_idx].item())
                base_node_feat = graph.node_features[base_node_id]
                base_obj_poses[b, obs_idx] = base_node_feat.clone()

        robot_body_pose_tensor = extract_robot_body_pose(inital_full_state)
        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3].repeat(len(obstruction_node_ids), 1),
            robot_body_pose_tensor[:, 3:7].repeat(len(obstruction_node_ids), 1),
        )
        base_obj_pose = Pose.create_from_pq(
            base_obj_poses.view(-1, 7)[:, :3], base_obj_poses.view(-1, 7)[:, 3:]
        )
        obstruction_rel_poses = Pose.create_from_pq(
            extracted_relative_states.view(-1, 8)[:, 1:4],
            extracted_relative_states.view(-1, 8)[:, 4:],
        )
        new_obstruction_poses_rel = base_obj_pose * obstruction_rel_poses
        new_obstruction_poses = robot_body_pose * new_obstruction_poses_rel
        new_obs_poses = new_obstruction_poses.raw_pose.view(
            B, len(obstruction_node_ids), 7
        )

        if "1" in CFG.scenario:
            # NOTE: Overwrite can
            new_obs_poses[:, 0, 2] = CFG.c_room_goal_z + CFG.c_room_can_z
            original_q = new_obs_poses[:, 0, 3:].clone()
            original_q = original_q / original_q.norm(dim=-1, keepdim=True)
            xoy_proj = quat_to_yaw_only(original_q)
            new_obs_poses[:, 0, 3:] = xoy_proj

        if "2" in CFG.scenario:
            # NOTE: Overwrite block: its z value is fixed, only yaw is allowed to sample
            # Otherwise it might collect with drawer and makes simulator unstable.
            new_obs_poses[:, 1, 2] = CFG.c_room_chair_z
            original_q = new_obs_poses[:, 1, 3:].clone()
            original_q = original_q / original_q.norm(dim=-1, keepdim=True)
            xoy_proj = quat_to_yaw_only(original_q)
            new_obs_poses[:, 1, 3:] = xoy_proj

        new_obs = _replace_obstruction_pose(inital_full_state, new_obs_poses)
        return new_obs
