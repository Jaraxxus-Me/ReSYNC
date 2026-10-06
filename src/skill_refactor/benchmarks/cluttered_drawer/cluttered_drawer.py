"""ClutteredTable environment planning components."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
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
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer_env import (
    ClutteredDrawerEnv,
)
from skill_refactor.benchmarks.cluttered_drawer.utils import (
    extract_blocking_drawer_pose,
    extract_drawer_q,
    extract_grasp_hammer_pose,
    extract_object_pos,
    extract_object_pose,
    extract_robot_body_pose,
    extract_robot_hand_pose,
    extract_robot_joints,
    extract_supporting_drawer_pose,
    extract_target_hammer_pose,
    is_grasping,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    FINGER_ACTION_INDEX,
    get_frozen_action,
    get_normalize_action_range,
)
from skill_refactor.utils.motion_planning import SpotMotion, WaypointTracker
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

SCENARIO_OBSTRUCTIONS = ["drawer", "block", "wall"]
# Default hand pose in front of the robot body
DEFAULT_BODY_2_HAND_POSE = torch.tensor([0.5405, 0.0, 0.55, 0.7245, 0.0, 0.68928, 0.0])


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


def _replace_obstruction_pose(
    obs: Tensor, new_obs_q: Tensor, new_obs_poses: Tensor
) -> Tensor:
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
            # drawer
            blocking_s, blocking_e = 35, 42
            new_obs[:, blocking_s:blocking_e] = new_obs_poses[:, idx]
            new_obs[:, 42:43] = new_obs_q[:, idx]
        elif idx == 1:
            # block
            block_s, block_e = 65, 72
            new_obs[:, block_s:block_e] = new_obs_poses[:, idx]
        elif idx == 2:
            # wall
            supporting_s, supporting_e = 43, 50
            new_obs[:, supporting_s:supporting_e] = new_obs_poses[:, idx]
            new_obs[:, 50:51] = new_obs_q[:, idx]
        else:
            raise ValueError(f"Unknown obstruction index: {idx}")

        # NOTE: Avoid blocking and supporting drawer collision,
        # we will commit to moving supporting drawer if too close.
        support_drawer_pose_tensor = extract_supporting_drawer_pose(new_obs)
        blocking_drawer_pose_tensor = extract_blocking_drawer_pose(new_obs)
        support_drawer_need_moving = (
            torch.abs(
                support_drawer_pose_tensor[:, 1] - blocking_drawer_pose_tensor[:, 1]
            )
            < CFG.c_drawer_hammer_min_dist_y / 2
        )
        if support_drawer_need_moving.any():
            # First find the hammer's pose
            grasp_hammer_pose = extract_grasp_hammer_pose(
                new_obs[support_drawer_need_moving]
            )
            target_hammer_pose = extract_target_hammer_pose(
                new_obs[support_drawer_need_moving]
            )

            dist_y_grasp = torch.abs(
                grasp_hammer_pose[:, 1]
                - support_drawer_pose_tensor[support_drawer_need_moving, 1]
            )
            dist_y_target = torch.abs(
                target_hammer_pose[:, 1]
                - support_drawer_pose_tensor[support_drawer_need_moving, 1]
            )

            #  Find the source hammer id
            moving_hammer_is_grasp = dist_y_grasp < dist_y_target
            source_hammer_pos_tensor = torch.where(
                moving_hammer_is_grasp.unsqueeze(-1),
                grasp_hammer_pose[:, :3],
                target_hammer_pose[:, :3],
            )
            to_hammer_pos_tensor = torch.where(
                moving_hammer_is_grasp.unsqueeze(-1),
                target_hammer_pose[:, :3],
                grasp_hammer_pose[:, :3],
            )
            support_drawer_pos_moving = support_drawer_pose_tensor[
                support_drawer_need_moving, 0:3
            ].clone()
            # Drawer only has translation w.r.t hammers
            rel_pose = support_drawer_pos_moving - source_hammer_pos_tensor
            new_support_drawer_pos = to_hammer_pos_tensor + rel_pose
            new_obs[support_drawer_need_moving, 43:46] = new_support_drawer_pos
    return new_obs


class ClutteredDrawerTypes(TypeContainer):
    """Container for StickButton types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = Type("robot")
        self.hammer = Type("hammer")
        self.drawer = Type("drawer")
        self.block = Type("block")
        self.wall = Type("wall")

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {self.robot, self.hammer, self.drawer, self.block, self.wall}

    def as_dict(self) -> dict[str, Type]:
        """Convert to dictionary of types."""
        return {
            "robot": self.robot,
            "hammer": self.hammer,
            "drawer": self.drawer,
            "block": self.block,
            "wall": self.wall,
        }


class ClutteredDrawerPredicates(PredicateContainer):
    """Container for StickButton predicates."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize predicates."""
        bodyreadygrasp = Predicate("BodyReadyGrasp", [types.robot, types.hammer])
        bodyreadyplace = Predicate(
            "BodyReadyPlace", [types.robot, types.hammer, types.hammer]
        )
        handreadygrasp = Predicate("HandReadyGrasp", [types.robot, types.hammer])
        handreadyplace = Predicate(
            "HandReadyPlace", [types.robot, types.hammer, types.hammer]
        )
        holding = Predicate("Holding", [types.robot, types.hammer])
        on = Predicate("On", [types.hammer, types.hammer])
        self.predicates = {
            "BodyReadyGrasp": bodyreadygrasp,
            "BodyReadyPlace": bodyreadyplace,
            "HandReadyGrasp": handreadygrasp,
            "HandReadyPlace": handreadyplace,
            "Holding": holding,
            "On": on,
        }


class ClutteredDrawerObjectsSC1(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.drawer = Object("drawer", types.drawer)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.drawer,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        drawer_name = self.drawer.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            drawer_name: self.drawer,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.drawer: 3,
        }


class ClutteredDrawerObjectsSC12(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.drawer = Object("drawer", types.drawer)
        self.block = Object("block", types.block)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.drawer,
            self.block,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        drawer_name = self.drawer.name
        block_name = self.block.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            drawer_name: self.drawer,
            block_name: self.block,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.drawer: 3,
            self.block: 4,
        }


class ClutteredDrawerObjectsSC2(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.block = Object("block", types.block)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.block,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        block_name = self.block.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            block_name: self.block,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.block: 3,
        }


class ClutteredDrawerObjectsSC3(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.wall = Object("wall", types.wall)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.wall,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        wall_name = self.wall.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            wall_name: self.wall,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.wall: 3,
        }


class ClutteredDrawerObjectsSC13(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.drawer = Object("drawer", types.drawer)
        self.wall = Object("wall", types.wall)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.drawer,
            self.wall,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        drawer_name = self.drawer.name
        wall_name = self.wall.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            drawer_name: self.drawer,
            wall_name: self.wall,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.drawer: 3,
            self.wall: 4,
        }


class ClutteredDrawerObjectsSC23(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.block = Object("block", types.block)
        self.wall = Object("wall", types.wall)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.block,
            self.wall,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        block_name = self.block.name
        wall_name = self.wall.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            block_name: self.block,
            wall_name: self.wall,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.block: 3,
            self.wall: 4,
        }


class ClutteredDrawerObjectsSC123(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: ClutteredDrawerTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_hammer = Object("grasp_hammer", types.hammer)
        self.target_hammer = Object("target_hammer", types.hammer)
        self.drawer = Object("drawer", types.drawer)
        self.block = Object("block", types.block)
        self.wall = Object("wall", types.wall)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_hammer,
            self.target_hammer,
            self.drawer,
            self.block,
            self.wall,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_hammer_name = self.grasp_hammer.name
        target_hammer_name = self.target_hammer.name
        drawer_name = self.drawer.name
        block_name = self.block.name
        wall_name = self.wall.name
        return {
            robot_name: self.robot,
            grasp_hammer_name: self.grasp_hammer,
            target_hammer_name: self.target_hammer,
            drawer_name: self.drawer,
            block_name: self.block,
            wall_name: self.wall,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_hammer: 1,
            self.target_hammer: 2,
            self.drawer: 3,
            self.block: 4,
            self.wall: 5,
        }


def _extract_current_objects() -> ObjectContainer:
    if CFG.scenario == "1":
        return ClutteredDrawerObjectsSC1(ClutteredDrawerTypes())
    elif CFG.scenario == "1,2":
        return ClutteredDrawerObjectsSC12(ClutteredDrawerTypes())
    elif CFG.scenario == "2":
        return ClutteredDrawerObjectsSC2(ClutteredDrawerTypes())
    elif CFG.scenario == "3":
        return ClutteredDrawerObjectsSC3(ClutteredDrawerTypes())
    elif CFG.scenario == "1,3":
        return ClutteredDrawerObjectsSC13(ClutteredDrawerTypes())
    elif CFG.scenario == "2,3":
        return ClutteredDrawerObjectsSC23(ClutteredDrawerTypes())
    elif CFG.scenario == "1,2,3":
        return ClutteredDrawerObjectsSC123(ClutteredDrawerTypes())
    raise ValueError(f"Unknown scenario: {CFG.scenario}")


class BaseClutteredDrawerEnvSkill(LiftedOperatorSkill):
    """Base class for ClutteredDrawerEnv environment skills."""

    def __init__(self, env: ClutteredDrawerEnv, operators: set[LiftedOperator]) -> None:
        """Initialize skill."""
        super().__init__()
        self._all_operators = operators
        self._control_mode = env.control_mode
        self._lifted_operator = self.get_lifted_operator()
        self.device = env.device
        self.spot_motion_generator = SpotMotion(
            device=env.device,
        )
        (
            self.normalize_action,
            self.arm_action_low,
            self.arm_action_high,
        ) = get_normalize_action_range(env, self._control_mode)
        self._current_plan: List[Tensor] | None = []
        self.spot_motion_tracker: Optional[WaypointTracker] = None

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
            self.spot_motion_tracker = WaypointTracker(
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


class BodyReachToGraspSkill(BaseClutteredDrawerEnvSkill):
    """Skill for Reach an Object for Grasping."""

    def get_operator_name(self) -> str:
        return "BodyReachToGrasp"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        tgt_obj = objects[1]
        reach_actions: list[Tensor] = []
        # get transformation matrix of the tcp pose, is default batched and on torch
        object_pose_tensor = extract_object_pose(obs, tgt_obj)
        target_p = object_pose_tensor[:, 0:3].clone()
        holding_obj = is_grasping(obs).to(torch.bool)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        from_pose_p = robot_hand_pose_tensor[:, 0:3].clone()
        from_pose_q = robot_hand_pose_tensor[:, 3:7].clone()
        from_pose = Pose.create_from_pq(from_pose_p, from_pose_q)
        from_joints = extract_robot_joints(obs).clone()

        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        default_body2hand = Pose.create_from_pq(
            DEFAULT_BODY_2_HAND_POSE[0:3]
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(self.device),
            DEFAULT_BODY_2_HAND_POSE[3:7]
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(self.device),
        )
        default_hand_pose = robot_body_pose * default_body2hand

        # 1. Move hand to default pose
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                robot_body_pose,
                from_joints,
                from_pose,
                default_hand_pose,
                closing=holding_obj,
                interpolate_steps=30,
            )
        )

        # 2. Then move body to align with target object
        from_joints = reach_actions[-1].clone()
        target_p[:, 2] = robot_body_pose_tensor[:, 2]
        target_p[:, 0] += CFG.c_drawer_body_ready_grasp_dx
        target_body_pose = Pose.create_from_pq(
            target_p,
            q=euler2quat(0.0, 0.0, 0.0),
        )
        reach_actions.extend(
            self.spot_motion_generator.move_body_from_to_pose(
                robot_worldF_curr=robot_body_pose,
                robot_worldF_tgt=target_body_pose,
                curr_joint_positions=from_joints,
                closing=holding_obj,
                interpolate_steps=40,
            )
        )
        return reach_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using ReadyGrasp predicate logic.

        Immediately terminate if there is no current plan or collision is detected.
        """

        assert self._current_plan is not None

        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(collision, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(collision, dtype=torch.bool)
        )
        terminated = not_have_current_plan_tensor | collision
        return terminated


class BodyReachToPlaceSkill(BodyReachToGraspSkill):
    """Skill for Reach an Object for Placing."""

    def get_operator_name(self) -> str:
        return "BodyReachToPlace"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Get kinematic plan for reaching to place.

        Unlike Grasp, this skill won't change the orientation of the robot hand.
        """
        tgt_obj = objects[2]
        reach_actions: list[Tensor] = []
        # get transformation matrix of the tcp pose, is default batched and on torch
        object_pose_tensor = extract_object_pose(obs, tgt_obj)
        target_p = object_pose_tensor[:, 0:3].clone()
        holding_obj = is_grasping(obs).to(torch.bool)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        from_pose_p = robot_hand_pose_tensor[:, 0:3].clone()
        from_pose_q = robot_hand_pose_tensor[:, 3:7].clone()
        from_pose = Pose.create_from_pq(from_pose_p, from_pose_q)
        from_joints = extract_robot_joints(obs).clone()

        robot_body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3], robot_body_pose_tensor[:, 3:7]
        )
        lifting_p = from_pose_p.clone()
        lifting_p[:, 2] += CFG.c_drawer_reachtoplace_lift_hand_z
        lifting_pose = Pose.create_from_pq(lifting_p, from_pose_q)

        # 1. Move hand to default pose
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                robot_body_pose,
                from_joints,
                from_pose,
                lifting_pose,
                closing=holding_obj,
                interpolate_steps=30,
            )
        )

        # 2. Then move body to align with target object
        from_joints = reach_actions[-1].clone()
        target_p[:, 2] = robot_body_pose_tensor[:, 2]
        target_p[:, 0] += CFG.c_drawer_body_ready_grasp_dx
        target_body_pose = Pose.create_from_pq(
            target_p,
            q=euler2quat(0.0, 0.0, 0.0),
        )
        reach_actions.extend(
            self.spot_motion_generator.move_body_from_to_pose(
                robot_worldF_curr=robot_body_pose,
                robot_worldF_tgt=target_body_pose,
                curr_joint_positions=from_joints,
                closing=holding_obj,
                interpolate_steps=40,
            )
        )
        return reach_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using CurrentPlan and Collision logic for this skill.
        """
        assert self._current_plan is not None

        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(collision, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(collision, dtype=torch.bool)
        )
        terminated = not_have_current_plan_tensor | collision
        return terminated


class HandReachToGraspSkill(BaseClutteredDrawerEnvSkill):
    """Skill for Reach an Object for Grasping."""

    def get_operator_name(self) -> str:
        return "HandReachToGrasp"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        tgt_obj = objects[1]
        B = obs.shape[0]
        holding_obj = is_grasping(obs)
        approaching = torch.tensor([0, 0, -1], dtype=torch.float32).to(self.device)
        # get transformation matrix of the tcp pose, is default batched and on torch
        object_pose_tensor = extract_object_pose(obs, tgt_obj)
        object_pose = Pose.create_from_pq(
            p=object_pose_tensor[:, 0:3], q=object_pose_tensor[:, 3:7]
        )
        object_pose_mat = object_pose.to_transformation_matrix()
        object_y_axis = object_pose_mat[:, :3, 1]
        object_y_axis_proj = object_y_axis.clone()
        object_y_axis_proj[..., 2] = 0
        object_y_axis_proj = F.normalize(object_y_axis_proj, dim=-1)
        # project to xoy plane
        object_center = object_pose_tensor[:, 0:3]
        approaching = approaching.unsqueeze(0).repeat(B, 1)
        grasp_pose = self.spot_motion_generator.build_grasp_pose(
            approaching, object_y_axis_proj, object_center
        )

        from_joints = extract_robot_joints(obs).clone()
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        from_pose_p = robot_hand_pose_tensor[:, 0:3].clone()
        from_pose_q = robot_hand_pose_tensor[:, 3:7].clone()
        from_pose = Pose.create_from_pq(from_pose_p, from_pose_q)

        # all envs have the same body pose
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3],
            robot_body_pose_tensor[:, 3:7],
        )
        reach_actions: list[Tensor] = []
        # First lift hand
        lift_pos = grasp_pose.p.clone()
        lift_pos[:, 2] += CFG.c_drawer_reachtograsp_lift_hand_z  # lift hand up by 30cm
        lift_pose = Pose.create_from_pq(lift_pos, grasp_pose.q)
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                from_pose,
                lift_pose,
                closing=holding_obj,
                interpolate_steps=40,
            )
        )
        # Then move to the grasp pose
        from_joints = reach_actions[-1].clone()  # use the last action as starting point
        # and finally move to the grasp pose
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                lift_pose,
                grasp_pose,
                closing=holding_obj,
                interpolate_steps=20,
            )
        )
        return reach_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using ReadyGrasp predicate logic.

        Immediately terminate if there is no current plan or collision is detected.
        """
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_hammer", "target_hammer"]
        obj0_p = extract_object_pos(obs, objects[0])
        obj1_p = extract_object_pos(obs, objects[1])
        distance_xy = torch.norm(
            obj0_p[:, :2] - obj1_p[:, :2],
            dim=-1,
        )
        distance_z = torch.abs(obj0_p[:, 2] - obj1_p[:, 2])
        holding = is_grasping(obs).to(torch.bool)
        distance_ok = (
            (distance_xy < CFG.c_drawer_ready_grasp_xy)
            & (distance_z < CFG.c_drawer_ready_grasp_z)
            & (~holding)
        )  # shape (B, num_pairs)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(distance_ok, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(distance_ok, dtype=torch.bool)
        )
        terminated = distance_ok | not_have_current_plan_tensor
        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        terminated = terminated | collision
        return terminated


class HandReachToPlaceSkill(HandReachToGraspSkill):
    """Skill for Reach an Object for Placing."""

    def get_operator_name(self) -> str:
        return "HandReachToPlace"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        """Get kinematic plan for reaching to place.

        Unlike Grasp, this skill won't change the orientation of the robot hand.
        """
        tgt_obj = objects[2]
        B = obs.shape[0]
        holding_obj = is_grasping(obs)
        approaching = torch.tensor([0, 0, -1], dtype=torch.float32).to(self.device)
        # get transformation matrix of the tcp pose, is default batched and on torch
        object_pose_tensor = extract_object_pose(obs, tgt_obj)
        object_pose = Pose.create_from_pq(
            p=object_pose_tensor[:, 0:3], q=object_pose_tensor[:, 3:7]
        )
        object_pose_mat = object_pose.to_transformation_matrix()
        object_y_axis = object_pose_mat[:, :3, 1]
        object_y_axis_proj = object_y_axis.clone()
        object_y_axis_proj[..., 2] = 0
        object_y_axis_proj = F.normalize(object_y_axis_proj, dim=-1)
        # project to xoy plane
        object_center = object_pose_tensor[:, 0:3]
        approaching = approaching.unsqueeze(0).repeat(B, 1)
        place_pose = self.spot_motion_generator.build_grasp_pose(
            approaching,
            object_y_axis_proj,
            object_center,
            CFG.c_drawer_place_rel_dx,
            CFG.c_drawer_place_rel_dy,
            CFG.c_drawer_place_rel_dz,
        )

        from_joints = extract_robot_joints(obs).clone()
        robot_hand_pose_tensor = extract_robot_hand_pose(obs)
        robot_body_pose_tensor = extract_robot_body_pose(obs)
        from_pose_p = robot_hand_pose_tensor[:, 0:3].clone()
        from_pose_q = robot_hand_pose_tensor[:, 3:7].clone()
        from_pose = Pose.create_from_pq(from_pose_p, from_pose_q)

        # all envs have the same body pose
        body_pose = Pose.create_from_pq(
            robot_body_pose_tensor[:, 0:3],
            robot_body_pose_tensor[:, 3:7],
        )
        reach_actions: list[Tensor] = []
        # First lift hand
        lift_pos = place_pose.p.clone()
        lift_pos[:, 2] += CFG.c_drawer_reachtoplace_lift_hand_z
        lift_pose = Pose.create_from_pq(lift_pos, place_pose.q)
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                from_pose,
                lift_pose,
                closing=holding_obj,
                interpolate_steps=30,
            )
        )

        # Finally move to the place pose
        from_joints = reach_actions[-1].clone()  # use the last action as starting point
        # and finally move to the grasp pose
        reach_actions.extend(
            self.spot_motion_generator.move_hand_from_to_pose(
                body_pose,
                from_joints,
                lift_pose,
                place_pose,
                closing=holding_obj,
                interpolate_steps=20,
            )
        )
        return reach_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using ReadyPlace predicate logic.
        """
        assert len(objects) == 3
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_hammer", "target_hammer"]
        assert objects[2].name in ["grasp_hammer", "target_hammer"]
        obj1_p = extract_object_pos(obs, objects[1])
        obj2_p = extract_object_pos(obs, objects[2])
        distance_xy = torch.norm(
            obj1_p[:, :2] - obj2_p[:, :2],
            dim=-1,
        )
        distance_z = obj1_p[:, 2] - obj2_p[:, 2]
        holding = is_grasping(obs).to(torch.bool)
        distance_ok = (
            (distance_xy < CFG.c_drawer_goal_radius)
            & (distance_z > 0)
            & (distance_z < CFG.c_drawer_place_height_tol)
            & holding
        )
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(distance_ok, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(distance_ok, dtype=torch.bool)
        )
        terminated = distance_ok | not_have_current_plan_tensor
        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        terminated = terminated | collision
        return terminated


class GraspSkill(BaseClutteredDrawerEnvSkill):
    """Skill for grasping the hammer."""

    freeze_action: Any = None  # type: ignore[assignment]
    grasped: Any = None  # type: ignore[assignment]

    def get_operator_name(self) -> str:
        return "Grasp"

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Get action given objects and observation.

        Grasp action will adapt the action plan based on the current grasp information
        and observation.
        """
        if not self._current_plan:
            self.grasped = torch.zeros(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
            self._current_plan = self._get_kinematic_plan_given_objects(objects, obs)
            self.freeze_action = get_frozen_action(
                self._current_plan[0],
                arm_action_low=self.arm_action_low,
                arm_action_high=self.arm_action_high,
                normalize_action=self.normalize_action,
                control_mode=self._control_mode,
            )
            self.spot_motion_tracker = WaypointTracker(
                plan=self._current_plan,
                normalize_action=self.normalize_action,
                arm_action_low=self.arm_action_low,
                arm_action_high=self.arm_action_high,
                waypoint_threshold=CFG.waypoint_threshold,
                angular_threshold=CFG.angular_threshold,
                device=self.device,
            )

        # save the action that made them grasp
        # If a plan exists, execute the next action in the plan
        # Type narrowing for mypy
        assert self.spot_motion_tracker is not None, "spot_motion_tracker must be set"
        skill_action = self.spot_motion_tracker.compute_delta_actions(
            extract_robot_joints(obs)
        )
        self._current_plan = self.spot_motion_tracker.plan

        curr_grasp = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        newly = curr_grasp & ~self.grasped  # those that flipped from False→True

        exec_action = torch.where(
            self.grasped.unsqueeze(-1),  # shape (NUM_ENVS, 1)
            self.freeze_action,  # shape (NUM_ENVS, action_dim)
            skill_action,  # shape (NUM_ENVS, action_dim)
        )

        # We will actually use the frozen action in the next step even
        # if the hammer is already grasped now.
        assert self.freeze_action is not None
        assert self.grasped is not None
        self.freeze_action[newly] = exec_action[newly]
        # update the overall mask
        self.grasped |= curr_grasp

        # finally, for grasped, hack the gripper
        curr_gripper = extract_robot_joints(obs)[
            :, FINGER_ACTION_INDEX
        ].clone()  # gripper position
        exec_action[self.grasped, -1] = curr_gripper[self.grasped] + 0.001
        return exec_action

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        del objects
        from_joints = extract_robot_joints(obs).clone()  # get initial joint positions
        grasp_actions = self.spot_motion_generator.close_gripper(from_joints, t=30)
        return grasp_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when the block is grasped."""
        assert len(objects) == 2
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_hammer", "target_hammer"]
        holding = is_grasping(obs).to(torch.bool)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(holding, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(holding, dtype=torch.bool)
        )
        terminated = holding | not_have_current_plan_tensor
        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        terminated = terminated | collision
        return terminated


class PlaceSkill(BaseClutteredDrawerEnvSkill):
    """Skill for placing the hammar."""

    def get_operator_name(self) -> str:
        return "Place"

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Get action given objects and observation.

        Grasp action will adapt the action plan based on the current grasp information
        and observation.
        """
        if not self._current_plan:
            self.freeze_action = torch.zeros_like(
                extract_robot_joints(obs)
            )  # shape (B, action_dim)
            self.grasped = torch.zeros(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
            self._current_plan = self._get_kinematic_plan_given_objects(objects, obs)
            self.spot_motion_tracker = WaypointTracker(
                plan=self._current_plan,
                normalize_action=self.normalize_action,
                arm_action_low=self.arm_action_low,
                arm_action_high=self.arm_action_high,
                waypoint_threshold=CFG.waypoint_threshold,
                angular_threshold=CFG.angular_threshold,
                device=self.device,
            )

        # Type narrowing for mypy
        assert self.spot_motion_tracker is not None, "spot_motion_tracker must be set"
        action = self.spot_motion_tracker.compute_delta_actions(
            extract_robot_joints(obs)
        )
        self._current_plan = self.spot_motion_tracker.plan
        return action

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        obs: Tensor,
    ) -> List[Tensor]:
        del objects
        from_joints = extract_robot_joints(obs).clone()
        place_actions = self.spot_motion_generator.open_gripper(from_joints, t=20)
        return place_actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when the placing is done.

        Using On predicate logic.
        """
        assert len(objects) == 3
        assert objects[1].name in ["grasp_hammer", "target_hammer"]
        assert objects[2].name in ["grasp_hammer", "target_hammer"]
        assert objects[1] != objects[2]
        is_on_tensor = obs[:, -2].to(torch.bool)  # shape (B,)
        holding = is_grasping(obs).to(torch.bool)
        is_on = is_on_tensor & (~holding)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(is_on, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(is_on, dtype=torch.bool)
        )
        terminated = is_on | not_have_current_plan_tensor
        collision = obs[:, -1].to(torch.bool)  # shape (B,)
        terminated = terminated | collision
        return terminated


class ClutteredTablePerceiver(Perceiver):
    """Perceiver for ClutteredTable environment."""

    def __init__(
        self, predicates: ClutteredDrawerPredicates, types: ClutteredDrawerTypes
    ) -> None:
        """Initialize with required types."""
        self.predicates_container = predicates
        self._types = types
        self.predicate_interpreters = {
            predicates["BodyReadyGrasp"]: self._interpret_bodyreadygrasp,
            predicates["BodyReadyPlace"]: self._interpret_bodyreadyplace,
            predicates["HandReadyGrasp"]: self._interpret_handreadygrasp,
            predicates["HandReadyPlace"]: self._interpret_handreadyplace,
            predicates["Holding"]: self._interpret_holding,
            predicates["On"]: self._interpret_on,
        }

    def reset(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        # self.objects remains the same from any task
        assert (
            obs.shape[0] == 1
        ), "Expected batch size of 1 for observation for perceiver reset."
        # Note that for now we assume all the tasks have the same initial atoms (so the same task plan).
        atoms = self._get_atoms(obs, info)
        latest_objects = _extract_current_objects()
        grasp_hammer = latest_objects.as_dict()["grasp_hammer"]
        place_hammer = latest_objects.as_dict()["target_hammer"]
        goal = {self.predicates_container["On"]([grasp_hammer, place_hammer])}
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
        """Convert a batch of observations into a List of GroundAtom-sets, one per batch
        element.

        Args:
            obs: Tensor of shape (B, …)

        Returns:
            List of length B, where each entry is the set of GroundAtoms for that obs.
        """
        B = obs.shape[0]
        latest_objects = _extract_current_objects()
        # start with an empty set for each batch element
        atoms_List: List[set[GroundAtom]] = [set() for _ in range(B)]

        # each interpreter now returns a List[Set[GroundAtom]] of length B
        desired_predicates = list(self.predicates_container.predicates.keys())
        if info is not None and "desired_predicates" in info:
            # If info is provided, we can filter the predicates to only those desired
            desired_predicates = [
                p
                for p in info["desired_predicates"]
                if p in self.predicates_container.predicates
            ]
        for predicate, interpret_fn in self.predicate_interpreters.items():
            if predicate.name not in desired_predicates:
                # If the predicate is not desired, skip it
                continue
            # First get the arguments for the predicate
            # which will extend the number of observation inputs

            # Now we assume all the batch has the same number of objects
            # meaning that we can use the same arguments for all batches.
            input_args: List[List[Object]] = []
            for args in get_object_combinations(
                latest_objects.as_set(),
                predicate.types,
                skip_self=True,
            ):
                input_args.append(args)
            # args: List[List[Object]] of length num_grounding
            # stacked_res: Tensor of shape (B, num_grounding)
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

    def _interpret_bodyreadygrasp(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        obj_positions = []
        hand_dz = []
        hand_x_axis_degree = []
        for obj_pair in objects:
            assert obj_pair[0].name == "robot"
            obj0_p = extract_robot_body_pose(obs)[:, 0:3]
            obj0_hand_dz = extract_robot_hand_pose(obs)[:, 2] - obj0_p[:, 2]
            hand_pose_q = extract_robot_hand_pose(obs)[:, 3:7]
            hand_rot_mat = quaternion_to_matrix(hand_pose_q)
            hand_x_axis = hand_rot_mat[:, :, 0]
            negative_z_axis = torch.zeros_like(hand_x_axis)
            negative_z_axis[:, 2] = -1.0
            cos_angle = torch.clamp(
                torch.sum(hand_x_axis * negative_z_axis, dim=1), -1.0, 1.0
            )
            angle = torch.abs(torch.acos(cos_angle))  # in radians
            hand_x_axis_degree.append(angle)
            obj1_p = extract_object_pos(obs, obj_pair[1])
            obj_positions.append(torch.stack([obj0_p, obj1_p], dim=1))
            hand_dz.append(obj0_hand_dz)
        # NOTE: We don't want Place to add BodyReadyGrasp
        # Since the two hammer are colliding now.
        is_on = obs[:, -2].to(torch.bool)
        is_on = is_on.unsqueeze(1).repeat(1, len(objects))  # shape (B, num_pairs)
        hand_dz_stacked = torch.stack(hand_dz, dim=1)  # shape (B, num_pairs)
        hand_x_axis_degree_stacked = torch.stack(
            hand_x_axis_degree, dim=1
        )  # shape (B, num_pairs)
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 3)
        distance_x = torch.abs(
            torch.abs(
                obj_position_stacked[:, :, 0, 0] - obj_position_stacked[:, :, 1, 0]
            )
            - abs(CFG.c_drawer_body_ready_grasp_dx)
        )
        distance_y = torch.abs(
            obj_position_stacked[:, :, 0, 1] - obj_position_stacked[:, :, 1, 1]
        )
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(
            1, obj_position_stacked.shape[1]
        )  # shape (B, num_pairs)
        distance_ok = (
            (distance_x < CFG.c_drawer_body_predicate_transition_thresh)
            & (distance_y < CFG.c_drawer_body_predicate_transition_thresh)
            & (hand_dz_stacked > CFG.c_drawer_body_ready_grasp_hand_dz_thresh)
            & (
                hand_x_axis_degree_stacked
                < CFG.c_drawer_body_ready_grasp_hand_x_axis_degree
            )
            & (~holding)
            & (~is_on)
        )  # shape (B, num_pairs)
        return distance_ok

    def _interpret_bodyreadyplace(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        obj_positions02 = []
        obj_positions12 = []
        obj1_is_grasping_hammer = []
        for obj_pair in objects:
            assert obj_pair[0].name == "robot"
            obj0_p = extract_robot_body_pose(obs)[:, 0:3]
            obj1_p = extract_object_pos(obs, obj_pair[1])
            obj2_p = extract_object_pos(obs, obj_pair[2])
            obj_positions02.append(torch.stack([obj0_p, obj2_p], dim=1))
            obj_positions12.append(torch.stack([obj1_p, obj2_p], dim=1))
            if obj_pair[1].name == "grasp_hammer":
                obj1_is_grasping_hammer.append(
                    torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
                )
            else:
                obj1_is_grasping_hammer.append(
                    torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
                )
        obj1_is_grasping_hammer_tensor = torch.stack(
            obj1_is_grasping_hammer, dim=1
        )  # shape (B, num_pairs)
        obj_position02_stacked = torch.stack(
            obj_positions02, dim=1
        )  # shape (B, num_pairs, 2, 3)
        obj_positions12_stacked = torch.stack(
            obj_positions12, dim=1
        )  # shape (B, num_pairs, 2, 3)
        distance_x = torch.abs(
            torch.abs(
                obj_position02_stacked[:, :, 0, 0] - obj_position02_stacked[:, :, 1, 0]
            )
            - abs(CFG.c_drawer_body_ready_grasp_dx)
        )
        distance_y = torch.abs(
            obj_position02_stacked[:, :, 0, 1] - obj_position02_stacked[:, :, 1, 1]
        )
        distance_xy = torch.norm(
            obj_positions12_stacked[:, :, 0, :2] - obj_positions12_stacked[:, :, 1, :2],
            dim=-1,
        )
        distance_z = (
            obj_positions12_stacked[:, :, 0, 2] - obj_positions12_stacked[:, :, 1, 2]
        )
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(
            1, obj_position02_stacked.shape[1]
        )  # shape (B, num_pairs)
        distance_ok = (
            (distance_x < CFG.c_drawer_body_predicate_transition_thresh)
            & (distance_y < CFG.c_drawer_body_predicate_transition_thresh)
            & (distance_xy < CFG.c_drawer_goal_radius * 4)
            & (distance_z > 0)
            & (holding)
            & (obj1_is_grasping_hammer_tensor)
        )  # shape (B, num_pairs)
        return distance_ok

    def _interpret_handreadygrasp(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret ReadyGrasp predicate."""
        obj_positions = []
        for obj_pair in objects:
            obj0_p = extract_object_pos(obs, obj_pair[0])
            obj1_p = extract_object_pos(obs, obj_pair[1])
            obj_positions.append(torch.stack([obj0_p, obj1_p], dim=1))
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 3)
        distance_xy = torch.norm(
            obj_position_stacked[:, :, 0, :2] - obj_position_stacked[:, :, 1, :2],
            dim=-1,
        )
        distance_z = torch.abs(
            obj_position_stacked[:, :, 0, 2] - obj_position_stacked[:, :, 1, 2]
        )
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(
            1, obj_position_stacked.shape[1]
        )  # shape (B, num_pairs)
        distance_ok = (
            (distance_xy < CFG.c_drawer_ready_grasp_xy)
            & (distance_z < CFG.c_drawer_ready_grasp_z)
            & (~holding)
        )  # shape (B, num_pairs)
        return distance_ok

    def _interpret_handreadyplace(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret ReadyPlace predicate."""
        obj_positions = []
        for obj_pair in objects:
            obj1_p = extract_object_pos(obs, obj_pair[1])
            obj2_p = extract_object_pos(obs, obj_pair[2])
            obj_positions.append(torch.stack([obj1_p, obj2_p], dim=1))
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 3)
        distance_xy = torch.norm(
            obj_position_stacked[:, :, 0, :2] - obj_position_stacked[:, :, 1, :2],
            dim=-1,
        )
        distance_z = obj_position_stacked[:, :, 0, 2] - obj_position_stacked[:, :, 1, 2]
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(
            1, obj_position_stacked.shape[1]
        )  # shape (B, num_pairs)
        distance_ok = (
            (distance_xy < CFG.c_drawer_goal_radius)
            & (distance_z > 0)
            & (distance_z < CFG.c_drawer_place_height_tol)
            & holding
        )
        return distance_ok

    def _interpret_on(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret On predicate."""
        diff_obj = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for obj_pair in objects:
            if (obj_pair[0].name != obj_pair[1].name) and (
                obj_pair[0].name == "grasp_hammer"
            ):
                # If both objects are the same, we cannot compute distance
                diff_obj[:, objects.index(obj_pair)] = True
        is_on = obs[:, -2].to(torch.bool)
        is_on = is_on.unsqueeze(1).repeat(1, len(objects))  # shape (B, num_pairs)
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(1, len(objects))  # shape (B, num_pairs)

        is_on = is_on & (~holding) & diff_obj
        return is_on

    def _interpret_holding(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Holding predicate."""
        holding = is_grasping(obs).to(torch.bool)  # bool tensor (NUM_ENVS,)
        holding_obj = []
        for obj in objects:
            if obj[1].name == "grasp_hammer":
                holding_obj.append(holding)
            else:
                # NOTE: An assumption that there is only one graspable object
                holding_obj.append(
                    torch.zeros_like(holding, dtype=torch.bool, device=holding.device)
                )
        return torch.stack(holding_obj, dim=1)  # shape (B, num_pairs)


class ClutteredDrawerRLTAMPSystem(BaseRLTAMPSystem):
    """Base TAMP system for 2D blocks graph-based environment."""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ClutteredDrawer TAMP system."""
        self._render_mode = render_mode
        self.env_kwargs = {
            "obs_mode": "state",
            "render_mode": render_mode,
            "sim_backend": "gpu",
            "control_mode": CFG.control_mode,
            "reward_mode": "sparse",
        }
        self.env_name = "ClutteredDrawer-v1"
        super().__init__(
            planning_components, name="ClutteredDrawerRLTAMPSystem", seed=seed  # type: ignore
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        envs = gym.make(
            self.env_name,
            num_envs=CFG.num_envs,
            reconfiguration_freq=None,
            **self.env_kwargs,
        )
        return envs

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "ClutteredDrawer-domain"

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
        """Create planning components for graph-based StickButton system."""
        types_container = ClutteredDrawerTypes()

        predicates = ClutteredDrawerPredicates(types_container)
        BodyReadyGrasp = predicates["BodyReadyGrasp"]
        BodyReadyPlace = predicates["BodyReadyPlace"]
        HandReadyGrasp = predicates["HandReadyGrasp"]
        HandReadyPlace = predicates["HandReadyPlace"]
        On = predicates["On"]
        Holding = predicates["Holding"]

        perceiver = ClutteredTablePerceiver(predicates, types_container)

        robot = Variable("?robot", types_container.robot)
        grasp_hammer = Variable("?grasp_hammer", types_container.hammer)
        target_hammer = Variable("?target_hammer", types_container.hammer)

        operators = {
            LiftedOperator(
                "BodyReachToGrasp",
                [robot, grasp_hammer],
                preconditions=set(),
                add_effects={
                    BodyReadyGrasp([robot, grasp_hammer]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "BodyReachToPlace",
                [robot, grasp_hammer, target_hammer],
                preconditions={
                    Holding([robot, grasp_hammer]),
                },
                add_effects={
                    BodyReadyPlace([robot, grasp_hammer, target_hammer]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "HandReachToGrasp",
                [robot, grasp_hammer],
                preconditions={
                    BodyReadyGrasp([robot, grasp_hammer]),
                },
                add_effects={
                    HandReadyGrasp([robot, grasp_hammer]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "HandReachToPlace",
                [robot, grasp_hammer, target_hammer],
                preconditions={
                    Holding([robot, grasp_hammer]),
                    BodyReadyPlace([robot, grasp_hammer, target_hammer]),
                },
                add_effects={
                    HandReadyPlace([robot, grasp_hammer, target_hammer]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "Grasp",
                [robot, grasp_hammer],
                preconditions={
                    HandReadyGrasp([robot, grasp_hammer]),
                    BodyReadyGrasp([robot, grasp_hammer]),
                },
                add_effects={
                    Holding([robot, grasp_hammer]),
                },
                delete_effects={
                    HandReadyGrasp([robot, grasp_hammer]),
                    BodyReadyGrasp([robot, grasp_hammer]),
                },
            ),
            LiftedOperator(
                "Place",
                [robot, grasp_hammer, target_hammer],
                preconditions={
                    HandReadyPlace([robot, grasp_hammer, target_hammer]),
                    BodyReadyPlace([robot, grasp_hammer, target_hammer]),
                    Holding([robot, grasp_hammer]),
                },
                add_effects={
                    On([grasp_hammer, target_hammer]),
                },
                delete_effects={
                    HandReadyPlace([robot, grasp_hammer, target_hammer]),
                    BodyReadyPlace([robot, grasp_hammer, target_hammer]),
                    Holding([robot, grasp_hammer]),
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
    ) -> ClutteredDrawerRLTAMPSystem:
        """Factory method for creating system with default components."""
        assert (
            not CFG.delta_finger_control
        ), "Delta finger control not supported in ClutteredDrawer."
        assert (
            CFG.dreaming_noise_base_var == 0.0
        ), "Dreaming in ClutteredDrawer should have 0.0 base noise."
        planning_components = cls._create_planning_components()
        system = cls(
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        assert isinstance(system.env.unwrapped, ClutteredDrawerEnv)
        skills = {
            BodyReachToGraspSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            BodyReachToPlaceSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            HandReachToGraspSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            HandReachToPlaceSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            GraspSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
            PlaceSkill(system.env.unwrapped, system.components.operators),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)  # type: ignore
        return system

    def state_to_graph(self, state: Tensor, training: bool = False) -> List[GraphData]:
        """Convert Batched ClutteredTable environment state tensor to graph
        representation.

        Args:
            state: State tensor from ClutteredTable environment, batched
            training: Whether in training mode (unused in this implementation)

        Returns:
            List of GraphData with nodes representing objects and edges representing relationships
        """
        # Get objects from the perceiver's object container
        b = state.shape[0]
        num_nodes = len(self.perceiver.objects.object_to_node.keys())

        # Node features: finger (1) + drawer_q (1) + position (3) + quaternion rotation (4)
        # = 9 dimensions
        node_feature_dim = 9
        node_features = torch.zeros(b, num_nodes, node_feature_dim)

        # Extract features for each object using existing helper functions
        robot_node_idx = None
        drawer_node_idx = []
        for obj, i in self.perceiver.objects.object_to_node.items():
            # Extract object pose (position + rotation)
            obj_pose = extract_object_pose(state, obj)
            if training:
                # Add noise to object pose during training for NN robustness
                noise_xyz = (
                    torch.randn_like(obj_pose[:, :3])
                    * CFG.cluttered_drawer_reach_drift_trans
                )
                # random eular angle noise
                # all zero for now
                noise_quanternion = torch.zeros_like(obj_pose[:, 3:7])
                # noise_eular = (
                #     torch.randn_like(obj_pose[:, 0:3]) * CFG.blocked2d_reach_drift_rot
                # )
                # noise_quanternion = matrix_to_quaternion(
                #     euler_angles_to_matrix(noise_eular, convention="XYZ")
                # )
                # original_euler = matri
                noise = torch.cat([noise_xyz, noise_quanternion], dim=-1)
                obj_pose += noise
            node_features[:, i, 2:] = obj_pose
            if obj.name == "robot":
                robot_node_idx = i
                node_features[:, i, 0] = is_grasping(state).cpu()  # finger closed
            elif obj.name in ("drawer", "wall"):
                drawer_node_idx.append(i)
                drawer_q = extract_drawer_q(state, obj)
                node_features[:, i, 1:2] = drawer_q.cpu()  # drawer joint position

        # Create edges: fully connected graph between all objects
        edge_list = []
        edge_features_list = []
        edge_feature_dim = 9  # relative pose: finger (1) + drawer_q (1) + position (3) + quaternion (4)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_list.append([i, j])

                    # Compute proper relative pose between objects using SE(3) transformation
                    pos_i = node_features[:, i, 2:5]  # position of object i
                    quat_i = node_features[:, i, 5:9]  # quaternion of object i
                    pos_j = node_features[:, j, 2:5]  # position of object j
                    quat_j = node_features[:, j, 5:9]  # quaternion of object j

                    # Create poses and compute relative transformation
                    if (i not in drawer_node_idx) and (j not in drawer_node_idx):
                        pose_i = Pose.create_from_pq(pos_i, quat_i)
                        pose_j = Pose.create_from_pq(pos_j, quat_j)
                        relative_pose = pose_i.inv() * pose_j
                    else:
                        # Drawer always has 0.0 yaw angle, so the relative pose
                        # is only translation in world frame
                        rel_p = pos_j - pos_i
                        if i in drawer_node_idx:
                            rel_q = quat_i
                        else:
                            rel_q = quat_j
                        relative_pose = Pose.create_from_pq(p=rel_p, q=rel_q)

                    # Extract relative pose features (position + quaternion)
                    rel_pos = relative_pose.p  # [3]
                    rel_quat = relative_pose.q  # [4]
                    grasping = torch.zeros(
                        (b, 1), dtype=rel_pos.dtype, device=rel_pos.device
                    )
                    drawer_q = torch.zeros(
                        (b, 1), dtype=rel_pos.dtype, device=rel_pos.device
                    )

                    if i == robot_node_idx:
                        grasping = is_grasping(state).cpu().unsqueeze(-1)  # [B, 1]

                    if j in drawer_node_idx:
                        drawer_q_actual = (
                            node_features[:, j, 1].cpu().unsqueeze(-1)
                        )  # [B, 1]
                        rel_dy = rel_pos[:, 1:2]
                        matched = torch.abs(rel_dy) < CFG.c_drawer_matched_dy
                        drawer_q = drawer_q_actual * matched.to(drawer_q_actual.dtype)

                    edge_feat = torch.cat(
                        [grasping, drawer_q, rel_pos, rel_quat], dim=-1
                    )  # 8-dim edge features
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

    def full_state_to_relative_state(
        self, full_state: Tensor, relative_name: str = ""
    ) -> Tensor:
        """Convert full state to relative state.

        Here, relative state means the relative pose between
        the obstructions and the blocks.

        Args:
            full_state: Full state tensor from BlockedStacking environment, batched

        Returns:
            Relative state tensor, batched.
            BxMx5, where M is number of obstructions, 1 is the base object (grasp block),
            5 represents (base_node_id, dx, dy, cos(theta), sin(theta))
        """

        state_graphs = self.state_to_graph(full_state)
        b = full_state.shape[0]
        obstruction_node_ids = []
        obstruction_node_index = []
        hammer_node_id = None

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
                    hammer_node_id = self.perceiver.objects.object_to_node[obj]

        assert hammer_node_id is not None, "Block node id not found."
        assert len(obstruction_node_ids) == len(
            CFG.scenario.split(",")
        ), "Obstruction node ids length mismatch."
        # NOTE: relative states will always be 3 obstructions, even if fewer obstructions exist
        # we will pad with zeros for missing obstructions by default.
        # 9-dim = (id, qpos, rel_pose_3d)
        relative_states = torch.zeros((b, 3, 9))
        for batch_idx in range(b):
            graph = state_graphs[batch_idx]
            for idx, node_obstruction in enumerate(obstruction_node_ids):
                edge_idx = None
                # Find the edge index from block to obstruction
                for ei in range(graph.edge_indices.shape[1]):
                    if (
                        graph.edge_indices[0, ei].item() == hammer_node_id
                        and graph.edge_indices[1, ei].item() == node_obstruction
                    ):
                        edge_idx = ei
                        break
                if edge_idx is not None:
                    edge_feat = graph.edge_features[edge_idx]
                    obs_idx = obstruction_node_index[idx]
                    relative_states[batch_idx, obs_idx, 0] = hammer_node_id
                    relative_states[batch_idx, obs_idx, 1] = graph.node_features[
                        node_obstruction, 1
                    ]  # qpos
                    relative_states[batch_idx, obs_idx, 2:] = edge_feat[2:]  # rel pose

        return relative_states

    def relative_state_to_full_state(
        self, inital_full_state: Tensor, relative_state: Tensor
    ) -> Tensor:
        """Convert relative state back to full state.

        Here, relative state means the relative pose between
        the obstructions and the blocks.

        Args:
            initial_full_state: Initial full state tensor from BlockedStacking environment, batched
            relative_state: Relative state tensor, batched, as defined above.
            BxMx5, where M is number of obstructions, N is number of blocks,
            5 represents (base_node_id, dx, dy, cos(theta), sin(theta))
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

        extracted_relative_states = torch.zeros((B, len(obstruction_node_ids), 9))
        for num, obstruction_id in enumerate(obstruction_idx):
            assert relative_state[
                :, obstruction_id, 1:
            ].any(), "Relative state for obstruction is empty."
            extracted_relative_states[:, num, :] = relative_state[:, obstruction_id, :]
        hammer_node_ids = extracted_relative_states[:, :, 0].long()

        base_obj_poses = torch.zeros((B, len(obstruction_node_ids), 7))
        for b in range(B):
            graph = init_state_graphs[b]
            for obs_idx, _ in enumerate(obstruction_node_ids):
                base_node_id = int(hammer_node_ids[b, obs_idx].item())
                base_node_feat = graph.node_features[base_node_id]
                base_obj_poses[b, obs_idx] = base_node_feat[2:]

        base_obj_pose = Pose.create_from_pq(
            base_obj_poses.view(-1, 7)[:, :3], base_obj_poses.view(-1, 7)[:, 3:]
        )
        obstruction_rel_poses = Pose.create_from_pq(
            extracted_relative_states.view(-1, 9)[:, 2:5],
            extracted_relative_states.view(-1, 9)[:, 5:],
        )
        new_obstruction_poses = base_obj_pose * obstruction_rel_poses
        new_obs_poses = new_obstruction_poses.raw_pose.view(
            B, len(obstruction_node_ids), 7
        )

        if "1" in CFG.scenario:
            # NOTE: Overwrite drawer as its relative pose is not SE3, but Vec3 in world frame.
            drawer_pose_p = (
                base_obj_poses[:, 0, :3] + extracted_relative_states[:, 0, 2:5]
            )
            drawer_pose_q = extracted_relative_states[:, 0, 5:]
            new_obs_poses[:, 0, :2] = drawer_pose_p[:, :2]
            new_obs_poses[:, 0, 2] = CFG.c_drawer_drawer_height
            new_obs_poses[:, 0, 3:] = drawer_pose_q

        if "2" in CFG.scenario:
            # NOTE: Overwrite block: its z value is fixed, only yaw is allowed to sample
            # Otherwise it might collect with drawer and makes simulator unstable.
            new_obs_poses[:, 1, 2] = CFG.c_drawer_hammer_height + CFG.c_drawer_block_dz
            # This is a litte hacky, we know the y-qlimit of the block is [-0.2, 0.2]
            min_y = base_obj_poses[:, 1, 1] - 0.2
            max_y = base_obj_poses[:, 1, 1] + 0.2
            new_obs_poses[:, 1, 1] = torch.clamp(
                new_obs_poses[:, 1, 1], min=min_y, max=max_y
            )
            original_q = new_obs_poses[:, 1, 3:].clone()
            original_q = original_q / original_q.norm(dim=-1, keepdim=True)
            xoy_proj = quat_to_yaw_only(original_q)
            new_obs_poses[:, 1, 3:] = xoy_proj

        if "3" in CFG.scenario:
            # NOTE: Overwrite wall as its relative pose is not SE3, but Vec3 in world frame.
            wall_pose_p = (
                base_obj_poses[:, 2, :3] + extracted_relative_states[:, 2, 2:5]
            )
            wall_pose_q = extracted_relative_states[:, 2, 5:]

            new_obs_poses[:, 2, :2] = wall_pose_p[:, :2]
            new_obs_poses[:, 2, 2] = CFG.c_drawer_drawer_height
            new_obs_poses[:, 2, 3:] = wall_pose_q

        new_obs_q = extracted_relative_states[:, :, 1].view(
            B, len(obstruction_node_ids), 1
        )

        new_obs = _replace_obstruction_pose(inital_full_state, new_obs_q, new_obs_poses)
        return new_obs

    def skill_obs_space(self) -> gym.Space:
        """Get observation and action spaces for skill learning.

        Returns:
            Observation space and action space for skill learning
        """
        obs_low = np.array([-np.inf] * 49, dtype=np.float32)
        obs_high = np.array([np.inf] * 49, dtype=np.float32)
        obs_low[:10] = np.array(
            [
                -2.0,
                -2.0,
                0.0,
                -2.6179938,
                -3.1415927,
                0.0,
                -2.7925267,
                -1.8325957,
                -2.8797932,
                -1.57,
            ],
            dtype=np.float32,
        )
        obs_high[:10] = np.array(
            [
                2.0,
                2.0,
                2.0,
                3.1415927,
                0.5235988,
                3.1415927,
                2.7925267,
                1.8325957,
                2.8797932,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_low[10:20] = np.array([-2.0] * 10, dtype=np.float32)
        obs_high[10:20] = np.array([2.0] * 10, dtype=np.float32)
        obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        return obs_space

    def state_to_vec(self, state: Tensor, objects: Sequence[Object]) -> Tensor:
        """Convert Batched ClutteredDrawer environment state tensor to sub vector
        representation. Mainly used for skill learning.

        Args:
            state: State tensor from ClutteredDrawer environment, batched
            objects: Sequence of objects to include in the state vector
        Returns:
            State vector tensor, batched
        """
        # Get objects from the perceiver's object container
        num_nodes = len(objects)
        node_features: List[Tensor] = []
        has_obj = torch.zeros(num_nodes, dtype=torch.bool)

        # Extract features for each object using existing helper functions
        for obj, _ in self.perceiver.objects.object_to_node.items():
            if obj in objects:
                obj_idx = objects.index(obj)
                if obj.name == "robot":
                    node_features.append(
                        state[:, :20].clone()
                    )  # robot joints + joint vel
                    node_features.append(
                        extract_robot_hand_pose(state).clone()
                    )  # robot hand pose
                    node_features.append(
                        extract_robot_body_pose(state).clone()
                    )  # robot hand pose
                elif "drawer" in obj.name:
                    feat = torch.cat(
                        [
                            extract_object_pose(state, obj),  # drawer pose
                            extract_drawer_q(state, obj),  # drawer joint pos
                        ],
                        dim=-1,
                    )
                    node_features.append(feat)
                elif "hammer" in obj.name:
                    node_features.append(extract_object_pose(state, obj))
                else:
                    feat = torch.cat(
                        [
                            extract_object_pose(state, obj),  # object pose
                            torch.zeros(
                                (state.shape[0], 1),
                                dtype=state.dtype,
                                device=state.device,
                            ),  # placeholder for drawer joint pos
                        ],
                        dim=-1,
                    )
                    node_features.append(feat)
                has_obj[obj_idx] = True
        # Flatten node features
        assert has_obj.all(), "Some objects not found in the state."
        state_vec = torch.cat(
            node_features, dim=-1
        )  # shape (B, num_nodes * feature_dim)
        return state_vec
