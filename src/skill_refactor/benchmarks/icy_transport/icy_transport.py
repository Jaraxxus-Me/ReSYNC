"""IcyTransport environment planning components."""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence, cast

import gymnasium as gym
import numpy as np
import torch
from prbench.envs.dynamic2d.object_types import (
    Dynamic2DType,
    KinRectangleType,
    SmallCircleType,
)
from relational_structs import (
    GroundAtom,
    Object,
    PDDLDomain,
    Type,
    Variable,
)
from torch import Tensor

from skill_refactor.benchmarks.base import (
    BaseRLTAMPSystem,
    GraphData,
    TensorPlanningComponents,
)
from skill_refactor.benchmarks.icy_transport.icy_transport_env import (
    CarRobotType,
    IcyRegionType,
    IcyTransport2DEnv,
    MuddyRegionType,
    ObjectType,
    RobotType,
)
from skill_refactor.benchmarks.icy_transport.utils import (
    DOORWAYS,
    ROOM_CENTERS,
    ROOM_PLANS,
    extract_handempty,
    extract_object_held,
    extract_object_pose,
    extract_region_activated,
    extract_robot_pose,
    extract_robot_vel,
    object_in_room,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvWrapper,
    NormalizeActionMultiEnvWrapper,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action
from skill_refactor.utils.motion_planning import (
    CarPDController,
)
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

# Room type removed - room logic now handled internally by skills


def get_scenario_regions() -> list[str]:
    """Get the region names based on the scenario configuration."""
    region_names = ["icy_region", "muddy_region", "sandy_region"]
    scenarios = CFG.scenario.split(",")
    return [region_names[int(s) - 1] for s in scenarios]


def xy_cos_sin_to_xy_theta(xy_cos_sin: Tensor) -> Tensor:
    """Convert (x, y, cos(theta), sin(theta)) to (x, y, theta)."""
    assert xy_cos_sin.shape[1] == 4
    x = xy_cos_sin[:, 0]
    y = xy_cos_sin[:, 1]
    cos_theta = xy_cos_sin[:, 2]
    sin_theta = xy_cos_sin[:, 3]
    theta = torch.atan2(sin_theta, cos_theta)
    return torch.stack([x, y, theta], dim=1)  # (B, 3)


def xy_cos_sin_to_matrix(xy_cos_sin: Tensor) -> Tensor:
    """Convert (x, y, cos(theta), sin(theta)) to 3x3 transformation matrix."""
    B = xy_cos_sin.shape[0]
    assert xy_cos_sin.shape[1] == 4
    x = xy_cos_sin[:, 0]
    y = xy_cos_sin[:, 1]
    cos_theta = xy_cos_sin[:, 2]
    sin_theta = xy_cos_sin[:, 3]
    zeros = torch.zeros(B, device=xy_cos_sin.device, dtype=xy_cos_sin.dtype)
    ones = torch.ones(B, device=xy_cos_sin.device, dtype=xy_cos_sin.dtype)
    matrix = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, x], dim=1),
            torch.stack([sin_theta, cos_theta, y], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )  # (B, 3, 3)
    return matrix


def matrix_to_xy_cos_sin(matrix: Tensor) -> Tensor:
    """Convert 3x3 transformation matrix to (x, y, cos(theta), sin(theta))."""
    assert matrix.shape[1] == 3 and matrix.shape[2] == 3
    x = matrix[:, 0, 2]
    y = matrix[:, 1, 2]
    cos_theta = matrix[:, 0, 0]
    sin_theta = matrix[:, 1, 0]
    return torch.stack([x, y, cos_theta, sin_theta], dim=1)  # (B, 4)


def xy_cos_sin_relative(xy_cos_sin1: Tensor, xy_cos_sin2: Tensor) -> Tensor:
    """Compute relative pose from xy_cos_sin1 to xy_cos_sin2.

    This returns xy_cos_sin1 defined in xy_cos_sin2's frame.
    """
    matrix1 = xy_cos_sin_to_matrix(xy_cos_sin1)  # (B, 3, 3)
    matrix2 = xy_cos_sin_to_matrix(xy_cos_sin2)  # (B, 3, 3)
    relative_matrix = torch.bmm(torch.inverse(matrix2), matrix1)  # [B, 3, 3]
    x = relative_matrix[:, 0, 2]
    y = relative_matrix[:, 1, 2]
    cos_theta = relative_matrix[:, 0, 0]
    sin_theta = relative_matrix[:, 1, 0]
    return torch.stack([x, y, cos_theta, sin_theta], dim=1)  # (B, 4)


def xy_cos_sin_relative_inverse(xy_cos_sin1: Tensor, xy_cos_sin2: Tensor) -> Tensor:
    """Compute world pose of xy_cos_sin1 world xy_cos_sin2."""
    matrix1 = xy_cos_sin_to_matrix(xy_cos_sin1)  # (B, 3, 3)
    matrix2 = xy_cos_sin_to_matrix(xy_cos_sin2)  # (B, 3, 3)
    world_matrix1 = torch.bmm(matrix2, matrix1)  # [B, 3, 3]
    x = world_matrix1[:, 0, 2]
    y = world_matrix1[:, 1, 2]
    cos_theta = world_matrix1[:, 0, 0]
    sin_theta = world_matrix1[:, 1, 0]
    return torch.stack([x, y, cos_theta, sin_theta], dim=1)  # (B, 4)


def _replace_region_pose(
    obs: Tensor, new_reg_poses_robot_frame: Tensor, new_reg_activated: Tensor
) -> Tensor:
    """Replace region position in observation.

    Note that these are initially defined in the robot's frame. idx is 0-indexed region
    number.
    """
    new_obs = obs.clone()
    num_obstructions = len(CFG.scenario.split(","))
    robot_pose = extract_robot_pose(obs)  # (B, 3)
    robot_xy_cos_sin = torch.cat(
        [
            robot_pose[:, 0:2],
            torch.cos(robot_pose[:, 2:3]),
            torch.sin(robot_pose[:, 2:3]),
        ],
        dim=1,
    )  # (B, 4)

    assert new_reg_poses_robot_frame.shape[1] == num_obstructions
    assert new_reg_poses_robot_frame.shape[0] == obs.shape[0]

    if num_obstructions == 1:
        new_reg_poses = xy_cos_sin_relative_inverse(
            new_reg_poses_robot_frame.squeeze(1), robot_xy_cos_sin
        )  # (B, 4)
        new_obs[:, 0:3] = xy_cos_sin_to_xy_theta(new_reg_poses)
        new_obs[:, 14] = new_reg_activated.squeeze()
    else:
        for idx in range(num_obstructions):
            new_reg_poses = xy_cos_sin_relative_inverse(
                new_reg_poses_robot_frame[:, idx, :], robot_xy_cos_sin
            )  # (B, 4)
            new_reg_act = new_reg_activated[:, idx]
            new_pose = xy_cos_sin_to_xy_theta(new_reg_poses)  # (B, 3)
            s = 20 * idx
            e = s + 3
            new_obs[:, s:e] = new_pose
            new_obs[:, s + 14] = new_reg_act.squeeze()
    return new_obs


class IcyTransportTypes(TypeContainer):
    """Container for IcyTransport types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = RobotType
        self.car_robot = CarRobotType
        self.transport_obj = ObjectType
        self.kinematic_rectangle = KinRectangleType
        self.dynamic2d = Dynamic2DType
        self.small_circle = SmallCircleType
        self.icy_region = IcyRegionType
        self.muddy_region = MuddyRegionType

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {
            self.robot,
            self.car_robot,
            self.transport_obj,
            self.dynamic2d,
            self.kinematic_rectangle,
            self.small_circle,
            self.icy_region,
            self.muddy_region,
        }

    def as_dict(self) -> dict[str, Type]:
        """Convert to dictionary of types."""
        return {
            "car_robot": self.car_robot,
            "robot": self.robot,
            "transport_obj": self.transport_obj,
            "dynamic2d": self.dynamic2d,
            "kinematic_rectangle": self.kinematic_rectangle,
            "small_circle": self.small_circle,
            "icy_region": self.icy_region,
            "muddy_region": self.muddy_region,
        }


class IcyTransportPredicates(PredicateContainer):
    """Container for IcyTransport predicates."""

    def __init__(self, types: IcyTransportTypes) -> None:
        """Initialize predicates."""
        handempty = Predicate("HandEmpty", [types.robot])
        holding = Predicate("Holding", [types.robot, types.transport_obj])
        delivered = Predicate("Delivered", [types.transport_obj, types.transport_obj])
        self.predicates = {
            "HandEmpty": handempty,
            "Holding": holding,
            "Delivered": delivered,
        }


class IcyTransportObjectsSC1(ObjectContainer):
    """Container for IcyTransport objects."""

    def __init__(self, types: IcyTransportTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.transport_obj1 = Object("transport_obj1", types.transport_obj)
        self.transport_obj2 = Object("transport_obj2", types.transport_obj)
        self.target_obj = Object("target_obj", types.transport_obj)
        # Icy region
        self.icy_region = Object("icy_region", types.icy_region)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.transport_obj1,
            self.transport_obj2,
            self.target_obj,
            self.icy_region,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        return {
            "robot": self.robot,
            "transport_obj1": self.transport_obj1,
            "transport_obj2": self.transport_obj2,
            "target_obj": self.target_obj,
            "icy_region": self.icy_region,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.transport_obj1: 1,
            self.transport_obj2: 2,
            self.target_obj: 3,
            self.icy_region: 4,
        }


class IcyTransportObjectsSC2(ObjectContainer):
    """Container for IcyTransport objects."""

    def __init__(self, types: IcyTransportTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.transport_obj1 = Object("transport_obj1", types.transport_obj)
        self.transport_obj2 = Object("transport_obj2", types.transport_obj)
        self.target_obj = Object("target_obj", types.transport_obj)
        # Icy region
        self.muddy_region = Object("muddy_region", types.muddy_region)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.transport_obj1,
            self.transport_obj2,
            self.target_obj,
            self.muddy_region,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        return {
            "robot": self.robot,
            "transport_obj1": self.transport_obj1,
            "transport_obj2": self.transport_obj2,
            "target_obj": self.target_obj,
            "muddy_region": self.muddy_region,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.transport_obj1: 1,
            self.transport_obj2: 2,
            self.target_obj: 3,
            self.muddy_region: 4,
        }


class IcyTransportObjectsSC12(ObjectContainer):
    """Container for IcyTransport objects."""

    def __init__(self, types: IcyTransportTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.transport_obj1 = Object("transport_obj1", types.transport_obj)
        self.transport_obj2 = Object("transport_obj2", types.transport_obj)
        self.target_obj = Object("target_obj", types.transport_obj)
        # Icy region
        self.icy_region = Object("icy_region", types.icy_region)
        self.muddy_region = Object("muddy_region", types.muddy_region)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.transport_obj1,
            self.transport_obj2,
            self.target_obj,
            self.icy_region,
            self.muddy_region,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        return {
            "robot": self.robot,
            "transport_obj1": self.transport_obj1,
            "transport_obj2": self.transport_obj2,
            "target_obj": self.target_obj,
            "icy_region": self.icy_region,
            "muddy_region": self.muddy_region,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.transport_obj1: 1,
            self.transport_obj2: 2,
            self.target_obj: 3,
            self.icy_region: 4,
            self.muddy_region: 5,
        }


def _extract_current_objects() -> ObjectContainer:
    if CFG.scenario == "1":
        return IcyTransportObjectsSC1(IcyTransportTypes())
    elif CFG.scenario == "2":
        return IcyTransportObjectsSC2(IcyTransportTypes())
    elif CFG.scenario == "1,2":
        return IcyTransportObjectsSC12(IcyTransportTypes())
    raise ValueError(f"Unknown scenario: {CFG.scenario}")


def _get_room_name_from_position(pos: Tensor) -> List[str]:
    """Determine which room each position (in batch) is in.

    Args:
        pos: Position tensor (B, 2)

    Returns:
        List of room names for each batch element
    """
    batch_size = pos.shape[0]

    # Check all rooms in batch - returns (B,) boolean tensor for each room
    room_names_list = ["room_bl", "room_br", "room_tl", "room_tr"]

    # Stack room membership checks into (B, 4) tensor
    room_masks = torch.stack(
        [
            object_in_room(pos, room_name, CFG.i_trans_in_room_offset)
            for room_name in room_names_list
        ],
        dim=1,
    )  # (B, 4)

    # Find first True value for each batch element (priority order: bl, br, tl, tr)
    # argmax returns index of first True (or 0 if all False)
    room_indices = torch.argmax(room_masks.int(), dim=1)  # (B,)

    # Check if any room matched for each batch element
    any_match = torch.any(room_masks, dim=1)  # (B,)

    # Build result list - use room_bl as default if no match
    room_names = [
        room_names_list[int(room_indices[b].item())] if any_match[b] else "room_bl"
        for b in range(batch_size)
    ]

    return room_names


class BaseIcyTransportSkill(LiftedOperatorSkill):
    """Base class for IcyTransport environment skills."""

    def __init__(self, env: IcyTransport2DEnv, operators: set[LiftedOperator]) -> None:
        """Initialize skill."""
        super().__init__()
        self._all_operators = operators
        self._lifted_operator = self.get_lifted_operator()
        action_space = cast(gym.spaces.Box, env.action_space)
        self.action_dim = action_space.shape[0] if action_space.shape is not None else 0
        self.device = CFG.device
        self.normalize_action = CFG.normalize_action

        # Create PD controller for car navigation
        self.pd_controller = CarPDController(
            kp_pos=CFG.kp_pos,
            kv_pos=CFG.kv_pos,
            kp_ang=CFG.kp_ang,
            kv_ang=CFG.kv_ang,
            device=self.device,
        )

        # Waypoint tracking state (per environment in batch)
        self._waypoints: List[torch.Tensor] | None = None  # N x (B, 3) waypoints
        self._current_waypoint: torch.Tensor | None = (
            None  # (B, 3) current index per env
        )
        self._waypoint_try_count: torch.Tensor | None = None  # Try count per env

        # Action space bounds for clipping
        self.action_low = torch.tensor(
            action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.tensor(
            action_space.high, dtype=torch.float32, device=self.device
        )

        # NOTE: In icy_transport, we don't have a meaningful motion plan since
        # the length depends on control frequency and dynamics.
        # this is a placeholder so that new skills never terminate unless predicted by
        # the terminal predicate or exhausted due to max steps.
        self._current_plan = [
            torch.tensor(action_space.sample(), device=self.device, dtype=torch.float32)
        ]

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        """Reset skill with ground operator."""
        self._waypoints = None
        self._current_waypoint = None
        return super().reset(ground_operator, obs)

    @abc.abstractmethod
    def _get_waypoints_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> List[torch.Tensor]:
        """Get waypoints given objects and observation.

        Returns:
            Waypoint tensor (N, 3) as [x, y, theta]
        """
        raise NotImplementedError

    def _check_waypoint_achieved(self, obs: Tensor, target: Tensor) -> Tensor:
        """Check if current waypoint is achieved for each environment.

        Args:
            obs: Batched observation (B, obs_dim)
            target: Target waypoint (B, 3) as [x, y, theta]

        Returns:
            Boolean tensor (B,) indicating achievement
        """
        robot_pose = extract_robot_pose(obs)  # (B, 3)
        robot_pos = robot_pose[:, :2]  # (B, 2)
        target_pos = target[:, :2]  # (B, 2)

        # robot_vel = extract_robot_vel(obs)
        # static = torch.all(torch.abs(robot_vel) < 1e-3, dim=1)  # (B,)

        robot_angle = robot_pose[:, 2]  # (B,)
        target_angle = target[:, 2]  # (B,)
        # robot_ang_vel = extract_robot_vel(obs)[:, 2]  # (B,)

        # convert angles to [-pi, pi]
        target_angle = (target_angle + torch.pi) % (2 * torch.pi) - torch.pi
        ang_error = target_angle - robot_angle
        ang_error = (ang_error + torch.pi) % (2 * torch.pi) - torch.pi

        distance = torch.sqrt(torch.sum((target_pos - robot_pos) ** 2, dim=1))
        ang_diff = torch.abs(ang_error)
        achieved = (distance < CFG.i_trans_waypoint_threshold_trans) & (
            ang_diff < CFG.i_trans_waypoint_threshold_rot
        )

        return achieved

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Get action given objects and observation."""
        batch_size = obs.shape[0]

        # Initialize waypoints if needed
        if self._waypoints is None:
            self._waypoints = self._get_waypoints_given_objects(objects, obs)
            self._current_waypoint = self._waypoints.pop(0)
            self._waypoint_try_count = torch.zeros(
                (batch_size,), dtype=torch.long, device=self.device
            )

        # Check if current waypoint achieved
        assert self._current_waypoint is not None
        achieved = self._check_waypoint_achieved(obs, self._current_waypoint)

        # Update waypoint indices and try counts
        if (
            achieved.all()
            or (self._waypoint_try_count >= CFG.i_trans_max_waypoint_tries).all()
        ):
            # Move all to next waypoint
            self._waypoint_try_count = torch.zeros(
                (batch_size,), dtype=torch.long, device=self.device
            )
            if len(self._waypoints) > 0:
                self._current_waypoint = self._waypoints.pop(0)
            else:
                return get_frozen_action(
                    torch.zeros((batch_size, self.action_dim), device=self.device),
                    self.action_low,
                    self.action_high,
                    self.normalize_action,
                    CFG.control_mode,
                    obs=obs,
                )

        # Compute control using PD controller
        assert self._current_waypoint is not None
        control = self.pd_controller.compute_control(obs, self._current_waypoint)
        assert self._waypoint_try_count is not None
        self._waypoint_try_count += 1

        # Clip to action space bounds
        clipped_control = torch.clamp(control, self.action_low, self.action_high)

        # Normalize action if needed
        # Normalize if needed
        if self.normalize_action:
            # self.arm_action_low is -0.2, self.arm_action_high is 0.2
            # NOT the actual delta min/max
            low = self.action_low.unsqueeze(0).repeat(clipped_control.shape[0], 1)
            high = self.action_high.unsqueeze(0).repeat(clipped_control.shape[0], 1)
            clipped_control_norm = (clipped_control - 0.5 * (low + high)) / (
                0.5 * (high - low)
            )
            return clipped_control_norm

        return clipped_control

    def get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op for op in self._all_operators if op.name == self.get_operator_name()
        )

    def get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class GoToPickObjectSkill(BaseIcyTransportSkill):
    """Skill for navigating to and picking up an object.

    Handles internal room-level planning:
    1. Determines robot's current room
    2. Determines transport object's room
    3. If different rooms, navigates through doorways
    4. Once in same room, navigates to object
    """

    def get_operator_name(self) -> str:
        return "GoToPickObject"

    def _get_waypoints_given_object_pos(
        self, robot_pose: Tensor, target_pose: Tensor
    ) -> List[torch.Tensor]:
        """Get waypoints for navigation given robot and target positions.

        Args:
            robot_pos: Robot position tensor (B, 2)
            target_pos: Target object position tensor (B, 2)
        Returns:
            List of waypoint tensors, each (B, 3) as [x, y, theta]
        """
        batch_size = robot_pose.shape[0]
        waypoints: List[torch.Tensor] = []

        robot_pos = robot_pose[:, :2]  # (B, 2)
        target_pos = target_pose[:, :2]  # (B, 2)

        # Determine rooms for each batch element
        robot_rooms = _get_room_name_from_position(robot_pos)
        target_rooms = _get_room_name_from_position(target_pos)

        # Get room transitions for each batch element using precomputed ROOM_PLANS
        # ROOM_PLANS[(from, to)] = [(from, mid), (mid, to)] or [(from, to)]
        # If from == to, there are no transitions needed
        room_transitions_list: list[list[tuple[str, str]]] = []
        for robot_room, target_room in zip(robot_rooms, target_rooms):
            if robot_room == target_room:
                # Already in target room, no transitions needed
                room_transitions_list.append([])
            else:
                # Get precomputed path
                room_transitions_list.append(ROOM_PLANS[(robot_room, target_room)])

        # Determine max transitions needed across batch
        transition_lengths = [len(transitions) for transitions in room_transitions_list]
        max_transitions = max(transition_lengths) if transition_lengths else 0

        # Generate batched waypoints for room-level navigation
        if max_transitions > 0:
            # Need to navigate through rooms
            for i in range(max_transitions):
                # Current pose is last waypoint or robot's current pose
                current_pose = waypoints[-1] if waypoints else robot_pose.clone()

                # Check which environments still have transitions at step i (batched)
                has_transition = torch.tensor(
                    [i < len(room_transitions_list[b]) for b in range(batch_size)],
                    dtype=torch.bool,
                    device=robot_pose.device,
                )  # (B,)

                # For environments with transitions, get from_room and to_room
                from_rooms = []
                to_rooms = []
                for b in range(batch_size):
                    if has_transition[b]:
                        from_rooms.append(room_transitions_list[b][i][0])
                        to_rooms.append(room_transitions_list[b][i][1])
                    else:
                        # No transition, stay in place (use dummy values)
                        from_rooms.append(robot_rooms[b])
                        to_rooms.append(robot_rooms[b])

                # Get room center positions for from_room (batched)
                room_centers_from = torch.stack(
                    [
                        torch.tensor(
                            ROOM_CENTERS[from_rooms[b]],
                            dtype=robot_pose.dtype,
                            device=robot_pose.device,
                        )
                        for b in range(batch_size)
                    ],
                    dim=0,
                )  # (B, 2)

                # Get doorway positions for to_room (batched)
                doorways_to = torch.stack(
                    [
                        torch.tensor(
                            (
                                DOORWAYS[(from_rooms[b], to_rooms[b])]
                                if from_rooms[b] != to_rooms[b]
                                else ROOM_CENTERS[to_rooms[b]]
                            ),  # Stay in same room
                            dtype=robot_pose.dtype,
                            device=robot_pose.device,
                        )
                        for b in range(batch_size)
                    ],
                    dim=0,
                )  # (B, 2)

                # Waypoint 1: Face room center (from side) or stay in place
                wp1 = current_pose.clone()
                # Compute direction angles for all batch elements (batched)
                direction_angle1 = (
                    torch.atan2(
                        room_centers_from[:, 1] - current_pose[:, 1],
                        room_centers_from[:, 0] - current_pose[:, 0],
                    )
                    - torch.pi / 2
                )  # (B,)
                # Only update angle for envs with transitions using mask
                wp1[:, 2] = torch.where(
                    has_transition, direction_angle1, current_pose[:, 2]
                )
                waypoints.append(wp1)

                # Waypoint 2: Move to room center or stay in place
                wp2 = current_pose.clone()
                # Use masked operations to update x, y, theta
                wp2[:, 0] = torch.where(
                    has_transition, room_centers_from[:, 0], current_pose[:, 0]
                )
                wp2[:, 1] = torch.where(
                    has_transition, room_centers_from[:, 1], current_pose[:, 1]
                )
                wp2[:, 2] = torch.where(
                    has_transition, direction_angle1, current_pose[:, 2]
                )
                waypoints.append(wp2)

                # Waypoint 3: Face through doorway or stay in place
                wp3 = wp2.clone()
                # Compute direction angles to doorways (batched)
                direction_angle2 = (
                    torch.atan2(
                        doorways_to[:, 1] - wp2[:, 1],
                        doorways_to[:, 0] - wp2[:, 0],
                    )
                    - torch.pi / 2
                )  # (B,)
                wp3[:, 2] = torch.where(has_transition, direction_angle2, wp2[:, 2])
                waypoints.append(wp3)

                # Waypoint 4: Move through doorway or stay in place
                wp4 = wp3.clone()
                wp4[:, 0] = torch.where(has_transition, doorways_to[:, 0], wp3[:, 0])
                wp4[:, 1] = torch.where(has_transition, doorways_to[:, 1], wp3[:, 1])
                # Keep wp3's angle
                waypoints.append(wp4)

        # Final waypoints: navigate to target object
        current_pose = waypoints[-1] if waypoints else robot_pose.clone()

        # Waypoint N-1: Face target object
        wp_face = current_pose.clone()
        direction_to_target = (
            torch.atan2(
                target_pose[:, 1] - current_pose[:, 1],
                target_pose[:, 0] - current_pose[:, 0],
            )
            - torch.pi / 2
        )
        wp_face[:, 2] = direction_to_target
        waypoints.append(wp_face)

        # Waypoint N: Move to target object
        wp_target = wp_face.clone()
        wp_target[:, 0] = target_pose[:, 0]
        wp_target[:, 1] = target_pose[:, 1]
        waypoints.append(wp_target)

        return waypoints

    def _get_waypoints_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> List[torch.Tensor]:
        """Get waypoints for navigation.

        Args:
            objects: [robot, transport_obj] for pick, [robot, transport_obj, target_obj] for place
            obs: Current observation

        Returns:
            List of waypoint tensors, each (B, 3) as [x, y, theta]
        """
        assert objects[0].name == "robot"
        # assert "transport_obj" in objects[1].name
        # Get robot and target object positions
        robot_pose = extract_robot_pose(obs)  # (B, 3)

        # For pick: target is transport_obj (objects[1])
        target_obj = objects[1]  # Second object is always the navigation target
        target_pose = extract_object_pose(obs, target_obj)  # (B, 3)

        return self._get_waypoints_given_object_pos(robot_pose, target_pose)

    def terminate_with_objects(self, objects: Sequence[Object], obs: Tensor) -> Tensor:
        """Terminate when object is held (collision detected)."""
        del objects  # Unused
        colliding = obs[:, -1].to(torch.bool)  # (B,) last entry is collision flag
        held = ~extract_handempty(obs).to(torch.bool)  # (B,)
        static = torch.all(torch.abs(extract_robot_vel(obs)) < 1e-2, dim=1)  # (B,)
        held = held & static
        assert self._waypoints is not None
        assert self._waypoint_try_count is not None
        if len(self._waypoints) == 0:
            current_wp_maxed_out = (
                self._waypoint_try_count >= CFG.i_trans_max_waypoint_tries
            )
            return held | current_wp_maxed_out | colliding
        else:
            return held | colliding


class GoToPlaceObjectSkill(GoToPickObjectSkill):
    """Skill for navigating to target location and placing object.

    Handles internal room-level planning:
    1. Determines robot's current room (holding transport object)
    2. Determines target object's room
    3. If different rooms, navigates through doorways
    4. Once in same room, navigates to target
    """

    def get_operator_name(self) -> str:
        return "GoToPlaceObject"

    def _get_waypoints_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> List[torch.Tensor]:
        """Get waypoints for navigation.

        Args:
            objects: [robot, transport_obj] for pick, [robot, transport_obj, target_obj] for place
            obs: Current observation

        Returns:
            List of waypoint tensors, each (B, 3) as [x, y, theta]
        """
        assert objects[0].name == "robot"
        assert "target_obj" == objects[2].name
        # Get robot and target object positions
        robot_pose = extract_robot_pose(obs)  # (B, 3)

        target_obj = objects[2]
        target_pose = extract_object_pose(obs, target_obj)  # (B, 3)

        return self._get_waypoints_given_object_pos(robot_pose, target_pose)

    def terminate_with_objects(self, objects: Sequence[Object], obs: Tensor) -> Tensor:
        """Terminate when transport object is delivered to target."""
        # objects: [robot, transport_obj, target_obj]
        colliding = obs[:, -1].to(torch.bool)  # (B,) last entry is collision flag
        target_pose = extract_object_pose(obs, objects[2])  # target_obj (B, 3)
        robot_pose = extract_robot_pose(obs)  # (B, 3)
        distance = torch.sqrt(
            torch.sum((target_pose[:, :2] - robot_pose[:, :2]) ** 2, dim=1)
        )
        close_enough = distance < CFG.i_transport_object_radius
        static = torch.all(torch.abs(extract_robot_vel(obs)) < 1e-2, dim=1)  # (B,)
        handempty = extract_handempty(obs).to(torch.bool)  # (B,)
        close_enough = close_enough & static & handempty
        assert self._waypoints is not None
        assert self._waypoint_try_count is not None
        if len(self._waypoints) == 0:
            current_wp_maxed_out = (
                self._waypoint_try_count >= CFG.i_trans_max_waypoint_tries
            )
            return close_enough | current_wp_maxed_out | colliding
        else:
            return close_enough | colliding


class IcyTransportPerceiver(Perceiver):
    """Perceiver for IcyTransport environment."""

    def __init__(
        self, predicates: IcyTransportPredicates, types: IcyTransportTypes
    ) -> None:
        """Initialize with required types."""
        self.predicates_container = predicates
        self._types = types

        self.predicate_interpreters = {
            predicates["HandEmpty"]: self._interpret_handempty,
            predicates["Holding"]: self._interpret_holding,
            predicates["Delivered"]: self._interpret_delivered,
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
        latest_objects = _extract_current_objects()

        # Goal: transport object to target object position
        transport_obj1 = latest_objects.as_dict()["transport_obj1"]
        transport_obj2 = latest_objects.as_dict()["transport_obj2"]
        target_obj = latest_objects.as_dict()["target_obj"]
        goal = {
            self.predicates_container["Delivered"]([transport_obj1, target_obj]),
            self.predicates_container["Delivered"]([transport_obj2, target_obj]),
        }

        return latest_objects.as_set(), atoms[0], goal

    @property
    def objects(self) -> ObjectContainer:
        """Get the object container."""
        return _extract_current_objects()

    def step(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Step perceiver with observation."""
        return self._get_atoms(obs, info)

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
            if (len(input_args) == 0) and not predicate.quantified:
                # No valid arguments for basic predicate, so skip
                # Quantified predicates can still be True with zero arguments (e.g., ForAll)
                continue
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

    def _interpret_holding(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Holding predicate."""
        holding_obj = []
        for obj in objects:
            holding_obj.append(extract_object_held(obs, obj[1]).to(torch.bool))
        return torch.stack(holding_obj, dim=1)  # shape (B, num_pairs)

    def _interpret_handempty(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret HandEmpty predicate."""
        handempty = extract_handempty(obs).to(torch.bool)
        not_held = handempty.unsqueeze(1).repeat(1, len(objects))
        return not_held

    def _interpret_delivered(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Delivered predicate (object in target room)."""
        diff_obj = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        distance = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.float32, device=obs.device
        )
        for obj_pair in objects:
            if (
                (obj_pair[0].name != obj_pair[1].name)
                and ("transport_obj" in obj_pair[0].name)
                and (obj_pair[1].name == "target_obj")
            ):
                # If both objects are the same, we cannot compute distance
                diff_obj[:, objects.index(obj_pair)] = True

            obj1_pose = extract_object_pose(obs, obj_pair[0])
            obj2_pose = extract_object_pose(obs, obj_pair[1])
            dist = torch.sqrt(
                torch.sum((obj1_pose[:, :2] - obj2_pose[:, :2]) ** 2, dim=1)
            )
            distance[:, objects.index(obj_pair)] = dist

        close_enough = distance <= CFG.i_transport_object_radius * 2
        return diff_obj & close_enough


class IcyTransportRLTAMPSystem(BaseRLTAMPSystem):
    """Base TAMP system for IcyTransport environment."""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize IcyTransport2D TAMP system."""
        self._render_mode = render_mode
        self.env_kwargs = {
            "render_mode": render_mode,
        }
        self.env_name = "skill_ref/IcyTransport2D-v0"
        super().__init__(
            planning_components, name="IcyTransportTAMPSystem", seed=seed  # type: ignore
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""

        assert (
            CFG.control_mode == "force_torque"
        ), "IcyTransport only supports force_torque control."

        def make_env():
            return gym.make(
                self.env_name,
                **self.env_kwargs,
            )

        if CFG.normalize_action:
            norm_envs = NormalizeActionMultiEnvWrapper(  # type: ignore
                make_env,
                num_envs=CFG.num_envs,
                auto_reset=False,
                to_tensor=True,
                device=CFG.device,
                max_episode_steps=CFG.max_env_steps,
            )
            return norm_envs  # type: ignore
        envs = MultiEnvWrapper(
            make_env,
            num_envs=CFG.num_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
        return envs

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "IcyTransport2D-domain"

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
        """Create planning components for IcyTransport system."""
        types_container = IcyTransportTypes()
        predicates = IcyTransportPredicates(types_container)

        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Delivered = predicates["Delivered"]

        perceiver = IcyTransportPerceiver(predicates, types_container)

        robot = Variable("?robot", types_container.robot)
        transport_obj = Variable("?transport_obj", types_container.transport_obj)
        transport_target = Variable("?transport_target", types_container.transport_obj)

        operators = {
            LiftedOperator(
                "GoToPickObject",
                parameters=[robot, transport_obj],
                preconditions={
                    HandEmpty([robot]),
                },
                add_effects={
                    Holding([robot, transport_obj]),
                },
                delete_effects={
                    HandEmpty([robot]),
                },
            ),
            LiftedOperator(
                "GoToPlaceObject",
                parameters=[robot, transport_obj, transport_target],
                preconditions={
                    Holding([robot, transport_obj]),
                },
                add_effects={
                    Delivered([transport_obj, transport_target]),
                    HandEmpty([robot]),
                },
                delete_effects={
                    Holding([robot, transport_obj]),
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
    ) -> IcyTransportRLTAMPSystem:
        """Factory method for creating system with default components."""
        planning_components = cls._create_planning_components()
        system = cls(
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        assert isinstance(system.env.unwrapped[0], IcyTransport2DEnv)  # type: ignore[index]
        skills = {
            GoToPickObjectSkill(
                system.env.unwrapped[0], planning_components.operators  # type: ignore[index]
            ),
            GoToPlaceObjectSkill(
                system.env.unwrapped[0], planning_components.operators  # type: ignore[index]
            ),
        }
        system.components.skills.update(skills)  # type: ignore
        return system

    def state_to_graph(self, state: Tensor, training: bool = False) -> List[GraphData]:
        """Convert Batched IcyTransport environment state tensor to graph
        representation.

        Args:
            state: State tensor from IcyTransport environment, batched
                Note that in IcyTransport, everything is defined in the robot's frame.
            _training: Whether in training mode (adds noise, currently unused)

        Returns:
            List of GraphData with nodes representing objects and edges representing relationships
        """
        # Get objects from the perceiver's object container
        b = state.shape[0]
        num_nodes = len(self.perceiver.objects.object_to_node.keys())

        # Node features 5: activated (1) +
        # SE2 in robot frame (x, y, cos(theta), sin(theta)) (4)
        node_feature_dim = 5
        device = state.device
        dtype = state.dtype
        node_features = torch.zeros(
            b, num_nodes, node_feature_dim, device=device, dtype=dtype
        )

        # Extract features for each object using existing helper functions
        robot_pose = extract_robot_pose(state)[:, :3]
        region_idxs = []
        robot_xy_cos_sin = torch.cat(
            [
                robot_pose[:, 0:1],
                robot_pose[:, 1:2],
                torch.cos(robot_pose[:, 2:3]),
                torch.sin(robot_pose[:, 2:3]),
            ],
            dim=-1,
        )
        for obj, i in self.perceiver.objects.object_to_node.items():
            # Extract object pose (position + rotation)
            obj_pose = extract_object_pose(state, obj)[:, :3]
            if training:
                # Add noise to object pose during training for NN robustness
                noise_xy = (
                    torch.randn_like(obj_pose[:, :2])
                    * CFG.icy_transport_reach_drift_trans
                )
                noise_theta = (
                    torch.randn_like(obj_pose[:, 2:3])
                    * CFG.icy_transport_reach_drift_rot
                )
                noise = torch.cat([noise_xy, noise_theta], dim=-1)
                obj_pose += noise
            normalized_x = obj_pose[:, 0:1]
            normalized_y = obj_pose[:, 1:2]
            cos_theta = torch.cos(obj_pose[:, 2:3])
            sin_theta = torch.sin(obj_pose[:, 2:3])
            obj_pose_norm = torch.cat(
                [normalized_x, normalized_y, cos_theta, sin_theta], dim=-1
            )
            rel_obj_pose = xy_cos_sin_relative(obj_pose_norm, robot_xy_cos_sin)
            # Fill in node features
            node_features[:, i, -4:] = rel_obj_pose
            if "region" in obj.name:
                # Regions have zero mass/moment/force/vel
                region_activated = extract_region_activated(state, obj)
                node_features[:, i, 0:1] = region_activated
                region_idxs.append(i)

        # Create edges: fully connected graph between all objects
        edge_list = []
        edge_features_list = []
        # Edge features (5): Similar to node, but the relative pose is
        # between the two objects (9 features + 4 relative pose)
        edge_feature_dim = 5

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_list.append([i, j])
                    # Compute proper relative pose between objects using SE(2) transformation
                    pos_mat_i = xy_cos_sin_to_matrix(
                        node_features[:, i, -4:]
                    )  # [B, 3, 3]
                    pos_mat_j = xy_cos_sin_to_matrix(
                        node_features[:, j, -4:]
                    )  # [B, 3, 3]
                    rel_pos_mat = torch.bmm(
                        torch.inverse(pos_mat_i), pos_mat_j
                    )  # [B, 3, 3]
                    rel_pos = matrix_to_xy_cos_sin(
                        rel_pos_mat
                    )  # [B, 4] relative pose in SE(2)

                    if j in region_idxs:
                        region_activated = node_features[:, j, 0:1]  # [B, 1]
                        rel_dx = rel_pos[:, 0:1]
                        rel_dy = rel_pos[:, 1:2]
                        rel_cos = rel_pos[:, 2:3]
                        matched = (
                            (rel_dx > 0)
                            & (torch.abs(rel_dy) < CFG.i_trans_obj_pos_rnd_range * 2)
                            & (rel_cos > 0.9)
                        )
                        matched_activated = region_activated * matched.to(dtype)

                        edge_feat = torch.cat(
                            [matched_activated, rel_pos], dim=-1
                        )  # 13-dim edge features (1 + 8 + 4)
                    else:
                        zero_features = torch.zeros(
                            (b, 1), dtype=rel_pos.dtype, device=rel_pos.device
                        )
                        edge_feat = torch.cat([zero_features, rel_pos], dim=-1)
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
                node_features=node_features[batch_idx].to(
                    torch.float32
                ),  # [num_nodes, node_feature_dim]
                edge_features=edge_features[batch_idx].to(
                    torch.float32
                ),  # [num_edges, edge_feature_dim]
                edge_indices=edge_indices,  # [2, num_edges] - same for all batch elements
                global_features=None,
                object_to_node=self.perceiver.objects.object_to_node.copy(),
            )
            graph_data_list.append(graph_data)

        return graph_data_list

    def state_to_vec(self, state: Tensor, objects: Sequence[Object]) -> Tensor:
        """Simply concatenate the node features used in the graph representation."""
        graph_data_list = self.state_to_graph(state)
        b = state.shape[0]
        vecs = []
        for batch_idx in range(b):
            graph_data = graph_data_list[batch_idx]
            node_features = graph_data.node_features  # [num_nodes, node_feature_dim]
            assert graph_data.object_to_node is not None
            obj_indices = [
                graph_data.object_to_node[obj] for obj in objects
            ]  # Indices of the specified objects
            selected_node_features = node_features[
                obj_indices, :
            ]  # [num_selected_objects, node_feature_dim]
            vec = selected_node_features.flatten()  # Flatten to a vector
            vecs.append(vec)
        return torch.stack(vecs, dim=0)  # [batch_size, vector_dim]

    def skill_obs_space(self) -> gym.Space:
        """Get skill observation space."""
        # Get objects from the perceiver's object container
        obs_low = np.array([-5] * 15, dtype=np.float32)
        obs_high = np.array([5] * 15, dtype=np.float32)
        return gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def full_state_to_relative_state(
        self, full_state: Tensor, relative_name: str = ""
    ) -> Tensor:
        """Convert full state to relative state.

        For IcyTransport, relative state is simply the relative pose between the region
        and the objects.

        Args:
            full_state: Full state tensor from IcyTransport environment, batched
            relative_name: transport target or target region.

        Returns:
            Relative state tensor, batched.
        """
        state_graphs = self.state_to_graph(full_state)
        b = full_state.shape[0]
        region_node_ids = []
        region_node_index = []
        object_node_id = None

        # NOTE: Force obstruction order to be consistent with
        # how observation is created.
        scenario_regions = get_scenario_regions()
        for name in scenario_regions:
            for key_name, obj in (self.perceiver.objects.as_dict()).items():
                if key_name == name:
                    region_node_ids.append(self.perceiver.objects.object_to_node[obj])
                    region_node_index.append(scenario_regions.index(name))
                if key_name == relative_name:
                    object_node_id = self.perceiver.objects.object_to_node[obj]

        assert object_node_id is not None, "Block node id not found."
        assert len(region_node_ids) == len(
            CFG.scenario.split(",")
        ), "Obstruction node ids length mismatch."
        # NOTE: relative states will always be 3 obstructions, even if fewer obstructions exist
        # we will pad with zeros for missing obstructions by default.
        # rel_node_id, activated, rel_pose
        device = full_state.device
        dtype = full_state.dtype
        relative_states = torch.zeros((b, 3, 6), device=device, dtype=dtype)
        for batch_idx in range(b):
            graph = state_graphs[batch_idx]
            for idx, node_obstruction in enumerate(region_node_ids):
                edge_idx = None
                # Find the edge index from block to obstruction
                for ei in range(graph.edge_indices.shape[1]):
                    if (
                        graph.edge_indices[0, ei].item() == object_node_id
                        and graph.edge_indices[1, ei].item() == node_obstruction
                    ):
                        edge_idx = ei
                        break
                if edge_idx is not None:
                    edge_feat = graph.edge_features[edge_idx]
                    obs_idx = region_node_index[idx]
                    relative_states[batch_idx, obs_idx, 0] = object_node_id
                    relative_states[batch_idx, obs_idx, 1:] = edge_feat[-5:]

        return relative_states

    def relative_state_to_full_state(
        self, inital_full_state: Tensor, relative_state: Tensor
    ) -> Tensor:
        """Convert relative state back to full state.

        Args:
            inital_full_state: Initial full state tensor
            _relative_state: Relative state tensor (currently unused)

        Returns:
            New full state tensor with updated positions
        """

        init_state_graphs = self.state_to_graph(inital_full_state)
        B = inital_full_state.shape[0]
        region_node_ids = []
        region_node_index = []

        # NOTE: Force obstruction order to be consistent with
        # how observation is created.
        scenario_regions = get_scenario_regions()
        for name in scenario_regions:
            for key_name, obj in (self.perceiver.objects.as_dict()).items():
                if key_name == name:
                    region_node_ids.append(self.perceiver.objects.object_to_node[obj])
                    region_node_index.append(scenario_regions.index(name))

        device = inital_full_state.device
        dtype = inital_full_state.dtype
        extracted_relative_states = torch.zeros(
            (B, len(region_node_ids), 6), device=device, dtype=dtype
        )
        for num, region_id in enumerate(region_node_index):
            assert relative_state[
                :, region_id, 1:
            ].any(), "Relative state for obstruction is empty."
            extracted_relative_states[:, num, :] = relative_state[:, region_id, :]
        object_node_ids = extracted_relative_states[:, :, 0].long()

        base_obj_poses = torch.zeros(
            (B, len(region_node_ids), 4), device=device, dtype=dtype
        )
        for b in range(B):
            graph = init_state_graphs[b]
            for reg_idx, _ in enumerate(region_node_ids):
                base_node_id = int(object_node_ids[b, reg_idx].item())
                base_node_feat = graph.node_features[base_node_id]
                base_obj_poses[b, reg_idx] = base_node_feat[-4:]

        base_obj_pose_mats = xy_cos_sin_to_matrix(
            base_obj_poses.view(-1, 4)
        )  # [B*M, 3, 3]
        rel_pose_mats = xy_cos_sin_to_matrix(
            extracted_relative_states[:, :, 2:].view(-1, 4)
        )  # [B*M, 3, 3]
        new_reg_pose_mats = torch.bmm(base_obj_pose_mats, rel_pose_mats)  # [B*M, 3, 3]

        new_reg_poses = matrix_to_xy_cos_sin(new_reg_pose_mats).view(
            B, -1, 4
        )  # [B, M, 4]
        new_reg_activated = extracted_relative_states[:, :, 1].unsqueeze(
            -1
        )  # [B, M, 1]
        new_obs = _replace_region_pose(
            inital_full_state, new_reg_poses, new_reg_activated
        )
        return new_obs
