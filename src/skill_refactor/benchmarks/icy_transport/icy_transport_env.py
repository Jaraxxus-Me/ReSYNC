"""Icy Four Room Environment based on MiniGrid's Four Room Environment + PyMunk.

A 4-room environment where a CarRobot must transport an object from one room to another.
The robot can push objects by collision and must navigate through doorways.
"""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium
import numpy as np
import pymunk
from gymnasium.utils import seeding
from numpy.typing import NDArray
from prbench.core import (
    ConstantObjectPRBenchEnv,
    ObjectCentricPRBenchEnv,
    PRBenchEnvConfig,
    RobotActionSpace,
)
from prbench.envs.dynamic2d.object_types import (
    CarRobotType,
    Dynamic2DRobotEnvTypeFeatures,
    KinRectangleType,
    SmallCircleType,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    CarRobot,
    CarRobotActionSpace,
    create_walls_from_world_boundaries,
    get_car_robot_action_from_gui_input,
    on_collision_w_object_car,
    on_collision_w_static_car,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose, ZOrder
from prbench.envs.utils import (
    BLACK,
    BROWN,
    ORANGE,
    object_to_multibody2d,
    render_2dstate,
    sample_se2_pose,
    state_2d_has_collision,
)
from pymunk.vec2d import Vec2d
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict
from tomsgeoms2d.utils import geom2ds_intersect

from skill_refactor.settings import CFG

# Define custom types for this environment
RobotType = Type("robot", parent=CarRobotType)
ObjectType = Type("transport_obj", parent=SmallCircleType)
IcyRegionType = Type("icy_region", parent=KinRectangleType)
MuddyRegionType = Type("muddy_region", parent=KinRectangleType)
SandyRegionType = Type("sandy_region", parent=KinRectangleType)

# Update type features
Dynamic2DRobotEnvTypeFeatures[RobotType] = list(
    Dynamic2DRobotEnvTypeFeatures[CarRobotType] + ["is_colliding"]
)
Dynamic2DRobotEnvTypeFeatures[ObjectType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)
Dynamic2DRobotEnvTypeFeatures[IcyRegionType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRectangleType]
    + [
        "activated",
        "local_x",
        "local_y",
        "fx",
        "fy",
        "torque",
    ]
)
Dynamic2DRobotEnvTypeFeatures[MuddyRegionType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRectangleType]
    + [
        "activated",
        "local_x",
        "local_y",
        "fx",
        "fy",
        "torque",
    ]
)


@dataclass(frozen=True)
class IcyTransport2DRobotEnvConfig(PRBenchEnvConfig):
    """Scene config for IcyTransport2DRobotEnv."""

    # The world is oriented like a standard X/Y coordinate frame.
    world_min_x: float = 0.0
    world_max_x: float = CFG.i_trans_world_size[0]
    world_min_y: float = 0.0
    world_max_y: float = CFG.i_trans_world_size[1]

    # Action space parameters.
    min_forward: float = -2.0
    max_forward: float = 2.0
    min_fsteer: float = -1.0
    max_fsteer: float = 1.0
    max_initial_state_sampling_attempts: int = 100

    # Robot parameters
    robot_base_width: float = CFG.robot_base_width
    robot_base_length: float = CFG.robot_base_length
    robot_mass: float = CFG.robot_base_mass
    robot_init_pose: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x / 4 - CFG.i_trans_obj_pos_rnd_range,
            world_max_y / 4 - CFG.i_trans_obj_pos_rnd_range,
            -np.pi,
        ),
        SE2Pose(
            world_max_x / 4 + CFG.i_trans_obj_pos_rnd_range,
            world_max_y / 4 + CFG.i_trans_obj_pos_rnd_range,
            np.pi,
        ),
    )

    # Room layout parameters
    wall_thickness: float = CFG.i_trans_wall_thickness
    door_width: float = CFG.i_trans_door_width

    # Object parameters, right bottom room
    object1_mass: float = 0.2  # Mass of each object to be transported
    object1_radius: float = (
        CFG.i_transport_object_radius
    )  # Radius of each circular object
    obj1_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x * 9 / 10 + -2 * CFG.i_trans_obj_pos_rnd_range,
            world_max_y / 4 - CFG.i_trans_obj_pos_rnd_range,
            -np.pi,
        ),
        SE2Pose(
            world_max_x * 9 / 10,
            world_max_y / 4 + CFG.i_trans_obj_pos_rnd_range,
            -np.pi + 0.0001,
        ),
    )
    transport1_obj_rgb: tuple[float, float, float] = ORANGE

    # Object parameters, right top room
    object2_mass: float = 0.2  # Mass of each object to be transported
    object2_radius: float = (
        CFG.i_transport_object_radius
    )  # Radius of each circular object
    obj2_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x * 3 / 4 - CFG.i_trans_obj_pos_rnd_range,
            world_max_y * 9 / 10 - 2 * CFG.i_trans_obj_pos_rnd_range,
            -np.pi / 2,
        ),
        SE2Pose(
            world_max_x * 3 / 4 + CFG.i_trans_obj_pos_rnd_range,
            world_max_y * 9 / 10,
            -np.pi / 2 + 0.0001,
        ),
    )
    transport2_obj_rgb: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Target zone parameters, left top room
    target_region_zone_radius: float = CFG.i_transport_object_radius * 2
    target_region_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x / 4 - CFG.i_trans_obj_pos_rnd_range,
            world_max_y * 9 / 10 - 2 * CFG.i_trans_obj_pos_rnd_range,
            -np.pi / 2,
        ),
        SE2Pose(
            world_max_x / 4 + CFG.i_trans_obj_pos_rnd_range,
            world_max_y * 9 / 10,
            -np.pi / 2 + 0.0001,
        ),
    )
    target_region_rgb: tuple[float, float, float] = BROWN

    # Unique region properties
    unique_region_shape: tuple[float, float] = CFG.i_transport_unique_region_shape
    unique_region_offset: float = CFG.i_transport_unique_region_offset
    icy_region_rgb: tuple[float, float, float] = (0.5, 0.8, 1.0)  # Light blue
    icy_region_feature: tuple[float, float, float, float, float] = (
        CFG.i_trans_icy_region_feature
    )

    muddy_region_rgb: tuple[float, float, float] = (0.96, 0.87, 0.70)  # Sandy color
    muddy_region_feature: tuple[float, float, float, float, float] = (
        CFG.i_trans_muddy_region_feature
    )

    # Physics parameters
    sim_dt: float = CFG.sim_dt  # Simulation timestep
    damping: float = 0.1  # Damping applied to all dynamic bodies
    collision_slop: float = 0.001  # Allow small interpenetration

    # For rendering.
    render_dpi: int = 50


_ConfigType = TypeVar("_ConfigType", bound=IcyTransport2DRobotEnvConfig)


class ObjectCentricIcyTransport2DEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, Array, _ConfigType],
    Generic[_ConfigType],
):
    """Four-room transport environment using PyMunk physics with CarRobot.

    The robot must transport an object from one room to another through doorways.

    Room Layout (5x5 world): +-------+-------+ |       |       | |  R1   D1  R2  |  (R =
    Room, D = Door) |       |       | +--D2---+---D3--+ |       |       | |  R3   D4  R4
    | |       |       | +-------+-------+
    """

    def __init__(
        self,
        config: IcyTransport2DRobotEnvConfig = IcyTransport2DRobotEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)  # type: ignore[arg-type]

        # PyMunk physics space
        self.pymunk_space: pymunk.Space | None = None
        self.robot: CarRobot | None = None

        # Initialized by reset().
        self._current_state: ObjectCentricState | None = None
        # Maps state objects to pymunk bodies
        self._state_obj_to_pymunk_body: dict[Object, pymunk.Body | None] = {}
        # Used for collision checking with Geom2D.
        self._static_object_body_cache: dict[Object, MultiBody2D] = {}

        # Track transport and target obj
        self._robot_obj: Object | None = None
        self._transport_obj1: Object | None = None
        self._transport_obj2: Object | None = None
        self._target_obj: Object | None = None
        self._region_objs: dict[int, Object] = {}

        self.region_ids = [int(idx) for idx in CFG.scenario.split(",")]  # 1 or 2 or 3
        # id must be ordered as 1,2, 2,3, 1,3, or 1,2,3 for now.
        assert (
            sorted(self.region_ids) == self.region_ids
        ), "Obstruction ids must be ordered"

        self.robot_is_colliding = False
        self.success = False
        self._prev_colliding = {key: False for key in self.region_ids}

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        types = set(self.type_features)
        return ObjectCentricStateSpace(types)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return CarRobotActionSpace(
            min_forward=config.min_forward,
            max_forward=config.max_forward,
            min_fsteer=config.min_fsteer,
            max_fsteer=config.max_fsteer,
        )

    def _create_constant_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        return create_state_from_dict(initial_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space."""
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, 0  # No gravity in top-down view
        self.pymunk_space.damping = self.config.damping
        self.pymunk_space.collision_slop = self.config.collision_slop

        # Create robot
        init_x = self.config.robot_init_pose[0].x
        init_y = self.config.robot_init_pose[0].y
        init_theta = self.config.robot_init_pose[0].theta
        self.robot = CarRobot(
            init_pos=pymunk.Vec2d(init_x, init_y),
            init_theta=init_theta,
            base_width=self.config.robot_base_width,
            base_length=self.config.robot_base_length,
            mass=self.config.robot_mass,
        )
        self.robot.add_to_space(self.pymunk_space)

        # Set up collision handlers
        # Static wall collisions
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_static_car,
            data=self.robot,
        )
        # Colliding with transport object will
        # directly load it onto the robot.
        self.pymunk_space.on_collision(
            DYNAMIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_object_car,
            data=self.robot,
        )

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create the 4-room layout with walls and doorways."""
        init_state_dict: dict[Object, dict[str, float]] = {}

        wt = self.config.wall_thickness
        dw = self.config.door_width
        world_w = self.config.world_max_x - self.config.world_min_x
        world_h = self.config.world_max_y - self.config.world_min_y

        # Outer walls (world boundaries)
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            -wt / 2,
            wt / 2,
            -wt / 2,
            wt / 2,
        )
        init_state_dict.update(wall_state_dict)

        # Vertical divider (splits left/right) with door D1
        # Bottom part of vertical divider
        vert_div_bottom = Object("vert_wall_divider_bottom", KinRectangleType)
        init_state_dict[vert_div_bottom] = {
            "x": world_w / 2,
            "y": self.config.world_min_y + (world_h / 2 - dw) / 4,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": wt,
            "height": (world_h / 2 - dw) / 2,
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Top part of vertical divider
        vert_div_top = Object("vert_wall_divider_top", KinRectangleType)
        init_state_dict[vert_div_top] = {
            "x": world_w / 2,
            "y": self.config.world_max_y * 3 / 4,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": wt,
            "height": world_h / 2,
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Middle part of vertical divider (between D1 and D4)
        vert_div_middle = Object("vert_wall_divider_middle", KinRectangleType)
        init_state_dict[vert_div_middle] = {
            "x": world_w / 2,
            "y": world_h / 2,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": wt,
            "height": (world_h / 2 - dw),
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Horizontal divider (splits top/bottom) with doors D2, D3, D4
        # Left part (before D2)
        horiz_div_left = Object("horiz_wall_divider_left", KinRectangleType)
        init_state_dict[horiz_div_left] = {
            "x": self.config.world_min_x + (world_h / 2 - dw) / 4,
            "y": world_h / 2,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": (world_w / 2 - dw) / 2,
            "height": wt,
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Middle part (between D2 and D3)
        horiz_div_middle = Object("horiz_wall_divider_middle", KinRectangleType)
        init_state_dict[horiz_div_middle] = {
            "x": world_w / 2,
            "y": world_h / 2,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": world_w / 2 - dw,
            "height": wt,
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        # Right part (after D3)
        horiz_div_right = Object("horiz_wall_divider_right", KinRectangleType)
        init_state_dict[horiz_div_right] = {
            "x": self.config.world_max_x - (world_h / 2 - dw) / 4,
            "y": world_h / 2,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": (world_w / 2 - dw) / 2,
            "height": wt,
            "static": True,
            "held": False,
            "color_r": BLACK[0],
            "color_g": BLACK[1],
            "color_b": BLACK[2],
            "z_order": ZOrder.ALL.value,
        }

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample initial positions for robot and object."""
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose,
                self.np_random,
            )
            # Sample transport object in bottom-right room (R4)
            pose1 = sample_se2_pose(
                self.config.obj1_init_pose_bounds,
                self.np_random,
            )
            pose2 = sample_se2_pose(
                self.config.obj2_init_pose_bounds,
                self.np_random,
            )
            pose3 = sample_se2_pose(
                self.config.target_region_pose_bounds,
                self.np_random,
            )

            poses = [pose1, pose2, pose3]
            rng_order = self.np_random.permutation(3)
            target_zone_pose = poses[rng_order[0]]
            # NOTE: Planning will bias to obj1 to produce shorter low-level plans.
            if rng_order[0] == 0:
                obj1_pose = pose3
                obj2_pose = pose2
            elif rng_order[0] == 1:
                obj1_pose = pose1
                obj2_pose = pose3
            else:
                obj1_pose = pose1
                obj2_pose = pose2

            infront_of_obj1_bools = [
                False,  # placeholder for 0 index
                CFG.icy_infront_of_transport1,
                CFG.muddy_infront_of_transport1,
            ]
            infront_of_obj2_bools = [
                False,  # placeholder for 0 index
                CFG.icy_infront_of_transport2,
                CFG.muddy_infront_of_transport2,
            ]
            infront_of_target_bools = [
                False,  # placeholder for 0 index
                CFG.icy_infront_of_target,
                CFG.muddy_infront_of_target,
            ]

            regions: dict[int, tuple[SE2Pose, tuple[float, float], bool]] = {}
            for reg_id in self.region_ids:
                obstruction_shape = self.config.unique_region_shape
                rel_pose = SE2Pose(
                    self.config.unique_region_offset,
                    0.0,
                    0.0,
                )
                activated = False
                infront_of_transprot1 = infront_of_obj1_bools[reg_id]
                infront_of_transprot2 = infront_of_obj2_bools[reg_id]
                infront_of_target = infront_of_target_bools[reg_id]
                if infront_of_transprot1:
                    region_pose = obj1_pose * rel_pose
                    activated = True
                elif infront_of_transprot2:
                    region_pose = obj2_pose * rel_pose
                    activated = True
                elif infront_of_target:
                    region_pose = target_zone_pose * rel_pose
                    activated = True
                else:
                    region_pose = obj1_pose * rel_pose

                regions[reg_id] = (region_pose, obstruction_shape, activated)

            state = self._create_initial_state(
                robot_init_pose=robot_pose,
                object1_init_pose=obj1_pose,
                object2_init_pose=obj2_pose,
                target_zone_pose=target_zone_pose,
                regions=regions,
            )
            # Check initial state validity: goal not satisfied and no collisions.
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            for region_obj in self._region_objs.values():
                # Remove region objects for collision checking
                all_objects.remove(region_obj)
            # We use Geom2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_init_pose: SE2Pose,
        object1_init_pose: SE2Pose,
        object2_init_pose: SE2Pose,
        target_zone_pose: SE2Pose,
        regions: dict[int, tuple[SE2Pose, tuple[float, float], bool]],
    ) -> ObjectCentricState:

        init_state_dict: dict[Object, dict[str, float]] = {}

        # Just have a robot for now
        robot = Object("robot", RobotType)
        self._robot_obj = robot
        init_state_dict[robot] = {
            "x": robot_init_pose.x,
            "y": robot_init_pose.y,
            "theta": robot_init_pose.theta,
            "vx_base": 0.0,
            "vy_base": 0.0,
            "omega_base": 0.0,
            "width": self.config.robot_base_width,
            "height": self.config.robot_base_length,
            "fx": 0.0,
            "fy": 0.0,
            "torque": 0.0,
            "mass": self.config.robot_mass,
            "moment": 1.0,  # dummy
            "static": False,
            "is_colliding": 0.0,
        }

        # Transport object1
        self._transport_obj1 = Object("transport_obj1", ObjectType)
        init_state_dict[self._transport_obj1] = {
            "x": object1_init_pose.x,
            "y": object1_init_pose.y,
            "theta": object1_init_pose.theta,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "radius": self.config.object1_radius,
            "mass": self.config.object1_mass,
            "static": False,
            "held": False,
            "color_r": self.config.transport1_obj_rgb[0],
            "color_g": self.config.transport1_obj_rgb[1],
            "color_b": self.config.transport1_obj_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Transport object2
        self._transport_obj2 = Object("transport_obj2", ObjectType)
        init_state_dict[self._transport_obj2] = {
            "x": object2_init_pose.x,
            "y": object2_init_pose.y,
            "theta": object2_init_pose.theta,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "radius": self.config.object2_radius,
            "mass": self.config.object2_mass,
            "static": False,
            "held": False,
            "color_r": self.config.transport2_obj_rgb[0],
            "color_g": self.config.transport2_obj_rgb[1],
            "color_b": self.config.transport2_obj_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Target zone
        self._target_obj = Object("target_obj", ObjectType)
        init_state_dict[self._target_obj] = {
            "x": target_zone_pose.x,
            "y": target_zone_pose.y,
            "theta": target_zone_pose.theta,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "radius": self.config.target_region_zone_radius,
            "mass": 0.0,  # dummy
            "static": True,
            "held": False,
            "color_r": self.config.target_region_rgb[0],
            "color_g": self.config.target_region_rgb[1],
            "color_b": self.config.target_region_rgb[2],
            "z_order": ZOrder.FLOOR.value,
        }

        for reg_id, (region_pose, region_shape, activated) in regions.items():
            # Initialize to avoid pylint possibly-used-before-assignment warning
            region_type = IcyRegionType  # Default type
            region_rgb = self.config.icy_region_rgb
            region_feature = self.config.icy_region_feature

            if reg_id == 1:
                region_type = IcyRegionType
                region_rgb = self.config.icy_region_rgb
                region_feature = self.config.icy_region_feature
            elif reg_id == 2:
                region_type = MuddyRegionType
                region_rgb = self.config.muddy_region_rgb
                region_feature = self.config.muddy_region_feature
            elif reg_id == 3:
                raise NotImplementedError("Sandy region not implemented yet.")

            region_obj = Object(f"region{reg_id}", region_type)
            self._region_objs[reg_id] = region_obj
            init_state_dict[region_obj] = {
                "x": region_pose.x,
                "y": region_pose.y,
                "theta": region_pose.theta,
                "vx": 0.0,
                "vy": 0.0,
                "omega": 0.0,
                "width": region_shape[0],
                "height": region_shape[1],
                "static": False,  # for rendering purposes
                "held": False,
                "activated": float(activated),
                "local_x": region_feature[0],
                "local_y": region_feature[1],
                "fx": region_feature[2],
                "fy": region_feature[3],
                "torque": region_feature[4],
                "color_r": region_rgb[0],
                "color_g": region_rgb[1],
                "color_b": region_rgb[2],
                "z_order": ZOrder.FLOOR.value,
            }

        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _reset_robot_in_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Reset the robot in the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        robot_base_x = state.get(obj, "x")
        robot_base_y = state.get(obj, "y")
        robot_theta = state.get(obj, "theta")
        robot_base_vx = state.get(obj, "vx_base")
        robot_base_vy = state.get(obj, "vy_base")
        robot_base_omega = state.get(obj, "omega_base")
        robot_base_fx = state.get(obj, "fx")
        robot_base_fy = state.get(obj, "fy")
        robot_base_torque = state.get(obj, "torque")
        robot_base_vel = (Vec2d(robot_base_vx, robot_base_vy), robot_base_omega)
        robot_base_force = (Vec2d(robot_base_fx, robot_base_fy), robot_base_torque)

        assert self.robot is not None, "Robot not initialized"
        self.robot.reset_positions(
            base_x=robot_base_x,
            base_y=robot_base_y,
            base_theta=robot_theta,
            base_vel=robot_base_vel,
            base_force=robot_base_force,
        )

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        # Add all objects to space
        for obj in state:
            if obj.is_instance(RobotType):
                self._reset_robot_in_space(obj, state)
            elif obj.is_instance(ObjectType) and (not state.get(obj, "static")):
                # Dynamic circular object
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                theta = state.get(obj, "theta")
                radius = state.get(obj, "radius")
                mass = state.get(obj, "mass")

                moment = pymunk.moment_for_circle(mass, 0, radius)
                body = pymunk.Body(mass=mass, moment=moment)
                circle_shape: pymunk.Shape = pymunk.Circle(body, radius)
                circle_shape.friction = 1.0
                circle_shape.density = 1.0
                circle_shape.mass = mass
                circle_shape.collision_type = DYNAMIC_COLLISION_TYPE

                self.pymunk_space.add(body, circle_shape)
                body.angle = theta
                body.position = x, y

                self._state_obj_to_pymunk_body[obj] = body

            elif obj.is_instance(KinRectangleType):
                if "region" in obj.name:
                    # region is not a physical object
                    continue
                # Static wall
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")

                # Static kinematic body
                b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                vs = [
                    (-width / 2, -height / 2),
                    (-width / 2, height / 2),
                    (width / 2, height / 2),
                    (width / 2, -height / 2),
                ]
                poly_shape: pymunk.Shape = pymunk.Poly(b2, vs)
                poly_shape.friction = 1.0
                poly_shape.density = 1.0
                poly_shape.mass = 1.0
                poly_shape.elasticity = 0.99
                poly_shape.collision_type = STATIC_COLLISION_TYPE
                self.pymunk_space.add(b2, poly_shape)
                b2.position = x, y
                b2.angle = theta
                self._state_obj_to_pymunk_body[obj] = b2

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if "wall" in obj.name or "region" in obj.name:
                # Static walls don't change
                continue
            if obj.is_instance(RobotType):
                # Update robot state from its body
                assert self.robot is not None, "Robot not initialized"
                state.set(obj, "x", self.robot.base_pose.x)
                state.set(obj, "y", self.robot.base_pose.y)
                state.set(obj, "theta", self.robot.base_pose.theta)
                state.set(obj, "vx_base", self.robot.base_vel[0].x)
                state.set(obj, "vy_base", self.robot.base_vel[0].y)
                state.set(obj, "omega_base", self.robot.base_vel[1])
                # Set force values to 0 (they're set during actions)
                state.set(obj, "fx", self.robot.base_force[0].x)
                state.set(obj, "fy", self.robot.base_force[0].y)
                state.set(obj, "torque", self.robot.base_force[1])
                state.set(obj, "mass", self.robot.base_mass_moment[0])
                state.set(obj, "moment", self.robot.base_mass_moment[1])
                if self.robot_is_colliding:
                    state.set(obj, "is_colliding", 1.0)
                else:
                    state.set(obj, "is_colliding", 0.0)
            else:
                if obj.is_instance(ObjectType) and state.get(obj, "static"):
                    # Static objects (regions) don't change here
                    continue
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                if pymunk_body is None:
                    state.set(obj, "held", False)
                    state.set(obj, "static", True)
                    # already delivered object
                    continue
                # Update object state from body
                # NOTE: Clip positions to be within world boundaries, for numerical stability
                pos_x = np.clip(
                    pymunk_body.position.x,
                    self.config.world_min_x,
                    self.config.world_max_x,
                )
                pos_y = np.clip(
                    pymunk_body.position.y,
                    self.config.world_min_y,
                    self.config.world_max_y,
                )
                state.set(obj, "x", pos_x)
                state.set(obj, "y", pos_y)
                state.set(obj, "theta", pymunk_body.angle)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)
                if self.robot is not None and pymunk_body == self.robot._base_body:
                    # If the object is being held by the robot, mark it as held
                    state.set(obj, "held", True)

        for reg_id, region_obj in self._region_objs.items():
            region_x = state.get(region_obj, "x")
            region_y = state.get(region_obj, "y")
            region_theta = state.get(region_obj, "theta")
            region_se2 = SE2Pose(region_x, region_y, region_theta)
            assert self._robot_obj is not None
            robot_x = state.get(self._robot_obj, "x")
            robot_y = state.get(self._robot_obj, "y")
            robot_theta = state.get(self._robot_obj, "theta")
            robot_se2 = SE2Pose(robot_x, robot_y, robot_theta)
            rel_se2 = region_se2.inverse * robot_se2
            rel_x_ok = rel_se2.x <= 0.0  # Behind the robot, drived through
            if (
                not self.robot_is_colliding
                and self._prev_colliding[reg_id]
                and rel_x_ok
            ):
                self._prev_colliding[reg_id] = False
                # Deactivate region after collision ends
                state.set(region_obj, "activated", 0.0)
        # Update the current state
        self._current_state = state

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on object position."""
        assert self._current_state is not None
        assert self._transport_obj1 is not None, "Transport object 1 not initialized"
        assert self._transport_obj2 is not None, "Transport object 2 not initialized"
        assert self._target_obj is not None, "Target object not initialized"

        # Check if either transport object reaches the target
        transport_obj1_x = self._current_state.get(self._transport_obj1, "x")
        transport_obj1_y = self._current_state.get(self._transport_obj1, "y")
        transport_obj2_x = self._current_state.get(self._transport_obj2, "x")
        transport_obj2_y = self._current_state.get(self._transport_obj2, "y")
        target_obj_x = self._current_state.get(self._target_obj, "x")
        target_obj_y = self._current_state.get(self._target_obj, "y")

        dist_to_goal1 = np.sqrt(
            (transport_obj1_x - target_obj_x) ** 2
            + (transport_obj1_y - target_obj_y) ** 2
        )
        dist_to_goal2 = np.sqrt(
            (transport_obj2_x - target_obj_x) ** 2
            + (transport_obj2_y - target_obj_y) ** 2
        )

        if (
            dist_to_goal1 <= self.config.target_region_zone_radius
            and dist_to_goal2 <= self.config.target_region_zone_radius
        ):
            self.success = True
            return 1.0, True

        return 0.0, False

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Dynamic2DRobotEnvTypeFeatures

    def _get_obs(self) -> ObjectCentricState:
        """Get observation by reading from the physics simulation."""
        self._read_state_from_space()
        assert self._current_state is not None, "Need to call reset()"
        return self._current_state.copy()

    def _get_info(self) -> dict:
        """Return info dict with collision and goal information."""
        return {
            "is_colliding": self.robot_is_colliding,
            "success": self.success,
        }

    @property
    def full_state(self) -> ObjectCentricState:
        """Get the full state, which includes both dynamic and static objects."""
        if self._current_state is None:
            raise RuntimeError("Current state is not initialized")
        full_state = self._current_state.copy()
        # Merge the initial constant state with the current state.
        full_state.data.update(self.initial_constant_state.data)
        return full_state

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        # Reset the random seed.
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        gymnasium.Env.reset(self, seed=seed)
        self.robot_is_colliding = False
        self.success = False
        self._prev_colliding = {key: False for key in self.region_ids}

        # Clear existing physics space
        if self.pymunk_space:
            # Remove all bodies and shapes
            for body in list(self.pymunk_space.bodies):
                for shape in list(body.shapes):
                    if body in self.pymunk_space.bodies:
                        self.pymunk_space.remove(body, shape)
            for shape in list(self.pymunk_space.shapes):
                # Some shapes are not attached to bodies (e.g., static lines)
                self.pymunk_space.remove(shape)

        # Set up new physics space
        self._setup_physics_space()
        self._static_object_body_cache = {}
        self._state_obj_to_pymunk_body = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()
            # logging.warning(
            #     "Resetting dynamic2d with a provided initial state is unstable, \
            #     replaying the same action won't produce the same result."
            # )
        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        # Add objects to physics space
        self._add_state_to_space(self.full_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        f_forward_y, f_forward_x, f_steer = action
        assert self._current_state is not None, "Need to call reset()"
        assert self.pymunk_space is not None, "Space not initialized"
        assert self.robot is not None, "Robot not initialized"
        assert self._robot_obj is not None, "Robot object not initialized"

        # First see if the held object is delivered to the target
        if len(self.robot.held_objects) > 0:
            assert self._transport_obj1 is not None, "Transport obj1 not initialized"
            assert self._transport_obj2 is not None, "Transport obj2 not initialized"
            if self._current_state.get(self._transport_obj1, "held"):
                held_obj = self._transport_obj1
            else:
                assert self._current_state.get(self._transport_obj2, "held")
                held_obj = self._transport_obj2
            assert self._target_obj is not None, "Target object not initialized"
            target_x = self._current_state.get(self._target_obj, "x")
            target_y = self._current_state.get(self._target_obj, "y")
            held_obj_x = self._current_state.get(held_obj, "x")
            held_obj_y = self._current_state.get(held_obj, "y")
            dist_to_target = np.sqrt(
                (held_obj_x - target_x) ** 2 + (held_obj_y - target_y) ** 2
            )
            if dist_to_target <= self.config.target_region_zone_radius:
                # Delivered the object, remove it from the robot and simulation
                self.robot.remove_from_cart()
                assert held_obj in self._state_obj_to_pymunk_body
                self._state_obj_to_pymunk_body[held_obj] = None

        # Apply forces to robot
        self.robot.update(
            forward_force_y=f_forward_y,
            forward_force_x=f_forward_x,
            steering_force=f_steer,
        )
        current_held_obj_num = len(self.robot.held_objects)

        robot_multibody = object_to_multibody2d(
            self._robot_obj, self._current_state, self._static_object_body_cache
        )
        self.robot_is_colliding = False
        for reg_id, region_obj in self._region_objs.items():
            if self._current_state.get(region_obj, "activated") < 0.5:
                continue
            region_multibody = object_to_multibody2d(
                region_obj, self.full_state, self._static_object_body_cache
            )
            assert len(region_multibody.bodies) == 1
            if geom2ds_intersect(
                robot_multibody.bodies[0].geom, region_multibody.bodies[0].geom
            ):
                self.robot_is_colliding = True
                self._prev_colliding[reg_id] = True
                # Apply region effect
                local_x = self._current_state.get(region_obj, "local_x")
                local_y = self._current_state.get(region_obj, "local_y")
                fx = self._current_state.get(region_obj, "fx")
                fy = self._current_state.get(region_obj, "fy")
                torque = self._current_state.get(region_obj, "torque")
                if current_held_obj_num > 0:
                    # If holding an object, halve additional slowing down effect
                    fy += CFG.icy_inhand_fy_penalty
                # Rotate local force to world frame
                self.robot.external_effect_force_torque(
                    Vec2d(fx, fy), Vec2d(local_x, local_y), torque
                )

        # Step physics simulation
        self.pymunk_space.step(self.config.sim_dt)

        new_held_obj_num = len(self.robot.held_objects)
        if new_held_obj_num > current_held_obj_num:
            assert current_held_obj_num == 0, "Can only hold one object at a time"
            assert new_held_obj_num == current_held_obj_num + 1
            for state_obj, _ in self._state_obj_to_pymunk_body.items():
                if state_obj.name in ["transport_obj1", "transport_obj2"]:
                    obj_x = self._current_state.get(state_obj, "x")
                    obj_y = self._current_state.get(state_obj, "y")
                    robot_x = self._current_state.get(self._robot_obj, "x")
                    robot_y = self._current_state.get(self._robot_obj, "y")
                    dist = np.sqrt((obj_x - robot_x) ** 2 + (obj_y - robot_y) ** 2)
                    if dist < self.config.robot_base_length:
                        # Transport object being held, its body is now the robot's base body
                        self._state_obj_to_pymunk_body[state_obj] = (
                            self.robot._base_body
                        )
                        break

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the current state."""
        assert self.render_mode == "rgb_array"
        assert self._current_state is not None, "Need to call reset()"
        render_input_state = self._current_state.copy()
        # Merge the initial constant state with the current state.
        render_input_state.data.update(self.initial_constant_state.data)
        return render_2dstate(
            render_input_state,
            self._static_object_body_cache,
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            self.config.render_dpi,
        )

    def get_action_from_gui_input(self, gui_input: dict[str, Any]) -> NDArray[Any]:
        """Get the mapping from human inputs to actions."""
        assert isinstance(self.action_space, CarRobotActionSpace)
        return get_car_robot_action_from_gui_input(self.action_space, gui_input)  # type: ignore[arg-type]


class IcyTransport2DEnv(ConstantObjectPRBenchEnv):
    """Icy Transport 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricIcyTransport2DEnv:
        return ObjectCentricIcyTransport2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        # All objects are constant (walls, robot, transport objects)
        constant_objects = []
        for i in CFG.scenario.split(","):
            # obstructions are always named by scenario index
            constant_objects.append(f"region{i}")
        constant_objects.extend(
            ["transport_obj1", "transport_obj2", "target_obj", "robot"]
        )
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""
        return "Four-room transport environment where a car robot must push an object to a goal room"

    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""
        return "Sparse reward: +100 when object reaches goal room, 0 otherwise"

    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""
        return "Inspired by MiniGrid Four Rooms environment"
