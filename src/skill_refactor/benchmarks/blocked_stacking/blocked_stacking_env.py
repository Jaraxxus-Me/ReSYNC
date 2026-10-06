"""Blocked Stacking Environment based on PRBench Dynamic2D Environment."""

import glob
import pickle
from dataclasses import dataclass

import numpy as np
import pymunk
from gymnasium.utils import seeding
from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from prbench.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    FINGER_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobot,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
    on_collision_w_static,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose, ZOrder
from prbench.envs.utils import (
    BLACK,
    PURPLE,
    rectangle_object_to_geom,
    sample_se2_pose,
    state_2d_has_collision,
)
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from skill_refactor.settings import CFG
from skill_refactor.utils.structs import Task

BLOCK_COLLISION_TYPE = 99
BLOCK_HELD_COLLISION_TYPE = 100

# Define custom object types for the obstruction environment
RobotType = Type("robot", parent=KinRobotType)
BlockType = Type("block", parent=DynRectangleType)
ObstructionRecType = Type("obstruction_rec", parent=KinRectangleType)
ObstructionTraType = Type("obstruction_tra", parent=KinRectangleType)
ObstructionThiType = Type("obstruction_thi", parent=KinRectangleType)
Dynamic2DRobotEnvTypeFeatures[BlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType]
)
Dynamic2DRobotEnvTypeFeatures[ObstructionRecType] = [
    "x",
    "y",
    "theta",
    "vx",
    "vy",
    "omega",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
    "width",
    "height",
]
Dynamic2DRobotEnvTypeFeatures[RobotType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRobotType] + ["is_colliding"]
)
Dynamic2DRobotEnvTypeFeatures[ObstructionTraType] = list(
    Dynamic2DRobotEnvTypeFeatures[ObstructionRecType]
)
Dynamic2DRobotEnvTypeFeatures[ObstructionThiType] = list(
    Dynamic2DRobotEnvTypeFeatures[ObstructionRecType]
)


def on_gripper_grasp(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot
) -> None:
    """Collision callback for gripper grasping objects.

    Here the grasp object becomes BLOCK_HELD_COLLISION_TYPE so that grasping will not
    make robot frozen.
    """
    dynamic_body = arbiter.bodies[0]
    if robot.is_grasping(arbiter.contact_point_set, dynamic_body):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        points = arbiter.shapes[0].get_vertices()
        shape = pymunk.Poly(kinematic_body, points)
        shape.friction = 1
        shape.density = 1.0
        # Held object becomes part of the robot.
        shape.collision_type = BLOCK_HELD_COLLISION_TYPE
        space.add(kinematic_body, shape)
        robot.add_to_hand((kinematic_body, [shape]), dynamic_body.mass)
        # Remove the dynamic body from the space
        space.remove(dynamic_body, arbiter.shapes[0])


@dataclass(frozen=True)
class BlockedStackingEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene specification for BlockedStacking2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 1.4 + 1.4 * np.sqrt(5)  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 2.0

    # Robot parameters
    init_robot_pos: tuple[float, float] = (1.5, 1.5)
    robot_base_radius: float = CFG.blocked2d_robot_base_radius
    robot_arm_length_max: float = CFG.blocked2d_robot_arm_length_max
    gripper_base_width: float = CFG.blocked2d_gripper_base_width
    gripper_base_height: float = CFG.blocked2d_gripper_base_height
    gripper_finger_width: float = CFG.blocked2d_gripper_finger_width
    gripper_finger_height: float = CFG.blocked2d_gripper_finger_height

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x + (robot_base_radius + gripper_base_height), 1.2, -np.pi),
        SE2Pose(world_max_x - (robot_base_radius + gripper_base_height), 1.5, 0.0),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined at the center
    table_pose: SE2Pose = SE2Pose(
        world_min_x + table_width / 2, world_min_y + table_height / 2, 0.0
    )

    # Grasp block.
    grasp_block_rgb: tuple[float, float, float] = PURPLE
    block_height: float = gripper_base_height - 1.6 * gripper_finger_height
    block_width: float = gripper_base_height - 1.6 * gripper_finger_height
    block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 6 * block_height,
            table_pose.y + table_height * 0.8 + 1e-6,
            0.0,
        ),
        SE2Pose(
            world_max_x - (robot_base_radius + gripper_base_height),
            table_pose.y + table_height * 0.8 + 1e-6,
            0.0,
        ),
    )
    grasp_block_mass: float = 1.0

    # Base block.
    base_block_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    base_block_mass: float = 1.0

    min_block_dist: float = 8 * block_width  # min distance between blocks in x axis

    # Obstruction.
    obstruction_rgb: tuple[float, float, float] = BLACK

    # Scenario 1
    obstruction_shape1: tuple[float, float] = (
        5 * gripper_base_height / 12,
        1.1 * gripper_base_width,
    )
    obstruction1_rel_pose_bounds = (
        SE2Pose(gripper_base_height * 0.50, 1.8 * gripper_finger_height, -np.pi / 12),
        SE2Pose(gripper_base_height * 0.52, 2 * gripper_finger_height, np.pi / 12),
    )

    # Scenario 2
    obstruction_shape2: tuple[float, float] = (block_width * 1.6, block_width * 2.0)
    obstruction2_rel_pose_bounds = (
        SE2Pose(-block_width * 1.35, block_width * 2.0, 0.0),
        SE2Pose(-block_width * 1.32, block_width * 2.0, 0.0),
    )

    # Scenario 3
    obstruction_shape3: tuple[float, float] = (gripper_finger_height, block_width * 2.0)
    obstruction3_rel_pose_bounds = (
        SE2Pose(
            -block_width * 0.85 - gripper_finger_height,
            block_width * 0.4,
            -np.pi / 6 - 0.05,
        ),
        SE2Pose(
            -block_width * 0.8 - gripper_finger_height, block_width * 0.45, -np.pi / 6
        ),
    )

    # Goal parameters.
    on_tol_dy_overlap: float = 1e-2  # distance tolerance for "on" relation
    on_tol_dy_relative: float = 1e-2  # distance tolerance for "on" relation
    on_tol_vel: float = 1e-2  # velocity tolerance for "on" relation

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 100


class ObjectCentricBlockedStacking2DEnv(
    ObjectCentricDynamic2DRobotEnv[BlockedStackingEnvConfig]
):
    """**Task Description:** A simple task where the objective is to stack a block on
    top of another block with possible obstructions.

    **(Important) Observation:**
      13*num_obstructions
    - [0:6]: [x, y, theta, vx, vy, omega] of obstruction1
    - [11:13]: [w, h] of the obstruction1
    - [13:19]: [x, y, theta, vx, vy, omega] of obstruction2 (if any)
    - [24:26]: [w, h] of the obstruction2 (if any)
    - [26:32]: [x, y, theta, vx, vy, omega] of obstruction3 (if any)
    - [37:39]: [w, h] of the obstruction3 (if any)

    - [13*num_obstructions:13*num_obstructions+6]:
        [x, y, theta, vx, vy, omega] of the grasp block
    - [13*num_obstructions+12:13*num_obstructions+14]:
        [w, h] of the grasp block
    - [13*num_obstructions+7]:
        is_grasped for the grasp block (1 if the robot is holding the block, 0 otherwise)
    - [13*num_obstructions+15, 13*num_obstructions+21]:
        [x, y, theta, vx, vy, omega] of the base block
    - [13*num_obstructions+27, 13*num_obstructions+29]:
        [w, h] of the base block
    - [13*num_obstructions+22]:
        is_grasped for the base block (1 if the robot is holding the block, 0 otherwise)
    - [13*num_obstructions+30, 13*num_obstructions+36]:
        [x, y, theta, vx, vy, omega] of the robot base
    - [13*num_obstructions+47]:
        robot arm joint position
    - [13*num_obstructions+51]:
        robot gripper gap
    - [-1]: is_colliding (1 if the robot is colliding with the static objects, 0 otherwise)

    **Success Conditions:**
    - the object position is within `goal_thresh` euclidean distance of the goal position
    - the object is not held
    """

    def __init__(
        self,
        config: BlockedStackingEnvConfig = BlockedStackingEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.obstruction_ids = [
            int(idx) for idx in CFG.scenario.split(",")
        ]  # 1 or 2 or 3
        # id must be ordered as 1,2, 2,3, 1,3, or 1,2,3 for now.
        assert (
            sorted(self.obstruction_ids) == self.obstruction_ids
        ), "Obstruction ids must be ordered"

        # Store object references for tracking
        self._robot_obj: Object | None = None
        self._grasp_block: Object | None = None
        self._base_block: Object | None = None
        self._obstruction: Object | None = None
        # Provided in info dict
        self.elapsed_steps = 0
        self.robot_is_colliding = False
        self.grasped_obj_name = ""
        self.success = False

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space."""
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, self.config.gravity_y
        self.pymunk_space.collision_slop = self.config.collision_slop

        # Create robot
        self.robot = KinRobot(
            init_pos=pymunk.Vec2d(*self.config.init_robot_pos),
            base_radius=self.config.robot_base_radius,
            arm_length_max=self.config.robot_arm_length_max,
            gripper_base_width=self.config.gripper_base_width,
            gripper_base_height=self.config.gripper_base_height,
            gripper_finger_width=self.config.gripper_finger_width,
            gripper_finger_height=self.config.gripper_finger_height,
            base_collision_type=ROBOT_COLLISION_TYPE,
            arm_collision_type=ROBOT_COLLISION_TYPE,
            finger_collision_type=FINGER_COLLISION_TYPE,
        )
        self.robot.add_to_space(self.pymunk_space)

        # Set up collision handlers
        # Only allow grasping blocks
        self.pymunk_space.on_collision(
            BLOCK_COLLISION_TYPE,
            FINGER_COLLISION_TYPE,
            post_solve=on_gripper_grasp,
            data=self.robot,
        )
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_static,
            data=self.robot,
        )
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            FINGER_COLLISION_TYPE,
            pre_solve=on_collision_w_static,
            data=self.robot,
        )
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            BLOCK_HELD_COLLISION_TYPE,
            pre_solve=on_collision_w_static,
            data=self.robot,
        )

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", ObstructionRecType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "vx": 0.0,
            "y": self.config.table_pose.y,
            "vy": 0.0,
            "theta": self.config.table_pose.theta,
            "omega": 0.0,
            "width": self.config.table_width,
            "height": self.config.table_height,
            "static": True,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.FLOOR.value,
        }

        # Create room walls.
        assert isinstance(self.action_space, KinRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose_bounds, self.np_random
            )
            grasp_block_pose = sample_se2_pose(
                self.config.block_init_pose_bounds, self.np_random
            )
            base_block_pose = sample_se2_pose(
                self.config.block_init_pose_bounds, self.np_random
            )
            if abs(grasp_block_pose.x - base_block_pose.x) < self.config.min_block_dist:
                # Ensure the two blocks are not too close in x axis.
                continue

            grasp_block_shape = (self.config.block_width, self.config.block_height)
            base_block_shape = (self.config.block_width, self.config.block_height)

            obstruction_shapes = [
                (0.0, 0.0),  # placeholder for 0 index
                self.config.obstruction_shape1,
                self.config.obstruction_shape2,
                self.config.obstruction_shape3,
            ]
            rel_poses = [
                SE2Pose(0.0, 0.0, 0.0),  # placeholder for 0 index
                sample_se2_pose(
                    self.config.obstruction1_rel_pose_bounds,
                    self.np_random,
                ),
                sample_se2_pose(
                    self.config.obstruction2_rel_pose_bounds,
                    self.np_random,
                ),
                sample_se2_pose(
                    self.config.obstruction3_rel_pose_bounds,
                    self.np_random,
                ),
            ]
            adjust_ys = [
                0.0,  # placeholder for 0 index
                (self.config.block_height + obstruction_shapes[0][1]) / 2,
                0.0,
                0.0,
            ]
            blocking_grasp_bools = [
                False,  # placeholder for 0 index
                CFG.obstruction1_blocking_grasp,
                CFG.obstruction2_blocking_grasp,
                CFG.obstruction3_blocking_grasp,
            ]
            blocking_stacking_bools = [
                False,  # placeholder for 0 index
                CFG.obstruction1_blocking_stacking,
                CFG.obstruction2_blocking_stacking,
                CFG.obstruction3_blocking_stacking,
            ]

            obstructions: dict[int, tuple[SE2Pose, tuple[float, float]]] = {}
            for obs_id in self.obstruction_ids:
                obstruction_shape = obstruction_shapes[obs_id]
                rel_pose = rel_poses[obs_id]
                adjust_y = adjust_ys[obs_id]
                blocking_grasp = blocking_grasp_bools[obs_id]
                blocking_stacking = blocking_stacking_bools[obs_id]
                assert (not blocking_grasp and blocking_stacking) or (
                    blocking_grasp and not blocking_stacking
                ), "One obstruction must block either grasping or stacking"

                adjusted_rel_pose = SE2Pose(
                    rel_pose.x,
                    rel_pose.y + adjust_y,
                    rel_pose.theta,
                )
                if blocking_grasp:
                    obstruction_pose = grasp_block_pose * adjusted_rel_pose
                else:
                    assert (
                        blocking_stacking
                    ), "One obstruction must block either grasping or stacking"
                    obstruction_pose = base_block_pose * adjusted_rel_pose
                obstructions[obs_id] = (obstruction_pose, obstruction_shape)

            state = self._create_initial_state(
                robot_pose,
                grasp_block_pose,
                grasp_block_shape,
                base_block_pose,
                base_block_shape,
                obstructions,
            )

            # Check initial state validity: goal not satisfied and no collisions.
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            # We use Geom2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        grasp_block_pose: SE2Pose,
        grasp_block_shape: tuple[float, float],
        base_block_pose: SE2Pose,
        base_block_shape: tuple[float, float],
        obstructions: dict[int, tuple[SE2Pose, tuple[float, float]]],
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", RobotType)
        self._robot_obj = robot
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "vx_base": 0.0,
            "vy_base": 0.0,
            "omega_base": 0.0,
            "vx_arm": 0.0,
            "vy_arm": 0.0,
            "omega_arm": 0.0,
            "vx_gripper_l": 0.0,
            "vy_gripper_l": 0.0,
            "omega_gripper_l": 0.0,
            "vx_gripper_r": 0.0,
            "vy_gripper_r": 0.0,
            "omega_gripper_r": 0.0,
            "static": False,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length_max,
            "gripper_base_width": self.config.gripper_base_width,
            "gripper_base_height": self.config.gripper_base_height,
            "finger_gap": self.config.gripper_base_height,
            "finger_height": self.config.gripper_finger_height,
            "finger_width": self.config.gripper_finger_width,
            "is_colliding": 0.0,
        }

        # Create the block to be grasped.
        grasp_block = Object("grasp_block", BlockType)
        self._grasp_block = grasp_block
        init_state_dict[grasp_block] = {
            "x": grasp_block_pose.x,
            "vx": 0.0,
            "y": grasp_block_pose.y + grasp_block_shape[1] / 2,
            "vy": 0.0,
            "theta": grasp_block_pose.theta,
            "omega": 0.0,
            "width": grasp_block_shape[0],
            "height": grasp_block_shape[1],
            "static": False,
            "mass": self.config.grasp_block_mass,
            "color_r": self.config.grasp_block_rgb[0],
            "color_g": self.config.grasp_block_rgb[1],
            "color_b": self.config.grasp_block_rgb[2],
            "z_order": ZOrder.ALL.value,
            "held": 0.0,
        }

        # Create the base block.
        base_block = Object("base_block", BlockType)
        self._base_block = base_block
        init_state_dict[base_block] = {
            "x": base_block_pose.x,
            "vx": 0.0,
            "y": base_block_pose.y + base_block_shape[1] / 2,
            "vy": 0.0,
            "theta": base_block_pose.theta,
            "omega": 0.0,
            "width": base_block_shape[0],
            "height": base_block_shape[1],
            "static": False,
            "mass": self.config.base_block_mass,
            "color_r": self.config.base_block_rgb[0],
            "color_g": self.config.base_block_rgb[1],
            "color_b": self.config.base_block_rgb[2],
            "z_order": ZOrder.ALL.value,
            "held": 0.0,
        }

        # Create obstructions.
        static_s = [False, True, False, True]
        types = [
            ObstructionRecType,
            ObstructionRecType,
            ObstructionTraType,
            ObstructionThiType,
        ]
        for obs_id, (obstruction_pose, obstruction_shape) in obstructions.items():
            y_s = [
                0.0,
                obstruction_pose.y + obstruction_shape[1] / 2,
                obstruction_pose.y,
                obstruction_pose.y,
            ]
            static = static_s[obs_id]
            y = y_s[obs_id]
            obstruction_type = types[obs_id]

            obstruction = Object(f"obstruction{obs_id}", obstruction_type)
            init_state_dict[obstruction] = {
                "x": obstruction_pose.x,
                "vx": 0.0,
                "y": y,
                "vy": 0.0,
                "theta": obstruction_pose.theta,
                "omega": 0.0,
                "width": obstruction_shape[0],
                "height": obstruction_shape[1],
                "static": static,
                "color_r": self.config.obstruction_rgb[0],
                "color_g": self.config.obstruction_rgb[1],
                "color_b": self.config.obstruction_rgb[2],
                "z_order": ZOrder.SURFACE.value,  # To avoid collision with table
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        # Add static objects (table, walls)
        for obj in state:
            if obj.is_instance(RobotType):
                self._reset_robot_in_space(obj, state)
            elif obj.is_instance(DynRectangleType) or obj.is_instance(KinRectangleType):
                # Everything else are rectangles in this environment.
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")

                if state.get(obj, "static"):
                    # Static objects
                    # We use Pymunk kinematic bodies for static objects
                    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(b2, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.mass = 1.0
                    shape.elasticity = 0.99
                    if obj.name == "table":
                        # NOTE: Table should not collide with held blocks
                        # otherwise robot finger will be frozen.
                        shape.collision_type = BLOCK_COLLISION_TYPE
                    else:
                        shape.collision_type = STATIC_COLLISION_TYPE
                    self.pymunk_space.add(b2, shape)
                    b2.position = x, y
                    b2.angle = theta
                    self._state_obj_to_pymunk_body[obj] = b2
                else:
                    # Dynamic objects
                    mass = self.config.grasp_block_mass
                    moment = pymunk.moment_for_box(mass, (width, height))
                    body = pymunk.Body()
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(body, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    if "block" in obj.name:
                        shape.collision_type = BLOCK_COLLISION_TYPE
                    else:
                        shape.collision_type = DYNAMIC_COLLISION_TYPE
                    shape.mass = mass
                    assert shape.body is not None
                    shape.body.moment = moment
                    shape.body.mass = mass
                    self.pymunk_space.add(body, shape)
                    body.angle = theta
                    body.position = x, y
                    self._state_obj_to_pymunk_body[obj] = body

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if state.get(obj, "static"):
                continue
            if obj.is_instance(KinRobotType):
                # Update robot state from its body
                assert self.robot is not None, "Robot not initialized"
                robot_obj = state.get_objects(KinRobotType)[0]
                state.set(robot_obj, "x", self.robot.base_pose.x)
                state.set(robot_obj, "y", self.robot.base_pose.y)
                state.set(robot_obj, "theta", self.robot.base_pose.theta)
                state.set(robot_obj, "vx_base", self.robot.base_vel[0].x)
                state.set(robot_obj, "vy_base", self.robot.base_vel[0].y)
                state.set(robot_obj, "omega_base", self.robot.base_vel[1])
                state.set(robot_obj, "arm_joint", self.robot.curr_arm_length)
                state.set(robot_obj, "vx_arm", self.robot.gripper_base_vel[0].x)
                state.set(robot_obj, "vy_arm", self.robot.gripper_base_vel[0].y)
                state.set(robot_obj, "omega_arm", self.robot.gripper_base_vel[1])
                state.set(robot_obj, "finger_gap", self.robot.curr_gripper)
                state.set(robot_obj, "vx_gripper_l", self.robot.finger_vel_l[0].x)
                state.set(robot_obj, "vy_gripper_l", self.robot.finger_vel_l[0].y)
                state.set(robot_obj, "omega_gripper_l", self.robot.finger_vel_l[1])
                state.set(robot_obj, "vx_gripper_r", self.robot.finger_vel_r[0].x)
                state.set(robot_obj, "vy_gripper_r", self.robot.finger_vel_r[0].y)
                state.set(robot_obj, "omega_gripper_r", self.robot.finger_vel_r[1])
            else:
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
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
                if obj.is_instance(BlockType):
                    if obj.name == self.grasped_obj_name:
                        state.set(obj, "held", 1.0)
                    else:
                        state.set(obj, "held", 0.0)

        # Update the current state
        self._current_state = state

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target condition is satisfied.

        This is borrowed from geom2d obstruction env for now.
        """
        # Find grasp and base blocks dynamically
        grasp_block = None
        base_block = None
        for obj in state:
            if obj.name == "grasp_block":
                grasp_block = obj
            elif obj.name == "base_block":
                base_block = obj

        if grasp_block is None or base_block is None:
            return False
        geom_overlap = False
        top_geom = rectangle_object_to_geom(
            state, grasp_block, static_object_body_cache
        )
        bottom_geom = rectangle_object_to_geom(
            state, base_block, static_object_body_cache
        )
        offset_top_geom = Rectangle(
            top_geom.x,
            top_geom.y - self.config.on_tol_dy_overlap,
            top_geom.width,
            top_geom.height,
            top_geom.theta,
        )
        if geom2ds_intersect(offset_top_geom, bottom_geom):
            geom_overlap = True
        still_vel = False
        vels = np.array(
            [
                state.get(grasp_block, "vx"),
                state.get(grasp_block, "vy"),
                state.get(grasp_block, "omega"),
                state.get(base_block, "vx"),
                state.get(base_block, "vy"),
                state.get(base_block, "omega"),
            ]
        )
        # Ensure both blocks are still
        if np.linalg.norm(vels) < self.config.on_tol_vel:
            still_vel = True

        rel_dy = state.get(grasp_block, "y") - state.get(base_block, "y")
        above = abs(
            rel_dy
            - (
                state.get(base_block, "height") / 2
                + state.get(grasp_block, "height") / 2
            )
        )

        above_relation = above < self.config.on_tol_dy_relative
        return (
            geom_overlap
            and still_vel
            and above_relation
            and (self.grasped_obj_name == "")
        )

    def _get_reward_and_done(self):
        """Calculate reward and termination."""
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        if terminated:
            self.success = True
            return 1.0, terminated
        return 0.0, terminated

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        """Step the environment with the given action.

        Here we additional detect grasping and collisions.
        """
        assert self.robot is not None, "Robot not initialized"
        dx, dy, dtheta, darm, dgripper = action
        # Calculate target positions
        tgt_x = self.robot.base_pose.x + dx
        tgt_y = self.robot.base_pose.y + dy
        tgt_theta = self.robot.base_pose.theta + dtheta
        tgt_arm = max(
            min(self.robot.curr_arm_length + darm, self.robot.arm_length_max),
            self.robot.base_radius,
        )
        tgt_gripper = max(
            min(self.robot.curr_gripper + dgripper, self.robot.gripper_gap_max),
            self.robot.gripper_finger_height * 2,
        )
        # Potentially check for held object collisions
        assert self._robot_obj is not None, "Robot object not initialized"
        assert self._current_state is not None, "Current state not initialized"

        tgt_held_x = None
        tgt_held_y = None
        tgt_held_theta = None
        held_obj = None
        if self.grasped_obj_name != "":
            # If holding an object, check its collisions too
            assert self._grasp_block is not None and self._base_block is not None
            held_obj = (
                self._grasp_block
                if self.grasped_obj_name == self._grasp_block.name
                else self._base_block
            )
            # It is fixed to the robot gripper, so its relative pose to the robot
            # remains the same.
            tgt_held_x = self._current_state.get(held_obj, "x") + dx
            tgt_held_y = self._current_state.get(held_obj, "y") + dy
            tgt_held_theta = self._current_state.get(held_obj, "theta") + dtheta
        collision_state = self._current_state.copy()
        collision_state.set(self._robot_obj, "x", tgt_x)
        collision_state.set(self._robot_obj, "y", tgt_y)
        collision_state.set(self._robot_obj, "theta", tgt_theta)
        collision_state.set(self._robot_obj, "arm_joint", tgt_arm)
        collision_state.set(self._robot_obj, "finger_gap", tgt_gripper)
        if self.grasped_obj_name != "":
            assert held_obj is not None, "Held object should be set when grasping"
            collision_state.set(held_obj, "x", tgt_held_x)
            collision_state.set(held_obj, "y", tgt_held_y)
            collision_state.set(held_obj, "theta", tgt_held_theta)
        # Check for collisions
        # NOTE: Simply check imagined new robot position against obstructions
        collision_checking_objects0 = {self._robot_obj}
        if held_obj is not None:
            collision_checking_objects0.add(held_obj)
        collision_checking_objects1 = [
            obj for obj in collision_state if "obstruction" in obj.name
        ]
        self.robot_is_colliding = state_2d_has_collision(
            collision_state,
            collision_checking_objects0,
            set(collision_checking_objects1),
            self._static_object_body_cache,
        )

        obs, reward, terminated, truncated, _ = super().step(action)

        if self.robot_is_colliding:
            obs.set(self._robot_obj, "is_colliding", 1.0)
        else:
            obs.set(self._robot_obj, "is_colliding", 0.0)

        # Get the held object info for grasping detection
        assert self._grasp_block is not None and self._base_block is not None
        if len(self.robot.held_objects):
            kin_obj, _, _ = self.robot.held_objects[0]
            held_obj_id = kin_obj[0].id
            for obj in (self._grasp_block, self._base_block):
                if self._state_obj_to_pymunk_body[obj].id == held_obj_id:
                    self.grasped_obj_name = obj.name
                    obs.set(obj, "held", 1.0)
                    break
        else:
            self.grasped_obj_name = ""
            for obj in (self._grasp_block, self._base_block):
                obs.set(obj, "held", 0.0)
        self.elapsed_steps += 1
        # Force get info to be the end of step
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        """Reset the environment."""
        self.elapsed_steps = 0
        self.grasped_obj_name = ""
        self.robot_is_colliding = False
        self.success = False

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

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
            potential_current_state = options["init_state"].copy()
            full_state = potential_current_state.copy()
            full_state.data.update(self.initial_constant_state.data)
            # Check validity of the provided initial state
            all_objects = set(full_state)
            obstruction_objects = {
                obj for obj in full_state if "obstruction" in obj.name
            }
            collision = state_2d_has_collision(
                full_state,
                obstruction_objects,
                all_objects,
                {},
            )
            if collision:
                # Dreamed initial state is invalid,
                # fall back to loading provided init state.
                self._current_state = self.load_init_state()
            else:
                self._current_state = potential_current_state
        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        # Add objects to physics space
        self._add_state_to_space(self.full_state)

        # Calculate simulation parameters
        dt = 1.0 / self.config.sim_hz
        # Stepping physics to let things settle
        assert self.pymunk_space is not None, "Space not initialized"
        for _ in range(self.config.sim_hz):
            self.pymunk_space.step(dt)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self) -> dict:
        return {
            "elapsed_steps": self.elapsed_steps,
            "is_colliding": self.robot_is_colliding,
            "is_grasped": self.grasped_obj_name != "",
            "success": self.success,
        }  # no extra info provided right now

    def load_init_state(self) -> ObjectCentricState:
        """Load an initial state for testing purposes."""
        constant_objects_names = []
        for i in CFG.scenario.split(","):
            # obstructions are always named by scenario index
            constant_objects_names.append(f"obstruction{i}")
        constant_objects_names.extend(["grasp_block", "base_block", "robot"])
        assert isinstance(self.observation_space, ObjectCentricStateSpace)
        exemplar_object_centric_state, _ = self.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        constant_objects = [obj_name_to_obj[name] for name in constant_objects_names]
        vec_observation_space = self.observation_space.to_box(
            constant_objects, self.type_features, self.action_space.dtype
        )
        task_files = glob.glob(
            f"{CFG.specified_task_path}/sc{CFG.scenario}_task_seed{CFG.seed}_*.pkl"
        )
        assert len(task_files) == 10, "There should be 10 specified task files."
        rnd_task_id = self.np_random.integers(0, 10)
        task_file = task_files[rnd_task_id]
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
        init_state_vec = task_data.init[0].cpu().numpy().astype(self.action_space.dtype)
        return vec_observation_space.devectorize(init_state_vec)


class BlockedStacking2DEnv(ConstantObjectPRBenchEnv):
    """Dynamic Obstruction 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricBlockedStacking2DEnv:
        return ObjectCentricBlockedStacking2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = []
        for i in CFG.scenario.split(","):
            # obstructions are always named by scenario index
            constant_objects.append(f"obstruction{i}")
        constant_objects.extend(["grasp_block", "base_block", "robot"])
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"

    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"

    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"
