"""Cluttered Drawer Environment borrowed from ManiSkill Benchmark."""

from functools import cached_property
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

from skill_refactor import PACKAGE_ASSET_DIR
from skill_refactor.benchmarks.cluttered_drawer.utils import (
    extract_block_pose,
    extract_blocking_drawer_pose,
    extract_blocking_drawer_q,
    extract_grasp_hammer_pose,
    extract_robot_joints,
    extract_supporting_drawer_pose,
    extract_supporting_drawer_q,
    extract_target_hammer_pose,
)
from skill_refactor.robots.spot_arm import SpotArmMobMani
from skill_refactor.settings import CFG

CABINET_COLLISION_BIT = 29


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
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

    return yaw


@register_env(
    "ClutteredDrawer-v1",
    max_episode_steps=CFG.max_env_steps,
)
class ClutteredDrawerEnv(BaseEnv):
    """**Task Description:** A more complicated task to pick and place a hammer from one
    drawer to another drawer. The drawers could be initially nearly closed and the robot
    needs to use the hammer as a hook to pull the drawer open.

    **Observation:**
    Always includes in any scenario:
    - [0, 14]: qpos and qvel of the robot
    - [14]: is_grasped (1 if the hammer is grasped, 0 otherwise)
    - [15, 22]: tcp pose (x, y, z, qw, qx, qy, qz)
    - [22, 29]: drawer_grasp pose (x, y, z, qw, qx, qy, qz)
    - [29]: drawer_grasp joint (left_drawer_qpos)
    - [30, 37]: drawer_target pose (x, y, z, qw, qx, qy, qz)
    - [37]: drawer_target joint (right_drawer_qpos)
    - [38, 45]: hammer grasp pose (x, y, z, qw, qx, qy, qz)
    - [45, 52]: hammer target pose (x, y, z, qw, qx, qy, qz)
    - [-2]: is_on (1 if the grasp hammer is on the target hammer, 0 otherwise)
    - [-1]: is_colliding
    Obstructions if exist (for scenario > 1):
    - [52+(num_obstructions-1), 59+(num_obstructions-1)]: obstruction n pose (x, y, z, qw, qx, qy, qz)

    **Success Conditions:**
    - the grasp hammer's position is within `goal_thresh` euclidean distance of the target hammer's position
    - the grasp hammer is not held
    """

    SUPPORTED_ROBOTS = [
        "spot_arm_mobmani",
    ]
    agent: SpotArmMobMani
    handle_types = ["prismatic"]
    hammer_grasp_urdf_path = f"{PACKAGE_ASSET_DIR}/hammer_grasp.urdf"
    hammer_tgt_urdf_path = f"{PACKAGE_ASSET_DIR}/hammer_tgt.urdf"
    obstacle_urdf_path = f"{PACKAGE_ASSET_DIR}/obstacle.urdf"
    drawer_urdf_path = f"{PACKAGE_ASSET_DIR}/drawer.urdf"
    drawer_nail_urdf_path = f"{PACKAGE_ASSET_DIR}/drawer_nail.urdf"

    def __init__(
        self,
        *args,
        robot_uids="spot_arm_mobmani",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ) -> None:
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.sensor_cam_eye_pos = CFG.c_drawer_sensor_cam_eye_pos
        self.sensor_cam_target_pos = CFG.c_drawer_sensor_cam_target_pos
        self.human_cam_eye_pos = CFG.c_drawer_human_cam_eye_pos
        self.human_cam_target_pos = CFG.c_drawer_human_cam_target_pos
        self.drawer_q_max: Optional[torch.Tensor] = None
        self.drawer_q_min: Optional[torch.Tensor] = None
        self.drawer_opened: Optional[torch.Tensor] = None
        self.wall_q_min: Optional[torch.Tensor] = None
        self.hammer_grasp_head: Optional[Link] = None
        self.hammer_tgt_head: Optional[Link] = None
        self.hammer_tgt_drawer_link: Optional[Link] = None
        self.hammer_tgt_drawer_rel_pose: Optional[Pose] = None
        self.block_drawer: Optional[Articulation] = None
        self.block_drawer_init_q: Optional[Pose] = None
        self.collision_links: list[Link] = []
        self.hammer_colliding_link: Optional[Link] = None
        self.obstruction_ids = [
            int(idx) for idx in CFG.scenario.split(",")
        ]  # 1 or 2 or 3
        # id must be ordered as 1,2, 2,3, 1,3, or 1,2,3 for now.
        assert (
            sorted(self.obstruction_ids) == self.obstruction_ids
        ), "Obstruction ids must be ordered"
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @cached_property
    def single_observation_space(self) -> gym.Space:
        """The unbatched observation space of the environment."""
        # observation space
        shape = common.to_numpy(self._init_raw_obs).shape[1:]
        low = np.array([-np.inf] * shape[0])
        high = np.array([np.inf] * shape[0])
        return gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        """Add rendering camera."""
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(
        self,
        options: dict,
        initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
    ):
        """Load the agent (robot) into the environment."""
        del initial_agent_poses, build_separate
        super()._load_agent(
            options,
            sapien.Pose(p=[CFG.c_drawer_spot_body_x, 0, CFG.c_drawer_spot_body_z]),
        )

    def _load_scene(self, options: dict):
        """Load the scene."""
        # Load static objects like ground, table, and robot
        self.ground = build_ground(self.scene)
        from skill_refactor.robots.spot_arm import SPOT_LLEG_COLLISION_BIT

        self.ground.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.ground.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )

        # Load the drawers
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.disable_self_collisions = True
        loader.scale = CFG.c_drawer_spawn_scale
        loader.name = "drawer_blocking"
        self.drawer_blocking = loader.load(self.drawer_urdf_path)
        for joint in self.drawer_blocking.joints:
            if joint.type[0] in self.handle_types:
                self.drawer_q_max = joint.get_limits()[:, 1]
                self.drawer_q_min = joint.get_limits()[:, 0]
                break
        self.collision_links.append(self.drawer_blocking.links_map["tabletop"])
        self.hammer_colliding_link = self.drawer_blocking.links_map["tabletop"]
        self.collision_links.append(self.drawer_blocking.links_map["drawer"])

        # DEBUG: YCB object and hacked link
        builder = actors.get_actor_builder(self.scene, id=f"ycb:024_bowl")
        builder.initial_pose = sapien.Pose(p=[0.8, 0.0, 0.7], q=[1, 0, 0, 0])
        self.apple_actor = builder.build(name="debug_apple")

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.disable_self_collisions = True
        loader.scale = CFG.c_drawer_spawn_scale
        loader.name = "drawer_supporting"
        if "3" in CFG.scenario:
            self.drawer_supporting = loader.load(self.drawer_nail_urdf_path)
        else:
            self.drawer_supporting = loader.load(self.drawer_urdf_path)
        for joint in self.drawer_supporting.joints:
            if joint.type[0] in self.handle_types:
                self.wall_q_min = joint.get_limits()[:, 0]
                break
        self.collision_links.append(self.drawer_supporting.links_map["tabletop"])
        self.collision_links.append(self.drawer_supporting.links_map["drawer"])
        if "3" in CFG.scenario:
            self.wall = self.drawer_supporting.links_map["nail"]
            self.collision_links.append(self.wall)
        else:
            self.wall = self.drawer_supporting.links_map["tabletop"]

        # spot arm picks hammer
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.name = "hammer_grasp"
        self.hammer_grasp = loader.load(self.hammer_grasp_urdf_path)
        self.hammer_grasp_handel = self.hammer_grasp.links_map["handle_link"]
        self.hammer_grasp_head = self.hammer_grasp.links_map["head_link"]

        # Goal indicator that only moves with direct resetting
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.name = "hammer_tgt"
        self.hammer_tgt = loader.load(self.hammer_tgt_urdf_path)
        self.hammer_tgt_handel = self.hammer_tgt.links_map["handle_link"]
        self.hammer_tgt_head = self.hammer_tgt.links_map["head_link"]

        # All hammer links do not collide
        self.hammer_grasp_handel.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.hammer_grasp_head.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.hammer_tgt_handel.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.hammer_tgt_head.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )

        # Add a block to the scene
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.name = "block"
        self.obstacle = loader.load(self.obstacle_urdf_path)
        self.obstacle_base = self.obstacle.links_map["body_link"]
        self.obstacle_attach = self.obstacle.links_map["attachment_link"]
        self.collision_links.append(self.obstacle_base)
        self.collision_links.append(self.obstacle_attach)

        # Wall and block do not collide with hammer (for simulation stability)
        self.obstacle_base.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.obstacle_attach.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )
        self.wall.set_collision_group_bit(
            group=2, bit_idx=SPOT_LLEG_COLLISION_BIT, bit=1
        )

        # Wall and block do not collide with drawer links
        self.wall.set_collision_group_bit(group=2, bit_idx=CABINET_COLLISION_BIT, bit=1)
        self.obstacle_base.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
        self.obstacle_attach.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
        self.drawer_supporting.links_map["drawer"].set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
        self.drawer_blocking.links_map["drawer"].set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
        self.drawer_supporting.links_map["tabletop"].set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
        self.drawer_blocking.links_map["tabletop"].set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.05,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode."""
        if "init_state" in options:
            # if reset states are provided, use them to initialize the environment
            self._initialize_episode_from_state(env_idx, options["init_state"])
            return
        with torch.device(self.device):
            # Type narrowing for mypy
            assert self.drawer_q_max is not None, "drawer_q_max must be initialized"
            assert self.drawer_q_min is not None, "drawer_q_min must be initialized"
            b = len(env_idx)
            # 1. Sample hammer poses
            # Sample a large batch of y-coordinates and filter for valid pairs
            # that satisfy the minimum distance constraint
            y1_samples = torch.empty(0, device=self.device)
            y2_samples = torch.empty(0, device=self.device)

            while len(y1_samples) < b:
                # Sample 100x more candidates than needed for efficiency
                n_needed = b - len(y1_samples)
                n_candidates = n_needed * 50
                y1_candidates = randomization.uniform(
                    CFG.c_drawer_hammer_init_sampling_min[1],
                    CFG.c_drawer_hammer_init_sampling_max[1],
                    size=(n_candidates,),
                )
                y2_candidates = randomization.uniform(
                    CFG.c_drawer_hammer_init_sampling_min[1],
                    CFG.c_drawer_hammer_init_sampling_max[1],
                    size=(n_candidates,),
                )

                # Find pairs that satisfy the minimum distance constraint
                valid_mask = (
                    torch.abs(y1_candidates - y2_candidates)
                    >= CFG.c_drawer_hammer_min_dist_y
                )

                # Accumulate valid samples
                valid_y1 = y1_candidates[valid_mask]
                valid_y2 = y2_candidates[valid_mask]
                y1_samples = torch.cat([y1_samples, valid_y1])
                y2_samples = torch.cat([y2_samples, valid_y2])

            # Concatenate and select exactly b samples
            y1_samples = y1_samples[:b]
            y2_samples = y2_samples[:b]
            x_samples = randomization.uniform(
                CFG.c_drawer_hammer_init_sampling_min[0],
                CFG.c_drawer_hammer_init_sampling_max[0],
                size=(2 * b,),
            )
            theta_samples = randomization.uniform(
                CFG.c_drawer_hammer_init_sampling_min[2],
                CFG.c_drawer_hammer_init_sampling_max[2],
                size=(2 * b,),
            )

            # IF debug, use fixed positions
            if CFG.debug_env:
                y1_samples = torch.full((b,), -0.5, device=self.device)
                y2_samples = torch.full((b,), 0.5, device=self.device)
                x_samples = torch.full((2 * b,), 0.3, device=self.device)
                theta_samples = torch.zeros((2 * b,), device=self.device)

            euler_angles = torch.zeros((2 * b, 3), device=self.device)
            euler_angles[:, 2] = theta_samples
            quanternions = matrix_to_quaternion(
                euler_angles_to_matrix(euler_angles, "XYZ")
            )
            grasp_hammer_pose = Pose.create_from_pq(
                p=torch.stack(
                    [
                        x_samples[:b],
                        y1_samples,
                        torch.full(
                            (b,), CFG.c_drawer_hammer_height, device=self.device
                        ),
                    ],
                    dim=-1,
                ),
                q=quanternions[:b],
            )
            target_hammer_pose = Pose.create_from_pq(
                p=torch.stack(
                    [
                        x_samples[b:],
                        y2_samples,
                        torch.full(
                            (b,), CFG.c_drawer_hammer_height, device=self.device
                        ),
                    ],
                    dim=-1,
                ),
                q=quanternions[b:],
            )

            # 2. Calcuate relative pose and joint for a drawer
            # either blocking or supporting the task
            dx_samples = randomization.uniform(
                CFG.c_drawer_hammer_drawer_delta_x[0],
                CFG.c_drawer_hammer_drawer_delta_x[1],
                size=(2 * b,),
            )
            dy_samples = randomization.uniform(
                CFG.c_drawer_hammer_drawer_delta_y[0],
                CFG.c_drawer_hammer_drawer_delta_y[1],
                size=(2 * b,),
            )
            if CFG.debug_env:
                dx_samples = torch.full(
                    (2 * b,),
                    (
                        CFG.c_drawer_hammer_drawer_delta_x[0]
                        + CFG.c_drawer_hammer_drawer_delta_x[1]
                    )
                    / 2,
                    device=self.device,
                )
                dy_samples = torch.full(
                    (2 * b,),
                    (
                        CFG.c_drawer_hammer_drawer_delta_y[0]
                        + CFG.c_drawer_hammer_drawer_delta_y[1]
                    )
                    / 2,
                    device=self.device,
                )

            drawer_supporting_p = torch.zeros((b, 3), device=self.device)
            # Supporting drawer is fully open but there might be a wall fixed to drawer
            # we use wall to reverse calculate the drawer position
            # NOTE: No matter we have wall or not, we always does so for distribution consistency.
            # first calculate the nail pose
            wall_rel_offset = torch.zeros((b, 3), dtype=torch.float32)
            if not CFG.debug_env:
                wall_rel_offset[:, 0] = (
                    torch.rand(b) * (CFG.c_drawer_wall_dx[1] - CFG.c_drawer_wall_dx[0])
                    + CFG.c_drawer_wall_dx[0]
                )
                wall_rel_offset[:, 1] = (
                    torch.rand(b) * (CFG.c_drawer_wall_dy[1] - CFG.c_drawer_wall_dy[0])
                    + CFG.c_drawer_wall_dy[0]
                )
            else:
                mid_x = (
                    0.5 * (CFG.c_drawer_wall_dx[1] - CFG.c_drawer_wall_dx[0])
                    + CFG.c_drawer_wall_dx[0]
                )
                mid_y = (
                    0.5 * (CFG.c_drawer_wall_dy[1] - CFG.c_drawer_wall_dy[0])
                    + CFG.c_drawer_wall_dy[0]
                )
                wall_rel_offset[:, 0] = mid_x
                wall_rel_offset[:, 1] = mid_y
            wall_rel_pose = Pose.create_from_pq(
                wall_rel_offset, euler2quat(0.0, 0.0, 0.0)
            )
            wall_p = torch.zeros((b, 3), device=self.device)
            if CFG.wall_blocking_grasp:
                wall_pose = grasp_hammer_pose * wall_rel_pose
                wall_p = wall_pose.p - grasp_hammer_pose.p
            else:
                assert CFG.wall_blocking_stacking, "One wall blocking mode must be True"
                wall_pose = target_hammer_pose * wall_rel_pose
                wall_p = wall_pose.p - target_hammer_pose.p
            drawer_supporting_p[:, 0] += wall_p[:, 0]
            drawer_supporting_p[:, 0] += CFG.nail_drawer_offset_x
            drawer_supporting_p[:, 1] += wall_p[:, 1]
            drawer_supporting_p[:, 2] = (
                CFG.c_drawer_drawer_height - CFG.c_drawer_hammer_height
            )

            drawer_blocking_p = torch.zeros((b, 3), device=self.device)
            drawer_blocking_p[:, 0] += dx_samples[b:]
            drawer_blocking_p[:, 1] += dy_samples[b:]
            drawer_blocking_p[:, 0] += self.drawer_q_min + CFG.c_drawer_close_frac * (
                self.drawer_q_max - self.drawer_q_min
            )
            drawer_blocking_p[:, 2] = (
                CFG.c_drawer_drawer_height - CFG.c_drawer_hammer_height
            )

            # # 3) Place obstructions if any
            block_rel_offset = torch.empty((b, 3), dtype=torch.float32)
            if not CFG.debug_env:
                block_rel_offset[:, 0] = (
                    torch.rand(b)
                    * (CFG.c_drawer_block_dx[1] - CFG.c_drawer_block_dx[0])
                    + CFG.c_drawer_block_dx[0]
                )
                block_rel_offset[:, 1] = (
                    torch.rand(b)
                    * (CFG.c_drawer_block_dy[1] - CFG.c_drawer_block_dy[0])
                    + CFG.c_drawer_block_dy[0]
                )
            else:
                mid_x = (
                    0.5 * (CFG.c_drawer_block_dx[1] - CFG.c_drawer_block_dx[0])
                    + CFG.c_drawer_block_dx[0]
                )
                mid_y = (
                    0.5 * (CFG.c_drawer_block_dy[1] - CFG.c_drawer_block_dy[0])
                    + CFG.c_drawer_block_dy[0]
                )
                block_rel_offset[:, 0] = mid_x
                block_rel_offset[:, 1] = mid_y
            block_rel_offset[:, 2] = CFG.c_drawer_block_dz
            block_rel_rot_mat = torch.tensor(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32
            )
            block_rel_rot_q = matrix_to_quaternion(
                block_rel_rot_mat.unsqueeze(0).repeat(b, 1, 1)
            )
            block_rel_pose = Pose.create_from_pq(block_rel_offset, block_rel_rot_q)
            block_far_xyz = torch.zeros((b, 3), device=self.device)
            block_far_xyz[:, 0] += CFG.c_drawer_world_min[0]
            block_far_xyz[:, 1] += CFG.c_drawer_world_min[1]
            block_far_xyz[:, 2] = CFG.c_drawer_world_min[2]
            # Default to far enough pose with drawer blocking
            block_pose = Pose.create_from_pq(
                p=block_far_xyz,
                q=euler2quat(0.0, 0.0, 0.0),
            )
            self.block_drawer = self.drawer_blocking

            drawer_set = False
            for i in self.obstruction_ids:

                if not drawer_set:
                    # Drawer must be set no matter what scenario it is
                    drawer_set = True
                    if CFG.drawer_blocking_grasp:
                        # if "3" in CFG.scenario and "1" in CFG.scenario:
                        #     assert CFG.wall_blocking_stacking
                        drawer_supporting_p += target_hammer_pose.p
                        drawer_blocking_p += grasp_hammer_pose.p
                        self.hammer_tgt_drawer_link = self.drawer_supporting.links_map[
                            "drawer"
                        ]
                    elif CFG.drawer_blocking_stacking:
                        # if "3" in CFG.scenario and "1" in CFG.scenario:
                        #     assert CFG.wall_blocking_grasp
                        drawer_supporting_p += grasp_hammer_pose.p
                        drawer_blocking_p += target_hammer_pose.p
                        self.hammer_tgt_drawer_link = self.drawer_blocking.links_map[
                            "drawer"
                        ]
                    else:
                        # Should never reach here due to assertion, but added for safety
                        raise ValueError(
                            "Either blocking_grasp or blocking_stacking must be True"
                        )
                if i == 2:
                    if CFG.block_blocking_grasp:
                        if CFG.drawer_blocking_grasp:
                            self.block_drawer = self.drawer_blocking
                        else:
                            self.block_drawer = self.drawer_supporting
                        block_pose = grasp_hammer_pose * block_rel_pose
                    elif CFG.block_blocking_stacking:
                        if CFG.drawer_blocking_stacking:
                            self.block_drawer = self.drawer_blocking
                        else:
                            self.block_drawer = self.drawer_supporting
                        block_pose = target_hammer_pose * block_rel_pose
                    else:
                        # Should never reach here due to assertion, but added for safety
                        raise ValueError(
                            "Either blocking_grasp or blocking_stacking must be True"
                        )

            # Set drawer poses
            drawer_supporting_pose = Pose.create_from_pq(
                p=drawer_supporting_p,
                q=euler2quat(0.0, 0.0, 0.0),
            )
            drawer_blocking_pose = Pose.create_from_pq(
                p=drawer_blocking_p,
                q=euler2quat(0.0, 0.0, 0.0),
            )
            self.drawer_blocking.set_pose(drawer_blocking_pose)
            self.drawer_supporting.set_pose(drawer_supporting_pose)
            # Set drawer joint positions
            blocking_qpos = self.drawer_q_min + CFG.c_drawer_close_frac * (
                self.drawer_q_max - self.drawer_q_min
            )
            blocking_qpos = blocking_qpos.unsqueeze(1)
            self.drawer_blocking.set_qpos(blocking_qpos)
            self.drawer_blocking.set_qvel(blocking_qpos * 0)
            if "3" in CFG.scenario:
                assert self.wall_q_min is not None
                supporting_qpos = self.wall_q_min.clone()
                supporting_qpos = supporting_qpos.unsqueeze(1)
            else:
                supporting_qpos = self.drawer_q_max.clone()
                supporting_qpos = supporting_qpos.unsqueeze(1)
            self.drawer_supporting.set_qpos(supporting_qpos)
            self.drawer_supporting.set_qvel(supporting_qpos * 0)

            # Set hammer poses
            self.hammer_grasp.set_pose(grasp_hammer_pose)
            self.hammer_tgt.set_pose(target_hammer_pose)

            # Set block poses
            block_yaw = quat_to_yaw(block_pose.q)  # align with grasp hammer yaw
            block_pose_zero_rot = Pose.create_from_pq(
                p=block_pose.p, q=euler2quat(0.0, 0.0, 0.0)
            )
            # NOTE: obstacle is always set with zero yaw,
            # the rotation is set via qpos
            self.obstacle.set_pose(block_pose_zero_rot)
            qpos = torch.zeros((b, 3), device=self.device)
            qpos[:, -1] = block_yaw
            self.obstacle.set_qpos(qpos)

            if self.gpu_sim_enabled:
                for _ in range(5):
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene.px.step()
                    self.scene._gpu_fetch_all()

            # A fixed relative pose that will be used to calculate goal indicator later
            assert self.hammer_tgt_drawer_link is not None
            self.hammer_tgt_drawer_rel_pose = (
                self.hammer_tgt_drawer_link.pose.inv() * target_hammer_pose
            )
            assert self.block_drawer is not None
            self.block_drawer_init_q = self.block_drawer.get_qpos().clone()
            # Reset the robot
            # NOTE: qpos
            # x, y, z, arm_6, gripper 1
            qpos = torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.784,
                    1.96,
                    0.0,
                    1.345,
                    0.0,
                    -1.481,
                ]
            )
            qpos = qpos.repeat(b).reshape(b, -1)
            dist = randomization.uniform(
                CFG.c_drawer_spot_body_init_dist[0],
                CFG.c_drawer_spot_body_init_dist[1],
                size=(b,),
            )
            theta = randomization.uniform(
                CFG.c_drawer_spot_body_init_theta[0],
                CFG.c_drawer_spot_body_init_theta[1],
                size=(b,),
            )
            if CFG.debug_env:
                dist = torch.full((b,), 0.0, device=self.device)
                theta = torch.zeros((b,), device=self.device)
            xy = torch.zeros((b, 2))
            xy[:, 0] += torch.cos(theta) * dist
            xy[:, 1] += torch.sin(theta) * dist
            qpos[:, :2] = xy
            qpos[:, 2] = theta
            self.agent.reset(qpos)
            self.agent.robot.set_pose(
                sapien.Pose([CFG.c_drawer_spot_body_x, 0, CFG.c_drawer_spot_body_z])
            )

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] += CFG.c_drawer_world_max[0]
            goal_xyz[:, 1] += CFG.c_drawer_world_max[1]
            goal_xyz[:, 2] = CFG.c_drawer_world_max[2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            self.drawer_opened = torch.zeros((b,), device=self.device)

    def _initialize_episode_from_state(
        self, env_idx: torch.Tensor, init_state: torch.Tensor
    ):
        """Initialize episode."""
        with torch.device(self.device):
            b = len(env_idx)
            assert (
                init_state.shape[0] == b
            ), f"init_state should have shape ({b}, ...), but got {init_state.shape}"
            # Type narrowing for mypy
            assert self.drawer_q_max is not None, "drawer_q_max must be initialized"
            assert self.drawer_q_min is not None, "drawer_q_min must be initialized"

            # 1) Initialize the Drawers
            # Extract drawer joint positions from state [b, 4] where last 2 are left/right qpos
            blocking_drawer_qs = extract_blocking_drawer_q(init_state)
            blocking_drawer_pose_tensor = extract_blocking_drawer_pose(init_state)
            self.drawer_blocking.set_qpos(blocking_drawer_qs)
            self.drawer_blocking.set_qvel(torch.zeros_like(blocking_drawer_qs))
            blocking_drawer_pose = Pose.create_from_pq(
                p=blocking_drawer_pose_tensor[:, :3],
                q=blocking_drawer_pose_tensor[:, 3:7],
            )
            self.drawer_blocking.set_pose(blocking_drawer_pose)

            supporting_drawer_pose_tensor = extract_supporting_drawer_pose(init_state)
            supporting_drawer_qs = extract_supporting_drawer_q(init_state)
            # supporting drawer is always fully open
            self.drawer_supporting.set_qpos(supporting_drawer_qs)
            self.drawer_supporting.set_qvel(torch.zeros_like(supporting_drawer_qs))
            supporting_drawer_pose = Pose.create_from_pq(
                p=supporting_drawer_pose_tensor[:, :3],
                q=supporting_drawer_pose_tensor[:, 3:7],
            )
            self.drawer_supporting.set_pose(supporting_drawer_pose)

            # 2) set hammer poses
            object_pose_grasp = extract_grasp_hammer_pose(init_state)
            object_pose_grasp[:, 2] = CFG.c_drawer_hammer_height  # ensure height
            object_pose_tgt = extract_target_hammer_pose(init_state)
            object_pose_tgt[:, 2] = CFG.c_drawer_hammer_height  # ensure height
            self.hammer_grasp.set_pose(
                Pose.create_from_pq(object_pose_grasp[:, :3], object_pose_grasp[:, 3:7])
            )
            self.hammer_tgt.set_pose(
                Pose.create_from_pq(object_pose_tgt[:, :3], object_pose_tgt[:, 3:7])
            )

            # 3) set block pose
            block_pose = extract_block_pose(init_state)
            block_pose[:, 2] = (
                CFG.c_drawer_block_dz + CFG.c_drawer_hammer_height
            )  # ensure height
            block_yaw = quat_to_yaw(block_pose[:, 3:7])  # align with grasp hammer yaw
            block_pose_zero_rot = Pose.create_from_pq(
                p=block_pose[:, :3], q=euler2quat(0.0, 0.0, 0.0)
            )
            # NOTE: obstacle is always set with zero yaw,
            # the rotation is set via qpos
            qpos = torch.zeros((b, 3), device=self.device)
            qpos[:, -1] = block_yaw

            self.obstacle.set_pose(block_pose_zero_rot)
            self.obstacle.set_qpos(qpos)

            if self.gpu_sim_enabled:
                for _ in range(10):
                    self.scene._gpu_apply_all()
                    self.scene.px.gpu_update_articulation_kinematics()
                    self.scene.px.step()
                    self.scene._gpu_fetch_all()

            # Update hammer_drawer link and relative pose based on distance
            dist_y_block = torch.abs(
                object_pose_tgt[:, 1] - blocking_drawer_pose_tensor[:, 1]
            )
            dist_y_support = torch.abs(
                object_pose_tgt[:, 1] - supporting_drawer_pose_tensor[:, 1]
            )
            if (dist_y_block < dist_y_support).all():
                self.hammer_tgt_drawer_link = self.drawer_blocking.links_map["drawer"]
            else:
                assert (
                    dist_y_support < dist_y_block
                ).all(), "Inconsistent drawer distance calculation"
                self.hammer_tgt_drawer_link = self.drawer_supporting.links_map["drawer"]
            assert self.hammer_tgt_drawer_link is not None
            self.hammer_tgt_drawer_rel_pose = (
                self.hammer_tgt_drawer_link.pose.inv() * self.hammer_tgt.pose
            )

            # Update block_drawer link and relative pose based on distance
            dist_y_block = torch.abs(
                block_pose[:, 1] - blocking_drawer_pose_tensor[:, 1]
            )
            dist_y_block_mean = dist_y_block.mean()
            dist_y_support = torch.abs(
                block_pose[:, 1] - supporting_drawer_pose_tensor[:, 1]
            )
            dist_y_support_mean = dist_y_support.mean()
            if (dist_y_block_mean < dist_y_support_mean).all():
                self.block_drawer = self.drawer_blocking
            else:
                self.block_drawer = self.drawer_supporting
            assert self.block_drawer is not None
            self.block_drawer_init_q = self.block_drawer.get_qpos().clone()

            # 4) Set robot qpos and qvel
            robot_qpos = extract_robot_joints(init_state)
            self.agent.robot.set_qpos(robot_qpos)
            # NOTE: robot pose is always set to the default position
            # the actual body pose is encoded in joint qpos
            self.agent.robot.set_pose(
                sapien.Pose([CFG.c_drawer_spot_body_x, 0, CFG.c_drawer_spot_body_z])
            )

    def _after_control_step(self):
        # after each control step, we update the goal position of the handle link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        assert self.hammer_tgt_drawer_link is not None
        assert self.hammer_tgt_drawer_rel_pose is not None
        current_target_pose = (
            self.hammer_tgt_drawer_link.pose * self.hammer_tgt_drawer_rel_pose
        )
        self.hammer_tgt.set_pose(current_target_pose)

        drawer_actual_q = self.drawer_blocking.get_qpos()
        drawer_q_set = drawer_actual_q.clone()

        # NOTE: A hack that once a drawer is opened, it will be forced to open throughout
        # Since the sim is not very stable when the block is in contact with the drawer,
        # sometimes the drawer would be pushed back in slightly causing the robot to fail
        if self.drawer_opened.any():
            # Allow drawer to move only if previously opened
            drawer_q_set[self.drawer_opened] = self.drawer_q_max.unsqueeze(1)[
                self.drawer_opened
            ]
        self.drawer_blocking.set_qpos(drawer_q_set)

        block_q_current = self.block_drawer.get_qpos()
        change = block_q_current - self.block_drawer_init_q
        block_curr_q = self.obstacle.get_qpos().clone()
        # Add dx to block current qpos if drawer has moved
        block_curr_q[:, 0:1] -= change
        self.obstacle.set_qpos(block_curr_q)
        self.block_drawer_init_q = block_q_current.clone()

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _get_obs_extra(self, info: Dict):
        """Environment specific observation."""
        # Clip all poses to be within world bounds for numerical stability
        min_pos = torch.tensor(CFG.c_drawer_world_min, device=self.device)
        max_pos = torch.tensor(CFG.c_drawer_world_max, device=self.device)

        body_pos = torch.clamp(self.agent.body_pose.p, min=min_pos, max=max_pos)
        body_pose = torch.cat([body_pos, self.agent.body_pose.q], dim=-1)

        tcp_pos = torch.clamp(self.agent.tcp_pose.p, min=min_pos, max=max_pos)
        tcp_pose = torch.cat([tcp_pos, self.agent.tcp_pose.q], dim=-1)
        # Use left drawer as reference pose (right drawer is just shifted in y)
        drawer_pos_blocking = torch.clamp(
            self.drawer_blocking.pose.p, min=min_pos, max=max_pos
        )
        drawer_pose_blocking = torch.cat(
            [drawer_pos_blocking, self.drawer_blocking.pose.q], dim=-1
        )
        drawer_pos_supporting = torch.clamp(
            self.drawer_supporting.pose.p, min=min_pos, max=max_pos
        )
        drawer_pose_supporting = torch.cat(
            [drawer_pos_supporting, self.drawer_supporting.pose.q], dim=-1
        )

        grasping_hammer_pos = torch.clamp(
            self.hammer_grasp_handel.pose.p, min=min_pos, max=max_pos
        )
        grasping_hammer_pose = torch.cat(
            [grasping_hammer_pos, self.hammer_grasp_handel.pose.q],
            dim=-1,
        )
        target_hammer_pos = torch.clamp(
            self.hammer_tgt_handel.pose.p, min=min_pos, max=max_pos
        )
        target_hammer_pose = torch.cat(
            [target_hammer_pos, self.hammer_tgt_handel.pose.q],
            dim=-1,
        )

        block_pos = torch.clamp(self.obstacle_base.pose.p, min=min_pos, max=max_pos)
        block_pose = torch.cat(
            [block_pos, self.obstacle_base.pose.q],
            dim=-1,
        )

        obs = dict(
            body_pose=body_pose,
            is_grasped=info["is_grasped"],
            tcp_pose=tcp_pose,
            drawer_pose_blocking=drawer_pose_blocking,
            drawer_joint_blocking=self.drawer_blocking.get_qpos(),
            drawer_pose_supporting=drawer_pose_supporting,
            drawer_joint_target=self.drawer_supporting.get_qpos(),
            grasping_hammer_pose=grasping_hammer_pose,
            target_hammer_pose=target_hammer_pose,
            block_pose=block_pose,
            is_on=info["is_on"],
            is_colliding=info["is_colliding"],
        )
        return obs

    def evaluate(self):
        """Success conditions for the task."""
        is_grasped = self.agent.is_grasping(self.hammer_grasp_handel)
        is_colliding_robot = self.agent.is_colliding(self.collision_links)
        hammer_is_colliding = self.hammer_colliding()
        is_colliding = is_colliding_robot | hammer_is_colliding
        # success if the object is placed close to the goal and not grasped
        success = self.hammar_head_touched() & ~is_grasped
        # is_robot_static = self.agent.is_static(0.2)
        self.drawer_opened = (
            self.drawer_blocking.get_qpos() >= 0.9 * self.drawer_q_max.unsqueeze(1)
        )
        return {
            "success": success,
            "is_obj_placed": self.hammar_head_touched(),
            "is_grasped": is_grasped,
            "is_on": self.hammar_head_touched(),
            "is_colliding": is_colliding,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward based on the current observation, action, and info."""
        return torch.zeros(len(action), device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Compute normalized dense reward based on the current observation, action, and
        info.

        The reward is normalized to be in the range [0, 1].
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def hammer_colliding(self) -> torch.Tensor:
        """Check if the hammer is colliding with any collision links."""
        assert (
            self.hammer_colliding_link is not None
        ), "hammer_colliding_link must be initialized"
        contact_force_head = self.scene.get_pairwise_contact_forces(
            self.hammer_colliding_link, self.hammer_grasp_head
        )
        contact_force_handle = self.scene.get_pairwise_contact_forces(
            self.hammer_colliding_link, self.hammer_grasp_handel
        )
        contact_force = torch.stack([contact_force_head, contact_force_handle], dim=1)
        contact_forces = torch.linalg.norm(contact_force, axis=2)
        return torch.any(contact_forces >= 3.0, dim=1)

    def hammar_head_touched(self) -> torch.Tensor:
        """Check if the hammer head is touching each other with nearly static
        condition."""
        assert (
            self.hammer_grasp_head is not None
        ), "hammer_grasp_head must be initialized"
        assert self.hammer_tgt_head is not None, "hammer_tgt_head must be initialized"
        rel_pose = self.hammer_grasp_head.pose.inv() * self.hammer_tgt_head.pose
        dist_xy = torch.linalg.norm(rel_pose.p[:, :2], dim=-1)
        dist_z = torch.abs(rel_pose.p[:, 2])
        is_touched = (dist_xy < CFG.c_drawer_goal_radius) & (
            dist_z < CFG.c_drawer_goal_height
        )
        return is_touched
