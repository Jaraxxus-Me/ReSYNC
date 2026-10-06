"""Cluttered Room Environment using ReplicaCAD scenes from MSHAB."""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import sapien
import torch
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

from skill_refactor import PACKAGE_ASSET_DIR
from skill_refactor.benchmarks.cluttered_room.cluttered_room_env import ClutteredRoomEnv

# Import for scene builder registration side effect
from skill_refactor.benchmarks.cluttered_room.scene_builder import (  # pylint: disable=unused-import
    ReplicaCADRearrangeSceneBuilder,
)
from skill_refactor.benchmarks.cluttered_room.utils import (
    CloseSubtask,
    OpenSubtask,
    Subtask,
    TaskPlan,
    extract_can_pose,
    extract_obj1_goal_pose,
    extract_obj1_pose,
    extract_obj2_goal_pose,
    extract_obj2_pose,
    extract_robot_joints,
    plan_data_from_file,
)
from skill_refactor.robots.spot_arm import SpotArmFetchManiHeld
from skill_refactor.settings import CFG


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
    "ClutteredRoom-held-v1",
    max_episode_steps=CFG.max_env_steps,
)
class ClutteredRoomHeldEnv(SceneManipulationEnv):
    """**Task Description:** A robot manipulation task in realistic cluttered room
    scenes. The robot must pick and place objects in cluttered indoor environments using
    ReplicaCAD photorealistic scenes.

    **Observation:**
    - Robot state: qpos, qvel, TCP pose
    - Object states: poses of manipulable objects
    - Goal: target placement location
    - Scene: articulated objects (drawers, doors, etc.)

    **Success Conditions:**
    - Target object is placed at goal location
    - Object is not grasped by robot
    - Object is static
    """

    SUPPORTED_ROBOTS = [
        "spot_arm_fetch_held",
    ]
    agent: SpotArmFetchManiHeld
    can_urdf_path = f"{PACKAGE_ASSET_DIR}/can/can.urdf"

    # Configuration for object manipulation
    OBJ_GOAL_THRESH = 0.15  # Distance threshold for goal
    EE_REST_POS_WRT_BASE = (0.5, 0, 1.25)

    def __init__(
        self,
        *args,
        robot_uids="spot_arm_fetch_held",
        task_name: str = "set_table",  # "set_table", "tidy_house", "prepare_groceries"
        split: str = "train",  # "train" or "val"
        subset_bc_id: str = "v3_sc1_staging_00",  # use only task plans from this build config
        init_config_idx: int = 5,  # For debugging
        build_config_idxs: Optional[List[int]] = None,
        **kwargs,
    ):
        """Initialize the cluttered room environment.

        Args:
            task_name: Name of the task (set_table, tidy_house, prepare_groceries)
            split: Dataset split (train or val)
            task_plans: Pre-loaded task plans (if None, will load from default path)
            build_config_idxs: Specific scene configurations to use
        """
        # Check that MS_ASSET_DIR environment variable is set
        assert os.getenv("MS_ASSET_DIR") is not None, (
            "MS_ASSET_DIR environment variable must be set before creating ClutteredRoom environment. "
            "Set it to the path containing scene_datasets/replica_cad_dataset/..."
        )

        self.task_name = task_name
        self.split = split

        plan_data = plan_data_from_file(
            CFG.maniskill_rearrange_dir
            / "task_plans"
            / task_name
            / "sequential"
            / split
            / "all.json"
        )
        task_plans = plan_data.plans
        # Check unique plans
        plan_sequences = set()
        all_objects = set()
        all_articulations = set()
        number_of_pick_place_objects = set()
        number_of_articulations = set()
        for tp in task_plans:
            subtask = tp.subtasks
            plan_sequence = []
            pick_place_objects = set()
            artic_objects = set()
            for st in subtask:
                plan_sequence.append(st.type)
                if st.type in ["pick", "place"] and hasattr(st, "obj_id"):
                    assert st.obj_id is not None
                    pick_place_objects.add(st.obj_id)
                    all_objects.add(st.obj_id)
                if st.type in ["open", "close"] and hasattr(st, "articulation_id"):
                    assert st.articulation_id is not None
                    artic_objects.add(st.articulation_id)
                    all_articulations.add(st.articulation_id)
            plan_sequences.add(tuple(plan_sequence))
            number_of_pick_place_objects.add(len(pick_place_objects))
            number_of_articulations.add(len(artic_objects))

        self.base_task_plans = task_plans
        self.use_task_plans = []
        self.task_plan_idx = 0
        self.bc_to_task_plans: Dict[str, List[TaskPlan]] = defaultdict(list)
        for tp in task_plans:
            if tp.build_config_name and subset_bc_id in tp.build_config_name:
                self.use_task_plans.append(tp)
                self.bc_to_task_plans[tp.build_config_name].append(tp)

        self._init_config_names = [
            tp.init_config_name for tp in self.use_task_plans if tp.init_config_name
        ]
        self.tp0 = self.use_task_plans[0]

        # Initialize tracking variables for task
        self.rearrange_obj1: Optional[Actor] = None
        self.rearrange_obj2: Optional[Actor] = None
        self.can: Optional[Articulation] = None
        self.can_body: Optional[Actor] = None
        # self.articulated_obj1: Optional[Articulation] = None
        # self.articulated_obj2: Optional[Articulation] = None
        self.rearrange_obj1_goal: Optional[Pose] = None
        self.rearrange_obj2_goal: Optional[Pose] = None
        self.init_config_idx = init_config_idx
        # Pre-initialize to avoid None error during super().__init__
        self.init_config_idxs: Optional[List[int]] = None
        self.collision_links: list[Link] = []
        self._num_obstructions = len(CFG.scenario.split(","))
        # Use a customized scene builder for cluttered room rearrangement
        # where we have a subset of room layouts and we are using TidySpot instead of Fetch.
        self.obstruction_ids = [
            int(idx) for idx in CFG.scenario.split(",")
        ]  # 1 or 2 or 3
        # id must be ordered as 1,2, 2,3, 1,3, or 1,2,3 for now.
        assert (
            sorted(self.obstruction_ids) == self.obstruction_ids
        ), "Obstruction ids must be ordered"

        super().__init__(
            *args,
            robot_uids=robot_uids,
            scene_builder_cls="MSHAB_SetTableRearrange",
            build_config_idxs=build_config_idxs,
            **kwargs,
        )
        # Now set it after super().__init__
        if self.init_config_idxs is None:
            self.init_config_idxs = [self.init_config_idx] * self.num_envs

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                temp_buffer_capacity=2**24,
                max_rigid_contact_count=2**23,
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
            ),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        """Camera configuration for human rendering."""
        # Room camera - overhead view
        room_camera_pose = sapien_utils.look_at([-0.8, -3.6, 1.5], [-1.5, -1.0, 1.0])
        # room_camera_pose = sapien_utils.look_at([4, -3.5, 3.5], [1.5, -3.5, 0])
        room_camera_config = CameraConfig(
            "render_camera",
            room_camera_pose,
            1920,
            1080,
            1,
            0.01,
            10,
        )
        # Robot camera - mounted on torso (currently unused)
        robot_camera_pose = sapien_utils.look_at([-0.5, -0.5, 1.5], [-2, 1.0, 0.5])
        _robot_camera_config = CameraConfig(
            "robot_render_camera",
            robot_camera_pose,
            512,
            512,
            1.75,
            0.01,
            10,
        )
        return room_camera_config

    def _load_scene(self, options):

        self.build_config_idx_to_task_plans: Dict[int, List[TaskPlan]] = dict()
        for bc, task_plans in self.bc_to_task_plans.items():
            self.build_config_idx_to_task_plans[
                self.scene_builder.build_config_names_to_idxs[bc]
            ] = task_plans

        # if num_bcis < self.num_envs, repeat bcis and truncate at self.num_envs
        self.build_config_idxs: List[int] = options.get(
            "build_config_idxs",
            (
                self._episode_rng.choice(
                    list(self.build_config_idx_to_task_plans.keys()),
                    size=self.num_envs,
                    replace=True,
                ).tolist()
            ),
        )
        self.num_task_plans_per_bci = torch.tensor(
            [
                len(self.build_config_idx_to_task_plans[bci])
                for bci in self.build_config_idxs
            ],
            device=self.device,
        )
        self.scene_builder.build(self.build_config_idxs, self._init_config_names)

        self.obj1_site = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[0, 1, 0, 1],
            name="obj1_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.obj1_goal_site = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[0, 0.6, 0.4, 1],
            name="obj1_goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.obj2_site = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[1, 0, 0, 1],
            name="obj2_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.obj2_goal_site = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[0.6, 0, 0.4, 1],
            name="obj2_goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        # Add a block to the scene
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.name = "can"
        self.can = loader.load(self.can_urdf_path)
        self.can_body = self.can.links_map["body"]

    def _get_actor_entity(self, actor_id: str, env_num: int):
        actor = self.scene_builder.movable_objects[actor_id]
        return actor._objs[actor._scene_idxs.tolist().index(env_num)]

    def create_merged_actor_from_base_ids(
        self,
        base_ids: List[str],
        name: Optional[str] = None,
    ) -> Actor:
        """Create merged actor from base YCB IDs (e.g., 024_bowl).

        Finds the visible (non-hidden) instance of each base ID and creates a merged
        actor from those instances.
        """
        entities = []
        hidden_threshold = -100  # Objects below x=-100 are considered hidden

        for env_num, base_id in enumerate(base_ids):
            # Get all instances of this YCB object
            ycb_objs = self.scene_builder.ycb_objs_per_env[env_num][base_id]

            # Find the first visible instance (not at hidden position)
            visible_obj = None
            for obj in ycb_objs:
                pos = obj.pose.p
                if pos[0, 0] > hidden_threshold:
                    visible_obj = obj
                    break

            if visible_obj is not None:
                entity = visible_obj._objs[
                    visible_obj._scene_idxs.tolist().index(env_num)
                ]
                entities.append(entity)
            else:
                # If no visible object found, use the first one (fallback)
                entity = ycb_objs[0]._objs[
                    ycb_objs[0]._scene_idxs.tolist().index(env_num)
                ]
                entities.append(entity)

        merged_obj = Actor.create_from_entities(
            entities,
            scene=self.scene,
            scene_idxs=torch.arange(len(entities), dtype=torch.long),
        )
        if name is not None:
            merged_obj.name = name
        return merged_obj

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode."""
        if "init_state" in options:
            # if reset states are provided, use them to initialize the environment
            self._initialize_episode_from_state(env_idx, options["init_state"])
            return
        with torch.device(self.device):
            b = len(env_idx)

            # Sample a task plan for each environment
            # For simplicity, use the same task plan or sample randomly
            assert b == self.num_envs, "Only full reset supported currently."
            # Initialize scene builder
            assert self.scene_builder.init_configs is not None
            # Ensure init_config_idxs is set
            if self.init_config_idxs is None:
                self.init_config_idxs = [self.init_config_idx] * self.num_envs

            self.scene_builder.initialize(env_idx, list(self.init_config_idxs))

            # NOTE: init_config_idxs is a constant list for now
            # which is in correspondence with the task plans.
            # In the future we can sample different ones for different envs.
            self.process_task_plan(
                sampled_subtask_lists=[
                    self.use_task_plans[task_id].subtasks
                    for task_id in self.init_config_idxs
                ],
            )

            assert (
                self.can_body is not None
            ), "Can obstruction must be created in _load_scene"
            self.collision_links.append(self.can_body)
            assert self.rearrange_obj1 is not None
            assert self.rearrange_obj1_goal is not None
            assert self.rearrange_obj2 is not None
            assert self.rearrange_obj2_goal is not None

            self.obj1_site.set_pose(
                Pose.create_from_pq(
                    self.rearrange_obj1.pose.p,
                    self.rearrange_obj1.pose.q,
                )
            )
            self.obj1_goal_site.set_pose(self.rearrange_obj1_goal)
            self.obj2_site.set_pose(
                Pose.create_from_pq(
                    self.rearrange_obj2.pose.p,
                    self.rearrange_obj2.pose.q,
                )
            )
            self.obj2_goal_site.set_pose(self.rearrange_obj2_goal)

            # 3) Place can obstruction using relative pose
            # Create relative offset for can placement
            can_rel_offset = torch.empty((b, 3), dtype=torch.float32)
            if not CFG.debug_env:
                can_rel_offset[:, 0] = (
                    torch.rand(b)
                    * (CFG.can_blocking_rel_dx[1] - CFG.can_blocking_rel_dx[0])
                    + CFG.can_blocking_rel_dx[0]
                )
                can_rel_offset[:, 1] = (
                    torch.rand(b)
                    * (CFG.can_blocking_rel_dy[1] - CFG.can_blocking_rel_dy[0])
                    + CFG.can_blocking_rel_dy[0]
                )
                # Randomize can rotation
                can_rel_theta = (
                    torch.rand(b)
                    * (
                        CFG.can_blocking_theta_range[1]
                        - CFG.can_blocking_theta_range[0]
                    )
                    + CFG.can_blocking_theta_range[0]
                )
            else:
                mid_x = (
                    0.5 * (CFG.can_blocking_rel_dx[1] - CFG.can_blocking_rel_dx[0])
                    + CFG.can_blocking_rel_dx[0]
                )
                mid_y = (
                    0.5 * (CFG.can_blocking_rel_dy[1] - CFG.can_blocking_rel_dy[0])
                    + CFG.can_blocking_rel_dy[0]
                )
                can_rel_offset[:, 0] = mid_x
                can_rel_offset[:, 1] = mid_y
                # Use middle of theta range
                can_rel_theta = torch.full(
                    (b,),
                    0.5
                    * (
                        CFG.can_blocking_theta_range[1]
                        + CFG.can_blocking_theta_range[0]
                    ),
                )
            can_rel_offset[:, 2] = CFG.c_room_can_z

            # Create relative rotation quaternions
            can_rel_q = torch.stack(
                [
                    torch.from_numpy(euler2quat(0.0, 0.0, theta.item()))
                    for theta in can_rel_theta
                ],
                dim=0,
            )

            # Create relative pose
            can_rel_pose = Pose.create_from_pq(can_rel_offset, can_rel_q)

            # Default can pose: far away from the scene
            can_far_xyz = torch.zeros((b, 3), device=self.device)
            can_far_xyz[:, 0] += CFG.c_room_world_min[0]
            can_far_xyz[:, 1] += CFG.c_room_world_min[1]
            can_far_xyz[:, 2] = CFG.c_room_world_min[2]
            can_pose = Pose.create_from_pq(
                p=can_far_xyz,
                q=euler2quat(0.0, 0.0, 0.0),
            )

            # Determine can placement based on what it's blocking
            # Get reference poses for obj1 (bowl) and obj2 (box)
            _obj1_init_pose = Pose.create_from_pq(
                self.rearrange_obj1.pose.p, self.rearrange_obj1.pose.q
            )
            _obj2_init_pose = Pose.create_from_pq(
                self.rearrange_obj2.pose.p, self.rearrange_obj2.pose.q
            )
            obj1_goal_pose = Pose.create_from_pq(
                self.rearrange_obj1_goal.p, self.rearrange_obj1_goal.q
            )
            obj2_goal_pose = Pose.create_from_pq(
                self.rearrange_obj2_goal.p, self.rearrange_obj2_goal.q
            )

            for i in self.obstruction_ids:
                if i == 1:
                    # Scenario 1: can blocks bowl transport
                    if CFG.can_blocking_target1:
                        can_pose = obj1_goal_pose * can_rel_pose
                    elif CFG.can_blocking_target2:
                        can_pose = obj2_goal_pose * can_rel_pose
                    else:
                        raise ValueError(
                            "Either can_blocking_transport1 or can_blocking_target1 must be True for scenario 1"
                        )
                elif i == 2:
                    raise ValueError(
                        "Either can_blocking_transport2 or can_blocking_target2 must be True for scenario 2"
                    )

            # Set can pose
            can_pose_zero_rot = Pose.create_from_pq(
                p=can_pose.p,
                q=euler2quat(0.0, 0.0, 0.0),
            )
            can_yaw = quat_to_yaw(can_rel_q)
            can_q = torch.zeros((b, 3), dtype=torch.float32)
            can_q[:, -1] = can_yaw
            assert self.can is not None
            self.can.set_pose(can_pose_zero_rot)
            self.can.set_qpos(can_q)

    def _initialize_episode_from_state(
        self, env_idx: torch.Tensor, init_state: torch.Tensor
    ):
        """Initialize episode."""
        with torch.device(self.device):
            b = len(env_idx)
            assert (
                init_state.shape[0] == b
            ), f"init_state should have shape ({b}, ...), but got {init_state.shape}"

            assert self.init_config_idxs is not None
            self.scene_builder.initialize(env_idx, list(self.init_config_idxs))

            # NOTE: init_config_idxs is a constant list for now
            # which is in correspondence with the task plans.
            # In the future we can sample different ones for different envs.
            self.process_task_plan(
                sampled_subtask_lists=[
                    self.use_task_plans[task_id].subtasks
                    for task_id in self.init_config_idxs
                ],
            )

            # Type narrowing for mypy
            assert self.rearrange_obj1 is not None
            assert self.rearrange_obj2 is not None
            assert self.can is not None

            # 1) Initialize the objects
            # Extract drawer joint positions from state [b, 4] where last 2 are left/right qpos
            rearrange_obj1_pose_tnsr = extract_obj1_pose(init_state)
            _rearrange_obj2_pose_tnsr = extract_obj2_pose(init_state)
            obj1_goal_pose_tnsr = extract_obj1_goal_pose(init_state)
            obj2_goal_pose_tnsr = extract_obj2_goal_pose(init_state)
            can_pose_tnsr = extract_can_pose(init_state)

            self.rearrange_obj1.set_pose(
                Pose.create_from_pq(
                    p=rearrange_obj1_pose_tnsr[:, :3],
                    q=rearrange_obj1_pose_tnsr[:, 3:7],
                )
            )
            # self.rearrange_obj2.set_pose(
            #     Pose.create_from_pq(
            #         p=rearrange_obj2_pose_tnsr[:, :3],
            #         q=rearrange_obj2_pose_tnsr[:, 3:7],
            #     )
            # )
            can_pos = can_pose_tnsr[:, :3].clone()
            can_pos[:, 2] = CFG.c_room_can_z + CFG.c_room_goal_z
            self.can.set_pose(
                Pose.create_from_pq(p=can_pos, q=euler2quat(0.0, 0.0, 0.0))
            )
            can_yaw = quat_to_yaw(can_pose_tnsr[:, 3:7])
            can_q = torch.zeros((b, 3), dtype=torch.float32)
            can_q[:, -1] = can_yaw
            self.can.set_qpos(can_q)
            self.rearrange_obj1_goal = Pose.create_from_pq(
                p=obj1_goal_pose_tnsr[:, :3],
                q=obj1_goal_pose_tnsr[:, 3:7],
            )
            self.rearrange_obj2_goal = Pose.create_from_pq(
                p=obj2_goal_pose_tnsr[:, :3],
                q=obj2_goal_pose_tnsr[:, 3:7],
            )

            # 2) Update visual markers
            self.obj1_site.set_pose(
                Pose.create_from_pq(
                    self.rearrange_obj1.pose.p,
                    self.rearrange_obj1.pose.q,
                )
            )
            self.obj1_goal_site.set_pose(self.rearrange_obj1_goal)
            self.obj2_goal_site.set_pose(self.rearrange_obj2_goal)

            if self.scene.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # 4) Set robot qpos and qvel
            robot_qpos = extract_robot_joints(init_state)
            self.agent.reset(robot_qpos)
            # NOTE: robot pose is always set to the default position
            # the actual body pose is encoded in joint qpos
            self.agent.robot.set_pose(
                sapien.Pose([CFG.c_room_spot_body_x, CFG.c_room_spot_body_y, 0.0])
            )

            self.obj2_site.set_pose(
                Pose.create_from_pq(
                    self.rearrange_obj2.pose.p,
                    self.rearrange_obj2.pose.q,
                )
            )

    def _sample_object_poses_and_goals(
        self, b: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample object poses and goals with region-based constraints.

        Args:
            b: Batch size (number of environments)

        Returns:
            Tuple of (obj1_pos, obj2_pos, obj1_goal_pos, obj2_goal_pos)
            Each is a tensor of shape (b, 3) containing [x, y, z] positions
        """
        # Define regions
        region1_x = CFG.c_room_region1_x
        region1_y = CFG.c_room_region1_y
        region2_x = CFG.c_room_region2_x
        region2_y = CFG.c_room_region2_y

        y1_samples = torch.empty(0, device=self.device)
        y2_samples = torch.empty(0, device=self.device)

        while len(y1_samples) < b:
            # Sample 100x more candidates than needed for efficiency
            n_needed = b - len(y1_samples)
            n_candidates = n_needed * 50
            y1_candidates = randomization.uniform(
                region1_y[0],
                region1_y[1],
                size=(n_candidates,),
            )
            y2_candidates = randomization.uniform(
                region1_y[0],
                region1_y[1],
                size=(n_candidates,),
            )

            # Find pairs that satisfy the minimum distance constraint
            valid_mask = (
                torch.abs(y1_candidates - y2_candidates) >= CFG.c_room_obj_min_dist
            )

            # Accumulate valid samples
            valid_y1 = y1_candidates[valid_mask]
            valid_y2 = y2_candidates[valid_mask]
            y1_samples = torch.cat([y1_samples, valid_y1])
            y2_samples = torch.cat([y2_samples, valid_y2])

        y3_samples = torch.empty(0, device=self.device)
        y4_samples = torch.empty(0, device=self.device)

        while len(y3_samples) < b:
            # Sample 100x more candidates than needed for efficiency
            n_needed = b - len(y3_samples)
            n_candidates = n_needed * 50
            y3_candidates = randomization.uniform(
                region2_y[0],
                region2_y[1],
                size=(n_candidates,),
            )
            y4_candidates = randomization.uniform(
                region2_y[0],
                region2_y[1],
                size=(n_candidates,),
            )

            # Find pairs that satisfy the minimum distance constraint
            valid_mask = (
                torch.abs(y3_candidates - y4_candidates) >= CFG.c_room_obj_min_dist
            )

            # Accumulate valid samples
            valid_y3 = y3_candidates[valid_mask]
            valid_y4 = y4_candidates[valid_mask]
            y3_samples = torch.cat([y3_samples, valid_y3])
            y4_samples = torch.cat([y4_samples, valid_y4])

        x12 = randomization.uniform(
            region1_x[0],
            region1_x[1],
            size=(2 * b,),
            device=self.device,
        )
        x34 = randomization.uniform(
            region2_x[0],
            region2_x[1],
            size=(2 * b,),
            device=self.device,
        )

        # Stack x and y coordinates to create 2D positions
        region1_sampled_xy1 = torch.stack([x12[:b], y1_samples[:b]], dim=-1)
        region1_sampled_xy2 = torch.stack([x12[b : 2 * b], y2_samples[:b]], dim=-1)
        region2_sampled_xy1 = torch.stack([x34[:b], y3_samples[:b]], dim=-1)
        region2_sampled_xy2 = torch.stack([x34[b : 2 * b], y4_samples[:b]], dim=-1)

        # Initialize output tensors
        obj1_init_pos = torch.zeros((b, 3), device=self.device)
        obj1_goal_pos = torch.zeros((b, 3), device=self.device)
        obj2_init_pos = torch.zeros((b, 3), device=self.device)
        obj2_goal_pos = torch.zeros((b, 3), device=self.device)

        # Randomly decide which region each object starts in (per batch element)
        # True = start in region1 (goal in region2), False = start in region2 (goal in region1)
        obj1_starts_region1 = torch.rand(b, device=self.device) > 0.5
        obj2_starts_region1 = torch.rand(b, device=self.device) > 0.5

        # Assign x,y positions based on random region assignment
        # Object 1: init and goal are in different regions
        obj1_init_pos[:, :2] = torch.where(
            obj1_starts_region1.unsqueeze(-1),
            region1_sampled_xy1,
            region2_sampled_xy1,
        )
        obj1_goal_pos[:, :2] = torch.where(
            obj1_starts_region1.unsqueeze(-1),
            region2_sampled_xy1,
            region1_sampled_xy1,
        )

        # Object 2: init and goal are in different regions (independent randomization)
        obj2_init_pos[:, :2] = torch.where(
            obj2_starts_region1.unsqueeze(-1),
            region1_sampled_xy2,
            region2_sampled_xy2,
        )
        obj2_goal_pos[:, :2] = torch.where(
            obj2_starts_region1.unsqueeze(-1),
            region2_sampled_xy2,
            region1_sampled_xy2,
        )

        # Set z-coordinates
        obj1_init_pos[:, 2] = CFG.c_room_obj1_z
        obj1_goal_pos[:, 2] = CFG.c_room_goal_z
        obj2_init_pos[:, 2] = CFG.c_room_obj2_z
        obj2_goal_pos[:, 2] = CFG.c_room_goal_z

        return obj1_init_pos, obj2_init_pos, obj1_goal_pos, obj2_goal_pos

    def process_task_plan(
        self,
        sampled_subtask_lists: List[List[Subtask]],
    ):
        """Process sampled task plans to setup episode.

        Extracts objects and goals from subtasks and initializes:
        - rearrange_obj1/2: Objects to pick and place
        - rearrange_obj1/2_goal: Goal poses for placement
        - articulated_obj1/2: Articulated objects (doors, drawers)
        """
        # Extract objects from pick/place subtasks
        # Organize data per environment
        articulation_ids = []
        articulation_joint_ids = []

        for _, subtask_list in enumerate(sampled_subtask_lists):
            for subtask in subtask_list:
                if isinstance(subtask, (OpenSubtask, CloseSubtask)):
                    if subtask.articulation_id not in articulation_ids:
                        articulation_ids.append(subtask.articulation_id)
                        articulation_joint_ids.append(
                            subtask.articulation_handle_active_joint_idx
                        )

        # Create merged actors for rearrange objects by finding visible instances
        # Use unique objects, one per environment
        obj1_ids = ["024_bowl"] * self.num_envs

        self.rearrange_obj1 = self.create_merged_actor_from_base_ids(
            obj1_ids, name="rearrange_obj1"
        )
        self.rearrange_obj2 = self.agent.held_obj_link

        # Sample object poses and goals with region-based constraints
        obj1_init_pos, _, obj1_goal_pos, obj2_goal_pos = (
            self._sample_object_poses_and_goals(self.num_envs)
        )

        # Set object initial poses
        obj1_pose = Pose.create_from_pq(p=obj1_init_pos, q=euler2quat(0.0, 0.0, 0.0))
        self.rearrange_obj1.set_pose(obj1_pose)

        # Set goal poses
        self.rearrange_obj1_goal = Pose.create_from_pq(p=obj1_goal_pos)
        self.rearrange_obj2_goal = Pose.create_from_pq(p=obj2_goal_pos)

    def objects_colliding(self, collision_links: list[Link]) -> torch.Tensor:
        """Check if any objects are colliding with the given links."""
        obj1_forces_list = []
        obj2_forces_list = []

        for link in collision_links:
            obj1_forces = self.scene.get_pairwise_contact_forces(
                link, self.rearrange_obj1
            )
            obj2_forces = self.scene.get_pairwise_contact_forces(
                link, self.rearrange_obj2
            )
            obj1_forces_list.append(obj1_forces)
            obj2_forces_list.append(obj2_forces)
        total_obj1_forces = torch.stack(obj1_forces_list, dim=1)
        total_obj2_forces = torch.stack(obj2_forces_list, dim=1)
        total_forces = torch.cat([total_obj1_forces, total_obj2_forces], dim=1)
        total_forces = torch.linalg.norm(total_forces, dim=-1)
        contact_forces_all_links = torch.any(total_forces >= 1.0, dim=1)

        return contact_forces_all_links

    def evaluate(self):
        """Evaluate success conditions."""
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # NOTE: Need to add held object check in the future
        is_colliding_robot = self.agent.is_colliding(self.collision_links)
        is_colliding_objects = self.objects_colliding(self.collision_links)

        assert self.rearrange_obj1 is not None
        assert self.rearrange_obj1_goal is not None
        assert self.rearrange_obj2 is not None
        assert self.rearrange_obj2_goal is not None
        # Check if object is grasped
        is_obj1_grasped = self.agent.is_grasping(self.rearrange_obj1, max_angle=30)
        is_obj2_grasped = self.agent.is_grasping(self.rearrange_obj2, max_angle=30)

        # Check if objects are at goal
        obj1_pos = self.rearrange_obj1.pose.p
        obj2_pos = self.rearrange_obj2.pose.p
        obj1_goal_pos = self.rearrange_obj1_goal.p
        obj2_goal_pos = self.rearrange_obj2_goal.p
        obj1_dist_to_goal = torch.norm(obj1_pos - obj1_goal_pos, dim=-1)
        obj2_dist_to_goal = torch.norm(obj2_pos - obj2_goal_pos, dim=-1)
        is_obj1_at_goal = obj1_dist_to_goal < CFG.c_room_obj_goal_thresh
        is_obj2_at_goal = obj2_dist_to_goal < CFG.c_room_obj_goal_thresh

        success = (
            (~is_obj1_grasped) & is_obj1_at_goal & (~is_obj2_grasped) & is_obj2_at_goal
        )

        return {
            "success": success,
            "is_obj1_grasped": is_obj1_grasped,
            "is_obj1_at_goal": is_obj1_at_goal,
            "is_obj2_grasped": is_obj2_grasped,
            "is_obj2_at_goal": is_obj2_at_goal,
            "is_colliding": is_colliding_robot | is_colliding_objects,
        }

    def _get_obs_extra(self, info: Dict):
        """Get task-specific observations."""
        obs = dict()

        # Robot TCP pose
        body_pose = self.agent.body_pose.raw_pose
        tcp_pose = self.agent.tcp_pose.raw_pose

        # Rearrange object 1 pose
        assert self.rearrange_obj1 is not None
        obj1_pose = torch.cat(
            [self.rearrange_obj1.pose.p, self.rearrange_obj1.pose.q], dim=-1
        )

        # Rearrange object 2 pose
        assert self.rearrange_obj2 is not None
        obj2_pose = torch.cat(
            [self.rearrange_obj2.pose.p, self.rearrange_obj2.pose.q], dim=-1
        )

        assert self.can_body is not None
        can_pose = torch.cat([self.can_body.pose.p, self.can_body.pose.q], dim=-1)

        # Rearrange object 1 goal
        assert self.rearrange_obj1_goal is not None
        obj1_goal = torch.cat(
            [self.rearrange_obj1_goal.p, self.rearrange_obj1_goal.q], dim=-1
        )

        # Rearrange object 2 goal
        assert self.rearrange_obj2_goal is not None
        obj2_goal = torch.cat(
            [self.rearrange_obj2_goal.p, self.rearrange_obj2_goal.q], dim=-1
        )

        obs = dict(
            body_pose=body_pose,
            tcp_pose=tcp_pose,
            rearrange_obj1_pose=obj1_pose,
            rearrange_obj1_goal=obj1_goal,
            is_obj1_grasped=info["is_obj1_grasped"],
            is_obj1_at_goal=info["is_obj1_at_goal"],
            rearrange_obj2_pose=obj2_pose,
            rearrange_obj2_goal=obj2_goal,
            is_obj2_grasped=info["is_obj2_grasped"],
            is_obj2_at_goal=info["is_obj2_at_goal"],
            can_pose=can_pose,
            is_colliding=info["is_colliding"],
        )
        return obs

    def compute_dense_reward(self, _obs: Any, _action: torch.Tensor, info: Dict):
        """Compute dense reward (sparse for now)."""
        # Simple sparse reward: 1 for success, 0 otherwise
        return info["success"].float()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Compute normalized dense reward."""
        return self.compute_dense_reward(_obs=obs, _action=action, info=info)
