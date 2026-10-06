"""Borrowed TaskPlan and SubTask definitions from MSHAB."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sapien
import shortuuid
import torch
import yaml
from dacite import from_dict
from mani_skill.utils.structs import Pose
from relational_structs import (
    Object,
)
from torch import Tensor

from skill_refactor.settings import CFG

PointTuple = Union[Tuple[float, float, float], List[float]]
RectCorners = Union[
    Tuple[PointTuple, PointTuple, PointTuple, PointTuple], List[PointTuple]
]
HandleJointIdxAndRelativeHandlePosition = Tuple[int, PointTuple]


def extract_robot_body_pose(obs: Tensor) -> Tensor:
    """Extract robot data from observation."""
    return obs[:, 22:29].clone()


def extract_robot_hand_pose(obs: Tensor) -> Tensor:
    """Extract robot data from observation."""
    return obs[:, 29:36].clone()


def extract_robot_joints(obs: Tensor) -> Tensor:
    """Extract robot joint positions from observation."""
    return obs[:, 0:11].clone()  # shape (B, 11)


def extract_robot_joint_vels(obs: Tensor) -> Tensor:
    """Extract robot joint positions from observation."""
    return obs[:, 11:22].clone()  # shape (B, 11)


def extract_obj1_pose(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 36:43].clone()


def extract_obj1_goal_pose(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 43:50].clone()


def extract_obj1_grasped(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 50:51].clone()


def extract_obj1_goal_achieved(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    dist_ok = obs[:, 51:52].clone().to(torch.bool)
    is_obj1_grasped = extract_obj1_grasped(obs).to(torch.bool)
    is_hand_empty = extract_hand_empty(obs).unsqueeze(1).to(torch.bool)

    goal_achieved = dist_ok & (~is_obj1_grasped) & is_hand_empty
    return goal_achieved


def extract_obj2_pose(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 52:59].clone()


def extract_obj2_goal_pose(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 59:66].clone()


def extract_obj2_grasped(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    return obs[:, 66:67].clone()


def extract_obj2_goal_achieved(obs: Tensor) -> Tensor:
    """Extract object 1 pose from observation."""
    dist_ok = obs[:, 67:68].clone().to(torch.bool)
    is_obj2_grasped = extract_obj2_grasped(obs).to(torch.bool)

    # NOTE: Obj2 goal does not require hand to be empty
    # otherwise pick obj1 will remove this obj2 goal achievement
    goal_achieved = dist_ok & (~is_obj2_grasped)
    return goal_achieved


def extract_can_pose(obs: Tensor) -> Tensor:
    """Extract can (obstruction) pose from observation."""
    return obs[:, 68:75].clone()


def extract_object_pose(obs: Tensor, obj: Object) -> Tensor:
    """Extract object pose from observation based on object name."""
    if obj.name == "bowl":
        return extract_obj1_pose(obs)
    elif obj.name == "bowl_goal":
        return extract_obj1_goal_pose(obs)
    elif obj.name == "box":
        return extract_obj2_pose(obs)
    elif obj.name == "box_goal":
        return extract_obj2_goal_pose(obs)
    elif obj.name == "can":
        return extract_can_pose(obs)
    else:
        raise ValueError(f"Unknown object name: {obj.name}")


def extract_object_held(obs: Tensor, obj: Object) -> Tensor:
    """Extract object pose from observation based on object name."""
    if obj.name == "bowl":
        return extract_obj1_grasped(obs)
    elif obj.name == "box":
        return extract_obj2_grasped(obs)
    else:
        return torch.zeros_like(obs[:, 0:1])


def extract_hand_empty(obs: Tensor) -> Tensor:
    """Extract whether the robot hand is empty from observation."""
    finger_openings = extract_robot_joints(obs)[:, -1]
    return finger_openings > CFG.c_room_hand_empty_finger_opening_thresh


def extract_goal_achieved(obs: Tensor, obj: Object) -> Tensor:
    """Extract whether the object has achieved its goal from observation based on object
    name."""
    if obj.name == "bowl":
        return extract_obj1_goal_achieved(obs)
    elif obj.name == "box":
        return extract_obj2_goal_achieved(obs)
    else:
        return torch.zeros_like(obs[:, 0:1])


@dataclass
class ArticulationConfig:
    """Configuration for articulated objects (doors, drawers, etc.)."""

    articulation_type: str
    articulation_id: str
    articulation_handle_link_idx: int
    articulation_handle_active_joint_idx: int


@dataclass
class Subtask:
    """Base class for all subtask types."""

    type: str = field(init=False)
    uid: str = field(init=False)
    composite_subtask_uids: List[str] = field(init=False)

    def __post_init__(self):
        assert self.type in ["pick", "place", "navigate", "open", "close"]
        if getattr(self, "uid", None) is None:
            self.uid = self.type + "_" + shortuuid.ShortUUID().random(length=6)
        if getattr(self, "composite_subtask_uids", None) is None:
            self.composite_subtask_uids = [self.uid]


@dataclass
class SubtaskConfig:
    """Base configuration for subtask execution parameters."""

    task_id: int
    horizon: int = 200
    robot_cumulative_force_limit: float = torch.inf
    ee_rest_thresh: float = 0.05
    robot_resting_qpos_tolerance: float = 0.2
    robot_resting_qpos_tolerance_grasping: float = 0.6

    def __post_init__(self):
        assert self.horizon > 0
        assert self.ee_rest_thresh >= 0
        assert self.robot_resting_qpos_tolerance >= 0
        assert self.robot_resting_qpos_tolerance_grasping >= 0

    def update(self, update_dict: Dict):
        for k, v in update_dict.items():
            if getattr(self, k, None) is not None:
                setattr(self, k, v)
        return self


@dataclass
class PickSubtask(Subtask):
    """Subtask for picking up an object."""

    obj_id: str
    articulation_config: Optional[ArticulationConfig] = None

    def __post_init__(self):
        self.type = "pick"
        super().__post_init__()


@dataclass
class PickSubtaskConfig(SubtaskConfig):
    """Configuration for pick subtasks."""

    task_id: int = 0
    robot_cumulative_force_limit: float = 5000


@dataclass
class PlaceSubtask(Subtask):
    """Subtask for placing an object at a goal location."""

    obj_id: str
    goal_rectangle_corners: Optional[
        Union[List[str], RectCorners, List[RectCorners]]
    ] = None
    goal_pos: Optional[Union[PointTuple, List[PointTuple], str]] = None
    validate_goal_rectangle_corners: bool = True
    articulation_config: Optional[ArticulationConfig] = None

    def __post_init__(self):
        self.type = "place"
        super().__post_init__()
        if (
            self.validate_goal_rectangle_corners
            and self.goal_rectangle_corners is not None
        ):
            self.goal_rectangle_corners = self._parse_rect_corners(
                self.goal_rectangle_corners
            )

        if isinstance(self.goal_pos, str):
            self.goal_pos = [float(coord) for coord in self.goal_pos.split(",")]

    def _parse_rect_corners(self, rect_corners):
        for i, corner in enumerate(rect_corners):
            if isinstance(corner, str):
                rect_corners[i] = [float(coord) for coord in corner.split(",")]
        # make sure have exactly 4 corners at the same height
        assert len(rect_corners) == 4, "Goal rectangle must have exactly 4 corners"
        A, B, C, D = [np.array(corner) for corner in rect_corners]
        sides0 = np.array([B - A, C - B, D - C, A - D])
        sides1 = np.array([D - A, A - B, B - C, C - D])
        points_angles = np.rad2deg(
            np.arccos(
                np.sum(sides0 * sides1, axis=1)
                / (np.linalg.norm(sides0, axis=1) * np.linalg.norm(sides1, axis=1))
            )
        )
        assert np.all(
            np.abs(points_angles - 90) < 1e-3
        ), f"Should have points in ABCD order, but got angles {points_angles} between sides AB/AD, BC/BA, CD/CB, DA/DC"
        return rect_corners


@dataclass
class PlaceSubtaskConfig(SubtaskConfig):
    """Configuration for place subtasks."""

    task_id: int = 1
    obj_goal_thresh: float = 0.15
    goal_type: str = "sphere"
    robot_cumulative_force_limit: float = 7500

    def __post_init__(self):
        super().__post_init__()
        assert self.obj_goal_thresh >= 0
        # cylinder means cylindrical goal centered at place_subtask.goal_pos
        # zone means use place_subtask.goal_rectangle_corners to establish rectangular zone
        assert self.goal_type in ["zone", "cylinder", "sphere"]


@dataclass
class NavigateSubtask(Subtask):
    """Subtask for navigating the robot to a target location."""

    obj_id: Optional[str] = None
    goal_pos: Optional[PointTuple] = None
    prev_goal_pos: Optional[PointTuple] = None
    articulation_config: Optional[ArticulationConfig] = None
    # NOTE (arth): see note in OpenSubtask
    remove_obj_id: Optional[str] = None

    def __post_init__(self):
        self.type = "navigate"
        super().__post_init__()


@dataclass
class NavigateSubtaskConfig(SubtaskConfig):
    """Configuration for navigate subtasks."""

    task_id: int = 2
    horizon: int = 500
    navigated_successfully_dist: float = 2
    navigated_successfully_rot: float = 0.5
    ignore_arm_checkers: bool = False


@dataclass
class OpenSubtask(Subtask, ArticulationConfig):
    """Subtask for opening an articulated object (door, drawer, etc.)."""

    obj_id: str
    articulation_relative_handle_pos: Union[PointTuple, sapien.Pose, Pose]

    def __post_init__(self):
        self.type = "open"
        super().__post_init__()


@dataclass
class OpenSubtaskConfig(SubtaskConfig):
    """Configuration for open subtasks."""

    task_id: int = 3
    robot_cumulative_force_limit: float = 10_000
    joint_qpos_open_thresh_frac: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self.joint_qpos_open_thresh_frac, dict)
        assert (
            "default" in self.joint_qpos_open_thresh_frac
        ), "joint_qpos_open_thresh_frac requires default value to cover cases where a different articulation is opened"
        for v in self.joint_qpos_open_thresh_frac.values():
            assert (
                isinstance(v, float) and 0 <= v <= 1
            ), f"joint_qpos_open_thresh_frac should be a float in [0, 1], instead got {v}"


@dataclass
class CloseSubtask(Subtask, ArticulationConfig):
    """Subtask for closing an articulated object (door, drawer, etc.)."""

    articulation_relative_handle_pos: Union[PointTuple, sapien.Pose, Pose]
    # NOTE (arth): this is somewhat of a band-aid solution to easily
    #       remove the bowl inside the kitchen_counter drawer
    #       in a future version, maybe can make a list of objects to
    #       teleport to specific location in gen_spawn_positions.py
    remove_obj_id: Optional[str] = None

    def __post_init__(self):
        self.type = "close"
        super().__post_init__()


@dataclass
class CloseSubtaskConfig(SubtaskConfig):
    """Configuration for close subtasks."""

    task_id: int = 4
    robot_cumulative_force_limit: float = 10_000
    joint_qpos_close_thresh_frac: float = 0.01


@dataclass
class TaskPlan:
    """A complete task plan consisting of a sequence of subtasks."""

    subtasks: List[Subtask]
    build_config_name: Optional[str] = None
    init_config_name: Optional[str] = None


# Reading Task Plan from file


@dataclass
class PlanData:
    """Container for task plan data loaded from file."""

    dataset: str
    plans: List[TaskPlan]


def plan_data_from_file(config_path: Union[str, Path]) -> PlanData:
    """Load task plan data from a JSON or YAML file.

    Args:
        config_path: Path to the task plan configuration file

    Returns:
        PlanData object containing the dataset name and list of task plans

    Raises:
        AssertionError: If the config path does not exist
        NotImplementedError: If an unsupported subtask type is encountered
        ValueError: If the file suffix is not .json or .yml
    """
    path = Path(config_path)
    assert path.exists(), f"Path {path} not found"

    suffix = path.suffix
    if suffix == ".json":
        with open(path, "rb") as f:
            plan_data = json.load(f)
    elif suffix == ".yml":
        with open(path, encoding="utf-8") as f:
            plan_data = yaml.safe_load(f)
    else:
        raise ValueError(f"File suffix {suffix} not supported (must be .json or .yml)")

    plans: List[TaskPlan] = []
    for task_plan_data in plan_data["plans"]:
        build_config_name = task_plan_data["build_config_name"]
        init_config_name = task_plan_data["init_config_name"]
        subtasks: List[Subtask] = []
        for subtask in task_plan_data["subtasks"]:
            subtask_type = subtask["type"]
            cls: type[Subtask]
            if subtask_type == "pick":
                cls = PickSubtask
            elif subtask_type == "place":
                cls = PlaceSubtask
            elif subtask_type == "navigate":
                cls = NavigateSubtask
            elif subtask_type == "open":
                cls = OpenSubtask
            elif subtask_type == "close":
                cls = CloseSubtask
            else:
                raise NotImplementedError(f"Subtask {subtask_type} not implemented yet")
            subtasks.append(from_dict(data_class=cls, data=subtask))
        plans.append(
            TaskPlan(
                subtasks=subtasks,
                build_config_name=build_config_name,
                init_config_name=init_config_name,
            )
        )

    return PlanData(dataset=plan_data["dataset"], plans=plans)
