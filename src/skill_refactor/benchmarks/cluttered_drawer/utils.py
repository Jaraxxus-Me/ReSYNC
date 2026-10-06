"""Utility functions for ClutteredDrawer benchmark."""

from relational_structs import (
    Object,
)
from torch import Tensor


def is_grasping(obs: Tensor) -> Tensor:
    """Check if the robot is currently grasping anything."""
    return obs[:, 27].clone().bool()  # bool tensor (NUM_ENVS,)


def extract_robot_body_pose(obs: Tensor) -> Tensor:
    """Extract robot data from observation."""
    return obs[:, 20:27].clone()


def extract_robot_hand_pose(obs: Tensor) -> Tensor:
    """Extract robot data from observation."""
    return obs[:, 28:35].clone()


def extract_grasp_hammer_pose(obs: Tensor) -> Tensor:
    """Extract grasp_hammer position from observation."""
    return obs[:, 51:58].clone()


def extract_target_hammer_pose(obs: Tensor) -> Tensor:
    """Extract hammer pose from observation."""
    return obs[:, 58:65].clone()


def extract_blocking_drawer_pose(obs: Tensor) -> Tensor:
    """Extract drawer pose from observation."""
    return obs[:, 35:42].clone()


def extract_supporting_drawer_pose(obs: Tensor) -> Tensor:
    """Extract drawer pose from observation."""
    return obs[:, 43:50].clone()


def extract_blocking_drawer_q(obs: Tensor) -> Tensor:
    """Extract drawer joint position from observation."""
    return obs[:, 42:43].clone()


def extract_supporting_drawer_q(obs: Tensor) -> Tensor:
    """Extract drawer joint position from observation."""
    return obs[:, 50:51].clone()


def extract_robot_joints(obs: Tensor) -> Tensor:
    """Extract robot joint positions from observation."""
    return obs[:, 0:10].clone()  # shape (B, 7)


def extract_block_pose(obs: Tensor) -> Tensor:
    """Extract blocking drawer pose from observation."""
    return obs[:, 65:72].clone()


def extract_object_pose(obs: Tensor, obj: Object) -> Tensor:
    """Extract object position from observation."""
    if obj.name == "robot":
        # NOTE: Temporarily return hand pose for robot
        # need to add body pose if necessary
        return extract_robot_hand_pose(obs)
    if obj.name == "grasp_hammer":
        return extract_grasp_hammer_pose(obs)
    if obj.name == "target_hammer":
        return extract_target_hammer_pose(obs)
    if obj.name == "drawer":
        return extract_blocking_drawer_pose(obs)
    if obj.name == "block":
        return extract_block_pose(obs)
    if obj.name == "wall":
        return extract_supporting_drawer_pose(obs)
    raise ValueError(f"Unknown object: {obj.name}")


def extract_drawer_q(obs: Tensor, obj: Object) -> Tensor:
    """Extract drawer joint position from observation."""
    if obj.name == "drawer":
        return extract_blocking_drawer_q(obs)
    if obj.name == "wall":
        return extract_supporting_drawer_q(obs)
    raise ValueError(f"Unknown drawer object: {obj.name}")


def extract_object_pos(obs: Tensor, obj: Object) -> Tensor:
    """Extract object pose from observation."""
    return extract_object_pose(obs, obj)[:, 0:3]
