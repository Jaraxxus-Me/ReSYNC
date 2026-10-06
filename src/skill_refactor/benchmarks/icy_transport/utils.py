"""Utility functions for IcyTransport benchmark."""

import torch
from relational_structs import (
    Object,
)
from torch import Tensor

from skill_refactor.settings import CFG

ROOM_CENTERS = {
    "room_bl": (
        CFG.i_trans_world_size[0] / 4,
        CFG.i_trans_world_size[1] / 4,
    ),  # Bottom-left
    "room_br": (
        CFG.i_trans_world_size[0] * 3 / 4,
        CFG.i_trans_world_size[1] / 4,
    ),  # Bottom-right
    "room_tl": (
        CFG.i_trans_world_size[0] / 4,
        CFG.i_trans_world_size[1] * 3 / 4,
    ),  # Top-left
    "room_tr": (
        CFG.i_trans_world_size[0] * 3 / 4,
        CFG.i_trans_world_size[1] * 3 / 4,
    ),  # Top-right
}

# Doorway positions for navigation (center of each doorway + offset for entering)
DOORWAYS = {
    ("room_bl", "room_br"): (
        CFG.i_trans_world_size[0] / 2 + CFG.i_trans_in_room_offset * 1.2,
        CFG.i_trans_world_size[1] / 4 - CFG.i_trans_door_width / 16,
    ),  # Horizontal door between bottom rooms
    ("room_br", "room_bl"): (
        CFG.i_trans_world_size[0] / 2 - CFG.i_trans_in_room_offset * 1.2,
        CFG.i_trans_world_size[1] / 4 + CFG.i_trans_door_width / 16,
    ),
    ("room_tl", "room_tr"): (
        CFG.i_trans_world_size[0] / 2 + CFG.i_trans_in_room_offset * 1.2,
        CFG.i_trans_world_size[1] * 3 / 4 - CFG.i_trans_door_width / 16,
    ),  # Horizontal door between top rooms
    ("room_tr", "room_tl"): (
        CFG.i_trans_world_size[0] / 2 - CFG.i_trans_in_room_offset * 1.2,
        CFG.i_trans_world_size[1] * 3 / 4 + CFG.i_trans_door_width / 16,
    ),
    ("room_bl", "room_tl"): (
        CFG.i_trans_world_size[0] / 4 + CFG.i_trans_door_width / 16,
        CFG.i_trans_world_size[1] / 2 + CFG.i_trans_in_room_offset * 1.2,
    ),  # Vertical door between left rooms
    ("room_tl", "room_bl"): (
        CFG.i_trans_world_size[0] / 4 - CFG.i_trans_door_width / 16,
        CFG.i_trans_world_size[1] / 2 - CFG.i_trans_in_room_offset * 1.2,
    ),
    ("room_br", "room_tr"): (
        CFG.i_trans_world_size[0] * 3 / 4 + CFG.i_trans_door_width / 16,
        CFG.i_trans_world_size[1] / 2 + CFG.i_trans_in_room_offset * 1.2,
    ),  # Vertical door between right rooms
    ("room_tr", "room_br"): (
        CFG.i_trans_world_size[0] * 3 / 4 - CFG.i_trans_door_width / 16,
        CFG.i_trans_world_size[1] / 2 - CFG.i_trans_in_room_offset * 1.2,
    ),
}

ROOM_PLANS = {
    # One Move
    ("room_bl", "room_br"): [("room_bl", "room_br")],
    ("room_br", "room_bl"): [("room_br", "room_bl")],
    ("room_tl", "room_tr"): [("room_tl", "room_tr")],
    ("room_tr", "room_tl"): [("room_tr", "room_tl")],
    ("room_br", "room_tr"): [("room_br", "room_tr")],
    ("room_tr", "room_br"): [("room_tr", "room_br")],
    ("room_bl", "room_tl"): [("room_bl", "room_tl")],
    ("room_tl", "room_bl"): [("room_tl", "room_bl")],
    # Two Moves
    ("room_br", "room_tl"): [("room_br", "room_bl"), ("room_bl", "room_tl")],
    ("room_bl", "room_tr"): [("room_bl", "room_br"), ("room_br", "room_tr")],
    ("room_tl", "room_br"): [("room_tl", "room_bl"), ("room_bl", "room_br")],
    ("room_tr", "room_bl"): [("room_tr", "room_br"), ("room_br", "room_bl")],
    # Three Moves
    ("room_tl", "room_tr"): [
        ("room_tl", "room_bl"),
        ("room_bl", "room_br"),
        ("room_br", "room_tr"),
    ],
    ("room_tr", "room_tl"): [
        ("room_tr", "room_br"),
        ("room_br", "room_bl"),
        ("room_bl", "room_tl"),
    ],
}


def extract_robot_pose(obs: Tensor) -> Tensor:
    """Extract robot pose from observation."""
    return obs[:, -14:-11].clone()


def extract_robot_vel(obs: Tensor) -> Tensor:
    """Extract robot velocity from observation."""
    return obs[:, -11:-8].clone()


def extract_robot_force(obs: Tensor) -> Tensor:
    """Extract robot force from observation."""
    return obs[:, -6:-3].clone()


def extract_robot_mass_moment(obs: Tensor) -> Tensor:
    """Extract robot mass and moment from observation."""
    return obs[:, -3:-1].clone()


def extract_target_obj_pose(obs: Tensor) -> Tensor:
    """Extract target object pose from observation."""
    return obs[:, -28:-25].clone()


def extract_transport_obj1_pose(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -56:-53].clone()


def extract_transport_obj1_held(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -49].clone()


def extract_transport_obj1_static(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -50].clone()


def extract_transport_obj2_pose(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -42:-39].clone()


def extract_transport_obj2_held(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -35].clone()


def extract_transport_obj2_static(obs: Tensor) -> Tensor:
    """Extract transport pose from observation."""
    return obs[:, -36].clone()


def extract_region_pose(obs: Tensor, region_obj: Object) -> Tensor:
    """Extract region activated status from observation."""
    assert "region" in region_obj.name, "Object is not a region."
    names = ["icy_region", "muddy_region", "sandy_region"]
    scenarios = CFG.scenario.split(",")
    name_id = names.index(region_obj.name)
    scenario_idx = scenarios.index(str(name_id + 1))
    s = 20 * scenario_idx
    e = s + 3
    return obs[:, s:e].clone()


def extract_room_pose(obs: Tensor, room_obj: Object) -> Tensor:
    """Extract room pose from observation."""
    assert "room" in room_obj.name, "Object is not a room."
    pose = torch.zeros_like(obs[:, :3])
    pose[:, 0], pose[:, 1] = ROOM_CENTERS[room_obj.name]
    return pose


def extract_object_pose(obs: Tensor, obj: Object) -> Tensor:
    """Extract specified object pose from observation."""
    if obj.name == "robot":
        return extract_robot_pose(obs)
    elif obj.name == "transport_obj1":
        return extract_transport_obj1_pose(obs)
    elif obj.name == "transport_obj2":
        return extract_transport_obj2_pose(obs)
    elif obj.name == "target_obj":
        return extract_target_obj_pose(obs)
    elif "region" in obj.name:
        return extract_region_pose(obs, obj)
    else:
        raise ValueError(f"Unknown object name: {obj.name}")


def extract_object_held(obs: Tensor, obj: Object) -> Tensor:
    """Extract specified object held status from observation."""
    if obj.name == "transport_obj1":
        return extract_transport_obj1_held(obs)
    elif obj.name == "transport_obj2":
        return extract_transport_obj2_held(obs)
    else:
        return torch.zeros_like(obs[:, 0])  # Not held


def extract_handempty(obs: Tensor) -> Tensor:
    """Extract handempty status from observation."""
    transport_obj1_held = extract_transport_obj1_held(obs).to(torch.bool)
    transport_obj2_held = extract_transport_obj2_held(obs).to(torch.bool)
    handempty = ~(transport_obj1_held | transport_obj2_held)
    return handempty


def extract_region_activated(obs: Tensor, region_obj: Object) -> Tensor:
    """Extract region activated status from observation."""
    assert "region" in region_obj.name, "Object is not a region."
    names = ["icy_region", "muddy_region", "sandy_region"]
    scenarios = CFG.scenario.split(",")
    name_id = names.index(region_obj.name)
    scenario_idx = scenarios.index(str(name_id + 1))
    s = 20 * scenario_idx + 14
    e = s + 1
    return obs[:, s:e].clone()


def object_in_room(object_pos: Tensor, room_name: str, door_offset: float) -> Tensor:
    """Check if robot is in the specified room."""
    room_x, room_y = ROOM_CENTERS[room_name]
    room_center = torch.zeros_like(object_pos)
    room_center[:, 0] = room_x
    room_center[:, 1] = room_y
    center_offset = CFG.i_trans_world_size[0] / 4 - door_offset
    return (
        (object_pos[:, 0] > room_center[:, 0] - center_offset)
        & (object_pos[:, 0] < room_center[:, 0] + center_offset)
        & (object_pos[:, 1] > room_center[:, 1] - center_offset)
        & (object_pos[:, 1] < room_center[:, 1] + center_offset)
    )
