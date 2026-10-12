"""Observation, pose, checkpoint, and virtual-region helpers."""

from __future__ import annotations

import os
from typing import Any

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_rotate,
    subtract_frame_transforms,
    wrap_to_pi,
)
from torch import Tensor

from resync_isaac.corner_stacking.utils import (
    CornerStackingObsLayout,
    get_ee_pose_in_base,
    get_ee_pose_in_world,
    get_robot_body_pose_in_base,
    goal_pose_base_to_world,
    load_block_box_rl_policy,
    top_down_target_quat,
    top_down_wrist_target_quat,
)
from resync_isaac.flipped_stacking.task_cfg import POLICY_OBS_DIM
from skill_refactor.settings import CFG

TOP_CUBE = "top"
MID1_CUBE = "mid1"
MID2_CUBE = "mid2"
BOTTOM_CUBE = "bottom"
_CUBE_BOTTOM_OFFSET = torch.tensor([0.0, 0.0, -0.03])
_REGION_TOP_OFFSET = torch.tensor([0.0, 0.0, 0.0005])


def block_names(num_of_blocks: int | None = None) -> tuple[str, ...]:
    """Return blocks in final top-to-bottom order."""
    count = int(CFG.num_of_blocks if num_of_blocks is None else num_of_blocks)
    if count == 2:
        return (TOP_CUBE, BOTTOM_CUBE)
    if count == 3:
        return (TOP_CUBE, MID1_CUBE, BOTTOM_CUBE)
    if count == 4:
        return (TOP_CUBE, MID1_CUBE, MID2_CUBE, BOTTOM_CUBE)
    raise ValueError(f"num_of_blocks must be 2, 3, or 4, got {count}")


def cube_scene_name(cube_name: str) -> str:
    """Map a symbolic final-stack role to its Isaac scene entity."""
    scene_name = {
        TOP_CUBE: "insertive_object",
        MID1_CUBE: "mid1_object",
        MID2_CUBE: "mid2_object",
        BOTTOM_CUBE: "receptive_object",
    }.get(cube_name)
    if scene_name is None:
        raise ValueError(f"Unknown cube {cube_name!r}")
    return scene_name


def active_block_names(env: Any) -> tuple[str, ...]:
    """Return the block order from the instantiated scene, independent of CFG."""
    try:
        env.unwrapped.scene["mid1_object"]
    except KeyError:
        return block_names(2)
    try:
        env.unwrapped.scene["mid2_object"]
    except KeyError:
        return block_names(3)
    return block_names(4)


def resolve_flipped_stacking_checkpoint() -> str | None:
    path = os.path.abspath(os.path.expanduser(str(CFG.fstack_checkpoint)))
    return path if os.path.isfile(path) else None


def load_flip_rl_policy(raw_env: Any, agent_cfg: Any, checkpoint: str):
    """Load the ReSYNC cube/region policy against the 215-D canonical env."""
    return load_block_box_rl_policy(raw_env, agent_cfg, checkpoint)


def get_cube_pose_in_base(env: Any, cube_name: str) -> tuple[Tensor, Tensor]:
    scene_name = cube_scene_name(cube_name)
    cube = env.unwrapped.scene[scene_name]
    robot = env.unwrapped.scene["robot"]
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )


def get_finger_joint_pos(env: Any) -> Tensor:
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints("finger_joint")
    return robot.data.joint_pos[:, ids[0]]


def get_task_success(env: Any) -> Tensor:
    term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    return term.success.clone()


def top_down_wrist_goal(
    env: Any, target_pos_b: Tensor, gripper_height: float
) -> tuple[Tensor, Tensor]:
    """Wrist pose that places ``robotiq_base_link`` top-down over a target."""
    wrist_pos, wrist_quat = get_ee_pose_in_base(env)
    grip_pos, grip_quat = get_robot_body_pose_in_base(env, "robotiq_base_link")
    wrist_to_grip_pos, wrist_to_grip_quat = subtract_frame_transforms(
        wrist_pos, wrist_quat, grip_pos, grip_quat
    )
    inv_quat = torch.cat(
        (wrist_to_grip_quat[:, :1], -wrist_to_grip_quat[:, 1:]), dim=-1
    )
    inv_pos = -quat_rotate(inv_quat, wrist_to_grip_pos)
    desired_grip_pos = target_pos_b.clone()
    desired_grip_pos[:, 2] += float(gripper_height)
    desired_grip_quat = top_down_target_quat(grip_quat)
    return combine_frame_transforms(
        desired_grip_pos, desired_grip_quat, inv_pos, inv_quat
    )


def _pose6(
    root_pos: Tensor, root_quat: Tensor, target_pos: Tensor, target_quat: Tensor
) -> Tensor:
    pos, quat = subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
    return torch.cat((pos, axis_angle_from_quat(quat)), dim=-1)


class FlipPolicyObservation:
    """Build the exact ReSYNC 215-D actor input for a grounded cube.

    Robot/action histories come from the live environment. The three
    object-dependent terms are replaced by histories for the grounded cube
    and a virtual region whose base-frame pose is frozen at skill reset.
    """

    HISTORY = 5

    def __init__(self, env: Any, layout: CornerStackingObsLayout) -> None:
        self.env = env
        self.layout = layout
        self.cube_name: str | None = None
        self.region_pos_b: Tensor | None = None
        self.region_quat_b: Tensor | None = None
        self._cube_in_region: Tensor | None = None
        self._cube_in_ee: Tensor | None = None
        self._region_in_ee: Tensor | None = None

    def reset(self, cube_name: str, obs: Tensor) -> None:
        self.cube_name = cube_name
        self.region_pos_b, self.region_quat_b = sample_virtual_region(
            self.env, cube_name
        )
        current = self._current_terms(obs)
        self._cube_in_region = current[0][:, None, :].repeat(1, self.HISTORY, 1)
        self._cube_in_ee = current[1][:, None, :].repeat(1, self.HISTORY, 1)
        self._region_in_ee = current[2][:, None, :].repeat(1, self.HISTORY, 1)

    def _current_terms(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert self.cube_name is not None
        assert self.region_pos_b is not None and self.region_quat_b is not None
        cube_pos_b, cube_quat_b = get_cube_pose_in_base(self.env, self.cube_name)
        ee_pos_b, _ = self.layout.latest_ee_pose_in_base(obs)
        ee_quat_b = self.layout.latest_ee_quat_in_base(obs)
        return (
            _pose6(
                self.region_pos_b,
                self.region_quat_b,
                cube_pos_b,
                cube_quat_b,
            ),
            _pose6(ee_pos_b, ee_quat_b, cube_pos_b, cube_quat_b),
            _pose6(
                ee_pos_b,
                ee_quat_b,
                self.region_pos_b,
                self.region_quat_b,
            ),
        )

    @staticmethod
    def _roll(history: Tensor, value: Tensor) -> None:
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = value

    def build(self, obs: Tensor) -> Tensor:
        if self.cube_name is None:
            raise RuntimeError("Call FlipPolicyObservation.reset() first")
        assert self._cube_in_region is not None
        assert self._cube_in_ee is not None
        assert self._region_in_ee is not None
        cube_in_region, cube_in_ee, region_in_ee = self._current_terms(obs)
        self._roll(self._cube_in_region, cube_in_region)
        self._roll(self._cube_in_ee, cube_in_ee)
        self._roll(self._region_in_ee, region_in_ee)
        out = obs[..., :POLICY_OBS_DIM].clone()
        out[:, 0:30] = self._cube_in_region.flatten(1)
        out[:, 155:185] = self._cube_in_ee.flatten(1)
        out[:, 185:215] = self._region_in_ee.flatten(1)
        return out

    def assembled(self) -> Tensor:
        """ReSYNC goal: alignment points coincide and cube is top-up."""
        assert self.cube_name is not None
        assert self.region_pos_b is not None and self.region_quat_b is not None
        cube_pos, cube_quat = get_cube_pose_in_base(self.env, self.cube_name)
        n = cube_pos.shape[0]
        cube_off = _CUBE_BOTTOM_OFFSET.to(cube_pos).expand(n, -1)
        region_off = _REGION_TOP_OFFSET.to(cube_pos).expand(n, -1)
        cube_align = cube_pos + quat_rotate(cube_quat, cube_off)
        region_align = self.region_pos_b + quat_rotate(self.region_quat_b, region_off)
        roll, pitch, _ = euler_xyz_from_quat(cube_quat)
        upright = wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs()
        return (
            (cube_align - region_align).norm(dim=-1)
            < float(CFG.fstack_flip_position_tol)
        ) & (upright < float(CFG.fstack_flip_upright_tol))


def sample_virtual_region(env: Any, cube_name: str) -> tuple[Tensor, Tensor]:
    """Sample a fixed ReSYNC region near one cube and away from every other cube."""
    cube_pos, _ = get_cube_pose_in_base(env, cube_name)
    other_positions = torch.stack(
        [
            get_cube_pose_in_base(env, name)[0]
            for name in active_block_names(env)
            if name != cube_name
        ],
        dim=1,
    )
    n = cube_pos.shape[0]
    device = cube_pos.device
    result = torch.zeros_like(cube_pos)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    r_lo, r_hi = map(float, CFG.fstack_flip_region_radius_range)
    min_other_distance = float(CFG.fstack_flip_region_other_cube_min_xy_dist)
    x_lo, x_hi = map(float, CFG.fstack_cube_x_range)
    y_lo, y_hi = map(float, CFG.fstack_cube_y_range)
    for _ in range(int(CFG.fstack_flip_region_sample_attempts)):
        ids = torch.nonzero(pending, as_tuple=False).flatten()
        if ids.numel() == 0:
            break
        radius = r_lo + (r_hi - r_lo) * torch.rand(len(ids), device=device)
        angle = 2.0 * torch.pi * torch.rand(len(ids), device=device)
        candidate = cube_pos[ids].clone()
        candidate[:, 0] += radius * torch.cos(angle)
        candidate[:, 1] += radius * torch.sin(angle)
        # The lowest cube center gives a stable estimate of the table surface.
        candidate[:, 2] = other_positions[ids, :, 2].amin(dim=1) - 0.0305
        distances = (candidate[:, None, :2] - other_positions[ids, :, :2]).norm(dim=-1)
        away = distances.amin(dim=1) >= min_other_distance
        in_workspace = (
            (candidate[:, 0] >= x_lo)
            & (candidate[:, 0] <= x_hi)
            & (candidate[:, 1] >= y_lo)
            & (candidate[:, 1] <= y_hi)
        )
        accepted = away & in_workspace
        result[ids[accepted]] = candidate[accepted]
        pending[ids[accepted]] = False
    if bool(pending.any().item()):
        raise RuntimeError(
            "Could not sample a virtual ReSYNC region with required cube clearance"
        )
    yaw_lo, yaw_hi = map(float, CFG.fstack_flip_region_yaw_range)
    yaw = yaw_lo + (yaw_hi - yaw_lo) * torch.rand(n, device=device)
    zero = torch.zeros_like(yaw)
    return result, quat_from_euler_xyz(zero, zero, yaw)


__all__ = [
    "BOTTOM_CUBE",
    "FlipPolicyObservation",
    "MID1_CUBE",
    "MID2_CUBE",
    "TOP_CUBE",
    "active_block_names",
    "block_names",
    "cube_scene_name",
    "CornerStackingObsLayout",
    "get_cube_pose_in_base",
    "get_ee_pose_in_base",
    "get_ee_pose_in_world",
    "get_finger_joint_pos",
    "get_task_success",
    "goal_pose_base_to_world",
    "load_flip_rl_policy",
    "resolve_flipped_stacking_checkpoint",
    "sample_virtual_region",
    "top_down_wrist_target_quat",
    "top_down_wrist_goal",
]
