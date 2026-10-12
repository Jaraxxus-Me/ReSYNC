"""Isaac Lab task configuration for flipped stacking.

The scene contains two dynamic 6 cm ReSYNC cubes. Either cube, both cubes, or
neither cube can start with randomized roll/pitch. Unlike the upstream
OmniReset RL environment, the second scene object is not a kinematic receptive
object; ``receptive_object`` is simply the second dynamic cube so the canonical
215-column state observation can be reused unchanged.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    subtract_frame_transforms,
    wrap_to_pi,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    ObservationsCfg,
    RlStateSceneCfg,
    TerminationsCfg,
    Ur5eRobotiq2f140RelCartesianOSCEvalCfg,
    variants,
)
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import reset_scene_to_default

from skill_refactor.settings import CFG

FLIPPED_STACKING_TASK_ID = "FlippedStacking-Ur5eRobotiq2f140-v0"
POLICY_OBS_DIM = 215
POLICY_DT = 12.0 / 120.0


def _preview_material(
    stage: Any,
    path: str,
    color: tuple[float, float, float],
    *,
    emissive: bool = False,
):
    """Create a saturated, non-metallic USD preview material."""
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _arrow_mesh(
    stage: Any, path: str, points: list[tuple[float, float, float]], material: Any
) -> None:
    """Author one visual-only planar arrow component."""
    from pxr import Gf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(points)])
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
        [Gf.Vec3f(1.0, 0.0, 0.0)]
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def _add_up_arrows(stage: Any, prim_path: str) -> None:
    """Add red local-+Z arrows to the four vertical cube faces."""
    from pxr import UsdGeom

    visuals_path = f"{prim_path}/visuals"
    arrows_path = f"{visuals_path}/UpArrows"
    UsdGeom.Xform.Define(stage, arrows_path)
    red = _preview_material(
        stage,
        f"{arrows_path}/RedMaterial",
        (1.0, 0.0, 0.0),
        emissive=True,
    )
    face = 0.0305
    shaft_half = 0.0035
    head_half = 0.011
    shaft_bottom, shaft_top, tip = -0.019, 0.004, 0.022

    for label, sign in (("PosX", 1.0), ("NegX", -1.0)):
        x = sign * face
        _arrow_mesh(
            stage,
            f"{arrows_path}/{label}Shaft",
            [
                (x, -shaft_half, shaft_bottom),
                (x, shaft_half, shaft_bottom),
                (x, shaft_half, shaft_top),
                (x, -shaft_half, shaft_top),
            ],
            red,
        )
        _arrow_mesh(
            stage,
            f"{arrows_path}/{label}Head",
            [(x, -head_half, shaft_top), (x, head_half, shaft_top), (x, 0.0, tip)],
            red,
        )

    for label, sign in (("PosY", 1.0), ("NegY", -1.0)):
        y = sign * face
        _arrow_mesh(
            stage,
            f"{arrows_path}/{label}Shaft",
            [
                (-shaft_half, y, shaft_bottom),
                (shaft_half, y, shaft_bottom),
                (shaft_half, y, shaft_top),
                (-shaft_half, y, shaft_top),
            ],
            red,
        )
        _arrow_mesh(
            stage,
            f"{arrows_path}/{label}Head",
            [(-head_half, y, shaft_top), (head_half, y, shaft_top), (0.0, y, tip)],
            red,
        )


@clone
def spawn_directional_cube_from_usd(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    """Spawn the physical ReSYNC cube and replace only its visuals."""
    del kwargs
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation)
    stage = prim.GetStage()
    is_bottom = prim_path.endswith("/BottomCube")
    base_color = (0.02, 0.20, 0.95) if is_bottom else (0.50, 0.05, 0.80)
    base = _preview_material(stage, f"{prim_path}/visuals/SolidColor", base_color)

    from pxr import UsdShade

    cube_visual = stage.GetPrimAtPath(f"{prim_path}/visuals/cube")
    if not cube_visual.IsValid():
        raise RuntimeError(f"Missing ReSYNC cube visual under {prim_path}")
    UsdShade.MaterialBindingAPI.Apply(cube_visual).Bind(
        base, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    _add_up_arrows(stage, prim_path)
    return prim


def _sample_uniform(
    count: int, limits: tuple[float, float], device: torch.device | str
) -> torch.Tensor:
    lo, hi = float(limits[0]), float(limits[1])
    return lo + (hi - lo) * torch.rand(count, device=device)


def reset_flipped_stacking(
    env: Any,
    env_ids: torch.Tensor,
    flipped_cube: str,
    num_of_blocks: int,
    home_joints: tuple[float, ...],
    home_joint_noise: float,
    cube_x_range: tuple[float, float],
    cube_y_range: tuple[float, float],
    cube_center_z: float,
    cube_min_xy_dist: float,
    flipped_roll_range: tuple[float, float],
    flipped_pitch_range: tuple[float, float],
    cube_yaw_range: tuple[float, float],
) -> None:
    """Reset the robot and two/four cubes without a reset-state dataset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = len(env_ids)
    if num_of_blocks not in (2, 3, 4):
        raise ValueError(f"num_of_blocks must be 2, 3, or 4, got {num_of_blocks}")
    block_roles = {
        2: ("top", "bottom"),
        3: ("top", "mid1", "bottom"),
        4: ("top", "mid1", "mid2", "bottom"),
    }[num_of_blocks]
    valid_roles = (*block_roles, "both", "none", "random_pair")
    explicit_pair = tuple(flipped_cube.split(",")) if "," in flipped_cube else ()
    if explicit_pair and (
        len(explicit_pair) != 2
        or len(set(explicit_pair)) != 2
        or not set(explicit_pair) <= set(block_roles)
    ):
        raise ValueError(f"Invalid flipped pair: {flipped_cube}")
    if not explicit_pair and flipped_cube not in valid_roles:
        raise ValueError(
            f"fstack_flipped_cube must be one of {valid_roles}, "
            f"got {flipped_cube!r}"
        )

    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    arm_ids, _ = robot.find_joints(
        ["shoulder.*", "elbow.*", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    )
    if len(arm_ids) != 6:
        raise RuntimeError(f"Expected six UR5e joints, got {arm_ids}")
    home = torch.tensor(home_joints, dtype=joint_pos.dtype, device=env.device)
    noise = (2.0 * torch.rand((count, 6), device=env.device) - 1.0) * float(
        home_joint_noise
    )
    joint_pos[:, arm_ids] = home + noise
    # All gripper/mimic joints retain their open default positions.
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

    if num_of_blocks >= 3:
        # Several unconstrained rejection samples tend to occupy extreme corners.
        # Keep the final bottom at an accessible central stack site and
        # randomize the movable roles over separated staging sites.
        x_lo, x_hi = map(float, cube_x_range)
        y_lo, y_hi = map(float, cube_y_range)
        staging = torch.tensor(
            [
                [x_hi - 0.08, y_lo + 0.02],
                [x_hi - 0.08, (y_lo + y_hi) / 2.0],
                [x_hi - 0.08, y_hi - 0.02],
            ],
            device=env.device,
        )
        movable_count = num_of_blocks - 1
        staging_pool = staging[[0, 2]] if num_of_blocks == 3 else staging
        movable_xy = torch.stack(
            [
                staging_pool[
                    torch.randperm(len(staging_pool), device=env.device)[:movable_count]
                ]
                for _ in range(count)
            ]
        )
        bottom_xy = torch.tensor(
            [
                x_lo + (0.20 if num_of_blocks == 3 else 0.07),
                (y_lo + y_hi) / 2.0,
            ],
            device=env.device,
        ).expand(count, 1, 2)
        xy = torch.cat((movable_xy, bottom_xy), dim=1)
        xy += 0.005 * (2.0 * torch.rand_like(xy) - 1.0)
        pairwise = torch.cdist(xy, xy)
        pairwise += torch.eye(num_of_blocks, device=env.device)[None] * 10.0
        if bool((pairwise.amin(dim=(1, 2)) < float(cube_min_xy_dist)).any()):
            raise RuntimeError("Multi-block interior reset violated cube clearance")
    else:
        # Preserve the original two-block reset distribution for regression.
        xy = torch.empty((count, num_of_blocks, 2), device=env.device)
        for block_idx in range(num_of_blocks):
            pending = torch.ones(count, dtype=torch.bool, device=env.device)
            for _ in range(256):
                candidate = torch.stack(
                    (
                        _sample_uniform(count, cube_x_range, env.device),
                        _sample_uniform(count, cube_y_range, env.device),
                    ),
                    dim=-1,
                )
                if block_idx == 0:
                    accepted = torch.ones(count, dtype=torch.bool, device=env.device)
                else:
                    accepted = (candidate[:, None, :] - xy[:, :block_idx, :]).norm(
                        dim=-1
                    ).amin(dim=1) >= float(cube_min_xy_dist)
                newly_accepted = accepted & pending
                xy[newly_accepted, block_idx] = candidate[newly_accepted]
                pending[newly_accepted] = False
                if not bool(pending.any().item()):
                    break
            else:
                raise RuntimeError("Could not sample two collision-free cubes")

    flipped_mask = torch.zeros(
        (count, num_of_blocks), dtype=torch.bool, device=env.device
    )
    if explicit_pair:
        for role in explicit_pair:
            flipped_mask[:, block_roles.index(role)] = True
    elif flipped_cube in ("both", "random_pair"):
        for row in range(count):
            selected = torch.randperm(num_of_blocks, device=env.device)[:2]
            flipped_mask[row, selected] = True
    elif flipped_cube != "none":
        flipped_mask[:, block_roles.index(flipped_cube)] = True

    origins = env.scene.env_origins[env_ids]
    velocity = torch.zeros((count, 6), device=env.device)
    scene_names = {
        2: ("insertive_object", "receptive_object"),
        3: ("insertive_object", "mid1_object", "receptive_object"),
        4: (
            "insertive_object",
            "mid1_object",
            "mid2_object",
            "receptive_object",
        ),
    }[num_of_blocks]
    shared_yaw = None
    if num_of_blocks >= 3:
        shared_yaw = _sample_uniform(
            count, tuple(CFG.fstack_flip_region_yaw_range), env.device
        )
    for block_idx, scene_name in enumerate(scene_names):
        pos = origins.clone()
        pos[:, :2] += xy[:, block_idx]
        pos[:, 2] += float(cube_center_z)
        yaw = (
            shared_yaw.clone()
            if shared_yaw is not None
            else _sample_uniform(count, cube_yaw_range, env.device)
        )
        roll = torch.zeros(count, device=env.device)
        pitch = torch.zeros(count, device=env.device)
        selected = flipped_mask[:, block_idx]
        if bool(selected.any().item()):
            n = int(selected.sum().item())
            roll[selected] = _sample_uniform(n, flipped_roll_range, env.device)
            pitch[selected] = _sample_uniform(n, flipped_pitch_range, env.device)
            too_upright = selected & (
                wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs() < 0.25
            )
            while bool(too_upright.any().item()):
                n = int(too_upright.sum().item())
                roll[too_upright] = _sample_uniform(n, flipped_roll_range, env.device)
                pitch[too_upright] = _sample_uniform(n, flipped_pitch_range, env.device)
                too_upright = selected & (
                    wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs() < 0.25
                )
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        env.scene[scene_name].write_root_pose_to_sim(
            torch.cat((pos, quat), dim=-1), env_ids=env_ids
        )
        env.scene[scene_name].write_root_velocity_to_sim(velocity, env_ids=env_ids)


@configclass
class FlippedStackingSceneCfg(RlStateSceneCfg):
    """Parent OmniReset scene with optional middle dynamic ReSYNC cubes."""

    mid1_object: RigidObjectCfg | None = None
    mid2_object: RigidObjectCfg | None = None

    top_bottom_contact: ContactSensorCfg | None = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/TopCube",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BottomCube"],
    )
    top_mid1_contact: ContactSensorCfg | None = None
    mid1_bottom_contact: ContactSensorCfg | None = None
    mid1_mid2_contact: ContactSensorCfg | None = None
    mid2_bottom_contact: ContactSensorCfg | None = None


def block_supported_by(
    env: Any, upper_role: str, lower_role: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return support validity and the filtered/other force for one pair."""
    sensor_name = f"{upper_role}_{lower_role}_contact"
    sensor = env.unwrapped.scene[sensor_name]
    force_matrix = sensor.data.force_matrix_w
    net_forces = sensor.data.net_forces_w
    if force_matrix is None or net_forces is None:
        raise RuntimeError(f"Contact sensor {sensor_name!r} is not initialized")
    lower_vector = force_matrix[:, :, 0, :].sum(dim=1)
    net_vector = net_forces.sum(dim=1)
    lower_force = lower_vector.norm(dim=-1)
    other_force = (net_vector - lower_vector).norm(dim=-1)
    valid = (lower_force >= float(CFG.fstack_bottom_contact_min_force)) & (
        other_force <= float(CFG.fstack_other_contact_max_force)
    )
    return valid, lower_force, other_force


def top_supported_only_by_bottom(
    env: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return support validity and the bottom/other normal-force magnitudes."""
    return block_supported_by(env, "top", "bottom")


@configclass
class FlippedStackingEventCfg:
    reset_everything: EventTerm = EventTerm(
        func=reset_scene_to_default, mode="reset", params={}
    )
    reset_task: EventTerm = EventTerm(
        func=reset_flipped_stacking,
        mode="reset",
        params={
            "flipped_cube": "top",
            "num_of_blocks": 2,
            "home_joints": tuple(CFG.fstack_home_joints),
            "home_joint_noise": float(CFG.fstack_home_joint_noise),
            "cube_x_range": tuple(CFG.fstack_cube_x_range),
            "cube_y_range": tuple(CFG.fstack_cube_y_range),
            "cube_center_z": float(CFG.fstack_cube_center_z),
            "cube_min_xy_dist": float(CFG.fstack_cube_min_xy_dist),
            "flipped_roll_range": tuple(CFG.fstack_flipped_roll_range),
            "flipped_pitch_range": tuple(CFG.fstack_flipped_pitch_range),
            "cube_yaw_range": tuple(CFG.fstack_cube_yaw_range),
        },
    )

    def __post_init__(self) -> None:
        self.reset_task.params["flipped_cube"] = CFG.fstack_flipped_cube
        self.reset_task.params["num_of_blocks"] = int(CFG.num_of_blocks)


class StackedUprightSuccess(ManagerTermBase):
    """Sparse final goal: the requested ordered tower, all blocks top-up."""

    def __init__(self, cfg: RewTerm, env: Any) -> None:
        super().__init__(cfg, env)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: Any,
        upright_tol: float,
        xy_tol: float,
        z_offset: float,
        z_tol: float,
        yaw_tol: float,
        open_finger_max: float,
    ) -> torch.Tensor:
        try:
            env.scene["mid1_object"]
        except KeyError:
            roles: tuple[str, ...] = ("top", "bottom")
        else:
            try:
                env.scene["mid2_object"]
            except KeyError:
                roles = ("top", "mid1", "bottom")
            else:
                roles = ("top", "mid1", "mid2", "bottom")
        scene_names = {
            "top": "insertive_object",
            "mid1": "mid1_object",
            "mid2": "mid2_object",
            "bottom": "receptive_object",
        }
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for role in roles:
            block: RigidObject = env.scene[scene_names[role]]
            roll, pitch, _ = euler_xyz_from_quat(block.data.root_quat_w)
            valid &= wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs() < upright_tol
        for upper_role, lower_role in zip(roles[:-1], roles[1:]):
            upper: RigidObject = env.scene[scene_names[upper_role]]
            lower: RigidObject = env.scene[scene_names[lower_role]]
            rel_pos, rel_quat = subtract_frame_transforms(
                lower.data.root_pos_w,
                lower.data.root_quat_w,
                upper.data.root_pos_w,
                upper.data.root_quat_w,
            )
            _, _, rel_yaw = euler_xyz_from_quat(rel_quat)
            sensor = env.scene[f"{upper_role}_{lower_role}_contact"]
            pair_force = sensor.data.force_matrix_w
            if pair_force is None:
                raise RuntimeError("Final-stack contact sensor is not initialized")
            touching = pair_force[:, :, 0, :].sum(dim=1).norm(dim=-1) >= float(
                CFG.fstack_bottom_contact_min_force
            )
            valid &= (
                (rel_pos[:, :2].norm(dim=-1) < xy_tol)
                & ((rel_pos[:, 2] - z_offset).abs() < z_tol)
                & (wrap_to_pi(rel_yaw).abs() < yaw_tol)
                & touching
            )
        robot: Articulation = env.scene["robot"]
        finger_ids, _ = robot.find_joints("finger_joint")
        released = robot.data.joint_pos[:, finger_ids[0]] < open_finger_max
        self.success[:] = valid & released
        return self.success.float()


@configclass
class FlippedStackingRewardsCfg:
    progress_context: RewTerm = RewTerm(
        func=StackedUprightSuccess,
        weight=1.0,
        params={
            "upright_tol": float(CFG.fstack_upright_tol),
            "xy_tol": float(CFG.fstack_on_xy_tol),
            "z_offset": float(CFG.fstack_on_z_offset),
            "z_tol": float(CFG.fstack_on_z_tol),
            "yaw_tol": float(CFG.fstack_on_yaw_tol),
            "open_finger_max": float(CFG.fstack_flip_open_finger_max),
        },
    )


@configclass
class EmptyCommandsCfg:
    pass


@configclass
class FlippedStackingCfg(Ur5eRobotiq2f140RelCartesianOSCEvalCfg):
    scene: FlippedStackingSceneCfg = FlippedStackingSceneCfg(
        num_envs=1, env_spacing=1.5
    )
    observations: ObservationsCfg = ObservationsCfg()
    events: FlippedStackingEventCfg = FlippedStackingEventCfg()
    rewards: FlippedStackingRewardsCfg = FlippedStackingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: EmptyCommandsCfg = EmptyCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(CFG.num_of_blocks) not in (2, 3, 4):
            raise ValueError(
                f"CFG.num_of_blocks must be 2, 3, or 4, got {CFG.num_of_blocks}"
            )
        self.viewer.eye = (-2.0, 0.0, 0.75)
        self.events.reset_task.params["flipped_cube"] = CFG.fstack_flipped_cube
        self.events.reset_task.params["num_of_blocks"] = int(CFG.num_of_blocks)
        cube_cfg = variants["scene.insertive_object"]["cube_resync"]
        self.scene.insertive_object = cube_cfg.replace(
            prim_path="{ENV_REGEX_NS}/TopCube"
        )
        self.scene.receptive_object = cube_cfg.replace(
            prim_path="{ENV_REGEX_NS}/BottomCube"
        )
        self.scene.insertive_object.spawn.func = spawn_directional_cube_from_usd
        self.scene.receptive_object.spawn.func = spawn_directional_cube_from_usd
        self.scene.insertive_object.spawn.activate_contact_sensors = True
        self.scene.receptive_object.spawn.activate_contact_sensors = True
        if int(CFG.num_of_blocks) >= 3:
            self.scene.mid1_object = cube_cfg.replace(
                prim_path="{ENV_REGEX_NS}/Mid1Cube"
            )
            middles = [self.scene.mid1_object]
            if int(CFG.num_of_blocks) == 4:
                self.scene.mid2_object = cube_cfg.replace(
                    prim_path="{ENV_REGEX_NS}/Mid2Cube"
                )
                middles.append(self.scene.mid2_object)
            for middle in middles:
                middle.spawn.func = spawn_directional_cube_from_usd
                middle.spawn.activate_contact_sensors = True
            self.scene.top_bottom_contact = None
            self.scene.top_mid1_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/TopCube",
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Mid1Cube"],
            )
            if int(CFG.num_of_blocks) == 3:
                self.scene.mid1_bottom_contact = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Mid1Cube",
                    update_period=0.0,
                    history_length=3,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/BottomCube"],
                )
            else:
                self.scene.mid1_mid2_contact = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Mid1Cube",
                    update_period=0.0,
                    history_length=3,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Mid2Cube"],
                )
                self.scene.mid2_bottom_contact = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Mid2Cube",
                    update_period=0.0,
                    history_length=3,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/BottomCube"],
                )


gym.register(
    id=FLIPPED_STACKING_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:FlippedStackingCfg"},
)
