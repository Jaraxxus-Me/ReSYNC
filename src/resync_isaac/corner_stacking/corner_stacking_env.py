"""Corner Stacking environment wrapping UWLab OmniReset cube task.

This module provides a thin wrapper around the UWLab OmniReset
UR5e + Robotiq 2F-85 cube assembly environment (Isaac Sim / Isaac Lab).

IMPORTANT: Isaac Lab requires the simulator runtime to be launched before
any isaaclab/omni imports. Callers must invoke ``isaaclab.app.AppLauncher``
before importing this module. See ``tests/corner_stacking`` for examples.

Pose-extraction helpers live in :mod:`resync_isaac.corner_stacking.utils`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

# Register UWLab tasks so gym.make can find them
import uwlab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.event_manager import EventManager as _EventManager
from isaaclab.utils.math import compute_pose_error
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    Ur5eRobotiq2f140RelCartesianOSCEvalCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140.rl_state_cfg import (
    variants as omnireset_variants,
)

from skill_refactor.settings import CFG

# IsaacLab defers instantiation of class-based ``EventTerm`` ``func`` (subclasses
# of ``ManagerTermBase``) to a timeline PLAY callback that does not always fire
# in time before ``EventManager.apply(mode="startup")`` runs. Patch ``apply`` so
# it triggers scene-entity resolution synchronously when needed.
if not getattr(_EventManager, "_skill_refactor_apply_patched", False):
    _original_event_manager_apply = _EventManager.apply

    def _event_manager_apply_with_resolution(self, *args, **kwargs):
        if not getattr(self, "_is_scene_entities_resolved", True):
            self._resolve_terms_callback(None)
        return _original_event_manager_apply(self, *args, **kwargs)

    _EventManager.apply = _event_manager_apply_with_resolution
    _EventManager._skill_refactor_apply_patched = True

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

CUBE_TASK_ID = "OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0"

# OSC action scale matching rl_state_cfg training actions
ACTION_SCALE = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])

# Env timing: decimation=12, dt=1/120 → each policy step = 0.1s
DECIMATION = 12
SIM_DT = 1.0 / 120.0
POLICY_DT = DECIMATION * SIM_DT  # 0.1s per step


# ---------------------------------------------------------------------------
# Scripted controller
# ---------------------------------------------------------------------------


class ScriptedBringToGoal:
    """Scripted controller in the same action space as the RL policy.

    The RL policy outputs, per step, a normalized delta ``a ∈ [-1, 1]^6``
    which Isaac Lab's ``RelCartesianOSCAction`` scales by the live action
    term's ``scale_xyz_axisangle`` and feeds to the OSC as a change in EE
    target pose. We produce actions in the exact same space: given the
    remaining pose error, compute the
    per-step-normalized delta that would close it in one step, clamped to
    the admissible range. No proportional / PID gains — the action IS the
    per-step delta, so the controller is straight-line interpolation at
    the maximum admissible speed until within one step of the goal.

    Formally::

        action = clamp(pose_error / action_scale, -1, 1)

    For ``|err| ≥ action_scale[i]`` the axis saturates at ±1 (max speed);
    for ``|err| < action_scale[i]`` the output shrinks linearly and the
    next OSC update requests exactly the remaining delta.
    """

    def __init__(
        self,
        action_scale: torch.Tensor,
        num_envs: int,
        device: torch.device | str,
        env: Any | None = None,
    ) -> None:
        self.goal_pos: torch.Tensor | None = None
        self.goal_quat: torch.Tensor | None = None
        self.num_envs = num_envs
        self.device = device
        self._env = env
        arm = self._get_arm_action_term()
        if arm is not None and hasattr(arm, "_scale"):
            self.action_scale = arm._scale.detach().clone().to(device)
        else:
            self.action_scale = action_scale.to(device)

    def _get_arm_action_term(self) -> Any | None:
        """Return the OSC arm action term when this controller owns an env."""
        if self._env is None:
            return None
        unwrapped = getattr(self._env, "unwrapped", self._env)
        action_manager = getattr(unwrapped, "action_manager", None)
        terms = getattr(action_manager, "_terms", None)
        arm = terms.get("arm") if isinstance(terms, dict) else None
        if arm is None or not hasattr(arm, "_scale"):
            return None
        return arm

    def set_goal(
        self,
        goal_pos_b: torch.Tensor,
        goal_quat_b: torch.Tensor,
        env_mask: torch.Tensor | None = None,
    ) -> None:
        """Set the goal pose in robot base frame.

        Args:
            goal_pos_b: (N, 3) goal position in robot base frame.
            goal_quat_b: (N, 4) goal orientation in robot base frame.
            env_mask: Optional (N,) bool mask. When provided, only those
                rows are updated; this lets vectorized callers snapshot a
                static goal once per env as each env enters a phase.
        """
        goal_pos = goal_pos_b.to(self.device)
        goal_quat = goal_quat_b.to(self.device)
        if env_mask is None:
            self.goal_pos = goal_pos
            self.goal_quat = goal_quat
            return

        env_mask = env_mask.to(device=self.device, dtype=torch.bool).flatten()
        if not bool(env_mask.any().item()):
            return
        if self.goal_pos is None or self.goal_quat is None:
            self.goal_pos = goal_pos.clone()
            self.goal_quat = goal_quat.clone()
            return
        self.goal_pos[env_mask] = goal_pos[env_mask]
        self.goal_quat[env_mask] = goal_quat[env_mask]

    def compute(
        self, ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute normalized OSC actions toward the goal.

        Returns:
            arm_actions: (N, 6), each axis ``err / scale`` clamped to [-1, 1].
            pos_error_norm: (N,) ``‖pos_error‖`` (metres), reported for logging.
            rot_error_norm: (N,) ``‖axis_angle_error‖`` (rad), for logging.
        """
        assert (
            self.goal_pos is not None and self.goal_quat is not None
        ), "Call set_goal() before compute()"
        pos_error, rot_error_aa = compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            self.goal_pos,
            self.goal_quat,
            rot_error_type="axis_angle",
        )
        raw = torch.zeros(pos_error.shape[0], 6, device=pos_error.device)
        raw[:, :3] = pos_error / self.action_scale[:3]
        raw[:, 3:] = rot_error_aa / self.action_scale[3:]
        # Match compare_ur5e_wrist_torque.py: normalized deltas are clipped to
        # the OSC action bounds before the action term applies its own scale.
        arm_actions = raw.clamp(-3.0, 3.0)
        return arm_actions, pos_error.norm(dim=-1), rot_error_aa.norm(dim=-1)


# ---------------------------------------------------------------------------
# Observation adapter
# ---------------------------------------------------------------------------


class CubeCompatObsWrapper(gym.Wrapper):
    """Adapt cornered-stacking observations to the cube-policy obs format.

    Sits between the raw ``ManagerBasedRLEnv`` and ``RslRlVecEnvWrapper`` so
    a checkpoint trained on the original two-object cube task can run on the
    new three-object env. For each named obs group, columns beyond
    ``target_dims[group]`` are dropped from the right, on the assumption
    that any new obs terms (e.g. outside_block pose) are appended after the
    legacy ones.

    Currently :class:`CorneredStackingCfg` reuses the parent cube task's
    ``ObservationsCfg`` verbatim, so the wrapper is effectively a no-op
    when ``target_dims`` is left empty — but adding it to the test wires up
    the slicing seam so a future PolicyCfg extension does not silently
    misalign the cube checkpoint.

    NOTE: ``RslRlVecEnvWrapper.get_observations()`` calls
    ``self.unwrapped.observation_manager.compute()`` directly and so does
    NOT pass through this wrapper. Today that is fine because the obs
    layout still matches; once new terms land, swap this for a subclassed
    ``RslRlVecEnvWrapper`` that overrides ``get_observations`` / ``step``.
    """

    def __init__(
        self,
        env: gym.Env,
        target_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__(env)
        self._target_dims: dict[str, int] = (
            {k: int(v) for k, v in target_dims.items()} if target_dims else {}
        )

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return self._adapt(obs), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._adapt(obs), reward, terminated, truncated, info

    def _adapt(self, obs):
        if not self._target_dims or not isinstance(obs, dict):
            return obs
        adapted = dict(obs)
        for group, target in self._target_dims.items():
            tensor = adapted.get(group)
            if tensor is not None and tensor.shape[-1] > target:
                adapted[group] = tensor[..., :target]
        return adapted


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_corner_stacking_env(
    num_envs: int | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    env_cfg: ManagerBasedRLEnvCfg | None = None,
    task_id: str | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a corner-stacking environment.

    Wraps one of three task configs via ``gym.make``. The returned env is the
    raw Isaac Lab ``ManagerBasedRLEnv`` — callers can wrap it further (e.g.
    with ``RslRlVecEnvWrapper``).

    Args:
        num_envs: Number of parallel environments. Defaults to
            ``CFG.c_stack_num_envs``.
        max_steps: Maximum policy steps per episode.  Sets
            ``episode_length_s`` so the env's own ``time_out``
            termination handles truncation. Defaults to
            ``CFG.c_stack_max_steps``.
        device: Torch device string (e.g. ``"cuda:0"``).
        env_cfg: Optional override for the full env config.  When *None*,
            a fresh cfg matching ``task_id`` is instantiated.
        task_id: Gym id to register against. Defaults to the custom
            two-cube reset task. Supported ids:

            * :data:`CORNER_STACKING_CUSTOM_TASK_ID` (default) — upstream
              InsertiveCube/ReceptiveCube pair with uniform XY reset.
            * :data:`CUBE_TASK_ID` — stock UWLab cube Play task.
            * :data:`CORNERED_STACKING_TASK_ID` — three-object (box,
              grasp block, outside block) sparse-reward tower-grasp task;
              defined in :mod:`.cornered_stacking_task_cfg`.

    Returns:
        A gymnasium environment.
    """
    if num_envs is None:
        num_envs = CFG.c_stack_num_envs
    if max_steps is None:
        max_steps = CFG.c_stack_max_steps

    # Default task is the custom cube-reset task; upstream UWLab id is
    # retained as an opt-in fallback via ``task_id=CUBE_TASK_ID``.
    from resync_isaac.corner_stacking.task_cfg import (
        CORNER_STACKING_CUSTOM_TASK_ID,
        CornerStackingCustomCfg,
    )
    from resync_isaac.corner_stacking.cornered_stacking_task_cfg import (
        CORNERED_STACKING_TASK_ID,
        CorneredStackingCfg,
    )

    if task_id is None:
        task_id = CORNER_STACKING_CUSTOM_TASK_ID

    use_default_robot_cfg = env_cfg is None
    if env_cfg is None:
        if task_id == CORNER_STACKING_CUSTOM_TASK_ID:
            env_cfg = CornerStackingCustomCfg()
        elif task_id == CUBE_TASK_ID:
            env_cfg = Ur5eRobotiq2f140RelCartesianOSCEvalCfg()
        elif task_id == CORNERED_STACKING_TASK_ID:
            env_cfg = CorneredStackingCfg()
        else:
            raise ValueError(
                f"Unknown task_id: {task_id}. Pass a matching env_cfg instead."
            )

    env_cfg.scene.num_envs = num_envs

    if device is not None:
        env_cfg.sim.device = device

    # Set episode length so the env's time_out termination handles truncation
    env_cfg.episode_length_s = max_steps * POLICY_DT

    if use_default_robot_cfg:
        # Keep local corner-stacking tasks pinned to the same full 2F-140 robot
        # used by the UWLab eval/demo task. Object variants below may change,
        # but the robot USD, actuator model, defaults, and action term stay
        # aligned with OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0.
        eval_cfg = Ur5eRobotiq2f140RelCartesianOSCEvalCfg()
        env_cfg.scene.robot = eval_cfg.scene.robot.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        env_cfg.actions = eval_cfg.actions

    # Select USD variants (use module-level dict since hydra-instantiated
    # configs may not carry the class-level ``variants`` attribute).
    if task_id == CORNERED_STACKING_TASK_ID:
        # Cornered stacking uses the cornered_block pair; outside_block uses
        # ``block_blue.usd``, a thin USD override layer that references
        # ``block.usd`` and rewrites ONLY the green (PreviewSurface) material's
        # diffuseColor. The red top face (PreviewSurfaceFace), all geometry,
        # and physics/mass/collider data come through unchanged — so the two
        # blocks are visually distinguishable (blue sides vs green sides) but
        # share the red top and behave identically in sim.
        block_variant = omnireset_variants["scene.insertive_object"]["block"]
        env_cfg.scene.insertive_object = block_variant
        env_cfg.scene.receptive_object = omnireset_variants["scene.receptive_object"][
            "box"
        ]
        blue_usd_path = block_variant.spawn.usd_path.replace(
            "block/block.usd", "block/block_blue.usd"
        )
        env_cfg.scene.outside_block = block_variant.replace(
            prim_path="{ENV_REGEX_NS}/OutsideBlock",
            spawn=block_variant.spawn.replace(usd_path=blue_usd_path),
        )
    else:
        env_cfg.scene.insertive_object = omnireset_variants["scene.insertive_object"][
            "cube"
        ]
        env_cfg.scene.receptive_object = omnireset_variants["scene.receptive_object"][
            "cube"
        ]

    from resync_isaac.runtime import configure_scene

    configure_scene(env_cfg)
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
    return env
