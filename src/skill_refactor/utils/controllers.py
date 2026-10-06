"""Utility functions for controllers in the skill refactor project."""

from typing import Optional

import gymnasium as gym
import torch
from gymnasium import Env
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import Array
from torch import Tensor

from skill_refactor.benchmarks.icy_transport.utils import (
    extract_robot_mass_moment,
    extract_robot_pose,
    extract_robot_vel,
)
from skill_refactor.settings import CFG

FINGER_ACTION_INDEX = 9
"""Action index where finger actions start."""


def rotate_vectors(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # v: (B, 2)
    # theta: (B,)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x = v[:, 0]
    y = v[:, 1]

    # Standard 2D rotation formula
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta

    return torch.stack((new_x, new_y), dim=1)


def get_normalize_action_range(
    env: BaseEnv | Env, control_mode: str  # pylint: disable=unused-argument
) -> tuple[bool, Tensor, Tensor]:
    """Get the action range for normalization in the ManiSkill environment.

    Args:
        env (ManiSkillVectorEnv): The ManiSkill environment instance.

    Returns:
        bool: Whether to normalize actions.
        Tensor: Lower bounds of the action space.
        Tensor: Upper bounds of the action space.
    """
    normalize_action = False

    if isinstance(env.unwrapped, BaseEnv):
        normalize_action = env.unwrapped.agent.controller.configs[
            "arm"
        ].normalize_action
        if normalize_action:
            arm_action_low = env.unwrapped.agent.controller.controllers[
                "arm"
            ].action_space_low
            base_action_low = env.unwrapped.agent.controller.controllers[
                "base"
            ].action_space_low
            arm_action_high = env.unwrapped.agent.controller.controllers[
                "arm"
            ].action_space_high
            base_action_high = env.unwrapped.agent.controller.controllers[
                "base"
            ].action_space_high
            arm_action_low = torch.cat([base_action_low, arm_action_low], dim=0)
            arm_action_high = torch.cat([base_action_high, arm_action_high], dim=0)
        else:
            arm_action_low = torch.tensor([-1.0] * 9, device=env.unwrapped.device)
            arm_action_high = torch.tensor([1.0] * 9, device=env.unwrapped.device)
    else:
        normalize_action = CFG.normalize_action
        assert isinstance(env, Env)
        assert isinstance(env.action_space, gym.spaces.Box)
        if normalize_action:
            assert hasattr(env, "action_low") and hasattr(
                env, "action_high"
            ), "Environment must have 'action_low' and 'action_high' attributes for normalization."
            arm_action_low = torch.tensor(
                env.action_low, dtype=torch.float64, device=CFG.device
            )
            arm_action_high = torch.tensor(
                env.action_high, dtype=torch.float64, device=CFG.device
            )
        else:
            arm_action_low = torch.tensor(
                env.action_space.low, dtype=torch.float64, device=CFG.device
            )
            arm_action_high = torch.tensor(
                env.action_space.high, dtype=torch.float64, device=CFG.device
            )
    return normalize_action, arm_action_low, arm_action_high


def get_frozen_action(
    skill_action: Tensor,
    arm_action_low: Tensor,
    arm_action_high: Tensor,
    normalize_action: bool,
    control_mode: str,
    obs: Optional[Tensor] = None,
) -> Tensor:
    """Get the frozen action for the skill based on the action range.

    Args:
        skill_action (Tensor): The action from the skill.
        arm_action_low (Tensor): Lower bounds of the action space.
        arm_action_high (Tensor): Upper bounds of the action space.
        normalize_action (bool): Whether to normalize actions.
        control_mode (str): The control mode of the environment.

    Returns:
        Tensor: The frozen action.
    """
    frozen_action = skill_action.clone()
    if control_mode == "pd_joint_delta_pos":
        static_actions = torch.zeros_like(skill_action)
        if CFG.delta_finger_control:
            if normalize_action:
                low = arm_action_low.unsqueeze(0).repeat(static_actions.shape[0], 1)
                high = arm_action_high.unsqueeze(0).repeat(static_actions.shape[0], 1)
                delta_qpos_norm = (static_actions - 0.5 * (low + high)) / (
                    0.5 * (high - low)
                )
                # Ensure dtype matches skill_action
                delta_qpos_norm = delta_qpos_norm.to(skill_action.dtype)
            else:
                delta_qpos_norm = static_actions.clone().to(skill_action.dtype)
        else:
            if normalize_action:
                low = arm_action_low.unsqueeze(0).repeat(static_actions.shape[0], 1)
                high = arm_action_high.unsqueeze(0).repeat(static_actions.shape[0], 1)
                delta_arm_qpos = (
                    static_actions[:, 3:FINGER_ACTION_INDEX]
                    - 0.5 * (low[:, 3:] + high[:, 3:])
                ) / (0.5 * (high[:, 3:] - low[:, 3:]))
                delta_base_qpos = (
                    static_actions[:, :3] - 0.5 * (low[:, :3] + high[:, :3])
                ) / (0.5 * (high[:, :3] - low[:, :3]))
            else:
                delta_arm_qpos = static_actions[:, :FINGER_ACTION_INDEX].clone()
                delta_base_qpos = static_actions[:, FINGER_ACTION_INDEX + 1 :].clone()
            delta_qpos_norm = torch.cat(
                [
                    delta_base_qpos,
                    delta_arm_qpos,
                    skill_action[:, FINGER_ACTION_INDEX : FINGER_ACTION_INDEX + 1],
                ],
                dim=-1,
            )
            # Ensure dtype matches skill_action
            delta_qpos_norm = delta_qpos_norm.to(skill_action.dtype)

        frozen_action = delta_qpos_norm.clone()
    else:
        assert control_mode == "force_torque"
        assert obs is not None, "obs must be provided for force_torque control."
        if obs.shape[0] == 0:
            assert skill_action.shape[0] == 0
            return frozen_action
        curr_vel = extract_robot_vel(obs)
        curr_pose = extract_robot_pose(obs)
        robot_mass_moment = extract_robot_mass_moment(obs)
        vel_trans_error = -curr_vel[:, :2]  # (B, 2)
        vel_ang_error = -curr_vel[:, 2]  # (B,)

        # PD control: compute desired accelerations
        # Desired linear acceleration (to reduce position error and velocity)
        desired_linear_accel = CFG.kv_pos_rl * vel_trans_error

        # Desired angular acceleration (to reduce heading error and angular velocity)
        desired_angular_accel = CFG.kv_ang_rl * vel_ang_error

        # NOTE: We assume the robot is either rotating in place or moving forward,
        # so the norm of the desired force is the forward force.
        desired_force = robot_mass_moment[:, 0:1].repeat(1, 2) * desired_linear_accel
        desired_force_rotated = rotate_vectors(desired_force, -curr_pose[:, 2])
        # assert desired_force_rotated[:, 0].abs().max() < 1e-3, "Rotate first!"
        desired_force_norm_x = desired_force_rotated[:, 0:1]
        desired_force_norm_y = desired_force_rotated[:, 1:]  # (B, 1) forward force only
        # Desired torque is Bx1
        desired_torque = (robot_mass_moment[:, 1] * desired_angular_accel).unsqueeze(-1)
        # Steering force (norm) is torque divided by steering length
        # Note that steering force is x-positive, which means negative rotation
        steering_force_value = -desired_torque / CFG.robot_steering_length

        # Stack into (B, 3) control tensor
        control = torch.cat(
            [desired_force_norm_y, desired_force_norm_x, steering_force_value], dim=-1
        )

        # Clamp the control to be within action range
        frozen_action = torch.clamp(
            control, min=arm_action_low.unsqueeze(0), max=arm_action_high.unsqueeze(0)
        )
        if normalize_action:
            low = arm_action_low.unsqueeze(0).repeat(frozen_action.shape[0], 1)
            high = arm_action_high.unsqueeze(0).repeat(frozen_action.shape[0], 1)
            frozen_action = (frozen_action - 0.5 * (low + high)) / (0.5 * (high - low))
            # Ensure dtype matches skill_action
        frozen_action = frozen_action.to(skill_action.dtype)

    return frozen_action


class PDBasePosController(PDJointPosController):
    """PDJointVelController for ego-centric base movement."""

    def _initialize_action_space(self):
        # At least support xy-plane translation and z-axis rotation
        assert len(self.joints) >= 3, len(self.joints)
        super()._initialize_action_space()

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                if not hasattr(self, "_target_qpos"):
                    self._target_qpos = self._start_qpos
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


class PDBasePosControllerConfig(PDJointPosControllerConfig):
    """Configuration for PDBasePosController."""

    controller_cls = PDBasePosController
