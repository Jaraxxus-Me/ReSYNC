"""Flipped-stacking planning components and hybrid skills.

The symbolic model is intentionally incomplete for predicate/operator
learning. It exposes only ``HandEmpty(robot)`` and ``On(block, block)``.
``Pick`` and ``Place`` form the task-planned suffix; ``MoveToFlip`` and the
pretrained-RL ``Flip`` have empty schemas and are invoked directly as a
physical prefix when one block starts overturned.

Until a learned ``Holding`` predicate is introduced, ``On(block, block)`` is
the observable intermediate token for a held block. Real stacking remains
``On(top, bottom)`` and is contact-gated by the environment.
"""

from __future__ import annotations

import abc
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import gymnasium as gym
import torch
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_inv,
    quat_mul,
    quat_rotate,
    subtract_frame_transforms,
    wrap_to_pi,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    PDDLDomain,
    Type,
    Variable,
)
from tensordict import TensorDict
from torch import Tensor

from resync_isaac.corner_stacking.corner_stacking import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
)
from resync_isaac.corner_stacking.utils import (
    CornerStackingObsLayout,
    object_pitch_180_target_quat,
)
from resync_isaac.flipped_stacking.flipped_stacking_env import (
    ACTION_SCALE,
    ScriptedBringToGoal,
    make_flipped_stacking_env,
)
from resync_isaac.flipped_stacking.task_cfg import (
    block_supported_by,
)
from resync_isaac.flipped_stacking.utils import (
    BOTTOM_CUBE,
    MID1_CUBE,
    MID2_CUBE,
    TOP_CUBE,
    FlipPolicyObservation,
    active_block_names,
    block_names,
    get_cube_pose_in_base,
    get_finger_joint_pos,
    top_down_wrist_goal,
)
from skill_refactor.benchmarks.base import (
    BaseRLTAMPSystem,
    GraphData,
    TensorPlanningComponents,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    ObjectContainer,
    Perceiver,
    Predicate,
    PredicateContainer,
    TypeContainer,
)
from skill_refactor.utils.task_planning import get_object_combinations


class FlippedStackingTypes(TypeContainer):
    """Robot and block types for flipped stacking."""

    def __init__(self) -> None:
        self.robot = Type("robot")
        self.block = Type("block")

    def as_set(self) -> set[Type]:
        return {self.robot, self.block}

    def as_dict(self) -> dict[str, Type]:
        return {"robot": self.robot, "block": self.block}


class FlippedStackingPredicates(PredicateContainer):
    """The intentionally incomplete predicate vocabulary to be learned."""

    def __init__(self, types: FlippedStackingTypes) -> None:
        handempty = Predicate("HandEmpty", [types.robot])
        on = Predicate("On", [types.block, types.block])
        self.predicates = {
            "HandEmpty": handempty,
            "On": on,
        }


class FlippedStackingObjects(ObjectContainer):
    """The robot and ordered two/four-block final stack."""

    def __init__(self, types: FlippedStackingTypes, num_of_blocks: int) -> None:
        self.robot = Object("robot", types.robot)
        self.blocks = tuple(
            Object(name, types.block) for name in block_names(num_of_blocks)
        )
        for block in self.blocks:
            setattr(self, block.name, block)

    def as_set(self) -> set[Object]:
        return {self.robot, *self.blocks}

    def as_dict(self) -> dict[str, Object]:
        return {"robot": self.robot, **{block.name: block for block in self.blocks}}

    @property
    def object_to_node(self) -> Dict[Object, int]:
        return {obj: idx for idx, obj in enumerate((self.robot, *self.blocks))}


class FlippedStackingPerceiver(Perceiver):
    """Interpret the two-predicate vocabulary and fixed stacking goal."""

    def __init__(
        self,
        predicates: FlippedStackingPredicates,
        types: FlippedStackingTypes,
        objects: FlippedStackingObjects,
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        self.predicates_container = predicates
        self._types = types
        self._objects = objects
        self._layout = obs_layout
        self.env: Any | None = None
        self.predicate_interpreters = {
            predicates["HandEmpty"]: self._interpret_handempty,
            predicates["On"]: self._interpret_on,
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> "FlippedStackingPerceiver":
        """Copy offline perception state without copying the live Isaac env."""
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for name, value in self.__dict__.items():
            setattr(
                copied,
                name,
                None if name == "env" else copy.deepcopy(value, memo),
            )
        return copied

    @property
    def objects(self) -> FlippedStackingObjects:
        return self._objects

    def reset(self, obs: Tensor, info=None):
        atoms = self._get_atoms(obs, info)
        goal = {
            GroundAtom(self.predicates_container["On"], (upper, lower))
            for upper, lower in zip(self._objects.blocks[:-1], self._objects.blocks[1:])
        }
        goal.add(
            GroundAtom(self.predicates_container["HandEmpty"], (self._objects.robot,))
        )
        return self._objects.as_set(), atoms[0], goal

    def step(self, obs: Tensor, info=None) -> List[set[GroundAtom]]:
        return self._get_atoms(obs, info)

    def _cube_pose(self, obs: Tensor, name: str) -> tuple[Tensor, Tensor]:
        if name == TOP_CUBE:
            return (
                self._layout.latest_insertive_pos_in_base(obs),
                self._layout.latest_insertive_quat_in_base(obs),
            )
        if name == BOTTOM_CUBE:
            return (
                self._layout.latest_receptive_pos_in_base(obs),
                self._layout.latest_receptive_quat_in_base(obs),
            )
        if name in (MID1_CUBE, MID2_CUBE):
            if self.env is None:
                raise RuntimeError(f"Live environment required for {name} pose")
            return get_cube_pose_in_base(self.env, name)
        raise ValueError(name)

    def _is_holding(self, obs: Tensor, name: str) -> Tensor:
        if name == TOP_CUBE:
            pos, _ = self._layout.latest_insertive_pose_in_ee(obs)
        elif name == BOTTOM_CUBE:
            pos, _ = self._layout.latest_receptive_pose_in_ee(obs)
        else:
            if self.env is None:
                # Historical training observations contain only top/bottom.
                return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
            cube_pos, cube_quat = self._cube_pose(obs, name)
            ee_pos, _ = self._layout.latest_ee_pose_in_base(obs)
            ee_quat = self._layout.latest_ee_quat_in_base(obs)
            pos, _ = subtract_frame_transforms(ee_pos, ee_quat, cube_pos, cube_quat)
        finger = self._layout.latest_finger_pos(obs)
        closed = (finger > float(CFG.fstack_closed_finger_min)) & (
            finger < float(CFG.fstack_closed_finger_max)
        )
        return (
            closed
            & (pos[:, :2].norm(dim=-1) < float(CFG.fstack_holding_xy_tol))
            & (pos[:, 2].abs() < float(CFG.fstack_holding_z_tol))
        )

    def _get_atoms(self, obs: Tensor, info=None) -> List[set[GroundAtom]]:
        result: List[set[GroundAtom]] = [set() for _ in range(obs.shape[0])]
        desired_predicates = list(self.predicates_container.predicates)
        if info is not None and "desired_predicates" in info:
            desired_predicates = [
                name
                for name in info["desired_predicates"]
                if name in self.predicates_container.predicates
            ]
        with torch.inference_mode():
            for pred, fn in self.predicate_interpreters.items():
                if pred.name not in desired_predicates:
                    continue
                combos: List[Sequence[Object]] = list(
                    get_object_combinations(
                        self._objects.as_set(),
                        pred.types,
                        # In the incomplete model, On(block, block) is the
                        # observable placeholder for holding that links Pick
                        # to Place without defining a Holding predicate.
                        skip_self=pred.name != "On",
                    )
                )
                values = fn(obs, combos)
                for batch in range(obs.shape[0]):
                    for idx, args in enumerate(combos):
                        if bool(values[batch, idx].item()):
                            result[batch].add(GroundAtom(pred, args))
        return result

    def _blank(self, obs: Tensor, objects) -> Tensor:
        return torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )

    def _interpret_handempty(self, obs: Tensor, objects) -> Tensor:
        out = self._blank(obs, objects)
        empty = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        for block in self._objects.blocks:
            empty &= ~self._is_holding(obs, block.name)
        for i, args in enumerate(objects):
            if args[0].name == "robot":
                out[:, i] = empty
        return out

    def _interpret_on(self, obs: Tensor, objects) -> Tensor:
        out = self._blank(obs, objects)
        for i, args in enumerate(objects):
            a, b = args[0].name, args[1].name
            block_set = {block.name for block in self._objects.blocks}
            if a == b and a in block_set:
                out[:, i] = self._is_holding(obs, a)
                continue
            if a not in block_set or b not in block_set:
                continue
            if self.env is None and (
                a in (MID1_CUBE, MID2_CUBE) or b in (MID1_CUBE, MID2_CUBE)
            ):
                # The saved planner dataset predates four-block observations.
                continue
            pa, qa = self._cube_pose(obs, a)
            pb, qb = self._cube_pose(obs, b)
            relative_pos, relative_quat = subtract_frame_transforms(pb, qb, pa, qa)
            relative_roll, relative_pitch, relative_yaw = euler_xyz_from_quat(
                relative_quat
            )
            out[:, i] = (
                (relative_pos[:, :2].norm(dim=-1) < float(CFG.fstack_on_xy_tol))
                & (
                    (relative_pos[:, 2] - float(CFG.fstack_on_z_offset)).abs()
                    < float(CFG.fstack_on_z_tol)
                )
                & (
                    wrap_to_pi(relative_roll).abs() + wrap_to_pi(relative_pitch).abs()
                    < float(CFG.fstack_upright_tol)
                )
                & (wrap_to_pi(relative_yaw).abs() < float(CFG.fstack_on_yaw_tol))
            )
        return out


class BaseFlippedStackingSkill(LiftedOperatorSkill):
    """Shared phase-controller structure matching corner stacking."""

    debug: bool = False

    def __init__(
        self,
        env: Any,
        operators: set[LiftedOperator],
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        super().__init__()
        self._env = env
        self.device = env.unwrapped.device
        self._layout = obs_layout
        self._all_operators = operators
        self._lifted_operator = next(
            op for op in operators if op.name == self.get_operator_name()
        )
        self._controller: Optional[ScriptedBringToGoal] = None
        self._phase: Any = 0
        self._step_idx = 0
        self._phase_step_idx = 0
        self._gripper_timer = 0
        self._last_done: Optional[Tensor] = None

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        self._controller = None
        self._phase = 0
        self._step_idx = 0
        self._phase_step_idx = 0
        self._gripper_timer = 0
        self._last_done = None
        return super().reset(ground_operator, obs)

    def get_lifted_operator(self) -> LiftedOperator:
        return self._lifted_operator

    @abc.abstractmethod
    def get_operator_name(self) -> str:
        raise NotImplementedError

    def _build_controller(self, num_envs: int) -> ScriptedBringToGoal:
        return ScriptedBringToGoal(ACTION_SCALE, num_envs, self.device, env=self._env)

    def _ee_pose_b(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        ee_pos, _ = self._layout.latest_ee_pose_in_base(obs)
        return ee_pos, self._layout.latest_ee_quat_in_base(obs)

    @staticmethod
    def _compose_action(arm: Tensor, gripper: float) -> Tensor:
        return torch.cat(
            (arm, torch.full((arm.shape[0], 1), gripper, device=arm.device)), dim=-1
        )

    def _phase_timed_out(self) -> bool:
        max_steps = int(CFG.fstack_scripted_max_phase_steps)
        if len(active_block_names(self._env)) >= 3:
            max_steps *= 2
        return self._phase_step_idx >= max_steps

    def _update_phase_counter(self, previous_phase: int) -> None:
        self._phase_step_idx = (
            0 if self._phase != previous_phase else self._phase_step_idx + 1
        )

    def terminate_with_objects(self, objects, obs: Tensor) -> Tensor:
        del objects
        if self._last_done is None:
            return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        return self._last_done.clone().detach()


class _ScriptedReachSkill(BaseFlippedStackingSkill, abc.ABC):
    """Scripted skill marker matching corner stacking's public structure."""


class MoveToFlipSkill(_ScriptedReachSkill):
    """Move the gripper above the block before invoking its flip policy."""

    def get_operator_name(self) -> str:
        return "MoveToFlip"

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        super().reset(ground_operator, obs)
        self._controller = self._build_controller(obs.shape[0])
        self._last_done = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)

    def get_action_given_objects(self, objects, obs: Tensor) -> Tensor:
        assert self._controller is not None
        cube_name = objects[1].name
        cube_pos, _ = get_cube_pose_in_base(self._env, cube_name)
        goal, goal_quat = top_down_wrist_goal(
            self._env, cube_pos, float(CFG.fstack_move_pregrasp_height)
        )
        self._controller.set_goal(goal, goal_quat)
        ee_pos, ee_quat = self._ee_pose_b(obs)
        arm, pos_err, rot_err = self._controller.compute(ee_pos, ee_quat)
        self._step_idx += 1
        self._last_done = (pos_err < float(CFG.fstack_scripted_pos_tol)) & (
            rot_err < float(CFG.fstack_scripted_rot_tol)
        )
        if self._step_idx >= int(CFG.fstack_scripted_max_phase_steps):
            self._last_done[:] = True
        return self._compose_action(arm, GRIPPER_OPEN)


class FlipSkill(BaseFlippedStackingSkill):
    """Recover an overturned block with the pretrained policy and retreat."""

    RL, OPEN, RETREAT, DONE = range(4)

    def __init__(self, env, operators, layout) -> None:
        self._policy: Callable[[Any], Tensor] | None = None
        self._adapter = FlipPolicyObservation(env, layout)
        self._phase: Tensor | None = None
        super().__init__(env, operators, layout)

    def get_operator_name(self) -> str:
        return "Flip"

    def set_policy(self, policy: Callable[[Any], Tensor]) -> None:
        self._policy = policy

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        super().reset(ground_operator, obs)
        cube_name = ground_operator.parameters[1].name
        self._adapter.reset(cube_name, obs)
        self._phase = torch.full(
            (obs.shape[0],), self.RL, dtype=torch.int8, device=obs.device
        )

    def get_action_given_objects(self, objects, obs: Tensor) -> Tensor:
        assert self._phase is not None and self._policy is not None
        n = obs.shape[0]
        assembled = self._adapter.assembled()
        self._phase[(self._phase == self.RL) & assembled] = self.OPEN
        rl_timed_out = (self._phase == self.RL) & (
            self._step_idx >= int(CFG.fstack_flip_max_rl_steps)
        )
        self._phase[rl_timed_out] = self.OPEN
        opened = get_finger_joint_pos(self._env) < float(
            CFG.fstack_flip_open_finger_max
        )
        self._phase[(self._phase == self.OPEN) & opened] = self.RETREAT
        action = torch.zeros((n, 7), device=obs.device)
        rl = self._phase == self.RL
        if bool(rl.any().item()):
            policy_obs = self._adapter.build(obs)
            td = TensorDict({"policy": policy_obs}, batch_size=[n])
            with torch.inference_mode():
                policy_action = self._policy(td).clone().detach()
            action[rl] = policy_action[rl]
        opened_phase = self._phase == self.OPEN
        action[opened_phase, -1] = GRIPPER_OPEN
        retreat = self._phase == self.RETREAT
        if bool(retreat.any().item()):
            if self._controller is None:
                self._controller = self._build_controller(n)
                assert self._adapter.region_pos_b is not None
                goal, goal_quat = top_down_wrist_goal(
                    self._env,
                    self._adapter.region_pos_b,
                    float(CFG.fstack_flip_retreat_height),
                )
                self._controller.set_goal(goal, goal_quat)
            ee_pos, ee_quat = self._ee_pose_b(obs)
            arm, pos_err, rot_err = self._controller.compute(ee_pos, ee_quat)
            action[retreat, :6] = arm[retreat]
            action[retreat, -1] = GRIPPER_OPEN
            reached = (
                retreat
                & (pos_err < float(CFG.fstack_scripted_pos_tol))
                & (rot_err < float(CFG.fstack_scripted_rot_tol))
            )
            self._phase[reached] = self.DONE
        self._step_idx += 1
        self._last_done = self._phase == self.DONE
        return action


class PickSkill(_ScriptedReachSkill):
    """Scripted Pick: pre-grasp, grasp, close, then lift."""

    _PRE_GRASP, _GRASP, _CLOSE, _LIFT, _DONE = range(5)

    def __init__(self, env, operators, layout) -> None:
        self._target_quat_b: Tensor | None = None
        self._initial_block_pos_b: Tensor | None = None
        self._retry_count = 0
        super().__init__(env, operators, layout)

    def get_operator_name(self) -> str:
        return "Pick"

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        super().reset(ground_operator, obs)
        self._retry_count = 0
        block_name = ground_operator.parameters[1].name
        block_pos, block_quat = get_cube_pose_in_base(self._env, block_name)
        self._initial_block_pos_b = block_pos.detach().clone()
        self._target_quat_b = object_pitch_180_target_quat(block_quat).detach().clone()
        self._controller = self._build_controller(obs.shape[0])

    def get_action_given_objects(self, objects, obs: Tensor) -> Tensor:
        assert self._controller is not None
        assert self._target_quat_b is not None and self._initial_block_pos_b is not None
        block_pos, _ = get_cube_pose_in_base(self._env, objects[1].name)
        if self._phase == self._PRE_GRASP:
            goal, gripper = block_pos.clone(), GRIPPER_OPEN
            goal[:, 2] += float(CFG.fstack_pick_pregrasp_dz)
        elif self._phase in (self._GRASP, self._CLOSE):
            goal = block_pos.clone()
            retry_offsets = tuple(CFG.fstack_pick_retry_offsets)
            assert self._retry_count < len(retry_offsets)
            dx, dy, dz = retry_offsets[self._retry_count]
            goal[:, 0] += float(dx)
            goal[:, 1] += float(dy)
            goal[:, 2] += float(CFG.fstack_pick_grasp_dz) + float(dz)
            gripper = GRIPPER_OPEN if self._phase == self._GRASP else GRIPPER_CLOSE
        else:
            goal, gripper = self._initial_block_pos_b.clone(), GRIPPER_CLOSE
            goal[:, 2] += float(CFG.fstack_pick_lift_dz)
        self._controller.set_goal(goal, self._target_quat_b)
        ee_pos, ee_quat = self._ee_pose_b(obs)
        arm, pe, _ = self._controller.compute(ee_pos, ee_quat)
        pos_ok = bool((pe < float(CFG.fstack_scripted_pos_tol)).all().item())
        grasp_pos_error = goal - ee_pos
        grasp_ready = bool(
            (
                (
                    grasp_pos_error[:, :2].norm(dim=-1)
                    < float(CFG.fstack_pick_grasp_xy_tol)
                )
                & (grasp_pos_error[:, 2].abs() < float(CFG.fstack_pick_grasp_z_tol))
            )
            .all()
            .item()
        )
        _, _, ee_yaw = euler_xyz_from_quat(ee_quat)
        _, _, goal_yaw = euler_xyz_from_quat(self._target_quat_b)
        wrist_yaw_ok = bool(
            (wrap_to_pi(ee_yaw - goal_yaw).abs() < float(CFG.fstack_on_yaw_tol))
            .all()
            .item()
        )
        timeout = self._phase_timed_out()
        lifted = bool(
            (
                block_pos[:, 2]
                >= self._initial_block_pos_b[:, 2]
                + float(CFG.fstack_pick_lift_clearance)
            )
            .all()
            .item()
        )
        previous_phase = self._phase
        if self._phase == self._PRE_GRASP and ((pos_ok and wrist_yaw_ok) or timeout):
            self._phase = self._GRASP
        elif self._phase == self._GRASP and (grasp_ready or timeout):
            self._phase = self._CLOSE
        elif self._phase == self._CLOSE:
            self._gripper_timer += 1
            if self._gripper_timer >= int(CFG.fstack_gripper_dwell_steps) or timeout:
                self._phase = self._LIFT
        elif self._phase == self._LIFT:
            if lifted:
                self._phase = self._DONE
            elif self._phase_step_idx >= int(
                CFG.fstack_pick_retry_lift_steps
            ) and self._retry_count < int(CFG.fstack_pick_max_retries):
                self._retry_count += 1
                self._gripper_timer = 0
                self._phase = self._PRE_GRASP
            elif timeout:
                self._phase = self._DONE
        self._update_phase_counter(previous_phase)
        self._last_done = torch.full(
            (obs.shape[0],),
            self._phase == self._DONE,
            dtype=torch.bool,
            device=obs.device,
        )
        return self._compose_action(arm, gripper)


class PlaceSkill(_ScriptedReachSkill):
    """Scripted Place: pre-place, descend, open, then retract."""

    _LIFT_CLEAR, _PRE_PLACE, _PLACE, _OPEN, _RETRACT, _DONE = range(6)

    def __init__(self, env, operators, layout) -> None:
        self._target_quat_b: Tensor | None = None
        self._grasp_wrist_offset_b: Tensor | None = None
        self._lift_clear_xy_b: Tensor | None = None
        self._lift_start_z_b: Tensor | None = None
        super().__init__(env, operators, layout)

    def get_operator_name(self) -> str:
        return "Place"

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        super().reset(ground_operator, obs)
        grasp_name = ground_operator.parameters[1].name
        target_name = ground_operator.parameters[2].name
        grasp_pos, _ = get_cube_pose_in_base(self._env, grasp_name)
        _, target_quat = get_cube_pose_in_base(self._env, target_name)
        ee_pos, _ = self._ee_pose_b(obs)
        self._grasp_wrist_offset_b = (ee_pos - grasp_pos).detach().clone()
        self._grasp_wrist_offset_b[:, 2].clamp_min_(0.0)
        self._lift_clear_xy_b = ee_pos[:, :2].detach().clone()
        self._lift_start_z_b = ee_pos[:, 2].detach().clone()
        self._target_quat_b = object_pitch_180_target_quat(target_quat).detach().clone()
        self._controller = self._build_controller(obs.shape[0])

    def get_action_given_objects(self, objects, obs: Tensor) -> Tensor:
        assert self._controller is not None and self._target_quat_b is not None
        assert self._grasp_wrist_offset_b is not None
        assert self._lift_clear_xy_b is not None
        assert self._lift_start_z_b is not None
        grasp_pos, grasp_quat = get_cube_pose_in_base(self._env, objects[1].name)
        target_pos, target_quat = get_cube_pose_in_base(self._env, objects[2].name)
        collision_free_pre_dz = (
            self._grasp_wrist_offset_b[:, 2]
            + float(CFG.fstack_on_z_offset)
            + float(CFG.fstack_place_pre_clearance)
        )
        preplace_dz = torch.maximum(
            collision_free_pre_dz,
            torch.full_like(collision_free_pre_dz, float(CFG.fstack_place_pre_dz)),
        )
        safe_ee_z = torch.maximum(
            target_pos[:, 2] + preplace_dz,
            torch.full_like(preplace_dz, float(CFG.fstack_place_fixed_safe_ee_z)),
        )
        safe_ee_z = torch.maximum(safe_ee_z, self._lift_start_z_b)
        collision_free_place_dz = (
            self._grasp_wrist_offset_b[:, 2]
            + float(CFG.fstack_on_z_offset)
            + float(CFG.fstack_place_release_clearance)
        )
        place_dz = torch.maximum(
            collision_free_place_dz,
            torch.full_like(collision_free_place_dz, float(CFG.fstack_place_dz)),
        )
        if self._phase == self._LIFT_CLEAR:
            goal, gripper = target_pos.clone(), GRIPPER_CLOSE
            goal[:, :2] = self._lift_clear_xy_b
            goal[:, 2] = safe_ee_z
        elif self._phase == self._PRE_PLACE:
            goal, gripper = target_pos.clone(), GRIPPER_CLOSE
            goal[:, :2] += self._grasp_wrist_offset_b[:, :2]
            goal[:, 2] = safe_ee_z
        elif self._phase in (self._PLACE, self._OPEN):
            goal = target_pos.clone()
            goal[:, :2] += self._grasp_wrist_offset_b[:, :2]
            goal[:, 2] += place_dz
            gripper = GRIPPER_CLOSE if self._phase == self._PLACE else GRIPPER_OPEN
        else:
            goal, gripper = target_pos.clone(), GRIPPER_OPEN
            goal[:, :2] += self._grasp_wrist_offset_b[:, :2]
            goal[:, 2] = safe_ee_z

        self._controller.set_goal(goal, self._target_quat_b)
        ee_pos, ee_quat = self._ee_pose_b(obs)
        arm, pos_error, _ = self._controller.compute(ee_pos, ee_quat)
        near_tower = (ee_pos[:, :2] - target_pos[:, :2]).norm(dim=-1) < float(
            CFG.fstack_place_near_tower_xy_dist
        )
        near_tower_scale = float(CFG.fstack_place_near_tower_motion_scale)
        assert 0.0 < near_tower_scale <= 1.0
        arm = torch.where(
            near_tower[:, None],
            arm * near_tower_scale,
            arm,
        )
        pos_ok = bool((pos_error < float(CFG.fstack_scripted_pos_tol)).all().item())
        timeout = self._phase_step_idx >= int(CFG.fstack_place_max_phase_steps)
        cube_delta = grasp_pos - target_pos
        preplace_ready = bool(
            (
                (cube_delta[:, :2].norm(dim=-1) < float(CFG.fstack_place_xy_tol))
                & (
                    cube_delta[:, 2]
                    > float(CFG.fstack_on_z_offset)
                    + float(CFG.fstack_place_pre_clearance)
                )
            )
            .all()
            .item()
        )
        lift_clear_ready = bool(
            (
                cube_delta[:, 2]
                > float(CFG.fstack_on_z_offset) + float(CFG.fstack_place_pre_clearance)
            )
            .all()
            .item()
        )
        grasp_roll, grasp_pitch, grasp_yaw = euler_xyz_from_quat(grasp_quat)
        target_roll, target_pitch, target_yaw = euler_xyz_from_quat(target_quat)
        preplace_orientation_ready = bool(
            (
                (
                    wrap_to_pi(grasp_roll).abs() + wrap_to_pi(grasp_pitch).abs()
                    < float(CFG.fstack_upright_tol)
                )
                & (
                    wrap_to_pi(grasp_yaw - target_yaw).abs()
                    < float(CFG.fstack_place_pre_yaw_tol)
                )
            )
            .all()
            .item()
        )
        release_height_error = (
            cube_delta[:, 2]
            - float(CFG.fstack_on_z_offset)
            - float(CFG.fstack_place_release_clearance)
        ).abs()
        contact_height_error = (cube_delta[:, 2] - float(CFG.fstack_on_z_offset)).abs()
        placed_geometry = bool(
            (
                (cube_delta[:, :2].norm(dim=-1) < float(CFG.fstack_place_xy_tol))
                & (
                    (release_height_error < float(CFG.fstack_place_z_tol))
                    | (contact_height_error < float(CFG.fstack_place_z_tol))
                )
                & (
                    wrap_to_pi(grasp_roll).abs() + wrap_to_pi(grasp_pitch).abs()
                    < float(CFG.fstack_upright_tol)
                )
                & (
                    wrap_to_pi(target_roll).abs() + wrap_to_pi(target_pitch).abs()
                    < float(CFG.fstack_upright_tol)
                )
                & (
                    wrap_to_pi(grasp_yaw - target_yaw).abs()
                    < float(CFG.fstack_place_yaw_tol)
                )
            )
            .all()
            .item()
        )
        contact_ok, _, _ = block_supported_by(
            self._env, objects[1].name, objects[2].name
        )
        released_on_target = bool(
            (
                contact_ok
                & (
                    get_finger_joint_pos(self._env)
                    < float(CFG.fstack_flip_open_finger_max)
                )
            )
            .all()
            .item()
        )

        previous_phase = self._phase
        if self._phase == self._LIFT_CLEAR and (
            (pos_ok and lift_clear_ready) or timeout
        ):
            self._phase = self._PRE_PLACE
        elif self._phase == self._PRE_PLACE and (
            (preplace_ready and preplace_orientation_ready) or timeout
        ):
            self._phase = self._PLACE
        elif self._phase == self._PLACE and (placed_geometry or timeout):
            self._phase = self._OPEN
        elif self._phase == self._OPEN:
            self._gripper_timer += 1
            if (
                released_on_target
                and self._gripper_timer >= int(CFG.fstack_place_open_dwell_steps)
            ) or timeout:
                self._phase = self._RETRACT
        elif self._phase == self._RETRACT and (pos_ok or timeout):
            self._phase = self._DONE
        self._update_phase_counter(previous_phase)
        self._last_done = torch.full(
            (obs.shape[0],),
            self._phase == self._DONE,
            dtype=torch.bool,
            device=obs.device,
        )
        return self._compose_action(arm, gripper)


class FlippedStackingRLTAMPSystem(BaseRLTAMPSystem):
    """Combine the flipped stacking simulator with symbolic planning skills."""

    @property
    def perceiver(self) -> FlippedStackingPerceiver:
        return cast(FlippedStackingPerceiver, self.components.perceiver)

    def __init__(self, components, seed=None, render_mode=None, env_cfg=None) -> None:
        self.render_mode = render_mode
        self.env_cfg = env_cfg
        super().__init__(components, "FlippedStackingRLTAMPSystem", seed, render_mode)

    def _create_env(self) -> gym.Env:
        return make_flipped_stacking_env(
            CFG.fstack_num_envs,
            CFG.fstack_max_steps,
            env_cfg=self.env_cfg,
            render_mode=self.render_mode,
        )

    def _get_domain_name(self) -> str:
        return "FlippedStacking-domain"

    def get_domain(self) -> PDDLDomain:
        return PDDLDomain(
            self._get_domain_name(),
            self.operators,
            self.predicates,
            self.components.type_container.as_set(),
        )

    @classmethod
    def _create_planning_components(cls) -> TensorPlanningComponents:
        types_container = FlippedStackingTypes()
        predicates = FlippedStackingPredicates(types_container)
        objects = FlippedStackingObjects(types_container, int(CFG.num_of_blocks))
        HandEmpty = predicates["HandEmpty"]
        On = predicates["On"]

        robot = Variable("?robot", types_container.robot)
        grasp_block = Variable("?grasp_block", types_container.block)
        target_block = Variable("?target_block", types_container.block)
        flip_block = Variable("?flip_block", types_container.block)

        # With the deliberately incomplete two-predicate vocabulary, the
        # self-relation On(block, block) is the symbolic intermediate token
        # linking Pick to Place; the perceiver emits it only while held.
        picked = LiftedAtom(On, (grasp_block, grasp_block))
        operators = {
            LiftedOperator(
                "MoveToFlip",
                [robot, flip_block],
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
            ),
            LiftedOperator(
                "Flip",
                [robot, flip_block],
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
            ),
            LiftedOperator(
                "Pick",
                [robot, grasp_block],
                preconditions={LiftedAtom(HandEmpty, (robot,))},
                add_effects={picked},
                delete_effects={LiftedAtom(HandEmpty, (robot,))},
            ),
            LiftedOperator(
                "Place",
                [robot, grasp_block, target_block],
                preconditions={picked},
                add_effects={
                    LiftedAtom(On, (grasp_block, target_block)),
                    LiftedAtom(HandEmpty, (robot,)),
                },
                delete_effects={picked},
            ),
        }
        perceiver = FlippedStackingPerceiver(
            predicates,
            types_container,
            objects,
            obs_layout=None,  # type: ignore[arg-type]
        )
        return TensorPlanningComponents(
            type_container=types_container,
            predicate_container=predicates,
            operators=operators,
            skills=set(),
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
        env_cfg: Any = None,
    ) -> FlippedStackingRLTAMPSystem:
        components = cls._create_planning_components()
        system = cls(components, seed, render_mode, env_cfg)
        layout = CornerStackingObsLayout.from_env(system.env, has_outside=False)
        system.perceiver._layout = layout  # type: ignore[attr-defined]
        system.perceiver.env = system.env
        system.components.skills.update(
            {
                MoveToFlipSkill(system.env, system.operators, layout),
                FlipSkill(system.env, system.operators, layout),
                PickSkill(system.env, system.operators, layout),
                PlaceSkill(system.env, system.operators, layout),
            }
        )
        return system

    @property
    def flip_skill(self) -> FlipSkill:
        return next(skill for skill in self.skills if isinstance(skill, FlipSkill))

    @property
    def move_to_flip_skill(self) -> MoveToFlipSkill:
        return next(
            skill for skill in self.skills if isinstance(skill, MoveToFlipSkill)
        )

    @property
    def pick_skill(self) -> PickSkill:
        return next(skill for skill in self.skills if isinstance(skill, PickSkill))

    @property
    def place_skill(self) -> PlaceSkill:
        return next(skill for skill in self.skills if isinstance(skill, PlaceSkill))

    def state_to_graph(self, state: Tensor, training: bool = False) -> List[GraphData]:
        """Convert policy observations to robot plus two/four-block pose graphs."""
        del training
        obs = state
        batch_size = obs.shape[0]
        object_to_node = self.perceiver.objects.object_to_node
        num_nodes = len(object_to_node)
        node_features = torch.zeros(
            batch_size,
            num_nodes,
            7,
            dtype=obs.dtype,
            device=obs.device,
        )

        assert isinstance(self.perceiver, FlippedStackingPerceiver)
        layout = self.perceiver._layout  # pylint: disable=protected-access
        ee_pos_b, _ = layout.latest_ee_pose_in_base(obs)
        pose_by_name: dict[str, tuple[Tensor, Tensor]] = {
            "robot": (ee_pos_b, layout.latest_ee_quat_in_base(obs)),
            TOP_CUBE: (
                layout.latest_insertive_pos_in_base(obs),
                layout.latest_insertive_quat_in_base(obs),
            ),
            BOTTOM_CUBE: (
                layout.latest_receptive_pos_in_base(obs),
                layout.latest_receptive_quat_in_base(obs),
            ),
        }
        for name in (MID1_CUBE, MID2_CUBE):
            if name in self.perceiver.objects.as_dict():
                pose_by_name[name] = get_cube_pose_in_base(self.env, name)
        for obj, node_idx in object_to_node.items():
            pos_b, quat_b = pose_by_name[obj.name]
            node_features[:, node_idx] = torch.cat((pos_b, quat_b), dim=-1)

        edge_list: list[list[int]] = []
        edge_features_list: list[Tensor] = []
        for source_idx in range(num_nodes):
            for target_idx in range(num_nodes):
                if source_idx == target_idx:
                    continue
                edge_list.append([source_idx, target_idx])
                source_pos = node_features[:, source_idx, :3]
                source_quat = node_features[:, source_idx, 3:]
                target_pos = node_features[:, target_idx, :3]
                target_quat = node_features[:, target_idx, 3:]
                source_quat_inv = quat_inv(source_quat)
                relative_pos = quat_rotate(source_quat_inv, target_pos - source_pos)
                relative_pos[:, 2] = 0.0
                relative_quat = quat_mul(source_quat_inv, target_quat)
                edge_features_list.append(
                    torch.cat((relative_pos, relative_quat), dim=-1)
                )

        edge_indices = torch.tensor(edge_list, dtype=torch.long, device=obs.device).T
        edge_features = torch.stack(edge_features_list, dim=1)
        return [
            GraphData(
                node_features=node_features[batch_idx],
                edge_features=edge_features[batch_idx],
                edge_indices=edge_indices,
                global_features=None,
                object_to_node=object_to_node.copy(),
            )
            for batch_idx in range(batch_size)
        ]

    def state_to_vec(self, state: Tensor, objects: Sequence[Object]) -> Tensor:
        raise NotImplementedError

    def skill_obs_space(self) -> gym.Space:
        return self.env.observation_space

    def full_state_to_relative_state(
        self, full_state: Tensor, relative_name: str = ""
    ) -> Tensor:
        raise NotImplementedError

    def relative_state_to_full_state(
        self, inital_full_state: Tensor, relative_state: Tensor
    ) -> Tensor:
        raise NotImplementedError


__all__ = [
    "BaseFlippedStackingSkill",
    "FlipSkill",
    "FlippedStackingObjects",
    "FlippedStackingPerceiver",
    "FlippedStackingRLTAMPSystem",
    "MoveToFlipSkill",
    "PickSkill",
    "PlaceSkill",
    "_ScriptedReachSkill",
]
