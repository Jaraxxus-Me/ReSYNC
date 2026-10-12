"""CornerStacking (Isaac Sim) environment planning components.

IMPORTANT: Importing this module transitively imports ``corner_stacking_env``,
which requires Isaac Lab's ``AppLauncher`` to have been invoked first. Tests
must launch the simulator before importing anything from this file.

Runtime design
--------------
The scene has three entities — two blocks and a box — plus the UR5e +
Robotiq 2F-140 robot:

* ``block0`` is the *insertive* block (the one that starts in the box at the
  USD assembled offset when the upstream reset dataset is used).
* ``block1`` is the *outside* block (spawned near the box by
  ``CorneredStackingEventCfg``).
* ``box`` is the *receptive* object that ``block0`` sits inside of.

Three skills correspond to three operators:

* ``Pick(robot, block)`` — scripted: pre-grasp, grasp, lift.
* ``Place(robot, grasp_block, target_block)`` — scripted: lift, pre-place,
  place.
* ``RL(robot, block, box)`` — runs the pretrained block_box OSC policy until the
  insertive block is assembled in the box. Preconditions and effects are
  intentionally empty so the operator is a no-op for task planning; the
  skill is invoked directly from the hybrid controller.

The perceiver and scripted skills consume only the policy observation
tensor — they do not touch ``env`` during ``step``. Layout discovery
(joint count + Robotiq drive-joint index + presence of the outside-block
obs slice) happens once at construction via
:meth:`CornerStackingObsLayout.from_env`.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import torch
from isaaclab.utils.math import compute_pose_error, quat_inv, quat_mul, quat_rotate
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    PDDLDomain,
    Type,
    Variable,
)
from torch import Tensor

from resync_isaac.corner_stacking.corner_stacking_env import (
    ACTION_SCALE,
    ScriptedBringToGoal,
    make_corner_stacking_env,
)
from resync_isaac.corner_stacking.cornered_stacking_task_cfg import (
    BLOCK_BOX_POLICY_OBS_DIM,
    CORNERED_STACKING_TASK_ID,
)
from resync_isaac.corner_stacking.utils import (
    CornerStackingObsLayout,
    InsertiveAssembledChecker,
    get_ee_pose_in_base,
    get_finger_joint_pos,
    get_receptive_object_pose_in_base,
    object_pitch_180_target_quat,
    top_down_wrist_target_quat,
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

TOP_DOWN_QUAT = torch.tensor([0.7071, 0.0, 0.7071, 0.0])  # wxyz


# ---------------------------------------------------------------------------
# Types / predicates / objects
# ---------------------------------------------------------------------------


class CornerStackingTypes(TypeContainer):
    """Container for CornerStacking types."""

    def __init__(self) -> None:
        self.robot = Type("robot")
        self.block = Type("block")
        self.box = Type("box")

    def as_set(self) -> set[Type]:
        return {self.robot, self.block, self.box}

    def as_dict(self) -> dict[str, Type]:
        return {"robot": self.robot, "block": self.block, "box": self.box}


class CornerStackingPredicates(PredicateContainer):
    """Container for CornerStacking predicates.

    Three obs-only predicates:
    * ``HandEmpty(robot)`` — gripper is open and not holding a block.
    * ``Holding(robot, block)`` — gripper closed and block is at the EE.
    * ``On(block, block)`` — first block rests on the second block.
    """

    def __init__(self, types: CornerStackingTypes) -> None:
        handempty = Predicate("HandEmpty", [types.robot])
        holding = Predicate("Holding", [types.robot, types.block])
        on = Predicate("On", [types.block, types.block])
        self.predicates = {
            "HandEmpty": handempty,
            "Holding": holding,
            "On": on,
        }


class CornerStackingObjects(ObjectContainer):
    """Single object container (no scenarios).

    ``block0`` maps to the insertive_object scene entity (the in-box block);
    ``block1`` maps to the outside_block; ``box`` maps to the
    receptive_object.
    """

    def __init__(self, types: CornerStackingTypes) -> None:
        self.robot = Object("robot", types.robot)
        self.block0 = Object("block0", types.block)
        self.block1 = Object("block1", types.block)
        self.box = Object("box", types.box)

    def as_set(self) -> set[Object]:
        return {self.robot, self.block0, self.block1, self.box}

    def as_dict(self) -> dict[str, Object]:
        return {
            "robot": self.robot,
            "block0": self.block0,
            "block1": self.block1,
            "box": self.box,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        return {self.robot: 0, self.block0: 1, self.block1: 2, self.box: 3}


# Module-level singleton — the set of objects never changes in this env.
_TYPES = CornerStackingTypes()
_OBJECTS = CornerStackingObjects(_TYPES)


# ---------------------------------------------------------------------------
# Perceiver (obs-only)
# ---------------------------------------------------------------------------


class CornerStackingPerceiver(Perceiver):
    """Obs-only perceiver.

    All predicate interpreters operate on the policy observation tensor
    through :class:`CornerStackingObsLayout`. The layout is resolved once
    from the live env at construction time (see
    :meth:`CornerStackingRLTAMPSystem.create_default`), so evaluation at
    runtime does not touch the env.
    """

    def __init__(
        self,
        predicates: CornerStackingPredicates,
        types: CornerStackingTypes,
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        self.predicates_container = predicates
        self._types = types
        self._layout = obs_layout
        self.predicate_interpreters = {
            predicates["HandEmpty"]: self._interpret_handempty,
            predicates["Holding"]: self._interpret_holding,
            predicates["On"]: self._interpret_on,
        }

    @property
    def objects(self) -> ObjectContainer:
        return _OBJECTS

    def reset(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        assert (
            obs.shape[0] == 1
        ), "Expected batch size of 1 for observation for perceiver reset."
        atoms = self._get_atoms(obs, info)
        On = self.predicates_container["On"]
        goal_0on1 = bool(CFG.c_stack_goal_0on1)
        if info is not None:
            scenario_name = info.get("train_scenario_idx", info.get("scenario_name"))
            if isinstance(scenario_name, str):
                if "goal_0on1" in scenario_name:
                    goal_0on1 = True
                elif "goal_1on0" in scenario_name:
                    goal_0on1 = False
            if "c_stack_goal_0on1" in info:
                goal_0on1 = bool(info["c_stack_goal_0on1"])
        if goal_0on1:
            goal = {GroundAtom(On, (_OBJECTS.block0, _OBJECTS.block1))}
        else:
            goal = {GroundAtom(On, (_OBJECTS.block1, _OBJECTS.block0))}
        return _OBJECTS.as_set(), atoms[0], goal

    def step(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        return self._get_atoms(obs, info)

    def _get_atoms(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        B = obs.shape[0]
        atoms_list: List[set[GroundAtom]] = [set() for _ in range(B)]

        desired_predicates = list(self.predicates_container.predicates.keys())
        if info is not None and "desired_predicates" in info:
            desired_predicates = [
                p
                for p in info["desired_predicates"]
                if p in self.predicates_container.predicates
            ]

        with torch.inference_mode():
            for predicate, interpret_fn in self.predicate_interpreters.items():
                if predicate.name not in desired_predicates:
                    continue
                input_args: List[Sequence[Object]] = []
                for args in get_object_combinations(
                    _OBJECTS.as_set(),
                    predicate.types,
                    skip_self=True,
                ):
                    input_args.append(args)
                stacked_res = interpret_fn(obs, input_args)
                assert stacked_res.shape[0] == B
                assert stacked_res.shape[1] == len(input_args)
                for b in range(B):
                    for n in range(len(input_args)):
                        if bool(stacked_res[b, n].item()):
                            atoms_list[b].add(GroundAtom(predicate, input_args[n]))
        return atoms_list

    # ------------------------------------------------------------------
    # Helpers — obs-only.
    # ------------------------------------------------------------------

    def _is_gripper_open(self, obs: Tensor) -> Tensor:
        finger = self._layout.latest_finger_pos(obs)
        return finger < CFG.c_stack_gripper_closed_min

    def _is_gripper_closed(self, obs: Tensor) -> Tensor:
        finger = self._layout.latest_finger_pos(obs)
        return (finger > CFG.c_stack_gripper_closed_min) & (
            finger < CFG.c_stack_gripper_closed_max
        )

    def _is_holding(self, obs: Tensor, block_name: str) -> Tensor:
        """Gripper closed AND the named block is at the EE."""
        if block_name == "block0":
            pos_in_ee, _ = self._layout.latest_insertive_pose_in_ee(obs)
        elif block_name == "block1":
            pos_in_ee, _ = self._layout.latest_outside_pose_in_ee(obs)
        else:
            raise ValueError(f"Unknown block: {block_name}")
        near_ee = pos_in_ee[:, :2].norm(dim=-1) < CFG.c_stack_holding_ee_dist_xy
        near_ee_z = pos_in_ee[:, 2] < CFG.c_stack_holding_ee_dist_z
        return self._is_gripper_closed(obs) & near_ee & near_ee_z

    def _is_holding_any(self, obs: Tensor) -> Tensor:
        return self._is_holding(obs, "block0") | self._is_holding(obs, "block1")

    # ------------------------------------------------------------------
    # Interpreters
    # ------------------------------------------------------------------

    def _interpret_handempty(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        empty = ~self._is_holding_any(obs)
        out = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for i, args in enumerate(objects):
            if len(args) == 1 and args[0].name == "robot":
                out[:, i] = empty
        return out

    def _interpret_holding(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        out = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for i, pair in enumerate(objects):
            if pair[0].name != "robot":
                continue
            block_name = pair[1].name
            if block_name in ("block0", "block1"):
                out[:, i] = self._is_holding(obs, block_name)
        return out

    def _interpret_on(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        out = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for i, pair in enumerate(objects):
            a_name, b_name = pair[0].name, pair[1].name
            if a_name not in ("block0", "block1") or b_name not in (
                "block0",
                "block1",
            ):
                continue
            if a_name == b_name:
                continue
            pos_a = self._layout.latest_block_pos_in_base(obs, a_name)
            pos_b = self._layout.latest_block_pos_in_base(obs, b_name)
            diff = pos_a - pos_b
            dxy = diff[:, :2].norm(dim=-1)
            dz = diff[:, 2]
            stacked = (
                (dxy < CFG.c_stack_on_xy_thresh)
                & (dz > CFG.c_stack_on_z_min)
                & (dz < CFG.c_stack_on_z_max)
            )
            out[:, i] = stacked
        return out


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


# Isaac Lab's BinaryJointAction convention: negative action closes the
# gripper (finger_joint=0.785), non-negative opens (finger_joint=0.0).
GRIPPER_OPEN: float = 1.0
GRIPPER_CLOSE: float = -1.0


class BaseCornerStackingSkill(LiftedOperatorSkill):
    """Base skill with phase-based scripted control.

    Subclasses drive a :class:`ScriptedBringToGoal` controller through a
    small state machine. Scripted actions are derived from the observation
    via the shared :class:`CornerStackingObsLayout`; the env handle is also
    passed to the controller so it can use the live OSC action scale.
    """

    debug: bool = False

    def __init__(
        self,
        env: Any,
        operators: set[LiftedOperator],
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        super().__init__()
        self._all_operators = operators
        self._lifted_operator = self.get_lifted_operator()
        self._layout = obs_layout
        self._env = env
        self.device = env.unwrapped.device

        self._controller: Optional[ScriptedBringToGoal] = None
        self._phase: int = 0
        self._gripper_timer: int = 0
        self._last_done: Optional[Tensor] = None
        self._step_idx: int = 0
        self._phase_step_idx: int = 0

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        self._controller = None
        self._phase = 0
        self._gripper_timer = 0
        self._last_done = None
        self._step_idx = 0
        self._phase_step_idx = 0
        return super().reset(ground_operator, obs)

    def get_lifted_operator(self) -> LiftedOperator:
        return next(
            op for op in self._all_operators if op.name == self.get_operator_name()
        )

    @abc.abstractmethod
    def get_operator_name(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_controller(self, num_envs: int) -> ScriptedBringToGoal:
        controller = ScriptedBringToGoal(
            action_scale=ACTION_SCALE,
            num_envs=num_envs,
            device=self.device,
            env=self._env,
        )
        return controller

    def _ee_pose_b(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        ee_pos, _ = self._layout.latest_ee_pose_in_base(obs)
        ee_quat = self._layout.latest_ee_quat_in_base(obs)
        return ee_pos, ee_quat

    def _compose_action(self, arm: Tensor, gripper_cmd: float) -> Tensor:
        gripper = torch.full((arm.shape[0], 1), gripper_cmd, device=arm.device)
        return torch.cat([arm, gripper], dim=-1)

    def _phase_step_limit_reached(self) -> bool:
        max_phase_steps = int(getattr(CFG, "c_stack_scripted_max_phase_steps", 0))
        return 0 < max_phase_steps <= self._phase_step_idx + 1

    def _update_phase_step_counter(self, prev_phase: int) -> None:
        if self._phase != prev_phase:
            self._phase_step_idx = 0
        else:
            self._phase_step_idx += 1

    def terminate_with_objects(self, objects: Sequence[Object], obs: Tensor) -> Tensor:
        if self._last_done is None:
            return torch.zeros(obs.shape[0], dtype=torch.bool, device=self.device)
        return self._last_done.clone().detach()


class _ScriptedReachSkill(BaseCornerStackingSkill, abc.ABC):
    """Back-compat alias used by tests/visualisers.

    Exposes the same ``_controller``/``goal_pos``/``goal_quat`` attributes as
    the previous scripted skills so external code that introspects the
    scripted branch (e.g. for frame-marker visualisation) keeps working.
    """


class PickSkill(_ScriptedReachSkill):
    """Scripted pick: pre-grasp → grasp → close gripper → lift.

    ``objects = (robot, block)``.

    Heights: :data:`CFG.c_stack_pre_grasp_z_offset` (approach/lift) and
    :data:`CFG.c_stack_grasp_z_offset` (descent before closing). The target
    orientation snapshots the block's full orientation at reset and applies a
    180-degree pitch in the block frame.
    """

    _PRE_GRASP = 0
    _GRASP = 1
    _CLOSE = 2
    _LIFT = 3
    _DONE = 4

    _GRIPPER_CLOSE_STEPS = 15

    def __init__(
        self,
        env: Any,
        operators: set[LiftedOperator],
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        super().__init__(env, operators, obs_layout)
        self._target_quat_b: Optional[Tensor] = None
        self._block_name: Optional[str] = None
        # Block position snapshotted at Pick reset, used as a fixed frame
        # for the LIFT phase (once the block is grasped, its live obs pose
        # tracks the EE and would otherwise make the LIFT target recede
        # forever).
        self._initial_block_pos_b: Optional[Tensor] = None

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        self._target_quat_b = None
        self._block_name = None
        self._initial_block_pos_b = None
        return super().reset(ground_operator, obs)

    def get_operator_name(self) -> str:
        return "Pick"

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        assert objects[0].name == "robot"
        block = objects[1]
        assert block.name in ("block0", "block1"), f"Unknown block {block.name}"

        with torch.inference_mode():
            ee_pos_b, ee_quat_b = self._ee_pose_b(obs)

            # Lock controller + snapshot block orientation and initial pose
            # on first step.
            if self._controller is None:
                self._controller = self._build_controller(num_envs=obs.shape[0])
                self._block_name = block.name
                block_quat_b = self._layout.latest_block_quat_in_base(obs, block.name)
                self._target_quat_b = (
                    object_pitch_180_target_quat(block_quat_b).detach().clone()
                )
                self._initial_block_pos_b = (
                    self._layout.latest_block_pos_in_base(obs, block.name)
                    .detach()
                    .clone()
                )

            assert self._target_quat_b is not None
            assert self._initial_block_pos_b is not None
            goal_quat = self._target_quat_b

            # Live block pose drives approach/grasp (the gripper has not yet
            # closed on it); the reset-time snapshot drives LIFT so the
            # target does not recede as the block is lifted with the gripper.
            live_block_pos_b = self._layout.latest_block_pos_in_base(obs, block.name)
            pre_dz = CFG.c_stack_pre_grasp_z_offset
            grasp_dz = CFG.c_stack_grasp_z_offset

            if self._phase == self._PRE_GRASP:
                goal_pos = live_block_pos_b.clone()
                goal_pos[:, 2] += pre_dz
                gripper_cmd = GRIPPER_OPEN
            elif self._phase == self._GRASP:
                goal_pos = live_block_pos_b.clone()
                goal_pos[:, 2] += grasp_dz
                gripper_cmd = GRIPPER_OPEN
            elif self._phase == self._CLOSE:
                goal_pos = live_block_pos_b.clone()
                goal_pos[:, 2] += grasp_dz
                gripper_cmd = GRIPPER_CLOSE
            else:  # _LIFT or _DONE
                goal_pos = self._initial_block_pos_b.clone()
                goal_pos[:, 2] += pre_dz
                gripper_cmd = GRIPPER_CLOSE

            self._controller.set_goal(goal_pos, goal_quat)
            arm, pos_err, rot_err = self._controller.compute(ee_pos_b, ee_quat_b)
            action = self._compose_action(arm, gripper_cmd)

            pos_ok = pos_err < CFG.c_stack_pos_threshold
            rot_ok = rot_err < CFG.c_stack_rot_threshold

            if self.debug and (self._step_idx % 2 == 0):
                print(
                    f"    [Pick step={self._step_idx} phase={self._phase}] "
                    f"pos_err={pos_err.max().item():.3f} "
                    f"rot_err={rot_err.max().item():.3f} "
                    f"block_z={live_block_pos_b[:, 2].max().item():.3f} "
                    f"ee_z={ee_pos_b[:, 2].max().item():.3f}"
                )
            prev_phase = self._phase

            # Phase advance: PRE_GRASP requires orientation lock; the rest
            # only wait for position to converge.
            phase_timed_out = self._phase_step_limit_reached()
            if self._phase == self._PRE_GRASP and (
                (pos_ok & rot_ok).all() or phase_timed_out
            ):
                self._phase = self._GRASP
            elif self._phase == self._GRASP and (pos_ok.all() or phase_timed_out):
                self._phase = self._CLOSE
                self._gripper_timer = 0
            elif self._phase == self._CLOSE:
                self._gripper_timer += 1
                if self._gripper_timer >= self._GRIPPER_CLOSE_STEPS or phase_timed_out:
                    self._phase = self._LIFT
            elif self._phase == self._LIFT and (pos_ok.all() or phase_timed_out):
                self._phase = self._DONE

            if self.debug and self._phase != prev_phase:
                print(
                    f"    [Pick step={self._step_idx}] "
                    f"phase {prev_phase} → {self._phase}"
                )

            self._update_phase_step_counter(prev_phase)
            self._step_idx += 1

            self._last_done = torch.full(
                (obs.shape[0],),
                self._phase == self._DONE,
                dtype=torch.bool,
                device=self.device,
            )

        return action.clone().detach()


class PlaceSkill(_ScriptedReachSkill):
    """Scripted place: pre-place → place (open gripper) → retract.

    ``objects = (robot, grasp_block, target_block)``.

    Pick already leaves the grasped block lifted. Place moves directly to
    the pre-place pose over the target block, descends, then opens once
    position is locked.
    """

    _PRE_PLACE = 0
    _PLACE = 1
    _OPEN = 2
    _DONE = 3

    _GRIPPER_OPEN_STEPS = 15

    def __init__(
        self,
        env: Any,
        operators: set[LiftedOperator],
        obs_layout: CornerStackingObsLayout,
    ) -> None:
        super().__init__(env, operators, obs_layout)
        self._target_quat_b: Optional[Tensor] = None
        self._target_block_name: Optional[str] = None

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        self._target_quat_b = None
        self._target_block_name = None
        return super().reset(ground_operator, obs)

    def get_operator_name(self) -> str:
        return "Place"

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        assert objects[0].name == "robot"
        target_block = objects[2]
        assert target_block.name in (
            "block0",
            "block1",
        ), f"Unknown target {target_block.name}"

        with torch.inference_mode():
            ee_pos_b, ee_quat_b = self._ee_pose_b(obs)

            if self._controller is None:
                self._controller = self._build_controller(num_envs=obs.shape[0])
                self._target_block_name = target_block.name
                target_quat_b = self._layout.latest_block_quat_in_base(
                    obs, target_block.name
                )
                self._target_quat_b = (
                    object_pitch_180_target_quat(target_quat_b).detach().clone()
                )

            assert self._target_quat_b is not None
            goal_quat = self._target_quat_b

            tgt_pos_b = self._layout.latest_block_pos_in_base(obs, target_block.name)
            pre_dz = CFG.c_stack_pre_grasp_z_offset
            place_dz = CFG.c_stack_place_dz

            if self._phase == self._PRE_PLACE:
                goal_pos = tgt_pos_b.clone()
                goal_pos[:, 2] += pre_dz
                gripper_cmd = GRIPPER_CLOSE
            elif self._phase == self._PLACE:
                goal_pos = tgt_pos_b.clone()
                goal_pos[:, 2] += place_dz
                gripper_cmd = GRIPPER_CLOSE
            elif self._phase == self._OPEN:
                goal_pos = tgt_pos_b.clone()
                goal_pos[:, 2] += place_dz
                gripper_cmd = GRIPPER_OPEN
            else:  # _DONE
                goal_pos = tgt_pos_b.clone()
                goal_pos[:, 2] += pre_dz
                gripper_cmd = GRIPPER_OPEN

            self._controller.set_goal(goal_pos, goal_quat)
            arm, pos_err, rot_err = self._controller.compute(ee_pos_b, ee_quat_b)
            action = self._compose_action(arm, gripper_cmd)

            pos_ok = pos_err < CFG.c_stack_pos_threshold
            rot_ok = rot_err < CFG.c_stack_rot_threshold

            if self.debug and (self._step_idx % 2 == 0):
                print(
                    f"    [Place step={self._step_idx} phase={self._phase}] "
                    f"pos_err={pos_err.max().item():.3f} "
                    f"rot_err={rot_err.max().item():.3f} "
                    f"tgt_z={tgt_pos_b[:, 2].max().item():.3f} "
                    f"ee_z={ee_pos_b[:, 2].max().item():.3f}"
                )
            prev_phase = self._phase

            # PRE_PLACE locks orientation; PLACE/OPEN only wait on position.
            phase_timed_out = self._phase_step_limit_reached()
            if self._phase == self._PRE_PLACE and (
                (pos_ok & rot_ok).all() or phase_timed_out
            ):
                self._phase = self._PLACE
            elif self._phase == self._PLACE and (pos_ok.all() or phase_timed_out):
                self._phase = self._OPEN
                self._gripper_timer = 0
            elif self._phase == self._OPEN:
                self._gripper_timer += 1
                if self._gripper_timer >= self._GRIPPER_OPEN_STEPS or phase_timed_out:
                    self._phase = self._DONE

            if self.debug and self._phase != prev_phase:
                print(
                    f"    [Place step={self._step_idx}] "
                    f"phase {prev_phase} → {self._phase}"
                )

            self._update_phase_step_counter(prev_phase)
            self._step_idx += 1

            self._last_done = torch.full(
                (obs.shape[0],),
                self._phase == self._DONE,
                dtype=torch.bool,
                device=self.device,
            )

        return action.clone().detach()


class RLSkill(LiftedOperatorSkill):
    """Two-phase RL-then-retreat skill: RL inserts the block, scripted
    retreat opens the gripper and lifts the hand clear.

    ``objects = (robot, block, box)``.

    Phase 1 (``_PHASE_RL``): the pretrained block_box OSC policy drives
    the gripper. Per env, this phase ends when
    :meth:`InsertiveAssembledChecker.assembled_only` fires — alignment +
    upright, *no* gripper-state requirement. The policy may still be
    holding the block at that moment.

    Phase 2 (``_PHASE_OPEN`` → ``_PHASE_LIFT``): scripted,
    action-space identical to the policy (6 OSC + 1 binary gripper).

    * ``_PHASE_OPEN``: arm action zero, gripper action ``+1`` (open).
      Per env, ends when ``finger_joint`` drops below the open
      threshold (:meth:`InsertiveAssembledChecker.gripper_open`).
    * ``_PHASE_LIFT``: scripted pose control toward a static lifted,
      top-down gripper pose (Robotiq local ``+Z`` aligned with base/world
      ``-Z``), gripper held open. The target xy is snapped to the box
      center and target z is fixed at 0.5 m in robot-base coordinates.
      Per env, ends when base-frame EE-z reaches the target height, wrist
      rotation is within ``CFG.c_stack_rot_threshold``, and the arm joint
      velocities are below
      ``CFG.cstack_retreat_lift_joint_vel_max``. LIFT keeps tracking the
      static goal until termination so the target does not walk with the
      current pose.
    * ``_PHASE_DONE``: zero action; ``terminate_with_objects`` returns
      True. Gripper command stays at zero — which the binary action
      term maps to *open* (mask is ``actions < 0``), so it remains
      released.

    Phase tracking is per env, so envs that finish phase 1 at different
    steps each progress through phase 2 independently. The policy is
    only queried when at least one env is still in ``_PHASE_RL``.

    Policy loading is deferred — callers must provide a ``policy``
    callable via :meth:`set_policy` before the skill is invoked. The
    operator's preconditions and effects are intentionally empty so task
    planning ignores this skill; it is meant to be invoked directly.
    """

    _PHASE_RL: int = 0
    _PHASE_OPEN: int = 1
    _PHASE_LIFT: int = 2
    _PHASE_DONE: int = 3
    _LIFT_GOAL_Z_B: float = 0.5

    # Action layout: 6-dim OSC delta + 1-dim binary gripper. Gripper
    # convention (BinaryJointAction.process_actions): ``actions < 0`` →
    # close, ``actions ≥ 0`` → open. We use ``+1.0`` to open.
    _ACTION_DIM: int = 7
    _GRIPPER_OPEN_CMD: float = 1.0
    debug: bool = False

    def __init__(
        self,
        env: Any,
        operators: set[LiftedOperator],
    ) -> None:
        super().__init__()
        self._all_operators = operators
        self._lifted_operator = next(
            op for op in operators if op.name == self.get_operator_name()
        )
        self._env = env
        self.device = env.unwrapped.device
        self._policy: Optional[Callable[[Tensor], Tensor]] = None
        self._checker: Optional[InsertiveAssembledChecker] = None
        self._phase: Optional[Tensor] = None  # (N,) int8
        self._last_done: Optional[Tensor] = None
        self._lift_controller: Optional[ScriptedBringToGoal] = None
        self._lift_goal_initialized: Optional[Tensor] = None  # (N,) bool
        self._phase_step_count: Optional[Tensor] = None  # (N,) int32
        self._step_idx: int = 0

    def get_lifted_operator(self) -> LiftedOperator:
        return self._lifted_operator

    def get_operator_name(self) -> str:
        return "RL"

    def set_policy(self, policy: Callable[[Tensor], Tensor]) -> None:
        """Inject the pretrained policy callable (obs → action)."""
        self._policy = policy

    def get_lift_goal_in_base(self) -> tuple[Tensor, Tensor] | None:
        """Return the active LIFT wrist goal in robot base frame, if any."""
        if (
            self._phase is None
            or self._lift_controller is None
            or self._lift_controller.goal_pos is None
            or self._lift_controller.goal_quat is None
        ):
            return None
        lift_mask = self._phase == self._PHASE_LIFT
        if not bool(lift_mask.any().item()):
            return None
        return (
            self._lift_controller.goal_pos.clone().detach(),
            self._lift_controller.goal_quat.clone().detach(),
        )

    def _ensure_checker(self) -> InsertiveAssembledChecker:
        if self._checker is None:
            self._checker = InsertiveAssembledChecker(
                self._env,
                pos_tol=CFG.cstack_assembled_pos_tol,
                upright_tol=CFG.cstack_assembled_upright_tol,
                min_gripper_gap=CFG.cstack_assembled_min_gripper_gap,
            )
        return self._checker

    def _ensure_phase_state(self, num_envs: int) -> None:
        if self._phase is None or self._phase.shape[0] != num_envs:
            with torch.inference_mode(False):
                self._phase = torch.full(
                    (num_envs,),
                    self._PHASE_RL,
                    dtype=torch.int8,
                    device=self.device,
                )
                self._lift_goal_initialized = torch.zeros(
                    num_envs, dtype=torch.bool, device=self.device
                )
                self._phase_step_count = torch.zeros(
                    num_envs, dtype=torch.int32, device=self.device
                )

    def reset(self, ground_operator: GroundOperator, obs: Tensor) -> None:
        self._phase = None
        self._last_done = None
        self._lift_controller = None
        self._lift_goal_initialized = None
        self._phase_step_count = None
        self._step_idx = 0
        return super().reset(ground_operator, obs)

    def _scripted_phase_step_limit_reached(self, phase: int) -> Tensor:
        assert self._phase is not None
        max_phase_steps = int(getattr(CFG, "c_stack_scripted_max_phase_steps", 0))
        if max_phase_steps <= 0 or self._phase_step_count is None:
            return torch.zeros_like(self._phase, dtype=torch.bool)
        return (self._phase == phase) & (self._phase_step_count >= max_phase_steps)

    def _transition_phase(self, mask: Tensor, next_phase: int) -> None:
        assert self._phase is not None
        if not bool(mask.any().item()):
            return
        self._phase[mask] = next_phase
        if self._phase_step_count is not None:
            self._phase_step_count[mask] = 0

    def _update_phases(self) -> None:
        """Advance per-env phases based on current sim state."""
        assert self._phase is not None
        checker = self._ensure_checker()

        # RL → OPEN: the insertive block is now seated upright in the box.
        rl_mask = self._phase == self._PHASE_RL
        if bool(rl_mask.any().item()):
            assembled = checker.assembled_only(self._env)
            transition = rl_mask & assembled
            self._transition_phase(transition, self._PHASE_OPEN)

        # OPEN → LIFT: gripper has finished opening past the configured gap.
        open_mask = self._phase == self._PHASE_OPEN
        if bool(open_mask.any().item()):
            opened = checker.gripper_open(self._env)
            timed_out = self._scripted_phase_step_limit_reached(self._PHASE_OPEN)
            transition = open_mask & (opened | timed_out)
            self._transition_phase(transition, self._PHASE_LIFT)

    def _lift_rotation_ok(self) -> Tensor:
        """Per-env bool: active LIFT goal rotation is within threshold."""
        if (
            self._phase is None
            or self._lift_controller is None
            or self._lift_controller.goal_pos is None
            or self._lift_controller.goal_quat is None
        ):
            n = 0 if self._phase is None else self._phase.shape[0]
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        ee_pos_b, ee_quat_b = get_ee_pose_in_base(self._env)
        _, rot_error_aa = compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            self._lift_controller.goal_pos,
            self._lift_controller.goal_quat,
            rot_error_type="axis_angle",
        )
        return rot_error_aa.norm(dim=-1) < CFG.c_stack_rot_threshold

    def _lift_height_ok(self) -> Tensor:
        """Per-env bool: EE has reached the active lift goal height."""
        if (
            self._phase is None
            or self._lift_controller is None
            or self._lift_controller.goal_pos is None
        ):
            n = 0 if self._phase is None else self._phase.shape[0]
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        ee_pos_b, _ = get_ee_pose_in_base(self._env)
        return ee_pos_b[:, 2] >= (
            self._lift_controller.goal_pos[:, 2] - float(CFG.cstack_retreat_lift_z_tol)
        )

    def _lift_joint_velocity_ok(self) -> Tensor:
        """Per-env bool: OSC arm joints are slow enough to terminate."""
        n = 0 if self._phase is None else self._phase.shape[0]
        action_manager = getattr(self._env.unwrapped, "action_manager", None)
        terms = getattr(action_manager, "_terms", None)
        arm = terms.get("arm") if isinstance(terms, dict) else None
        robot = getattr(arm, "_asset", None)
        joint_ids = getattr(arm, "_joint_ids", None)
        if robot is None or joint_ids is None:
            return torch.ones(n, dtype=torch.bool, device=self.device)

        joint_vel = robot.data.joint_vel[:, joint_ids]
        max_joint_speed = joint_vel.abs().amax(dim=-1)
        return max_joint_speed < float(CFG.cstack_retreat_lift_joint_vel_max)

    def _lift_goal_reached(self) -> Tensor:
        """Per-env bool: LIFT can terminate while still tracking its goal."""
        if self._phase is None:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        lift_mask = self._phase == self._PHASE_LIFT
        if not bool(lift_mask.any().item()):
            return torch.zeros_like(lift_mask)
        return (
            lift_mask
            & self._lift_height_ok()
            & self._lift_rotation_ok()
            & self._lift_joint_velocity_ok()
        )

    def _query_policy(self, obs: Tensor) -> Tensor:
        """Run the pretrained block_box policy on the prefix-sliced obs."""
        assert (
            self._policy is not None
        ), "RLSkill has no policy — call set_policy() before running."
        # rsl_rl's inference policy expects a dict-like obs keyed by obs
        # group (see ``ActorCritic.get_actor_obs``); callers upstream pass
        # a flat tensor, so re-wrap the sliced prefix as a TensorDict with
        # the ``policy`` group key the pretrained actor was trained on.
        from tensordict import TensorDict

        with torch.inference_mode():
            sliced = obs[..., :BLOCK_BOX_POLICY_OBS_DIM]
            obs_dict = TensorDict({"policy": sliced}, batch_size=[sliced.shape[0]])
            return self._policy(obs_dict).clone().detach()

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        n = obs.shape[0]
        self._ensure_phase_state(n)
        assert self._phase is not None
        self._update_phases()

        action = torch.zeros((n, self._ACTION_DIM), device=obs.device)

        rl_mask = self._phase == self._PHASE_RL
        if bool(rl_mask.any().item()):
            policy_out = self._query_policy(obs)
            # Pad/truncate the policy's action width to our action dim in
            # case the loaded checkpoint differs (defensive — block_box
            # was trained on the same 7-dim layout).
            if policy_out.shape[-1] < self._ACTION_DIM:
                pad = torch.zeros(
                    n,
                    self._ACTION_DIM - policy_out.shape[-1],
                    device=policy_out.device,
                )
                policy_out = torch.cat([policy_out, pad], dim=-1)
            elif policy_out.shape[-1] > self._ACTION_DIM:
                policy_out = policy_out[..., : self._ACTION_DIM]
            action[rl_mask] = policy_out[rl_mask]

        open_mask = self._phase == self._PHASE_OPEN
        if bool(open_mask.any().item()):
            action[open_mask, -1] = self._GRIPPER_OPEN_CMD

        lift_mask = self._phase == self._PHASE_LIFT
        if bool(lift_mask.any().item()):
            assert self._lift_goal_initialized is not None
            if self._lift_controller is None or self._lift_controller.num_envs != n:
                self._lift_controller = ScriptedBringToGoal(
                    action_scale=ACTION_SCALE,
                    num_envs=n,
                    device=self.device,
                    env=self._env,
                )
                self._lift_goal_initialized = torch.zeros(
                    n, dtype=torch.bool, device=self.device
                )
            ee_pos_b, ee_quat_b = get_ee_pose_in_base(self._env)
            new_lift_mask = lift_mask & ~self._lift_goal_initialized
            if bool(new_lift_mask.any().item()):
                box_pos_b, _ = get_receptive_object_pose_in_base(self._env)
                lift_goal_pos_b = box_pos_b.clone()
                lift_goal_pos_b[:, 2] = self._LIFT_GOAL_Z_B
                self._lift_controller.set_goal(
                    lift_goal_pos_b,
                    top_down_wrist_target_quat(self._env),
                    env_mask=new_lift_mask,
                )
                self._lift_goal_initialized[new_lift_mask] = True
            lift_arm, pos_err, rot_err = self._lift_controller.compute(
                ee_pos_b, ee_quat_b
            )
            if self.debug and (self._step_idx % 2 == 0):
                print(
                    f"    [RLSkill step={self._step_idx} phase=LIFT] "
                    f"pos_err={pos_err.max().item():.3f} "
                    f"rot_err={rot_err.max().item():.3f} "
                    f"ee_z={ee_pos_b[:, 2].max().item():.3f} "
                    f"goal_z={self._LIFT_GOAL_Z_B:.3f}"
                )
            action[lift_mask, : self._ACTION_DIM - 1] = lift_arm[lift_mask]
            action[lift_mask, -1] = self._GRIPPER_OPEN_CMD

        # _PHASE_DONE leaves the row at zero; this should only be observed
        # after terminate_with_objects has already reported success.
        if self._phase_step_count is not None:
            active = self._phase != self._PHASE_DONE
            self._phase_step_count[active] += 1
        self._step_idx += 1
        return action

    def terminate_with_objects(self, objects: Sequence[Object], obs: Tensor) -> Tensor:
        n = obs.shape[0]
        self._ensure_phase_state(n)
        assert self._phase is not None
        done_now = self._lift_goal_reached() | self._scripted_phase_step_limit_reached(
            self._PHASE_LIFT
        )
        if bool(done_now.any().item()):
            self._transition_phase(done_now, self._PHASE_DONE)
        done = self._phase == self._PHASE_DONE
        self._last_done = done.clone().detach()
        return done


# ---------------------------------------------------------------------------
# TAMP system
# ---------------------------------------------------------------------------


class CornerStackingRLTAMPSystem(BaseRLTAMPSystem):
    """TAMP system for the 3-object Isaac Sim cornered-stacking task."""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        seed: int | None = None,
        render_mode: str | None = None,
        env_cfg: Any = None,
    ) -> None:
        self._render_mode = render_mode
        self._env_cfg = env_cfg
        super().__init__(
            planning_components, name="CornerStackingRLTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        return make_corner_stacking_env(
            num_envs=CFG.c_stack_num_envs,
            max_steps=CFG.c_stack_max_steps,
            env_cfg=self._env_cfg,
            task_id=CORNERED_STACKING_TASK_ID,
            render_mode=self._render_mode,
        )

    def _get_domain_name(self) -> str:
        return "CornerStacking-domain"

    def get_domain(self) -> PDDLDomain:
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.type_container.as_set(),
        )

    @classmethod
    def _create_planning_components(cls) -> TensorPlanningComponents:
        types_container = CornerStackingTypes()
        predicates = CornerStackingPredicates(types_container)
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]

        robot = Variable("?robot", types_container.robot)
        grasp_block = Variable("?grasp_block", types_container.block)
        target_block = Variable("?target_block", types_container.block)
        rl_block = Variable("?block", types_container.block)
        rl_box = Variable("?box", types_container.box)

        operators = {
            LiftedOperator(
                "Pick",
                [robot, grasp_block],
                preconditions={LiftedAtom(HandEmpty, (robot,))},
                add_effects={LiftedAtom(Holding, (robot, grasp_block))},
                delete_effects={LiftedAtom(HandEmpty, (robot,))},
            ),
            LiftedOperator(
                "Place",
                [robot, grasp_block, target_block],
                preconditions={LiftedAtom(Holding, (robot, grasp_block))},
                add_effects={
                    LiftedAtom(On, (grasp_block, target_block)),
                    LiftedAtom(HandEmpty, (robot,)),
                },
                delete_effects={LiftedAtom(Holding, (robot, grasp_block))},
            ),
            # Empty preconditions and effects: the RL operator is invisible
            # to task planning. The paired RLSkill is meant to be invoked
            # directly (e.g. to clear the block into the box before Pick/Place).
            LiftedOperator(
                "RL",
                [robot, rl_block, rl_box],
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
            ),
        }

        # Placeholder perceiver; create_default will replace it with one that
        # has an obs layout resolved from the live env.
        perceiver = CornerStackingPerceiver(
            predicates, types_container, obs_layout=None  # type: ignore[arg-type]
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
    ) -> CornerStackingRLTAMPSystem:
        planning_components = cls._create_planning_components()
        system = cls(
            planning_components,
            seed=seed,
            render_mode=render_mode,
            env_cfg=env_cfg,
        )

        # Resolve the obs layout from the live articulation exactly once.
        layout = CornerStackingObsLayout.from_env(system.env)
        assert isinstance(system.components.perceiver, CornerStackingPerceiver)
        system.components.perceiver._layout = layout  # type: ignore[attr-defined]

        skills: set[LiftedOperatorSkill] = {
            PickSkill(system.env, system.components.operators, layout),
            PlaceSkill(system.env, system.components.operators, layout),
            RLSkill(system.env, system.components.operators),
        }
        system.components.skills.update(skills)  # type: ignore[arg-type]
        return system

    @property
    def rl_skill(self) -> RLSkill:
        """Return the RLSkill instance in this system's skill set."""
        for skill in self.components.skills:
            if isinstance(skill, RLSkill):
                return skill
        raise RuntimeError("No RLSkill registered on this TAMP system.")

    # ------------------------------------------------------------------
    # Abstract-method stubs — graph/state-vec helpers are not needed for
    # the first iteration of this env (no skill refactorization yet).
    # ------------------------------------------------------------------

    def state_to_graph(self, state: Tensor, training: bool = False) -> List[GraphData]:
        """Convert policy observations into object-pose graph features.

        Nodes are ``robot``, ``block0``, ``block1``, and ``box`` in the
        perceiver's stable object order. Each node stores its pose in robot
        base frame as ``[x, y, z, qw, qx, qy, qz]``. Edges are directed and
        fully connected without self-loops; each edge stores the relative pose
        from source object frame to target object frame.
        """
        del training
        obs = state
        b = obs.shape[0]
        object_to_node = self.perceiver.objects.object_to_node
        num_nodes = len(object_to_node)
        node_feature_dim = 7
        node_features = torch.zeros(
            b,
            num_nodes,
            node_feature_dim,
            dtype=obs.dtype,
            device=obs.device,
        )

        assert isinstance(self.perceiver, CornerStackingPerceiver)
        layout = self.perceiver._layout  # pylint: disable=protected-access

        ee_pos_b, _ = layout.latest_ee_pose_in_base(obs)
        pose_by_name: dict[str, tuple[Tensor, Tensor]] = {
            "robot": (ee_pos_b, layout.latest_ee_quat_in_base(obs)),
            "block0": (
                layout.latest_insertive_pos_in_base(obs),
                layout.latest_insertive_quat_in_base(obs),
            ),
            "block1": (
                layout.latest_outside_pos_in_base(obs),
                layout.latest_outside_quat_in_base(obs),
            ),
            "box": (
                layout.latest_receptive_pos_in_base(obs),
                layout.latest_receptive_quat_in_base(obs),
            ),
        }
        for obj, node_idx in object_to_node.items():
            pos_b, quat_b = pose_by_name[obj.name]
            node_features[:, node_idx, :] = torch.cat([pos_b, quat_b], dim=-1)

        edge_list: list[list[int]] = []
        edge_features_list: list[Tensor] = []
        for src_idx in range(num_nodes):
            for dst_idx in range(num_nodes):
                if src_idx == dst_idx:
                    continue
                edge_list.append([src_idx, dst_idx])
                src_pos = node_features[:, src_idx, :3]
                src_quat = node_features[:, src_idx, 3:7]
                dst_pos = node_features[:, dst_idx, :3]
                dst_quat = node_features[:, dst_idx, 3:7]
                src_quat_inv = quat_inv(src_quat)
                rel_pos = quat_rotate(src_quat_inv, dst_pos - src_pos)
                rel_pos[
                    :, 2
                ] *= 0  # zero out relative z to encourage planar generalization
                rel_quat = quat_mul(src_quat_inv, dst_quat)
                edge_features_list.append(torch.cat([rel_pos, rel_quat], dim=-1))

        edge_indices = torch.tensor(edge_list, dtype=torch.long).T
        edge_features = torch.stack(edge_features_list, dim=1)

        return [
            GraphData(
                node_features=node_features[batch_idx],
                edge_features=edge_features[batch_idx],
                edge_indices=edge_indices,
                global_features=None,
                object_to_node=object_to_node.copy(),
            )
            for batch_idx in range(b)
        ]

    def state_to_vec(self, state: Tensor, objects: Sequence[Object]) -> Tensor:
        raise NotImplementedError

    def skill_obs_space(self) -> gym.Space:
        return self.env.observation_space

    def full_state_to_relative_state(
        self, full_state: Tensor, relative_name: str = ""
    ) -> Tensor:
        raise NotImplementedError(
            "Relative-state encoding is not defined for CornerStacking yet."
        )

    def relative_state_to_full_state(
        self, inital_full_state: Tensor, relative_state: Tensor
    ) -> Tensor:
        raise NotImplementedError(
            "Relative-state encoding is not defined for CornerStacking yet."
        )
