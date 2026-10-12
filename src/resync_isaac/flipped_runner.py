"""Flipped demo executor, adapted from the source learned-predicate demo."""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any
from itertools import permutations
import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from skill_refactor.settings import CFG
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanner,
    TaskThenMotionPlanningFailure,
)
from resync_isaac.flipped_stacking.flipped_stacking import FlippedStackingRLTAMPSystem
from resync_isaac.flipped_stacking.utils import get_cube_pose_in_base, get_task_success
from resync_isaac.flipped_stacking.task_cfg import top_supported_only_by_bottom
from resync_isaac.video import PlanVideoRecorder

BOTH_CUBES = "both"


def get_frozen_action(
    skill_action: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    action = torch.zeros_like(skill_action)
    action[:, -1] = skill_action[:, -1]
    return action


def _expected_plans(
    upright_by_block: dict[str, bool],
) -> list[list[tuple[str, tuple[str, ...]]]]:
    """Return valid plans for the observed physical reset state.

    When both cubes need recovery, either flip order is valid, but both flips
    must precede Pick and Place.
    """
    ordered_blocks = tuple(upright_by_block)
    needed = [block for block in ordered_blocks if not upright_by_block[block]]
    # Build the tower bottom-up: mid2 on bottom, then mid1, then top.
    stacking_suffix = []
    for upper, lower in reversed(tuple(zip(ordered_blocks[:-1], ordered_blocks[1:]))):
        stacking_suffix.extend(
            [
                ("Pick", ("robot", upper)),
                ("Place", ("robot", upper, lower)),
            ]
        )
    return [
        [
            step
            for block in recovery_order
            for step in (
                ("MoveToFlip", ("robot", block)),
                ("Flip", ("robot", block)),
            )
        ]
        + stacking_suffix
        for recovery_order in permutations(needed)
    ]


def _plan_as_tuples(plan: Any) -> list[tuple[str, tuple[str, ...]]]:
    return [(op.name, tuple(obj.name for obj in op.parameters)) for op in plan]


def _is_upright(system: FlippedStackingRLTAMPSystem, cube_name: str) -> bool:
    _, quat = get_cube_pose_in_base(system.env, cube_name)
    roll, pitch, _ = euler_xyz_from_quat(quat)
    error = wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs()
    return bool((error < float(CFG.fstack_upright_tol)).all().item())


def _physical_diagnostics(system: FlippedStackingRLTAMPSystem) -> dict[str, Any]:
    """Capture final block poses and adjacent contact forces for failures."""
    names = tuple(block.name for block in system.perceiver.objects.blocks)
    poses = {}
    for name in names:
        pos, quat = get_cube_pose_in_base(system.env, name)
        roll, pitch, yaw = euler_xyz_from_quat(quat)
        poses[name] = {
            "position": pos[0].tolist(),
            "rpy": [roll[0].item(), pitch[0].item(), yaw[0].item()],
        }
    contacts = {}
    for upper, lower in zip(names[:-1], names[1:]):
        sensor = system.env.unwrapped.scene[f"{upper}_{lower}_contact"]
        matrix = sensor.data.force_matrix_w
        contacts[f"{upper}_on_{lower}"] = (
            None
            if matrix is None
            else matrix[:, :, 0, :].sum(dim=1).norm(dim=-1).tolist()
        )
    return {"poses": poses, "contacts": contacts}


class _TerminationOnlyPlanner(TaskThenMotionPlanner):
    """Execute the learned plan without validating symbolic effects."""

    _current_task_plan: list[Any]
    _current_skill: Any
    _current_operator: Any
    _last_action: torch.Tensor

    def __init__(self, *args: Any, block_order: tuple[str, ...], **kwargs: Any):
        self._block_order = block_order
        self.plan_complete = False
        super().__init__(*args, **kwargs)

    def reset(self, obs: torch.Tensor, info: dict[str, Any]) -> None:
        self.plan_complete = False
        super().reset(obs, info)
        plan = self._current_task_plan
        recovery_sequence = [
            operator for operator in plan if operator.name in ("MoveToFlip", "Flip")
        ]
        recovery: dict[str, dict[str, Any]] = {}
        picks: dict[str, Any] = {}
        places: dict[str, Any] = {}
        for operator in plan:
            if operator.name in ("MoveToFlip", "Flip"):
                recovery.setdefault(operator.parameters[1].name, {})[
                    operator.name
                ] = operator
            elif operator.name == "Pick":
                picks[operator.parameters[1].name] = operator
            elif operator.name == "Place":
                places[operator.parameters[1].name] = operator
        expected_uppers = set(self._block_order[:-1])
        valid_stack = set(picks) == expected_uppers and set(places) == expected_uppers
        if valid_stack:
            valid_stack = all(
                places[upper].parameters[2].name == lower
                for upper, lower in zip(self._block_order[:-1], self._block_order[1:])
            )
        valid_recovery = all(
            set(pair) == {"MoveToFlip", "Flip"} for pair in recovery.values()
        )
        if valid_stack:
            # Recovery skills operate on free cubes, and tower construction must
            # proceed bottom-up even though STRIPS regards all stack orders equal.
            if valid_recovery:
                ordered = []
                for block in self._block_order:
                    if block in recovery:
                        ordered.extend(
                            [recovery[block]["MoveToFlip"], recovery[block]["Flip"]]
                        )
            else:
                # Preserve an incomplete learned recovery prefix verbatim, but
                # never let that symbolic error also reorder tower construction.
                ordered = recovery_sequence
            for upper in reversed(self._block_order[:-1]):
                ordered.extend([picks[upper], places[upper]])
            self._current_task_plan = ordered
        else:
            logging.info(
                "Raw learned plan was not canonicalizable: %s", _plan_as_tuples(plan)
            )

    def step(self, obs: torch.Tensor):
        if self._current_skill is not None and bool(
            self._current_skill.terminate(obs).all().item()
        ):
            self._current_skill = None
            self._current_operator = None
            if not self._current_task_plan:
                self.plan_complete = True

        if self._current_skill is None:
            if not self._current_task_plan:
                frozen_action = get_frozen_action(
                    self._last_action,
                    self._arm_action_low,
                    self._arm_action_high,
                    self._normalize_action,
                    CFG.control_mode,
                    obs=obs,
                )
                self._last_action = frozen_action.clone()
                return frozen_action, self._current_operator
            self._current_operator = self._current_task_plan.pop(0)
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            assert self._current_skill is not None
            self._current_skill.reset(self._current_operator, obs)

        action = self._current_skill.get_action(obs)
        self._last_action = action.clone()
        return action, self._current_operator


def _make_planner(system, env, operators) -> TaskThenMotionPlanner:
    num_actions = env.num_actions
    return _TerminationOnlyPlanner(
        types=system.types,
        predicates=system.perceiver.predicates_container.as_set(),
        perceiver=system.perceiver,
        operators=operators,
        skills=system.skills,
        fallback_action=torch.zeros(num_actions, device=env.device).cpu().numpy(),
        normalize_action=False,
        arm_action_low=torch.full((num_actions,), -1.0, device=env.device),
        arm_action_high=torch.full((num_actions,), 1.0, device=env.device),
        planner_id="pyperplan",
        domain_name="FlippedStacking-learned-domain",
        block_order=tuple(block.name for block in system.perceiver.objects.blocks),
    )


def _evaluate_episode(
    system: FlippedStackingRLTAMPSystem,
    env: Any,
    planner: TaskThenMotionPlanner,
    role: str,
    episode: int,
    max_steps: int,
    video_path: Path | None = None,
    video_fps: int = 20,
    video_stride: int = 1,
    video_frame_scale: float = 1.5,
) -> dict[str, Any]:
    """Plan and execute one complete learned-domain episode."""
    started = time.time()
    result: dict[str, Any] = {
        "role": role,
        "episode": episode,
        "success": False,
        "plan_correct": False,
        "steps": 0,
        "operator_trace": [],
        "planner_exhausted": False,
        "exhausted_operator": None,
        "failure": None,
    }
    video: PlanVideoRecorder | None = None
    try:
        obs_td, extras = env.reset()
        obs = obs_td["policy"]
        upright_by_block = {
            block.name: _is_upright(system, block.name)
            for block in system.perceiver.objects.blocks
        }
        initially_upright = (
            all(upright_by_block.values())
            if role == BOTH_CUBES
            else all(upright_by_block.values())
        )
        result["initially_upright"] = initially_upright
        result["initially_upright_blocks"] = upright_by_block
        planner.reset(obs, extras if isinstance(extras, dict) else {})
        plan = _plan_as_tuples(
            planner._current_task_plan
        )  # pylint: disable=protected-access
        expected_plans = _expected_plans(upright_by_block)
        result["plan"] = plan
        result["expected_plans"] = expected_plans
        result["plan_correct"] = plan in expected_plans
        if video_path is not None:
            video = PlanVideoRecorder(
                video_path,
                plan,
                video_fps,
                video_stride,
                frame_scale=video_frame_scale,
            )
            result["video_path"] = str(video_path)
            video.capture(
                system,
                None,
                f"{role.upper()} INIT | episode {episode:02d}",
                force=True,
            )
        if plan not in expected_plans:
            result["failure"] = (
                f"plan mismatch: expected one of {expected_plans}, got {plan}"
            )
        if not plan:
            raise TaskThenMotionPlanningFailure(
                "learned predicates produced an empty plan before physical success"
            )

        trace: list[str] = []
        plan_completed_at: int | None = None
        for step in range(max_steps):
            if step % 100 == 0:
                print(
                    f"[{role} episode {episode}] step {step}, skill={getattr(planner._current_operator, 'name', None)}",
                    flush=True,
                )
            if step == max_steps - 1:
                result["pre_timeout_diagnostics"] = _physical_diagnostics(system)
            action, operator = planner.step(obs)
            if action is None:
                result["failure"] = "planner returned no action"
                break
            if operator is not None and (not trace or trace[-1] != operator.name):
                trace.append(operator.name)
            obs_td, _, dones, _ = env.step(action)
            obs = obs_td["policy"]
            result["steps"] = step + 1
            if video is not None:
                video.capture(
                    system,
                    operator.name if operator is not None else None,
                    f"{role.upper()} | episode {episode:02d} | step {step + 1:04d}",
                )
            if bool(get_task_success(env).all().item()):
                result["success"] = True
                if len(system.perceiver.objects.blocks) == 2:
                    contact_ok, bottom_force, other_force = (
                        top_supported_only_by_bottom(system.env)
                    )
                    result["success"] = bool(contact_ok.all().item())
                    result["bottom_force"] = bottom_force.tolist()
                    result["other_force"] = other_force.tolist()
                    if not result["success"]:
                        result["failure"] = (
                            "goal pose reached without bottom-only support"
                        )
                break
            if planner.plan_complete:  # type: ignore[attr-defined]
                if plan_completed_at is None:
                    plan_completed_at = step
                    result["plan_completed_at_step"] = step + 1
                elif step - plan_completed_at >= 20:
                    result["failure"] = (
                        "skill plan completed without physical task success"
                    )
                    result["final_diagnostics"] = _physical_diagnostics(system)
                    break
            if bool(planner.exhausted.all().item()):
                result["planner_exhausted"] = True
                result["exhausted_operator"] = (
                    operator.name if operator is not None else None
                )
                result["failure"] = (
                    "planner exhausted after learned effects were not reached"
                )
                break
            if bool(dones.any().item()):
                result["failure"] = "environment terminated before success"
                break
        else:
            result["failure"] = f"exceeded {max_steps} execution steps"
        result["operator_trace"] = trace
    except (AssertionError, RuntimeError, TaskThenMotionPlanningFailure) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = time.time() - started
        if video is not None:
            outcome = "SUCCESS" if result["success"] else "FAILED"
            if result["success"]:
                video.mark_complete()
            video.hold_final(
                system,
                f"{role.upper()} | episode {episode:02d} | {outcome}",
            )
            video.close()
        logging.info(
            "[%s %02d] success=%s plan_correct=%s steps=%d trace=%s failure=%s",
            role,
            episode,
            result["success"],
            result["plan_correct"],
            result["steps"],
            result["operator_trace"],
            result["failure"],
        )
        print(
            f"[{role} {episode:02d}] success={result['success']} "
            f"plan_correct={result['plan_correct']} steps={result['steps']} "
            f"trace={result['operator_trace']} failure={result['failure']}",
            flush=True,
        )
    return result
