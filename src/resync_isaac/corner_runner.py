"""Cornered stacking: native RL recovery with a learned-domain Pick/Place plan."""

import json
import time
from argparse import Namespace

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from resync_isaac.corner_stacking.corner_stacking import CornerStackingRLTAMPSystem
from resync_isaac.corner_stacking.corner_stacking_env import CUBE_TASK_ID
from resync_isaac.corner_stacking.utils import load_block_box_rl_policy
from resync_isaac.domain import load_domain
from resync_isaac.paths import ROOT
from resync_isaac.video import PlanVideoRecorder
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


def run(args: Namespace) -> None:
    system = CornerStackingRLTAMPSystem.create_default(
        seed=args.seed, render_mode="rgb_array" if args.record_video else None
    )
    env = RslRlVecEnvWrapper(system.env, clip_actions=None)
    try:
        bundle = args.asset_root / "corner_stacking"
        agent_cfg = load_cfg_from_registry(CUBE_TASK_ID, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        policy, _ = load_block_box_rl_policy(
            system.env, agent_cfg, str(bundle / "policy.pt")
        )
        system.rl_skill.set_policy(policy)
        operators = load_domain(
            system,
            bundle / "predicates",
            ROOT / "config/inference/isaac/corner_stacking_predicates.yaml",
        )
        planner = TaskThenMotionPlanner(
            types=system.types,
            predicates=system.perceiver.predicates_container.as_set(),
            perceiver=system.perceiver,
            operators=operators,
            skills=system.skills,
            fallback_action=torch.zeros(env.num_actions).numpy(),
            normalize_action=False,
            arm_action_low=torch.full((env.num_actions,), -1.0),
            arm_action_high=torch.ones(env.num_actions),
            planner_id="pyperplan",
            domain_name="CorneredStacking-demo",
        )
        objects = system.perceiver.objects.as_dict()
        goal_names = (
            ("block0", "block1") if args.case == "1_g" else ("block1", "block0")
        )
        goal = system.perceiver.predicates_container["On"](
            [objects[n] for n in goal_names]
        )
        results = []
        for episode in range(args.episodes):
            video = None
            result = {
                "episode": episode,
                "case": args.case,
                "goal": list(goal_names),
                "success": False,
                "skills": [],
            }
            started = time.monotonic()
            try:
                obs_td, extras = env.reset()
                obs = obs_td["policy"]
                planner.reset(obs, extras if isinstance(extras, dict) else {})
                plan = list(planner._current_task_plan)
                result["learned_plan"] = [
                    (op.name, [o.name for o in op.parameters]) for op in plan
                ]
                # Source hybrid demo explicitly performs recovery even when the
                # learned predicates already admit a Pick/Place suffix.
                if not any(op.name == "RL" for op in plan):
                    op = next(op for op in operators if op.name == "RL")
                    plan.insert(
                        0,
                        op.ground(
                            (objects["robot"], objects["block0"], objects["box"])
                        ),
                    )
                    result["manual_recovery_prefix"] = True
                else:
                    result["manual_recovery_prefix"] = False
                result["execution_plan"] = [
                    (op.name, [o.name for o in op.parameters]) for op in plan
                ]
                if args.record_video:
                    video = PlanVideoRecorder(
                        args.output / f"corner_{args.case}_{episode:03d}.mp4",
                        result["execution_plan"],
                        fps=10,
                        stride=1,
                    )
                    video.capture(system, None, f"{args.case} INIT", force=True)
                total_steps = 0
                for op in plan:
                    skill = next(s for s in system.skills if s.can_execute(op))
                    skill.reset(op, obs)
                    terminated = False
                    budget = min(
                        1500 if op.name == "RL" else 600, args.max_steps - total_steps
                    )
                    print(
                        f"[{args.case}] {op.name}{tuple(o.name for o in op.parameters)}",
                        flush=True,
                    )
                    for _ in range(budget):
                        with torch.inference_mode():
                            action = skill.get_action(obs)
                            if action.shape[-1] != env.num_actions:
                                raise ValueError(
                                    f"Skill action width {action.shape[-1]} != {env.num_actions}"
                                )
                            obs_td, _, dones, _ = env.step(action)
                            obs = obs_td["policy"]
                        total_steps += 1
                        if video:
                            video.capture(
                                system, op.name, f"{args.case} | step {total_steps:04d}"
                            )
                        if bool(dones.any().item()):
                            raise RuntimeError(
                                "Environment ended during skill execution"
                            )
                        if bool(
                            skill.terminate_with_objects(op.parameters, obs)
                            .all()
                            .item()
                        ):
                            terminated = True
                            break
                    result["skills"].append(
                        {
                            "name": op.name,
                            "objects": [o.name for o in op.parameters],
                            "terminated": terminated,
                        }
                    )
                    if not terminated:
                        raise RuntimeError(
                            f"{op.name} did not terminate within {budget} steps"
                        )
                result["steps"] = total_steps
                result["success"] = goal in system.perceiver.step(obs)[0]
                if not result["success"]:
                    result["failure"] = "Skill sequence ended without the stacking goal"
            except (RuntimeError, AssertionError) as exc:
                result["failure"] = str(exc)
            finally:
                result["elapsed_seconds"] = time.monotonic() - started
                if video:
                    if result["success"]:
                        video.mark_complete()
                    video.hold_final(
                        system,
                        f'{args.case} | {"SUCCESS" if result["success"] else "FAILED"}',
                    )
                    video.close()
                results.append(result)
                (args.output / "episodes.json").write_text(
                    json.dumps(
                        {
                            "seed": args.seed,
                            "execution": "native recovery with learned predicates/operators",
                            "episodes": results,
                        },
                        indent=2,
                    )
                )
                print(json.dumps(result), flush=True)
    finally:
        env.close()
