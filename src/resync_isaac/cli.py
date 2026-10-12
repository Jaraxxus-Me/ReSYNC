"""Simulator lifecycle and argument parsing; safe to import without Isaac."""

import argparse
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env", choices=["flipped_stacking", "corner_stacking"], required=True
    )
    p.add_argument("--num-blocks", type=int, choices=[2, 3, 4], default=4)
    p.add_argument(
        "--flipped",
        nargs=2,
        metavar=("BLOCK", "BLOCK"),
        help="Two distinct roles: top, mid1, mid2, bottom; omitted = random pair",
    )
    p.add_argument("--case", choices=["1_b", "1_g"], default="1_b")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        choices=["cuda:0", "cpu"],
        default="cuda:0",
        help="Physics/policy device; video still requires an NVIDIA GPU",
    )
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/isaac_demos")
    p.add_argument("--asset-root", type=Path, default=ROOT / "inference_assets/isaac")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    roles = {
        2: ["top", "bottom"],
        3: ["top", "mid1", "bottom"],
        4: ["top", "mid1", "mid2", "bottom"],
    }[args.num_blocks]
    if args.flipped and (
        len(set(args.flipped)) != 2 or not set(args.flipped) <= set(roles)
    ):
        p.error(f"--flipped requires two distinct roles from {roles}")
    if args.episodes < 1 or args.max_steps < 1:
        p.error("episodes and max-steps must be positive")
    args.asset_root = args.asset_root.resolve()
    args.output = args.output.resolve()
    bundle = args.asset_root / args.env
    for path in [
        bundle / "policy.pt",
        bundle / "predicates/invented_predicates_operators_sc1.json",
    ]:
        if not path.is_file():
            p.error(f"Missing {path}; download the Isaac demo assets first")
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ["RESYNC_ISAAC_ASSETS"] = str(args.asset_root)
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=args.headless,
        enable_cameras=args.record_video,
        width=640,
        height=480,
        renderer="RayTracedLighting",
        fast_shutdown=True,
        device=args.device,
    ).app
    try:
        logging.basicConfig(
            filename=args.output / "demo.log", level=logging.INFO, force=True
        )
        from resync_isaac.settings import configure

        configure(
            device=args.device,
            seed=args.seed,
            num_envs=1,
            num_of_blocks=args.num_blocks,
            fstack_num_envs=1,
            c_stack_num_envs=1,
            fstack_max_steps=args.max_steps,
            c_stack_max_steps=args.max_steps,
            normalize_action=False,
            fstack_flipped_cube=(
                ",".join(args.flipped) if args.flipped else "random_pair"
            ),
            fstack_checkpoint=str(bundle / "policy.pt"),
            c_stack_checkpoint=str(bundle / "policy.pt"),
            c_stack_goal_0on1=args.case == "1_g",
            c_stack_reset_from_post_rl=False,
        )
        from resync_isaac.provenance import write_provenance

        write_provenance(args)
        import torch

        torch.set_num_threads(4)
        if args.env == "flipped_stacking":
            run_flipped(args)
        else:
            from resync_isaac.corner_runner import run

            run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        # Kit's fast shutdown exits with code zero even while unwinding an
        # exception. Preserve a failing CLI exit code after printing the error.
        import sys

        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


def run_flipped(args: argparse.Namespace) -> None:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from resync_isaac.corner_stacking.corner_stacking_env import CUBE_TASK_ID
    from resync_isaac.domain import load_domain, repair_flipped_composition
    from resync_isaac.flipped_runner import _evaluate_episode, _make_planner
    from resync_isaac.flipped_stacking.flipped_stacking import (
        FlippedStackingRLTAMPSystem,
    )
    from resync_isaac.flipped_stacking.utils import load_flip_rl_policy

    system = FlippedStackingRLTAMPSystem.create_default(
        seed=args.seed, render_mode="rgb_array" if args.record_video else None
    )
    env = RslRlVecEnvWrapper(system.env, clip_actions=None)
    try:
        bundle = args.asset_root / args.env
        agent_cfg = load_cfg_from_registry(CUBE_TASK_ID, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        policy, _ = load_flip_rl_policy(
            system.env, agent_cfg, str(bundle / "policy.pt")
        )
        system.flip_skill.set_policy(policy)
        ops = repair_flipped_composition(
            load_domain(
                system,
                bundle / "predicates",
                ROOT / "config/inference/isaac/flipped_stacking_predicates.yaml",
            )
        )
        planner = _make_planner(system, env, ops)
        results = []
        role = ",".join(args.flipped) if args.flipped else "random_pair"
        for episode in range(args.episodes):
            result = _evaluate_episode(
                system,
                env,
                planner,
                role,
                episode,
                args.max_steps,
                (
                    args.output / f"flipped_{episode:03d}.mp4"
                    if args.record_video
                    else None
                ),
                video_fps=10,
                video_stride=1,
                video_frame_scale=1.0,
            )
            results.append(result)
            (args.output / "episodes.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "num_blocks": args.num_blocks,
                        "flipped": role,
                        "execution": "learned plan with composition repairs and termination-only execution",
                        "episodes": results,
                    },
                    indent=2,
                )
            )
    finally:
        env.close()
