"""Run stage-1 inference using the unchanged main-branch implementation.

The configurations, episode seeds, action conversion, and stopping conditions
follow tests/{domain}/approaches/test_lifelong_ref.py. This entry point exposes
the episode count, avoids mandatory video recording, and writes structured results.
No policies or predicates are trained here.
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", choices=["blocked_stacking", "cluttered_drawer"], required=True
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = args.output or Path(f"results/sanity_{args.env}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output.with_suffix(".log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
    )

    import torch
    import yaml

    from skill_refactor import register_all_environments
    from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
    from skill_refactor.args import reset_config, update_config
    from skill_refactor.settings import CFG
    from skill_refactor.utils.ttmp import TaskThenMotionPlanningFailure

    if not torch.cuda.is_available():
        raise SystemExit(
            "The released configurations require an NVIDIA GPU and CUDA-enabled PyTorch."
        )
    torch.set_num_threads(4)
    drawer = args.env == "cluttered_drawer"
    if drawer:
        from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
            ClutteredDrawerRLTAMPSystem,
        )

        system_cls = ClutteredDrawerRLTAMPSystem
        predicate_dir = "c_drawer_sc1_pred_nets_seed0"
        policy_path = (
            "trained_policies/cluttered_drawer/skill_1231_sc1_seed0/best_ppo_ckpt.pt"
        )
        blocking_prefix = "drawer"
        recovery_name = "Pull"
    else:
        from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
            BlockedStackingRLTAMPSystem,
        )

        system_cls = BlockedStackingRLTAMPSystem
        predicate_dir = "skill_1127_sc1_pred_nets_seed0"
        policy_path = (
            "trained_policies/b_stacking/skill_1127_sc1_seed0/best_ppo_ckpt.pt"
        )
        blocking_prefix = "obstruction1"
        recovery_name = "Punch"

    config_path = f"config/lifelong_learning/{args.env}_sc1.yaml"
    stage = yaml.safe_load(Path(config_path).read_text())["scenarios"][1]
    dataset = (
        Path(stage["planner_learning_cfg_settings"]["planner_dataset_path"]) / "seed_0"
    )
    operator_path = Path(predicate_dir) / "invented_predicates_operators_sc1.json"
    required = [
        Path(policy_path),
        operator_path,
        dataset / "trajectories.pkl",
        dataset / "lifted_operator_plans.pkl",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(
            "Run scripts/download_release_assets.py first. Missing:\n"
            + "\n".join(missing)
        )

    reset_config(
        {
            "seed": 0,
            "env": args.env,
            "num_envs": 1,
            "scenario": "1",
            "debug_env": False,
            "delta_finger_control": not drawer,
            "lll_config": config_path,
            "control_mode": "pd_joint_delta_pos",
            "force_skip_pred_learning": True,
            "pred_net_save_dir": predicate_dir,
            "pre_trained_policy_path": policy_path,
            "max_env_steps": 250,
        }
    )
    register_all_environments()
    update_config(stage["planner_learning_cfg_settings"])
    # Explicitly keep the original trained models selected above.
    update_config(
        {
            "force_skip_pred_learning": True,
            "pred_net_save_dir": predicate_dir,
            "pre_trained_policy_path": policy_path,
        }
    )
    base = system_cls.create_default(render_mode="rgb_array", seed=42)
    approach = LifelongRefApproach(base, seed=0)
    approach.update_learning_info(1, stage)
    print(
        f"Loading {args.env}: policy, predicates, operators, and planner data...",
        flush=True,
    )
    started = time.time()
    approach.update_domain_knowledge(stage)
    operator_names = sorted(op.name for op in approach.operators)
    if recovery_name not in operator_names:
        raise RuntimeError(f"Learned recovery operator {recovery_name} was not loaded")
    report = {
        "environment": args.env,
        "seed": 0,
        "source_commit": "49f8d12a072166e89ef45a0eacc1d20adc8bdefa",
        "policy": policy_path,
        "predicates": predicate_dir,
        "config": config_path,
        "max_env_steps": CFG.max_env_steps,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "loaded_operators": operator_names,
        "loaded_predicate_count": len(approach.perceiver.predicates_container.as_set()),
        "configurations": {},
    }
    for name, grasp, stacking in [("1_b", False, True), ("1_g", True, False)]:
        update_config(
            {
                "num_envs": 1,
                "num_eval_episodes": args.episodes,
                f"{blocking_prefix}_blocking_grasp": grasp,
                f"{blocking_prefix}_blocking_stacking": stacking,
            }
        )
        system = system_cls.create_default(render_mode="rgb_array", seed=42)
        env = system.env
        episodes = []
        counts = Counter()
        try:
            for episode in range(args.episodes):
                obs, info = env.reset(seed=10 * episode, options={})
                success = False
                steps = 0
                error = None
                sequence = []
                try:
                    result = approach.reset(obs, info)
                    reward = torch.zeros(
                        CFG.num_envs, dtype=torch.float32, device=env.device
                    )
                    for step in range(CFG.max_env_steps + 1):
                        if result.op is not None:
                            op_name = result.op.name
                            counts[op_name] += 1
                            if not sequence or sequence[-1] != op_name:
                                sequence.append(op_name)
                        action = (
                            result.action if drawer else result.action.to(torch.float64)
                        )
                        obs, _, _, _, info = env.step(action)
                        steps = step + 1
                        success = bool(torch.as_tensor(info["success"]).all().item())
                        if success or (drawer and approach.exhausted.all()):
                            break
                        result = approach.step(obs, reward, False, False, info)
                except TaskThenMotionPlanningFailure as exc:
                    error = str(exc)
                episodes.append(
                    {
                        "episode": episode,
                        "reset_seed": 10 * episode,
                        "success": success,
                        "steps": steps,
                        "operators": sequence,
                        "planning_failure": error,
                    }
                )
                print(
                    f"{args.env} {name}: episode {episode + 1}/{args.episodes}, success={success}, steps={steps}",
                    flush=True,
                )
                report["configurations"][name] = {
                    "successes": sum(e["success"] for e in episodes),
                    "episodes": len(episodes),
                    "success_rate": sum(e["success"] for e in episodes) / len(episodes),
                    "operator_step_counts": dict(counts),
                    "rollouts": episodes,
                }
                report["elapsed_seconds"] = time.time() - started
                output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.close()
    base.env.close()
    for name, result in report["configurations"].items():
        print(
            f"{name}: {result['successes']}/{result['episodes']} ({result['success_rate']:.1%})"
        )
    print(f"Results: {output}")
    if any(
        r["successes"] == 0 or r["operator_step_counts"].get(recovery_name, 0) == 0
        for r in report["configurations"].values()
    ):
        raise SystemExit(
            "Sanity check failed: each configuration must succeed and execute the learned recovery skill."
        )


if __name__ == "__main__":
    main()
