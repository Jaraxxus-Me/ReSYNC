"""Evaluate inference assets on the ReSYNC experiment scenarios."""

import argparse
import hashlib
import importlib
import json
import logging
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    specs = json.loads(Path("config/inference/experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=specs)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--threads", type=int, default=4, help="PyTorch CPU thread count"
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--settings",
        type=Path,
        help="JSON configuration overrides",
    )
    args = parser.parse_args()
    overrides = json.loads(args.settings.read_text()) if args.settings else {}
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    spec = specs[args.experiment]
    if spec["environment"] in ("blocked_stacking", "icy_transport"):
        overrides.setdefault("device", "cpu")
    if spec["environment"] == "blocked_stacking":
        overrides.setdefault("blocked_stacking_legacy_skill_timing", True)
    if spec["environment"] == "cluttered_drawer" and spec["stage"] == 1:
        overrides.setdefault("c_drawer_legacy_physics", True)
        overrides.setdefault("c_drawer_place_height_tol", 0.09)
        overrides.setdefault("c_drawer_block_dz", 0.04)
    if spec["environment"] == "icy_transport":
        overrides.setdefault("icy_legacy_plan_order", True)
        overrides.setdefault("icy_reset_collision_history", False)
        overrides.setdefault("icy_inhand_fy_penalty", 0.0)
    cases = [c for c in spec["cases"] if args.cases is None or c["name"] in args.cases]
    if not cases or (args.cases and set(args.cases) - {c["name"] for c in cases}):
        parser.error("Unknown or empty case selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.output.with_suffix(".log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
    )
    import torch
    import yaml
    from skill_refactor import PACKAGE_ASSET_DIR, register_all_environments
    from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
    from skill_refactor.args import reset_config, update_config
    from skill_refactor.settings import CFG
    from skill_refactor.utils.ttmp import TaskThenMotionPlanningFailure

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    env_name, stage = spec["environment"], spec["stage"]
    physics3d = env_name in ("cluttered_drawer", "cluttered_room")
    class_names = {
        "blocked_stacking": "BlockedStacking",
        "icy_transport": "IcyTransport",
        "cluttered_drawer": "ClutteredDrawer",
        "cluttered_room": "ClutteredRoom",
    }
    module = importlib.import_module(f"skill_refactor.benchmarks.{env_name}.{env_name}")
    # Hash the source files used by this process before initialization.
    package_root = Path(module.__file__).resolve().parents[2]
    source_sha256 = {
        str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    system_cls = getattr(module, class_names[env_name] + "RLTAMPSystem")
    config_path = Path(f"config/inference/{env_name}_stage{stage}.yaml")
    stages = yaml.safe_load(config_path.read_text())["scenarios"]
    if overrides.get("c_drawer_legacy_physics"):
        legacy_obstacle = Path(PACKAGE_ASSET_DIR) / "obstacle_legacy.urdf"
        if not legacy_obstacle.is_file():
            raise FileNotFoundError(
                f"Missing {legacy_obstacle}; run scripts/download_release_assets.py --domains simulation_assets"
            )
    model_files = []
    for phase, data in stages.items():
        settings = data["planner_learning_cfg_settings"]
        policy = Path(settings["pre_trained_policy_path"])
        predicate_dir = Path(settings["pred_net_save_dir"])
        operator = predicate_dir / f"invented_predicates_operators_sc{phase}.json"
        dataset = Path(settings["inference_dataset_path"])
        for required in (
            policy,
            operator,
            dataset / "trajectories.pkl",
            dataset / "lifted_operator_plans.pkl",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        model_files.extend([policy, *predicate_dir.glob("*")])
    reset_config(
        {
            "env": env_name,
            "seed": 0,
            "num_envs": 1,
            "scenario": "1",
            "debug_env": False,
            "delta_finger_control": not physics3d,
            "lll_config": str(config_path),
            "force_skip_pred_learning": True,
            "control_mode": (
                "force_torque" if env_name == "icy_transport" else "pd_joint_delta_pos"
            ),
        }
    )
    if env_name == "cluttered_drawer":
        # Configure scene noise before constructing the base system.
        update_config({"dreaming_noise_base_var": 0.0})
    update_config(overrides)
    register_all_environments()
    systems = []
    if stage == 1:
        update_config(stages[1]["planner_learning_cfg_settings"])
    base = system_cls.create_default(render_mode="rgb_array", seed=42)
    systems.append(base)
    approach = LifelongRefApproach(base, seed=0)
    for phase, data in stages.items():
        update_config(data["planner_learning_cfg_settings"])
        update_config(overrides)
        latest = (
            base
            if stage == 1
            else system_cls.create_default(render_mode="rgb_array", seed=42)
        )
        if latest is not base:
            systems.append(latest)
        approach.update_learning_info(int(phase), data, latest_tamp_system=latest)
        print(f"Loading {env_name} phase {phase}", flush=True)
        approach.update_domain_knowledge(data)
    horizons = {
        "blocked_stacking": {2: 350, 3: 500},
        "cluttered_drawer": {2: 800},
        "icy_transport": {2: 5000},
        "cluttered_room": {},
    }
    horizon = horizons[env_name].get(stage, CFG.max_env_steps)
    report = {
        "experiment": args.experiment,
        "config_overrides": overrides,
        "evaluation_seed": 0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": str(config_path),
        "max_env_steps": horizon,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cases": cases,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "model_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in set(model_files)
            if p.is_file()
        },
        "configurations": {},
    }
    report["source_package"] = str(package_root)
    report["source_sha256"] = source_sha256
    report["runner_sha256"] = runner_sha256
    report["package_asset_urdf_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(PACKAGE_ASSET_DIR).glob("*.urdf"))
    }
    started = time.time()
    for case in cases:
        update_config(
            {
                **case["settings"],
                "num_envs": 1,
                "num_eval_episodes": args.episodes,
                "max_env_steps": horizon,
            }
        )
        system = system_cls.create_default(render_mode="rgb_array", seed=42)
        episodes, counts = [], Counter()
        try:
            for episode in range(args.episodes):
                obs, info = system.env.reset(seed=10 * episode, options={})
                success, steps, error, sequence = False, 0, None, []
                try:
                    result = approach.reset(obs, info)
                    reward = torch.zeros(1, device=CFG.device)
                    for step in range(horizon + 1):
                        if result.op is not None:
                            name = result.op.name
                            counts[name] += 1
                            if not sequence or sequence[-1] != name:
                                sequence.append(name)
                        action = (
                            result.action
                            if physics3d
                            else result.action.to(torch.float64)
                        )
                        obs, _, _, _, info = system.env.step(action)
                        steps = step + 1
                        success = bool(torch.as_tensor(info["success"]).all())
                        if success or (physics3d and approach.exhausted.all()):
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
                successes = sum(e["success"] for e in episodes)
                report["configurations"][case["name"]] = {
                    "successes": successes,
                    "episodes": len(episodes),
                    "success_rate": successes / len(episodes),
                    "operator_step_counts": dict(counts),
                    "rollouts": episodes,
                }
                report["elapsed_seconds"] = time.time() - started
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{case['name']} {episode+1}/{args.episodes}: {success}; rate={successes/len(episodes):.3f}",
                    flush=True,
                )
        finally:
            system.env.close()
    for system in systems:
        system.env.close()


if __name__ == "__main__":
    main()
