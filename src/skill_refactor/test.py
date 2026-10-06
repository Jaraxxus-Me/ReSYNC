"""Standalone test script for lifelong learning approaches.

Example usage (run from project root):
    # Single scenario test
    python -m skill_refactor.test --seed 0 --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml --scenario 1

    # Alternative: direct execution from root
    python src/skill_refactor/test.py --seed 0 --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml --scenario 1

    # Multi-scenario test
    python -m skill_refactor.test --seed 0 --lll_config config/lifelong_learning/cluttered_drawer_sc12_2.yaml --scenario 12_2

    # Multiple seeds
    python -m skill_refactor.test --seeds 0,1,2 --lll_config config/lifelong_learning/cluttered_drawer_sc1.yaml --scenario 1
"""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.args import get_git_commit_hash, reset_config, update_config
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
    ClutteredDrawerRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import ManiSkillsRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.ttmp import TaskThenMotionPlanningFailure


def get_project_root() -> Path:
    """Get the project root directory (where the repo is).

    Assumes the script is in src/skill_refactor/test.py
    """
    # Get the directory of this file
    current_file = Path(__file__).resolve()
    # Go up two levels: test.py -> skill_refactor -> src -> root
    return current_file.parent.parent.parent


def create_test_arg_parser() -> ArgumentParser:
    """Create argument parser for test script."""
    parser = ArgumentParser(description="Standalone test for lifelong learning")

    # Required arguments
    parser.add_argument(
        "--lll_config",
        type=str,
        required=True,
        help="Path to lifelong learning YAML config file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., '1', '12_2', '123_3')",
    )

    # Seed arguments (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seed",
        type=int,
        help="Single seed to test",
    )
    seed_group.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '0,1,2')",
    )

    # Test parameters
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per configuration (default: 50)",
    )
    parser.add_argument(
        "--max_env_steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )

    # Recording parameters
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save videos of episodes (only for first seed)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory to save videos (default: videos)",
    )

    # Logging parameters
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Log file path (default: logs/test_sc{scenario}_seed{seed}.log)",
    )
    parser.add_argument(
        "--debug_log",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Enable debug logging",
    )

    # Environment parameters
    parser.add_argument(
        "--control_mode",
        type=str,
        default="pd_joint_delta_pos",
        help="Control mode (default: pd_joint_delta_pos)",
    )
    parser.add_argument(
        "--delta_finger_control",
        action="store_true",
        help="Enable delta finger control",
    )
    parser.add_argument(
        "--force_skip_pred_learning",
        action="store_true",
        default=True,
        help="Skip predicate learning (default: True)",
    )

    return parser


def get_eval_configs_for_scenario(scenario: str) -> List[Dict[str, Any]]:
    """Get evaluation configurations based on scenario."""
    if scenario == "1":
        # Single obstacle (drawer) - test both configurations
        return [
            {
                "name": "1_b",
                "scenario": "1",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
            },
            {
                "name": "1_g",
                "scenario": "1",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
            },
        ]
    elif scenario == "1,2":
        # Two obstacles (drawer + block) - test 6 configurations
        return [
            {
                "name": "1_b_2_b",
                "scenario": "1,2",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
            {
                "name": "1_b_2_g",
                "scenario": "1,2",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": True,
                "block_blocking_stacking": False,
            },
            {
                "name": "1_g_2_b",
                "scenario": "1,2",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
            {
                "name": "1_g_2_g",
                "scenario": "1,2",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": True,
                "block_blocking_stacking": False,
            },
            {
                "name": "1_g",
                "scenario": "1",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
            {
                "name": "1_b",
                "scenario": "1",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
        ]
    elif scenario == "1,2,3":
        # Three obstacles (drawer + block + wall) - test 12 configurations
        return [
            # Three obstacles
            {
                "name": "1_g_2_g_3_b",
                "scenario": "1,2,3",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": True,
                "block_blocking_stacking": False,
                "wall_blocking_grasp": False,
                "wall_blocking_stacking": True,
            },
            {
                "name": "1_b_2_b_3_g",
                "scenario": "1,2,3",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
                "wall_blocking_grasp": True,
                "wall_blocking_stacking": False,
            },
            # Two obstacles (various combinations)
            {
                "name": "1_b_2_g",
                "scenario": "1,2",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": True,
                "block_blocking_stacking": False,
            },
            {
                "name": "1_g_2_b",
                "scenario": "1,2",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
            {
                "name": "1_g_2_g",
                "scenario": "1,2",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "block_blocking_grasp": True,
                "block_blocking_stacking": False,
            },
            {
                "name": "1_b_2_b",
                "scenario": "1,2",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "block_blocking_grasp": False,
                "block_blocking_stacking": True,
            },
            {
                "name": "1_b_3_g",
                "scenario": "1,3",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "wall_blocking_grasp": True,
                "wall_blocking_stacking": False,
            },
            {
                "name": "1_g_3_b",
                "scenario": "1,3",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "wall_blocking_grasp": False,
                "wall_blocking_stacking": True,
            },
            {
                "name": "1_g_3_g",
                "scenario": "1,3",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
                "wall_blocking_grasp": True,
                "wall_blocking_stacking": False,
            },
            {
                "name": "1_b_3_b",
                "scenario": "1,3",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
                "wall_blocking_grasp": False,
                "wall_blocking_stacking": True,
            },
            # Single obstacle
            {
                "name": "1_g",
                "scenario": "1",
                "drawer_blocking_grasp": True,
                "drawer_blocking_stacking": False,
            },
            {
                "name": "1_b",
                "scenario": "1",
                "drawer_blocking_grasp": False,
                "drawer_blocking_stacking": True,
            },
        ]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def run_single_scenario_test(seed: int, args: Dict[str, Any]) -> Dict[str, float]:
    """Run test for a single scenario (e.g., sc1).

    Returns:
        Dictionary mapping config names to success rates.
    """
    scenario = args["scenario"]
    lll_config_path = args["lll_config"]

    logging.info(f"\n{'='*80}")
    logging.info(f"Testing Scenario {scenario} with Seed {seed}")
    logging.info(f"{'='*80}\n")

    # Load lifelong learning config
    with open(lll_config_path, "rb") as f:
        lll_config_data = yaml.safe_load(f)

    # Get scenario info from YAML
    scenario_info = lll_config_data["scenarios"][int(scenario)]

    # Update config with scenario-specific settings
    world_setting = scenario_info.get("planner_learning_cfg_settings", {})
    update_config(world_setting)

    # Create TAMP system and approach
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=seed)

    # Update learning info and domain knowledge
    approach.update_learning_info(int(scenario), scenario_info)
    approach.update_domain_knowledge(scenario_info)

    # Get evaluation configurations for this scenario
    eval_configs = get_eval_configs_for_scenario(scenario)

    # Run evaluations
    results = {}
    for eval_config in eval_configs:
        success_rate = run_evaluation(eval_config, seed, scenario, approach, args)
        results[eval_config["name"]] = success_rate

    return results


def run_multi_scenario_test(seed: int, args: Dict[str, Any]) -> Dict[str, float]:
    """Run test for multiple scenarios (e.g., sc12_2, sc123_3).

    Returns:
        Dictionary mapping config names to success rates.
    """
    scenario = args["scenario"]
    lll_config_path = args["lll_config"]

    logging.info(f"\n{'='*80}")
    logging.info(f"Testing Multi-Scenario {scenario} with Seed {seed}")
    logging.info(f"{'='*80}\n")

    # Load lifelong learning config
    with open(lll_config_path, "rb") as f:
        lll_config_data = yaml.safe_load(f)

    # Create initial TAMP system and approach
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=seed)

    # Sequentially update through all scenarios
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        logging.info(f"\nUpdating with scenario {scenario_id}...")

        # Update CFG with planner_learning_cfg_settings
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)

        # Create new TAMP system with updated settings
        latest_tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )

        # Update approach with new scenario
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        approach.update_domain_knowledge(scenario_info)

    # Get evaluation configurations for this scenario
    eval_configs = get_eval_configs_for_scenario(scenario)

    # Run evaluations
    results = {}
    for eval_config in eval_configs:
        success_rate = run_evaluation(eval_config, seed, scenario, approach, args)
        results[eval_config["name"]] = success_rate

    return results


def run_evaluation(
    eval_config: Dict[str, Any],
    seed: int,
    scenario: str,
    approach: LifelongRefApproach,
    args: Dict[str, Any],
) -> float:
    """Run evaluation for a single configuration.

    Returns:
        Success rate as a float between 0 and 1.
    """
    eval_name = eval_config["name"]

    logging.info(f"\n{'='*80}")
    logging.info(f"Evaluating Configuration: {eval_name}")
    logging.info(f"{'='*80}\n")

    # Update config for this evaluation
    test_config = {
        "num_envs": args["num_envs"],
        "num_eval_episodes": args["num_eval_episodes"],
        "max_env_steps": args["max_env_steps"],
    }

    # Add environment-specific blocking settings
    for key in eval_config:
        if key not in ["name", "scenario"]:
            test_config[key] = eval_config[key]

    # Update scenario if specified in eval_config
    if "scenario" in eval_config:
        test_config["scenario"] = eval_config["scenario"]

    update_config(test_config)

    # Create TAMP system for evaluation
    tamp_system = ClutteredDrawerRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Setup video recording if requested
    envs: Any
    if args.get("save_video", False) or seed == 0:
        video_folder = Path(args["video_dir"]) / f"sc{scenario}_seed{seed}_{eval_name}"
        envs = ManiSkillsRecordVideo(
            tamp_system.env,
            output_dir=video_folder,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=CFG.max_env_steps,
            video_fps=30,
        )
        logging.info(f"Recording videos to {video_folder}")
    else:
        envs = tamp_system.env

    # Run episodes
    success = []
    rnd_seed = list(range(0, CFG.num_eval_episodes * 20, 10))

    for epi in range(CFG.num_eval_episodes):
        reset_options: dict = {}
        obs, info = envs.reset(
            options=reset_options, seed=rnd_seed[epi] + seed
        )  # type: ignore[no-untyped-call]

        # Try to reset approach (may fail with planning failure)
        try:
            step_result = approach.reset(obs, info)
        except TaskThenMotionPlanningFailure as e:
            logging.info(
                f"Episode {epi} failed during reset with TaskThenMotionPlanningFailure: {e}"
            )
            success.append(False)
            continue

        # Initialize episode state
        total_reward = torch.tensor(
            [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
        )
        epi_success = torch.zeros(CFG.num_envs, dtype=torch.bool, device=envs.device)

        # Run episode steps
        for step in range(CFG.max_env_steps + 1):
            obs, _, _, _, info = envs.step(step_result.action)
            bool_success = torch.tensor(
                info["success"], dtype=torch.bool, device=epi_success.device
            )
            epi_success |= bool_success

            # Check for early success
            if epi_success.all():
                logging.info(f"Episode {epi} succeeded at step {step}.")
                break

            # Check for skill exhaustion
            if approach.exhausted.all():
                logging.info(f"Episode {epi} exhausted skill attempts at step {step}.")
                break

            # Get next action
            step_result = approach.step(obs, total_reward, False, False, info)  # type: ignore[arg-type]

        success.extend(epi_success.cpu().numpy().tolist())
        logging.info(f"Episode {epi} final success: {epi_success}")

    # Calculate success rate
    success_rate = sum(success) / len(success)

    logging.info(f"\n{'='*80}")
    logging.info(f"Configuration {eval_name} - Success Rate: {success_rate:.2%}")
    logging.info(f"{'='*80}\n")

    envs.close()  # type: ignore[no-untyped-call]

    return success_rate


def main() -> None:
    """Main entry point for standalone test script."""
    # Change to project root directory to ensure relative paths work correctly
    project_root = get_project_root()
    os.chdir(project_root)

    # Parse arguments
    parser = create_test_arg_parser()
    args = parser.parse_args()

    # Parse seeds
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Resolve lll_config path relative to project root
    lll_config_path = Path(args.lll_config)
    if not lll_config_path.is_absolute():
        lll_config_path = project_root / lll_config_path
    args.lll_config = str(lll_config_path)

    # Setup base configuration
    base_config = {
        "num_envs": args.num_envs,
        "lll_config": args.lll_config,
        "control_mode": args.control_mode,
        "delta_finger_control": args.delta_finger_control,
        "force_skip_pred_learning": args.force_skip_pred_learning,
        "dreaming_noise_base_var": 0.0,
        "max_env_steps": args.max_env_steps,
        "num_eval_episodes": args.num_eval_episodes,
        "scenario": args.scenario,
    }

    # Register environments
    register_all_environments()

    # Run tests for each seed
    all_results = {}
    for seed in seeds:
        # Setup logging for this seed
        if args.log_file:
            log_file = args.log_file
        else:
            log_file = f"logs/test_sc{args.scenario}_seed{seed}.log"

        # Resolve log file path relative to project root
        log_file_path = Path(log_file)
        if not log_file_path.is_absolute():
            log_file_path = project_root / log_file_path
        log_file = str(log_file_path)

        base_config["seed"] = seed
        base_config["log_file"] = log_file
        base_config["loglevel"] = args.loglevel

        # Reset config for this seed
        reset_config(base_config)

        # Setup logging handlers
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        if CFG.log_file:
            Path(CFG.log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(CFG.log_file, mode="w"))

        logging.basicConfig(
            level=CFG.loglevel,
            format="%(message)s",
            handlers=handlers,
            force=True,
        )
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

        logging.info(f"Logging to {CFG.log_file}")
        logging.info(f"Git commit hash: {get_git_commit_hash()}")
        logging.info(f"Configuration:\n{CFG}")

        # Determine if single or multi-scenario test
        test_args = {
            "scenario": args.scenario,
            "lll_config": args.lll_config,
            "num_eval_episodes": args.num_eval_episodes,
            "max_env_steps": args.max_env_steps,
            "num_envs": args.num_envs,
            "save_video": args.save_video,
            "video_dir": args.video_dir,
        }

        if args.scenario in ["1", "2", "3"]:
            # Single scenario test
            results = run_single_scenario_test(seed, test_args)
        else:
            # Multi-scenario test (e.g., "12_2", "123_3")
            results = run_multi_scenario_test(seed, test_args)

        all_results[seed] = results

    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("FINAL RESULTS SUMMARY")
    logging.info(f"{'='*80}\n")

    for seed, results in all_results.items():
        logging.info(f"\nSeed {seed}:")
        for config_name, success_rate in results.items():
            logging.info(f"  {config_name}: {success_rate:.2%}")

    # Calculate average across seeds if multiple seeds
    if len(seeds) > 1:
        logging.info(f"\nAverage across seeds:")
        all_config_names = list(all_results[seeds[0]].keys())
        for config_name in all_config_names:
            avg_success = sum(all_results[s][config_name] for s in seeds) / len(seeds)
            logging.info(f"  {config_name}: {avg_success:.2%}")

    logging.info(f"\n{'='*80}")
    logging.info("Test completed successfully!")
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as _err:
        logging.exception("test.py crashed")
        raise _err
