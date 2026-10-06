"""Main entry point for running approaches in environments.

Example usage with learning NSRTs:
    python predicators/main.py --env stick_button --approach tamp --seed 0
"""

import logging
import os
import sys
from typing import Callable, List

from tabulate import tabulate

from skill_refactor import register_all_environments
from skill_refactor.args import get_git_commit_hash, parse_args, update_config
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.cluttered_drawer.cluttered_drawer import (
    ClutteredDrawerRLTAMPSystem,
)
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.pipelines import PIPELINE_BUILDER
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.settings import CFG


def get_available_system(env_name: str) -> type[BaseRLTAMPSystem]:
    """Get list of available TAMP systems."""
    if env_name == "icy_transport":
        assert CFG.control_mode == "force_torque"
        return IcyTransportRLTAMPSystem
    if env_name == "cluttered_drawer":
        assert not CFG.delta_finger_control
        return ClutteredDrawerRLTAMPSystem
    if env_name == "blocked_stacking":
        assert CFG.delta_finger_control
        return BlockedStackingRLTAMPSystem
    raise ValueError("Unknown system name")


def save_results(results: dict[tuple[str, str], Metrics]) -> None:
    """Save experiment results."""
    table_data = []
    headers = [
        "System",
        "Approach",
        "Success Rate",
        "Avg Length",
        "Avg Reward",
        "Train Time (s)",
        "Total Time (s)",
    ]

    for (system_name, approach_name), metrics in sorted(results.items()):
        table_data.append(
            [
                system_name,
                approach_name,
                f"{metrics.success_rate:.2%}",
                f"{metrics.avg_episode_length:.2f}",
                f"{metrics.avg_reward:.2f}",
                f"{metrics.training_time:.2f}",
                f"{metrics.total_time:.2f}",
            ]
        )

    table = tabulate(table_data, headers=headers, tablefmt="grid")
    os.makedirs(CFG.results_dir, exist_ok=True)
    results_file = CFG.results_dir / "experiment_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(table)

    logging.info(f"\nResults saved to {results_file}")


def main() -> None:
    """Main entry point for running approaches in environments."""
    # Parse & validate args
    args = parse_args()
    update_config(args)
    register_all_environments()
    str_args = " ".join(sys.argv)
    # Log to stderr.
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
    logging.basicConfig(
        level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(CFG)
    logging.info(f"Git commit hash: {get_git_commit_hash()}")

    system_cls = get_available_system(CFG.env)
    results: dict[tuple[str, str], Metrics] = {}

    logging.info(f"\n{'='*20} Testing {CFG.env} {'='*20}")
    logging.info(f"\n{'-'*10} Testing {CFG.approach} {'-'*10}")

    pipeline: Callable[..., Metrics] = PIPELINE_BUILDER[
        CFG.approach.replace("_Loaded", "")
    ]
    metrics = pipeline(system_cls=system_cls, approach_name=CFG.approach)

    results[(CFG.env, CFG.approach)] = metrics
    logging.info(f"\nResults for {CFG.env} with {CFG.approach}:")
    logging.info(f"Success Rate: {metrics.success_rate:.2%}")
    logging.info(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    logging.info(f"Average Reward: {metrics.avg_reward:.2f}")

    # Save results
    save_results(results)
    logging.info("Experiment completed.")


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main.py crashed")
        raise _err
