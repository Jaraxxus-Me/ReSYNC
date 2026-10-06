"""Evaluate pure TAMP on a system."""

import logging
import time
from typing import TypeVar

import numpy as np

from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.pipelines.eval_episode import run_evaluation_episode
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.settings import CFG
from skill_refactor.utils.gpu_utils import set_torch_seed

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def evaluate_pure_tamp(
    system: BaseRLTAMPSystem,
    approach_name: str = "pure_tamp",
) -> Metrics:
    """Train and evaluate pure TAMP on a system."""
    logging.info(f"\nInitializing pure TAMP baseline training for {system.name}...")
    seed = CFG.seed
    set_torch_seed(seed)
    start_time = time.time()

    approach = PureTAMPApproach(
        system=system,
        seed=seed,
    )

    # Run evaluation
    logging.info(f"\nEvaluating pure {approach_name} on {system.name}...")
    rewards, lengths, successes = run_evaluation_episode(
        system,
        approach,
        total_episodes=CFG.num_eval_episodes,
    )

    total_time = time.time() - start_time
    return Metrics(
        success_rate=float(sum(successes) / len(successes)),
        avg_episode_length=float(np.mean(lengths)),
        avg_reward=float(np.mean(rewards)),
        training_time=0.0,
        total_time=total_time,
    )
