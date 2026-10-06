"""Pipeline builder for different training and evaluation strategies. This module
contains the pipeline builders for different training and evaluation strategies.

Currently, it supports:
- pure TAMP
- pure RL
- SLAP (shortcut learning and planning)
"""

from typing import Callable, Dict

from skill_refactor.pipelines.eval_pure_tamp import evaluate_pure_tamp
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.pipelines.train_val_bc import train_and_evaluate_bc
from skill_refactor.pipelines.train_val_lifelong_ref import (
    train_and_evaluate_ll_ref,
)
from skill_refactor.pipelines.train_val_pure_rl import train_and_evaluate_pure_rl

PIPELINE_BUILDER: Dict[str, Callable[..., Metrics]] = {
    "pure_rl": train_and_evaluate_pure_rl,
    "pure_tamp": evaluate_pure_tamp,
    "rl_planning_states": train_and_evaluate_ll_ref,
    "bc": train_and_evaluate_bc,
}
