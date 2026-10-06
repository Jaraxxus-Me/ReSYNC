"""Metrics module for training and evaluation."""

from dataclasses import dataclass


@dataclass
class Metrics:
    """Training and evaluation metrics."""

    success_rate: float
    avg_episode_length: float
    avg_reward: float
    training_time: float = 0.0
    total_time: float = 0.0
