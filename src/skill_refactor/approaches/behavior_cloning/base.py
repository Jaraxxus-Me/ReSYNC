from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

from relational_structs import Object
from torch import Tensor

from skill_refactor.utils.structs import PlannerDataset


class BCPolicy(ABC):
    """Base class for offline behavior cloning policies (GNN / Transformer)."""

    args: Any

    def __init__(self, seed: int) -> None:
        self._seed = seed

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this BC model requires offline training."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> None:
        """Optional init (e.g., build nets after seeing exemplar dims)."""

    @abstractmethod
    def can_initiate(self, obs: Tensor) -> bool:
        """Whether BC can produce a valid decision in current context."""
        raise NotImplementedError

    @abstractmethod
    def predict_operator(self, graph: Dict, object_to_node: Dict[Object, int]) -> str:
        """Predict operator name from graph.

        Args:
            graph: Graph dict with nodes, edges, globals
            object_to_node: Mapping from objects to node indices

        Returns:
            Tuple of (operator_name, object_parameters)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, obs: Tensor | None = None) -> None:
        """Reset recurrent state if any."""
        raise NotImplementedError

    @abstractmethod
    def train_offline(self) -> None:
        """Train on offline planner dataset (graphs)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError
