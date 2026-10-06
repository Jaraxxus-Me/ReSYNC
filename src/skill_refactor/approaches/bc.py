"""Behavior cloning baseline approach using GNN."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from relational_structs import Object
from torch import Tensor

from skill_refactor.approaches.base import BaseApproach
from skill_refactor.approaches.behavior_cloning.base import BCPolicy
from skill_refactor.approaches.behavior_cloning.gnn_bc import GNNBCPolicy
from skill_refactor.benchmarks.base import BaseRLTAMPSystem, GraphData
from skill_refactor.utils.structs import (
    ApproachStepResult,
    GroundOperator,
    LiftedOperatorSkill,
)


def graphdata_to_dict(g: GraphData) -> Dict:
    """Convert GraphData to dict format expected by GNN."""

    def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    nodes = to_numpy(g.node_features).astype(np.float32)
    edges = to_numpy(g.edge_features).astype(np.float32)
    senders = to_numpy(g.edge_indices[0]).astype(np.int64)
    receivers = to_numpy(g.edge_indices[1]).astype(np.int64)

    n_node = np.array([nodes.shape[0]], dtype=np.int64)
    n_edge = np.array([edges.shape[0]], dtype=np.int64)

    if g.global_features is None:
        globals_in = np.zeros((1,), dtype=np.float32)
    else:
        globals_in = to_numpy(g.global_features).astype(np.float32).reshape(-1)

    return {
        "nodes": nodes,
        "edges": edges,
        "globals": globals_in,
        "senders": senders,
        "receivers": receivers,
        "n_node": n_node,
        "n_edge": n_edge,
    }


def strip_operator_suffix(op_name: str) -> str:
    """Strip type-specific suffix from operator name.

    During lifelong learning, operators are specialized with type suffixes
    (e.g., ReachToGrasp_Punch, ReachToGrasp_Rectangle). But skills are
    registered with just the base operator name (e.g., ReachToGrasp).

    Args:
        op_name: Operator name potentially with suffix (e.g., "ReachToGrasp_Punch")

    Returns:
        Base operator name without suffix (e.g., "ReachToGrasp")
    """
    # Known suffixes from lifelong learning
    suffixes = ["_Punch", "_Rectangle", "_Triangle", "_Tra"]

    for suffix in suffixes:
        if op_name.endswith(suffix):
            return op_name[: -len(suffix)]

    # If no known suffix found, return as is
    return op_name


def map_operator_to_base(op_name: str) -> str:
    """Map lifelong learning operators to base operators.

    Some operators from lifelong learning don't exist in the base system.
    This function maps them to semantically similar base operators.

    Args:
        op_name: Operator name from BC-GNN prediction

    Returns:
        Mapped operator name that exists in base system
    """
    # First strip any type-specific suffixes
    base_name = strip_operator_suffix(op_name)

    # Map lifelong learning operators to base operators
    operator_mapping = {
        "Punch": "ReachToGrasp",  # Punch obstruction → reach to grasp it
    }

    return operator_mapping.get(base_name, base_name)


class BehaviorCloningApproach(BaseApproach):
    """Behavior Cloning approach using a BCPolicy for Baseline.

    Implements skill-based execution where:
    - BC-GNN predicts operator + objects
    - Skill executes until termination (collision or success)
    - Only then does BC-GNN predict next operator
    """

    def __init__(
        self, system: BaseRLTAMPSystem, bc_policy: BCPolicy, seed: int
    ) -> None:
        super().__init__(system, seed)
        self.bc_policy = bc_policy

        # Skill-based execution state
        self._current_skill: Optional[LiftedOperatorSkill] = None
        self._current_operator: Optional[GroundOperator] = None
        self._current_objects: List[Object] = []
        self._last_action: Optional[Tensor] = None

    @classmethod
    def get_name(cls) -> str:
        return "bc"

    def reset(self, obs: Tensor, info: dict[str, Any]) -> ApproachStepResult:
        self.bc_policy.reset(obs)

        # Reset perceiver at the start of each episode
        if hasattr(self.system, "perceiver"):
            self.system.perceiver.reset(obs, info)
            logging.info("BC Approach: Reset perceiver at episode start")

        # Reset skill-based execution state
        self._current_skill = None
        self._current_operator = None
        self._current_objects = []
        self._last_action = None

        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        # Check if current skill has terminated
        should_switch_skill = False
        if self._current_skill is not None:
            # Check termination condition
            terminate_mask = self._current_skill.terminate(obs)
            if terminate_mask.any():
                logging.info(
                    f"Skill {self._current_skill.get_operator_name()} terminated"
                )
                should_switch_skill = True
        else:
            # No current skill, need to predict one
            should_switch_skill = True

        # Switch to new skill if needed
        if should_switch_skill:  # this is to predict new operator and objects
            # Convert observation tensor to graph representation
            graph_data_list = self.system.state_to_graph(obs, training=False)
            graph_data = graph_data_list[0]
            object_to_node = graph_data.object_to_node
            assert (
                object_to_node is not None
            ), "object_to_node must be provided by state_to_graph"

            # Convert GraphData to dict format for GNN
            graph_dict = graphdata_to_dict(graph_data)

            # Get operator and object predictions from BC policy
            if isinstance(self.bc_policy, GNNBCPolicy):
                # For predicate augmentation, we need current atoms and goal
                # Get atoms from perceiver if available
                if (
                    hasattr(self.system, "perceiver")
                    and self.bc_policy.tamp_system is not None
                ):
                    # Perceiver's step() returns atoms for the current state
                    # Pass info dict with desired_predicates as predicate names (strings)
                    # predicates_container.predicates is Dict[str, Predicate], keys are strings
                    perceiver_info = {
                        "desired_predicates": list(
                            self.system.perceiver.predicates_container.predicates.keys()
                        )
                    }
                    current_atoms_list = self.system.perceiver.step(obs, perceiver_info)
                    current_atoms = (
                        current_atoms_list[0] if current_atoms_list else set()
                    )
                    logging.info(
                        f"BC Approach: Extracted {len(current_atoms)} atoms from perceiver for BC-GNN"
                    )
                else:
                    current_atoms = set()

                # Predict operator name and object parameters
                op_name, predicted_objects = (
                    self.bc_policy.predict_operator_and_objects(
                        graph_dict, object_to_node, current_atoms
                    )
                )

                logging.info(
                    f"BC-GNN predicted operator: {op_name} with objects: {[str(o) for o in predicted_objects]}"
                )

                # Map operator to base system (handles both suffix stripping and fallback mapping)
                # e.g., ReachToGrasp_Punch -> ReachToGrasp, Punch -> ReachToGrasp
                # base_op_name = map_operator_to_base(op_name)
                base_op_name = op_name
                if base_op_name != op_name:
                    logging.info(f"Mapped operator: {op_name} -> {base_op_name}")

                # Find the matching skill for this operator
                new_skill = None
                for skill in self.system.skills:
                    if skill.get_operator_name() == base_op_name:
                        new_skill = skill
                        break

                if new_skill is None:
                    logging.warning(
                        f"No skill found for operator {base_op_name} (original: {op_name}), returning zero action"
                    )
                    device = obs.device
                    env = self.system.env
                    if hasattr(env, "single_action_space"):
                        action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
                    elif isinstance(env.unwrapped, list):
                        action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
                    else:
                        action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

                    action = torch.zeros(obs.shape[0], action_dim, device=device)
                    self._last_action = action
                    return ApproachStepResult(_action=action)

                # Create ground operator
                # Find the lifted operator from the system (use base name)
                lifted_op = None
                for op in self.system.operators:
                    if op.name == base_op_name:
                        lifted_op = op
                        break

                if lifted_op is None:
                    logging.error(
                        f"No lifted operator found for {base_op_name} (original: {op_name})"
                    )
                    device = obs.device
                    env = self.system.env
                    if hasattr(env, "single_action_space"):
                        action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
                    elif isinstance(env.unwrapped, list):
                        action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
                    else:
                        action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

                    action = torch.zeros(obs.shape[0], action_dim, device=device)
                    self._last_action = action
                    return ApproachStepResult(_action=action)

                # Create ground operator with predicted objects
                # Filter predicted objects to match the lifted operator's parameters
                # The BC-GNN might predict more objects (from specialized operators),
                # but we need to match only those required by the base operator
                operator_objects = []
                for param in lifted_op.parameters:
                    # Find the first predicted object that matches this parameter's type
                    for obj in predicted_objects:
                        if obj.is_instance(param.type) and obj not in operator_objects:
                            operator_objects.append(obj)
                            break

                if len(operator_objects) != len(lifted_op.parameters):
                    logging.error(
                        f"Could not match all parameters for {base_op_name}. "
                        f"Expected {len(lifted_op.parameters)} objects, found {len(operator_objects)}. "
                        f"Predicted objects: {[str(o) for o in predicted_objects]}"
                    )
                    device = obs.device
                    env = self.system.env
                    if hasattr(env, "single_action_space"):
                        action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
                    elif isinstance(env.unwrapped, list):
                        action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
                    else:
                        action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

                    action = torch.zeros(obs.shape[0], action_dim, device=device)
                    self._last_action = action
                    return ApproachStepResult(_action=action)

                # Use the LiftedOperator's ground() method to properly substitute variables
                # ground() requires a tuple for hashing
                ground_op = lifted_op.ground(tuple(operator_objects))

                # Reset the skill with the new operator
                try:
                    new_skill.reset(ground_op, obs)
                    self._current_skill = new_skill
                    self._current_operator = ground_op
                    self._current_objects = predicted_objects
                    logging.info(
                        f"Initialized skill {op_name} with objects {[str(o) for o in predicted_objects]}"
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(f"Failed to reset skill {op_name}: {e}")
                    device = obs.device
                    env = self.system.env
                    if hasattr(env, "single_action_space"):
                        action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
                    elif isinstance(env.unwrapped, list):
                        action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
                    else:
                        action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

                    action = torch.zeros(obs.shape[0], action_dim, device=device)
                    self._last_action = action
                    return ApproachStepResult(_action=action)

            else:
                raise NotImplementedError(
                    f"BC policy type {type(self.bc_policy)} not supported"
                )

        # Execute current skill to get action
        if self._current_skill is not None:
            try:
                action = self._current_skill.get_action(obs)
                self._last_action = action
                return ApproachStepResult(_action=action)
            except Exception as e:  # pylint: disable=broad-except
                logging.error(
                    f"Failed to execute skill {self._current_skill.get_operator_name()}: {e}"
                )
                device = obs.device
                env = self.system.env
                if hasattr(env, "single_action_space"):
                    action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
                elif isinstance(env.unwrapped, list):
                    action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
                else:
                    action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

                action = torch.zeros(obs.shape[0], action_dim, device=device)
                self._last_action = action
                return ApproachStepResult(_action=action)
        else:
            # Shouldn't reach here, but return zero action as fallback
            logging.error("No current skill available")
            device = obs.device
            env = self.system.env
            if hasattr(env, "single_action_space"):
                action_dim = env.single_action_space.shape[0]  # type: ignore[union-attr]
            elif isinstance(env.unwrapped, list):
                action_dim = env.unwrapped[0].action_space.shape[0]  # type: ignore[union-attr]
            else:
                action_dim = env.unwrapped.action_space.shape[0]  # type: ignore[union-attr,index]

            action = torch.zeros(obs.shape[0], action_dim, device=device)
            self._last_action = action
            return ApproachStepResult(_action=action)
