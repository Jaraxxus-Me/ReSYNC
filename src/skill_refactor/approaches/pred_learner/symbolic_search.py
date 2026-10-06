"""Symbolic search module for generating action-effect vectors in bilevel learning."""

from __future__ import annotations

import abc
import copy
import itertools
import logging
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    cast,
)

import torch
from relational_structs.pddl import LiftedAtom
from torch import Tensor

from skill_refactor.approaches.operator_learner import (
    OpEffBelief,
    learn_operator_from_data,
)
from skill_refactor.approaches.operator_learner.segmentation import segment_trajectory
from skill_refactor.approaches.pred_learner.utils import one2two
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    GroundAtom,
    GroundAtomTrajectory,
    GroundOperator,
    LiftedOperator,
    LowLevelTrajectory,
    Object,
    Predicate,
    Segment,
    Task,
)
from skill_refactor.utils.task_planning import (
    create_task_planning_heuristic,
    get_object_combinations,
    prune_ground_atom_dataset,
    segment_trajectory_to_atoms_sequence,
    task_plan,
    task_plan_grounding,
)

### Start of MCTS classes for AE vector generation ###


def tensor_in_list(tensor: Tensor, list_of_tensors: List) -> bool:
    """Check if tensor exists in list of tensors."""
    return any(torch.equal(tensor, x) for x in list_of_tensors)


class AEMCTSNode:
    """MCTS Node for AE vector search.

    State representation: flattened AE vector where each element can be:
    - 0: unassigned (no effect)
    - 1: add effect
    - -1: delete effect
    """

    def __init__(
        self,
        state: Tensor,
        level: int,
        parent: Optional["AEMCTSNode"] = None,
        search_region: Optional[Tensor] = None,
    ):
        self.state = state.clone()  # Current AE vector state (flattened)
        self.parent = parent
        self.level = level
        self.visits = 0
        self.guidance = (
            torch.zeros_like(state, dtype=torch.float32) - 1.0
        )  # Guidance values
        self.value = 0.0
        self.next_states: Set[Tuple[int, ...]] = set()
        self.search_region = search_region  # Store search region

        # Pre-compute valid next states (only change 0 -> 1 or 0 -> -1)
        # Only consider positions within the search region
        for i in range(len(self.state)):
            if self.state[i] == 0 and self._is_in_search_region(i):
                for new_val in [1, -1]:  # Add or delete effect
                    next_state = self.state.clone()
                    next_state[i] = new_val
                    self.next_states.add(tuple(next_state.tolist()))

    def _is_in_search_region(self, position: int) -> bool:
        """Check if a flattened position is within the search region.

        Args:
            position: Flattened position index in state vector

        Returns:
            True if position is in search region or no search region specified
        """
        if self.search_region is None:
            return True

        return bool(self.search_region[position].item())

    def is_fully_expanded(self) -> bool:
        """Check if all possible next states have been explored."""
        return len(self.next_states) == 0

    def update_value(self, zero_loss: Tensor, guidance_threshold: float) -> None:
        """Update node value based on guidance and zero loss."""
        non_zero_indices = torch.where(self.state != 0)[0]

        if len(non_zero_indices) == 0:
            # Root node - use all zero losses
            self.value = zero_loss.mean().item()
        else:
            non_zero_guidance = self.guidance[non_zero_indices]

            # Check if guidance is reasonable for non-zero positions
            if non_zero_guidance.mean() < guidance_threshold:
                # It will be a good parent
                local_zero_loss = zero_loss.clone()
                local_zero_loss[non_zero_indices] = 0.0  # Ignore non-zero positions
                self.value = local_zero_loss.mean().item()

                # Bonus for good overall guidance
                if self.guidance.mean() > guidance_threshold:
                    self.value += self.guidance.mean().item() - guidance_threshold
            else:
                # Poor guidance on assigned positions - penalize heavily
                self.value = float("-inf")

    def expand(
        self, zero_loss: Tensor, evaluated_states: Optional[Set[Tuple[int, ...]]] = None
    ) -> Optional["AEMCTSNode"]:
        """Expand node by creating a child with next best assignment.

        Args:
            zero_loss: Zero loss tensor for guidance
            evaluated_states: Set of already evaluated states to avoid

        Returns:
            New child node, or None if no valid unexplored state found
        """
        next_state = self.get_successor(zero_loss, evaluated_states)
        if next_state is None:
            return None
        return AEMCTSNode(
            next_state,
            level=self.level + 1,
            parent=self,
            search_region=self.search_region,
        )

    def get_successor(
        self, zero_loss: Tensor, evaluated_states: Optional[Set[Tuple[int, ...]]] = None
    ) -> Optional[Tensor]:
        """Get next state prioritizing positions with highest zero loss.

        Args:
            zero_loss: Zero loss tensor for guidance
            evaluated_states: Set of already evaluated states to avoid

        Returns:
            Next valid state, or None if no valid unexplored state found
        """
        if evaluated_states is None:
            evaluated_states = set()

        local_zero_loss = zero_loss.clone()
        non_zero_indices = torch.where(self.state != 0)[0]
        local_zero_loss[non_zero_indices] = -1  # Mark assigned positions

        # Mark positions outside search region as unavailable
        for i in range(len(local_zero_loss)):
            if not self._is_in_search_region(i):
                local_zero_loss[i] = -1

        # Try to find highest loss unassigned position
        while not (local_zero_loss == -1).all():
            max_loss_idx = int(torch.argmax(local_zero_loss).item())

            # Try both add and delete effects for this position
            for new_val in [1, -1]:
                next_state = self.state.clone()
                next_state[max_loss_idx] = new_val
                next_state_tuple = tuple(next_state.tolist())

                # Check if this state is in our next_states and not already evaluated
                if (
                    next_state_tuple in self.next_states
                    and next_state_tuple not in evaluated_states
                ):
                    self.next_states.remove(next_state_tuple)
                    return next_state

            # Position already tried, mark as unavailable
            local_zero_loss[max_loss_idx] = -1

        # Fallback to random selection if guided selection exhausted
        remaining_valid_states = [
            state_tuple
            for state_tuple in self.next_states
            if state_tuple not in evaluated_states
        ]

        if remaining_valid_states:
            chosen_idx = int(torch.randint(0, len(remaining_valid_states), (1,)).item())
            chosen_state_tuple = remaining_valid_states[chosen_idx]
            chosen_state = torch.tensor(chosen_state_tuple, dtype=self.state.dtype)
            self.next_states.remove(chosen_state_tuple)
            return chosen_state

        # No valid unexplored states found
        return None


class HierarchicalAEMCTSSearcher:
    """Hierarchical MCTS searcher for AE vector generation."""

    def __init__(
        self,
        vector_dim: int,
        max_level: int,
        guidance_threshold: float,
        search_region: Optional[Tensor] = None,
    ):
        self.visits = 0
        self.guidance_threshold = min(guidance_threshold * vector_dim, 0.5)
        self.max_level = max_level
        self.search_region = search_region

        # Initialize root node (all zeros = no effects)
        root_state = torch.zeros(vector_dim, dtype=torch.int32)
        root = AEMCTSNode(root_state, level=0, search_region=search_region)
        root.guidance = torch.zeros(root_state.shape, dtype=torch.float32)

        self.global_zero_loss = torch.zeros(root_state.shape, dtype=torch.float32)
        root.update_value(self.global_zero_loss, self.guidance_threshold)

        self.frontier = [root]
        self.evaluated_states: Set[Tuple[int, ...]] = {tuple(root_state.tolist())}

    def uct_selection(
        self, nodes: List[AEMCTSNode], batch_size: int
    ) -> List[AEMCTSNode]:
        """Select nodes using Upper Confidence Bound for Trees."""
        visits = torch.tensor([node.visits for node in nodes], dtype=torch.float32)
        values = torch.tensor([node.value for node in nodes], dtype=torch.float32)

        # Avoid division by zero
        visits = torch.where(visits == 0, 1e-5, visits)
        total_visits = torch.log(torch.tensor(self.visits + 1, dtype=torch.float32))

        # UCT calculation with exploration parameter
        exploration_param = 14.1 * self.guidance_threshold
        uct_values = values / visits + exploration_param * torch.sqrt(
            total_visits / visits
        )

        # Select top batch_size nodes
        _, top_indices = torch.topk(uct_values, min(batch_size, len(nodes)))
        return [nodes[i] for i in top_indices.tolist()]

    def update_value(self, state: Tensor, guidance: Tensor) -> None:
        """Update node values based on new guidance feedback."""
        self.visits += 1

        if torch.isfinite(guidance).all():
            # Valid guidance - update global statistics
            zero_mask = state == 0
            self.global_zero_loss[zero_mask] += guidance[zero_mask]

            # Update matching frontier nodes
            for node in self.frontier:
                if torch.equal(node.state, state):
                    node.guidance = guidance.clone()

                # Recompute value with updated global information
                avg_zero_loss = self.global_zero_loss / max(self.visits, 1)
                node.update_value(avg_zero_loss, self.guidance_threshold)
        else:
            # Invalid/unsatisfiable guidance
            for node in self.frontier:
                if torch.equal(node.state, state):
                    node.guidance = torch.ones_like(node.guidance)
                    avg_zero_loss = self.global_zero_loss / max(self.visits, 1)
                    node.update_value(avg_zero_loss, self.guidance_threshold)

        self.update_frontier()

    def update_frontier(self) -> None:
        """Remove fully expanded or invalid nodes from frontier."""
        new_frontier = []
        for node in self.frontier:
            if not node.is_fully_expanded() and node.value > float("-inf"):
                new_frontier.append(node)
            else:
                logging.debug(f"Removing node from frontier: {node.state}")

        if len(new_frontier) != len(self.frontier):
            logging.debug(f"Frontier size: {len(self.frontier)} -> {len(new_frontier)}")

        self.frontier = new_frontier

    def propose(self) -> Optional[Tensor]:
        """Propose next AE vector state to evaluate."""
        self.update_frontier()

        if not self.frontier:
            return None

        # UCT selection and expansion
        selected_nodes = self.uct_selection(self.frontier, 1)

        for node in selected_nodes:
            if not node.is_fully_expanded() and node.value > float("-inf"):
                node.visits += 1

                # Try to expand with duplicate avoidance
                child = node.expand(self.global_zero_loss, self.evaluated_states)

                if child is not None:
                    # Successfully created a new child with unexplored state
                    child_state_tuple = tuple(child.state.tolist())
                    self.evaluated_states.add(child_state_tuple)

                    # Add to frontier if within level limit
                    if child.level <= self.max_level:
                        self.frontier.append(child)

                    return child.state

        return None


### Start of AE vector generation classes ###


class BaseAEVectorGenerator:
    """Generator for action-effect vectors.

    Enumerates all possible effect distributions of a single predicate across different
    operators. Each AE vector has shape [num_operators, 2] where the columns represent
    [add_effect, delete_effect].
    """

    def __init__(self, num_operators: int, arity: int):
        """Initialize the generator.

        Args:
            num_operators: Number of operators in the domain
        """
        self.num_operators = num_operators
        self.arity = arity

    def generate_all_vectors(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate all possible AE vectors.

        For each operator, the predicate can have:
        - No effect: [0, 0]
        - Add effect: [1, 0]
        - Delete effect: [0, 1]

        Yields:
            Tensor: AE vector of shape [num_operators, 2]
        """
        # Each operator can have one of 3 effect types: none, add, or delete
        effect_options = [(0, 0), (1, 0), (0, 1)]

        # Generate all combinations across all operators
        for combination in itertools.product(effect_options, repeat=self.num_operators):
            # Skip the all-zero vector (no effects)
            if all(effect == (0, 0) for effect in combination):
                continue
            ae_vector = torch.tensor(combination, dtype=torch.float32)
            yield ae_vector, torch.zeros(self.arity, dtype=torch.float32)

    def count_total_vectors(self) -> int:
        """Count total number of vectors that will be generated.

        Returns:
            int: Total number of AE vectors (3^num_operators)
        """
        return 3**self.num_operators - 1

    def update_belief(
        self,
        feed_back: Tensor,
    ) -> None:
        """Update generator's belief based on neural learning feedback.

        This method can be used by adaptive generators to update their
        generation strategy based on feedback from neural predicate learning.

        Args:
            feed_back: Feedback tensor from neural learning (e.g., validation loss)
        """
        del feed_back  # Placeholder for future use

    @classmethod
    def get_name(cls) -> str:
        """Get the name identifier for this generator class.

        Returns:
            str: Name identifier for the generator
        """
        return "base"


class ExhaustAEVectorGenerator(BaseAEVectorGenerator):
    """Exhaustive AE vector generator that enumerates all possible combinations within a
    search region (if provuded)."""

    def __init__(
        self,
        num_operators: int,
        arity: int,
        ae_var_ids: List[int],
        search_region: Optional[List[int]] = None,
    ):
        """Initialize the generator.

        Args:
            num_operators: Number of operators in the domain
            arity: Arity of predicates
            ae_var_ids: Variable IDs for the AE vector
            search_region: Optional search region restriction
        """
        super().__init__(num_operators, arity)
        self.num_operators = num_operators
        self.arity = arity
        self.ae_var_ids = torch.tensor(ae_var_ids)
        if search_region is not None:
            assert (
                len(search_region) == num_operators
            ), "Search region length must match number of operators."
            self.search_region = torch.tensor(search_region, dtype=torch.bool)
        else:
            self.search_region = torch.ones(num_operators, dtype=torch.bool)

    def generate_all_vectors(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate all possible AE vectors.

        For each operator, the predicate can have:
        - No effect: [0, 0]
        - Add effect: [1, 0]
        - Delete effect: [0, 1]

        Yields:
            Tensor: AE vector of shape [num_operators, 2]
        """
        # Each operator can have one of 3 effect types: none, add, or delete
        effect_options = [(0, 0), (1, 0), (0, 1)]

        # Generate all combinations across all operators
        for combination in itertools.product(effect_options, repeat=self.num_operators):
            # Skip the all-zero vector (no effects)
            if all(effect == (0, 0) for effect in combination):
                continue
            ae_vector = torch.tensor(combination, dtype=torch.float32)
            all_in_region = ae_vector[~self.search_region].sum() == 0
            if all_in_region:
                yield ae_vector, self.ae_var_ids

    def count_total_vectors(self) -> int:
        """Count total number of vectors that will be generated.

        Returns:
            int: Total number of AE vectors (3^num_operators)
        """
        return int(3 ** int(self.search_region.sum().item()) - 1)

    @classmethod
    def get_name(cls) -> str:
        """Get the name identifier for this generator class.

        Returns:
            str: Name identifier for the generator
        """
        return "exhaustive"


class FixedAEVectorGenerator(BaseAEVectorGenerator):
    """Fixed AE vector generator that yields predefined vectors one by one."""

    def __init__(
        self,
        num_operators: int,
        arity: int,
        ae_vectors: List[List[int]],
        ae_var_ids: List[List[int]],
    ):
        """Initialize the generator with fixed AE vectors.

        Args:
            num_operators: Number of operators in the domain
            ae_vectors: List of predefined AE vectors to yield
        """
        super().__init__(num_operators, arity)
        self.ae_vectors: List[Tensor] = []
        self.ae_var_ids: List[Tensor] = []

        # Validate that all vectors have the correct shape
        for i, vec in enumerate(ae_vectors):
            ae_vec_two = one2two(torch.tensor(vec).unsqueeze(1)).squeeze()
            ae_var_ids_tensor = torch.tensor(ae_var_ids[i])
            self.ae_vectors.append(ae_vec_two)
            self.ae_var_ids.append(ae_var_ids_tensor)

    def generate_all_vectors(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate the predefined AE vectors one by one.

        Yields:
            Tensor: Predefined AE vector of shape [num_operators, 2]
        """
        for i, ae_vector in enumerate(self.ae_vectors):
            yield ae_vector.clone(), self.ae_var_ids[i].clone()

    def count_total_vectors(self) -> int:
        """Count total number of vectors that will be generated.

        Returns:
            int: Number of predefined AE vectors
        """
        return len(self.ae_vectors)

    @classmethod
    def get_name(cls) -> str:
        """Get the name identifier for this generator class.

        Returns:
            str: Name identifier for the generator
        """
        return "fixed"


class MCTExpAEVectorGenerator(BaseAEVectorGenerator):
    """MCTS Expansion AE vector generator that uses Monte Carlo Tree Search to
    intelligently explore the space of AE vectors based on neural learning feedback."""

    def __init__(
        self,
        num_operators: int,
        arity: int,
        ae_var_ids: List[int],
        search_region: Optional[List[int]] = None,
        guidance_threshold: float = 0.1,
        max_iterations: int = 50,
    ):
        """Initialize MCTS-based AE vector generator.

        Args:
            num_operators: Number of operators in the domain
            arity: Arity of predicates
            ae_var_ids: Variable IDs for the AE vector
            search_region: Optional search region restriction (list of operator indices)
            guidance_threshold: Threshold for guidance-based value updates
            max_iterations: Maximum number of MCTS iterations
        """
        super().__init__(num_operators, arity)
        self.ae_var_ids = torch.tensor(ae_var_ids)

        # Set up search region
        if search_region is not None:
            assert (
                len(search_region) == num_operators
            ), "Search region length must match number of operators."
            self.search_region = torch.tensor(search_region, dtype=torch.bool)
        else:
            self.search_region = torch.ones(num_operators, dtype=torch.bool)

        # Calculate max_level from search region sum
        self.max_level = int(self.search_region.sum().item())
        self.guidance_threshold = guidance_threshold
        self.max_iterations = max_iterations

        # Initialize MCTS searcher for flattened AE vector (num_operators * 2)
        self.mcts_searcher = HierarchicalAEMCTSSearcher(
            vector_dim=num_operators,
            max_level=self.max_level,
            guidance_threshold=guidance_threshold,
            search_region=self.search_region,
        )

        self.iteration_count = 0
        self.current_vector: Optional[Tensor] = None

    def generate_all_vectors(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Generate AE vectors using MCTS exploration.

        This iterator will propose new vectors based on MCTS search until either
        max_iterations is reached or MCTS search is exhausted.
        """
        while self.iteration_count < self.max_iterations:
            # Get next vector proposal from MCTS
            proposed_state = self.mcts_searcher.propose()

            if proposed_state is None:
                # MCTS search exhausted
                logging.info("MCTS search exhausted, stopping generation")
                break

            self.iteration_count += 1
            # Store current vector (now tensor from MCTS)
            self.current_vector = proposed_state

            # Skip all-zero vector (no effects)
            if torch.all(proposed_state == 0):
                continue

            # Validate that effects are within search region
            ae_vector = one2two(proposed_state.unsqueeze(1)).squeeze()
            if not self._is_ae_vector_in_search_region(ae_vector):
                continue

            yield ae_vector, self.ae_var_ids.clone()

    def _is_ae_vector_in_search_region(self, ae_vector: Tensor) -> bool:
        """Check if AE vector respects search region constraints.

        Args:
            ae_vector: AE vector tensor [num_operators, 2]

        Returns:
            True if all effects are within search region
        """
        for i in range(self.num_operators):
            if not self.search_region[i]:  # Operator not in search region
                if ae_vector[i, 0] > 0 or ae_vector[i, 1] > 0:  # Has effects
                    return False
        return True

    def update_belief(
        self,
        feed_back: Tensor,
    ) -> None:
        """Update MCTS beliefs based on neural learning feedback.

        Args:
            ae_vector: Action-effect vector that was used for learning
            var_bind_idx: Variable binding indices for the predicate
            validation_loss: Validation loss from neural training
            learned_ae_vector: Actual AE vector learned by the neural model
        """
        if self.current_vector is None:
            logging.warning("No current vector to update belief for")
            return
        # Update MCTS with guidance (both are tensors now)
        self.mcts_searcher.update_value(self.current_vector, feed_back)

        logging.info(
            f"Updated MCTS belief with current guidance {feed_back}, "
            f"current vector {self.current_vector}, "
        )

    def count_total_vectors(self) -> int:
        """Estimate total number of vectors that could be generated.

        For MCTS, this is limited by max_iterations rather than exhaustive enumeration.
        """
        return min(self.max_iterations, 3 ** int(self.search_region.sum().item()) - 1)

    @classmethod
    def get_name(cls) -> str:
        """Get the name identifier for this generator class."""
        return "mct_expansion"


def get_ae_generator_by_name(name: str) -> Type[BaseAEVectorGenerator]:
    """Get AE vector generator class by name.

    Args:
        name: Name of the generator ("base", "exhaustive", or "fixed")

    Returns:
        Type[BaseAEVectorGenerator]: The generator class

    Raises:
        ValueError: If the generator name is not recognized
    """
    generators: Dict[str, Type[BaseAEVectorGenerator]] = {
        "base": BaseAEVectorGenerator,
        "exhaustive": ExhaustAEVectorGenerator,
        "fixed": FixedAEVectorGenerator,
        "mct_expansion": MCTExpAEVectorGenerator,
    }

    if name not in generators:
        available_names = list(generators.keys())
        raise ValueError(
            f"Unknown generator name '{name}'. Available generators: {available_names}"
        )

    return generators[name]


### End of AE vector generation classes ###

### Start of predicate selection class ###


@dataclass(frozen=True, eq=False, repr=False)
class OperatorBeliefScoreFunction(abc.ABC):
    """A score function for guiding search over predicate **sets**."""

    _atom_dataset: List[GroundAtomTrajectory]  # data with all candidate predicates
    _train_tasks: List[Task]  # all of the train tasks
    _row_names: List[LiftedOperator]  # all of the operator names
    metric_name: str  # num_nodes_created or num_nodes_expanded

    def evaluate(
        self,
        candidate_ae_matrix: Tensor,
        candidate_eff_predicates: List[Predicate],
        precond_predicates: List[Predicate],
        predicates_ent_idx: List[Tensor],
        curr_best_score: float,
        basic_operators: Optional[Set[LiftedOperator]] = None,
    ) -> Tuple[float, Set[LiftedOperator]]:
        """Evaluate candidate predicates for operator belief scoring."""
        total_cost = len(candidate_eff_predicates) + len(precond_predicates)
        logging.info(
            f"Evaluating predicates: {candidate_eff_predicates + precond_predicates}, with "
            f"total cost {total_cost}"
        )
        assert candidate_ae_matrix.shape[0] == len(self._row_names)
        candidate_predicates: FrozenSet[Predicate] = frozenset(
            candidate_eff_predicates + precond_predicates
        )
        pruned_atom_data = prune_ground_atom_dataset(
            self._atom_dataset,
            candidate_predicates,
        )
        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        try:
            # delete the pre-condition only predicates now
            operator_belief = OpEffBelief(
                row_names=self._row_names,
                col_names=candidate_eff_predicates,
                col_var_idx=predicates_ent_idx,
                ae_matrix=candidate_ae_matrix,
            )
            if basic_operators is not None:
                operators, _, _, num_samples_out = learn_operator_from_data(
                    "belief_prog",
                    low_level_trajs,
                    self._train_tasks,
                    set(candidate_predicates),
                    given_operators=basic_operators,
                    ground_atom_dataset=pruned_atom_data,
                    operator_belief=operator_belief,
                )
            else:
                operators, _, _, num_samples_out = learn_operator_from_data(
                    "belief",
                    low_level_trajs,
                    self._train_tasks,
                    set(candidate_predicates),
                    given_operators=set(self._row_names),
                    ground_atom_dataset=pruned_atom_data,
                    operator_belief=operator_belief,
                )

        except TimeoutError:
            logging.info("Warning: Operator Learning timed out! Skipping evaluation.")
            return float("inf"), set()
        logging.debug(
            f"Learned {len(operators)} operators for this predicate-matrix set."
        )
        if len(operators) == 0:
            logging.info("Warning: No operators learned! Skipping evaluation.")
            # larger than any possible score
            op_score = (
                CFG.pred_search_expected_nodes_upper_bound * len(self._train_tasks) * 2
            )
            return op_score, set()
        strips_ops = [op for op in operators if isinstance(op, LiftedOperator)]
        assert len(strips_ops) == len(
            self._row_names
        ), f"Expected {len(self._row_names)} operators, got {len(strips_ops)}"
        # We assume that the operators are ordered in the same way as the
        segmented_trajs = [
            segment_trajectory(traj, atom_seq=atom_seq)
            for traj, atom_seq in pruned_atom_data
        ]
        op_score = self.evaluate_with_operators(
            candidate_predicates,
            low_level_trajs,
            segmented_trajs,
            strips_ops,
            curr_best_score,
        )
        logging.info(f"Adding out-of-sample penalty for {num_samples_out} samples.")
        logging.info(f"Operator score before adding: {op_score}")
        op_score += (
            num_samples_out * CFG.pred_search_operator_learning_out_of_sample_penalty
        ) / len(self._train_tasks)

        return op_score, operators

    def evaluate_with_operators(
        self,
        candidate_predicates: FrozenSet[Predicate],
        low_level_trajs: List[LowLevelTrajectory],
        segmented_trajs: List[List[Segment]],
        operators: List[LiftedOperator],
        _curr_best_score: float,
        max_traj_used: Optional[int] = -1,
    ) -> float:
        """Evaluate candidate predicates with given operators."""
        assert self.metric_name in ("num_nodes_created", "num_nodes_expanded")
        score = 0.0
        seen_demos = 0
        low_level_trajs = low_level_trajs[:max_traj_used]
        segmented_trajs = segmented_trajs[:max_traj_used]
        assert len(low_level_trajs) == len(segmented_trajs)
        for ll_traj, seg_traj in zip(low_level_trajs, segmented_trajs):
            objects = [
                t
                for t in self._train_tasks
                if t.train_task_idx == ll_traj.train_task_idx
            ][0].objects
            demo_atoms_sequence = segment_trajectory_to_atoms_sequence(seg_traj)
            _ = [segment.op for segment in seg_traj]  # demo_ground_op_sequences
            seen_demos += 1
            init_atoms = demo_atoms_sequence[0]
            goal = [
                t
                for t in self._train_tasks
                if t.train_task_idx == ll_traj.train_task_idx
            ][0].goal
            ground_operators, reachable_atoms = task_plan_grounding(
                init_atoms, objects, operators, allow_noops=True
            )
            try:
                heuristic = create_task_planning_heuristic(
                    CFG.sesame_task_planning_heuristic,
                    init_atoms,
                    goal,
                    ground_operators,
                    set(candidate_predicates),
                    objects,
                )
            except Exception:
                # If the heuristic is not defined, we will skip this trajectory
                # assuming the task can't be solved
                score += CFG.pred_search_expected_nodes_upper_bound
                continue
            # The expected time needed before a low-level plan is found. We
            # approximate this using node creations and by adding a penalty
            # for every skeleton after the first to account for backtracking.
            expected_planning_time = 0.0
            # Keep track of the probability that a refinable skeleton has still
            # not been found, updated after each new goal-reaching skeleton is
            # considered.
            refinable_skeleton_not_found_prob = 1.0
            # When the operator is already in a good shape, make the objective more strict
            max_skeletons = CFG.pred_search_max_skeletons_optimized
            generator = task_plan(
                init_atoms,
                goal,
                ground_operators,
                reachable_atoms,
                heuristic,
                CFG.seed,
                CFG.predicate_search_task_planning_timeout,
                max_skeletons,
            )
            try:
                for idx, (_, plan_atoms_sequence, metrics) in enumerate(generator):
                    assert goal.issubset(plan_atoms_sequence[-1])
                    # Estimate the probability that this skeleton is refinable.

                    # Part 1: based on atom seq length
                    refinement_prob = self._get_refinement_prob_atom(
                        demo_atoms_sequence, plan_atoms_sequence
                    )
                    # if (refinement_prob_atom == CFG.pred_search_expected_nodes_optimal_demo_prob) and \
                    #         enable_op_comparison:
                    #     # Part 1.5: based on operator name and object match
                    #     # if the atom-based probability is already optimal
                    #     refinement_prob_op = self._get_refinement_prob_op(
                    #         demo_ground_op_sequences, plan_op_sequence
                    #     )
                    #     refinement_prob = refinement_prob_atom * refinement_prob_op
                    # else:
                    #     refinement_prob = refinement_prob_atom

                    # Part 2: based on num_nodes_created/expanded during planning
                    # Get the number of nodes that have been created or
                    # expanded so far.
                    assert self.metric_name in metrics
                    num_nodes = metrics[self.metric_name]
                    # This contribution to the expected number of nodes is for
                    # the event that the current skeleton is refinable, but no
                    # previous skeleton has been refinable.
                    p = refinable_skeleton_not_found_prob * refinement_prob
                    expected_planning_time += p * num_nodes
                    # Apply a penalty to account for the time that we'd spend
                    # in backtracking if the last skeleton was not refinable.
                    if idx > 0:
                        w = CFG.pred_search_expected_nodes_backtracking_cost
                        expected_planning_time += p * w
                    # Update the probability that no skeleton yet is refinable.
                    refinable_skeleton_not_found_prob *= 1 - refinement_prob
            except AssertionError:
                # Note if we failed to find any skeleton, the next lines add
                # the upper bound with refinable_skeleton_not_found_prob = 1.0,
                # so no special action is required.
                pass
            # After exhausting the skeleton budget or timeout, we use this
            # probability to estimate a "worst-case" planning time, making the
            # soft assumption that some skeleton will eventually work.
            ub = CFG.pred_search_expected_nodes_upper_bound
            expected_planning_time += refinable_skeleton_not_found_prob * ub
            # The score is simply the total expected planning time.
            score += expected_planning_time
        return score

    @staticmethod
    def _get_refinement_prob_atom(
        demo_atoms_sequence: Sequence[Set[GroundAtom]],
        plan_atoms_sequence: Sequence[Set[GroundAtom]],
    ) -> float:
        """Estimate the probability that plan_atoms_sequence is refinable using the
        demonstration demo_atoms_sequence."""
        # Make a soft assumption that the demonstrations are optimal,
        # using a geometric distribution.
        demo_len = len(demo_atoms_sequence)
        plan_len = len(plan_atoms_sequence)
        # The exponent is the difference in plan lengths.
        exponent = abs(demo_len - plan_len)
        p = CFG.pred_search_expected_nodes_optimal_demo_prob
        return p * (1 - p) ** exponent

    @staticmethod
    def _get_refinement_prob_op(
        demo_ground_op_sequences: Sequence[GroundOperator],
        plan_op_sequence: Sequence[GroundOperator],
    ) -> float:
        """Estimate the probability that plan_atoms_sequence is refinable using the
        demonstration demo_atoms_sequence."""
        # Make a soft assumption that the demonstrations are optimal,
        # using a geometric distribution.
        name_match = [
            1 if demo_op.name == plan_op.name else 0
            for demo_op, plan_op in zip(demo_ground_op_sequences, plan_op_sequence)
        ]
        objects_match = [
            1 if demo_op.parameters == plan_op.parameters else 0
            for demo_op, plan_op in zip(demo_ground_op_sequences, plan_op_sequence)
        ]
        percentage_name_mismatch = name_match.count(0)
        percentage_object_mismatch = objects_match.count(0)
        # The exponent is the difference in plan lengths.
        exponent = abs(percentage_name_mismatch + percentage_object_mismatch)
        p = CFG.pred_search_expected_nodes_optimal_demo_prob
        return p * (1 - p) ** exponent


### Start of hill-climbing search class ###


class HillClimbingSearch:
    """Hill-climbing search for optimal predicate selection.

    At each step, adds the single predicate that most improves the score, continuing
    until no improvement is possible.
    """

    def __init__(
        self,
        score_function: OperatorBeliefScoreFunction,
        provided_effect_predicates: List[Predicate],
        provided_prec_predicates: List[Predicate],
        basic_matrix: List[Tensor],
        basic_pred_var_idx: List[Tensor],
        verbose: bool = True,
        basic_operators: Optional[Set[LiftedOperator]] = None,
    ):
        """Initialize the hill-climbing search.

        Args:
            score_function: Function to evaluate predicate sets
            provided_effect_predicates: Initial effect predicates
            provided_prec_predicates: Initial precondition-only predicates
            basic_matrix: Initial AE vectors for effect predicates
            basic_pred_var_idx: Initial variable binding indices
            verbose: Whether to log search progress
        """
        self.score_function = score_function
        self.provided_effect_predicates = provided_effect_predicates
        self.provided_prec_predicates = provided_prec_predicates
        self.basic_matrix = basic_matrix
        self.basic_pred_var_idx = basic_pred_var_idx
        self.verbose = verbose

        # Initialize current state
        self.current_effect_predicates = copy.deepcopy(provided_effect_predicates)
        self.current_matrix = (
            torch.stack(basic_matrix, dim=1) if basic_matrix else torch.empty(0, 0)
        )
        self.current_var_idx = copy.deepcopy(basic_pred_var_idx)
        self.basic_operators = basic_operators

        # Evaluate initial score
        self.current_score, self.current_operators = self.score_function.evaluate(
            self.current_matrix,
            self.current_effect_predicates,
            self.provided_prec_predicates,
            self.current_var_idx,
            curr_best_score=float("inf"),
            basic_operators=self.basic_operators,
        )

        if self.verbose:
            logging.info(f"Initial hill-climbing score: {self.current_score}")

    def search(
        self,
        candidate_predicates: List[Predicate],
        candidate_ae_vectors: List[Tensor],
        candidate_var_indices: List[Tensor],
    ) -> Tuple[List[Predicate], List[Tensor], Set[LiftedOperator], float]:
        """Perform hill-climbing search to select optimal predicates.

        Args:
            candidate_predicates: List of candidate predicates to consider
            candidate_ae_vectors: Corresponding AE vectors for each candidate
            candidate_var_indices: Variable binding indices for each candidate

        Returns:
            Tuple of:
            - Final selected effect predicates
            - Final AE matrix as list of vectors
            - Final learned operators
            - Final score
        """
        if self.verbose:
            logging.info(
                f"Starting hill-climbing search with {len(candidate_predicates)} candidates"
            )

        # Track which candidates haven't been added yet
        available_indices = set(range(len(candidate_predicates)))
        iteration = 0

        while available_indices:
            iteration += 1
            if self.verbose:
                logging.info(
                    f"Hill-climbing iteration {iteration}, {len(available_indices)} candidates remaining"
                )

            best_improvement = None
            best_candidate_idx = None
            best_score = self.current_score
            best_operators = None

            # Try adding each remaining candidate
            for candidate_idx in available_indices:
                # Create new predicate set by adding this candidate
                trial_effect_predicates = self.current_effect_predicates + [
                    candidate_predicates[candidate_idx]
                ]
                trial_var_idx = self.current_var_idx + [
                    candidate_var_indices[candidate_idx]
                ]

                # Create new AE matrix with this candidate
                if self.current_matrix.numel() == 0:
                    # Handle case where we start with empty matrix
                    trial_matrix = candidate_ae_vectors[candidate_idx].unsqueeze(1)
                else:
                    trial_matrix = torch.cat(
                        [
                            self.current_matrix,
                            candidate_ae_vectors[candidate_idx].unsqueeze(1),
                        ],
                        dim=1,
                    )

                # Evaluate this configuration
                try:
                    trial_score, trial_operators = self.score_function.evaluate(
                        trial_matrix,
                        trial_effect_predicates,
                        self.provided_prec_predicates,
                        trial_var_idx,
                        best_score,
                        self.basic_operators,
                    )

                    if self.verbose:
                        logging.info(
                            f"  Candidate {candidate_predicates[candidate_idx].name}: score {trial_score}"
                        )

                    # Check if this is the best improvement so far
                    if trial_score < best_score:
                        best_score = trial_score
                        best_candidate_idx = candidate_idx
                        best_improvement = (
                            trial_effect_predicates,
                            trial_matrix,
                            trial_var_idx,
                        )
                        best_operators = trial_operators

                except Exception as e:
                    if self.verbose:
                        logging.warning(
                            f"  Candidate {candidate_predicates[candidate_idx].name} failed: {e}"
                        )
                    continue

            # If no improvement found, stop search
            if best_improvement is None or best_score >= self.current_score:
                if self.verbose:
                    logging.info(
                        f"No improvement found in iteration {iteration}, stopping search"
                    )
                break

            # Apply the best improvement
            assert best_candidate_idx is not None and best_operators is not None
            (
                self.current_effect_predicates,
                self.current_matrix,
                self.current_var_idx,
            ) = best_improvement
            self.current_score = best_score
            self.current_operators = best_operators
            available_indices.remove(best_candidate_idx)

            if self.verbose:
                logging.info(
                    f"  Added predicate {candidate_predicates[best_candidate_idx].name}"
                )
                logging.info(f"  New score: {self.current_score}")

        if self.verbose:
            logging.info(f"Hill-climbing search completed after {iteration} iterations")
            logging.info(f"Final score: {self.current_score}")
            logging.info(
                f"Selected {len(self.current_effect_predicates) - len(self.provided_effect_predicates)} new predicates"
            )

        # Convert matrix back to list of vectors for compatibility
        final_matrix_list = [
            self.current_matrix[:, i] for i in range(self.current_matrix.size(1))
        ]

        return (
            self.current_effect_predicates,
            final_matrix_list,
            self.current_operators,
            self.current_score,
        )


### End of hill-climbing search class ###


### Start of precondition filtering class ###


def filter_operator_preconditions(
    operators: Set[LiftedOperator],
    train_tasks: List[Task],
    predicate_interpreters: Dict[
        Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
    ],
    unsolvable_threshold: float = 0.3,
) -> Set[LiftedOperator]:
    """Filter operator preconditions by removing quantified predicates that cause
    unsolvability.

    This function relaxes operator preconditions by:
    1. Identifying all predicates with "unexpected" preconditions (variables outside operator
       parameters) and tracking where they appear across all operators
    2. Creating a fully relaxed operator set with all unexpected preconditions removed
    3. For each unexpected predicate, testing adding it back to all operators where it
       originally appeared
    4. Keeping predicates that don't make >threshold fraction of tasks unsolvable

    Args:
        operators: Original set of operators to filter
        train_tasks: Training tasks to test solvability
        predicate_interpreters: Dict mapping predicates to interpreter functions
        unsolvable_threshold: Maximum fraction of tasks that can be unsolvable when adding
            a predicate back. Predicates exceeding this are discarded from all operators.
        timeout: Timeout for each planning attempt (seconds)

    Returns:
        New set of operators with filtered preconditions (same names, same effects,
        relaxed preconditions)
    """
    logging.info(f"Filtering preconditions for {len(operators)} operators")

    # Convert to list for consistent ordering
    operators_list = sorted(operators, key=lambda op: op.name)
    considered_predicate_names = [pred.name for pred in predicate_interpreters.keys()]

    # Step 1: Identify all unexpected preconditions and track their locations
    # Map from predicate name to list of (operator_idx, precondition_atom)
    unexpected_precond_locations: Dict[str, List[Tuple[int, LiftedAtom]]] = {}
    # Track expected preconditions for each operator
    expected_preconditions_by_op: List[Set[LiftedAtom]] = []

    for op_idx, operator in enumerate(operators_list):
        op_param_vars_types = set(param.type for param in operator.parameters)
        expected_preconditions: Set[LiftedAtom] = set()

        for precond in operator.preconditions:
            pre_cond_predicate = cast(Predicate, precond.predicate)
            pred_name = precond.predicate.name
            if pred_name not in considered_predicate_names:
                expected_preconditions.add(precond)
                continue
            # Check if predicate is quantified (hasattr handles both custom and base Predicate)
            if not getattr(pre_cond_predicate, "quantified", False):
                expected_preconditions.add(precond)
                continue
            precond_types = getattr(
                pre_cond_predicate, "base_types", pre_cond_predicate.types
            )
            if set(precond_types).issubset(op_param_vars_types) or (
                set(precond_types) == op_param_vars_types
            ):
                expected_preconditions.add(precond)
                continue
            logging.info(
                f"Operator {operator.name} has unexpected precondition: {precond}"
            )
            if pred_name not in unexpected_precond_locations:
                unexpected_precond_locations[pred_name] = []
            unexpected_precond_locations[pred_name].append((op_idx, precond))

        expected_preconditions_by_op.append(expected_preconditions)

    logging.info(
        f"Found {len(unexpected_precond_locations)} predicates with unexpected preconditions"
    )
    for pred_name, locations in unexpected_precond_locations.items():
        logging.info(f"  {pred_name}: appears in {len(locations)} operators")

    # Step 2: Create fully relaxed operator set (only expected preconditions)
    # and get all of the solvable tasks with this relaxed set
    current_preconditions_by_op = [
        preconds.copy() for preconds in expected_preconditions_by_op
    ]
    # Build trial operator set
    trial_operators = set()
    for op_idx, operator in enumerate(operators_list):
        trial_op = LiftedOperator(
            name=operator.name,
            parameters=operator.parameters,
            preconditions=set(current_preconditions_by_op[op_idx]),
            add_effects=operator.add_effects,
            delete_effects=operator.delete_effects,
        )
        trial_operators.add(trial_op)

    filtered_tasks: List[Task] = []
    for task in train_tasks:
        solvable = _task_plan_test(
            task,
            trial_operators,
            predicate_interpreters,
        )
        if solvable:
            filtered_tasks.append(task)

    logging.info(
        f"After full relaxation, {len(filtered_tasks)}/{len(train_tasks)} tasks remain solvable"
    )

    # Step 3: Test each unexpected predicate to see if it should be added back
    for pred_name, locations in unexpected_precond_locations.items():
        logging.info(f"Testing predicate: {pred_name}")

        for op_idx_loc, precond in locations:
            # Create trial operator set with this predicate added back to specified locations
            trial_preconditions_by_op = [
                preconds.copy() for preconds in current_preconditions_by_op
            ]
            trial_preconditions_by_op[op_idx_loc].add(precond)

            # Build trial operator set
            trial_operators = set()
            for op_idx, operator in enumerate(operators_list):
                trial_op = LiftedOperator(
                    name=operator.name,
                    parameters=operator.parameters,
                    preconditions=set(trial_preconditions_by_op[op_idx]),
                    add_effects=operator.add_effects,
                    delete_effects=operator.delete_effects,
                )
                trial_operators.add(trial_op)

            # Count how many tasks become unsolvable with this predicate
            unsolvable_count = _count_unsolvable_tasks(
                trial_operators,
                filtered_tasks,
                predicate_interpreters,
            )

            total_tested = len(filtered_tasks)
            unsolvable_fraction = (
                unsolvable_count / total_tested if total_tested > 0 else 0.0
            )

            logging.info(
                f"  Unsolvable fraction: {unsolvable_fraction:.2f} "
                f"({unsolvable_count}/{total_tested})"
            )

            # Keep predicate in all operators if it doesn't make too many tasks unsolvable
            if unsolvable_fraction <= unsolvable_threshold:
                logging.info(
                    f"  Keeping predicate {pred_name} in operators {operators_list[op_idx_loc].name} "
                )
                # Update current preconditions to include this predicate
                # This will be cumulative across locations
                current_preconditions_by_op[op_idx_loc].add(precond)
            else:
                logging.info(
                    f"  Discarding predicate {pred_name} in operators {operators_list[op_idx_loc].name} "
                )

    # Step 4: Build final filtered operator set
    filtered_operators: Set[LiftedOperator] = set()
    for op_idx, operator in enumerate(operators_list):
        filtered_op = LiftedOperator(
            name=operator.name,
            parameters=operator.parameters,
            preconditions=set(current_preconditions_by_op[op_idx]),
            add_effects=operator.add_effects,
            delete_effects=operator.delete_effects,
        )
        filtered_operators.add(filtered_op)

        original_preconds = len(operator.preconditions)
        filtered_preconds = len(current_preconditions_by_op[op_idx])
        logging.info(
            f"Operator {operator.name}: {original_preconds} -> {filtered_preconds} preconditions"
        )

    return filtered_operators


def _count_unsolvable_tasks(
    operators: Set[LiftedOperator],
    tasks: List[Task],
    predicate_interpreters: Dict[
        Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
    ],
) -> int:
    """Count how many tasks are unsolvable with given operators.

    Args:
        operators: Set of operators to test
        tasks: Tasks to test
        predicate_interpreters: Dict mapping predicates to interpreter functions
        timeout: Planning timeout per task (seconds)

    Returns:
        Number of unsolvable tasks
    """
    unsolvable_count = 0

    for task in tasks:
        # Get initial atoms from task.init using predicate interpreters
        solvable = _task_plan_test(
            task,
            operators,
            predicate_interpreters,
        )
        if not solvable:
            unsolvable_count += 1

    return unsolvable_count


def _task_plan_test(
    task: Task,
    operators: Set[LiftedOperator],
    predicate_interpreters: Dict[
        Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
    ],
) -> bool:
    """Test if a task is solvable with given operators.

    Args:
        init_atoms: Initial ground atoms
        goal: Goal ground atoms
        operators: Set of operators to test
        objects: Objects in the environment
        predicate_interpreters: Dict mapping predicates to interpreter functions
        timeout: Planning timeout (seconds)
    Returns:
        True if task is solvable, False otherwise
    """
    try:
        goal = task.goal
        objects = task.objects
        init_atoms = _get_atoms_from_obs(
            task.init.unsqueeze(0),  # Add batch dimension
            objects,
            predicate_interpreters,
        )
        _, reachable_atoms = task_plan_grounding(
            init_atoms, objects, operators, allow_noops=True
        )

        return goal.issubset(reachable_atoms)

    except Exception as e:
        # Planning failed (timeout, heuristic error, etc.) - count as unsolvable
        logging.debug(f"  Planning failed during test: {e}")
        return False


def _get_atoms_from_obs(
    obs: Tensor,
    objects: Set[Object],
    predicate_interpreters: Dict[
        Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]
    ],
) -> Set[GroundAtom]:
    """Get ground atoms from an observation using predicate interpreters.

    Similar to Perceiver._get_atoms but simplified for single observation.

    Args:
        obs: Observation tensor of shape (1, ...) with batch dimension
        objects: Set of objects in the environment
        predicate_interpreters: Dict mapping predicates to interpreter functions

    Returns:
        Set of ground atoms that are true in the observation
    """
    atoms: Set[GroundAtom] = set()

    for predicate, interpret_fn in predicate_interpreters.items():
        # Generate all possible object combinations for this predicate
        input_args: List[Sequence[Object]] = []
        for args in get_object_combinations(
            objects,
            predicate.types,
            skip_self=True,
        ):
            input_args.append(args)

        if len(input_args) == 0:
            # No valid object combinations for this predicate
            continue

        # Call interpreter function
        # Returns tensor of shape (1, num_grounding) for batch size 1
        stacked_res = interpret_fn(obs, input_args)

        # Extract atoms from first (and only) batch element
        for n in range(len(input_args)):
            if stacked_res[0, n]:
                obj_args: Sequence[Object] = input_args[n]
                if predicate.arity > 0:
                    atoms.add(predicate(obj_args))
                else:
                    atoms.add(GroundAtom(predicate, []))

    return atoms


### End of precondition filtering class ###
