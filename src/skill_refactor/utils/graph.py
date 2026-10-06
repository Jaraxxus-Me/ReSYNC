"""Planning graph representation for improvisational TAMP."""

import heapq
import itertools
import logging
from dataclasses import dataclass, field
from typing import Generator

from relational_structs import GroundAtom

from skill_refactor.utils.structs import GroundOperator


@dataclass
class PlanningGraphNode:
    """Node in the planning graph representing a set of atoms."""

    atoms: frozenset[GroundAtom]
    id: int

    def __hash__(self) -> int:
        return hash(self.atoms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphNode):
            return False
        return self.atoms == other.atoms


@dataclass
class PlanningGraphEdge:
    """Edge in the planning graph representing a transition."""

    source: PlanningGraphNode
    target: PlanningGraphNode
    operator: GroundOperator | None = None
    cost: float = float("inf")
    is_shortcut: bool = False

    # Store path-dependent costs: (path, source_node_id) -> cost
    # where path is a tuple of node IDs
    costs: dict[tuple[tuple[int, ...], int], float] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.operator))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphEdge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.operator == other.operator
        )

    def get_cost(self, path: tuple[int, ...]) -> float:
        """Get the cost of this edge when coming via the specified path."""
        if not self.costs:
            return self.cost

        # Try to find exact path match
        for (p, _), cost in self.costs.items():
            if p == path:
                return cost

        # If no exact match, look for a path ending with the same node
        for (p, node_id), cost in self.costs.items():
            if p and p[-1] == self.source.id and node_id == self.source.id:
                return cost

        # Default to the minimum cost if no matching path is found
        return self.cost


class PlanningGraph:
    """Graph representation of a task plan."""

    def __init__(self) -> None:
        self.nodes: list[PlanningGraphNode] = []
        self.edges: list[PlanningGraphEdge] = []
        self.node_to_incoming_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_to_outgoing_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_map: dict[frozenset[GroundAtom], PlanningGraphNode] = {}
        self.goal_nodes: list[PlanningGraphNode] = []

    def add_node(self, atoms: set[GroundAtom]) -> PlanningGraphNode:
        """Add a node to the graph."""
        frozen_atoms = frozenset(atoms)
        assert frozen_atoms not in self.node_map
        node_id = len(self.nodes)
        node = PlanningGraphNode(frozen_atoms, node_id)
        self.nodes.append(node)
        self.node_map[frozen_atoms] = node
        self.node_to_incoming_edges[node] = []
        self.node_to_outgoing_edges[node] = []
        return node

    def add_edge(
        self,
        source: PlanningGraphNode,
        target: PlanningGraphNode,
        operator: GroundOperator | None = None,
        cost: float = float("inf"),
        is_shortcut: bool = False,
    ) -> PlanningGraphEdge:
        """Add an edge to the graph."""
        edge = PlanningGraphEdge(source, target, operator, cost, is_shortcut)
        self.edges.append(edge)
        self.node_to_incoming_edges[edge.target].append(edge)
        self.node_to_outgoing_edges[edge.source].append(edge)
        return edge

    def find_shortest_paths(
        self,
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> Generator[list[PlanningGraphEdge], None, None]:
        """Return a generator over *all* shortest paths (as lists of edges) from the
        initial node to any node whose atoms superset the goal set.

        Each yielded path is a list of PlanningGraphEdge in source->target order.

        Path-aware costs are respected via PlanningGraphEdge.get_cost(path_prefix).

        Yields:
            List[PlanningGraphEdge]
        """
        assert self.nodes, "Graph is empty, cannot find paths."

        initial_node = self.node_map[frozenset(init_atoms)]
        goal_nodes = [n for n in self.nodes if goal.issubset(n.atoms)]
        assert goal_nodes, "No goal node found"

        # Distances keyed by (node, path_signature)
        distances: dict[tuple[PlanningGraphNode, tuple[int, ...]], float] = {}
        # Best distance per node (for pruning)
        best_node_dist: dict[PlanningGraphNode, float] = {initial_node: 0.0}

        # Predecessor map:
        # state -> list of (prev_state, edge) for ALL minimal predecessors
        predecessors: dict[
            tuple[PlanningGraphNode, tuple[int, ...]],
            list[tuple[tuple[PlanningGraphNode, tuple[int, ...]], PlanningGraphEdge]],
        ] = {}

        start_sig: tuple[int, ...] = tuple()
        start_state = (initial_node, start_sig)
        distances[start_state] = 0.0
        predecessors[start_state] = []

        counter = itertools.count()
        pq: list[tuple[float, int, tuple[PlanningGraphNode, tuple[int, ...]]]] = [
            (0.0, next(counter), start_state)
        ]

        # Track the current best goal distance for pruning
        # best_goal_dist = float("inf")
        best_goal_dist = 6

        max_path_length = len(self.nodes) * 2  # same safeguard

        while pq:
            cur_dist, _, cur_state = heapq.heappop(pq)
            node, path_sig = cur_state

            # Prune outdated queue entries
            if cur_dist > distances.get(cur_state, float("inf")):
                continue
            if cur_dist > best_goal_dist:
                # Already worse than best known goal; prune
                continue
            if len(path_sig) > max_path_length:
                continue

            # If this is a goal node, update best goal distance
            # if node in goal_nodes and cur_dist < best_goal_dist:
            #     best_goal_dist = cur_dist

            # Outgoing edges
            for edge in self.node_to_outgoing_edges.get(node, []):
                edge_cost = edge.get_cost(path_sig)
                if edge_cost == float("inf"):
                    continue
                new_dist = cur_dist + edge_cost
                if new_dist > best_goal_dist:
                    continue  # cannot beat current best goal
                new_sig = path_sig + (node.id,)
                new_state = (edge.target, new_sig)

                # Node-level pruning / update
                prev_best_node = best_node_dist.get(edge.target, float("inf"))
                if new_dist < prev_best_node:
                    best_node_dist[edge.target] = new_dist

                old = distances.get(new_state, float("inf"))
                if new_dist < old - 1e-12:
                    distances[new_state] = new_dist
                    predecessors[new_state] = [(cur_state, edge)]
                    heapq.heappush(pq, (new_dist, next(counter), new_state))
                elif abs(new_dist - old) <= 1e-12:
                    # Another minimal predecessor for same state
                    predecessors[new_state].append((cur_state, edge))

        # Collect terminal states for goal nodes that achieve the minimal goal distance
        terminal_states: list[tuple[PlanningGraphNode, tuple[int, ...]]] = []
        for g in goal_nodes:
            # minimal cost for this goal node is any state whose node == g with minimal distances
            if g not in best_node_dist:
                continue
            g_best = best_node_dist[g]
            if abs(g_best - best_goal_dist) > 1e-12:
                # Not globally minimal, skip
                continue
            # All states of this goal node that match g_best
            for (n, sig), dval in distances.items():
                if n == g and abs(dval - g_best) <= 1e-12:
                    terminal_states.append((n, sig))

        assert terminal_states, "No goal state found (after search)."

        # --- Lazy enumeration of all shortest paths ---

        def enumerate_paths() -> Generator[list[PlanningGraphEdge], None, None]:
            """Depth-first backtracking over predecessor graph."""
            # We may have multiple distinct terminal states (different path signatures)

            def backtrack(
                state: tuple[PlanningGraphNode, tuple[int, ...]],
            ) -> Generator[list[PlanningGraphEdge], None, None]:
                """Backtrack from a terminal state to the initial state."""
                if state == start_state:
                    # Return empty path (will be built upward)
                    yield []
                    return
                preds = predecessors.get(state, [])
                for prev_state, edge in preds:
                    for partial in backtrack(prev_state):
                        # append edge AFTER recursive yield to maintain forward order
                        yield partial + [edge]

            # De-duplicate identical edge sequences if multiple terminal signatures collapse
            # Use tuple of (edge.source.id, edge.target.id, operator_id_or_name) as a hash
            produced = set()
            for term_state in terminal_states:
                for path_edges in backtrack(term_state):
                    key = tuple(
                        (e.source.id, e.target.id, getattr(e.operator, "name", None))
                        for e in path_edges
                    )
                    if key in produced:
                        continue
                    produced.add(key)
                    yield path_edges

            logging.info(
                "Enumerated %d distinct shortest path(s) with total cost %.4f",
                len(produced),
                best_goal_dist,
            )

        return enumerate_paths()
