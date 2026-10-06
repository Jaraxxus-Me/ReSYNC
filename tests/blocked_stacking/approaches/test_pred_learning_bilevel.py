"""Unit tests for bilevel predicate learning and MCTS expansion."""

import logging

import torch

from skill_refactor.approaches.pred_learner.symbolic_search import (
    AEMCTSNode,
    HierarchicalAEMCTSSearcher,
    MCTExpAEVectorGenerator,
    get_ae_generator_by_name,
)


def test_mct_expansion() -> None:
    """Test MCTS expansion AE vector generator."""
    # Set up logging to see MCTS progress
    logging.basicConfig(level=logging.INFO)

    # Test parameters
    num_operators = 3
    arity = 2
    ae_var_ids = [0, 1]
    _max_level = 3
    guidance_threshold = 0.1
    max_iterations = 10

    # Create MCTS generator
    generator = MCTExpAEVectorGenerator(
        num_operators=num_operators,
        arity=arity,
        ae_var_ids=ae_var_ids,
        search_region=[True, True, False],  # Only search first 2 operators
        guidance_threshold=guidance_threshold,
        max_iterations=max_iterations,
    )

    assert generator.get_name() == "mct_expansion"
    # Note: count_total_vectors() returns actual searchable states, not max_iterations
    total_vectors = generator.count_total_vectors()
    assert total_vectors > 0, "Should have at least some vectors to search"
    print(f"Total searchable vectors: {total_vectors}")

    # Test vector generation
    generated_vectors = []
    generated_var_ids = []

    for i, (ae_vector, var_bind_idx) in enumerate(generator.generate_all_vectors()):
        print(f"Iteration {i + 1}: Generated AE vector:\n{ae_vector}")
        print(f"Variable binding indices: {var_bind_idx}")

        # Verify vector properties
        assert ae_vector.shape == (num_operators, 2)
        # AE vectors are returned as int64 (discrete values 0,1,2)
        assert ae_vector.dtype in [torch.int64, torch.float32]
        assert torch.equal(var_bind_idx, torch.tensor(ae_var_ids))

        # Store generated vectors
        generated_vectors.append(ae_vector.clone())
        generated_var_ids.append(var_bind_idx.clone())

        if i >= 5:  # Test first few iterations
            break

    # Verify we generated some vectors
    assert len(generated_vectors) > 0
    print(f"Successfully generated {len(generated_vectors)} AE vectors using MCTS")


def test_ae_mcts_node() -> None:
    """Test AEMCTSNode functionality."""
    # Create a simple state (now using torch tensor)
    state = torch.tensor([0, 1, 0, 2, 0, 0], dtype=torch.int32)
    node = AEMCTSNode(state, level=1)

    # Test basic properties
    assert node.level == 1
    assert node.visits == 0
    assert node.value == 0.0
    assert len(node.guidance) == len(state)

    # Test next state generation
    assert len(node.next_states) > 0  # Should have some next states

    # Test expansion
    zero_loss = torch.rand(len(state), dtype=torch.float32)
    if not node.is_fully_expanded():
        child = node.expand(zero_loss)
        if child is not None:
            assert child.level == node.level + 1
            assert child.parent == node

    print("AEMCTSNode test passed")


def test_hierarchical_mcts_searcher() -> None:
    """Test HierarchicalAEMCTSSearcher functionality."""
    vector_dim = 6  # 3 operators * 2 effects (flattened)
    max_level = 2
    guidance_threshold = 0.1
    # Search region should match flattened vector dim (6 elements, not 3)
    # Each operator has 2 effects, so we repeat the operator mask
    search_region = torch.tensor(
        [True, True, True, True, False, False], dtype=torch.bool
    )  # Only first 2 operators (4 effect positions)

    searcher = HierarchicalAEMCTSSearcher(
        vector_dim=vector_dim,
        max_level=max_level,
        guidance_threshold=guidance_threshold,
        search_region=search_region,
    )

    # Test initial state
    assert len(searcher.frontier) == 1
    assert searcher.visits == 0

    # Test proposal generation and duplicate avoidance
    proposed_states = []
    for i in range(10):
        proposed_state = searcher.propose()
        if proposed_state is not None:
            state_tuple = tuple(proposed_state.tolist())
            proposed_states.append(state_tuple)
            print(f"Proposed state {i}: {proposed_state}")

            # Simulate guidance feedback
            guidance = torch.rand(len(proposed_state), dtype=torch.float32) * 0.5
            searcher.update_value(proposed_state, guidance)
        else:
            print(f"Search exhausted at iteration {i}")
            break

    # Verify no duplicates were proposed
    unique_states = set(proposed_states)
    assert len(unique_states) == len(proposed_states), (
        f"Found duplicate states! Proposed {len(proposed_states)} states "
        f"but only {len(unique_states)} unique"
    )
    print(f"✓ No duplicate states found ({len(proposed_states)} unique proposals)")

    print("HierarchicalAEMCTSSearcher test passed")


def test_generator_factory() -> None:
    """Test that the MCTS generator can be created via factory."""
    generator_class = get_ae_generator_by_name("mct_expansion")
    assert generator_class == MCTExpAEVectorGenerator

    # Test instantiation via factory
    generator = MCTExpAEVectorGenerator(
        num_operators=2,
        arity=1,
        ae_var_ids=[0],
        search_region=[True, True],  # Search both operators
        guidance_threshold=0.1,
        max_iterations=5,
    )

    assert isinstance(generator, MCTExpAEVectorGenerator)
    print("Generator factory test passed")
