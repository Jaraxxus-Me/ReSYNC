"""Regression checks for scenario-specific predicate training data."""

from unittest.mock import Mock, call, patch, sentinel

from skill_refactor.approaches.pred_learner.individual_recomposer import (
    IndividualPredicateRecomposer,
)


def test_candidate_generation_partitions_trajectories_by_scenario() -> None:
    learner = Mock(spec=IndividualPredicateRecomposer)
    learner.init_predicate_ae_vectors = {}
    learner.all_scenarios = ["1", "2"]
    learner.predicate_configures = []
    first = Mock(train_scenario_idx="1")
    second = Mock(train_scenario_idx="2")
    unrelated = Mock(train_scenario_idx="3")
    learner._ground_atom_dataset = [
        (first, sentinel.first_atoms),
        (second, sentinel.second_atoms),
        (unrelated, sentinel.unrelated_atoms),
    ]
    learner._create_transition_dataset.return_value = []

    with patch(
        "skill_refactor.approaches.pred_learner.individual_recomposer.segment_trajectory",
        side_effect=[[sentinel.first_segment], [sentinel.second_segment]],
    ) as segment:
        IndividualPredicateRecomposer._generate_candidates(learner)

    assert segment.call_args_list == [
        call(first, sentinel.first_atoms),
        call(second, sentinel.second_atoms),
    ]
    assert learner._create_transition_dataset.call_args_list == [
        call([[sentinel.first_segment]]),
        call([[sentinel.second_segment]]),
    ]
