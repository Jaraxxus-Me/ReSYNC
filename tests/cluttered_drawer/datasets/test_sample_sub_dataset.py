"""Unit Tests for predicate learning in Cluttered Table environment, using topdown
effect supervised neural optimization approach."""

from pathlib import Path

from skill_refactor.utils.structs import (
    PlannerDataset,
)


# @pytest.mark.skip(reason="The script is used to run experiments locally")
def test_sample_sub_dataset():
    """Test the entire predicate invention process in Blocked Stacking environment."""

    planner_dataset_path = (
        "training_data/cluttered_drawer/Planner_data/scenario12_2/seed_1_full"
    )

    dataset_path = Path(planner_dataset_path)
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=-1)

    sub_dataset = planner_dataset.sample_sub_dataset(traj_per_scenario=5)
    sub_dataset_path = Path(
        "training_data/cluttered_drawer/Planner_data/scenario12_2/seed_1"
    )
    sub_dataset.save(sub_dataset_path)
