"""Unit tests for failure detection learner in ClutteredRoom environment."""

import glob
import logging
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.pred_learner.initiation_function_learner import (
    FailureDataset,
    InitiationFunctionLearner,
)
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.settings import CFG


# @pytest.mark.skip(reason="The script requires local data")
# @pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_train_failure_detector_sc1() -> None:
    """Test training failure detector for ClutteredRoom scenario 1."""
    sc = 1
    seed = 0
    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": str(sc),
        "lll_config": f"config/recovery_chaining/cluttered_room_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "delta_finger_control": False,
        "loglevel": logging.INFO,
        "log_file": f"logs/cluttered_room_failure_det_train_sc{sc}_seed{seed}.log",
    }
    register_all_environments()
    reset_config(test_config)

    # Set up logging
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
    logging.basicConfig(
        level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")

    # Load scenario info
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][sc]

    # Create TAMP system (only needed for interpreter creation later, not for training)
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Load failure detection config
    failure_nn_config_path = scenario_info["failure_learning_cfg_settings"][
        "failure_nn_config"
    ]
    with open(failure_nn_config_path, "r", encoding="utf-8") as f:
        failure_config = yaml.safe_load(f)["failure_detector"]

    # Load collected failure data
    data_path = Path(
        f"training_data/cluttered_room/Failure_data/scenario_{sc}/seed_{seed}/state_labels.pt"
    )
    assert data_path.exists(), f"Data file not found: {data_path}"

    logging.info(f"Loading failure data from {data_path}")
    dataset = FailureDataset.load(data_path)

    # Log dataset statistics
    num_positive = torch.sum(dataset.labels == 1).item()
    num_negative = torch.sum(dataset.labels == 0).item()
    logging.info(
        f"Loaded dataset: {len(dataset.states)} states "
        f"({num_positive} positive, {num_negative} negative)"
    )

    # Split into train/val
    val_split = failure_config.get("val_split", 0.2)
    num_val = int(len(dataset.states) * val_split)
    num_train = len(dataset.states) - num_val

    indices = torch.randperm(len(dataset.states))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = FailureDataset(
        states=[dataset.states[i] for i in train_indices],
        labels=dataset.labels[train_indices],
    )
    val_dataset = FailureDataset(
        states=[dataset.states[i] for i in val_indices],
        labels=dataset.labels[val_indices],
    )

    logging.info(f"Split: {num_train} train, {num_val} val")

    # Get failed operator info
    failed_skill = scenario_info.get("failed_skill", "GoToPickObject")
    failed_objects = scenario_info.get("failed_objects", "robot,transport_obj1").split(
        ","
    )

    # Create learner
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = InitiationFunctionLearner(
        tamp_system=tamp_system,
        failed_op=failed_skill,
        failed_objects=failed_objects,
        config=failure_config,
        device=device,
    )

    logging.info(f"Training on device: {device}")

    # Train
    save_dir = Path(
        f"training_data/cluttered_room/Failure_models/scenario_{sc}/seed_{seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = learner.train(
        train_dataset=train_dataset, val_dataset=val_dataset, save_dir=save_dir
    )

    # Check that training produced reasonable results
    assert len(metrics["train_loss"]) > 0, "No training metrics"
    assert metrics["train_loss"][-1] < metrics["train_loss"][0], "Loss did not decrease"


if __name__ == "__main__":
    # Run test for seed 0
    test_train_failure_detector_sc1()
