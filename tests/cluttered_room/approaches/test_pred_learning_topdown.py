"""Unit Tests for predicate learning in Cluttered Table environment, using topdown
effect supervised neural optimization approach."""

import logging
import os
from pathlib import Path
from typing import List

import pytest
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.pred_learner.sequential_finetuner import (
    SequentialPredicateFinetuner,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.cluttered_room.cluttered_room import (
    ClutteredRoomRLTAMPSystem,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    PlannerDataset,
)


# @pytest.mark.skip(reason="The script is used to run experiments locally")
@pytest.mark.parametrize("seeed", [0, 2, 3])
def test_fixed_predicate_invention_cluttered_room_sc1_or_2_or_3(seeed) -> None:
    """Test the entire predicate invention process in Blocked Stacking environment."""
    sc = "1"
    seed = seeed
    test_config = {
        "delta_finger_control": False,
        "seed": seed,
        "log_file": f"0118_pred_learning_sc{sc}_seed{seed}.log",
        "lll_config": f"config/lifelong_learning/cluttered_room_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
        "pred_net_save_dir": f"trained_pred_nets_sc1_debug3_seed{seed}",
        "loglevel": logging.INFO,
    }
    reset_config(test_config)
    register_all_environments()
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
    # sc1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][int(sc)]

    cfg_settings = scenario_info.get("planner_learning_cfg_settings", {})
    update_config(cfg_settings)
    # Create a simple TAMP system
    tamp_system = ClutteredRoomRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    dataset_path = Path(
        os.path.join(cfg_settings["planner_dataset_path"], f"seed_{CFG.seed}")
    )
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=-1)

    topdown_learner = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        verbose=True,
        scenario=sc.split("_", maxsplit=1)[0],
    )

    topdown_learner.invent()
