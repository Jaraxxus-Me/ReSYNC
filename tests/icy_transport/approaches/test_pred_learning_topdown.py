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
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    PlannerDataset,
)


@pytest.mark.skip(reason="The script is used to run experiments locally")
def test_fixed_predicate_invention_icy_transport_sc1_or_2_or_3() -> None:
    """Test the entire predicate invention process in Icy Transport environment."""
    sc = "1"
    test_config = {
        "seed": 0,
        "control_mode": "force_torque",
        "log_file": f"0101_pred_search_sc{sc}.log",
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
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
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    dataset_path = Path(os.path.join(CFG.planner_dataset_path, f"seed_{CFG.seed}"))
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=10)

    topdown_learner = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        verbose=True,
        scenario=sc.split("_", maxsplit=1)[0],
    )

    topdown_learner.invent()


@pytest.mark.skip(reason="The script is used to run experiments locally")
def test_fixed_predicate_invention_icy_transport_sc12_2() -> None:
    """Test the entire predicate invention process in Icy Transport environment."""
    sc = "12_2"
    test_config = {
        "seed": 2,
        "log_file": f"1205_{sc}_research200.log",
        "lll_config": f"config/lifelong_learning/icy_transport_sc12_2_seed2.yaml",
        "control_mode": "pd_joint_delta_pos",
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
        lll_config_data = yaml.safe_load(f)

    # Create TAMP system for the latest scenario
    scenario_info2 = lll_config_data["scenarios"][int(sc.rsplit("_", maxsplit=1)[-1])]
    world_setting = scenario_info2.get("planner_learning_cfg_settings", {})
    update_config(world_setting)

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    dataset_path = Path(
        os.path.join(world_setting["planner_dataset_path"], f"seed_{CFG.seed}")
    )
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=10)

    # 1). Use latest planner dataset to fine-tune predicates from all previous scenarios
    # for scenario_id, scenario_info in lll_config_data["scenarios"].items():
    #     if scenario_info.get("trained"):
    #         # For all the trained scenarios, use the current data to fine-tine the predicates
    #         predicate_config = scenario_info["planner_learning_cfg_settings"]\
    #             .get("predicate_config", None)
    #         # Load trajectory dataset
    #         dataset_path_str = scenario_info["planner_learning_cfg_settings"]\
    #             .get("planner_dataset_path")
    #         if dataset_path_str is None:
    #             raise ValueError("dataset_path must be specified in learning_info")
    #         old_dataset_path = Path(dataset_path_str)
    #         num_traj = scenario_info["planner_learning_cfg_settings"].get("planner_num_traj", -1)
    #         old_planner_dataset = PlannerDataset.load(old_dataset_path, num_traj=num_traj)
    #         assert predicate_config is not None
    #         with open(predicate_config, "rb") as f:
    #             old_config_data = yaml.safe_load(f)
    #         old_predicate_configures = old_config_data["predicates"]
    #         sequential_finetuner = SequentialPredicateFinetuner(
    #             old_planner_dataset=old_planner_dataset,
    #             scenario=str(scenario_id),
    #             dataset=planner_dataset,
    #             tamp_system=tamp_system,
    #             predicate_configures=old_predicate_configures,
    #             quantify_basic_predicates=False,
    #             verbose=True,
    #         )
    #         sequential_finetuner.finetune()

    # 2). Instantiate approach to load these fine-tuned predicates
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        if scenario_info.get("trained"):
            approach.update_domain_knowledge(scenario_info)

    # 3). Finally, invent new predicates using the latest dataset and latest approach perceiver
    topdown_learner = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        latest_perceiver=approach.perceiver,
        quantify_basic_predicates=False,
        verbose=True,
        scenario=sc.rsplit("_", maxsplit=1)[-1],
    )

    invented_pred_interpr_sofar = approach.get_invented_predicate_interpretr_so_far()

    invented_pred_interpr, op_set = topdown_learner.invent()

    invented_pred_interpr_sofar.update(invented_pred_interpr)
    filtered_op_set = topdown_learner.filter_preconditions(
        op_set,
        invented_pred_interpr_sofar,
        dataset=planner_dataset,
        num_traj_per_scenario=5,
    )
    json_path = Path(
        os.path.join(
            CFG.pred_net_save_dir, CFG.invented_pred_op_json + f"_sc2_filtered.json"
        )
    )
    topdown_learner.save_invented_predicates_and_operators(
        invented_pred_interpr, filtered_op_set, json_path
    )


@pytest.mark.skip(reason="The script is used to run experiments locally")
def test_fixed_predicate_invention_icy_transport_sc123_3() -> None:
    """Test the entire predicate invention process in Icy Transport environment."""
    sc = "123_3"
    test_config = {
        "log_file": f"1124_pred_learning_sc{sc}.log",
        "lll_config": f"config/lifelong_learning/icy_transport_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
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
        lll_config_data = yaml.safe_load(f)

    # Create TAMP system for the latest scenario
    scenario_info2 = lll_config_data["scenarios"][int(sc.rsplit("_", maxsplit=1)[-1])]
    world_setting = scenario_info2.get("planner_learning_cfg_settings", {})
    update_config(world_setting)

    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    dataset_path = Path(world_setting["planner_dataset_path"])
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=-1)
    use_dataset = planner_dataset.sample_sub_dataset(traj_per_scenario=20)
    planner_dataset = use_dataset

    # 1). Use latest planner dataset to fine-tune predicates from all previous scenarios
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        if scenario_info.get("trained"):
            # For all the trained scenarios, use the current data to fine-tine the predicates
            predicate_config = scenario_info["planner_learning_cfg_settings"].get(
                "predicate_config", None
            )
            # Load trajectory dataset
            dataset_path_str = scenario_info["planner_learning_cfg_settings"].get(
                "planner_dataset_path"
            )
            if dataset_path_str is None:
                raise ValueError("dataset_path must be specified in learning_info")
            old_dataset_path = Path(dataset_path_str)
            num_traj = scenario_info["planner_learning_cfg_settings"].get(
                "planner_num_traj", -1
            )
            old_planner_dataset = PlannerDataset.load(
                old_dataset_path, num_traj=num_traj
            )
            assert predicate_config is not None
            with open(predicate_config, "rb") as f:
                old_config_data = yaml.safe_load(f)
            old_predicate_configures = old_config_data["predicates"]
            sequential_finetuner = SequentialPredicateFinetuner(
                old_planner_dataset=old_planner_dataset,
                scenario=str(scenario_id),
                dataset=planner_dataset,
                tamp_system=tamp_system,
                predicate_configures=old_predicate_configures,
                quantify_basic_predicates=False,
                verbose=True,
            )
            sequential_finetuner.finetune()

    # 2). Instantiate approach to load these fine-tuned predicates
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)
    for scenario_id, scenario_info in lll_config_data["scenarios"].items():
        # Update CFG with planner_learning_cfg_settings (right after planner refactor in sc1)
        cfg_settings_after_sc = scenario_info.get("planner_learning_cfg_settings", {})
        update_config(cfg_settings_after_sc)
        latest_tamp_system = IcyTransportRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        approach.update_learning_info(
            int(scenario_id),
            scenario_info,
            latest_tamp_system=latest_tamp_system,
        )
        if scenario_info.get("trained"):
            approach.update_domain_knowledge(scenario_info)

    # 3). Finally, invent new predicates using the latest dataset and latest approach perceiver
    topdown_learner = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        latest_perceiver=approach.perceiver,
        quantify_basic_predicates=False,
        verbose=True,
        scenario=sc.rsplit("_", maxsplit=1)[-1],
    )

    invented_pred_interpr_sofar = approach.get_invented_predicate_interpretr_so_far()

    invented_pred_interpr, op_set = topdown_learner.invent()

    invented_pred_interpr_sofar.update(invented_pred_interpr)
    filtered_op_set = topdown_learner.filter_preconditions(
        op_set,
        invented_pred_interpr_sofar,
        dataset=planner_dataset,
        num_traj_per_scenario=5,
    )
    json_path = Path(
        os.path.join(
            CFG.pred_net_save_dir, CFG.invented_pred_op_json + f"_sc{sc}_filtered.json"
        )
    )
    topdown_learner.save_invented_predicates_and_operators(
        invented_pred_interpr, filtered_op_set, json_path
    )
