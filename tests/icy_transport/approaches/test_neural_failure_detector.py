"""Test neural failure detector integration in RecoveryChainApproach."""

import logging
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.recovery_chain import RecoveryChainApproach
from skill_refactor.args import reset_config, update_config
from skill_refactor.benchmarks.icy_transport.icy_transport import (
    IcyTransportRLTAMPSystem,
)
from skill_refactor.settings import CFG


@pytest.mark.skip(reason="The script requires local data")
def test_neural_failure_detector_loads() -> None:
    """Test that neural failure detector is loaded during domain knowledge update."""
    sc = 1
    seed = 0

    # Check that trained model and policy exist
    model_path = Path(
        f"training_data/icy_transport/Failure_models/scenario_{sc}/seed_{seed}/best_model.pt"
    )
    policy_path = Path(
        f"trained_policies/icy_transport/skill_0101_i_transport_sc1_seed{seed}/best_ppo_ckpt.pt"
    )
    if not model_path.exists():
        pytest.skip(f"Trained model not found at {model_path}")
    if not policy_path.exists():
        pytest.skip(f"Trained policy not found at {policy_path}")

    test_config = {
        "seed": seed,
        "num_envs": 1,
        "scenario": str(sc),
        "lll_config": f"config/recovery_chaining/icy_transport_sc{sc}.yaml",
        "control_mode": "force_torque",
        "pre_trained_policy_path": f"trained_policies/icy_transport/skill_0101_i_transport_sc1_seed{seed}/best_ppo_ckpt.pt",
        "loglevel": logging.INFO,
    }
    register_all_environments()
    reset_config(test_config)

    # Set up logging
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    logging.basicConfig(
        level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Load scenario info
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        lll_config_data = yaml.safe_load(f)
    scenario_info = lll_config_data["scenarios"][sc]

    # Update config with failure learning settings (includes failured_det_operator_objects)
    failure_learning_cfg = scenario_info.get("failure_learning_cfg_settings", {})
    update_config(failure_learning_cfg)

    # Create TAMP system
    tamp_system = IcyTransportRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    # Create approach
    approach = RecoveryChainApproach(tamp_system, seed=CFG.seed)

    # Update learning info
    approach.update_learning_info(sc, scenario_info)

    # Verify that failure detector is initially not loaded for this scenario
    assert (
        sc not in approach._failure_detectors
    ), "Failure detector should not exist before update_domain_knowledge"

    # Update domain knowledge (this should load the neural classifier)
    approach.update_domain_knowledge(scenario_info)

    # Verify that failure detector was loaded
    assert (
        sc in approach._failure_detectors
    ), "Failure detector should be loaded after update_domain_knowledge"

    logging.info("✓ Neural failure detector loaded successfully")
    logging.info(
        "✓ Integration test passed - neural classifier is now being used instead of hand-coded thresholds"
    )


if __name__ == "__main__":
    test_neural_failure_detector_loads()
    print("\n✅ Neural failure detector integration test passed!")
