"""Tests for PPO-C policy implementation."""

from pathlib import Path

import pytest
import torch

from skill_refactor import register_all_environments
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skip(reason="The script is used to run experiments locally")
@pytest.mark.parametrize("system_cls", [BlockedStackingRLTAMPSystem])
def test_ppoc_policy_blocked_stacking(system_cls):
    """Test PPO-C Policy with BlockedStacking environment."""
    test_config = {
        "debug_env": True,
        "num_envs": 8,  # Smaller for faster testing
        "max_rl_steps": 20,
        "rl_static_steps": 3,
        "max_env_steps": 180,
        "obstruction_blocking_grasp_prob": 1.0,
        "obstruction_blocking_stacking_prob": 0.0,
        "lll_config": "config/lifelong_learning/blocked_stacking.yaml",
        "rl_config": "config/pure_rl/blocked_stacking_ppo_n20_finite_clip01.yaml",
        "control_mode": "pd_joint_delta_pos",
        "exp_name": "test_ppoc_blocked_stacking",
    }
    reset_config(test_config)
    register_all_environments()
    scenario_info = {
        "tgt_skill": "Punch(robot,block,obstruction)",
        "train_objects": "robot,grasp_block,obstruction",
        "max_skill_steps": 100,
    }
    # Create TAMP system
    tamp_system = system_cls.create_default(render_mode="rgb_array", seed=42)
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    envs = tamp_system.env
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/test-ppoc-stacking")

    policy = PPOCPolicy(seed=0)
    envs_mani = PlanningStatesVectorEnv(
        envs,
        tamp_system.perceiver,
        scenario_info,
        planner,
        CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    policy.initialize(envs_mani)

    # Check that PPO agent and classifier are initialized
    assert policy.agent is not None
    assert policy.terminal_cls is not None
    assert policy.replay_buffer is not None
    assert not policy.c_learning_enabled  # Should start disabled
    assert not policy._trained

    # Test that get_action works (PPO only mode)
    obs, _ = envs_mani.reset()
    action = policy.get_action(obs)
    assert action.shape == (CFG.num_envs, envs_mani.single_action_space.shape[0])

    # Manually enable classifier to test that path
    policy.c_learning_enabled = True
    action_with_cls = policy.get_action(obs)
    assert action_with_cls.shape == (
        CFG.num_envs,
        envs_mani.single_action_space.shape[0],
    )

    # Test terminate function
    terminate_flags = policy.terminate(obs)
    assert terminate_flags.shape == (CFG.num_envs,)
    assert terminate_flags.dtype == torch.bool

    # Test save/load
    save_path = Path("test_ppoc_save.pt")
    policy.save(save_path)
    assert save_path.exists()

    policy2 = PPOCPolicy(seed=0)
    policy2.initialize(envs_mani)
    policy2.load(save_path)
    assert policy2._trained

    # Cleanup
    save_path.unlink()


def test_ppoc_replay_buffer():
    """Test the ReplayBuffer implementation for PPO-C."""
    from skill_refactor.approaches.rl_policies.ppo_c import ReplayBuffer

    buffer = ReplayBuffer(
        buffer_size=100,
        obs_shape=(10,),
        device=torch.device("cpu"),
        train_split=0.8,
    )

    # Test initial state
    assert len(buffer) == 0
    assert not buffer.full
    assert buffer.train_split == 0.8

    # Add some transitions (12 iterations × 4 items/batch = 48 items)
    for _ in range(12):
        obs = torch.randn(4, 10)  # Batch of 4
        rewards = torch.randint(0, 2, (4,)).float()  # Binary rewards (0 or 1)
        dones = torch.zeros(4, dtype=torch.bool)

        buffer.add(obs, rewards, dones)

    assert len(buffer) == 48  # 12 batches × 4 items
    assert not buffer.full

    # Verify train/val split is applied
    # Each sample has 0.8 probability of being in train set
    # With 48 samples, we expect roughly 38-40 in train and 8-10 in val
    train_count = buffer.is_train[:48].sum().item()
    val_count = 48 - train_count
    assert 30 <= train_count <= 46  # Allow variance due to randomness
    assert 2 <= val_count <= 18

    # Test sampling from full buffer (should sample balanced 50/50 positive/negative)
    sample = buffer.sample(32)
    assert len(sample) == 3  # obs, rewards, dones
    obs_sample, rewards_sample, dones_sample = sample
    assert obs_sample.shape == (32, 10)
    assert rewards_sample.shape == (32,)
    assert dones_sample.shape == (32,)

    # Check balanced sampling (should have roughly 16 positive, 16 negative)
    positive_count = (rewards_sample == 1).sum().item()
    negative_count = (rewards_sample == 0).sum().item()
    assert positive_count == 16
    assert negative_count == 16

    # Test train/val split sampling
    train_sample = buffer.sample_train(16)
    assert train_sample[0].shape == (16, 10)

    val_sample = buffer.sample_val(16)
    assert val_sample[0].shape == (16, 10)

    # Fill buffer (need 52 more items)
    for _ in range(13):
        obs = torch.randn(4, 10)
        rewards = torch.randint(0, 2, (4,)).float()
        dones = torch.zeros(4, dtype=torch.bool)

        buffer.add(obs, rewards, dones)

    assert len(buffer) == 100
    assert buffer.full

    # Test save/load
    save_path = Path("test_replay_buffer.pt")
    buffer.save(save_path)
    assert save_path.exists()

    # Create new buffer and load
    buffer2 = ReplayBuffer(
        buffer_size=100,
        obs_shape=(10,),
        device=torch.device("cpu"),
        train_split=0.8,
    )
    buffer2.load(save_path)

    assert len(buffer2) == 100
    assert buffer2.full
    assert buffer2.pos == buffer.pos
    assert torch.allclose(buffer2.observations, buffer.observations)
    assert torch.allclose(buffer2.rewards, buffer.rewards)
    assert torch.equal(buffer2.dones, buffer.dones)
    assert torch.equal(buffer2.is_train, buffer.is_train)

    # Cleanup
    save_path.unlink()


def test_ppoc_classifier_network():
    """Test the ClassifierNetwork forward pass."""
    from skill_refactor.approaches.rl_policies.ppo_c import ClassifierNetwork

    classifier = ClassifierNetwork(obs_shape=(10,))

    # Test forward pass
    obs = torch.randn(32, 10)
    logits = classifier(obs)

    assert logits.shape == (32, 2)  # Binary classification

    # Test that logits are reasonable (not NaN or Inf)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # Test that argmax works (simulating inference)
    predictions = torch.argmax(logits, dim=1)
    assert predictions.shape == (32,)
    assert predictions.min() >= 0
    assert predictions.max() <= 1
