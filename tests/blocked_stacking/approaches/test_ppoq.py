"""Tests for PPO-Q policy implementation."""

from pathlib import Path

import pytest

from skill_refactor import register_all_environments
from skill_refactor.approaches.rl_policies.ppo_q import PPOQPolicy
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
from skill_refactor.utils.structs import RLDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skip(reason="The script is used to run experiments locally")
@pytest.mark.parametrize("system_cls", [BlockedStackingRLTAMPSystem])
def test_ppoq_policy_blocked_stacking(system_cls):
    """Test PPO-Q Policy with BlockedStacking environment."""
    data_path = Path("training_data/blocked_stacking/RL_data/scenario_1")
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
        "exp_name": "test_ppoq_blocked_stacking",
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
    envs = MultiEnvRecordVideo(tamp_system.env, "videos/test-ppoq-stacking")

    policy = PPOQPolicy(seed=0)
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

    # Check that Q-networks are initialized
    assert policy.q_network is not None
    assert policy.target_q_network is not None
    assert policy.replay_buffer is not None
    assert not policy.q_learning_enabled  # Should start disabled

    # Load training data
    train_data = RLDataset.load(data_path, num_states=100)  # Small for testing
    envs_mani.configure_training(train_data)

    # Train for a small number of iterations to verify it works
    # Note: For actual training, you'd run policy.train(envs_mani, None, train_data)
    # but for testing we just verify initialization works

    # Test that get_action works
    obs, _ = envs_mani.reset()
    action = policy.get_action(obs)
    assert action.shape == (CFG.num_envs, envs_mani.single_action_space.shape[0])

    # Manually enable Q-learning to test that path
    policy.q_learning_enabled = True
    action_with_q = policy.get_action(obs)
    assert action_with_q.shape == (CFG.num_envs, envs_mani.single_action_space.shape[0])

    # Test save/load
    save_path = Path("test_ppoq_save.pt")
    policy.save(save_path)
    assert save_path.exists()

    policy2 = PPOQPolicy(seed=0)
    policy2.initialize(envs_mani)
    policy2.load(save_path)
    assert policy2._trained

    # Cleanup
    save_path.unlink()


def test_ppoq_replay_buffer():
    """Test the ReplayBuffer implementation."""
    import torch

    from skill_refactor.approaches.rl_policies.ppo_q import ReplayBuffer

    buffer = ReplayBuffer(
        buffer_size=100,
        obs_shape=(10,),
        device=torch.device("cpu"),
    )

    # Test initial state
    assert len(buffer) == 0
    assert not buffer.full

    # Add some transitions (12 iterations × 4 items/batch = 48 items)
    for _ in range(12):
        obs = torch.randn(4, 10)  # Batch of 4
        next_obs = torch.randn(4, 10)
        actions = torch.randint(0, 2, (4,))
        rewards = torch.randn(4)
        dones = torch.zeros(4, dtype=torch.bool)

        buffer.add(obs, next_obs, actions, rewards, dones)

    assert len(buffer) == 48  # 12 batches × 4 items
    assert not buffer.full

    # Test sampling
    sample = buffer.sample(32)
    assert len(sample) == 5  # obs, next_obs, actions, rewards, dones
    assert sample[0].shape == (32, 10)  # observations

    # Fill buffer (need 52 more items, so 13 batches × 4 = 52)
    for _ in range(13):
        obs = torch.randn(4, 10)
        next_obs = torch.randn(4, 10)
        actions = torch.randint(0, 2, (4,))
        rewards = torch.randn(4)
        dones = torch.zeros(4, dtype=torch.bool)

        buffer.add(obs, next_obs, actions, rewards, dones)

    assert len(buffer) == 100  # 48 + 52 = 100
    assert buffer.full


def test_ppoq_q_network():
    """Test the Q-Network forward pass."""
    import torch

    from skill_refactor.approaches.rl_policies.ppo_q import QNetwork

    q_net = QNetwork(obs_shape=(10,))

    # Test forward pass
    obs = torch.randn(32, 10)
    q_values = q_net(obs)

    assert q_values.shape == (32, 2)  # Binary action space

    # Test that Q-values are reasonable (not NaN or Inf)
    assert not torch.isnan(q_values).any()
    assert not torch.isinf(q_values).any()
