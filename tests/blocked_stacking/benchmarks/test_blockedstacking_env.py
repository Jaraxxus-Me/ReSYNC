"""Tests for core BlockedStacking environment."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.wrappers import MultiEnvWrapper
from skill_refactor.settings import CFG


def test_blocked_stacking_env_blocked():
    """Test basic functionality of BlockedStacking environment with obstructions."""
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
    }
    reset_config(test_config)
    register_all_environments()

    env = gym.make("skill_ref/BlockedStacking2D-v0")

    # Test initial state sampling
    _, _ = env.reset(seed=0)

    # Test random actions
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            _, _ = env.reset()


def test_parallel_blocked_stacking():
    """Test MultiEnvWrapper with BlockedStacking environments."""
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "num_envs": 16,
    }
    reset_config(test_config)
    register_all_environments()

    # Create environment factory function
    def env_fn():
        return gym.make("skill_ref/BlockedStacking2D-v0")

    # Test with 2 parallel environments
    multi_env = MultiEnvWrapper(env_fn, num_envs=CFG.num_envs)

    # Test observation and action spaces
    single_env = env_fn()
    _, _ = single_env.reset()

    # Check that spaces are properly batched
    assert (
        multi_env.observation_space.shape
        == (CFG.num_envs,) + single_env.observation_space.shape
    )
    assert (
        multi_env.action_space.shape == (CFG.num_envs,) + single_env.action_space.shape
    )

    # Test reset
    obs_batch, info_batch = multi_env.reset(seed=42)
    assert obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    assert len(info_batch) >= 0  # Info dict should exist

    # Test step
    actions = multi_env.action_space.sample()
    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)

    assert obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    assert rewards.shape == (CFG.num_envs,)
    assert terminated.shape == (CFG.num_envs,)
    assert truncated.shape == (CFG.num_envs,)
    assert len(info_batch) >= 0

    # Test render - should return tiled image or None
    _ = multi_env.render()

    # Clean up
    multi_env.close()
    single_env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_parallel_blocked_stacking_tensor():
    """Test MultiEnvWrapper with torch tensor support."""

    # Create environment factory function
    test_config = {
        "obstruction1_blocking_grasp": True,
        "obstruction1_blocking_stacking": False,
        "num_envs": 16,
        "device": "cuda:0",
    }
    reset_config(test_config)
    register_all_environments()

    def env_fn():
        return gym.make("skill_ref/BlockedStacking2D-v0")

    # Test with tensor support enabled
    multi_env = MultiEnvWrapper(
        env_fn, num_envs=CFG.num_envs, to_tensor=True, device=CFG.device
    )
    single_env = env_fn()
    _, _ = single_env.reset()

    # Check that spaces are properly batched (still numpy-based)
    assert (
        multi_env.observation_space.shape
        == (CFG.num_envs,) + single_env.observation_space.shape
    )
    assert (
        multi_env.action_space.shape == (CFG.num_envs,) + single_env.action_space.shape
    )

    # Test reset - should return tensor
    obs_batch, info_batch = multi_env.reset(seed=42)
    assert torch.is_tensor(
        obs_batch
    ), "Observations should be tensors when to_tensor=True"
    assert obs_batch.device == torch.device(
        "cuda:0"
    ), "Tensor should be on correct device"
    assert obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    assert len(info_batch) >= 0

    # Check info batching for scalar numeric values
    for key, value in info_batch.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.shape[0] == CFG.num_envs
            ), f"Info '{key}' should be batched with shape[0]={CFG.num_envs}"
            assert value.device == torch.device(
                "cuda:0"
            ), f"Info tensor '{key}' should be on correct device"
        elif isinstance(value, np.ndarray):
            assert (
                value.shape[0] == CFG.num_envs
            ), f"Info array '{key}' should be batched with shape[0]={CFG.num_envs}"

    # Test step with numpy actions - should work with automatic conversion
    actions_np = multi_env.action_space.sample()
    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions_np)

    assert torch.is_tensor(obs_batch), "Observations should be tensors"
    assert torch.is_tensor(rewards), "Rewards should be tensors"
    assert torch.is_tensor(terminated), "Terminated should be tensors"
    assert torch.is_tensor(truncated), "Truncated should be tensors"
    assert obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    assert rewards.shape == (CFG.num_envs,)
    assert terminated.shape == (CFG.num_envs,)
    assert truncated.shape == (CFG.num_envs,)

    # Check info batching after step
    for key, value in info_batch.items():
        if isinstance(value, torch.Tensor):
            assert (
                value.shape[0] == CFG.num_envs
            ), f"Info '{key}' should be batched with shape[0]={CFG.num_envs}"
            assert value.device == torch.device(
                "cuda:0"
            ), f"Info tensor '{key}' should be on correct device"
        elif isinstance(value, np.ndarray):
            assert (
                value.shape[0] == CFG.num_envs
            ), f"Info array '{key}' should be batched with shape[0]={CFG.num_envs}"

    # Test step with tensor actions
    actions_tensor = torch.from_numpy(multi_env.action_space.sample()).float()
    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(
        actions_tensor
    )

    assert torch.is_tensor(obs_batch), "Observations should be tensors"
    assert torch.is_tensor(rewards), "Rewards should be tensors"
    assert obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape

    # Test without tensor support for comparison
    multi_env_numpy = MultiEnvWrapper(env_fn, num_envs=CFG.num_envs, to_tensor=False)
    obs_numpy, _ = multi_env_numpy.reset(seed=42)
    assert isinstance(
        obs_numpy, np.ndarray
    ), "Should return numpy array when to_tensor=False"

    # Clean up
    multi_env.close()
    multi_env_numpy.close()
    single_env.close()
