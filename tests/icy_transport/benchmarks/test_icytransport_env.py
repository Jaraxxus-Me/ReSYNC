"""Tests for core IcyTransport environment."""

import time

import gymnasium as gym
import numpy as np
import pytest
import torch

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.wrappers import (
    MultiEnvRecordVideo,
    MultiEnvWrapper,
    NormalizeActionMultiEnvWrapper,
)
from skill_refactor.settings import CFG


def test_icy_transport_env_basic():
    """Test basic functionality of IcyTransport environment."""
    test_config = {
        "num_envs": 4,
        "device": "cpu",
    }
    reset_config(test_config)
    register_all_environments()

    def make_env():
        return gym.make("skill_ref/IcyTransport2D-v0")

    if CFG.normalize_action:
        envs = NormalizeActionMultiEnvWrapper(  # type: ignore
            make_env,
            num_envs=CFG.num_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
    else:
        envs = MultiEnvWrapper(
            make_env,
            num_envs=CFG.num_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )

    # envs = MultiEnvRecordVideo(
    #     envs,
    #     f"videos/rand_actions_icy_transport/",
    #     episode_trigger=lambda episode_id: True,
    # )

    # Test initial state sampling
    _, _ = envs.reset(seed=0)
    forward_action = envs.action_space.sample()
    forward_action[:, 0] = 0.0  # full forward throttle
    forward_action[:, 1] = 1.0

    # Test random actions
    s = time.time()
    for _ in range(150):
        _, _, done, truncated, _ = envs.step(forward_action)
        if done.any() or truncated.any():
            _, _ = envs.reset()
            break
    print("Stepping time for 400 steps:", time.time() - s)
    envs.close()


def test_icy_transport_env_deterministic():
    """Test that environment is deterministic with same seed."""
    reset_config({})
    register_all_environments()

    env1 = gym.make("skill_ref/IcyTransport2D-v0")
    env2 = gym.make("skill_ref/IcyTransport2D-v0")

    # Reset with same seed
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    assert np.array_equal(
        obs1, obs2
    ), "Observations should match after reset with same seed"

    env1.close()
    env2.close()


def test_parallel_icy_transport():
    """Test MultiEnvWrapper with IcyTransport environments."""
    test_config = {
        "num_envs": 4,
    }
    reset_config(test_config)
    register_all_environments()

    # Create environment factory function
    def env_fn():
        return gym.make("skill_ref/IcyTransport2D-v0")

    # Test with parallel environments
    multi_env = MultiEnvWrapper(env_fn, num_envs=CFG.num_envs)

    # Test observation and action spaces
    single_env = env_fn()
    single_env.reset()

    # Check that spaces are properly batched
    assert (
        multi_env.observation_space.shape
        == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Observation space should be batched"
    assert (
        multi_env.action_space.shape == (CFG.num_envs,) + single_env.action_space.shape
    ), "Action space should be batched"

    # Test reset
    obs_batch, info_batch = multi_env.reset(seed=42)
    assert (
        obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Observation batch shape incorrect"
    assert len(info_batch) >= 0, "Info dict should exist"

    # Note: goal_room and init_obj_room not yet implemented (no transport object yet)

    # Test step
    actions = multi_env.action_space.sample()
    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)

    assert (
        obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Step obs shape incorrect"
    assert rewards.shape == (CFG.num_envs,), "Rewards shape should be (num_envs,)"
    assert terminated.shape == (CFG.num_envs,), "Terminated shape should be (num_envs,)"
    assert truncated.shape == (CFG.num_envs,), "Truncated shape should be (num_envs,)"
    assert len(info_batch) >= 0, "Info dict should exist after step"

    # Test render - Skip for now as CarRobotType rendering not implemented yet
    # _ = multi_env.render()

    # Clean up
    multi_env.close()
    single_env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_parallel_icy_transport_tensor():
    """Test MultiEnvWrapper with torch tensor support."""

    # Create environment factory function
    test_config = {
        "num_envs": 4,
        "device": "cuda:0",
    }
    reset_config(test_config)
    register_all_environments()

    def env_fn():
        return gym.make("skill_ref/IcyTransport2D-v0")

    # Test with tensor support enabled
    multi_env = MultiEnvWrapper(
        env_fn, num_envs=CFG.num_envs, to_tensor=True, device=CFG.device
    )
    single_env = env_fn()
    single_env.reset()

    # Check that spaces are properly batched (still numpy-based)
    assert (
        multi_env.observation_space.shape
        == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Observation space should be batched"
    assert (
        multi_env.action_space.shape == (CFG.num_envs,) + single_env.action_space.shape
    ), "Action space should be batched"

    # Test reset - should return tensor
    obs_batch, info_batch = multi_env.reset(seed=42)
    assert torch.is_tensor(
        obs_batch
    ), "Observations should be tensors when to_tensor=True"
    assert obs_batch.device == torch.device(
        "cuda:0"
    ), "Tensor should be on correct device"
    assert (
        obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Observation tensor shape incorrect"
    assert len(info_batch) >= 0, "Info dict should exist"

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
    assert (
        obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Step obs shape incorrect"
    assert rewards.shape == (CFG.num_envs,), "Rewards shape incorrect"
    assert terminated.shape == (CFG.num_envs,), "Terminated shape incorrect"
    assert truncated.shape == (CFG.num_envs,), "Truncated shape incorrect"

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
    assert (
        obs_batch.shape == (CFG.num_envs,) + single_env.observation_space.shape
    ), "Observation shape with tensor actions incorrect"

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
