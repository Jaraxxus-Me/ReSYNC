"""General wrapper for environments supporting improvisational policies."""

import os
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import logger
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.monitoring import video_recorder
from gymnasium.wrappers.record_video import capped_cubic_video_schedule
from mani_skill.envs.sapien_env import BaseEnv as ManiSkillBaseEnv
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.wrappers.record import RecordEpisode  # type: ignore

from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    get_frozen_action,
    get_normalize_action_range,
)
from skill_refactor.utils.structs import RLDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiEnvWrapper(gym.Env):
    """A batched single-Env wrapper over multiple Gym environments.

    This class exposes N identical sub-environments as ONE `gym.Env` whose
    observation_space and action_space are the batched versions of the
    single-env spaces (via `batch_space`). This makes it compatible with
    wrappers like `gymnasium.wrappers.RecordVideo` that expect `gym.Env`,
    while still enabling vectorized stepping.

    It supports optional PyTorch tensor IO for Deep RL training, and a
    tiled `rgb_array` render for video recording.

    Args:
        env_fn: A callable that creates a single environment instance
        num_envs: Number of sub-environments to create
        auto_reset: Whether to automatically reset terminated environments
            (default: True)
        to_tensor: If True, observations and returns will be converted to PyTorch
            tensors, and tensor actions will be accepted (default: False)
        device: Device to place tensors on if to_tensor=True (default: "cpu")
        render_mode: Render mode; should be "rgb_array" to use RecordVideo

    Example:
        >>> import prbench
        >>> prbench.register_all_environments()
        >>> env_fn = lambda: prbench.make(
        ...     "prbench/StickButton2D-b5-v0", render_mode="rgb_array")
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4, render_mode="rgb_array")
        >>> obs_batch, info_batch = multi_env.reset(seed=123)
        >>> obs_batch.shape
        (4, observation_dim)
        >>> actions = multi_env.action_space.sample()
        >>> obs_batch, rewards, terminated, truncated, info_batch = (
        ...     multi_env.step(actions))

        With tensor support:
        >>> multi_env = MultiEnvWrapper(
        ...     env_fn, num_envs=4, to_tensor=True, device="cuda",
        ...     render_mode="rgb_array")
        >>> obs_batch, _ = multi_env.reset()  # returns torch.Tensor on cuda
        >>> actions = torch.randn(
        ...     (4, *multi_env.single_action_space.shape), device="cuda")
        >>> obs, rewards, done, truncated, _ = multi_env.step(actions)
    """

    # Make sure RecordVideo recognizes rgb_array rendering
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        max_episode_steps: int | None = None,
        auto_reset: bool = True,
        to_tensor: bool = False,
        device: str = "cpu",
        render_mode: str | None = "rgb_array",
    ):

        super().__init__()
        self.env_fn = env_fn
        self.num_envs = int(num_envs)
        assert self.num_envs >= 1, "num_envs must be >= 1"
        self.auto_reset = auto_reset
        self.to_tensor = to_tensor
        self.device = device
        self.render_mode = render_mode

        # Create all sub-environments
        # TIP: Prefer env_fn that accepts render_mode="rgb_array" for recording.
        self.envs = [env_fn() for _ in range(self.num_envs)]

        # Spaces
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        assert isinstance(
            self.single_observation_space, gym.spaces.Box
        ), "Only Box observation space is supported"
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Validate homogeneous spaces
        for i, env in enumerate(self.envs):
            assert env.action_space == self.single_action_space, (
                f"Environment {i} has different action space: {env.action_space} "
                f"vs expected {self.single_action_space}"
            )
            assert env.observation_space == self.single_observation_space, (
                f"Environment {i} has different observation space: "
                f"{env.observation_space} vs expected {self.single_observation_space}"
            )

        # Buffers
        self._observations = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._env_needs_reset = np.ones((self.num_envs,), dtype=np.bool_)

        # Copy metadata and annotate autoreset status
        self.metadata = dict(getattr(self.envs[0], "metadata", {}))
        self.metadata["render_modes"] = list(
            set(self.metadata.get("render_modes", []) + ["rgb_array"])
        )
        self.metadata["autoreset_mode"] = "next_step" if auto_reset else "disabled"

        elapsed_steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.elapsed_steps = self._to_tensor(elapsed_steps)
        self._max_episode_steps = max_episode_steps
        if max_episode_steps is not None:
            print(
                "Warning: max_episode_steps is now enforced by "
                "MultiEnvWrapper, will ignore per env truncation."
            )

    # ------------------------- Utilities -------------------------

    def _to_tensor(self, array: np.ndarray) -> np.ndarray | torch.Tensor:
        if self.to_tensor:
            return torch.from_numpy(array).to(self.device)
        return array

    def _to_numpy(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return data

    # --------------------------- API -----------------------------

    def reset(
        self, *, seed: int | Sequence[int] | None = None, options: dict | None = None
    ) -> tuple[np.ndarray | torch.Tensor, dict]:
        """Reset all sub-environments and return batched observation and info."""
        # Distribute seeds
        if seed is not None:
            if isinstance(seed, int):
                seeds_final: list[int | None] = [seed + i for i in range(self.num_envs)]
            else:
                seed = list(seed)
                assert (
                    len(seed) == self.num_envs
                ), f"Seed list length {len(seed)} doesn't match num_envs {self.num_envs}"
                seeds_final = seed  # type: ignore
        else:
            seeds_final = [None] * self.num_envs

        # Reset
        infos: dict[str, Any] = {}
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds_final)):
            # NOTE: Need to handle reset init_states properly here
            # for each sub-env. we assume the init_states must be
            # provided as a batch of states for all sub-envs.
            local_options = None
            if options is not None:
                local_options = dict(options)
                if "init_state" in options.keys():
                    assert (
                        isinstance(options["init_state"], (np.ndarray, torch.Tensor))
                        and options["init_state"].shape[0] == self.num_envs
                    ), (
                        "If providing init_state in options, it must be a "
                        "batch of states for all sub-envs"
                    )
                    local_options = dict(options)
                    local_options["init_state"] = self._to_numpy(
                        options["init_state"][i]
                    )

            obs, info = env.reset(seed=env_seed, options=local_options)
            # Write obs into buffer
            self._observations[i] = obs

            # Batch info
            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs
                infos[key][i] = value

        # Convert info lists to arrays for scalar values (float/int)
        for key, value_list in list(infos.items()):
            if all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in value_list
                if v is not None
            ):
                array = np.array(value_list, dtype=np.float32)
                infos[key] = self._to_tensor(array)

        # Reset trackers
        self._env_needs_reset.fill(False)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._rewards.fill(0.0)
        self.elapsed_steps = self._to_tensor(np.zeros((self.num_envs,), dtype=np.int32))

        observations = np.array(self._observations)
        return self._to_tensor(observations), infos

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Step all sub-environments with batched actions."""
        actions_np = self._to_numpy(actions)
        assert self.envs[0].action_space.contains(
            actions_np[0]
        ), "Actions not in action space"
        self.elapsed_steps += 1

        infos: dict[str, Any] = {}

        for i, env in enumerate(self.envs):
            # Auto-reset paths
            if self._env_needs_reset[i] and self.auto_reset:
                obs, reset_info = env.reset()
                self._observations[i] = obs
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
                self._env_needs_reset[i] = False
                self.elapsed_steps[i] = 0

                for key, value in reset_info.items():
                    if key not in infos:
                        infos[key] = [None] * self.num_envs
                    infos[key][i] = value
                continue

            # Normal step
            action = actions_np[i]
            obs, reward, terminated, truncated, info = env.step(action)

            self._observations[i] = obs
            self._rewards[i] = np.float32(reward)
            if self._max_episode_steps is None:
                self._terminations[i] = bool(terminated)
            else:
                # NOTE: If max_episode_steps is set, we ignore env-provided
                # termination signal to avoid inconsistency across sub-envs.
                self._terminations[i] = False
            if self._max_episode_steps is not None:
                truncated = self.elapsed_steps[i].item() >= self._max_episode_steps
            self._truncations[i] = bool(truncated)

            # If done, mark for auto-reset next call
            # NOTE: We ignore env-provided termination and truncation
            # if max_episode_steps is set, since it may be inconsistent
            # across sub-envs.
            if (terminated and self._max_episode_steps is None) or truncated:
                self._env_needs_reset[i] = True

            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs
                infos[key][i] = value

        # Convert info lists to arrays for scalar values (float/int)
        for key, value_list in list(infos.items()):
            if all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in value_list
                if v is not None
            ):
                array = np.array(value_list, dtype=np.float32)
                infos[key] = self._to_tensor(array)

        observations = np.array(self._observations)
        rewards = self._rewards.copy()
        terminations = self._terminations.copy()
        truncations = self._truncations.copy()

        return (
            self._to_tensor(observations),
            self._to_tensor(rewards),
            self._to_tensor(terminations),
            self._to_tensor(truncations),
            infos,
        )

    def render(self) -> np.ndarray | None:  # type: ignore
        """Render at most 16 environments and tile them in a 4x4 grid.

        Returns:
            Tiled image as numpy array with shape (height, width, 3) or None
        """
        results: list[np.ndarray] = []
        for env in self.envs:
            rendered_img: np.ndarray | list | None = env.render()
            assert isinstance(
                rendered_img, np.ndarray
            ), "Sub-environment render() must return an image as numpy array"
            results.append(rendered_img)

        if not results:
            return None

        # Tile images in a 4x4 grid (max 16 environments)
        max_envs = min(len(results), 16)
        results = results[:max_envs]

        # Calculate grid dimensions
        grid_cols = min(4, max_envs)
        grid_rows = (max_envs + grid_cols - 1) // grid_cols

        # Get dimensions from first image
        img_height, img_width = results[0].shape[:2]
        channels = results[0].shape[2] if len(results[0].shape) == 3 else 1

        # Create tiled image
        tiled_height = grid_rows * img_height
        tiled_width = grid_cols * img_width

        if channels == 1:
            tiled_image = np.zeros((tiled_height, tiled_width), dtype=results[0].dtype)
        else:
            tiled_image = np.zeros(
                (tiled_height, tiled_width, channels), dtype=results[0].dtype
            )

        # Fill tiled image
        for i, img in enumerate(results):
            row = i // grid_cols
            col = i % grid_cols

            start_row = row * img_height
            end_row = start_row + img_height
            start_col = col * img_width
            end_col = start_col + img_width

            if channels == 1:
                tiled_image[start_row:end_row, start_col:end_col] = img
            else:
                tiled_image[start_row:end_row, start_col:end_col] = img

        return tiled_image

    def close(self, **kwargs):
        """Close all environments."""
        del kwargs  # Unused parameter required by VectorEnv interface
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    # -------------------------- Misc -----------------------------

    @property
    def unwrapped(self):
        """Return the underlying sub-environments list."""
        return self.envs


class NormalizeActionMultiEnvWrapper(MultiEnvWrapper):
    """A `MultiEnvWrapper` that normalizes the action space to [-1, 1].

    It assumes the action space of the wrapped environment is a Box space.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.single_action_space, gym.spaces.Box
        ), "Only Box action space is supported"

        # Pre-compute normalized action space parameters
        self.action_low = self.single_action_space.low
        self.action_high = self.single_action_space.high
        self._action_mean = (self.action_high + self.action_low) / 2
        self._action_half_range = (self.action_high - self.action_low) / 2.0

        # New normalized action space
        norm_action_space = gym.spaces.Box(
            low=-np.ones_like(self.action_low, dtype=self.single_action_space.dtype),
            high=np.ones_like(self.action_high, dtype=self.single_action_space.dtype),
            shape=self.single_action_space.shape,
            dtype=self.single_action_space.dtype,
        )
        self.action_space = batch_space(norm_action_space, self.num_envs)
        self.single_action_space = norm_action_space

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Step all sub-environments with normalized batched actions in [-1, 1]."""
        actions_np = self._to_numpy(actions)
        assert self.action_space.contains(actions_np), "Actions not in action space"
        # Denormalize actions to original space
        denorm_actions = self._action_mean + actions_np * self._action_half_range
        return super().step(denorm_actions)


class MultiEnvRecordVideo(gym.Wrapper):
    """A `RecordVideo` wrapper for `MultiEnvWrapper` that records tiled rgb_array
    renders of all sub-environments.

    We need this because the standard `RecordVideo` expects a
    boolean terminal / truncated signal from the `step()` call, but
    `MultiEnvWrapper` returns a batch of such signals, one per
    sub-environment.

    NOTE: This wrapper currently only supports episode based recording.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Optional[Callable[[int], bool]] = None,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        gym.Wrapper.__init__(self, env)
        assert isinstance(
            env, MultiEnvWrapper
        ), "MultiEnvRecordVideo only works with MultiEnvWrapper"

        assert env.render_mode == "rgb_array", (
            "MultiEnvRecordVideo requires the wrapped env to have "
            'render_mode="rgb_array" for image rendering'
        )
        if episode_trigger is None:
            episode_trigger = capped_cubic_video_schedule
        self.episode_trigger = episode_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0

        self.recording = False
        self.terminated: bool = False
        self.truncated: bool = False
        self.recorded_frames = 0
        self.episode_id = 0

        self.is_vector_env = True

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video
        enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        self.episode_id += 1
        self.step_id = 0
        if self._video_enabled():
            # Force start recording on reset if enabled
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()  # type: ignore[no-untyped-call]
        self.recorded_frames = 1
        self.recording = True

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def _video_enabled(self) -> bool:  # type: ignore[no-untyped-call]
        return self.episode_trigger(self.episode_id)  # type: ignore[no-untyped-call]

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Steps through the environment using actions, recording observations if
        :attr:`self.recording`."""
        assert isinstance(self.env, MultiEnvWrapper)
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(actions)
        self.step_id += 1

        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.capture_frame()  # type: ignore[no-untyped-call]
            self.recorded_frames += 1
        elif self._video_enabled():
            self.start_video_recorder()  # type: ignore[no-untyped-call]

        return observations, rewards, terminateds, truncateds, infos

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during
        initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()


class PlanningStatesVectorEnv(VectorEnv):
    """A wrapper for intergated planning and RL environments. Note that this wrapper
    assumes the base environment is a vectorized (batched) environment. This can be
    either from Maniskill or wrapped from prpl_utils.gym_utils.MultiEnvWrapper.

    It does three things:
    1. It contructs a local mdp for RL learning (finite fixed horizon, no termination).
       The local mdp could have different observation and action spaces from the
         base environment.
       The local mdp is initialed from the initial states of the planning (failure) data.
    2. The observation space of the local mdp is clipped based on the observation space.
    3. It evaluates the "subgoal"/"reward" conditions based on the latest tamp system,
       e.g., after the rl policy roll out, it checks whether the task can be achieved by
       the planner.
    """

    def __init__(
        self,
        env: Union[ManiSkillBaseEnv, "MultiEnvWrapper"],
        tamp_system: BaseRLTAMPSystem,
        scenario_info: dict[str, Any],
        planner: TaskThenMotionPlanner,
        num_envs: int = 1,
        auto_reset: bool = True,
        ignore_terminations: bool = False,
        record_metrics: bool = False,
    ):
        self._env = env
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics
        self.spec = self._env.spec
        self.tamp_system = tamp_system
        self.initial_states: Optional[torch.Tensor] = None
        # local mdp horizon and planner horizon
        self.skill_max_steps = scenario_info.get("max_rl_steps", 20)
        self.rl_static_steps = scenario_info.get("rl_static_steps", 3)
        self.planner_max_steps = (
            scenario_info.get("max_env_steps", CFG.max_env_steps)
            - self.rl_static_steps
            - self.skill_max_steps
        )
        self.failed_skill = scenario_info.get("failed_skill", None)
        # Right now the planners are not parallizable, so we just copy the planner
        # for each environment.
        self.planner = planner
        (
            self.normalize_action,
            self.arm_action_low,
            self.arm_action_high,
        ) = get_normalize_action_range(env, CFG.control_mode)
        # Construct local observation and action spaces
        train_objects = scenario_info.get("train_objects", "").split(",")
        self.train_objects = []
        for name, obj in self.tamp_system.perceiver.objects.as_dict().items():
            if name in train_objects:
                self.train_objects.append(obj)
        assert len(self.train_objects) == len(
            train_objects
        ), "Some training objects are not found in the perceiver objects."
        # Assume all observations are continuous and unbounded for now
        local_observation_space = self.tamp_system.skill_obs_space()

        super().__init__(
            num_envs,
            local_observation_space,
            self._env.get_wrapper_attr("single_action_space"),
        )
        if not self.ignore_terminations and auto_reset:
            if isinstance(self._env, ManiSkillBaseEnv):
                assert (
                    self._env.reconfiguration_freq == 0 or self._env.num_envs == 1
                ), "With partial resets, environment cannot be reconfigured automatically"

        if self.record_metrics:
            self.success_once = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self.fail_once = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self.returns = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            )

        # NOTE: local obs is float32 for neural network training,
        # actual env might be float64 for physics simulation
        self.clip_low = torch.as_tensor(
            self.observation_space.low, device=self.device, dtype=torch.float32  # type: ignore
        )
        self.clip_high = torch.as_tensor(
            self.observation_space.high, device=self.device, dtype=torch.float32  # type: ignore
        )

        self.reset_id = 12

        # Load intrinsic reward function if planner_eval is disabled
        self.intrinsic_reward_fn = None
        if not CFG.planner_eval:
            assert (
                CFG.intrinsic_reward_path
            ), "intrinsic_reward_path must be set when planner_eval is False"
            import importlib.util
            from pathlib import Path

            reward_path = Path(CFG.intrinsic_reward_path)
            assert (
                reward_path.exists() and reward_path.suffix == ".py"
            ), f"intrinsic_reward_path must be a valid Python file: {reward_path}"

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(
                "intrinsic_reward_module", reward_path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the intrinsic_rwd function
            assert hasattr(
                module, "intrinsic_rwd"
            ), f"intrinsic_reward_path file must define 'intrinsic_rwd' function"
            self.intrinsic_reward_fn = module.intrinsic_rwd
            logger.info(f"Loaded intrinsic reward function from {reward_path}")

    @property
    def device(self):
        """Return the device on which the environment is running."""
        return self._env.device

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self._env.unwrapped

    def configure_training(
        self,
        training_data: RLDataset,
    ) -> None:
        """Configure environment for training phase."""
        self.initial_states = torch.stack(
            training_data.states,
            dim=0,
        ).to(self.device, dtype=torch.float64)

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset the environment."""
        if options is None:
            options = {}
        if self.initial_states is not None:
            if "env_idx" in options:
                num_states = len(options["env_idx"])
            else:
                num_states = self.num_envs
            reset_state_idx = torch.randint(
                0, len(self.initial_states), (num_states,), device=self.device
            )
            logger.info(f"Reset states idx: {reset_state_idx}")
            init_state = self.initial_states[reset_state_idx]
            options["init_state"] = init_state
            self.reset_id = self.reset_id + num_states
        origin_obs, info = self._env.reset(seed=seed, options=options)  # type: ignore
        if "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)  # type: ignore
            mask[env_idx] = True  # type: ignore
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0
        return (
            self.clip_observation(torch.as_tensor(origin_obs, device=self.device)),
            info,
        )

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        """Step the environment with the given actions."""
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float64
        )
        original_obs, rew, terminations, truncations, infos = self._env.step(actions)  # type: ignore
        original_obs = torch.as_tensor(original_obs, device=self.device)
        rw_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        truncations = self._env.elapsed_steps >= self.skill_max_steps
        if truncations.any():
            assert truncations.all(), "Truncations should be homogeneous"
            # Once truncated, start planning based evaluation for rewards
            # RL with Planning Feedback and Planning with RL Refactorization
            planner_obs = torch_clone_dict(original_obs)
            frozon_actions = get_frozen_action(
                actions_tensor,
                self.arm_action_low,
                self.arm_action_high,
                self.normalize_action,
                CFG.control_mode,
                planner_obs,
            )
            # Step the env for some static steps before calling the planner
            # Since RL policy is not very stable
            for _ in range(self.rl_static_steps):
                planner_obs, _, _, _, _ = self._env.step(frozon_actions)
                frozon_actions = get_frozen_action(
                    actions_tensor,
                    self.arm_action_low,
                    self.arm_action_high,
                    self.normalize_action,
                    CFG.control_mode,
                    planner_obs if isinstance(planner_obs, torch.Tensor) else None,
                )
            if CFG.planner_eval:
                # Use planner based evaluation
                self.planner.reset(planner_obs, infos)
                for _ in range(self.planner_max_steps):
                    actions_planner, _ = self.planner.step(planner_obs)
                    if actions_planner is None:
                        # No symbolic plan, -1 reward
                        rew_planner_tensor = torch.zeros(
                            self.num_envs, device=self.device, dtype=torch.float32
                        )
                        rew_planner_tensor -= 1.0
                        break
                    planner_obs, rew_planner, _, _, infos = self._env.step(
                        actions_planner
                    )
                    if self.planner.exhausted.all():
                        # We need one more step to truncate the envs for video recording.
                        self._env.elapsed_steps[:] = (
                            self.skill_max_steps
                            + self.rl_static_steps
                            + self.planner_max_steps
                            - 1
                        )
                        _, _, _, _, _ = self._env.step(actions_planner)
                        break
                    if (
                        CFG.partial_planner_eval
                        and self.planner.last_operator_reached_effects.any()
                        and self.planner.last_operator is not None
                        and (
                            self.planner.last_operator.parent.name == self.failed_skill
                        )
                    ):
                        # Partial planner evaluation: only evaluate the current failed skill
                        # reached effects
                        rew_planner_partial = torch.zeros(
                            self.num_envs, device=self.device, dtype=torch.float32
                        )
                        rew_planner_partial[
                            self.planner.last_operator_reached_effects
                        ] += 1.0
                        rew_planner_tensor = rew_planner_partial
                        # We need one more step to truncate the envs for video recording.
                        self._env.elapsed_steps[:] = (
                            self.skill_max_steps
                            + self.rl_static_steps
                            + self.planner_max_steps
                            - 1
                        )
                        _, _, _, _, _ = self._env.step(actions_planner)
                        break
                    rew_planner_tensor = torch.as_tensor(
                        rew_planner, device=self.device, dtype=torch.float32
                    )
                # Return is at most 1.0 for an episode
                rw_tensor += torch.clamp(rew_planner_tensor, max=1.0)
            else:
                # Use intrinsic reward function instead of planner evaluation
                # For fast iteration of rl tuning purposes
                assert (
                    self.intrinsic_reward_fn is not None
                ), "intrinsic_reward_fn must be loaded when planner_eval is False"
                # Compute intrinsic reward from current observation
                intrinsic_reward = self.intrinsic_reward_fn(planner_obs)
                intrinsic_reward_tensor = torch.as_tensor(
                    intrinsic_reward, device=self.device, dtype=torch.float32
                )
                rw_tensor += intrinsic_reward_tensor
                # We need one more step to truncate the envs for video recording.
                self._env.elapsed_steps[:] = (
                    self.skill_max_steps
                    + self.rl_static_steps
                    + self.planner_max_steps
                    - 1
                )
                _, _, _, _, _ = self._env.step(frozon_actions)

        if self.record_metrics:
            episode_info = dict()
            self.returns += rw_tensor
            if "success" in infos:
                self.success_once = infos["success"].to(torch.bool)
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = torch.as_tensor(
                self._env.elapsed_steps, device=self.device
            ).clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )

        # Assume terminations are tensors
        terminations_tensor = torch.as_tensor(terminations, device=self.device).clone()

        if self.ignore_terminations:
            terminations_tensor[:] = False
            if self.record_metrics:
                if "success" in infos:
                    episode_info["success_at_end"] = (
                        infos["success"].to(torch.bool).clone()
                    )
                if "fail" in infos:
                    episode_info["fail_at_end"] = infos["fail"].clone()
        if self.record_metrics:
            infos["episode"] = episode_info

        trunc_tensor = torch.as_tensor(
            truncations, device=self.device, dtype=torch.bool
        )
        dones = torch.logical_or(terminations_tensor, trunc_tensor)

        if dones.any() and self.auto_reset:
            assert dones.all(), "Dones should be homogeneous"
            final_obs = torch_clone_dict(original_obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = torch_clone_dict(infos)
            # Note that self.reset return sub_obs, not in the complete observation space
            sub_obs, infos = self.reset(options=dict(env_idx=env_idx))
            # NOTE: PPO only has access to the last obs before planner
            # we convert it to a sub-obs here and save as final_observation
            infos["final_observation"] = self.clip_observation(final_obs)
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE: return sub_obs for next episode start
            return (
                sub_obs,
                rw_tensor,
                terminations_tensor,
                truncations,
                infos,
            )
        return (
            self.clip_observation(original_obs),
            rw_tensor,
            terminations_tensor,
            truncations,
            infos,
        )

    def call(self, name: str, *args, **kwargs):
        """Call a method on the environment."""
        function = getattr(self._env, name)
        return function(*args, **kwargs)

    def render(self):
        """Render the environment."""
        return self._env.render()

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Wait for the reset to complete."""
        del seed, options  # Unused parameters

    def step_wait(self, **kwargs) -> Tuple[Any, Array, Array, Array, Dict]:
        """Wait for the step to complete."""
        del kwargs  # Unused parameter
        raise NotImplementedError(
            "step_wait is not implemented for PlanningStatesVectorEnv"
        )

    def clip_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Clip the observation to the local observation space."""
        sub_obs = self.tamp_system.state_to_vec(obs, self.train_objects)
        # clip the observation to the local observation space
        sub_obs = torch.clip(sub_obs, self.clip_low, self.clip_high)
        return sub_obs


class PlanningStatesVectorRoomEnv(PlanningStatesVectorEnv):
    """A wrapper for intergated planning and RL environments. Note that this wrapper
    assumes the base environment is a vectorized (batched) environment. This can be
    either from Maniskill or wrapped from prpl_utils.gym_utils.MultiEnvWrapper.

    It does three things:
    1. It contructs a local mdp for RL learning (finite fixed horizon, no termination).
       The local mdp could have different observation and action spaces from the
         base environment.
       The local mdp is initialed from the initial states of the planning (failure) data.
    2. The observation space of the local mdp is clipped based on the observation space.
    3. It evaluates the "subgoal"/"reward" conditions based on the latest tamp system,
       e.g., after the rl policy roll out, it checks whether the task can be achieved by
       the planner.
    """

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        """Step the environment with the given actions."""
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float64
        )
        original_obs, rew, terminations, truncations, infos = self._env.step(actions)  # type: ignore
        original_obs = torch.as_tensor(original_obs, device=self.device)
        rw_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        truncations = self._env.elapsed_steps >= self.skill_max_steps
        if truncations.any():
            assert truncations.all(), "Truncations should be homogeneous"
            # Once truncated, start planning based evaluation for rewards
            # RL with Planning Feedback and Planning with RL Refactorization
            planner_obs = torch_clone_dict(original_obs)
            frozon_actions = get_frozen_action(
                actions_tensor,
                self.arm_action_low,
                self.arm_action_high,
                self.normalize_action,
                CFG.control_mode,
                planner_obs,
            )
            # Step the env for some static steps before calling the planner
            # Since RL policy is not very stable
            for _ in range(self.rl_static_steps):
                planner_obs, _, _, _, _ = self._env.step(frozon_actions)
                frozon_actions = get_frozen_action(
                    actions_tensor,
                    self.arm_action_low,
                    self.arm_action_high,
                    self.normalize_action,
                    CFG.control_mode,
                    planner_obs if isinstance(planner_obs, torch.Tensor) else None,
                )
            if CFG.planner_eval:
                # Use planner based evaluation
                self.planner.reset(planner_obs, infos)
                for _ in range(self.planner_max_steps):
                    actions_planner, _ = self.planner.step(planner_obs)
                    if actions_planner is None:
                        # No symbolic plan, -1 reward
                        rew_planner_tensor = torch.zeros(
                            self.num_envs, device=self.device, dtype=torch.float32
                        )
                        rew_planner_tensor -= 1.0
                        break
                    planner_obs, rew_planner, _, _, infos = self._env.step(
                        actions_planner
                    )
                    if self.planner.exhausted.all():
                        # We need one more step to truncate the envs for video recording.
                        self._env.elapsed_steps[:] = (
                            self.skill_max_steps
                            + self.rl_static_steps
                            + self.planner_max_steps
                            - 1
                        )
                        _, _, _, _, _ = self._env.step(actions_planner)
                        break
                    if (
                        CFG.partial_planner_eval
                        and self.planner.last_operator_reached_effects.any()
                        and self.planner.last_operator is not None
                        and (
                            self.planner.last_operator.parent.name == self.failed_skill
                        )
                    ):
                        # Partial planner evaluation: only evaluate the current failed skill
                        # reached effects
                        rew_planner_partial = torch.zeros(
                            self.num_envs, device=self.device, dtype=torch.float32
                        )
                        rew_planner_partial[
                            self.planner.last_operator_reached_effects
                        ] += 1.0
                        rew_planner_tensor = rew_planner_partial
                        # We need one more step to truncate the envs for video recording.
                        self._env.elapsed_steps[:] = (
                            self.skill_max_steps
                            + self.rl_static_steps
                            + self.planner_max_steps
                            - 1
                        )
                        _, _, _, _, _ = self._env.step(actions_planner)
                        break
                    rew_planner_tensor = torch.as_tensor(
                        rew_planner, device=self.device, dtype=torch.float32
                    )
                # Return is at most 1.0 for an episode
                rw_tensor += torch.clamp(rew_planner_tensor, max=1.0)
            else:
                # Use intrinsic reward function instead of planner evaluation
                # For fast iteration of rl tuning purposes
                assert (
                    self.intrinsic_reward_fn is not None
                ), "intrinsic_reward_fn must be loaded when planner_eval is False"
                # Compute intrinsic reward from current observation
                intrinsic_reward = self.intrinsic_reward_fn(planner_obs)
                intrinsic_reward_tensor = torch.as_tensor(
                    intrinsic_reward, device=self.device, dtype=torch.float32
                )
                rw_tensor += intrinsic_reward_tensor
                # We need one more step to truncate the envs for video recording.
                self._env.elapsed_steps[:] = (
                    self.skill_max_steps
                    + self.rl_static_steps
                    + self.planner_max_steps
                    - 1
                )
                _, _, _, _, _ = self._env.step(frozon_actions)

        if self.record_metrics:
            episode_info = dict()
            self.returns += rw_tensor
            if "success" in infos:
                self.success_once = infos["success"].to(torch.bool)
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = torch.as_tensor(
                self._env.elapsed_steps, device=self.device
            ).clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )

        # Assume terminations are tensors
        terminations_tensor = torch.as_tensor(terminations, device=self.device).clone()

        if self.ignore_terminations:
            terminations_tensor[:] = False
            if self.record_metrics:
                if "success" in infos:
                    episode_info["success_at_end"] = (
                        infos["success"].to(torch.bool).clone()
                    )
                if "fail" in infos:
                    episode_info["fail_at_end"] = infos["fail"].clone()
        if self.record_metrics:
            infos["episode"] = episode_info

        trunc_tensor = torch.as_tensor(
            truncations, device=self.device, dtype=torch.bool
        )
        dones = torch.logical_or(terminations_tensor, trunc_tensor)

        if dones.any() and self.auto_reset:
            assert dones.all(), "Dones should be homogeneous"
            final_obs = torch_clone_dict(original_obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = torch_clone_dict(infos)
            # Note that self.reset return sub_obs, not in the complete observation space
            sub_obs, infos = self.reset(options=dict(env_idx=env_idx))
            # NOTE: PPO only has access to the last obs before planner
            # we convert it to a sub-obs here and save as final_observation
            infos["final_observation"] = self.clip_observation(final_obs)
            infos["final_info"] = final_info
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE: return sub_obs for next episode start
            return (
                sub_obs,
                rw_tensor,
                terminations_tensor,
                truncations,
                infos,
            )
        return (
            self.clip_observation(original_obs),
            rw_tensor,
            terminations_tensor,
            truncations,
            infos,
        )


class ManiSkillsRecordVideo(RecordEpisode):
    """Record trajectories or videos for episodes. You generally should always apply
    this wrapper last, particularly if you include observation wrappers which modify the
    returned observations. The only wrappers that may go after this one is any of the
    vector env interface wrappers that map the maniskill env to a e.g. gym vector env
    interface.

    Trajectory data is saved with two files, the actual data in a .h5 file via H5py and metadata in a JSON file of the same basename.

    Each JSON file contains:

    - `env_info` (Dict): task (also known as environment) information, which can be used to initialize the task
    - `env_id` (str): task id
    - `max_episode_steps` (int)
    - `env_kwargs` (Dict): keyword arguments to initialize the task. **Essential to recreate the environment.**
    - `episodes` (List[Dict]): episode information
    - `source_type` (Optional[str]): a simple category string describing what process generated the trajectory data. ManiSkill official datasets will usually write one of "human", "motionplanning", or "rl" at the moment.
    - `source_desc` (Optional[str]): a longer explanation of how the data was generated.

    The episode information (the element of `episodes`) includes:

    - `episode_id` (int): a unique id to index the episode
    - `reset_kwargs` (Dict): keyword arguments to reset the task. **Essential to reproduce the trajectory.**
    - `control_mode` (str): control mode used for the episode.
    - `elapsed_steps` (int): trajectory length
    - `info` (Dict): information at the end of the episode.

    With just the meta data, you can reproduce the task the same way it was created when the trajectories were collected as so:

    ```python
    env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
    episode = env_info["episodes"][0] # picks the first
    env.reset(**episode["reset_kwargs"])
    ```

    Each HDF5 demonstration dataset consists of multiple trajectories. The key of each trajectory is `traj_{episode_id}`, e.g., `traj_0`.

    Each trajectory is an `h5py.Group`, which contains:

    - actions: [T, A], `np.float32`. `T` is the number of transitions.
    - terminated: [T], `np.bool_`. It indicates whether the task is terminated or not at each time step.
    - truncated: [T], `np.bool_`. It indicates whether the task is truncated or not at each time step.
    - env_states: [T+1, D], `np.float32`. Environment states. It can be used to set the environment to a certain state via `env.set_state_dict`. However, it may not be enough to reproduce the trajectory.
    - success (optional): [T], `np.bool_`. It indicates whether the task is successful at each time step. Included if task defines success.
    - fail (optional): [T], `np.bool_`. It indicates whether the task is in a failure state at each time step. Included if task defines failure.
    - obs (optional): [T+1, D] observations.

    Note that env_states is in a dictionary form (and observations may be as well depending on obs_mode), where it is formatted as a dictionary of lists. For example, a typical environment state looks like this:

    ```python
    env_state = env.get_state_dict()
    \"\"\"
    env_state = {
    "actors": {
        "actor_id": [...numpy_actor_state...],
        ...
    },
    "articulations": {
        "articulation_id": [...numpy_articulation_state...],
        ...
    }
    }
    \"\"\"
    ```
    In the trajectory file env_states will be the same structure but each value/leaf in the dictionary will be a sequence of states representing the state of that particular entity in the simulation over time.

    In practice it is may be more useful to use slices of the env_states data (or the observations data), which can be done with

    ```python
    import mani_skill.trajectory.utils as trajectory_utils
    env_states = trajectory_utils.dict_to_list_of_dicts(env_states)
    # now env_states[i] is the same as the data env.get_state_dict() returned at timestep i
    i = 10
    env_state_i = trajectory_utils.index_dict(env_states, i)
    # now env_state_i is the same as the data env.get_state_dict() returned at timestep i
    ```

    Args:
        env: the environment to record
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        info_on_video: whether to write data about reward, action, and data in the info object to the video. The first video frame is generally the result
            of the first env.reset() (visualizing the first observation). Text is written on frames after that, showing the action taken to get to that
            environment state and reward.
        save_on_reset: whether to save the previous trajectory (and video of it if `save_video` is True) automatically when resetting.
            Not that for environments simulated on the GPU (to leverage fast parallel rendering) you must
            set `max_steps_per_video` to a fixed number so that every `max_steps_per_video` steps a video is saved. This is
            required as there may be partial environment resets which makes it ambiguous about how to save/cut videos.
        save_video_trigger: a function that takes the current number of elapsed environment steps and outputs a bool. If output is True, will start saving that timestep to the video.
        max_steps_per_video: how many steps can be recorded into a single video before flushing the video. If None this is not used. A internal step counter is maintained to do this.
            If the video is flushed at any point, the step counter is reset to 0.
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
        record_reward: whether to record the reward in the trajectory data
        record_env_state: whether to record the environment state in the trajectory data
        video_fps (int): The FPS of the video to generate if save_video is True
        render_substeps (bool): Whether to render substeps for video. This is captures an image of the environment after each physics step. This runs slower but generates more image frames
            per environment step which when coupled with a higher video FPS can yield a smoother video.
        avoid_overwriting_video (bool): If true, the wrapper will iterate over possible video names to avoid overwriting existing videos in the output directory. Useful for resuming training runs.
        source_type (Optional[str]): a word to describe the source of the actions used to record episodes (e.g. RL, motionplanning, teleoperation)
        source_desc (Optional[str]): A longer description describing how the demonstrations are collected
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
        **kwargs,
    ):
        if self.save_on_reset:
            if self.save_video:
                self.flush_video()
        return super().reset(*args, seed=seed, options=options, **kwargs)
