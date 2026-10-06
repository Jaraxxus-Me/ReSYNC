"""
Soft Actor-Critic (SAC) implementation which is largely based on
the implementation from ManiSkill3:
https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/sac/sac.py
"""

from __future__ import annotations

# === Standard library ===
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# === Third-party ===
import dacite
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from gymnasium import spaces

# === Local ===
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.utils import polyak_update
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from skill_refactor.approaches.rl_policies.base import (
    Logger,
    Policy,
)
from skill_refactor.settings import CFG

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Default arguments for SAC
@dataclass
class SACArgs:
    """Arguments for the Soft Actor-Critic (SAC) algorithm."""

    exp_name: str = CFG.exp_name
    """The name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""
    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    track: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project_name: str = "ManiSkill"
    """The wandb's project name."""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project."""
    wandb_group: str = "SAC"
    """The group of the run for wandb."""
    capture_video: bool = True
    """Whether to capture videos of the agent performances (check out `videos`
    folder)"""
    save_trajectory: bool = False
    """Whether to save trajectory data into the `videos` folder."""
    save_model: bool = True
    """Whether to save model into the `runs/{run_name}` folder."""
    evaluate: bool = False
    """If toggled, only runs evaluation with the given model checkpoint and saves the
    evaluation trajectories."""
    checkpoint: Optional[str] = None
    """Path to a pretrained checkpoint file to start evaluation/training from."""
    log_freq: int = 1
    """Logging frequency in terms of environment steps."""

    # Environment specific arguments
    num_envs: int = 16
    """The number of parallel environments."""
    num_eval_envs: int = 1
    """The number of parallel evaluation environments."""
    partial_reset: bool = False
    """Whether to let parallel environments reset upon termination instead of
    truncation."""
    eval_partial_reset: bool = False
    """Whether to let parallel evaluation environments reset upon termination instead of
    truncation."""
    num_steps: int = 50
    """The number of steps to run in each environment per policy rollout."""
    num_eval_steps: int = 50
    """The number of steps to run in each evaluation environment during evaluation."""
    reconfiguration_freq: Optional[int] = None
    """How often to reconfigure the environment during training."""
    eval_reconfiguration_freq: Optional[int] = 1
    """For benchmarking purposes we want to reconfigure the eval environment each reset
    to ensure objects are randomized in some tasks."""
    eval_freq: int = 25
    """Evaluation frequency in terms of iterations."""
    save_train_video_freq: Optional[int] = None
    """Frequency to save training videos in terms of iterations."""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """The control mode to use for the environment."""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """Total timesteps of the experiments."""
    buffer_size: int = 1_000_000
    """The replay memory buffer size."""
    buffer_device: str = "cuda"
    """Where the replay buffer is stored.

    Can be 'cpu' or 'cuda' for GPU
    """
    gamma: float = 0.8
    """The discount factor gamma."""
    tau: float = 0.01
    """Target smoothing coefficient."""
    batch_size: int = 1024
    """The batch size of sample from the replay memory."""
    learning_starts: int = 128
    """Timestep to start learning."""
    policy_lr: float = 3e-4
    """The learning rate of the policy network optimizer."""
    q_lr: float = 3e-4
    """The learning rate of the Q network network optimizer."""
    policy_frequency: int = 1
    """The frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """The frequency of updates for the target nerworks."""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """Automatic tuning of the entropy coefficient."""
    training_freq: int = 64
    """Training frequency (in steps)"""
    utd: float = 0.5
    """Update to data ratio."""
    bootstrap_at_done: str = "always"
    """The bootstrap method to use when a done signal is received.

    Can be 'always' or 'never'
    """

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """The number of gradient updates per iteration."""
    steps_per_env: int = 0
    """The number of steps each parallel env takes per iteration."""


@dataclass
class ReplayBufferSample:
    """A sample from the replay buffer."""

    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """A replay buffer for storing transitions."""

    def __init__(
        self,
        env: ManiSkillVectorEnv,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        """Initializes the replay buffer.

        Args:
            env: The environment from which to derive observation and action spaces.
            num_envs: Number of parallel environments.
            buffer_size: Total size of the replay buffer.
            storage_device: Device where the buffer will be stored (e.g., 'cpu' or 'cuda').
            sample_device: Device from which samples will be drawn (e.g., 'cpu' or 'cuda').
        """

        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + env.single_observation_space.shape
        ).to(storage_device)
        self.next_obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + env.single_observation_space.shape
        ).to(storage_device)
        self.actions = torch.zeros(
            (self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape
        ).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Adds a new transition to the replay buffer.

        Args:
            obs: Current observations from the environment.
            next_obs: Next observations after taking the action.
            action: Actions taken in the environment.
            reward: Rewards received after taking the action.
            done: Done flags indicating if the episode has ended.
        """
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size: Number of transitions to sample.
        Returns:
            A `ReplayBufferSample` containing sampled observations, next observations,
            actions, rewards, and done flags.
        """
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """Soft Q-Network for the Soft Actor-Critic algorithm."""

    def __init__(self, env: ManiSkillVectorEnv):
        """Initialize the Soft Q-Network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod()
                + np.prod(env.single_action_space.shape),
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.cat([x, a], 1)
        return self.net(x)


class Actor(nn.Module):
    """Actor network for the Soft Actor-Critic algorithm."""

    def __init__(self, env: ManiSkillVectorEnv):
        """Initialize the Actor network."""
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_space = env.single_action_space
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.action_dist = SquashedDiagGaussianDistribution(
            env.single_action_space.shape
        )
        self.action_low = torch.tensor(self.action_space.low)
        self.action_high = torch.tensor(self.action_space.high)

    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Get the parameters for the action distribution.

        :param obs:
        :return: Mean, standard deviation and optional keyword arguments.
        """
        latent_pi = self.backbone(obs)
        mean_actions = self.fc_mean(latent_pi)

        # Unstructured exploration (Original implementation)
        log_std = self.fc_logstd(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Forward pass through the Actor network."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the log probability of the action given the observation."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, x: torch.Tensor, deterministic=False) -> torch.Tensor:
        """Get deterministic action for evaluation."""
        return self(x, deterministic=deterministic)

    def predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get the policy action from an observation (and optional hidden state).

        Includes sugar-coating to handle different observations (e.g. normalizing
        images).
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.eval()
        with torch.no_grad():
            # unscaled actions
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.reshape((-1, *self.action_space.shape))
        actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
        return actions

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale the action from [low, high] to [-1, 1] (no need for symmetric action
        space)

        :param action: Action to scale
        :return: Scaled action
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to scale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.action_low, self.action_high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """Rescale the action from [-1, 1] to [low, high] (no need for symmetric action
        space)

        :param scaled_action: Action to un-scale
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.action_low, self.action_high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def to(self, *args: Any, **kwargs: Any) -> Actor:
        """Move the Actor network to the specified device or dtype."""
        self.action_low = self.action_low.to(*args, **kwargs)
        self.action_high = self.action_high.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self


def load_sac_args(config_path: str) -> SACArgs:
    """Load SAC arguments from a YAML configuration file."""
    with open(config_path, "rb") as f:
        config_data = yaml.safe_load(f)

    return dacite.from_dict(SACArgs, config_data)


class SACPolicy(Policy):
    """Soft Actor-Critic (SAC) policy implementation."""

    def __init__(
        self,
        seed: int,
        rl_config: str = "",
    ) -> None:
        super().__init__(seed, rl_config)
        self.device = CFG.device
        self.args: SACArgs = load_sac_args(rl_config)
        log_path = Path(CFG.tb_log_dir) / f"{CFG.exp_name}"
        writer = SummaryWriter(log_path)  # type: ignore
        writer.add_text(  # type: ignore
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )
        self.logger = Logger(log_wandb=CFG.log_wandb, tensorboard=writer)
        self._trained = False

    @property
    def requires_training(self) -> bool:
        """Check if the policy requires training."""
        return True

    def initialize(self, env: ManiSkillVectorEnv) -> None:
        """Initialize the SAC policy with the given environment."""
        self.actor = Actor(env).to(self.device)
        self.qf1 = SoftQNetwork(env).to(self.device)
        self.qf2 = SoftQNetwork(env).to(self.device)
        self.qf1_target = SoftQNetwork(env).to(self.device)
        self.qf2_target = SoftQNetwork(env).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.qf1_target.eval()
        self.qf2_target.eval()

        self.q_optimizer = Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.policy_lr)

        if self.args.autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(env.single_action_space.shape)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha

        self.replay_buffer = ReplayBuffer(
            env=env,
            num_envs=self.args.num_envs,
            buffer_size=self.args.buffer_size,
            storage_device=torch.device(self.args.buffer_device),
            sample_device=self.device,
        )

        # update the args with the environment-specific values
        num_envs = env.num_envs
        if num_envs != self.args.num_envs:
            logging.warning(
                f"Number of environments in the provided environment ({num_envs}) "
                f"does not match the configured number of environments ({self.args.num_envs}). "
                f"Using {num_envs} instead."
            )
            self.args.num_envs = num_envs
        self.args.steps_per_env = self.args.training_freq // self.args.num_envs

    def can_initiate(self) -> bool:
        return True

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        del obs  # No internal state to reset

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor.predict(obs, deterministic=False)
        return action

    def _sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action according to the exploration policy. This is either done by
        sampling the probability distribution of the policy, or sampling a random action
        (from a uniform distribution over the action space) or by adding noise to the
        deterministic output.

        :param action_noise: Action noise that will be used for exploration Required for
            deterministic policy (e.g. TD3). This can also be used in addition to the
            stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment and scaled action that will be stored
            in the replay buffer. The two differs when the action space is not
            normalized (bounds are not [-1, 1]).
        """
        # Note: when using continuous actions,
        # we assume that the policy uses tanh to scale the action
        # We use non-deterministic action in the case of SAC, for TD3, it does not matter
        unscaled_action = self.actor.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        assert isinstance(self.actor.action_space, spaces.Box)
        scaled_action = self.actor.scale_action(unscaled_action)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.actor.unscale_action(scaled_action)
        return action, buffer_action

    def train(
        self,
        env: ManiSkillVectorEnv,
        eval_env: Optional[ManiSkillVectorEnv] = None,
        train_data: Optional[object] = None,
    ) -> None:
        del train_data  # Unused
        obs, _ = env.reset(seed=self._seed)
        global_step = 0
        global_update = 0
        learning_has_started = False

        global_steps_per_iteration = self.args.num_envs * (self.args.steps_per_env)
        pbar = tqdm.tqdm(range(self.args.total_timesteps))
        cumulative_times: defaultdict[str, float] = defaultdict(float)

        while global_step < self.args.total_timesteps:
            # Perform evaluation if needed
            if (
                self.args.eval_freq > 0
                and (
                    (global_step // global_steps_per_iteration) % self.args.eval_freq
                    == 0
                )
                and (eval_env is not None)
            ):
                # copy the normalizer of env
                if hasattr(env._env, "_obs_rms"):  # pylint: disable=protected-access
                    eval_env._env.obs_rms = (  # pylint: disable=protected-access
                        env._env.get_wrapper_attr(  # pylint: disable=protected-access
                            "obs_rms"
                        )
                    )
                # evaluate
                self.actor.eval()
                stime = time.perf_counter()
                eval_obs, _ = eval_env.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(self.args.num_eval_steps):
                    with torch.no_grad():
                        eval_obs, _, _, _, eval_infos = eval_env.step(
                            self.get_action(eval_obs)
                        )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
                eval_metrics_mean = {}
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    eval_metrics_mean[k] = mean
                    self.logger.add_scalar(f"eval/{k}", mean, global_step)
                success_once = eval_metrics_mean.get("success_once", torch.tensor(0.0))
                return_ = eval_metrics_mean.get("return", torch.tensor(0.0))
                pbar.set_description(
                    f"success_once: {success_once:.2f}, " f"return: {return_:.2f}"
                )
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                self.logger.add_scalar("time/eval_time", eval_time, global_step)
                if self.args.evaluate:
                    break
                self.actor.train()

                if self.args.save_model:
                    model_path = (
                        Path(CFG.rl_policy_save_dir)
                        / f"runs/{self.args.exp_name}/ckpt_{global_step}.pt"
                    )
                    base_path = (
                        Path(CFG.rl_policy_save_dir) / "runs" / self.args.exp_name
                    )
                    base_path.mkdir(parents=True, exist_ok=True)
                    self.save(model_path)
                    logging.info(f"model saved to {model_path}")

            # Collect rollouts from environemnts
            rollout_time = time.perf_counter()
            self.actor.train(False)

            for _ in range(self.args.steps_per_env):
                global_step += 1 * self.args.num_envs
                if not learning_has_started:
                    actions = torch.tensor(
                        env.action_space.sample(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # Following SB3, buffer actions are the scaled actions to [-1, 1]
                    # Direct actions (from actor._predict) are unscaled, also used for step the env.
                    buffer_actions = self.actor.scale_action(actions)
                else:
                    actions, buffer_actions = self._sample_action(obs)

                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                real_next_obs = next_obs.clone()
                if self.args.bootstrap_at_done == "never":
                    need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                    # always stop bootstrap when episode ends
                    stop_bootstrap = truncations | terminations
                else:
                    if self.args.bootstrap_at_done == "always":
                        # always need final obs when episode ends
                        need_final_obs = truncations | terminations
                        # never stop bootstrap
                        stop_bootstrap = torch.zeros_like(
                            terminations, dtype=torch.bool
                        )
                    else:  # bootstrap at truncated
                        # only need final obs when truncated and not terminated
                        need_final_obs = truncations & (~terminations)
                        # only stop bootstrap when terminated, don't stop when truncated
                        stop_bootstrap = terminations
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    real_next_obs[need_final_obs] = infos["final_observation"][
                        need_final_obs
                    ]
                    for k, v in final_info["episode"].items():
                        if k == "success_at_end":
                            continue  # skip success_once, it's a boolean
                        self.logger.add_scalar(
                            f"train/{k}", v[done_mask].float().mean(), global_step
                        )

                self.replay_buffer.add(
                    obs, real_next_obs, buffer_actions, rewards, stop_bootstrap
                )
                obs = next_obs

            rollout_time = time.perf_counter() - rollout_time
            cumulative_times["rollout_time"] += rollout_time
            pbar.update(self.args.num_envs * self.args.steps_per_env)

            # ALGO LOGIC: training.
            if global_step < self.args.learning_starts:
                continue

            update_time = time.perf_counter()
            learning_has_started = True
            self.actor.train(True)
            self.qf1.train(True)
            self.qf2.train(True)
            for _ in range(self.args.grad_steps_per_iteration):
                global_update += 1
                data = self.replay_buffer.sample(self.args.batch_size)

                actions_pi, log_prob = self.actor.action_log_prob(data.obs)
                log_prob = log_prob.reshape(-1, 1)

                # Critic Loss
                with torch.no_grad():
                    next_state_actions, next_state_log_pi = self.actor.action_log_prob(
                        data.next_obs
                    )
                    qf1_next_target = self.qf1_target(data.next_obs, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_obs, next_state_actions)
                    min_qf_next_target = torch.min(
                        qf1_next_target, qf2_next_target
                    ) - self.alpha * next_state_log_pi.reshape(-1, 1)
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.obs, data.actions).view(-1)
                qf2_a_values = self.qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = (qf1_loss + qf2_loss) / 2.0

                self.q_optimizer.zero_grad()
                qf_loss.backward()  # type: ignore
                self.q_optimizer.step()

                # Actor Loss
                qf1_pi = self.qf1(data.obs, actions_pi)
                qf2_pi = self.qf2(data.obs, actions_pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_prob) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()  # type: ignore
                self.actor_optimizer.step()

                if global_step % self.args.target_network_frequency == 0:
                    polyak_update(
                        self.qf1.parameters(),
                        self.qf1_target.parameters(),
                        self.args.tau,
                    )
                    polyak_update(
                        self.qf2.parameters(),
                        self.qf2_target.parameters(),
                        self.args.tau,
                    )

            update_time = time.perf_counter() - update_time
            cumulative_times["update_time"] += update_time

            # Log training-related data
            if (
                (global_step - self.args.training_freq) // self.args.log_freq
                < global_step // self.args.log_freq
                and learning_has_started
            ):
                self.logger.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                self.logger.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                self.logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                self.logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                self.logger.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, global_step
                )
                self.logger.add_scalar(
                    "losses/actor_loss", actor_loss.item(), global_step
                )
                self.logger.add_scalar("time/update_time", update_time, global_step)
                self.logger.add_scalar("time/rollout_time", rollout_time, global_step)
                self.logger.add_scalar(
                    "time/rollout_fps",
                    global_steps_per_iteration / rollout_time,
                    global_step,
                )
                for k, v in cumulative_times.items():
                    self.logger.add_scalar(f"time/total_{k}", v, global_step)
                self.logger.add_scalar(
                    "time/total_rollout+update_time",
                    cumulative_times["rollout_time"] + cumulative_times["update_time"],
                    global_step,
                )

        if self.args.save_model:
            model_path = (
                Path(CFG.rl_policy_save_dir)
                / f"runs/{self.args.exp_name}/final_ckpt.pt"
            )
            self.save(model_path)
            logging.info(f"model saved to {model_path}")
        env.close()

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "qf1": self.qf1_target.state_dict(),
                "qf2": self.qf2_target.state_dict(),
                "log_alpha": getattr(self, "log_alpha", None),
            },
            path.as_posix(),
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path.as_posix(), map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.qf1.load_state_dict(ckpt["qf1"])
        self.qf2.load_state_dict(ckpt["qf2"])
        if "log_alpha" in ckpt and ckpt["log_alpha"] is not None:
            self.log_alpha = ckpt["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
        self._trained = True
