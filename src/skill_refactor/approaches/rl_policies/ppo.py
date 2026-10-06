"""
Proximal Policy Optimization (PPO) implementation which is largely based on
the implementation from ManiSkill3:
https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py
"""

from __future__ import annotations

# === Standard library ===
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# === Third-party ===
import dacite
import numpy as np
import torch
import yaml

# === Local ===
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from skill_refactor.approaches.rl_policies.base import (
    Logger,
    Policy,
)
from skill_refactor.settings import CFG


# Default arguments for PPO
@dataclass
class PPOArgs:
    """Arguments for the Soft Actor-Critic (SAC) algorithm."""

    seed: int = CFG.seed
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
    wandb_group: str = "PPO"
    """The group of the run for wandb."""
    capture_video: bool = True
    """Whether to capture videos of the self.agent performances (check out `videos`
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

    # Environment specific arguments
    num_envs: int = 512
    """The number of parallel environments."""
    hidden_sz: int = 256
    """The hidden size of the MLP networks."""
    num_eval_envs: int = 16
    """The number of parallel evaluation environments."""
    partial_reset: bool = True
    """Whether to let parallel environments reset upon termination instead of
    truncation."""
    eval_partial_reset: bool = False
    """Whether to let parallel evaluation environments reset upon termination instead of
    truncation."""
    num_steps: int = 50
    """The number of steps to run in each environment per policy rollout."""
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
    total_timesteps: int = 10_000_000
    """Total timesteps of the experiments."""
    learning_rate: float = 3e-4
    """The learning rate of the optimizer."""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks."""
    gamma: float = 0.8
    """The discount factor gamma."""
    gae_lambda: float = 0.9
    """The lambda for the general advantage estimation."""
    num_minibatches: int = 32
    """The number of mini-batches."""
    update_epochs: int = 4
    """The K epochs to update the policy."""
    norm_adv: bool = True
    """Toggles advantages normalization."""
    clip_coef: float = 0.2
    """The surrogate clipping coefficient."""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the
    paper."""
    ent_coef: float = 0.0
    """Coefficient of the entropy."""
    vf_coef: float = 0.5
    """Coefficient of the value function."""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping."""
    target_kl: float = 0.1
    """The target KL divergence threshold."""
    reward_scale: float = 1.0
    """Scale the reward by this factor."""
    finite_horizon_gae: bool = False
    normalize_obs: bool = False
    """Whether to normalize observations using running mean and std."""
    use_dual_agent: bool = False
    """Whether to use DualAgent with separate networks for two state clusters."""
    cluster_dim: int = 0
    """Observation dimension to use for clustering (if use_dual_agent=True)."""

    # to be filled in runtime
    batch_size: int = 0
    """The batch size (computed in runtime)"""
    minibatch_size: int = 0
    """The mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """The number of iterations (computed in runtime)"""


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize a layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer


def load_ppo_args(config_path: str) -> PPOArgs:
    """Load PPO arguments from a YAML configuration file."""
    with open(config_path, "rb") as f:
        config_data = yaml.safe_load(f)

    return dacite.from_dict(PPOArgs, config_data)


class Agent(nn.Module):
    """Proximal Policy Optimization (PPO) agent."""

    def __init__(self, envs: ManiSkillVectorEnv, hidden_sz: int = 256) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), hidden_sz
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sz, hidden_sz)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sz, hidden_sz)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sz, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), hidden_sz
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sz, hidden_sz)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sz, hidden_sz)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_sz, np.prod(envs.single_action_space.shape)),  # type: ignore
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5  # type: ignore
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get the value of the state."""
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action from the policy."""
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)  # type: ignore
        return probs.sample()  # type: ignore

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an action and its value from the policy."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)  # type: ignore
        if action is None:
            action = probs.sample()  # type: ignore
        return (
            action,
            probs.log_prob(action).sum(1),  # type: ignore
            probs.entropy().sum(1),  # type: ignore
            self.critic(x),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the agent."""
        return self.get_action(x, deterministic=True)


class DualAgent(nn.Module):
    """Dual PPO agent with separate networks for two clusters of states.

    This agent handles states that fall into two distinct clusters (e.g., one cluster
    with negative values in certain dimensions, another with positive values). Using
    separate actor-critic networks for each cluster prevents NaN gradients that can
    occur when a single network tries to handle both clusters.

    Args:
        envs: Vector environment
        cluster_dim: Observation dimension index to use for clustering (default: 0)
    """

    def __init__(self, envs: ManiSkillVectorEnv, cluster_dim: int = 0) -> None:
        super().__init__()
        self.cluster_dim = cluster_dim

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)  # type: ignore

        # Cluster 0 networks (for negative values in cluster_dim)
        self.critic_0 = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean_0 = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd_0 = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        # Cluster 1 networks (for positive values in cluster_dim)
        self.critic_1 = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean_1 = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd_1 = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def _get_cluster_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which samples belong to which cluster.

        Args:
            x: Observation tensor [batch_size, obs_dim]

        Returns:
            Tuple of (mask_0, mask_1) where mask_i is a boolean tensor indicating
            which samples belong to cluster i
        """
        cluster_indicator = x[:, self.cluster_dim]
        mask_0 = cluster_indicator < 0  # Negative values -> cluster 0
        mask_1 = ~mask_0  # Non-negative values -> cluster 1
        return mask_0, mask_1

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get the value of the state using cluster-specific critics.

        Args:
            x: Observation tensor [batch_size, obs_dim]

        Returns:
            Value estimates [batch_size, 1]
        """
        mask_0, mask_1 = self._get_cluster_masks(x)
        values = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

        if mask_0.any():
            values[mask_0] = self.critic_0(x[mask_0])
        if mask_1.any():
            values[mask_1] = self.critic_1(x[mask_1])

        return values

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action from the policy using cluster-specific actors.

        Args:
            x: Observation tensor [batch_size, obs_dim]
            deterministic: If True, return mean action; if False, sample from distribution

        Returns:
            Actions [batch_size, action_dim]
        """
        mask_0, mask_1 = self._get_cluster_masks(x)
        action_dim = self.actor_logstd_0.shape[1]
        actions = torch.zeros(x.shape[0], action_dim, device=x.device, dtype=x.dtype)

        # Process cluster 0
        if mask_0.any():
            action_mean_0 = self.actor_mean_0(x[mask_0])
            if deterministic:
                actions[mask_0] = action_mean_0
            else:
                action_logstd_0 = self.actor_logstd_0.expand_as(action_mean_0)
                action_std_0 = torch.exp(action_logstd_0)
                probs_0 = Normal(action_mean_0, action_std_0)  # type: ignore
                actions[mask_0] = probs_0.sample()  # type: ignore

        # Process cluster 1
        if mask_1.any():
            action_mean_1 = self.actor_mean_1(x[mask_1])
            if deterministic:
                actions[mask_1] = action_mean_1
            else:
                action_logstd_1 = self.actor_logstd_1.expand_as(action_mean_1)
                action_std_1 = torch.exp(action_logstd_1)
                probs_1 = Normal(action_mean_1, action_std_1)  # type: ignore
                actions[mask_1] = probs_1.sample()  # type: ignore

        return actions

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an action and its value using cluster-specific networks.

        Args:
            x: Observation tensor [batch_size, obs_dim]
            action: Optional action tensor [batch_size, action_dim]. If None, samples new actions.

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        mask_0, mask_1 = self._get_cluster_masks(x)
        batch_size = x.shape[0]
        action_dim = self.actor_logstd_0.shape[1]

        # Initialize output tensors
        if action is None:
            actions = torch.zeros(
                batch_size, action_dim, device=x.device, dtype=x.dtype
            )
        else:
            actions = action
        logprobs = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        entropies = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        values = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        # Process cluster 0
        if mask_0.any():
            x_0 = x[mask_0]
            action_mean_0 = self.actor_mean_0(x_0)
            action_logstd_0 = self.actor_logstd_0.expand_as(action_mean_0)
            action_std_0 = torch.exp(action_logstd_0)
            probs_0 = Normal(action_mean_0, action_std_0)  # type: ignore

            if action is None:
                actions[mask_0] = probs_0.sample()  # type: ignore
                action_0 = actions[mask_0]
            else:
                action_0 = action[mask_0]

            logprobs[mask_0] = probs_0.log_prob(action_0).sum(1)  # type: ignore
            entropies[mask_0] = probs_0.entropy().sum(1)  # type: ignore
            values[mask_0] = self.critic_0(x_0)

        # Process cluster 1
        if mask_1.any():
            x_1 = x[mask_1]
            action_mean_1 = self.actor_mean_1(x_1)
            action_logstd_1 = self.actor_logstd_1.expand_as(action_mean_1)
            action_std_1 = torch.exp(action_logstd_1)
            probs_1 = Normal(action_mean_1, action_std_1)  # type: ignore

            if action is None:
                actions[mask_1] = probs_1.sample()  # type: ignore
                action_1 = actions[mask_1]
            else:
                action_1 = action[mask_1]

            logprobs[mask_1] = probs_1.log_prob(action_1).sum(1)  # type: ignore
            entropies[mask_1] = probs_1.entropy().sum(1)  # type: ignore
            values[mask_1] = self.critic_1(x_1)

        return actions, logprobs, entropies, values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the agent."""
        return self.get_action(x, deterministic=True)


class PPOPolicy(Policy):
    """Soft Actor-Critic (SAC) policy implementation."""

    def __init__(
        self,
        seed: int,
        rl_config: str = "",
    ) -> None:
        super().__init__(seed)
        self.device = CFG.device
        self.args: PPOArgs = load_ppo_args(rl_config)
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
        """Initialize the PPO policy with the given environment."""
        # update the args with the environment-specific values
        num_envs = env.num_envs
        if num_envs != self.args.num_envs:
            logging.warning(
                f"Number of environments in the provided environment ({num_envs}) "
                f"does not match the configured number of environments ({self.args.num_envs}). "
                f"Using {num_envs} instead."
            )
            self.args.num_envs = num_envs

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(
            self.args.batch_size // self.args.num_minibatches
        )
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size

        # Create agent (either standard or dual)
        agent: Union[Agent, DualAgent]
        if self.args.use_dual_agent:
            logging.info(f"Using DualAgent with cluster_dim={self.args.cluster_dim}")
            agent = DualAgent(env, cluster_dim=self.args.cluster_dim).to(self.device)
        else:
            agent = Agent(env).to(self.device)
        self.agent = agent

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.args.num_steps, self.args.num_envs)
            + (
                tuple(env.single_observation_space.shape)
                if env.single_observation_space.shape is not None
                else ()
            )
        ).to(self.device)
        self.actions = torch.zeros(
            (self.args.num_steps, self.args.num_envs)
            + (
                tuple(env.single_action_space.shape)
                if env.single_action_space.shape is not None
                else ()
            )
        ).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        del obs  # No internal state to reset

    def can_initiate(self) -> bool:
        return True

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using current running mean and std."""
        if not self.args.normalize_obs:
            return obs
        return (obs - self.curr_obs_mean) / (self.curr_obs_std + 1e-8)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = self.agent.get_action(obs, deterministic=True)
        return action

    def train(
        self,
        env: ManiSkillVectorEnv,
        eval_env: Optional[ManiSkillVectorEnv] = None,
        train_data: Optional[object] = None,
    ) -> None:
        del train_data  # Unused

        # Initialize observation normalization variables
        obs_shape = env.single_observation_space.shape
        if obs_shape is None:
            obs_shape = ()
        self.curr_obs_mean = torch.zeros(obs_shape, device=self.device)
        self.curr_obs_std = torch.ones(obs_shape, device=self.device)

        next_obs, _ = env.reset(seed=self.args.seed)
        if eval_env is not None:
            eval_obs, _ = eval_env.reset(seed=self.args.seed)
        next_done = torch.zeros(self.args.num_envs, device=self.device)
        global_step = 0

        action_space_low, action_space_high = torch.from_numpy(
            env.single_action_space.low  # type: ignore
        ).to(self.device), torch.from_numpy(
            env.single_action_space.high
        ).to(  # type: ignore
            self.device
        )

        def clip_action(action: torch.Tensor):
            return torch.clamp(action.detach(), action_space_low, action_space_high)

        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            logging.info(f"Epoch: {iteration}, global_step={global_step}")
            final_values = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            self.agent.eval()
            if iteration % self.args.eval_freq == 1 and eval_env is not None:
                logging.info("Evaluating")
                eval_obs, _ = eval_env.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(CFG.max_rl_steps):
                    with torch.no_grad():
                        normalized_eval_obs = self.normalize_obs(eval_obs)
                        eval_obs, _, _, _, eval_infos = eval_env.step(
                            clip_action(self.get_action(normalized_eval_obs))
                        )
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                evaluated_steps = CFG.max_rl_steps * self.args.num_eval_envs
                logging.info(
                    f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
                )
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if self.logger is not None:
                        self.logger.add_scalar(f"eval/{k}", mean, global_step)
                    logging.info(f"eval_{k}_mean={mean}")
                if self.args.evaluate:
                    break
            if self.args.save_model and iteration % self.args.eval_freq == 1:
                model_path = (
                    Path(CFG.rl_policy_save_dir)
                    / f"runs/{CFG.exp_name}/ckpt_{global_step}.pt"
                )
                base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                base_path.mkdir(parents=True, exist_ok=True)
                self.save(model_path)
                logging.info(f"model saved to {model_path}")
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            rollout_time = time.time()
            # ALGO LOGIC: collect data
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    normalized_obs = self.normalize_obs(next_obs)
                    action, logprob, _, value = self.agent.get_action_and_value(
                        normalized_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = env.step(
                    clip_action(action)
                )
                next_done = torch.logical_or(terminations, truncations).to(
                    torch.float32
                )
                self.rewards[step] = reward.view(-1) * self.args.reward_scale

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k, v in final_info["episode"].items():
                        self.logger.add_scalar(
                            f"train/{k}", v[done_mask].float().mean(), global_step
                        )
                    with torch.no_grad():
                        final_values[
                            step,
                            torch.arange(self.args.num_envs, device=self.device)[
                                done_mask
                            ],
                        ] = self.agent.get_value(
                            self.normalize_obs(infos["final_observation"][done_mask])
                        ).view(
                            -1
                        )
            rollout_time = time.time() - rollout_time

            # Update observation normalization statistics
            if self.args.normalize_obs:
                with torch.no_grad():
                    # Calculate mean and std from current rollout observations
                    batch_obs = self.obs.reshape(-1, *self.obs.shape[2:])
                    self.curr_obs_mean = batch_obs.mean(dim=0)
                    self.curr_obs_std = batch_obs.std(dim=0)

            # bootstrap value according to termination and truncation
            with torch.no_grad():
                normalized_next_obs = self.normalize_obs(next_obs)
                next_value = self.agent.get_value(normalized_next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = torch.tensor(0.0, device=self.device)
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    real_next_values = (
                        next_not_done * nextvalues + final_values[t]
                    )  # t instead of t+1
                    # next_not_done means nextvalues is computed from the correct next_obs
                    # if next_not_done is 1, final_values is always 0
                    # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                    if self.args.finite_horizon_gae:
                        if t == self.args.num_steps - 1:  # initialize
                            lam_coef_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the first term
                            reward_term_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the second term
                            value_term_sum = torch.tensor(
                                0.0, device=self.device
                            )  # the sum of the third term
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + self.args.gae_lambda * lam_coef_sum
                        reward_term_sum = (
                            self.args.gae_lambda * self.args.gamma * reward_term_sum
                            + lam_coef_sum * self.rewards[t]
                        )
                        value_term_sum = (
                            self.args.gae_lambda * self.args.gamma * value_term_sum
                            + self.args.gamma * real_next_values
                        )

                        advantages[t] = (
                            reward_term_sum + value_term_sum
                        ) / lam_coef_sum - self.values[t]
                    else:
                        delta = (
                            self.rewards[t]
                            + self.args.gamma * real_next_values
                            - self.values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.args.gamma
                            * self.args.gae_lambda
                            * next_not_done
                            * lastgaelam
                        )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                returns = advantages + self.values

            # Normalize observations before agent update
            if self.args.normalize_obs:
                self.obs = self.normalize_obs(self.obs)

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # ALGO LOGIC: update the agent with the collected data
            self.agent.train()
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            update_time = time.time()
            for _ in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    try:
                        (
                            _,
                            newlogprob,
                            entropy,
                            newvalue,
                        ) = self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                    except ValueError as e:
                        logging.error(
                            f"Error during action and value computation: {e}. "
                        )
                        logging.error(
                            f"b_obs x min: {b_obs[:, [0, 7, 13]].min()}, "
                            f"b_obs x max: {b_obs[:, [0, 7, 13]].max()}"
                            f"b_obs max: {b_obs.max()}"
                            f"b_obs min: {b_obs.min()}"
                            f"b_obs Nan: {b_obs.isnan().any()}"
                        )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    if (
                        self.args.target_kl is not None
                        and approx_kl > self.args.target_kl
                    ):
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.args.ent_coef * entropy_loss
                        + v_loss * self.args.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()  # type: ignore
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.args.max_grad_norm
                    )
                    self.optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            update_time = time.time() - update_time

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            self.logger.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.logger.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            clipfracs_log = float(np.mean(clipfracs))
            self.logger.add_scalar("losses/clipfrac", clipfracs_log, global_step)
            self.logger.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            elapsed_time = time.time() - start_time
            self.logger.add_scalar(
                "charts/SPS", int(global_step / elapsed_time), global_step
            )
            self.logger.add_scalar("time/step", global_step, global_step)
            self.logger.add_scalar("time/update_time", update_time, global_step)
            self.logger.add_scalar("time/rollout_time", rollout_time, global_step)
            self.logger.add_scalar(
                "time/rollout_fps",
                self.args.num_envs * self.args.num_steps / rollout_time,
                global_step,
            )
        if not self.args.evaluate:
            if self.args.save_model:
                model_path = (
                    Path(CFG.rl_policy_save_dir) / f"runs/{CFG.exp_name}/final_ckpt.pt"
                )
                self.save(model_path)
                logging.info(f"model saved to {model_path}")
            self.logger.close()

    def save(self, path: Path) -> None:
        torch.save(
            {
                "agent": self.agent.state_dict(),
            },
            path.as_posix(),
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path.as_posix(), map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self._trained = True
