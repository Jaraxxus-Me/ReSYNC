"""
PPO-C Policy: Combines Proximal Policy Optimization (PPO) with Terminal Classifier learning.

Two-stage training approach:
1. Stage 1 (PPO Training): Train a standard PPO policy until it reaches a reward threshold
2. Stage 2 (Classifier Training): Train a binary classifier to decide when to call the planner:
   - Classifier output 0: Use the trained PPO policy
   - Classifier output 1: Call planner via frozen/static action

The classifier learns to identify states where the learned RL policy is insufficient
and should defer to symbolic planning.
"""

from __future__ import annotations

# === Standard library ===
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

# === Third-party ===
import dacite
import numpy as np
import torch
import yaml

# === Local ===
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from skill_refactor.approaches.rl_policies.base import Logger, Policy
from skill_refactor.approaches.rl_policies.ppo import Agent as PPOAgent
from skill_refactor.approaches.rl_policies.ppo import DualAgent as PPODualAgent
from skill_refactor.approaches.rl_policies.ppo import (
    PPOArgs,
    layer_init,
)
from skill_refactor.benchmarks.wrappers import PlanningStatesVectorEnv
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action


@dataclass
class PPOCArgs(PPOArgs):
    """Arguments for the PPO-C algorithm, extending PPOArgs."""

    # Classifier-specific arguments
    resume_ppo: str = ""
    """Path to a pre-trained PPO model to load before training."""
    resume_buffer: str = ""
    """Path to a saved replay buffer to load instead of collecting data."""
    c_buffer_size: int = 30000
    """Replay buffer size for classifier training."""
    c_learning_rate: float = 1e-4
    """Learning rate for classifier network."""
    c_batch_size: int = 128
    """Batch size for classifier updates."""
    c_num_iterations: int = 5000
    """Number of training iterations for classifier."""
    reward_threshold: float = 0.8
    """Average reward threshold to trigger classifier training (Stage 2)."""
    reward_continuing_episodes: int = 2
    """Last N episodes to consider for average reward."""
    eval_freq_c: int = 500
    """Evaluation frequency during classifier training (in timesteps)."""
    trivial_classifier: bool = False
    """If True, use a trivial classifier that never terminates."""
    arm_action_only: bool = False
    """If True, only use arm actions and base is frozen."""
    close_finger: bool = False


def load_ppoc_args(config_path: str) -> PPOCArgs:
    """Load PPO-C arguments from a YAML configuration file."""
    with open(config_path, "rb") as f:
        config_data = yaml.safe_load(f)

    return dacite.from_dict(PPOCArgs, config_data)


class TrivialClassifier(nn.Module):
    """A trivial classifier that never calls the planner."""

    def eval(self) -> "TrivialClassifier":  # type: ignore[override]
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always predicts class 0 (use PPO policy)."""
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 2), device=x.device)


class ClassifierNetwork(nn.Module):
    """Binary classifier network for deciding between PPO policy and planner.

    Outputs logits for two classes:
    - Class 0: Use PPO policy
    - Class 1: Call planner (frozen action)
    """

    def __init__(self, obs_shape: tuple) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 128)),
            nn.GELU(),
            nn.BatchNorm1d(128),
            layer_init(nn.Linear(128, 128)),
            nn.GELU(),
            nn.BatchNorm1d(128),
            layer_init(nn.Linear(128, 2)),  # Binary classification: 0=PPO, 1=planner
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier network.

        Args:
            x: Observation tensor

        Returns:
            Logits for binary classification (shape: [batch_size, 2])
        """
        return self.network(x)


class ReplayBuffer:
    """Replay buffer for classifier training.

    Stores (observation, reward, done) tuples and provides balanced sampling to handle
    class imbalance (50% positive rewards, 50% negative rewards). Supports train/val
    split with 80/20 probability.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        device: torch.device,
        train_split: float = 0.8,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device
        self.train_split = train_split

        self.observations = torch.zeros((buffer_size,) + obs_shape, device=device)
        self.rewards = torch.zeros((buffer_size,), device=device) - 1
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        # Track which indices are in train vs val set
        self.is_train = torch.zeros((buffer_size,), dtype=torch.bool, device=device)

    def add(
        self,
        obs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a transition to the buffer.

        Each new sample has train_split probability of being in the train set.
        """
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.observations[self.pos] = obs[i]
            self.rewards[self.pos] = reward[i]
            self.dones[self.pos] = done[i]
            # Randomly assign to train or val with train_split probability
            self.is_train[self.pos] = (
                torch.rand(1, device=self.device).item() < self.train_split
            )

            self.pos = (self.pos + 1) % self.buffer_size
            if self.pos == 0:
                self.full = True

    def _sample_from_indices(
        self, batch_size: int, valid_indices: torch.Tensor
    ) -> tuple:
        """Sample a balanced batch from the given valid indices.

        Args:
            batch_size: Number of samples to draw
            valid_indices: Indices to sample from (train or val set)

        Returns:
            Tuple of (observations, rewards, dones)
        """
        # Filter by valid indices
        positive_rwd_indices = valid_indices[self.rewards[valid_indices] == 1]
        negative_rwd_indices = valid_indices[self.rewards[valid_indices] == 0]

        # Torch-based balanced sampling (50/50 positive/negative)
        n_positive = batch_size // 2
        n_negative = batch_size - n_positive

        sampled_positive_indices = positive_rwd_indices[
            torch.randint(
                0, len(positive_rwd_indices), (n_positive,), device=self.device
            )
        ]
        sampled_negative_indices = negative_rwd_indices[
            torch.randint(
                0, len(negative_rwd_indices), (n_negative,), device=self.device
            )
        ]
        indices = torch.cat((sampled_positive_indices, sampled_negative_indices))

        return (
            self.observations[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the entire buffer.

        Returns:
            Tuple of (observations, rewards, dones)
        """
        size = self.buffer_size if self.full else self.pos
        all_indices = torch.arange(size, device=self.device)
        return self._sample_from_indices(batch_size, all_indices)

    def sample_train(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the training set only.

        Returns:
            Tuple of (observations, rewards, dones)
        """
        size = self.buffer_size if self.full else self.pos
        train_indices = torch.arange(size, device=self.device)[self.is_train[:size]]
        return self._sample_from_indices(batch_size, train_indices)

    def sample_val(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the validation set only.

        Returns:
            Tuple of (observations, rewards, dones)
        """
        size = self.buffer_size if self.full else self.pos
        val_indices = torch.arange(size, device=self.device)[~self.is_train[:size]]
        return self._sample_from_indices(batch_size, val_indices)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.buffer_size if self.full else self.pos

    def save(self, path: Path) -> None:
        """Save the replay buffer to a file.

        Args:
            path: Path to save the buffer to
        """
        torch.save(
            {
                "observations": self.observations,
                "rewards": self.rewards,
                "dones": self.dones,
                "is_train": self.is_train,
                "pos": self.pos,
                "full": self.full,
                "buffer_size": self.buffer_size,
                "train_split": self.train_split,
            },
            path.as_posix(),
        )

    def load(self, path: Path) -> None:
        """Load the replay buffer from a file.

        Args:
            path: Path to load the buffer from
        """
        checkpoint = torch.load(path.as_posix(), map_location=self.device)
        self.observations = checkpoint["observations"].to(self.device)
        self.rewards = checkpoint["rewards"].to(self.device)
        self.dones = checkpoint["dones"].to(self.device)
        self.is_train = checkpoint["is_train"].to(self.device)
        self.pos = checkpoint["pos"]
        self.full = checkpoint["full"]
        # Verify buffer size matches
        if checkpoint["buffer_size"] != self.buffer_size:
            logging.warning(
                f"Loaded buffer size {checkpoint['buffer_size']} does not match "
                f"current buffer size {self.buffer_size}"
            )
        if checkpoint["train_split"] != self.train_split:
            logging.warning(
                f"Loaded train_split {checkpoint['train_split']} does not match "
                f"current train_split {self.train_split}"
            )


class PPOCPolicy(Policy):
    """PPO-C policy combining PPO with classifier learning for deciding when to call the
    planner."""

    def __init__(
        self,
        seed: int,
        rl_config: str = "",
    ) -> None:
        super().__init__(seed)
        self.device = CFG.device
        self.args: PPOCArgs = load_ppoc_args(rl_config)
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
        self.c_learning_enabled = False
        self.recent_rewards: list[float] = []
        self.best_eval_reward = float("-inf")
        self.best_model_path: Optional[Path] = None
        self.terminal_cls: TrivialClassifier | ClassifierNetwork

    @property
    def requires_training(self) -> bool:
        """Check if the policy requires training."""
        return True

    def initialize(self, env: PlanningStatesVectorEnv) -> None:
        """Initialize the PPO-C policy with the given environment.

        Sets up PPO agent, classifier network, replay buffer, and training buffers.
        """
        # Update the args with the environment-specific values
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

        # Initialize PPO agent (either standard or dual)
        agent: PPOAgent | PPODualAgent
        if self.args.use_dual_agent:
            logging.info(f"Using DualAgent with cluster_dim={self.args.cluster_dim}")
            agent = PPODualAgent(env, cluster_dim=self.args.cluster_dim).to(self.device)
        else:
            agent = PPOAgent(env).to(self.device)
        self.agent = agent

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # Initialize classifier network
        self.obs_shape = env.single_observation_space.shape
        assert self.obs_shape is not None, "Observation space must have a shape"
        if self.args.trivial_classifier:
            self.terminal_cls = TrivialClassifier()
        else:
            self.terminal_cls = ClassifierNetwork(self.obs_shape).to(self.device)  # type: ignore[assignment]
            self.cls_optimizer = optim.AdamW(
                self.terminal_cls.parameters(), lr=self.args.c_learning_rate
            )
            self.cls_loss_fn = nn.CrossEntropyLoss()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.args.c_buffer_size,
            self.obs_shape,
            self.device,
        )

        # Store environment action space info for frozen actions
        self.action_space_low = torch.from_numpy(
            env.single_action_space.low  # type: ignore
        ).to(self.device)
        self.action_space_high = torch.from_numpy(
            env.single_action_space.high  # type: ignore
        ).to(self.device)

        # ALGO Logic: Storage setup for PPO
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
        self._last_obs = torch.zeros(
            (self.args.num_envs,) + self.obs_shape, device=self.device
        )

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        bs = obs.shape[0]
        self._last_obs = torch.zeros(
            (bs,) + cast(tuple, self.obs_shape), device=self.device
        )

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip actions to the environment's action space."""
        clipped_action = torch.clamp(
            action, self.action_space_low, self.action_space_high
        )
        if self.args.arm_action_only:
            assert (
                not CFG.delta_finger_control
            ), "Delta finger control not supported with arm_action_only"
            # Freeze base actions (first 3 dimensions)
            clipped_action[:, :3] *= 0.0
            # Freeze gripper action
            clipped_action[:, -1] = self.action_space_low[-1]
            if self.args.close_finger:
                clipped_action[:, -1] = CFG.c_room_finger_close_value
        return clipped_action

    def can_initiate(self) -> bool:
        return True

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using current running mean and std."""
        if not self.args.normalize_obs:
            return obs
        return (obs - self.curr_obs_mean) / (self.curr_obs_std + 1e-8)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the policy.

        If classifier is enabled, uses the classifier to decide between PPO policy and
        calling the planner (frozen action). Otherwise, uses PPO policy only.
        """
        if self.terminal_cls.training:
            self.terminal_cls.eval()
        with torch.no_grad():
            # if self.c_learning_enabled:
            #     # Use classifier to decide between PPO and planner
            #     c_logits = self.terminal_cls(obs)
            #     c_actions = torch.argmax(c_logits, dim=1)

            #     # Get PPO actions
            #     ppo_actions = self.agent.get_action(obs, deterministic=True)

            #     # Get frozen actions (triggers planner call)
            #     frozen_actions = get_frozen_action(
            #         ppo_actions,  # Use PPO actions as base
            #         self.action_space_low,
            #         self.action_space_high,
            #         normalize_action=True,
            #         control_mode=CFG.control_mode,
            #     )

            #     # Select based on classifier decision
            #     actions = torch.where(
            #         c_actions.unsqueeze(-1) == 0,
            #         ppo_actions,
            #         frozen_actions,
            #     )
            #     return actions

            # Use PPO policy only
            action = self.agent.get_action(obs, deterministic=True)
            clipped_action = self.clip_action(action)
            return clipped_action

    def terminate(self, obs: torch.Tensor) -> torch.Tensor:
        """Determine if the current skill should terminate (i.e., call planner).

        Returns True for states where the classifier predicts to call the planner.
            Note: This is only used during evaluation/testing, not training.
        """
        if self.terminal_cls.training:
            self.terminal_cls.eval()
        with torch.no_grad():
            c_logits = self.terminal_cls(obs)
            c_predictions = torch.argmax(c_logits, dim=1)
            scene_static = torch.all(
                torch.isclose(obs, self._last_obs, atol=1e-4), dim=1
            )
            self._last_obs = obs.clone()
            # scene static and classifier predicts to call planner
            terminal = scene_static & (c_predictions == 1)
            return terminal

    def evaluate(
        self,
        eval_env: PlanningStatesVectorEnv,
        global_step: int,
    ) -> tuple[int, int, dict]:
        """Evaluate the current policy.

        Args:
            eval_env: Evaluation environment
            global_step: Current global step for logging

        Returns:
            Tuple of (evaluated_steps, num_episodes, eval_metrics)
        """
        logging.info("Evaluating")
        eval_obs, _ = eval_env.reset()
        eval_metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(eval_env.skill_max_steps):
            with torch.no_grad():
                normalized_eval_obs = self.normalize_obs(eval_obs)
                eval_obs, _, _, _, eval_infos = eval_env.step(
                    self.clip_action(self.get_action(normalized_eval_obs))
                )
                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
        evaluated_steps = eval_env.skill_max_steps * eval_env.num_envs
        logging.info(
            f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
        )

        for k, v in eval_metrics.items():
            mean = torch.stack(v).float().mean()
            if self.logger is not None:
                self.logger.add_scalar(f"eval/{k}", mean, global_step)
            logging.info(f"eval_{k}_mean={mean}")

        return evaluated_steps, num_episodes, eval_metrics

    def train_ppo(
        self,
        env: PlanningStatesVectorEnv,
        eval_env: Optional[PlanningStatesVectorEnv],
        start_time: float,
    ) -> int:
        """PPO training loop (Stage 1).

        Trains PPO until reward threshold is reached or max iterations completed.
        Saves checkpoints and tracks best model for later classifier training.

        Args:
            env: Training environment
            eval_env: Evaluation environment
            start_time: Training start time for logging

        Returns:
            Final global_step
        """
        logging.info("=" * 50)
        logging.info("STAGE 1: PPO TRAINING")
        logging.info("=" * 50)

        # Initialize - exactly like ppo.py
        next_obs, _ = env.reset(seed=self.args.seed)
        next_done = torch.zeros(self.args.num_envs, device=self.device)
        global_step = 0

        # Main PPO training loop
        for iteration in range(1, self.args.num_iterations + 1):
            logging.info(f"Epoch: {iteration}, global_step={global_step}")
            final_values = torch.zeros(
                (self.args.num_steps, self.args.num_envs), device=self.device
            )
            self.agent.eval()

            # Evaluation - exactly like ppo.py
            if iteration % self.args.eval_freq == 1 and eval_env is not None:
                logging.info("Evaluating")
                # while True:
                #     self.c_learning_enabled = False
                evaluated_steps, num_episodes, eval_metrics = self.evaluate(
                    eval_env, global_step
                )
                logging.info(
                    f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
                )
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if self.logger is not None:
                        self.logger.add_scalar(f"eval/{k}", mean, global_step)
                    logging.info(f"eval_{k}_mean={mean}")

                    # Track best model and check threshold
                    if k == "return":
                        mean_return = mean.item()
                        self.recent_rewards.append(mean_return)

                        # Save checkpoint if best
                        if mean_return > self.best_eval_reward:
                            self.best_eval_reward = mean_return
                            self.best_model_path = (
                                Path(CFG.rl_policy_save_dir)
                                / f"runs/{CFG.exp_name}/best_ppo_ckpt.pt"
                            )
                            base_path = (
                                Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                            )
                            base_path.mkdir(parents=True, exist_ok=True)
                            self.save(self.best_model_path)
                            logging.info(
                                f"New best PPO model! Reward: {mean_return:.3f}, saved to {self.best_model_path}"
                            )

                        # Check if should end PPO training (early stopping)
                        if (
                            len(self.recent_rewards)
                            >= self.args.reward_continuing_episodes
                        ):
                            avg_reward = np.mean(
                                self.recent_rewards[
                                    -self.args.reward_continuing_episodes :
                                ]
                            )
                            if avg_reward >= self.args.reward_threshold:
                                logging.info(
                                    f"PPO reached reward threshold! "
                                    f"Avg reward: {avg_reward:.3f} >= {self.args.reward_threshold}"
                                )
                                logging.info(
                                    "Proceeding to classifier training stage..."
                                )
                                return global_step

                if self.args.evaluate:
                    return global_step

            # Save model - exactly like ppo.py
            if self.args.save_model and iteration % self.args.eval_freq == 1:
                model_path = (
                    Path(CFG.rl_policy_save_dir)
                    / f"runs/{CFG.exp_name}/ckpt_{global_step}.pt"
                )
                base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                base_path.mkdir(parents=True, exist_ok=True)
                self.save(model_path)
                logging.info(f"model saved to {model_path}")

            # Annealing the rate if instructed to do so - exactly like ppo.py
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            rollout_time = time.time()

            # ALGO LOGIC: collect data - exactly like ppo.py
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
                    self.clip_action(action)
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

            # Update observation normalization statistics - exactly like ppo.py
            if self.args.normalize_obs:
                with torch.no_grad():
                    batch_obs = self.obs.reshape(-1, *self.obs.shape[2:])
                    self.curr_obs_mean = batch_obs.mean(dim=0)
                    self.curr_obs_std = batch_obs.std(dim=0)

            # bootstrap value according to termination and truncation - exactly like ppo.py
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
                    real_next_values = next_not_done * nextvalues + final_values[t]

                    if self.args.finite_horizon_gae:
                        if t == self.args.num_steps - 1:
                            lam_coef_sum = torch.tensor(0.0, device=self.device)
                            reward_term_sum = torch.tensor(0.0, device=self.device)
                            value_term_sum = torch.tensor(0.0, device=self.device)
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
                        )
                returns = advantages + self.values

            # Normalize observations before agent update - exactly like ppo.py
            if self.args.normalize_obs:
                self.obs = self.normalize_obs(self.obs)

            # flatten the batch - exactly like ppo.py
            b_obs = self.obs.reshape(
                (-1,) + cast(tuple, env.single_observation_space.shape)
            )
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape(
                (-1,) + cast(tuple, env.single_action_space.shape)
            )
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # ALGO LOGIC: update the agent with the collected data - exactly like ppo.py
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
                        _, newlogprob, entropy, newvalue = (
                            self.agent.get_action_and_value(
                                b_obs[mb_inds], b_actions[mb_inds]
                            )
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

            # Logging - exactly like ppo.py
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

        logging.info("PPO training completed (max iterations reached)")
        return global_step

    def train_terminal(
        self,
        env: PlanningStatesVectorEnv,
        eval_env: Optional[PlanningStatesVectorEnv],
        start_global_step: int,
        start_time: float,
    ) -> int:
        """Classifier training loop (Stage 2).

        Loads best PPO checkpoint, then trains classifier to decide when to call planner.
        Collects data by enumerating different timesteps to call the planner.

        Args:
            env: Training environment
            eval_env: Evaluation environment for periodic evaluation
            start_global_step: Starting global step for logging
            start_time: Training start time for logging

        Returns:
            Final global_step
        """
        del start_time  # Unused in current implementation
        logging.info("=" * 50)
        logging.info("STAGE 2: CLASSIFIER TRAINING")
        logging.info("=" * 50)

        # Load best PPO checkpoint
        if self.best_model_path is not None and self.best_model_path.exists():
            logging.info(f"Loading best PPO checkpoint from {self.best_model_path}")
            self.load(self.best_model_path)
            logging.info(f"Best PPO model loaded (reward: {self.best_eval_reward:.3f})")
        else:
            logging.warning("No best PPO checkpoint found, using current policy")

        # Initialize
        global_step = start_global_step

        # Check if we should load a saved replay buffer
        if self.args.resume_buffer != "":
            buffer_path = Path(self.args.resume_buffer)
            if buffer_path.exists():
                logging.info(f"Loading replay buffer from {buffer_path}")
                self.replay_buffer.load(buffer_path)
                logging.info(
                    f"Replay buffer loaded: {len(self.replay_buffer)} samples "
                    f"(pos={self.replay_buffer.pos}, full={self.replay_buffer.full})"
                )
            else:
                logging.error(f"Replay buffer path {buffer_path} does not exist!")
                raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")
        else:
            # Collect data by enumerate calling planner in each rollout step
            logging.info("Collecting classifier training data...")
            obs, _ = env.reset(seed=self.args.seed)
            action_space_low, action_space_high = torch.from_numpy(
                env.single_action_space.low  # type: ignore
            ).to(self.device), torch.from_numpy(
                env.single_action_space.high  # type: ignore
            ).to(
                self.device
            )

            num_seeds = self.args.c_buffer_size // (
                self.args.num_envs * env.skill_max_steps
            )
            logging.info(
                f"Collecting data for {num_seeds} seeds of {env.skill_max_steps} steps each"
            )
            for s in range(num_seeds):
                logging.info(f"Data collection seed {s+1}/{num_seeds}")
                final_reward = torch.zeros(self.args.num_envs, device=self.device)
                for step in range(1, env.skill_max_steps):
                    logging.info(f"  Planner call step {step}/{env.skill_max_steps}")
                    logging.info(
                        f"  Current final rewards: {final_reward.cpu().numpy()}"
                    )
                    obs, _ = env.reset(seed=self.args.seed + s)
                    with torch.no_grad():
                        # Execute PPO for "step" steps to get to next planner call
                        # NOTE: We don't store data here, since we are enumerating planner calls
                        # these data "has been" collected in previous "steps" calls.
                        for _ in range(step):
                            normalized_obs = self.normalize_obs(obs)
                            # Get PPO actions
                            action, _, _, _ = self.agent.get_action_and_value(
                                normalized_obs
                            )
                            next_obs, _, _, _, _ = env.step(self.clip_action(action))
                            obs = next_obs
                        call_planner_step_obs = self.normalize_obs(obs)
                        # Execute planner call (frozen action) until episode terminates
                        frozen_action = get_frozen_action(
                            action,
                            action_space_low,
                            action_space_high,
                            normalize_action=True,
                            control_mode=CFG.control_mode,
                            obs=obs,
                        )
                        done = False
                        while not done:
                            _, reward, _, truncations, _ = env.step(
                                self.clip_action(frozen_action)
                            )
                            final_reward = reward  # Track final reward for labeling
                            done = truncations.all()
                        # Store data: observation at planner call time, final reward (0 or 1)
                        self.replay_buffer.add(
                            call_planner_step_obs,
                            final_reward,
                            truncations,
                        )
                    if (
                        final_reward.sum() / self.args.num_envs
                    ) >= self.args.reward_threshold:
                        # Statistically all episodes have succeeded
                        break

            # Save the collected replay buffer
            buffer_save_path = (
                Path(CFG.rl_policy_save_dir) / f"runs/{CFG.exp_name}/replay_buffer.pt"
            )
            base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
            base_path.mkdir(parents=True, exist_ok=True)
            self.replay_buffer.save(buffer_save_path)
            logging.info(
                f"Replay buffer saved: {len(self.replay_buffer)} samples to {buffer_save_path}"
            )

        # Train classifier with cross-entropy loss
        best_terminal_return = -1.0
        best_classifier_path: Optional[Path] = None

        for iteration in range(1, self.args.c_num_iterations + 1):
            # Training step
            self.terminal_cls.train()
            cls_obs, cls_rewards, _ = self.replay_buffer.sample_train(
                self.args.c_batch_size
            )
            cls_logits = self.terminal_cls(cls_obs)
            train_loss = self.cls_loss_fn(cls_logits, cls_rewards.long())
            self.cls_optimizer.zero_grad()
            train_loss.backward()  # type: ignore
            self.cls_optimizer.step()

            # Logging
            self.logger.add_scalar(
                "c_training/train_loss", train_loss.item(), iteration
            )

            # Periodic evaluation
            if iteration % self.args.eval_freq_c == 0:
                self.terminal_cls.eval()
                logging.info(
                    f"Classifier iteration {iteration}/{self.args.c_num_iterations}"
                )

                # Compute validation loss
                with torch.no_grad():
                    val_obs, val_rewards, _ = self.replay_buffer.sample_val(
                        self.args.c_batch_size
                    )
                    val_logits = self.terminal_cls(val_obs)
                    val_loss = self.cls_loss_fn(val_logits, val_rewards.long())

                    # Compute validation accuracy
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_accuracy = (val_preds == val_rewards.long()).float().mean()

                logging.info(
                    f"Train loss: {train_loss.item():.3f}, "
                    f"Val loss: {val_loss.item():.3f}, "
                    f"Val accuracy: {val_accuracy.item():.3f}"
                )

                # Log validation metrics
                self.logger.add_scalar(
                    "c_training/val_loss", val_loss.item(), iteration
                )
                self.logger.add_scalar(
                    "c_training/val_accuracy", val_accuracy.item(), iteration
                )

                # Save best model based on eva return
                # Evaluate on the actual environment (like PPO does)
                if eval_env is not None:
                    logging.info("Evaluating classifier on environment")
                    evaluated_steps, num_episodes, eval_metrics = self.evaluate(
                        eval_env, global_step
                    )
                    logging.info(
                        f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
                    )
                    for k, v in eval_metrics.items():
                        mean = torch.stack(v).float().mean()
                        if self.logger is not None:
                            self.logger.add_scalar(f"c_eval/{k}", mean, iteration)
                        logging.info(f"c_eval_{k}_mean={mean}")
                        if k == "return":
                            mean_return = mean.item()
                            # Save checkpoint if best
                            # With equal, prefer later ckpt if mean return is the same (more training)
                            if mean_return >= best_terminal_return:
                                best_terminal_return = mean_return
                                best_classifier_path = (
                                    Path(CFG.rl_policy_save_dir)
                                    / f"runs/{CFG.exp_name}/best_classifier_ckpt.pt"
                                )
                                base_path = (
                                    Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                                )
                                base_path.mkdir(parents=True, exist_ok=True)
                                self.save(best_classifier_path)
                                logging.info(
                                    f"New best classifier! Mean Return: {mean_return:.3f}, "
                                    f"saved to {best_classifier_path}"
                                )

        # Load best classifier at the end
        if best_classifier_path is not None and best_classifier_path.exists():
            logging.info(f"Loading best classifier from {best_classifier_path}")
            self.load(best_classifier_path)
            logging.info(f"Best classifier loaded (mean return: {mean_return:.3f})")

        logging.info("Classifier training completed")
        return global_step

    def train(
        self,
        env: PlanningStatesVectorEnv,
        eval_env: Optional[PlanningStatesVectorEnv] = None,
        train_data: Optional[object] = None,
    ) -> None:
        """Two-stage training: PPO followed by classifier.

        Stage 1: Train PPO until reward threshold or max iterations
        Stage 2: Load best PPO, train classifier to learn when to call planner
        """
        del train_data  # Unused

        # Initialize observation normalization variables
        obs_shape = env.single_observation_space.shape
        if obs_shape is None:
            obs_shape = ()
        self.curr_obs_mean = torch.zeros(obs_shape, device=self.device)
        self.curr_obs_std = torch.ones(obs_shape, device=self.device)

        start_time = time.time()

        # HACK: Debugging c learning here
        # self.c_learning_enabled = True
        # global_step = self.train_terminal(env, eval_env, 0, start_time)

        # ========== STAGE 1: PPO TRAINING ==========
        if self.args.resume_ppo == "":
            global_step = self.train_ppo(env, eval_env, start_time)
        else:
            resume_path = Path(self.args.resume_ppo)
            self.load(resume_path)
            logging.info(f"Resumed PPO model from {self.args.resume_ppo}")
            assert (
                eval_env is not None
            ), "Evaluation environment required when resuming PPO"
            _, _, eval_metrics = self.evaluate(eval_env, 0)  # Evaluate loaded model
            mean_return = 0.0
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/{k}", mean, 0)
                logging.info(f"eval_{k}_mean={mean}")

                # Track best model and check threshold
                if k == "return":
                    mean_return = mean.item()
            global_step = 0
            self.recent_rewards = [mean_return] * self.args.reward_continuing_episodes

        # If we're in evaluation-only mode, stop here
        if self.args.evaluate:
            self.logger.close()
            return

        # ========== STAGE 2: CLASSIFIER TRAINING ==========
        # Only proceed to classifier training if PPO reached the threshold
        if len(self.recent_rewards) >= self.args.reward_continuing_episodes:
            avg_reward = np.mean(
                self.recent_rewards[-self.args.reward_continuing_episodes :]
            )
            if avg_reward >= self.args.reward_threshold:
                logging.info(
                    "PPO training successful, proceeding to classifier stage..."
                )
                self.c_learning_enabled = True
                if self.args.trivial_classifier:
                    logging.info("Using trivial classifier that never terminates.")
                else:
                    logging.info("Using learned classifier network.")
                    global_step = self.train_terminal(
                        env, eval_env, global_step, start_time
                    )
            else:
                logging.info(
                    f"PPO did not reach threshold (avg reward: {avg_reward:.3f} < {self.args.reward_threshold})"
                )
                logging.info("Skipping classifier training stage")
        else:
            logging.warning(
                "Not enough evaluation data, skipping classifier training stage"
            )

        # Final save and cleanup
        if self.args.save_model:
            model_path = (
                Path(CFG.rl_policy_save_dir) / f"runs/{CFG.exp_name}/final_ckpt.pt"
            )
            base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
            base_path.mkdir(parents=True, exist_ok=True)
            self.save(model_path)
            logging.info(f"Final model saved to {model_path}")

        logging.info("Training completed!")
        self.logger.close()

    def save(self, path: Path) -> None:
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "terminal_cls": self.terminal_cls.state_dict(),
            },
            path.as_posix(),
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path.as_posix(), map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        if "terminal_cls" in ckpt:
            if self.args.trivial_classifier:
                self.terminal_cls = TrivialClassifier()
            else:
                self.terminal_cls = ClassifierNetwork(self.obs_shape).to(self.device)  # type: ignore[assignment,arg-type]
                self.terminal_cls.load_state_dict(ckpt["terminal_cls"])
            self.c_learning_enabled = True
        self._trained = True
