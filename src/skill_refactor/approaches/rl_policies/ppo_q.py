"""
PPO-Q Policy: Combines Proximal Policy Optimization (PPO) with DQN learning.

This policy first trains a standard PPO policy until it achieves high reward,
then triggers Q-learning where a Q-network learns to choose between:
- Action 0: Use the trained PPO policy
- Action 1: Use a frozen/static action

The implementation is based on ppo.py with additional DQN components from dqn.py.
"""

from __future__ import annotations

# === Standard library ===
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# === Third-party ===
import dacite
import numpy as np
import torch
import torch.nn.functional as F
import yaml

# === Local ===
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from skill_refactor.approaches.rl_policies.base import Logger, Policy
from skill_refactor.approaches.rl_policies.ppo import Agent as PPOAgent
from skill_refactor.approaches.rl_policies.ppo import (
    PPOArgs,
    layer_init,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action


@dataclass
class PPOQArgs(PPOArgs):
    """Arguments for the PPO-Q algorithm, extending PPOArgs."""

    # Q-learning specific arguments
    resume_ppo: str = ""
    """Path to a pre-trained PPO model to load before training."""
    q_learning_starts: int = 1000
    """Number of rollouts before starting Q-learning."""
    rnd_action_dist_biase: float = 0.5
    """Bias for random action selection during Q-learning.

    0.5 means the last step has 50% chance to select call planner,
    the earlier steps have smaller chance to call planner.
    """
    q_buffer_size: int = 100000
    """Replay buffer size for Q-learning."""
    q_learning_rate: float = 2.5e-4
    """Learning rate for Q-network."""
    q_batch_size: int = 128
    """Batch size for Q-network updates."""
    q_train_frequency: int = 10
    """Frequency of Q-network training."""
    q_target_network_frequency: int = 500
    """Frequency of target network updates."""
    q_tau: float = 1.0
    """Target network update rate (1.0 = hard update)."""
    q_gamma: float = 0.99
    """Discount factor for Q-learning."""
    epsilon_start: float = 1.0
    """Starting epsilon for Q-network exploration."""
    epsilon_end: float = 0.05
    """Ending epsilon for Q-network exploration."""
    epsilon_decay_steps: int = 500000
    """Number of steps over which to decay epsilon."""
    reward_threshold: float = 0.5
    """Average reward threshold to trigger Q-learning."""
    reward_continuing_episodes: int = 2
    """Last N episodes to consider for average reward."""
    eval_freq_q: int = (
        1000  # in batched timesteps (e.g., 32 envs * 1000 steps = 32,000 steps)
    )
    """Evaluation frequency during Q-learning (in timesteps)."""


def load_ppoq_args(config_path: str) -> PPOQArgs:
    """Load PPO-Q arguments from a YAML configuration file."""
    with open(config_path, "rb") as f:
        config_data = yaml.safe_load(f)

    return dacite.from_dict(PPOQArgs, config_data)


class QNetwork(nn.Module):
    """Q-Network for binary action selection (PPO policy vs static action)."""

    def __init__(self, obs_shape: tuple) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 120)),
            nn.ReLU(),
            layer_init(nn.Linear(120, 84)),
            nn.ReLU(),
            layer_init(nn.Linear(84, 2)),  # Binary action: 0=PPO, 1=static
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Q-network."""
        return self.network(x)


class ReplayBuffer:
    """Simple replay buffer for Q-learning."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.device = device

        self.observations = torch.zeros((buffer_size,) + obs_shape, device=device)
        self.next_observations = torch.zeros((buffer_size,) + obs_shape, device=device)
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((buffer_size,), device=device) - 1
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a transition to the buffer."""
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.observations[self.pos] = obs[i]
            self.next_observations[self.pos] = next_obs[i]
            self.actions[self.pos] = action[i]
            self.rewards[self.pos] = reward[i]
            self.dones[self.pos] = done[i]

            self.pos = (self.pos + 1) % self.buffer_size
            if self.pos == 0:
                self.full = True

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions."""
        positive_rwd_indices = (self.rewards > 0).nonzero(as_tuple=True)[0]
        negative_rwd_indices = (
            (self.rewards >= -0.5) & (self.rewards <= -0.0)
        ).nonzero(as_tuple=True)[0]

        # Torch-based balanced sampling
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
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.buffer_size if self.full else self.pos


class PPOQPolicy(Policy):
    """PPO-Q policy combining PPO with Q-learning for action selection."""

    def __init__(
        self,
        seed: int,
        rl_config: str = "",
    ) -> None:
        super().__init__(seed)
        self.device = CFG.device
        self.args: PPOQArgs = load_ppoq_args(rl_config)
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
        self.q_learning_enabled = False
        self.recent_rewards: list[float] = []
        self.best_eval_reward = float("-inf")
        self.best_model_path: Optional[Path] = None
        self.planner_called_mask: Optional[torch.Tensor] = (
            None  # Track which envs called planner
        )

    @property
    def requires_training(self) -> bool:
        """Check if the policy requires training."""
        return True

    def initialize(self, env: ManiSkillVectorEnv) -> None:
        """Initialize the PPO-Q policy with the given environment."""
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

        # Initialize PPO agent
        self.agent = PPOAgent(env).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5
        )

        # Initialize Q-networks
        obs_shape = env.single_observation_space.shape
        self.q_network = QNetwork(obs_shape).to(self.device)
        self.target_q_network = QNetwork(obs_shape).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.args.q_learning_rate
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.args.q_buffer_size,
            obs_shape,
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

    def reset(self, obs: torch.Tensor) -> None:
        """Reset the policy state."""
        del obs  # No internal state to reset

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip actions to the environment's action space."""
        return torch.clamp(action, self.action_space_low, self.action_space_high)

    def can_initiate(self) -> bool:
        return True

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using current running mean and std."""
        if not self.args.normalize_obs:
            return obs
        return (obs - self.curr_obs_mean) / (self.curr_obs_std + 1e-8)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the policy."""
        with torch.no_grad():
            if self.q_learning_enabled:
                # Use Q-network to decide between PPO and static action
                q_values = self.q_network(obs)
                q_actions = torch.argmax(q_values, dim=1)

                # Get PPO actions
                ppo_actions = self.agent.get_action(obs, deterministic=True)

                # Get frozen actions
                frozen_actions = get_frozen_action(
                    ppo_actions,  # Use PPO actions as base
                    self.action_space_low,
                    self.action_space_high,
                    normalize_action=True,
                    control_mode=CFG.control_mode,
                )

                # Select based on Q-network decision
                actions = torch.where(
                    q_actions.unsqueeze(-1) == 0,
                    ppo_actions,
                    frozen_actions,
                )
                return actions
            else:
                # Use PPO policy only
                action = self.agent.get_action(obs, deterministic=True)
                return action

    def terminate(self, obs: torch.Tensor) -> torch.Tensor:
        """Determine if the current skill should terminate."""
        with torch.no_grad():
            q_values = self.q_network(obs)
            q_actions = torch.argmax(q_values, dim=1)
            return q_actions == 1  # Terminate if static action is chosen

    def linear_schedule(
        self, start_val: float, end_val: float, duration: int, t: int
    ) -> float:
        """Linear schedule for epsilon decay."""
        slope = (end_val - start_val) / duration
        return max(slope * t + start_val, end_val)

    def evaluate(
        self,
        eval_env: ManiSkillVectorEnv,
        global_step: int,
    ) -> tuple[int, int, dict]:
        """Evaluate the current policy.

        Returns:
            Mean evaluation return
        """
        logging.info("Evaluating")
        eval_obs, _ = eval_env.reset()
        eval_metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(CFG.max_rl_steps):
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
        evaluated_steps = CFG.max_rl_steps * self.args.num_eval_envs
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
        env: ManiSkillVectorEnv,
        eval_env: Optional[ManiSkillVectorEnv],
        start_time: float,
    ) -> int:
        """Full PPO training loop (Stage 1) - exactly matches ppo.py::train.

        Trains PPO until reward threshold is reached or max iterations completed.
        Saves checkpoints and tracks best model.

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
                                    f"PPO reached reward threshold! Avg reward: {avg_reward:.3f} >= {self.args.reward_threshold}"
                                )
                                logging.info("Proceeding to DQN training stage...")
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
            b_obs = self.obs.reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + env.single_action_space.shape)
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

    def train_dqn(
        self,
        env: ManiSkillVectorEnv,
        eval_env: Optional[ManiSkillVectorEnv],
        start_global_step: int,
        start_time: float,
    ) -> int:
        """Full DQN training loop (Stage 2).

        Loads best PPO checkpoint, then trains Q-network to decide when to call planner.
        Tracks planner calls to prevent replay buffer contamination.

        Returns:
            Final global_step
        """
        logging.info("=" * 50)
        logging.info("STAGE 2: DQN TRAINING")
        logging.info("=" * 50)

        # Load best PPO checkpoint
        if self.best_model_path is not None and self.best_model_path.exists():
            logging.info(f"Loading best PPO checkpoint from {self.best_model_path}")
            self.load(self.best_model_path)
            logging.info(f"Best PPO model loaded (reward: {self.best_eval_reward:.3f})")
        else:
            logging.warning("No best PPO checkpoint found, using current policy")

        # Initialize planner call tracking
        self.planner_called_mask = torch.zeros(
            self.args.num_envs, dtype=torch.bool, device=self.device
        )
        self.planner_called_step = torch.zeros(
            self.args.num_envs, dtype=torch.int, device=self.device
        )

        # Initialize
        obs, _ = env.reset(seed=self.args.seed)
        call_planner_step_obs = obs.clone()
        global_step = start_global_step

        action_space_low, action_space_high = torch.from_numpy(
            env.single_action_space.low  # type: ignore
        ).to(self.device), torch.from_numpy(
            env.single_action_space.high
        ).to(  # type: ignore
            self.device
        )

        # DQN-specific parameters
        factored_num_steps = (
            self.args.total_timesteps - start_global_step
        ) // self.args.num_envs
        roll_out_step = 0  # control probability of planner call
        logging.info(f"DQN will run for {factored_num_steps} batched steps")

        # Main DQN training loop
        for global_step in range(
            start_global_step, self.args.total_timesteps, self.args.num_envs
        ):
            logging.info(f"Global Step: {global_step}")

            # Evaluation
            batched_steps = (global_step - start_global_step) // self.args.num_envs
            if batched_steps % self.args.eval_freq_q == 1 and eval_env is not None:
                evaluated_steps, num_episodes, eval_metrics = self.evaluate(
                    eval_env, global_step
                )
                logging.info(
                    f"Evaluated {evaluated_steps} steps resulting in {num_episodes} episodes"
                )
                if self.args.evaluate:
                    return global_step

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
                                / f"runs/{CFG.exp_name}/best_ppoq_ckpt.pt"
                            )
                            base_path = (
                                Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                            )
                            base_path.mkdir(parents=True, exist_ok=True)
                            self.save(self.best_model_path)
                            logging.info(
                                f"New best PPOQ model! Reward: {mean_return:.3f}, saved to {self.best_model_path}"
                            )

            # Save periodic checkpoint
            if self.args.save_model and batched_steps % self.args.eval_freq_q == 1:
                model_path = (
                    Path(CFG.rl_policy_save_dir)
                    / f"runs/{CFG.exp_name}/ppoq_ckpt_{global_step}.pt"
                )
                base_path = Path(CFG.rl_policy_save_dir) / "runs" / CFG.exp_name
                base_path.mkdir(parents=True, exist_ok=True)
                self.save(model_path)
                logging.info(f"DQN checkpoint saved to {model_path}")

            # Collect rollout data
            roll_out_step += 1
            with torch.no_grad():
                normalized_obs = self.normalize_obs(obs)

                # Epsilon-greedy for Q-network
                epsilon = self.linear_schedule(
                    self.args.epsilon_start,
                    self.args.epsilon_end,
                    self.args.epsilon_decay_steps,
                    global_step - start_global_step,
                )
                rnd_action_mask = (
                    torch.rand(self.args.num_envs, device=self.device) < epsilon
                )
                p = roll_out_step / CFG.max_rl_steps * self.args.rnd_action_dist_biase
                q_actions = torch.bernoulli(
                    torch.full((self.args.num_envs,), p, device=self.device)
                ).long()
                q_values = self.q_network(normalized_obs[~rnd_action_mask])
                q_actions_network = torch.argmax(q_values, dim=1)
                q_actions[~rnd_action_mask] = q_actions_network

                # Track planner calls
                planner_called_this_step = q_actions == 1
                newly_called = planner_called_this_step & (~self.planner_called_mask)
                self.planner_called_mask = (
                    self.planner_called_mask | planner_called_this_step
                )
                # ONLY update call_planner_step_obs for newly called envs
                call_planner_step_obs[newly_called] = normalized_obs[newly_called]
                self.planner_called_step[newly_called] = roll_out_step

                # Get PPO actions
                action, _, _, _ = self.agent.get_action_and_value(normalized_obs)

                # Get frozen/static actions
                frozen_actions = get_frozen_action(
                    action,
                    action_space_low,
                    action_space_high,
                    normalize_action=True,
                    control_mode=CFG.control_mode,
                )

                # Select based on Q-network (0=PPO, 1=static/planner)
                final_action = torch.where(
                    self.planner_called_mask.unsqueeze(-1) == 0,
                    action,
                    frozen_actions,
                )

                # Execute action
                next_obs, reward, terminations, truncations, infos = env.step(
                    self.clip_action(final_action)
                )
                normalized_obs_next = self.normalize_obs(next_obs)

                # NOTE: For all env planner called, we do not store in replay buffer
                # here, since the reward is fake (true reward is achieved with final observation)
                replay_obs = normalized_obs[~self.planner_called_mask]
                replay_next_obs = normalized_obs_next[~self.planner_called_mask]
                replay_actions = q_actions[~self.planner_called_mask]
                replay_rewards = reward.view(-1)[~self.planner_called_mask]
                replay_dones = torch.logical_or(terminations, truncations).to(
                    torch.float32
                )[~self.planner_called_mask]
                if len(replay_obs) > 0:
                    self.replay_buffer.add(
                        replay_obs,
                        replay_next_obs,
                        replay_actions,
                        replay_rewards,
                        replay_dones.bool(),
                    )

                obs = next_obs
                if "final_info" in infos:
                    roll_out_step = 0
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    assert (
                        done_mask.all()
                    ), "All envs should be done when final_info is present"
                    for k, v in final_info["episode"].items():
                        self.logger.add_scalar(
                            f"train/{k}", v[done_mask].float().mean(), global_step
                        )
                    self.logger.add_scalar(
                        "q_charts/planner_called_ratio",
                        self.planner_called_mask.float().mean().item(),
                        global_step,
                    )
                    self.logger.add_scalar(
                        "q_charts/planner_called_count_mean",
                        self.planner_called_step.float().mean().item(),
                        global_step,
                    )
                    self.logger.add_scalar(
                        "q_charts/planner_called_count_var",
                        self.planner_called_step.float().var().item(),
                        global_step,
                    )
                    # Add the call planner step transitions to replay buffer with true rewards
                    if self.planner_called_mask.any():
                        planner_called_obs = call_planner_step_obs[
                            self.planner_called_mask
                        ]
                        planner_called_next_obs = normalized_obs_next[
                            self.planner_called_mask
                        ]
                        planner_called_actions = torch.ones(
                            planner_called_obs.shape[0],
                            dtype=torch.long,
                            device=self.device,
                        )  # Action=1 for call_planner
                        planner_called_rewards = reward.view(-1)[
                            self.planner_called_mask
                        ]
                        planner_called_dones = torch.ones(
                            planner_called_obs.shape[0],
                            dtype=torch.bool,
                            device=self.device,
                        )  # Done=True for final step
                        self.replay_buffer.add(
                            planner_called_obs,
                            planner_called_next_obs,
                            planner_called_actions,
                            planner_called_rewards,
                            planner_called_dones,
                        )
                    # Reset planner call mask for envs that are done
                    self.planner_called_mask[done_mask] = False
                    call_planner_step_obs[done_mask] = next_obs[done_mask]
                    self.planner_called_step[done_mask] = 0

            # Update Q-network if enough data collected
            q_loss_value = 0.0
            q_update_time = time.time()

            if len(self.replay_buffer) >= self.args.q_learning_starts:
                if global_step % self.args.q_train_frequency == 0:
                    (
                        rb_obs,
                        rb_next_obs,
                        rb_actions,
                        rb_rewards,
                        rb_dones,
                    ) = self.replay_buffer.sample(self.args.q_batch_size)

                    with torch.no_grad():
                        target_max, _ = self.target_q_network(rb_next_obs).max(dim=1)
                        td_target = (
                            rb_rewards
                            + self.args.q_gamma * target_max * (~rb_dones).float()
                        )

                    old_val = (
                        self.q_network(rb_obs)
                        .gather(1, rb_actions.unsqueeze(1))
                        .squeeze()
                    )
                    q_loss = F.mse_loss(td_target, old_val)
                    q_loss_value = q_loss.item()

                    self.q_optimizer.zero_grad()
                    q_loss.backward()  # type: ignore[no-untyped-call]
                    self.q_optimizer.step()

                # Update target network
                if global_step % self.args.q_target_network_frequency == 0:
                    for target_param, param in zip(
                        self.target_q_network.parameters(),
                        self.q_network.parameters(),
                    ):
                        target_param.data.copy_(
                            self.args.q_tau * param.data
                            + (1.0 - self.args.q_tau) * target_param.data
                        )
            q_update_time = time.time() - q_update_time

            # Logging
            self.logger.add_scalar("q_losses/td_loss", q_loss_value, global_step)
            self.logger.add_scalar(
                "q_charts/buffer_size", len(self.replay_buffer), global_step
            )
            self.logger.add_scalar("q_charts/epsilon", epsilon, global_step)
            self.logger.add_scalar("time/q_update_time", q_update_time, global_step)

            elapsed_time = time.time() - start_time
            self.logger.add_scalar(
                "charts/SPS", int(global_step / elapsed_time), global_step
            )

        logging.info("DQN training completed")
        return global_step

    def train(
        self,
        env: ManiSkillVectorEnv,
        eval_env: Optional[ManiSkillVectorEnv] = None,
        train_data: Optional[object] = None,
    ) -> None:
        """Two-stage training: PPO followed by DQN.

        Stage 1: Train PPO until reward threshold or max iterations
        Stage 2: Load best PPO, train DQN to learn when to call planner
        """
        del train_data  # Unused

        # Initialize observation normalization variables
        obs_shape = env.single_observation_space.shape
        if obs_shape is None:
            obs_shape = ()
        self.curr_obs_mean = torch.zeros(obs_shape, device=self.device)
        self.curr_obs_std = torch.ones(obs_shape, device=self.device)

        start_time = time.time()
        # HACK: Debugging q learning here
        # self.q_learning_enabled = True
        # global_step = self.train_dqn(env, eval_env, 0, start_time)

        # ========== STAGE 1: PPO TRAINING ==========
        if self.args.resume_ppo == "":
            global_step = self.train_ppo(env, eval_env, start_time)
        else:
            resume_path = Path(self.args.resume_ppo)
            self.load(resume_path)
            logging.info(f"Resumed PPO model from {self.args.resume_ppo}")
            _, _, eval_metrics = self.evaluate(eval_env, 0)  # Evaluate loaded model
            mean_return = 0.0
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/{k}", mean, 0)
                logging.info(f"eval_{k}_mean={mean}")

                # Track best model and check threshold
                if k == "success_once":
                    mean_return = mean.item()
            global_step = 0
            self.recent_rewards = [mean_return] * self.args.reward_continuing_episodes

        # If we're in evaluation-only mode, stop here
        if self.args.evaluate:
            self.logger.close()
            return

        # ========== STAGE 2: DQN TRAINING ==========
        # Only proceed to DQN if PPO reached the threshold
        if len(self.recent_rewards) >= self.args.reward_continuing_episodes:
            avg_reward = np.mean(
                self.recent_rewards[-self.args.reward_continuing_episodes :]
            )
            if avg_reward >= self.args.reward_threshold:
                logging.info("PPO training successful, proceeding to DQN stage...")
                self.q_learning_enabled = True
                global_step = self.train_dqn(env, eval_env, global_step, start_time)
            else:
                logging.info(
                    f"PPO did not reach threshold (avg reward: {avg_reward:.3f} < {self.args.reward_threshold})"
                )
                logging.info("Skipping DQN training stage")
        else:
            logging.warning("Not enough evaluation data, skipping DQN training stage")

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
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
            },
            path.as_posix(),
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path.as_posix(), map_location=self.device)
        self.agent.load_state_dict(ckpt["agent"])
        if "q_network" in ckpt:
            self.q_network.load_state_dict(ckpt["q_network"])
            self.target_q_network.load_state_dict(ckpt["target_q_network"])
        self._trained = True
