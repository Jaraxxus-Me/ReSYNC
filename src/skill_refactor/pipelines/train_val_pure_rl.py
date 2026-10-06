"""Train and evaluate a pure RL policy on a system."""

import logging
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # type: ignore # pylint: disable=unused-import
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode  # type: ignore
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor.approaches.pure_rl import PureRLApproach
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.approaches.rl_policies.ppo import PPOPolicy
from skill_refactor.approaches.rl_policies.sac import SACPolicy
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.pipelines.eval_episode import run_evaluation_episode
from skill_refactor.pipelines.metric import Metrics
from skill_refactor.settings import CFG
from skill_refactor.utils.gpu_utils import set_torch_seed


def create_rl_policy(policy_name: str, seed: int) -> Policy:
    """Create customized SAC policy."""
    if policy_name == "SAC":
        return SACPolicy(seed=seed)
    assert policy_name == "PPO", f"Unsupported policy: {policy_name}"
    return PPOPolicy(seed=seed)


def train_and_evaluate_pure_rl(
    system: BaseRLTAMPSystem,
    approach_name: str,
) -> Metrics:
    """Train and evaluate a pure RL policy on a system."""
    logging.info(f"\nInitializing pure RL baseline training for {system.name}...")
    seed = CFG.seed
    set_torch_seed(seed)

    policy = create_rl_policy(CFG.rl_algo, seed)

    # Create mani-skill environment
    logging.info(f"Creating ManiSkill environment for {system.name}...")

    if policy.args.control_mode is not None:
        system.env_kwargs["control_mode"] = policy.args.control_mode

    # Base env
    pure_rl_env_train = gym.make(
        system.env_name,
        num_envs=CFG.num_envs,
        reconfiguration_freq=None,
        **system.env_kwargs,
    )
    pure_rl_env_eval = gym.make(
        system.env_name,
        num_envs=CFG.num_eval_envs,
        reconfiguration_freq=1,
        human_render_camera_configs={"shader_pack": "default"},
        **system.env_kwargs,
    )

    # Record
    if CFG.record_training and policy.args.save_train_video_freq is not None:
        video_folder = Path(f"videos/{system.name}_{approach_name}_train")
        video_folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving eval trajectories/videos to {video_folder}")

        save_video_trigger = (
            lambda x: (x // policy.args.num_steps) % policy.args.save_train_video_freq
            == 0
        )

        envs = RecordEpisode(
            pure_rl_env_train,
            output_dir=video_folder,
            save_trajectory=False,
            save_video_trigger=save_video_trigger,
            max_steps_per_video=CFG.max_env_steps,
            video_fps=30,
        )
    else:
        envs = pure_rl_env_train

    eval_output_dir = Path(f"videos/{system.name}_{approach_name}_eval")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_envs = RecordEpisode(
        pure_rl_env_eval,
        output_dir=eval_output_dir,
        save_trajectory=False,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=CFG.max_env_steps,
        video_fps=30,
    )

    # ManiSkill VectorEnv wrapper
    envs = ManiSkillVectorEnv(
        envs, policy.args.num_envs, ignore_terminations=True, record_metrics=True
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        policy.args.num_eval_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    # Initialize policy
    policy.initialize(envs)  # type: ignore

    # Train policy if needed
    start_time = time.time()
    if policy.requires_training:
        logging.info("\nTraining pure RL policy...")
        policy.train(envs, eval_envs, train_data=None)  # type: ignore
    else:
        approach_name = approach_name.replace("_Loaded", "")  # type: ignore
        pre_trained_policy_path = (
            Path(CFG.rl_policy_save_dir) / f"{system.name}_{approach_name}.zip"
        )
        assert (
            pre_trained_policy_path.exists()
        ), f"Pre-trained policy not found at {pre_trained_policy_path}"
        logging.info(f"Using pre-trained policy from {pre_trained_policy_path}")
        policy.load(pre_trained_policy_path)

    training_time = time.time() - start_time

    approach = PureRLApproach(system, policy, seed)

    # Run evaluation
    logging.info(f"\nEvaluating pure RL policy on {system.name}...")
    start_time = time.time()
    # Run evaluation
    logging.info(f"\nEvaluating pure {approach_name} on {system.name}...")
    rewards, lengths, successes = run_evaluation_episode(
        system,
        approach,
        total_episodes=CFG.num_eval_episodes,
    )

    total_time = time.time() - start_time
    return Metrics(
        success_rate=float(sum(successes) / len(successes)),
        avg_episode_length=float(np.mean(lengths)),
        avg_reward=float(np.mean(rewards)),
        training_time=training_time,
        total_time=total_time,
    )
