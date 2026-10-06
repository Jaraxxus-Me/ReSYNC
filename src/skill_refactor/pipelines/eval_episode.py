"""Evaluate a single episode of a skill learning + planning approach."""

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv as ManiskillBaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode  # type: ignore
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor.approaches.base import BaseApproach
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo, MultiEnvWrapper
from skill_refactor.settings import CFG


def run_evaluation_episode(
    system: BaseRLTAMPSystem,
    approach: BaseApproach,
    total_episodes: int,
) -> tuple[List, List, List]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if CFG.render and can_render:
        video_folder = Path(f"videos/{system.name}_{approach.get_name()}_test")
        if isinstance(system.env.unwrapped, ManiskillBaseEnv):
            envs = RecordEpisode(
                system.env.unwrapped,
                output_dir=video_folder,
                save_trajectory=False,
                save_video=True,
                trajectory_name="trajectory",
                max_steps_per_video=CFG.max_env_steps,
                video_fps=30,
            )
        else:
            assert isinstance(system.env, MultiEnvWrapper)
            envs = MultiEnvRecordVideo(
                system.env,
                video_folder=video_folder.as_posix(),
                episode_trigger=lambda episode_id: True,  # Save all evaluation episodes
            )
    else:
        envs = system.env

    if isinstance(system.env.unwrapped, ManiskillBaseEnv):
        envs = ManiSkillVectorEnv(
            envs, CFG.num_envs, ignore_terminations=True, record_metrics=True
        )
    rewards = []
    lengths = []
    successes = []

    for episode in range(total_episodes):
        if episode != 1:
            continue
        logging.info(f"\nEvaluation Episode {episode + 1}/{total_episodes}")
        obs, info = envs.reset(seed=episode)
        step_result = approach.reset(obs, info)

        total_reward = torch.tensor(
            [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
        )
        step_count = torch.tensor(
            [0] * CFG.num_envs, dtype=torch.int, device=envs.device
        )
        success = torch.tensor(
            [False] * CFG.num_envs, dtype=torch.bool, device=envs.device
        )

        # Rest of steps
        for _step in range(CFG.max_env_steps):
            obs, reward, _, _, info = envs.step(step_result.action)
            step_result = approach.step(obs, total_reward, False, False, info)
            total_reward[~success] += reward[~success]
            step_success = info["success"].to(torch.bool)
            success |= step_success
            step_count[~success] += 1

        rewards.append(total_reward.mean().cpu().numpy())
        average_length = step_count.sum() / step_count.shape[0]
        lengths.append(average_length.cpu().numpy())
        average_success = success.sum() / success.shape[0]
        successes.append(average_success.cpu().numpy())

        logging.info(f"Current Success Rate: {sum(successes)/(episode+1):.2%}")
        logging.info(f"Current Avg Episode Length: {np.mean(lengths):.2f}")
        logging.info(f"Current Avg Reward: {np.mean(rewards):.2f}")

    envs.close()
    return rewards, lengths, successes
