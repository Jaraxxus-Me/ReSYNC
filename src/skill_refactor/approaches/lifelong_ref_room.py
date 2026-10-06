"""Lifelong refactorization approach adapted for stochastic ManiSkill environments.

This variant removes the replay mechanism from data collection, as ManiSkill
environments are non-deterministic and replaying actions does not reproduce the same
states.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import Tensor

from skill_refactor.approaches.lifelong_ref_mani import LifelongRefManiApproach
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    ApproachStepResult,
    GroundOperator,
    LiftedOperator,
    LowLevelTrajectory,
    PlannerDataset,
)
from skill_refactor.utils.ttmp import TaskThenMotionPlanningFailure


class LifelongRefRoomApproach(LifelongRefManiApproach):
    """Lifelong refactorization approach for stochastic ManiSkill environments.

    This class inherits from LifelongRefApproach but redefines the planner data
    collection methods to work with non-deterministic environments where action replay
    does not produce identical states.
    """

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "lifelong_ref_room"

    def collect_planner_data(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        provided_init_states: List[Tensor],
        real_env_scenario_name: str = "sc1_pre_n1_sc2_pre_n1",
        dataset_save_path: Optional[Path] = None,
        save_frequency: int = 5,
    ) -> PlannerDataset:
        """Collect Planner training data from stochastic ManiSkill environment.

        Uses batched collision state tracking for both real and dreamed scenarios.
        All collection is done via the shared `_collect_trajectories_with_sampler` method.

        Supports incremental saving and automatic resumption from partial datasets.

        Args:
            envs: The vectorized environment
            policy: The policy to use for RL skill execution
            provided_init_states: Initial states to create the sampler from
            real_env_scenario_name: The scenario name for real environment trajectories
            dataset_save_path: Path to save dataset incrementally (if None, no saving)
            save_frequency: Save dataset every N episodes (default: 5)

        Returns:
            PlannerDataset with trajectories from both real environment and dreaming.
        """
        logging.info("Collecting training data with planner (ManiSkill variant)\n")

        # ===============================
        # Load existing dataset if available
        # ===============================
        scenario_trajectories: Dict[str, List[LowLevelTrajectory]] = {}

        if dataset_save_path is not None and dataset_save_path.exists():
            logging.info("=" * 80)
            logging.info("RESUMING FROM EXISTING DATASET")
            logging.info("=" * 80)

            # Load all scenario subdirectories
            for scenario_dir in dataset_save_path.iterdir():
                if scenario_dir.is_dir():
                    scenario_file = scenario_dir / "trajectories.pkl"
                    if scenario_file.exists():
                        try:
                            import pickle

                            with open(scenario_file, "rb") as f:
                                scenario_trajs = pickle.load(f)
                            scenario_trajectories[scenario_dir.name] = scenario_trajs
                            logging.info(
                                f"Loaded {len(scenario_trajs)} trajectories from {scenario_dir.name}"
                            )
                        except Exception as e:
                            logging.warning(f"Failed to load {scenario_file}: {e}")

            total_existing = sum(len(trajs) for trajs in scenario_trajectories.values())
            logging.info(f"Total existing trajectories: {total_existing}")
            logging.info(f"Scenarios loaded: {list(scenario_trajectories.keys())}")
            logging.info("%s\n", "=" * 80)

        # Helper function to save a scenario's trajectories
        def save_scenario_trajectories(
            scenario_name: str, trajectories: List[LowLevelTrajectory]
        ) -> None:
            """Save trajectories for a specific scenario."""
            if dataset_save_path is None or len(trajectories) == 0:
                return

            scenario_dir = dataset_save_path / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_file = scenario_dir / "trajectories.pkl"

            import pickle

            with open(scenario_file, "wb") as f:
                pickle.dump(trajectories, f)

            logging.info(f"Saved {len(trajectories)} trajectories to {scenario_file}")

        # ===============================
        # Phase 1: Collect Real Environment Trajectories
        # ===============================
        logging.info("=" * 80)
        logging.info("PHASE 1: Real Environment Data Collection")
        logging.info("%s\n", "=" * 80)

        # Calculate total episode count from existing data
        all_episode_count = sum(len(trajs) for trajs in scenario_trajectories.values())

        # Create sampler for real environment
        relative_state_sampler, basic_node, pre_post_list = (
            self.get_init_state_sampler_from_provided(provided_init_states, envs.device)
        )

        def real_init_sampler() -> Tensor:
            env_reset_init, _ = envs.reset()
            return self.sample_init_states_from_sampler(
                envs, env_reset_init, relative_state_sampler, basic_node, pre_post_list
            )

        # Collect real trajectories (or use existing)
        if real_env_scenario_name in scenario_trajectories:
            real_env_scenario = real_env_scenario_name
            real_trajectories = scenario_trajectories[real_env_scenario]
            logging.info(
                f"Using {len(real_trajectories)} existing trajectories for '{real_env_scenario}'"
            )
            num_remaining = max(
                0,
                CFG.num_train_episodes_planner_before_dreaming - len(real_trajectories),
            )
        else:
            real_env_scenario = real_env_scenario_name
            num_remaining = CFG.num_train_episodes_planner_before_dreaming
        real_trajectories, _, all_episode_count = (
            self._collect_trajectories_with_sampler(
                envs,
                policy,
                real_init_sampler,
                all_episode_count,
                num_episodes=num_remaining,
                scenario_name=real_env_scenario,
                dataset_save_path=dataset_save_path,
                save_frequency=save_frequency,
                existing_trajectories=scenario_trajectories.get(real_env_scenario, []),
            )
        )
        scenario_trajectories[real_env_scenario] = real_trajectories
        save_scenario_trajectories(real_env_scenario, real_trajectories)

        # Extract relative states from real trajectories
        relative_states: Dict[str, List[Tensor]] = {}
        rl_operator_names: List[str] = []

        for scenario in list(self.learning_info.keys()):
            relative_states[f"sc{scenario}_pre"] = []
            relative_states[f"sc{scenario}_post"] = []
            rl_op_name = (
                self.learning_info[scenario].get("tgt_skill", "").split("(")[0].strip()
            )
            rl_operator_names.append(rl_op_name)

        for traj in real_trajectories:
            self._extract_relative_states_from_trajectory_single_env(
                traj.states,
                [act.op for act in traj.actions],
                relative_states,
                rl_operator_names,
            )

        logging.info(
            f"Collected {len(real_trajectories)} real environment trajectories"
        )
        logging.info(
            f"Extracted relative states for {len(relative_states)} scenario keys"
        )

        # ===============================
        # Phase 2: Dreaming - Compositional Data Collection
        # ===============================
        logging.info("\n%s", "=" * 80)
        logging.info("PHASE 2: Dreaming Data Collection")
        logging.info("%s\n", "=" * 80)

        trajectories = list(real_trajectories)

        # Build all scenario combinations
        num_obstructions = len(CFG.scenario.split(","))
        assert self.potential_comp_nodes, "No potential composition nodes available"
        num_comp_nodes = len(self.potential_comp_nodes)
        total_scenarios = (2 * num_comp_nodes) ** num_obstructions
        desired_scenario_number = CFG.num_train_episodes_planner // total_scenarios

        import itertools

        all_scenario_combos: List[Tuple[Tuple[int, int, int], ...]] = []
        all_scenario_names: List[str] = []
        obstruction_choices = []

        for scenario in list(self.learning_info.keys()):
            obstruction_choice = []
            for pre_post in [0, 1]:  # 0=pre, 1=post
                for node_id in self.potential_comp_nodes:
                    obstruction_choice.append((scenario, pre_post, node_id))
            obstruction_choices.append(obstruction_choice)

        for combo in itertools.product(*obstruction_choices):
            all_scenario_combos.append(combo)
            scenario_name_parts = []
            for scenario, pre_post, node_id in combo:
                pre_post_str = "pre" if pre_post == 0 else "post"
                scenario_name_parts.append(f"sc{scenario}_{pre_post_str}_n{node_id}")
            all_scenario_names.append("_".join(scenario_name_parts))

        # Mark known invalid and real scenarios
        scenario_proven_invalid = [False] * total_scenarios
        known_invalid_scenarios = (
            self.learning_info[self.curr_learning_phase]
            .get("known_invalid_sc", "")
            .split(";")
        )
        for invalid_sc in known_invalid_scenarios:
            if invalid_sc and invalid_sc in all_scenario_names:
                idx = all_scenario_names.index(invalid_sc)
                scenario_proven_invalid[idx] = True
                logging.info(f"Marking known invalid scenario '{invalid_sc}'")

        # Mark real scenario as already collected
        assert (
            real_env_scenario_name in all_scenario_names
        ), f"real_env_scenario_name '{real_env_scenario_name}' not in valid scenarios"
        real_env_scenario_idx = all_scenario_names.index(real_env_scenario_name)

        logging.info(
            f"Target: {desired_scenario_number} trajectories per scenario "
            f"({total_scenarios} total scenarios = (2*{num_comp_nodes})^{num_obstructions})"
        )
        logging.info(
            f"Real environment scenario '{real_env_scenario_name}' already collected with {len(real_trajectories)} trajectories"
        )

        # Generate relative state sampler for dreaming
        relative_state_sampler_dream = self._gen_relative_dist_sampler(relative_states)

        # Collect each dreamed scenario
        for scenario_idx, (scenario_combo, scenario_name) in enumerate(
            zip(all_scenario_combos, all_scenario_names)
        ):
            # Skip real scenario (already collected) and known invalid scenarios
            if (
                scenario_idx == real_env_scenario_idx
                or scenario_proven_invalid[scenario_idx]
            ):
                continue

            logging.info(
                f"\nProcessing dreamed scenario {scenario_idx}: {scenario_name}"
            )

            # Create sampler for this specific scenario
            def dreamed_init_sampler(
                combo=scenario_combo,
            ) -> Tensor:  # Capture combo in closure
                init_obs, _ = envs.reset()

                # Build pre/post list
                pre_post_list: List[Tuple[int, ...]] = [
                    (sc_id,) for sc_id in list(self.learning_info.keys())
                ]
                basic_node = (
                    torch.zeros(3, dtype=torch.long, device=init_obs.device) - 100
                )

                for encounter_id, (sc_id, pre_post, node_id) in enumerate(combo):
                    pre_post_list[encounter_id] = pre_post_list[encounter_id] + (
                        pre_post,
                    )
                    basic_node[sc_id - 1] = node_id

                dreamed_rel_state_data = relative_state_sampler_dream(pre_post_list).to(
                    init_obs.device
                )
                dreamed_rel_state = torch.cat(
                    [basic_node.unsqueeze(1), dreamed_rel_state_data], dim=1
                )
                dreamed_rel_state = dreamed_rel_state.unsqueeze(0)
                return self.system.relative_state_to_full_state(
                    init_obs, dreamed_rel_state
                )

            # Validate dreamed state first
            try:
                test_state = dreamed_init_sampler()
                _, _ = envs.reset(options={"init_state": test_state})
            except ValueError as e:
                logging.error(f"Dreamed state for {scenario_name} has collisions: {e}")
                scenario_proven_invalid[scenario_idx] = True
                logging.warning(f"Marking scenario {scenario_idx} as INVALID")
                continue

            # Check if scenario already has enough trajectories
            existing_trajs = scenario_trajectories.get(scenario_name, [])
            existing_count = len(existing_trajs)
            if existing_count >= desired_scenario_number:
                logging.info(
                    f"Scenario '{scenario_name}' already has {existing_count} trajectories (target: {desired_scenario_number}). Skipping."
                )
                # Add existing trajectories to main list
                trajectories.extend(existing_trajs)
                continue

            # Collect remaining trajectories for this scenario
            remaining_needed = desired_scenario_number - existing_count
            logging.info(
                f"Scenario '{scenario_name}': collecting {remaining_needed} more trajectories "
                f"({existing_count} already collected)"
            )

            # pre the latest scenario
            allow_rl = scenario_combo[-1][1] == 0
            scenario_trajs, is_valid, all_episode_count = (
                self._collect_trajectories_with_sampler(
                    envs,
                    policy,
                    dreamed_init_sampler,
                    all_episode_count,
                    num_episodes=remaining_needed,
                    scenario_name=scenario_name,
                    allow_rl=allow_rl,
                    max_attempts=CFG.max_consecutive_failures,
                    dataset_save_path=dataset_save_path,
                    save_frequency=save_frequency,
                    existing_trajectories=scenario_trajectories.get(scenario_name, []),
                )
            )

            if is_valid:
                # Combine existing and new trajectories
                all_scenario_trajs = existing_trajs + scenario_trajs
                scenario_trajectories[scenario_name] = all_scenario_trajs
                # Add ALL trajectories (existing + new) to main list
                # Note: existing ones for this scenario were not yet added to trajectories
                trajectories.extend(all_scenario_trajs)

                # Save updated scenario
                save_scenario_trajectories(scenario_name, all_scenario_trajs)

                logging.info(
                    f"Successfully collected {len(scenario_trajs)} new trajectories for scenario '{scenario_name}' "
                    f"(total: {len(all_scenario_trajs)})"
                )
            else:
                scenario_proven_invalid[scenario_idx] = True
                logging.warning(
                    f"Scenario '{scenario_name}' marked as INVALID after max attempts"
                )

        # Final verification
        logging.info("\n%s", "=" * 80)
        logging.info("FINAL SUMMARY")
        logging.info("%s", "=" * 80)
        logging.info(f"Total trajectories collected: {len(trajectories)}")
        logging.info(
            f"Real environment: {len(real_trajectories)} trajectories (scenario: {real_env_scenario_name})"
        )
        logging.info(
            f"Dreamed scenarios: {len(trajectories) - len(real_trajectories)} trajectories"
        )
        logging.info(
            f"Invalid scenarios: {sum(scenario_proven_invalid)}/{total_scenarios}"
        )
        logging.info("%s\n", "=" * 80)

        # Return dataset
        training_scenario_indices = [traj.train_scenario_idx for traj in trajectories]
        return PlannerDataset(
            _trajectories=trajectories,
            _training_scenario_idx=training_scenario_indices,
        )

    def is_colliding(
        self,
        obs: Tensor,
    ) -> Tensor:
        """Check if the agent is colliding in the given observations.

        Args:
            obs: Observations tensor
        Returns:
            Boolean tensor indicating collision status for each environment
        """
        obj2_pos = obs[:, 52:59]
        can_pos = obs[:, 68:75]

        xy_dist = torch.norm(obj2_pos[:, :2] - can_pos[:, :2], dim=-1)
        z_diff = torch.abs(obj2_pos[:, 2] - can_pos[:, 2])

        is_colliding = (xy_dist < 0.05) & (z_diff < 0.075)
        return is_colliding

    def _execute_single_batch_episode(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        obs: Tensor,
        max_rl_steps: int,
        rl_static_steps: int,
        allow_rl: bool,
        scenario_info: dict,
        all_episode_count: int,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single batched episode and return collected data.

        Args:
            envs: Vectorized environment
            policy: RL policy
            obs: Initial observation
            max_rl_steps: Maximum RL execution steps
            rl_static_steps: Stabilization steps after RL
            scenario_info: Scenario configuration

        Returns:
            Dictionary with episode data if successful, None if episode failed to initialize
        """
        # Try to create initial plan
        try:
            approach_result = self.reset(obs, {})
        except TaskThenMotionPlanningFailure:
            return None

        if approach_result.op is None:
            return None

        if CFG.render_planner_collection_img != "":
            render_planner_collection_img = Path(f"{CFG.render_planner_collection_img}")
            render_planner_collection_img.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as iio

            batched_img = envs.render()
            for b in range(envs.num_envs):
                img = batched_img[b]
                iio.imwrite(
                    os.path.join(
                        CFG.render_planner_collection_img,
                        f"epi_{all_episode_count}_env_{b}_env_init.png",
                    ),
                    img.cpu().numpy(),
                )

        curr_op_name = approach_result.op.parent.name
        curr_op_param_names = [obj.name for obj in approach_result.op.parameters]

        # Data collection buffers
        episode_states: List[List[Tensor]] = [[] for _ in range(envs.num_envs)]
        episode_actions: List[List[Tensor]] = [[] for _ in range(envs.num_envs)]
        episode_operators: List[List[Optional[GroundOperator]]] = [
            [] for _ in range(envs.num_envs)
        ]
        colliding_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)
        success_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)

        # Store initial state
        for b in range(envs.num_envs):
            episode_states[b].append(obs[b].clone())

        colliding_envs = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)
        last_actions = torch.zeros_like(approach_result.action)

        # Phase 1: Execute plan and track collisions
        # NOTE: Here we do the max_env_steps, instead of using max_skill_steps.
        for _ in range(CFG.max_env_steps):
            action = approach_result.action
            execute_action = action.clone()
            # For colliding envs, use last non-colliding action
            execute_action[colliding_mask] = last_actions[colliding_mask]
            obs, _, _, _, infos = envs.step(execute_action)

            # Check for collisions during failed operator execution
            if (
                self.is_colliding(obs[0:1]).any()
                and (curr_op_name == self.failed_op)
                and all(
                    obj_name == param_name
                    for obj_name, param_name in zip(
                        self.failed_objects,
                        curr_op_param_names,
                    )
                )
            ):
                # NOTE: We only check collision for the first env as a proxy
                # This at least ensures that if one env is colliding, we stop all colliding envs
                colliding_envs = self.is_colliding(obs).to(torch.bool)
                colliding_mask |= colliding_envs
                last_actions[colliding_envs, -1] = action[colliding_envs, -1]
                break

            # Store data
            for b in range(envs.num_envs):
                episode_states[b].append(obs[b].clone())
                episode_actions[b].append(action[b].clone())
                episode_operators[b].append(approach_result.op)

            # Check for success
            if infos["success"].any():
                success_mask |= infos["success"].to(torch.bool)

            # Break if all succeeded or collided
            if torch.all(success_mask | colliding_mask | self.exhausted):
                break

            approach_result = self.step(obs, 0.0, False, False, {})
            if approach_result.op is None:
                break
            curr_op_name = approach_result.op.parent.name
            curr_op_param_names = [obj.name for obj in approach_result.op.parameters]

        # Phase 2: Execute RL for collided environments
        environments_to_process = colliding_mask.clone()
        if environments_to_process.any() and allow_rl:
            logging.info(
                f"Executing RL skill for {environments_to_process.sum().item()} collided environments"
            )

            if CFG.render_planner_collection_img != "":
                import imageio.v2 as iio

                batched_img = envs.render()
                for b in range(envs.num_envs):
                    img = batched_img[b]
                    env_collided = environments_to_process[b]
                    iio.imwrite(
                        os.path.join(
                            CFG.render_planner_collection_img,
                            f"epi_{all_episode_count}_env_{b}_env_pre_rl_collide_{env_collided}.png",
                        ),
                        img.cpu().numpy(),
                    )

            obs_after_rl, action = self._execute_rl_skill_for_batch(
                envs,
                policy,
                obs,
                action,
                episode_states,
                episode_actions,
                episode_operators,
                success_mask,
                max_rl_steps,
                rl_static_steps,
                scenario_info,
            )

            # Phase 3: Replan and continue
            try:
                approach_result = self.reset(obs_after_rl, {})
            except TaskThenMotionPlanningFailure:
                pass  # Continue with what we have
            else:
                for b in range(envs.num_envs):
                    for k in range(
                        len(episode_operators[b]) - 1,
                        len(episode_operators[b]) - rl_static_steps - 1,
                        -1,
                    ):
                        episode_operators[b][k] = approach_result.op
                # Continue execution
                if CFG.render_planner_collection_img != "":
                    import imageio.v2 as iio

                    batched_img = envs.render()
                    for b in range(envs.num_envs):
                        img = batched_img[b]
                        env_collided = environments_to_process[b]
                        iio.imwrite(
                            os.path.join(
                                CFG.render_planner_collection_img,
                                f"epi_{all_episode_count}_env_{b}_env_post_rl_collide_{env_collided}.png",
                            ),
                            img.cpu().numpy(),
                        )
                for _ in range(CFG.max_env_steps):
                    action = approach_result.action
                    obs_after_rl, _, _, _, infos = envs.step(action)
                    success_mask |= infos["success"].to(torch.bool)

                    for b in range(envs.num_envs):
                        episode_states[b].append(obs_after_rl[b].clone())
                        episode_actions[b].append(action[b].clone())
                        episode_operators[b].append(approach_result.op)

                    if success_mask[environments_to_process].all():
                        logging.info(
                            "All processed environments succeeded after replanning"
                        )
                        break

                    if self.exhausted[environments_to_process].all():
                        logging.info(
                            "All processed environments exhausted after replanning"
                        )
                        logging.info(f"Exhausted operator: {self._current_operator}")
                        break

                    approach_result = self.step(obs_after_rl, 0.0, False, False, {})

                if CFG.render_planner_collection_img != "":
                    import imageio.v2 as iio

                    batched_img = envs.render()
                    for b in range(envs.num_envs):
                        img = batched_img[b]
                        env_collided = environments_to_process[b]
                        succ = success_mask[b].item()
                        iio.imwrite(
                            os.path.join(
                                CFG.render_planner_collection_img,
                                f"epi_{all_episode_count}_env_{b}_env_final_collide_{env_collided}_succ_{succ}.png",
                            ),
                            img.cpu().numpy(),
                        )
                final_episode_states = [
                    episode_states[b]
                    for b in range(envs.num_envs)
                    if environments_to_process[b]
                ]
                final_episode_actions = [
                    episode_actions[b]
                    for b in range(envs.num_envs)
                    if environments_to_process[b]
                ]
                final_episode_operators = [
                    episode_operators[b]
                    for b in range(envs.num_envs)
                    if environments_to_process[b]
                ]
                final_success_mask = success_mask[environments_to_process]
                return {
                    "episode_states": final_episode_states,
                    "episode_actions": final_episode_actions,
                    "episode_operators": final_episode_operators,
                    "success_mask": final_success_mask,
                }

        else:
            if allow_rl:
                logging.info("should trigger RL but no environments to process")
                return None
        if CFG.render_planner_collection_img != "":
            import imageio.v2 as iio

            batched_img = envs.render()
            for b in range(envs.num_envs):
                img = batched_img[b]
                env_collided = environments_to_process[b]
                succ = success_mask[b].item()
                iio.imwrite(
                    os.path.join(
                        CFG.render_planner_collection_img,
                        f"epi_{all_episode_count}_env_{b}_env_final_collide_{env_collided}_succ_{succ}.png",
                    ),
                    img.cpu().numpy(),
                )
        return {
            "episode_states": episode_states,
            "episode_actions": episode_actions,
            "episode_operators": episode_operators,
            "success_mask": success_mask,
        }

    def _execute_rl_skill_for_batch(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        obs: Tensor,
        action: Tensor,
        episode_states: List[List[Tensor]],
        episode_actions: List[List[Tensor]],
        episode_operators: List[List[Optional[GroundOperator]]],
        success_mask: Tensor,
        max_rl_steps: int,
        rl_static_steps: int,
        scenario_info: dict,
    ) -> Tuple[Tensor, Tensor]:
        """Execute RL skill for a batch of environments.

        Returns:
            Updated obs and action tensors
        """

        policy.reset(obs)
        # Create failure operator
        train_objects = scenario_info.get("train_objects", "").split(",")
        operator_parameters = [self.perceiver.objects[obj] for obj in train_objects]
        parameter_tuple = tuple(operator_parameters)

        failure_op = LiftedOperator(
            name=f"{self.failed_op}_{self.new_operator.name}",
            parameters=self.new_operator.parameters,
            preconditions=set(),
            add_effects=set(),
            delete_effects=set(),
        )
        ground_failed_op = failure_op.ground(parameter_tuple)

        # Replace last failed operator
        for b in range(envs.num_envs):
            for k in range(len(episode_operators[b]) - 1, -1, -1):
                op = episode_operators[b][k]
                if op is not None:
                    is_failed_op = op.parent.name == self.failed_op
                    if is_failed_op:
                        episode_operators[b][k] = ground_failed_op
                    else:
                        break

        # Prepare static actions
        rl_static_actions = torch.zeros_like(action)
        if not CFG.delta_finger_control:
            for b in range(envs.num_envs):
                if len(episode_actions[b]) > 0:
                    rl_static_actions[b, -1] = episode_actions[b][-1][-1]

        # Pre-RL stabilize
        for _ in range(CFG.pre_rl_stabilize_steps):
            res, _ = self.step_policy_with_scenario(obs, policy, rl_static_actions)
            obs, _, _, _, infos = envs.step(res.action)
            success_mask |= infos["success"].to(torch.bool)

            for b in range(envs.num_envs):
                episode_states[b].append(obs[b].clone())
                episode_actions[b].append(res.action[b].clone())
                episode_operators[b].append(res.op)

        # RL execution
        for _ in range(max_rl_steps):
            res, terminal = self.step_policy_with_scenario(obs, policy)
            if terminal.all():
                break

            obs, _, _, _, infos = envs.step(res.action)
            success_mask |= infos["success"].to(torch.bool)

            for b in range(envs.num_envs):
                episode_states[b].append(obs[b].clone())
                episode_actions[b].append(res.action[b].clone())
                episode_operators[b].append(res.op)

        # Post-RL stabilize
        for _ in range(rl_static_steps):
            res, _ = self.step_policy_with_scenario(obs, policy, rl_static_actions)
            obs, _, _, _, infos = envs.step(res.action)
            success_mask |= infos["success"].to(torch.bool)

            for b in range(envs.num_envs):
                episode_states[b].append(obs[b].clone())
                episode_actions[b].append(res.action[b].clone())
                episode_operators[b].append(res.op)

        return obs, res.action

    def _collect_trajectories_with_sampler(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        init_state_sampler: Callable[[], Tensor],
        all_episode_count: int,
        num_episodes: int,
        scenario_name: str,
        allow_rl: bool = True,
        max_attempts: Optional[int] = None,
        dataset_save_path: Optional[Path] = None,
        save_frequency: int = 5,
        existing_trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[List[LowLevelTrajectory], bool, int]:
        """Collect trajectories using batched execution with a given initial state
        sampler.

        This core collection function is used by both real and dreamed scenario collection.
        Supports incremental saving every N episodes.

        Args:
            envs: Vectorized ManiSkill environment
            policy: RL policy for skill execution
            init_state_sampler: Callable that returns initial states [num_envs, state_dim]
            num_episodes: Number of successful episodes to collect
            scenario_name: Name/identifier for the scenario (for trajectory tagging)
            max_attempts: Maximum number of attempts before marking scenario as invalid
                         If None, will keep trying until num_episodes are collected
            dataset_save_path: Path to save trajectories incrementally (if None, no saving)
            save_frequency: Save trajectories every N new episodes
            existing_trajectories: Previously collected trajectories for this scenario

        Returns:
            trajectories: List of collected trajectories
            is_valid: True if scenario successfully collected enough episodes,
                     False if scenario proved invalid (max_attempts reached)
        """
        logging.info(
            f"Collecting {num_episodes} episodes for scenario '{scenario_name}'"
        )

        trajectories: List[LowLevelTrajectory] = []
        all_trajectories = existing_trajectories.copy() if existing_trajectories else []
        scenario_info = self.learning_info[self.curr_learning_phase]
        max_rl_steps = scenario_info.get("max_rl_steps", 20)
        rl_static_steps = scenario_info.get("rl_static_steps", 3)

        concective_attempt = 0
        success_count = 0
        episodes_since_last_save = 0

        # Helper to save current progress
        def save_progress() -> None:
            """Save all trajectories collected so far for this scenario."""
            if dataset_save_path is None:
                return

            scenario_dir = dataset_save_path / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_file = scenario_dir / "trajectories.pkl"

            import pickle

            with open(scenario_file, "wb") as f:
                pickle.dump(all_trajectories, f)

            logging.info(
                f"[CHECKPOINT] Saved {len(all_trajectories)} trajectories to {scenario_file}"
            )

        while success_count < num_episodes:
            if max_attempts is not None and concective_attempt >= max_attempts:
                # Save any collected trajectories before marking as invalid
                if episodes_since_last_save > 0:
                    save_progress()
                logging.warning(
                    f"Scenario '{scenario_name}' reached max attempts ({max_attempts}) "
                    f"with only {success_count} successes. Marking as invalid."
                )
                return trajectories, False, all_episode_count

            logging.info(
                f"Attempt {concective_attempt} for scenario '{scenario_name}' "
                f"({success_count}/{num_episodes} collected)"
            )

            # Sample initial states
            init_state_sampled = init_state_sampler()
            obs, _ = envs.reset(options={"init_state": init_state_sampled})

            # Execute episode and collect trajectory data
            episode_data = self._execute_single_batch_episode(
                envs,
                policy,
                obs,
                max_rl_steps,
                rl_static_steps,
                allow_rl,
                scenario_info,
                all_episode_count,
            )

            if episode_data is None:
                # Episode failed to initialize (planning failed)
                concective_attempt += 1
                continue

            # Store successful trajectories
            for b in range(envs.num_envs):
                if episode_data["success_mask"][b]:
                    traj = LowLevelTrajectory(
                        _states=episode_data["episode_states"][b],
                        _actions=[
                            ApproachStepResult(_action=a, op=o)
                            for a, o in zip(
                                episode_data["episode_actions"][b],
                                episode_data["episode_operators"][b],
                            )
                        ],
                        _train_scenario_idx=scenario_name,
                        _success=True,
                        _train_task_idx=all_episode_count,
                    )
                    trajectories.append(traj)
                    all_trajectories.append(traj)
                    success_count += 1
                    all_episode_count += 1
                    episodes_since_last_save += 1

                    # Incremental save every save_frequency episodes
                    if episodes_since_last_save >= save_frequency:
                        save_progress()
                        episodes_since_last_save = 0

                    if success_count >= num_episodes:
                        break

            if success_count > 0:
                concective_attempt = 0
                logging.info(
                    f"Collected {success_count}/{num_episodes} successful trajectories for '{scenario_name}'"
                )
            else:
                logging.info(f"No successful trajectories collected in this attempt.")
                concective_attempt += 1

        # Final save if there are unsaved trajectories
        if episodes_since_last_save > 0:
            save_progress()

        logging.info(
            f"Successfully collected {len(trajectories)} trajectories for scenario '{scenario_name}'"
        )
        return all_trajectories, True, all_episode_count
