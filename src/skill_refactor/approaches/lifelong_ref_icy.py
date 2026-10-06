"""Lifelong refactorization approach adapted for stochastic ManiSkill environments.

This variant removes the replay mechanism from data collection, as ManiSkill
environments are non-deterministic and replaying actions does not reproduce the same
states.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import Tensor

from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
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


class LifelongRefIcyApproach(LifelongRefApproach):
    """Lifelong refactorization approach for stochastic ManiSkill environments.

    This class inherits from LifelongRefApproach but redefines the planner data
    collection methods to work with non-deterministic environments where action replay
    does not produce identical states.
    """

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "lifelong_ref_mani"

    def collect_planner_data(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        provided_init_states: List[Tensor],
        real_env_scenario_name: str = "sc1_pre_n1_sc2_pre_n1",
    ) -> PlannerDataset:
        """Collect Planner training data from stochastic ManiSkill environment.

        Uses batched collision state tracking for both real and dreamed scenarios.
        All collection is done via the shared `_collect_trajectories_with_sampler` method.

        Args:
            envs: The vectorized environment
            policy: The policy to use for RL skill execution
            provided_init_states: Initial states to create the sampler from
            real_env_scenario_name: The scenario name for real environment trajectories

        Returns:
            PlannerDataset with trajectories from both real environment and dreaming.
        """
        logging.info("Collecting training data with planner (ManiSkill variant)\n")

        # ===============================
        # Phase 1: Collect Real Environment Trajectories
        # ===============================
        logging.info("=" * 80)
        logging.info("PHASE 1: Real Environment Data Collection")
        logging.info("%s\n", "=" * 80)
        all_episode_count = 0
        num_obstructions = len(CFG.scenario.split(","))
        if num_obstructions > 1:
            assert (
                envs.num_envs == 1
            ), "For multiple obstructions, num_envs must be 1 to ensure correct rl execution"

        # Create sampler for real environment
        relative_state_sampler, basic_node, pre_post_list = (
            self.get_init_state_sampler_from_provided(provided_init_states, envs.device)
        )

        def real_init_sampler() -> Tensor:
            env_reset_init, _ = envs.reset()
            return self.sample_init_states_from_sampler(
                envs, env_reset_init, relative_state_sampler, basic_node, pre_post_list
            )

        # Collect real trajectories
        real_trajectories, _, all_episode_count = (
            self._collect_trajectories_with_sampler(
                envs,
                policy,
                real_init_sampler,
                all_episode_count,
                num_episodes=CFG.num_train_episodes_planner_before_dreaming,
                scenario_name="real_env",
            )
        )

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

            # Collect trajectories for this scenario
            # pre the latest scenario
            allow_rl = scenario_combo[-1][1] == 0
            scenario_trajs, is_valid, all_episode_count = (
                self._collect_trajectories_with_sampler(
                    envs,
                    policy,
                    dreamed_init_sampler,
                    all_episode_count,
                    num_episodes=desired_scenario_number,
                    scenario_name=scenario_name,
                    allow_rl=allow_rl,
                    max_attempts=CFG.max_consecutive_failures,
                )
            )

            if is_valid:
                trajectories.extend(scenario_trajs)
                logging.info(
                    f"Successfully collected {len(scenario_trajs)} trajectories for scenario '{scenario_name}'"
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

    def _extract_relative_states_from_trajectory_single_env(
        self,
        states_list: List[Tensor],
        operators_list: List[Optional[GroundOperator]],
        relative_states: Dict[str, List[Tensor]],
        rl_operator_names: List[str],
    ) -> None:
        """Extract relative states from single environment trajectory.

        Args:
            states_list: List of state tensors from a single environment (no batch dim)
            operators_list: List of operators executed
            relative_states: Dict to update with pre/post relative states
            rl_operator_names: List of RL operator names to track
        """
        if len(operators_list) == 0 or operators_list[0] is None:
            return

        curr_op = operators_list[0]
        # Add batch dimension for compatibility with full_state_to_relative_state
        last_op_state = states_list[0].unsqueeze(0)
        last_op_state_w_id = self.system.full_state_to_relative_state(
            last_op_state, curr_op.parameters[1].name
        )

        for step, ground_op in enumerate(operators_list):
            if ground_op != curr_op:
                # Skip if operator is None
                if ground_op is None:
                    continue

                # Add batch dimension
                s = states_list[step].unsqueeze(0)
                curr_op_state_w_id = self.system.full_state_to_relative_state(
                    s, curr_op.parameters[1].name
                )

                if curr_op.parent.name in rl_operator_names:
                    scenario_idx = rl_operator_names.index(curr_op.parent.name)
                    scenario = list(self.learning_info.keys())[scenario_idx]

                    # Verify node IDs are in potential_comp_nodes
                    assert torch.isin(
                        curr_op_state_w_id[:, scenario - 1, 0],
                        torch.tensor(
                            self.potential_comp_nodes,
                            device=curr_op_state_w_id.device,
                        ),
                    ).all(), "All node IDs must be in potential_comp_nodes"

                    relative_states[f"sc{scenario}_pre"].append(
                        last_op_state_w_id[:, scenario - 1, 1:].clone()
                    )
                    relative_states[f"sc{scenario}_post"].append(
                        curr_op_state_w_id[:, scenario - 1, 1:].clone()
                    )

                curr_op = ground_op
                last_op_state_w_id = curr_op_state_w_id

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
    ) -> Tuple[List[LowLevelTrajectory], bool, int]:
        """Collect trajectories using batched execution with a given initial state
        sampler.

        This core collection function is used by both real and dreamed scenario collection.

        Args:
            envs: Vectorized ManiSkill environment
            policy: RL policy for skill execution
            init_state_sampler: Callable that returns initial states [num_envs, state_dim]
            num_episodes: Number of successful episodes to collect
            scenario_name: Name/identifier for the scenario (for trajectory tagging)
            max_attempts: Maximum number of attempts before marking scenario as invalid
                         If None, will keep trying until num_episodes are collected

        Returns:
            trajectories: List of collected trajectories
            is_valid: True if scenario successfully collected enough episodes,
                     False if scenario proved invalid (max_attempts reached)
        """
        logging.info(
            f"Collecting {num_episodes} episodes for scenario '{scenario_name}'"
        )

        trajectories: List[LowLevelTrajectory] = []
        scenario_info = self.learning_info[self.curr_learning_phase]
        max_rl_steps = scenario_info.get("max_rl_steps", 20)
        rl_static_steps = scenario_info.get("rl_static_steps", 3)

        concective_attempt = 0
        success_count = 0

        while success_count < num_episodes:
            if max_attempts is not None and concective_attempt >= max_attempts:
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
            )

            if episode_data is None:
                # Episode failed to initialize (planning failed)
                concective_attempt += 1
                continue

            # Downsample episode data to reduce trajectory length
            max_states_per_op_segment = 2 * CFG.num_middle_states
            episode_data = self.downsample_data(episode_data, max_states_per_op_segment)

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
                    success_count += 1
                    all_episode_count += 1

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

        logging.info(
            f"Successfully collected {len(trajectories)} trajectories for scenario '{scenario_name}'"
        )
        return trajectories, True, all_episode_count

    def downsample_data(
        self,
        episode_data: Dict[str, Any],
        max_states_per_operator: int,
    ) -> Dict[str, Any]:
        """Downsample episode data operator-wise to reduce trajectory length.

        For each operator execution segment, downsample states to at most
        max_states_per_operator if the segment is longer. This is done operator-wise,
        meaning each consecutive segment with the same operator is downsampled independently.

        Args:
            episode_data: Dictionary with 'episode_states', 'episode_actions',
                         'episode_operators', 'success_mask'
            max_states_per_operator: Maximum number of states per operator segment

        Returns:
            Downsampled episode data with same structure
        """
        downsampled_states: List[List[Tensor]] = []
        downsampled_actions: List[List[Tensor]] = []
        downsampled_operators: List[List[Optional[GroundOperator]]] = []

        num_envs = len(episode_data["episode_states"])
        total_original_states = 0
        total_downsampled_states = 0
        num_downsampled_segments = 0

        for b in range(num_envs):
            states = episode_data["episode_states"][b]
            actions = episode_data["episode_actions"][b]
            operators = episode_data["episode_operators"][b]

            original_length = len(states)

            if len(operators) == 0:
                downsampled_states.append(states)
                downsampled_actions.append(actions)
                downsampled_operators.append(operators)
                total_original_states += original_length
                total_downsampled_states += original_length
                continue

            # Find operator segments and downsample each
            new_states: List[Tensor] = []
            new_actions: List[Tensor] = []
            new_operators: List[Optional[GroundOperator]] = []

            segment_start = 0
            current_op = operators[0]

            for i in range(1, len(operators) + 1):
                # Check if operator changed or reached end
                # Need to handle None operators carefully
                operator_changed = False
                if i < len(operators):
                    next_op = operators[i]
                    if current_op is None and next_op is None:
                        operator_changed = False
                    elif current_op is None or next_op is None:
                        operator_changed = True
                    else:
                        operator_changed = next_op != current_op

                if i == len(operators) or operator_changed:
                    # Process segment [segment_start:i)
                    # States include one extra for the final state after last action
                    segment_states = states[segment_start : i + 1]
                    segment_actions = actions[segment_start:i]
                    segment_operators = operators[segment_start:i]

                    num_states = len(segment_states)

                    # Downsample if needed
                    if num_states > max_states_per_operator:
                        num_downsampled_segments += 1
                        # Select state indices uniformly, always including first and last
                        indices = torch.linspace(
                            0,
                            num_states - 1,
                            max_states_per_operator,
                            dtype=torch.long,
                        )

                        segment_states = [segment_states[idx] for idx in indices]

                        # For actions/operators, select at state indices (except last state)
                        action_indices = indices[:-1]
                        segment_actions = [
                            segment_actions[idx] for idx in action_indices
                        ]
                        segment_operators = [
                            segment_operators[idx] for idx in action_indices
                        ]

                    # Append to new lists
                    if len(new_states) == 0:
                        # First segment: include all states
                        new_states.extend(segment_states)
                    else:
                        # Subsequent segments: skip first state (already in previous segment)
                        new_states.extend(segment_states[1:])

                    new_actions.extend(segment_actions)
                    new_operators.extend(segment_operators)

                    # Move to next segment
                    if i < len(operators):
                        segment_start = i
                        current_op = operators[i]

            downsampled_states.append(new_states)
            downsampled_actions.append(new_actions)
            downsampled_operators.append(new_operators)

            total_original_states += original_length
            total_downsampled_states += len(new_states)

        if num_downsampled_segments > 0:
            reduction_pct = (
                100.0
                * (total_original_states - total_downsampled_states)
                / total_original_states
            )
            logging.info(
                f"Downsampled {num_downsampled_segments} segments: "
                f"{total_original_states} -> {total_downsampled_states} states "
                f"({reduction_pct:.1f}% reduction)"
            )

        return {
            "episode_states": downsampled_states,
            "episode_actions": downsampled_actions,
            "episode_operators": downsampled_operators,
            "success_mask": episode_data["success_mask"],
        }

    def _execute_single_batch_episode(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        obs: Tensor,
        max_rl_steps: int,
        rl_static_steps: int,
        allow_rl: bool,
        scenario_info: dict,
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

        curr_op_name = approach_result.op.parent.name
        curr_op_param_names = [obj.name for obj in approach_result.op.parameters]

        # Data collection buffers
        episode_states: List[List[Tensor]] = [[] for _ in range(envs.num_envs)]
        episode_actions: List[List[Tensor]] = [[] for _ in range(envs.num_envs)]
        episode_operators: List[List[Optional[GroundOperator]]] = [
            [] for _ in range(envs.num_envs)
        ]
        collision_states: List[Optional[Tensor]] = [None] * envs.num_envs
        collision_param_names: List[Optional[List[str]]] = [None] * envs.num_envs
        colliding_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)
        success_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)

        # Store initial state
        for b in range(envs.num_envs):
            episode_states[b].append(obs[b].clone())

        previous_obs = obs.clone()

        # Phase 1: Execute plan and track collisions
        # NOTE: Here we do the max_env_steps, instead of using max_skill_steps.
        for _ in range(CFG.max_env_steps):
            action = approach_result.action
            obs, _, _, _, infos = envs.step(action)

            # Check for collisions during failed operator execution
            # Match based on parameter name prefixes (excluding last character)
            # to allow matching e.g., "b1" with "b2", but only for names with 2+ chars
            # Single-character names must match exactly
            if (
                any(infos["is_colliding"])
                and (curr_op_name == self.failed_op)
                and all(
                    (
                        obj_name[:-1] == param_name[:-1]
                        if len(obj_name) >= 2 and len(param_name) >= 2
                        else obj_name == param_name
                    )
                    for obj_name, param_name in zip(
                        self.failed_objects,
                        curr_op_param_names,
                    )
                )
            ):
                colliding_envs = infos["is_colliding"].to(torch.bool)
                newly_colliding = colliding_envs & ~colliding_mask & ~self.exhausted
                if not allow_rl:
                    logging.info(
                        "Collision detected during failed operator execution, "
                        "but RL is not allowed. Ending episode."
                    )
                    return None

                for b in range(envs.num_envs):
                    if newly_colliding[b]:
                        collision_states[b] = previous_obs[b].clone()
                        collision_param_names[b] = curr_op_param_names.copy()
                        colliding_mask[b] = True

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

            previous_obs = obs.clone()
            approach_result = self.step(obs, 0.0, False, False, {})
            if approach_result.op is None:
                break
            curr_op_name = approach_result.op.parent.name
            curr_op_param_names = [obj.name for obj in approach_result.op.parameters]

        # Phase 2: Execute RL for collided environments
        environments_to_process = colliding_mask & ~success_mask
        if environments_to_process.any():
            logging.info(
                f"Executing RL skill for {environments_to_process.sum().item()} collided environments"
            )
            obs_after_rl, action = self._execute_rl_skill_for_batch(
                envs,
                policy,
                obs,
                action,
                collision_states,
                collision_param_names,
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
                # NOTE: For IcyTransport, we don't overwrite the
                # RL static steps with the first operator.
                # Since the static steps might be very useful after RL.
                # Continue execution
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

                    approach_result = self.step(obs_after_rl, 0.0, False, False, {})

        else:
            if allow_rl:
                logging.info("should trigger RL but no environments to process")
                return None
        return {
            "episode_states": episode_states,
            "episode_actions": episode_actions,
            "episode_operators": episode_operators,
            "success_mask": success_mask,
        }

    def _execute_rl_skill_for_batch(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        obs: Tensor,
        action: Tensor,
        collision_states: List[Optional[Tensor]],
        collision_param_names: List[Optional[List[str]]],
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
        # Collect valid collision states
        valid_collision_states = [
            collision_states[b]
            for b in range(envs.num_envs)
            if collision_states[b] is not None
        ]

        if not valid_collision_states:
            logging.warning("No valid collision states, skipping RL")
            return obs, action

        # Build collision state batch with trajectory copying
        valid_collision_env_indices = [
            b for b in range(envs.num_envs) if collision_states[b] is not None
        ]

        collision_state_list: List[Tensor] = []
        collision_param_names_list: List[List[str]] = []

        for b in range(envs.num_envs):
            if collision_states[b] is not None:
                collision_state_list.append(cast(Tensor, collision_states[b]))
                collision_param_names_list.append(
                    cast(List[str], collision_param_names[b])
                )
            else:
                # Randomly select and copy trajectory
                random_idx = self.rng.integers(0, len(valid_collision_states))
                collision_state_list.append(
                    cast(Tensor, valid_collision_states[random_idx])
                )

                source_env_idx = valid_collision_env_indices[random_idx]
                episode_states[b] = [s.clone() for s in episode_states[source_env_idx]]
                episode_actions[b] = [
                    a.clone() for a in episode_actions[source_env_idx]
                ]
                episode_operators[b] = list(episode_operators[source_env_idx])
                collision_param_names_list.append(
                    cast(List[str], collision_param_names[source_env_idx])
                )

        collision_state_batch = torch.stack(collision_state_list)
        obs, _ = envs.reset(options={"init_state": collision_state_batch})
        policy.reset(obs)

        # Create failure operator using actual collision parameter names
        # Use parameters from the first colliding environment
        actual_param_names = collision_param_names_list[0]
        train_objects = scenario_info.get("train_objects", "").split(",")
        assert train_objects[-1] not in actual_param_names
        actual_param_names.append(train_objects[-1])  # Add robot gripper
        operator_parameters = [
            self.perceiver.objects[obj_name] for obj_name in actual_param_names
        ]
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
