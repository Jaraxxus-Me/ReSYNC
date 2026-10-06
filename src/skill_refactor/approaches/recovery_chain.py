"""Recovery chain approach: detects failures and triggers RL recovery skills,
but does not use abstract planning to avoid failures.

This is a baseline that compares to the full lifelong_ref approach by:
- Using failure detection to trigger RL skills when collisions occur
- NOT learning new predicates or operators for proactive planning
- NOT refactoring existing skills with new operators
- Simply chaining: original plan → detect failure → execute recovery skill → replan

Key differences from lifelong_ref:
1. No predicate learning (no TopDownPredicateLearner)
2. No operator learning (keeps original operators only)
3. No skill refactorization (doesn't update domain with new operators)
4. Recovery is reactive (only triggered by failure detector) not proactive (no planning)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from relational_structs import Object
from torch import Tensor

from skill_refactor.approaches.base import (
    BaseRLTAMPSystem,
)
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.pred_learner.initiation_function_learner import (
    FailureDataset,
    InitiationFunctionLearner,
)
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.benchmarks.wrappers import PlanningStatesVectorEnv
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action
from skill_refactor.utils.structs import (
    ApproachStepResult,
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
)
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanner,
    TaskThenMotionPlanningFailure,
)


class RecoveryChainApproach(LifelongRefApproach):
    """Recovery chain baseline approach.

    This approach detects failures during execution and triggers a learned recovery
    skill, but does NOT update the symbolic planning model (operators/predicates). It
    represents a reactive baseline without proactive planning.
    """

    def __init__(
        self,
        system: BaseRLTAMPSystem,
        seed: int,
        planner_id: str = "pyperplan",
    ) -> None:
        """Initialize recovery chain approach."""
        super().__init__(system, seed, planner_id)

        # Recovery-specific state - support multiple recovery skills for progressive learning
        self._in_recovery_mode: Tensor = torch.zeros(1, dtype=torch.bool)
        self._activate_recovery_on_failure: bool = True
        self._recovery_skills: Dict[int, LiftedOperatorSkill] = (
            {}
        )  # scenario_id -> skill
        self._recovery_operators: Dict[int, LiftedOperator] = (
            {}
        )  # scenario_id -> operator
        self._recovery_ground_operators: Dict[int, GroundOperator | None] = (
            {}
        )  # scenario_id -> ground_op
        self._failure_detectors: Dict[int, Callable[[Tensor], Tensor] | None] = (
            {}
        )  # scenario_id -> detector
        self._failed_op_to_scenarios: Dict[str, List[int]] = (
            {}
        )  # failed_op_name -> list of scenario_ids
        self._current_recovery_scenario: int | None = (
            None  # Currently executing recovery scenario
        )
        self._failured_det_operator_objects: Dict[str, List[int]] = (
            CFG.failured_det_operator_objects
        )

        for op in self.operators:
            assert op.name in self._failured_det_operator_objects

        logging.info("Initialized RecoveryChainApproach (reactive recovery only)")

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "recovery_chain"

    def update_learning_info(
        self,
        episode: int,
        learning_info: dict,
        latest_tamp_system: BaseRLTAMPSystem | None = None,
    ) -> None:
        """Update learning info and build failed operator to scenario mapping.

        Args:
            episode: The scenario/episode ID
            learning_info: Dictionary containing scenario configuration
            latest_tamp_system: Optional updated TAMP system
        """
        # Call parent implementation
        super().update_learning_info(episode, learning_info, latest_tamp_system)

        # Build mapping from failed operator to scenario (support multiple scenarios per operator)
        failed_skill = learning_info.get("failed_skill", "")
        if failed_skill:
            if failed_skill not in self._failed_op_to_scenarios:
                self._failed_op_to_scenarios[failed_skill] = []
            self._failed_op_to_scenarios[failed_skill].append(episode)
            logging.info(
                f"Registered scenario {episode} for failed operator '{failed_skill}'"
            )

    def extract_failure_det_input(
        self,
        obs: Tensor,
        scenario_id: int,
    ) -> Tensor:
        """Extract input features for failure detection from observation.

        Args:
            obs: Observation tensor [batch_size, obs_dim]
        Returns:
            input_features: Input features for failure detection [batch_size, feature_dim]
        """
        assert self._current_operator is not None, "Current operator is not set"
        curr_object_ids = self._failured_det_operator_objects[
            self._current_operator.parent.name
        ]
        objects = []
        for obj_id in curr_object_ids:
            obj = self._current_operator.parameters[obj_id]
            objects.append(obj)

        train_objects = self.learning_info[scenario_id].get("train_objects", "")
        assert train_objects, "train_objects not specified in learning_info"
        region_name = train_objects.split(",")[-1]
        all_objects = self.system.perceiver.objects
        region_obj = all_objects.as_dict().get(region_name)
        assert region_obj is not None, f"Region object {region_name} not found"

        objects.append(region_obj)

        # Convert observations to graph data
        graph_data_list = self.system.state_to_graph(obs)

        node_dim = graph_data_list[0].node_features.shape[-1]
        feature_dim = node_dim * len(objects)
        batch_size = obs.shape[0]

        # Collect features for all batch×grounding combinations
        all_features = torch.empty((batch_size, feature_dim), dtype=torch.float32).to(
            obs.device
        )

        for batch_idx in range(batch_size):
            graph_data = graph_data_list[batch_idx]
            assert graph_data.object_to_node is not None
            node_feature_list = [
                graph_data.node_features[graph_data.object_to_node[obj]]
                for obj in objects
            ]
            features = torch.cat(node_feature_list, dim=-1)
            all_features[batch_idx] = features

        return all_features

    def collect_planner_data(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        envs: ManiSkillVectorEnv,
        provided_init_states: List[Tensor],
        real_env_scenario_name: str = "sc1_pre_n1_sc2_pre_n1",
    ) -> FailureDataset:
        """Collect failure detection training data from ManiSkill environment.

        Two-phase collection:
        - Phase 1 (Wake): Collect real environment data
        - Phase 2 (Sleep): Sample compositional "pre" states for dreaming

        Args:
            envs: The vectorized environment
            policy: RL policy (unused for recovery chain)
            provided_init_states: Initial states to create the sampler from
            real_env_scenario_name: The scenario name for real environment trajectories

        Returns:
            FailureDataset with states and labels (0=no collision, 1=collision).
        """
        logging.info("Collecting failure detection training data (Wake-Sleep)\n")

        # ===============================
        # Phase 1: Collect Real Environment Data (Wake)
        # ===============================
        logging.info("=" * 80)
        logging.info("PHASE 1: Real Environment Data Collection (Wake)")
        logging.info("%s\n", "=" * 80)

        self._activate_recovery_on_failure = (
            False  # Disable recovery during data collection
        )

        num_obstructions = len(CFG.scenario.split(","))
        if num_obstructions > 1:
            assert envs.num_envs == 1, "For multiple obstructions, num_envs must be 1"

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
        real_states, real_labels = self._collect_failure_states_with_sampler(
            envs,
            real_init_sampler,
            num_episodes=CFG.num_train_episodes_planner_before_dreaming,
            scenario_name="real_env",
        )

        logging.info(
            f"Collected {len(real_states)} real states "
            f"({sum(real_labels)} collisions, {len(real_labels) - sum(real_labels)} pre-collisions)"
        )

        # ===============================
        # Phase 2: Dreaming - Compositional Data Collection (Sleep)
        # ===============================
        logging.info("\n%s", "=" * 80)
        logging.info("PHASE 2: Dreaming Data Collection (Sleep)")
        logging.info("%s\n", "=" * 80)

        all_states = list(real_states)
        all_labels = list(real_labels)

        # Build all scenario combinations (only "pre" states for recovery)
        assert self.potential_comp_nodes, "No potential composition nodes available"
        num_comp_nodes = len(self.potential_comp_nodes)
        total_scenarios = num_comp_nodes**num_obstructions
        desired_scenario_number = (
            CFG.num_train_episodes_planner
            - CFG.num_train_episodes_planner_before_dreaming
        ) // total_scenarios

        import itertools

        all_scenario_combos: List[Tuple[Tuple[int, int, int], ...]] = []
        all_scenario_names: List[str] = []
        obstruction_choices = []

        for scenario in list(self.learning_info.keys()):
            obstruction_choice = []
            for pre_post in [0]:  # Only "pre" states for recovery detection
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
        assert (
            real_env_scenario_name in all_scenario_names
        ), f"real_env_scenario_name '{real_env_scenario_name}' not in valid scenarios"
        real_env_scenario_idx = all_scenario_names.index(real_env_scenario_name)
        scenario_proven_invalid[real_env_scenario_idx] = True  # Already collected

        logging.info(
            f"Target: {desired_scenario_number} episodes per scenario "
            f"({total_scenarios} total scenarios = {num_comp_nodes}^{num_obstructions})"
        )
        logging.info(
            f"Real environment scenario '{real_env_scenario_name}' already collected"
        )

        # Generate relative state sampler for dreaming
        relative_state_sampler_dream = relative_state_sampler

        # Collect each dreamed scenario
        for scenario_idx, (scenario_combo, scenario_name) in enumerate(
            zip(all_scenario_combos, all_scenario_names)
        ):
            # Skip scenarios that are invalid or already collected
            if scenario_proven_invalid[scenario_idx]:
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
                pre_post_list_dream: List[Tuple[int, ...]] = [
                    (sc_id,) for sc_id in list(self.learning_info.keys())
                ]
                basic_node_dream = (
                    torch.zeros(3, dtype=torch.long, device=init_obs.device) - 100
                )

                for encounter_id, (sc_id, pre_post, node_id) in enumerate(combo):
                    pre_post_list_dream[encounter_id] = pre_post_list_dream[
                        encounter_id
                    ] + (pre_post,)
                    basic_node_dream[sc_id - 1] = node_id

                dreamed_rel_state_data = relative_state_sampler_dream(
                    pre_post_list_dream
                ).to(init_obs.device)
                dreamed_rel_state = torch.cat(
                    [basic_node_dream.unsqueeze(1), dreamed_rel_state_data], dim=1
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

            # Collect states for this scenario
            try:
                scenario_states, scenario_labels = (
                    self._collect_failure_states_with_sampler(
                        envs,
                        dreamed_init_sampler,
                        num_episodes=desired_scenario_number,
                        scenario_name=scenario_name,
                    )
                )
                all_states.extend(scenario_states)
                all_labels.extend(scenario_labels)
                logging.info(
                    f"Successfully collected {len(scenario_states)} states for scenario '{scenario_name}'"
                )
            except RuntimeError:
                scenario_proven_invalid[scenario_idx] = True
                logging.warning(
                    f"Scenario '{scenario_name}' marked as INVALID after max attempts"
                )

        # Final verification
        logging.info("\n%s", "=" * 80)
        logging.info("FINAL SUMMARY")
        logging.info("%s", "=" * 80)
        logging.info(f"Total states collected: {len(all_states)}")
        logging.info(
            f"Real environment: {len(real_states)} states (scenario: {real_env_scenario_name})"
        )
        logging.info(f"Dreamed scenarios: {len(all_states) - len(real_states)} states")
        logging.info(
            f"Invalid scenarios: {sum(scenario_proven_invalid)}/{total_scenarios}"
        )
        logging.info(
            f"Label distribution: {sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative"
        )
        logging.info("%s\n", "=" * 80)

        return FailureDataset(
            states=all_states,
            labels=torch.tensor(all_labels, dtype=torch.long),
        )

    def _collect_failure_states_with_sampler(
        self,
        envs: ManiSkillVectorEnv,
        init_state_sampler: Callable[[], Tensor],
        num_episodes: int,
        scenario_name: str,
    ) -> Tuple[List[Tensor], List[int]]:
        """Collect failure detection states using a given initial state sampler.

        This core collection function is used by both wake and sleep phases.

        Args:
            envs: Vectorized ManiSkill environment
            init_state_sampler: Callable that returns initial states [num_envs, state_dim]
            num_episodes: Number of successful episodes to collect
            scenario_name: Name/identifier for the scenario (for logging)

        Returns:
            states: List of state tensors
            labels: List of labels (0=no collision, 1=collision)
        """
        logging.info(
            f"Collecting {num_episodes} episodes for scenario '{scenario_name}'"
        )

        all_states: List[Tensor] = []
        all_labels: List[int] = []
        consecutive_attempt = 0
        episode = 0

        while episode < num_episodes:

            logging.info(
                f"Attempt {consecutive_attempt} for scenario '{scenario_name}' "
                f"({episode}/{num_episodes} collected)"
            )

            # Sample initial states
            init_state_sampled = init_state_sampler()
            obs, _ = envs.reset(options={"init_state": init_state_sampled})
            previous_obs = obs.clone()

            # Try to create initial plan
            try:
                approach_result = self.reset(obs, {})
            except TaskThenMotionPlanningFailure:
                consecutive_attempt += 1
                continue

            if approach_result.op is None:
                consecutive_attempt += 1
                continue

            # Track collision states per environment
            # Store ALL pre-collision states (not just the last one) since detector runs at every step
            episode_pre_collision_states: List[List[Tensor]] = [
                [] for _ in range(envs.num_envs)
            ]
            episode_collision_states: List[Optional[Tensor]] = [None] * envs.num_envs
            colliding_mask = torch.zeros(
                envs.num_envs, dtype=torch.bool, device=obs.device
            )

            # Execute the task plan and collect states
            for _ in range(CFG.max_env_steps):
                action = approach_result.action
                obs, _, _, _, infos = envs.step(action)

                # Check for collisions during failed operator execution
                if any(infos["is_colliding"]):
                    colliding_envs = infos["is_colliding"].to(torch.bool)
                    newly_colliding = colliding_envs & ~colliding_mask & ~self.exhausted

                    # Store collision states (positive examples)
                    for b in range(envs.num_envs):
                        if newly_colliding[b]:
                            full_obs = previous_obs[b].clone()
                            state_vec = self.extract_failure_det_input(
                                full_obs.unsqueeze(0), int(CFG.scenario)
                            )
                            episode_collision_states[b] = state_vec.clone()
                            colliding_mask[b] = True

                # Store ALL pre-collision states (negative examples)
                # Collect every state before collision since detector runs at every step
                for b in range(envs.num_envs):
                    if not colliding_mask[b] and not self.exhausted[b]:
                        full_obs = previous_obs[b].clone()
                        state_vec = self.extract_failure_det_input(
                            full_obs.unsqueeze(0), int(CFG.scenario)
                        )
                        episode_pre_collision_states[b].append(state_vec.clone())

                # Break if all succeeded or collided
                if torch.all(colliding_mask | self.exhausted):
                    break

                previous_obs = obs.clone()
                approach_result = self.step(obs, 0.0, False, False, {})
                if approach_result.op is None:
                    break

            # Store the episode data
            has_new_data = False
            for b in range(envs.num_envs):
                # Add ALL pre-collision states (negative examples)
                if episode_pre_collision_states[b]:
                    for pre_collision_state in episode_pre_collision_states[b]:
                        all_states.append(pre_collision_state)
                        all_labels.append(0)  # No collision
                    has_new_data = True

                # Add collision state (positive example)
                if episode_collision_states[b] is not None:
                    collision_state = episode_collision_states[b]
                    assert collision_state is not None  # For type checker
                    all_states.append(collision_state)
                    all_labels.append(1)  # Collision
                    has_new_data = True

            if has_new_data:
                episode += 1
                consecutive_attempt = 0
            else:
                consecutive_attempt += 1

            if len(all_states) > 0 and episode % 10 == 0:
                logging.info(
                    f"Collected {len(all_states)} states "
                    f"({sum(all_labels)} collisions, {len(all_labels) - sum(all_labels)} pre-collisions)"
                )

        logging.info(
            f"Successfully collected {len(all_states)} states for scenario '{scenario_name}'"
        )
        return all_states, all_labels

    def _gen_relative_dist_sampler(
        self, relative_states: Dict[str, List[Tensor]]
    ) -> Callable[[List[tuple[int, ...]]], Tensor]:
        """Generate a sampler function for relative states based on collected data.

        Args:
            relative_states: A dictionary mapping scenario keys to lists of relative
                state tensors.
        Returns:
            sampler: A function that takes a scenario_combo tuple (0=pre, 1=post for
                each component) and returns a sampled relative state tensor.
        """

        # 1. Split data dict into lists of tensors per component
        all_relative_states: List[List[List[Tensor]]] = []  # SC[Pre/Post[List[Tensor]]]
        empty_relative_states = torch.zeros_like(list(relative_states.values())[0][0])
        for scenario_id in [1, 2, 3]:
            # NOTE: all_relative_states is always ordered by sc1, sc2, ...
            # Padded with zeros if no data collected for that scenario
            if f"sc{scenario_id}_pre" in relative_states:
                all_relative_states.append(
                    [
                        relative_states[f"sc{scenario_id}_pre"],
                        relative_states[f"sc{scenario_id}_post"],
                    ]
                )
            else:
                all_relative_states.append(
                    [[empty_relative_states], [empty_relative_states]]
                )

        # 2. Fit Mean and Variance of each component
        relative_states_means: List[List[Tensor]] = []
        relative_states_vars: List[List[Tensor]] = []
        for comp_states in all_relative_states:
            comp_means = []
            comp_vars = []
            for state_list in comp_states:
                if len(state_list) == 0:
                    raise ValueError("No relative states collected for component.")
                states_tensor = torch.cat(state_list, dim=0)  # (N, D)
                comp_means.append(states_tensor.mean(dim=0))
                comp_vars.append(
                    states_tensor.var(dim=0, unbiased=False)
                    + CFG.dreaming_noise_base_var
                )
            relative_states_means.append(comp_means)
            relative_states_vars.append(comp_vars)

        # --- Sampler function ---
        def sampler(scenario_combo: List[Tuple[int, ...]]) -> Tensor:
            """Sample relative states for a specific scenario combination.

            Args:
                scenario_combo: Tuple of 0s and 1s (0=pre, 1=post) for each component
            """

            samples: List[Tensor] = []
            input_sc_ids = [sc_id for sc_id, _ in scenario_combo]
            for sc_id in [1, 2, 3]:
                if sc_id in input_sc_ids:
                    pre_post_id = scenario_combo[input_sc_ids.index(sc_id)][1]
                    means = relative_states_means[sc_id - 1][pre_post_id]
                    variances = relative_states_vars[sc_id - 1][pre_post_id]
                    noise = torch.randn_like(means)
                    sample = means + noise * variances.sqrt()
                    samples.append(sample)
                else:
                    means = relative_states_means[sc_id - 1][0]
                    variances = relative_states_vars[sc_id - 1][0]
                    noise = torch.randn_like(means)
                    sample = means + noise * variances.sqrt()
                    samples.append(sample)
            return torch.stack(samples, dim=0)  # (num_components, D)

        return sampler

    def update_domain_knowledge(
        self,
        learning_info: dict,
    ) -> None:
        """Load recovery policy and create recovery skill but do NOT update
        predicates/operators.

        Unlike lifelong_ref, this method only creates a recovery skill from the policy
        and does NOT call TopDownPredicateLearner or update the domain.

        Args:
            learning_info: Dictionary containing learning configuration with keys:
                - planner_learning_cfg_settings: Config for policy loading
                    - rl_config: Path to RL config
                    - pre_trained_policy_path: Path to pre-trained policy weights
        """
        logging.info(
            "Loading recovery policy and creating recovery skill (NO predicate/operator learning)"
        )

        # Load recovery policy from config
        rl_config_path = CFG.rl_config
        pre_trained_policy_path_str = CFG.pre_trained_policy_path
        assert Path(rl_config_path).is_file(), "RL config path is invalid"
        assert Path(
            pre_trained_policy_path_str
        ).is_file(), "Pre-trained policy path is invalid"

        policy = PPOCPolicy(seed=CFG.seed, rl_config=rl_config_path)

        # Initialize the policy with environment (required before use)
        planner = TaskThenMotionPlanner(
            types=self.system.types,
            predicates=self.perceiver.predicates_container.as_set(),
            perceiver=self.perceiver,
            operators=self.operators,
            skills=self.skills,
            fallback_action=self._fallback_action,
            normalize_action=self._normalize_action,
            arm_action_low=self._arm_action_low,
            arm_action_high=self._arm_action_high,
            planner_id="pyperplan",
        )
        envs_mani = PlanningStatesVectorEnv(
            self.system.env,
            self.system,
            learning_info,
            planner,
            CFG.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        policy.initialize(envs_mani)
        pre_trained_policy_path = Path(pre_trained_policy_path_str)
        policy.load(pre_trained_policy_path)

        # Load neural failure detector if model exists
        failure_model_path = Path(CFG.failured_det_nn_path)

        if failure_model_path.is_file():
            logging.info(f"Loading neural failure detector from {failure_model_path}")

            # Load the trained model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with open(CFG.failure_nn_config, "r", encoding="utf-8") as f:
                failure_config = yaml.safe_load(f)["failure_detector"]
            learner = InitiationFunctionLearner.load(
                failure_model_path,
                self.failed_op,
                self.failed_objects,
                failure_config,
                self.system,
                device,
            )

            # Create and store the interpreter for this scenario
            # Note: The type annotation in create_interpreter is incorrect - it actually returns
            # a function that just takes Tensor, not (Tensor, dict, GroundOperator)
            interpreter = learner.create_interpreter()
            self._failure_detectors[self.curr_learning_phase] = cast(
                Callable[[Tensor], Tensor], interpreter
            )
            logging.info(
                f"Neural failure detector loaded successfully for scenario {self.curr_learning_phase}"
            )
        else:
            logging.info(
                f"No neural failure detector found at {failure_model_path}, using default detector"
            )
            self._failure_detectors[self.curr_learning_phase] = None

        # Create recovery skill using the parent class method
        # Use the new_operator created in update_learning_info
        assert (
            self.new_operator is not None
        ), "new_operator must be set before update_domain_knowledge"
        recovery_skill_name = learning_info.get("tgt_skill", "").split("(")[0].strip()

        # Store the operator for this scenario
        self._recovery_operators[self.curr_learning_phase] = self.new_operator

        # Create and store recovery skill for this scenario
        self._recovery_skills[self.curr_learning_phase] = (
            self.create_new_operator_skill(
                policy=policy,
                operator=self.new_operator,
                operator_name=recovery_skill_name,
                scenario_id=self.curr_learning_phase,
            )
        )

        # Initialize ground operator storage
        self._recovery_ground_operators[self.curr_learning_phase] = None

        logging.info(
            f"Created recovery skill '{recovery_skill_name}' for scenario {self.curr_learning_phase} from policy at {pre_trained_policy_path_str}\n"
            "Domain knowledge (operators/predicates) NOT updated - using original planning model"
        )

    def reset(
        self,
        obs: Tensor,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Reset approach with initial observation."""
        # Reset recovery state
        self._in_recovery_mode = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        self._current_recovery_scenario = None

        # Call parent reset (creates task plan with original operators)
        return super().reset(obs, info)

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation.

        Key difference from lifelong_ref:
        - Uses failure detector to trigger recovery skill
        - Executes recovery skill until termination
        - Replans after recovery
        - Does NOT use updated operators for planning
        """
        del reward, terminated, truncated  # Unused in this approach

        obs = obs.to(torch.float32)

        # If in recovery mode, execute recovery skill
        if self._in_recovery_mode.any():
            return self._execute_recovery_step(obs, info)

        # Otherwise, execute normal planning (from parent class)
        normal_result = super().step(obs, 0.0, False, False, info)

        # Check if we should trigger recovery mode using the appropriate failure detector
        should_trigger_recovery = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        triggered_scenario_id = None

        # Determine which detectors to check based on current operator
        # Multiple scenarios may have detectors for the same operator (e.g., icy vs muddy regions)
        if self._current_operator is not None:
            current_op_name = self._current_operator.parent.name
            scenario_ids = self._failed_op_to_scenarios.get(current_op_name, [])

            # Check all detectors for this operator and use the one that actually fires
            for scenario_id in scenario_ids:
                if str(scenario_id) not in CFG.scenario.split(","):
                    continue
                if scenario_id in self._recovery_skills:
                    recovery_skill = self._recovery_skills[scenario_id]
                    if hasattr(recovery_skill, "initiates_on_failure"):
                        obj_centric_obs = self.extract_failure_det_input(
                            obs, scenario_id
                        )
                        scenario_trigger = recovery_skill.initiates_on_failure(  # type: ignore[attr-defined]
                            obj_centric_obs, self._current_operator
                        )
                        if scenario_trigger.any():
                            # Use the scenario whose detector actually fires
                            # (not just the first in the list)
                            should_trigger_recovery |= scenario_trigger
                            triggered_scenario_id = scenario_id
                            logging.info(f"Scenario {scenario_id} detector fired")

        if not self._activate_recovery_on_failure:
            should_trigger_recovery = torch.zeros_like(should_trigger_recovery)

        # Enter recovery mode if triggered
        if should_trigger_recovery.any() and triggered_scenario_id is not None:
            newly_recovering = should_trigger_recovery & ~self._in_recovery_mode
            if newly_recovering.any():
                logging.info(
                    f"Failure detected for scenario {triggered_scenario_id}, triggering recovery for {newly_recovering.sum()} environments"
                )
                self._in_recovery_mode |= newly_recovering
                self._current_recovery_scenario = triggered_scenario_id

                # Initialize recovery skill with ground operator
                assert (
                    triggered_scenario_id in self._recovery_skills
                ), f"Recovery skill not initialized for scenario {triggered_scenario_id}"

                # NOTE: Allow recovery skill to be used with objects beyond training
                # (Relational)
                assert (
                    self._current_operator is not None
                ), "Current operator must exist when triggering recovery"
                curr_object_ids = self._failured_det_operator_objects[
                    self._current_operator.parent.name
                ]
                objects = []
                for obj_id in curr_object_ids:
                    obj = self._current_operator.parameters[obj_id]
                    objects.append(obj)

                train_objects = self.learning_info[triggered_scenario_id].get(
                    "train_objects", ""
                )
                assert train_objects, "train_objects not specified in learning_info"
                region_name = train_objects.split(",")[-1]
                all_objects = self.system.perceiver.objects
                region_obj = all_objects.as_dict().get(region_name)
                assert region_obj is not None, f"Region object {region_name} not found"

                objects.append(region_obj)
                parameter_tuple = tuple(objects)

                # Ground the recovery operator
                recovery_skill = self._recovery_skills[triggered_scenario_id]
                recovery_operator = self._recovery_operators[triggered_scenario_id]
                recovery_ground_op = recovery_operator.ground(parameter_tuple)
                self._recovery_ground_operators[triggered_scenario_id] = (
                    recovery_ground_op
                )

                # Reset the recovery skill
                recovery_skill.reset(recovery_ground_op, obs)

        return normal_result

    def _execute_recovery_step(
        self, obs: Tensor, info: dict[str, Any]
    ) -> ApproachStepResult:
        """Execute one step of the recovery skill.

        Args:
            obs: Current observation
            info: Environment info dict

        Returns:
            ApproachStepResult with recovery action
        """
        assert (
            self._current_recovery_scenario is not None
        ), "Current recovery scenario not set"
        assert (
            self._current_recovery_scenario in self._recovery_skills
        ), f"Recovery skill not initialized for scenario {self._current_recovery_scenario}"
        assert (
            self._recovery_ground_operators[self._current_recovery_scenario] is not None
        ), f"Recovery ground operator not set for scenario {self._current_recovery_scenario}"

        # Get the recovery skill and operator for current scenario
        recovery_skill = self._recovery_skills[self._current_recovery_scenario]
        recovery_ground_operator = self._recovery_ground_operators[
            self._current_recovery_scenario
        ]

        # Execute the recovery skill
        skill_action = recovery_skill.get_action(obs)

        # Check if skill has terminated
        skill_terminated = recovery_skill.terminate(obs)

        # If all environments have terminated, replan
        if skill_terminated.all():
            logging.info(
                f"Recovery skill for scenario {self._current_recovery_scenario} terminated, replanning..."
            )
            self._in_recovery_mode = torch.zeros_like(self._in_recovery_mode)
            self._recovery_ground_operators[self._current_recovery_scenario] = None
            self._current_recovery_scenario = None

            # Replan from current state
            try:
                return self.reset(obs, info)
            except TaskThenMotionPlanningFailure as e:
                logging.warning(f"Replanning after recovery failed: {e}")
                # Return frozen action if replanning fails
                frozen_action = self._last_action.clone()
                self._last_action = frozen_action
                return ApproachStepResult(
                    _action=frozen_action, op=self._current_operator
                )

        # Update last action
        self._last_action = skill_action.clone()

        # Return with recovery operator
        return ApproachStepResult(_action=skill_action, op=recovery_ground_operator)

    def create_new_operator_skill(
        self,
        policy: Policy,
        operator: LiftedOperator,
        operator_name: str,
        scenario_id: int | None = None,
    ) -> LiftedOperatorSkill:
        """Create a recovery skill with failure detection capability.

        This method creates a skill that can detect when it should be triggered based on
        collision detection and operator matching.

        Args:
            policy: The policy to use for this skill
            operator: The lifted operator for this skill
            operator_name: Name of the operator
            scenario_id: The scenario ID this skill belongs to (defaults to curr_learning_phase)

        Returns:
            A LiftedOperatorSkill with failure detection capability
        """
        if scenario_id is None:
            scenario_id = self.curr_learning_phase

        scenario_info = self.learning_info[scenario_id]
        assert operator_name in scenario_info.get(
            "tgt_skill", ""
        ), "Operator name does not match target skill"
        # Create a new skill for the operator
        max_rl_steps = scenario_info.get("max_rl_steps", 20)
        rl_static_steps = scenario_info.get("rl_static_steps", 3)
        tamp_system = self.system
        # NOTE: The following has bs=1 when created, but will be updated during reset
        # where it can be executed in batched environments.
        action_space_low = self.action_space_low
        action_space_high = self.action_space_high
        arm_action_low = self._arm_action_low
        arm_action_high = self._arm_action_high
        normalize_action = self._normalize_action
        fallback_action = self._fallback_action

        # Capture failure detector for this scenario
        failure_detector = self._failure_detectors.get(scenario_id, None)

        class RecoverySkill(LiftedOperatorSkill):
            """A skill that uses a policy to execute a lifted operator."""

            _step: int = 0
            _batch_size: int = 1
            _post_rl_static_steps: Tensor = torch.zeros(1, dtype=torch.bool)
            _last_action: Tensor = torch.tensor(fallback_action)
            _terminated: Tensor = torch.zeros(1, dtype=torch.bool)
            _policy: Policy = policy

            def __init__(  # pylint: disable=super-init-not-called,no-member
                self,
                env=self.system.env.unwrapped,
                operators=self.operators,
            ) -> None:
                """Initialize the RL-based lifted operator skill.

                This function exists just for keeping the same signature as other
                skills.
                """
                del env, operators

            def get_lifted_operator(self, op=operator) -> LiftedOperator:
                """Return the lifted operator for this skill."""
                return op

            def get_action_given_objects(
                self,
                objects: Sequence[Object],
                obs: Tensor,
                act_low=action_space_low,
                act_high=action_space_high,
                max_rl_steps=max_rl_steps,
            ) -> Tensor:
                """Defines an object-parameterized policy."""
                assert (
                    self._current_plan is not None
                ), "Current plan is not initialized."
                self._step += 1

                if self._terminated.any():
                    # If already terminated, return frozen action
                    frozen_action = get_frozen_action(
                        self._last_action[self._terminated],
                        arm_action_low,
                        arm_action_high,
                        normalize_action,
                        CFG.control_mode,
                        obs=obs[self._terminated],
                    )
                    self._post_rl_static_steps[self._terminated] += 1
                else:
                    # dummy
                    frozen_action = torch.zeros(
                        (0, self._last_action.shape[1]),
                        dtype=self._last_action.dtype,
                        device=obs.device,
                    )

                if (
                    self._step <= max_rl_steps + CFG.pre_rl_stabilize_steps
                    and self._step > CFG.pre_rl_stabilize_steps
                ):
                    # first clip the observation
                    sub_obs = tamp_system.state_to_vec(
                        obs,
                        objects,
                    )
                    policy_action = self._policy.get_action(sub_obs).to(
                        self._last_action.dtype
                    )
                    clipped_action = torch.clamp(
                        policy_action,
                        act_low,
                        act_high,
                    )
                elif self._step <= CFG.pre_rl_stabilize_steps:
                    # During pre-rl stabilize steps, just use zero actions
                    # NOTE: This is not necessarily frozen action.
                    rl_static_actions = torch.zeros_like(self._last_action)
                    if not CFG.delta_finger_control:
                        rl_static_actions[:, -1] = self._last_action[:, -1]
                    clipped_action = rl_static_actions.clone()
                else:
                    # We handel static steps in side the skill
                    # so that the planning execution loop is not affected.
                    clipped_action = get_frozen_action(
                        self._last_action,
                        arm_action_low,
                        arm_action_high,
                        normalize_action,
                        CFG.control_mode,
                        obs=obs,
                    )
                clipped_action[self._terminated] = frozen_action
                self._current_plan[0] = clipped_action
                self._last_action = clipped_action.clone()
                return self._current_plan.pop(0)

            def get_operator_name(self, op=operator) -> str:
                """Return the lifted operator name for this skill."""
                return op.name

            def terminate_with_objects(
                self,
                objects: Sequence[Object],
                obs: Tensor,
            ) -> Tensor:
                """Just terminate after max steps for now."""
                assert (
                    self._current_plan is not None
                ), "Current plan is not initialized."
                # first clip the observation
                sub_obs = tamp_system.state_to_vec(
                    obs,
                    objects,
                )
                # ignore objects if your policy doesn't need them
                if hasattr(self._policy, "terminate"):
                    policy_terminal = cast(
                        torch.Tensor, self._policy.terminate(sub_obs)
                    )
                else:
                    policy_terminal = torch.zeros(
                        obs.shape[0], dtype=torch.bool, device=obs.device
                    )
                not_have_current_plan = len(self._current_plan) == 0
                if not_have_current_plan:
                    # Exhausted, terminate immediately
                    return torch.ones_like(obs[:, 0], dtype=torch.bool)

                self._terminated |= policy_terminal

                # Stablize for a few more steps
                static_steps_ok = self._post_rl_static_steps >= rl_static_steps
                actual_terminate = self._terminated & static_steps_ok
                return actual_terminate

            def reset(
                self,
                ground_operator: GroundOperator,
                obs: Tensor,
                max_rl_steps=max_rl_steps,
                rl_static_steps=rl_static_steps,
            ) -> None:
                assert hasattr(tamp_system.env, "single_action_space")
                self._batch_size = obs.shape[0]
                base_action = tamp_system.env.single_action_space.sample()
                if len(base_action.shape) == 1:
                    base_action = base_action[np.newaxis, :]
                else:
                    assert len(base_action.shape) == 2
                rand_actions = torch.tensor(
                    tamp_system.env.single_action_space.sample(), device=CFG.device
                ).repeat(self._batch_size, 1)
                # _current_plan is protected, but we’re explicit here
                # One more static steps before RL starts
                self._current_plan = [rand_actions] * (
                    max_rl_steps + rl_static_steps + CFG.pre_rl_stabilize_steps
                )
                assert self.can_execute(ground_operator)
                self.current_ground_operator = ground_operator  # type: ignore # pylint: disable=protected-access
                self._last_action = rand_actions.clone()
                self._step = 0
                self._post_rl_static_steps = torch.zeros(
                    rand_actions.shape[0], device=rand_actions.device
                )
                self._terminated = torch.zeros(
                    rand_actions.shape[0], dtype=torch.bool, device=rand_actions.device
                )
                self._policy.reset(obs)

            def initiates_on_failure(
                self,
                obs: Tensor,
                current_ground_operator: GroundOperator | None,
            ) -> Tensor:
                """Detect if this recovery skill should be triggered based on failure.

                Uses a trained neural classifier if available, otherwise returns False.

                Args:
                    obs: Current observation [batch_size, obs_dim]
                    _info: Environment info dict (unused, for interface compatibility)
                    current_ground_operator: Currently executing ground operator

                Returns:
                    Boolean tensor indicating which environments should trigger recovery
                """
                # If no current operator, no recovery needed
                if current_ground_operator is None:
                    return torch.zeros(
                        obs.shape[0], dtype=torch.bool, device=obs.device
                    )

                # Use neural failure detector if available
                if failure_detector is not None:
                    return failure_detector(obs)

                # No detector available, return False (no recovery)
                return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)

        return RecoverySkill()
