"""An RL approach that uses planner reached states as the initial states and jointly
learns the policy and the initiation set."""

from __future__ import annotations

import copy
import itertools
import logging
import os
import pickle
import re
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import torch
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from relational_structs import (
    GroundAtom,
    Object,
    PDDLDomain,
    Variable,
)
from torch import Tensor

from skill_refactor.approaches.base import (
    ApproachStepResult,
    BaseApproach,
    BaseRLTAMPSystem,
)
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.approaches.rl_policies.base import Policy
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.benchmarks.wrappers import (
    ManiSkillBaseEnv,
    MultiEnvWrapper,
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import (
    get_frozen_action,
    get_normalize_action_range,
)
from skill_refactor.utils.graph import PlanningGraph
from skill_refactor.utils.structs import (
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    LowLevelTrajectory,
    PlannerDataset,
    Predicate,
    RLDataset,
)
from skill_refactor.utils.task_planning import (
    create_task_planning_heuristic,
    task_plan,
    task_plan_grounding,
)
from skill_refactor.utils.ttmp import (
    TaskThenMotionPlanner,
    TaskThenMotionPlanningFailure,
)


class LifelongRefApproach(BaseApproach):
    """General improvisational TAMP approach.

    This approach combines task-and-motion planning with learned policies for creating
    shortcuts between non-adjacent nodes in the plan.
    """

    def __init__(
        self,
        system: BaseRLTAMPSystem,
        seed: int,
        planner_id: str = "pyperplan",
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.planner_id = planner_id
        (
            self._normalize_action,
            self._arm_action_low,
            self._arm_action_high,
        ) = get_normalize_action_range(system.env, CFG.control_mode)

        # Get domain
        self.domain = system.get_domain()

        # Initialize lll containers
        # Operators and skills are memory that solves more and more tasks
        # We actually don't need the predicates here.
        self.perceiver = system.perceiver
        self.operators = copy.deepcopy(system.operators)
        self.skills = copy.deepcopy(system.skills)
        self.init_operator_names = [op.name for op in self.operators]
        # active operators are used to learn new predicates and skills
        self.new_operators: LiftedOperator | None = None
        # Training lifted plans loaded from dataset (used for plan validation)
        self.training_lifted_plans: List[List[LiftedOperator]] = []

        # For planning
        self._current_operator: GroundOperator | None = None
        self._current_skill: LiftedOperatorSkill | None = None
        self._skill_reached_effects: Tensor = torch.zeros(1, dtype=torch.bool)
        self._skill_exhausted: Tensor = torch.zeros(1, dtype=torch.bool)
        self._max_skill_steps: int = 100
        fallback_action = system.env.single_action_space.sample()  # type: ignore
        self._fallback_action = fallback_action
        self._last_action: Tensor = torch.tensor(fallback_action)
        self._goal: set[GroundAtom] = set()

        # For RL
        self.action_space_low, self.action_space_high = torch.from_numpy(
            system.env.single_action_space.low  # type: ignore[attr-defined]
        ).to(
            system.env.device  # type: ignore[attr-defined]
        ), torch.from_numpy(  # type: ignore[attr-defined]
            system.env.single_action_space.high  # type: ignore[attr-defined]
        ).to(  # type: ignore[attr-defined]
            system.env.device  # type: ignore[attr-defined]
        )

        # Graph-based planning state
        self.visited_lifted_plans: dict[str, set[list[LiftedOperator]]] = {}
        self.planning_graph: PlanningGraph | None = None
        self._goal_atoms: set[GroundAtom] = set()
        # all visited low-level states
        self.visited_states: dict[Tuple[int, int], list[Tensor]] = {}
        self.rng = np.random.default_rng(seed)
        # high-level life learning information from human/llm
        self.curr_learning_phase: int = 0
        self.potential_comp_nodes: List[int] = []
        self.learning_info: dict[int, dict] = {}

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "rl_planning_states"

    def get_invented_predicate_interpretr_so_far(self) -> Dict[Predicate, Callable]:
        """Get all invented predicate interpreters so far."""
        invented_pred_interpr: Dict[Predicate, Callable] = {}
        for perceiver_pred, interpr in self.perceiver.predicate_interpreters.items():
            if perceiver_pred not in self.system.perceiver.predicate_interpreters:
                invented_pred_interpr[perceiver_pred] = interpr
        return invented_pred_interpr

    def update_domain(
        self,
        new_operators: Set[LiftedOperator],
        new_skills: Set[LiftedOperatorSkill],
    ) -> None:
        """Update the current domain with new operators and skills."""
        self.operators = copy.deepcopy(new_operators)
        self.skills = copy.deepcopy(new_skills)
        self.domain = PDDLDomain(
            f"Domain_Scenario_{self.curr_learning_phase}",
            self.operators,  # type: ignore[arg-type]
            self.perceiver.predicates_container.as_set(),
            self.system.components.type_container.as_set(),
        )

    def update_learning_info(
        self,
        episode: int,
        learning_info: dict,
        latest_tamp_system: Optional[BaseRLTAMPSystem] = None,
    ) -> None:
        """Update learning information for the approach."""
        self.learning_info[episode] = learning_info
        self.curr_learning_phase = episode
        # Update the max steps during data collection
        self._max_skill_steps = learning_info.get("max_skill_steps", 100)
        # Update the domain with a new empty operator
        skill_str = learning_info.get("tgt_skill", "")
        skill_name = skill_str.split("(")[0].strip()
        m = re.match(r"^[^(]+\((.*?)\)\s*$", skill_str)
        if not m:
            raise ValueError(f"String does not match expected pattern: {skill_str!r}")
        inner = m.group(1)
        # Split on commas and strip whitespace
        types_str = [tok.strip() for tok in inner.split(",") if tok.strip()]
        basic_type_str = learning_info.get("basic_type")
        assert (
            basic_type_str in types_str
        ), "Basic type must be one of the operator types"
        operator_var = []
        curr_comp_nodes = []
        for type_str in types_str:
            # Get the type from the system's type container
            if type_str not in self.system.components.type_container.as_dict():
                raise ValueError(
                    f"Type {type_str} not found in system's type container."
                )
            pddl_type = self.system.components.type_container.as_dict()[type_str]
            if type_str == basic_type_str:
                for (
                    obj,
                    node_id,
                ) in self.system.perceiver.objects.object_to_node.items():
                    if obj.type == pddl_type:
                        curr_comp_nodes.append(node_id)

            pddl_var = Variable(name=f"?{type_str}", type=pddl_type)
            operator_var.append(pddl_var)
        # Create a new operator with the given types
        self.potential_comp_nodes = curr_comp_nodes
        new_operator = LiftedOperator(
            name=skill_name,
            parameters=operator_var,
            preconditions=set(),
            add_effects=set(),
            delete_effects=set(),
        )
        self.new_operator = new_operator
        # NOTE: For now we assume skills are only useful for recover from
        # the failed op when it was learned.
        self.failed_op: str = learning_info["failed_skill"]
        self.failed_objects: list[str] = learning_info["failed_objects"].split(",")
        self.basic_objects_sc: list[str] = learning_info["basic_objects"].split(",")
        if latest_tamp_system is not None:
            # Optionally update the TAMP system
            self.system = latest_tamp_system

    def create_new_operator_skill(
        self,
        policy: Policy,
        operator: LiftedOperator,
        operator_name: str,
    ) -> LiftedOperatorSkill:
        """Create a new skill for the operator using the given RL policy."""
        scenario_info = self.learning_info[self.curr_learning_phase]
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

        class RLiftedOpSkill(LiftedOperatorSkill):
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

        return RLiftedOpSkill()

    def update_domain_knowledge(
        self,
        learning_info: dict,
    ) -> None:
        """Update domain knowledge with new predicates, operators, and skills.

        This method encapsulates the workflow of:
        1. Creating and loading a policy from configuration
        2. Loading predicate configurations and datasets
        3. Using TopDownPredicateLearner to invent new predicates and operators
        4. Creating new skills for target operators
        5. Refactoring existing skills with new operators
        6. Updating the domain with new operators and skills

        Args:
            learning_info: Dictionary containing learning configuration with keys:
                - planner_learning_cfg_settings: Config for predicate learning
                    - predicate_config: Path to predicate YAML config
                    - dataset_path: Path to trajectory dataset
                    - num_traj: Number of trajectories to load
                    - rl_config: Path to RL config
                    - pre_trained_policy_path: Path to pre-trained policy weights

        Returns:
            The created and loaded policy instance.

        Note:
            The returned policy must be initialized with an appropriate environment
            before it can be used for training or inference.
        """
        # Extract planner learning settings
        planner_cfg = learning_info.get("planner_learning_cfg_settings", {})

        # Create and load policy from config
        # NOTE: This happens before predicate update.
        # And this is how the policy was learned
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

        # Load predicate configurations
        predicate_config_path = planner_cfg.get("predicate_config")
        if predicate_config_path is None:
            raise ValueError("predicate_config must be specified in learning_info")

        with open(predicate_config_path, "rb") as f:
            config_data = yaml.safe_load(f)
        predicate_configures = config_data["predicates"]

        # Load trajectory dataset
        dataset_path = Path(os.path.join(CFG.planner_dataset_path, f"seed_{CFG.seed}"))
        assert dataset_path.is_dir(), "Dataset path is invalid"
        num_traj = planner_cfg.get("planner_num_traj", -1)
        planner_dataset = PlannerDataset.load(dataset_path, num_traj=num_traj)

        # Load training lifted operator plans for plan validation
        lifted_plans_path = dataset_path / "lifted_operator_plans.pkl"
        assert lifted_plans_path.exists(), "Lifted plans path is invalid"
        with open(lifted_plans_path, "rb") as f:
            self.training_lifted_plans = pickle.load(f)
        logging.info(
            f"Loaded {len(self.training_lifted_plans)} training lifted operator plans"
        )

        # Use TopDownPredicateLearner to invent predicates and operators
        quantify_basic = planner_cfg.get("quantify_basic", True)
        pred_learner = TopDownPredicateLearner(
            dataset=planner_dataset,
            tamp_system=self.system,
            predicate_configures=predicate_configures,
            latest_perceiver=self.perceiver,
            quantify_basic_predicates=quantify_basic,
            verbose=True,
            scenario=str(self.curr_learning_phase),
        )

        invented_pred_interpr, op_set = pred_learner.invent()

        # Update the perceiver with invented predicates
        for pred, interp in invented_pred_interpr.items():
            type_list = pred.types
            base_types = pred.base_types
            self.perceiver.add_predicate_interpreter(
                pred.name, type_list, interp, base_types
            )

        # Create terminal predicate interpreters
        terminal_interp_dict = pred_learner.create_terminal_pred_interpretrs()

        # Deep copy existing skills and get their operator names
        old_skills = copy.deepcopy(self.skills)
        # NOTE: Always create an empty new set to avoid mutation issues
        new_skills = set()
        old_skill_operator_names = [skill.get_operator_name() for skill in old_skills]

        # Create new skill for target operator
        tgt_skill_op_name = None
        for operator in op_set:
            if operator.name in learning_info["tgt_skill"]:
                # Add RL-based new skill
                new_skill = self.create_new_operator_skill(
                    policy=policy,
                    operator=operator,
                    operator_name=operator.name,
                )
                new_skills.add(new_skill)
                tgt_skill_op_name = operator.name
                break

        if tgt_skill_op_name is None:
            raise ValueError(
                f"Target skill {learning_info['tgt_skill']} not found in operator set"
            )

        # Refactor existing skills with new operators
        for operator in op_set:
            # Check if this operator is a variant of an existing skill
            operator_base_name = operator.name.split(f"_{tgt_skill_op_name}")[0]
            if operator_base_name in old_skill_operator_names:
                # Case 1: It is from an existing skill
                # Initiate a new skill from the existing skill, but with a new operator name
                existing_skill = next(
                    skill
                    for skill in old_skills
                    if skill.get_operator_name() == operator_base_name
                )
                if isinstance(self.system.env.unwrapped, ManiSkillBaseEnv):
                    new_skill = existing_skill.__class__(  # type: ignore[call-arg]
                        env=self.system.env.unwrapped,  # type: ignore[index]
                        operators=op_set,
                    )
                else:
                    assert isinstance(self.system.env, MultiEnvWrapper)
                    new_skill = existing_skill.__class__(  # type: ignore[call-arg]
                        env=self.system.env.unwrapped[0],  # type: ignore[index]
                        operators=op_set,
                    )

                # Deep copy so we don't accidentally mutate the original
                local_op = copy.deepcopy(operator)

                # Capture local_op in the default arg to avoid late binding issues
                def _custom_get_operator_name(self, op=local_op) -> str:
                    del self
                    return op.name

                def _custom_get_lifted_operator(self, op=local_op) -> LiftedOperator:
                    del self
                    return op

                # Dynamically bind new methods to the skill
                new_skill.get_operator_name = types.MethodType(  # type: ignore[method-assign]
                    _custom_get_operator_name, new_skill
                )
                new_skill.get_lifted_operator = types.MethodType(  # type: ignore[method-assign]
                    _custom_get_lifted_operator, new_skill
                )

                # Add customized termination function if there is a learned terminal predicate
                terminal_predicate_name = f"terminal_{operator.name.lower()}"
                if terminal_predicate_name in terminal_interp_dict:
                    logging.info(
                        "using learned terminal predicate for operator %s",
                        operator.name,
                    )
                    terminal_interp = terminal_interp_dict[terminal_predicate_name]

                    def _custom_terminate_with_objects(
                        self,
                        objects: Sequence[Object],
                        obs: Tensor,
                        term_interp=terminal_interp,
                    ) -> Tensor:
                        assert (
                            self._current_plan
                            is not None  # pylint: disable=protected-access
                        )
                        # NOTE: Predicate cls predicates BxNx1, we want Bx1 here
                        pred_atoms = term_interp(obs, [objects]).squeeze(1)
                        not_have_current_plan = (
                            len(self._current_plan)
                            == 0  # pylint: disable=protected-access
                        )
                        not_have_current_plan_tensor = (
                            torch.ones_like(pred_atoms, dtype=torch.bool)
                            if not_have_current_plan
                            else torch.zeros_like(pred_atoms, dtype=torch.bool)
                        )
                        # Existing skills terminate immediately on collision
                        colliding = obs[:, -1].clone().bool()
                        return pred_atoms | not_have_current_plan_tensor | colliding

                    new_skill.terminate_with_objects = types.MethodType(  # type: ignore[method-assign]
                        _custom_terminate_with_objects, new_skill
                    )

                new_skills.add(new_skill)
            else:
                assert operator.name == tgt_skill_op_name, "Unexpected operator name"
        # Update domain with new operators and skills
        self.update_domain(op_set, new_skills)

    def reset(
        self,
        obs: Tensor,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Reset approach with initial observation."""
        # First reset perceiver to get current objects in this test episode
        # Use float32
        obs = obs.to(torch.float32)
        bs = obs.shape[0]
        task_plans = []

        if bs > 1:
            for b in range(bs - 1):
                objects, atoms, goal = self.perceiver.reset(obs[b : b + 1], info)
                batch_task_plan = self._create_task_plan(objects, atoms, goal)
                if len(batch_task_plan) > 0:
                    task_plans.append(batch_task_plan)
                if len(batch_task_plan) > 0:
                    break
        else:
            objects, atoms, goal = self.perceiver.reset(obs, info)
            batch_task_plan = self._create_task_plan(objects, atoms, goal)
            if len(batch_task_plan) > 0:
                task_plans.append(batch_task_plan)
            task_plans.append(batch_task_plan)
        self._last_action = (
            torch.tensor(self._fallback_action)
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(obs.device)
        )
        if len(task_plans[0]) == 0:
            # will freeze at default fall back action
            raise TaskThenMotionPlanningFailure(
                "No valid task plan found during reset."
            )

        objects, atoms, goal = self.perceiver.reset(obs[0:1], info)  # type: ignore
        self._goal = goal

        # graph, plans = self._create_planning_graph(objects, atoms)
        # Compute task plan
        self._current_task_plan = task_plans[0]
        # Reset state
        self._current_operator = None
        self._current_skill = None
        self._last_action = (
            torch.tensor(self._fallback_action, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(obs.device)
        )
        self._skill_reached_effects = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        # If some environments exhaust the skill, freeze them.
        self._skill_exhausted = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )

        return self.step(obs, 0.0, False, False, info)

    @property
    def exhausted(self) -> Tensor:
        """Get exhausted mask."""
        return self._skill_exhausted.clone()

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation.

        This is identical to how abstract planning (ttmp.py) works. We re-implement it
        here to allow for customization if needed.
        """
        del reward, terminated, truncated, info  # Unused in this approach

        obs = obs.to(torch.float32)
        # NOTE: We only step perceiver then the skill terminates.
        if self._current_skill is not None and self._current_skill.terminate(obs).any():
            terminate_mask = self._current_skill.terminate(obs)
            terminate_mask &= ~self._skill_reached_effects
            if terminate_mask.any():
                # All terminated envs have reached the effects.
                # pass perceiver step.
                terminate_obs = obs[terminate_mask]
                terminate_atoms = self.perceiver.step(terminate_obs)

                # new, assuming atoms: List[Set[GroundAtom]]
                assert self._current_operator is not None
                add_ok_list = [
                    self._current_operator.add_effects.issubset(atom_set)
                    for atom_set in terminate_atoms
                ]
                del_ok_list = [
                    not any(self._current_operator.delete_effects & atom_set)
                    for atom_set in terminate_atoms
                ]

                # Update skill reached effects
                self._skill_reached_effects[terminate_mask] |= torch.logical_and(
                    torch.tensor(add_ok_list), torch.tensor(del_ok_list)
                ).to(obs.device)
                self._skill_exhausted[terminate_mask] |= ~self._skill_reached_effects[
                    terminate_mask
                ]

        # Switch conditions:
        # If the current operator is None
        # or terminated/exhausted in all envs.
        # NOTE: A skill has to be terminiated to trigger reach effects or exhaust.
        # So we don't need to check for termination here.
        switch_condition = torch.all(
            self._skill_reached_effects | self._skill_exhausted
        )
        if self._current_skill is None or switch_condition:
            # If there is no more plan to execute, freeze.
            if not self._current_task_plan:
                assert self._last_action is not None
                frozen_action = get_frozen_action(
                    self._last_action,
                    self._arm_action_low,
                    self._arm_action_high,
                    self._normalize_action,
                    CFG.control_mode,
                    obs=obs,
                )
                return ApproachStepResult(
                    _action=frozen_action, op=self._current_operator
                )

            self._current_operator = self._current_task_plan.pop(0)
            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill(self._current_operator)
            assert self._current_skill is not None, "No skill for operator"
            self._current_skill.reset(self._current_operator, obs)
            self._skill_reached_effects = torch.zeros(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
            # NOTE: We do not reset exhausted here, because
            # exhausted envs should stay exhausted.

        assert self._current_skill is not None
        skill_action = self._current_skill.get_action(obs)
        # Freeze the skill if it has reached the desired effects.
        # (Or if it is exhausted.)
        frozen_action_reached_eff = get_frozen_action(
            skill_action[self._skill_reached_effects],
            self._arm_action_low,
            self._arm_action_high,
            self._normalize_action,
            CFG.control_mode,
            obs=obs[self._skill_reached_effects],
        )
        frozen_action_exhausted = get_frozen_action(
            self._last_action[self._skill_exhausted],
            self._arm_action_low,
            self._arm_action_high,
            self._normalize_action,
            CFG.control_mode,
            obs=obs[self._skill_exhausted],
        )
        frozen_action_exhausted = frozen_action_exhausted.to(skill_action.dtype)
        skill_action[self._skill_reached_effects] = frozen_action_reached_eff
        skill_action[self._skill_exhausted] = frozen_action_exhausted
        self._last_action = skill_action.clone()
        return ApproachStepResult(_action=skill_action, op=self._current_operator)

    def step_policy_with_scenario(
        self,
        obs: Tensor,
        policy: Policy,
        static_action: Tensor | None = None,
    ) -> tuple[ApproachStepResult, torch.Tensor]:
        """Create a new skill for the operator using the given RL policy."""
        assert self.new_operator is not None, "No new operator to create skill for"
        scenario_info = self.learning_info[self.curr_learning_phase]
        # Create a new skill for the operator
        terminal = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        train_objects = scenario_info.get("train_objects", "").split(",")
        operator_parameters = []
        for obj in train_objects:
            operator_parameters.append(self.perceiver.objects[obj])

        parameter_tuple = tuple(operator_parameters)
        if static_action is not None:
            assert static_action.shape[0] == obs.shape[0]
            return (
                ApproachStepResult(
                    _action=static_action,
                    op=self.new_operator.ground(parameter_tuple),
                ),
                terminal,
            )

        sub_obs = self.system.state_to_vec(
            obs,
            operator_parameters,
        )
        policy_action = policy.get_action(sub_obs)
        clipped_action = torch.clamp(
            policy_action,
            self.action_space_low,
            self.action_space_high,
        )
        terminal = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        if hasattr(policy, "terminate"):
            terminal = cast(torch.Tensor, policy.terminate(sub_obs))

        return (
            ApproachStepResult(
                _action=clipped_action,
                op=self.new_operator.ground(parameter_tuple),
            ),
            terminal,
        )

    def get_init_state_sampler_from_provided(
        self, provided_init_states: List[Tensor], device: torch.device
    ) -> Tuple[Callable, Tensor, List[Tuple[int, ...]]]:
        """Create an initial state sampler from provided initial states.

        Analyzes provided initial states and builds relative state distributions
        for each scenario (pre/post), then creates a sampler that can generate
        diverse initial states for data collection.

        Args:
            provided_init_states: List of initial states to analyze
            device: Device to place tensors on

        Returns:
            relative_state_sampler: Callable that samples relative states
            basic_node: Tensor of basic object node IDs [num_obstructions]
            pre_post_list: List of (scenario_id, pre/post) tuples for sampling
        """
        # NOTE: Obstructions select basic nodes.
        basic_node = (
            torch.zeros(3, dtype=torch.long, device=device) - 100
        )  # [num_obstructions] - 100

        # First analyse provided initial states
        # This can also be acquired by executing the plan and get the colliding objects.
        relative_states: dict[str, List[Tensor]] = {}
        for scenario in list(self.learning_info.keys()):
            relative_states[f"sc{scenario}_pre"] = []
            relative_states[f"sc{scenario}_post"] = []

        for init_state in provided_init_states:
            for scenario in list(self.learning_info.keys()):
                # Get the object-relative state
                basic_object_name = self.basic_objects_sc[scenario - 1]
                relative_state_w_id = self.system.full_state_to_relative_state(
                    init_state, basic_object_name
                )
                relative_states[f"sc{scenario}_pre"].append(
                    relative_state_w_id[:, scenario - 1, 1:].clone()
                )
                relative_states[f"sc{scenario}_post"].append(
                    relative_state_w_id[:, scenario - 1, 1:].clone()
                )
                basic_node[scenario - 1] = relative_state_w_id[
                    0, scenario - 1, 0
                ].long()

        relative_state_sampler = self._gen_relative_dist_sampler(relative_states)

        # Extract pre/post choices and node IDs from scenario combo
        # target_scenario_combo is like: ((0, 5), (1, 3), (0, 5))
        pre_post_list: List[Tuple[int, ...]] = [
            (sc_id, 0) for sc_id in list(self.learning_info.keys())
        ]

        return relative_state_sampler, basic_node, pre_post_list

    def sample_init_states_from_sampler(
        self,
        envs: ManiSkillVectorEnv,
        env_reset_init: Tensor,
        relative_state_sampler: Callable,
        basic_node: Tensor,
        pre_post_list: List[Tuple[int, ...]],
    ) -> Tensor:
        """Sample initial states using the relative state sampler.

        Args:
            envs: Vectorized environment
            env_reset_init: Initial reset states from envs.reset()
            relative_state_sampler: Sampler created by get_init_state_sampler_from_provided
            basic_node: Basic object node IDs
            pre_post_list: List of (scenario_id, pre/post) tuples

        Returns:
            init_state_sampled: Sampled initial states [num_envs, state_dim]
        """
        init_state_sampled = torch.zeros_like(env_reset_init)
        # Replace with sampled initial states
        for i in range(envs.num_envs):
            sampled_rel_state_data = relative_state_sampler(pre_post_list).to(
                init_state_sampled.device
            )  # [num_obstructions, state_dim]
            sampled_rel_state = torch.cat(
                [
                    basic_node.unsqueeze(1),
                    sampled_rel_state_data,
                ],
                dim=1,
            )  # [num_obstructions, state_dim + 1]
            # NOTE: Create the batch dim here.
            sampled_rel_state = sampled_rel_state.unsqueeze(0)
            init_state_sampled[i] = self.system.relative_state_to_full_state(
                env_reset_init[i : i + 1], sampled_rel_state
            )
        return init_state_sampled

    def collect_rl_data(
        self, envs: ManiSkillVectorEnv, provided_init_states: List[Tensor]
    ) -> RLDataset:
        """Collect RL training data from the environment, use planning failures."""
        logging.info(
            f"Collecting RL training data for scenario {self.curr_learning_phase}\n"
            f"Target skill: {self.learning_info[self.curr_learning_phase].get('tgt_skill', '')}\n"
        )
        logging.info(
            f"Collecting {CFG.num_train_episodes_rl} episodes from {envs.num_envs} environments..."
        )

        # Create initial state sampler from provided states
        relative_state_sampler, basic_node, pre_post_list = (
            self.get_init_state_sampler_from_provided(provided_init_states, envs.device)
        )

        training_states = (
            []
        )  # Last state for pre-skill, where the desired effects are not achieved.
        current_atoms_list = []
        goal_atoms_list = (
            []
        )  # The missing atoms that should be True after the pre-skill.

        episode = 0

        while episode < CFG.num_train_episodes_rl:
            logging.info(
                f"Collecting episode {episode + 1}/{CFG.num_train_episodes_rl}"
            )
            # Sample num_env environments initial states
            # wil be replaced to avoid introducing new tasks.
            env_reset_init, _ = envs.reset()
            init_state_sampled = self.sample_init_states_from_sampler(
                envs, env_reset_init, relative_state_sampler, basic_node, pre_post_list
            )
            obs, _ = envs.reset(options={"init_state": init_state_sampled})
            previous_obs = obs.clone()
            # obtain objects and initial atoms, for plan parsing
            try:
                approach_result = self.reset(obs, {})
            except TaskThenMotionPlanningFailure as e:
                logging.info(f"Initial planning failed for episode {episode + 1}: {e}")
                continue
            if approach_result.op is None:
                logging.info(f"No operator found for episode {episode + 1}")
                continue
            curr_op_name = approach_result.op.parent.name
            curr_op_param_names = [obj.name for obj in approach_result.op.parameters]

            # One state per environment
            episode_training_states = torch.zeros_like(obs)
            episode_current_atoms: List[set] = [set() for _ in range(envs.num_envs)]
            episode_goal_atoms: List[frozenset] = [
                frozenset() for _ in range(envs.num_envs)
            ]
            colliding = torch.zeros(envs.num_envs, dtype=torch.bool, device=obs.device)

            # Execute the task plan and collect initial states for target skill training
            for i in range(self._max_skill_steps):
                # Get the next action from the approach
                action = approach_result.action
                obs, _, _, _, infos = envs.step(action)
                # detect collisions with obstacles
                # NOTE: only collect states when the failed operator is being executed
                if (
                    any(infos["is_colliding"])
                    and (curr_op_name == self.failed_op)
                    and all(
                        obj_name == param_name
                        for obj_name, param_name in zip(
                            self.failed_objects,
                            curr_op_param_names,
                        )
                    )
                ):
                    colliding_masks = infos["is_colliding"].to(torch.bool)
                    # NOTE: Only collect the states that are newly colliding
                    # and have not triggered exhaustion yet.
                    newly = colliding_masks & ~colliding & ~self.exhausted
                    episode_training_states[newly] = previous_obs[newly]
                    colliding[newly] = True
                # if break early
                if torch.all(torch.any(episode_training_states, dim=1)) or torch.all(
                    colliding | self.exhausted
                ):
                    logging.info(
                        f"Episode {episode + 1} finished early at step {i + 1} with collisions."
                    )
                    break
                previous_obs = obs.clone()
                approach_result = self.step(obs, 0.0, False, False, {})
                assert approach_result.op is not None, "Planning failed mid-episode."
                curr_op_name = approach_result.op.parent.name
                curr_op_param_names = [
                    obj.name for obj in approach_result.op.parameters
                ]

            # Store the episode data
            has_new_data = False
            for b in range(envs.num_envs):
                # if planning fails to reach pre-condition, skip the data.
                if torch.any(episode_training_states[b]):
                    training_states.append(episode_training_states[b])
                    current_atoms_list.append(episode_current_atoms[b])
                    goal_atoms_list.append(episode_goal_atoms[b])
                    has_new_data = True
            if has_new_data:
                episode += 1
            else:
                logging.info(
                    f"No valid training data collected in episode {episode + 1}, retrying..."
                )

            logging.info(
                f"Collected {len(training_states)} initial states for target skill."
            )

        return RLDataset(
            states=training_states,
            current_atoms=current_atoms_list,
            goal_atoms=goal_atoms_list,
            config={},
        )

    def collect_planner_data(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        provided_init_states: List[Tensor],
        real_env_scenario_name: str = "sc1_pre_n1_sc2_pre_n1",
    ) -> PlannerDataset:
        """Collect Planner training data from the environment, use RL rollouts with
        replay mechanism to avoid mid-episode resets.

        Args:
            envs: The vectorized environment
            policy: The policy to use for RL skill execution
            provided_init_states: Initial states to create the sampler from
            real_env_scenario_name: The scenario name for real environment trajectories
                                    (must be one of the valid scenario combinations)

        The agent collects two kinds of data for planner training:
        1) Data from the hard tasks where the RL skill is learned.
        2) Compositionally derived data from the observed high-level state transitions in 1).
        """
        logging.info("Collecting training data with planner\n")

        assert CFG.num_envs == 1, "Only support single env for planner data collection"
        num_obstructions = len(CFG.scenario.split(","))
        # Get potential composition nodes
        assert self.potential_comp_nodes, "No potential composition nodes available"
        num_comp_nodes = len(self.potential_comp_nodes)

        # Total scenarios: (2 pre/post choices × num_comp_nodes) ^ num_obstructions
        total_scenarios = (2 * num_comp_nodes) ** num_obstructions

        # Create initial state sampler from provided states
        relative_state_sampler_real, basic_node_real, pre_post_list_real = (
            self.get_init_state_sampler_from_provided(provided_init_states, envs.device)
        )

        episode = 0
        self.failed_id = 0
        relative_state_sampler_dream = None
        trajectories: List[LowLevelTrajectory] = []
        relative_states: Dict[str, List[Tensor]] = {}
        rl_operator_names: List[str] = []
        all_scenarios_collected: List[bool] = [False] * total_scenarios
        desired_scenario_number = CFG.num_train_episodes_planner // total_scenarios
        num_scenarios_collected: List[int] = [0] * total_scenarios
        num_initial_attempts: List[int] = [0] * total_scenarios
        scenario_proven_possible: List[bool] = [False] * total_scenarios
        scenario_proven_invalid: List[bool] = [False] * total_scenarios
        num_failed_collected: List[int] = [0] * total_scenarios
        desired_failed_per_scenario = CFG.num_failed_episodes_planner // total_scenarios
        all_scenario_names: List[str] = []
        # We collect relative states before and after RL skill execution
        # NOTE: Here each tensor contains only one obstruction's relative states
        # based on scenario id.
        for scenario in list(self.learning_info.keys()):
            relative_states[f"sc{scenario}_pre"] = []
            relative_states[f"sc{scenario}_post"] = []
            rl_op_name = (
                self.learning_info[scenario].get("tgt_skill", "").split("(")[0].strip()
            )
            rl_operator_names.append(rl_op_name)

        # Build scenario combinations including node IDs
        # Each scenario_combo is a tuple of (pre_post, node_id) pairs
        # Example: ((0, 0, 5), (1, 1, 3)) means obstruction 0 uses pre+node_5, obstruction 1 uses post+node_3
        all_scenario_combos: List[Tuple[Tuple[int, int, int], ...]] = []

        # Create all combinations of (pre/post, node_id) for each obstruction
        obstruction_choices = []
        for scenario in list(self.learning_info.keys()):
            # For each obstruction: [(sc, pre/post, node_id), ...]
            obstruction_choice = []
            for pre_post in [0, 1]:  # 0=pre, 1=post
                for node_id in self.potential_comp_nodes:
                    obstruction_choice.append((scenario, pre_post, node_id))
            obstruction_choices.append(obstruction_choice)

        # Generate all scenario combinations
        for combo in itertools.product(*obstruction_choices):
            # combo is like: ((0, 0, 5), (1, 1, 3), (2, 0, 5))
            all_scenario_combos.append(combo)
            # Create human-readable name
            scenario_name_parts = []
            for scenario, pre_post, node_id in combo:
                pre_post_str = "pre" if pre_post == 0 else "post"
                scenario_name_parts.append(f"sc{scenario}_{pre_post_str}_n{node_id}")
            scenario_name = "_".join(scenario_name_parts)
            all_scenario_names.append(scenario_name)

        known_invalid_scenarios = (
            self.learning_info[scenario].get("known_invalid_sc", "").split(";")
        )
        for invalid_sc in known_invalid_scenarios:
            if invalid_sc in all_scenario_names:
                idx = all_scenario_names.index(invalid_sc)
                scenario_proven_invalid[idx] = True
                all_scenarios_collected[idx] = True
                logging.info(
                    f"Marking known invalid scenario '{invalid_sc}' as collected."
                )

        rl_skill_name = self.learning_info[self.curr_learning_phase].get(
            "tgt_skill", ""
        )
        assert rl_operator_names[-1] in rl_skill_name, "RL skill name mismatch"
        logging.info(f"Using {rl_skill_name} policy for RL skill execution.")
        logging.info(
            f"Target: {desired_scenario_number} trajectories per scenario "
            f"({total_scenarios} total scenarios = (2*{num_comp_nodes})^{num_obstructions})"
        )

        # Verify that real_env_scenario_name is valid and get its index
        assert (
            real_env_scenario_name in all_scenario_names
        ), f"real_env_scenario_name '{real_env_scenario_name}' not in valid scenarios: {all_scenario_names}"
        real_env_scenario_idx = all_scenario_names.index(real_env_scenario_name)
        logging.info(
            f"Real environment trajectories will be collected as scenario: {real_env_scenario_name} "
            f"(index {real_env_scenario_idx})"
        )

        # Loop until we collect enough real environment data and all scenarios are collected
        while episode < CFG.num_train_episodes_planner_before_dreaming or not all(
            all_scenarios_collected
        ):
            logging.info(f"Collecting episode {episode + 1}")

            # Track which scenario is being targeted
            target_scenario_idx = None
            target_scenario_name = None

            if episode < CFG.num_train_episodes_planner_before_dreaming:
                logging.info("Collecting data from real environment...")
                # Set scenario tracking for real environment episodes
                target_scenario_idx = real_env_scenario_idx
                target_scenario_name = real_env_scenario_name
                # Sample initial state instead of using default reset
                env_reset_init, _ = envs.reset()
                init_state_sampled = self.sample_init_states_from_sampler(
                    envs,
                    env_reset_init,
                    relative_state_sampler_real,
                    basic_node_real,
                    pre_post_list_real,
                )
                init_obs, _ = envs.reset(options={"init_state": init_state_sampled})
                states_steps, actions_steps, ops_steps, success = (
                    self.collect_planner_data_with_init_state(
                        envs,
                        policy,
                        init_obs,
                        allow_rl=True,
                    )
                )
                # Store relative states before after rl-op skill for dreaming
                if success:
                    curr_op = ops_steps[0]
                    # Skip if no valid operator in trajectory
                    if curr_op is None:
                        episode += 1
                        continue
                    last_op_state_w_id = self.system.full_state_to_relative_state(
                        states_steps[0], curr_op.parameters[1].name
                    )
                    for step, ground_op in enumerate(ops_steps):
                        # Only save states when operator changes
                        # (High-level transitions)
                        # State is before the action is taken

                        # NOTE: We have droped the last state in traj
                        # here, as it is not useful for current domains.
                        s = states_steps[step]
                        if ground_op != curr_op:
                            # Skip if current operator is None
                            if curr_op is None:
                                curr_op = ground_op
                                continue
                            curr_op_state_w_id = (
                                self.system.full_state_to_relative_state(
                                    s, curr_op.parameters[1].name
                                )
                            )  # B x num_obstructions x state_dim
                            # If after RL skill/operator execution
                            if curr_op is not None and (
                                curr_op.parent.name in rl_operator_names
                            ):
                                scenario_idx = rl_operator_names.index(
                                    curr_op.parent.name
                                )
                                scenario = list(self.learning_info.keys())[scenario_idx]
                                # NOTE: The order of relative states is fixed regardless of
                                # which scenario comes first, it is always ordered by sc1, sc2, ...
                                # so we use scenario-1 as index, instead of scenario_idx (order of
                                # appearance in env settings)
                                # Assert all node IDs are in potential_comp_nodes
                                assert torch.isin(
                                    curr_op_state_w_id[:, scenario - 1, 0],
                                    torch.tensor(
                                        self.potential_comp_nodes,
                                        device=curr_op_state_w_id.device,
                                    ),
                                ).all(), "All node IDs must be in potential_comp_nodes"
                                assert curr_op_state_w_id[:, scenario - 1, 1:].any()
                                relative_states[f"sc{scenario}_pre"].append(
                                    last_op_state_w_id[:, scenario - 1, 1:].clone()
                                )
                                relative_states[f"sc{scenario}_post"].append(
                                    curr_op_state_w_id[:, scenario - 1, 1:].clone()
                                )
                            curr_op = ground_op
                            last_op_state_w_id = curr_op_state_w_id
            else:
                logging.info("Collecting data from dreaming...")
                init_obs, _ = envs.reset()
                if relative_state_sampler_dream is None:
                    # Assume two components for now
                    relative_state_sampler_dream = self._gen_relative_dist_sampler(
                        relative_states
                    )

                # Choose which scenario to target (pick one that needs more samples)
                # Prioritize scenarios with fewer collected trajectories
                incomplete_scenarios = [
                    i
                    for i, collected in enumerate(all_scenarios_collected)
                    if not collected
                ]
                if not incomplete_scenarios:
                    logging.info("All scenarios collected!")
                    break

                # Choose scenario with minimum count among incomplete ones
                target_scenario_idx = min(
                    incomplete_scenarios, key=lambda i: num_scenarios_collected[i]
                )
                target_scenario_combo = all_scenario_combos[target_scenario_idx]
                target_scenario_name = all_scenario_names[target_scenario_idx]

                logging.info(
                    f"Targeting scenario {target_scenario_idx}: {target_scenario_name} "
                    f"({num_scenarios_collected[target_scenario_idx]}/{desired_scenario_number})"
                )

                # Extract pre/post choices and node IDs from scenario combo
                # target_scenario_combo is like: ((0, 5), (1, 3), (0, 5))
                pre_post_list: List[Tuple[int, ...]] = [
                    (sc_id,) for sc_id in list(self.learning_info.keys())
                ]
                # NOTE: Obstructions select basic nodes.
                basic_node = (
                    torch.zeros(3, dtype=torch.long, device=init_obs.device) - 100
                )  # [num_obstructions] - 100
                for encounter_id, (sc_id, pre_post, node_id) in enumerate(
                    target_scenario_combo
                ):
                    # NOTE: sc1 will be stored at index 0
                    # even if it might be the second novelty
                    pre_post_list[encounter_id] = pre_post_list[encounter_id] + (
                        pre_post,
                    )
                    basic_node[sc_id - 1] = node_id

                # Sample an initial relative state for the target scenario
                dreamed_rel_state_data = relative_state_sampler_dream(pre_post_list).to(
                    init_obs.device
                )  # [num_obstructions, state_dim]

                # Determine allow_rl from the last component (new skill)
                # 0=pre means we're before RL, so allow_rl=True
                # 1=post means we're after RL, so allow_rl=False
                # NOTE: pre_post_tuple is ordered by encountered order, not sc1, sc2, ...
                allow_rl = pre_post_list[-1][1] == 0

                dreamed_rel_state = torch.cat(
                    [
                        basic_node.unsqueeze(1),
                        dreamed_rel_state_data,
                    ],
                    dim=1,
                )  # [num_obstructions, state_dim + 1]
                # NOTE: Create the batch dim here.
                dreamed_rel_state = dreamed_rel_state.unsqueeze(0)
                dreamed_full_state = self.system.relative_state_to_full_state(
                    init_obs, dreamed_rel_state
                )
                try:
                    # Verify that the dreamed state has no collisions
                    _, _ = envs.reset(options={"init_state": dreamed_full_state})
                except ValueError as e:
                    logging.error(
                        f"Dreamed initial state for scenario {target_scenario_name} has collisions: {e}"
                    )
                    # Mark this scenario as completed and invalid without collecting any trajectories
                    scenario_proven_invalid[target_scenario_idx] = True
                    all_scenarios_collected[target_scenario_idx] = True
                    logging.warning(
                        f"Marking scenario {target_scenario_idx} ({target_scenario_name}) as INVALID and COMPLETED "
                        f"due to collision in dreamed initial state. No trajectories will be collected for this scenario."
                    )
                    continue
                try:
                    states_steps, actions_steps, ops_steps, success = (
                        self.collect_planner_data_with_init_state(
                            envs,
                            policy,
                            dreamed_full_state,
                            allow_rl=allow_rl,
                        )
                    )
                except AssertionError as e:
                    logging.error(
                        f"Dreaming episode for scenario {target_scenario_name} failed due to error: {e}"
                    )
                    continue

            # Build trajectories from collected data
            # Determine if we should collect this trajectory based on success and scenario validity
            should_collect = False
            assert target_scenario_idx is not None
            # For all episodes (both real environment and dreaming)
            if success and not scenario_proven_invalid[target_scenario_idx]:
                # Collect successful trajectories for valid scenarios
                should_collect = True
            elif not success and scenario_proven_invalid[target_scenario_idx]:
                # Collect failed trajectories for invalid scenarios
                if (
                    num_failed_collected[target_scenario_idx]
                    < desired_failed_per_scenario
                ):
                    should_collect = True

            if should_collect:
                states_tensor = torch.stack(states_steps, dim=1)  # [B, S, *obs_shape]
                if len(actions_steps) == 0:
                    # Some failed episodes may have no actions collected
                    actions_tensor = torch.empty(
                        states_tensor.shape[0],
                        0,
                        self.action_space_low.shape[0],
                        device=states_tensor.device,
                    )
                else:
                    actions_tensor = torch.stack(
                        actions_steps, dim=1
                    )  # [B, A, *act_shape]

                env_states = states_tensor.unbind(dim=0)
                env_actions = actions_tensor.unbind(dim=0)

                for _, (s, act) in enumerate(zip(env_states, env_actions)):
                    assert (
                        target_scenario_name is not None
                    ), "target_scenario_name must be set"
                    trajectories.append(
                        LowLevelTrajectory(
                            _states=list(s.unbind(0)),
                            _actions=[
                                ApproachStepResult(_action=a, op=o)
                                for a, o in zip(act.unbind(0), ops_steps)
                            ],
                            _train_scenario_idx=target_scenario_name,
                            _success=success,
                            _train_task_idx=episode,
                        )
                    )
                    episode += 1

                # Update scenario tracking (applies to both real environment and dreaming episodes)
                if success:
                    num_scenarios_collected[target_scenario_idx] += 1
                    # Mark scenario as proven possible (never give up on it)
                    scenario_proven_possible[target_scenario_idx] = True
                    if (
                        num_scenarios_collected[target_scenario_idx]
                        >= desired_scenario_number
                    ):
                        all_scenarios_collected[target_scenario_idx] = True
                        logging.info(
                            f"Scenario {target_scenario_idx} ({target_scenario_name}) "
                            f"reached target: {num_scenarios_collected[target_scenario_idx]}/{desired_scenario_number} successful trajectories"
                        )
                else:
                    # Track failed trajectory collection for invalid scenarios
                    num_failed_collected[target_scenario_idx] += 1
                    if (
                        num_failed_collected[target_scenario_idx]
                        >= desired_failed_per_scenario
                    ):
                        all_scenarios_collected[target_scenario_idx] = True
                        logging.info(
                            f"Scenario {target_scenario_idx} ({target_scenario_name}) "
                            f"reached target: {num_failed_collected[target_scenario_idx]}/{desired_failed_per_scenario} failed trajectories"
                        )

                logging.info(
                    f"Collected {len(trajectories)} trajectories for planner refactorization."
                )
                logging.info(
                    f"Progress: {sum(all_scenarios_collected)}/{len(all_scenarios_collected)} "
                    f"scenarios completed"
                )
                # # Visualize
                # if episode < 640:
                #     _, _ = envs.reset(options={"init_state": states_steps[0]})
                #     import imageio.v2 as iio

                #     iio.imwrite(
                #         f"debug/planner_data_episode_{episode}_success.png",
                #         envs.render(),
                #     )
                #     logging.info(
                #         f"Episode image saved to planner_data_episode_{episode}_success.png"
                #     )
            else:
                logging.info(f"Episode failed")
                # # Visualize
                # if episode < 640:
                #     _, _ = envs.reset(options={"init_state": states_steps[0]})
                #     import imageio.v2 as iio

                #     iio.imwrite(
                #         f"debug/planner_data_episode_{episode}_failed{self.failed_id}.png",
                #         envs.render(),
                #     )
                #     logging.info(
                #         f"Episode image saved to planner_data_episode_{episode}_failed{self.failed_id}.png"
                #     )
                self.failed_id += 1
                # Update failure counter for dreaming episodes
                if target_scenario_idx is not None:
                    # Only count attempts if scenario hasn't been proven possible yet
                    if not scenario_proven_possible[target_scenario_idx]:
                        num_initial_attempts[target_scenario_idx] += 1
                        logging.info(
                            f"Scenario {target_scenario_idx} ({target_scenario_name}) "
                            f"initial attempts: {num_initial_attempts[target_scenario_idx]}/{CFG.max_consecutive_failures}"
                        )
                        # Mark as invalid if scenario fails in all initial attempts
                        if (
                            num_initial_attempts[target_scenario_idx]
                            >= CFG.max_consecutive_failures
                        ):
                            scenario_proven_invalid[target_scenario_idx] = True
                            logging.warning(
                                f"Marking scenario {target_scenario_idx} ({target_scenario_name}) as INVALID "
                                f"after {CFG.max_consecutive_failures} failed initial attempts (never succeeded). "
                                f"Will now collect {desired_failed_per_scenario} failed trajectories."
                            )
                    else:
                        # Scenario has succeeded before, so keep trying
                        logging.info(
                            f"Scenario {target_scenario_idx} ({target_scenario_name}) failed but is proven possible. "
                            f"Will keep trying. Collected {num_scenarios_collected[target_scenario_idx]}/{desired_scenario_number} trajectories."
                        )

        # Verification: Check that we collected the expected trajectories
        logging.info("\n%s", "=" * 80)
        logging.info("FINAL VERIFICATION OF COLLECTED TRAJECTORIES")
        logging.info("%s", "=" * 80)

        # Count trajectories by scenario and success status
        scenario_counts: Dict[str, Dict[str, int]] = {}
        for traj in trajectories:
            scenario_name = traj.train_scenario_idx
            if scenario_name not in scenario_counts:
                scenario_counts[scenario_name] = {"success": 0, "failed": 0}
            if traj.success:
                scenario_counts[scenario_name]["success"] += 1
            else:
                scenario_counts[scenario_name]["failed"] += 1

        # Verify each scenario
        verification_passed = True
        for scenario_idx, scenario_name in enumerate(all_scenario_names):
            is_valid = scenario_proven_possible[scenario_idx]
            is_invalid = scenario_proven_invalid[scenario_idx]

            actual_success = scenario_counts.get(scenario_name, {}).get("success", 0)
            actual_failed = scenario_counts.get(scenario_name, {}).get("failed", 0)

            if is_valid:
                expected_success = num_scenarios_collected[scenario_idx]
                expected_failed = 0
                logging.info(
                    f"Scenario {scenario_idx} ({scenario_name}): VALID - "
                    f"Expected {expected_success} successful, 0 failed | "
                    f"Actual: {actual_success} successful, {actual_failed} failed"
                )
                if actual_success != expected_success or actual_failed != 0:
                    logging.error(f"  ✗ MISMATCH for scenario {scenario_name}!")
                    verification_passed = False
                else:
                    logging.info(f"  ✓ Verified")
            elif is_invalid:
                expected_success = 0
                expected_failed = desired_failed_per_scenario
                logging.info(
                    f"Scenario {scenario_idx} ({scenario_name}): INVALID - "
                    f"Expected 0 successful, {expected_failed} failed | "
                    f"Actual: {actual_success} successful, {actual_failed} failed"
                )
                if actual_success != 0 or actual_failed != expected_failed:
                    logging.error(f"  ✗ MISMATCH for scenario {scenario_name}!")
                    verification_passed = False
                else:
                    logging.info(f"  ✓ Verified")
            else:
                # Scenario was neither proven valid nor invalid (shouldn't happen if logic is correct)
                logging.warning(
                    f"Scenario {scenario_idx} ({scenario_name}): UNKNOWN STATE - "
                    f"Actual: {actual_success} successful, {actual_failed} failed"
                )

        logging.info("%s", "=" * 80)
        if verification_passed:
            logging.info("✓ ALL SCENARIOS VERIFIED SUCCESSFULLY")
        else:
            logging.error(
                "✗ VERIFICATION FAILED - Some scenarios have mismatched counts"
            )
        logging.info("Total trajectories collected: %d", len(trajectories))
        logging.info("%s\n", "=" * 80)

        # Extract scenario indices for PlannerDataset
        training_scenario_indices = [traj.train_scenario_idx for traj in trajectories]

        return PlannerDataset(
            _trajectories=trajectories,
            _training_scenario_idx=training_scenario_indices,
        )

    def collect_planner_data_with_init_state(
        self,
        envs: ManiSkillVectorEnv,
        policy: Policy,
        init_state: Tensor,
        allow_rl: bool = True,
    ) -> Tuple[List[Tensor], List[Tensor], List[Optional[GroundOperator]], bool]:
        """Collect Planner training data from the environment, starting from init_state.

        Returns:
            states: List of state tensors collected during execution.
            actions: List of action tensors collected during execution.
            operators: List of GroundOperators executed.
            success: Whether the task was successfully completed.
        """
        assert CFG.num_envs == 1, "Only support single env for planner data collection"
        # Phase 1: Planning phase with replay bag tracking collision
        obs, _ = envs.reset(options={"init_state": init_state})
        init_obs = obs.clone()  # Save initial observation for replay
        try:
            res = self.reset(obs, {})
        except TaskThenMotionPlanningFailure:
            logging.info("Initial planning failed, skipping episode.")
            return [init_obs], [], [], False

        colliding = False
        # Replay bag to store actions and collision states
        # List of (action, operator, colliding_mask) tuples
        replay_bag: List[Tuple[Tensor, GroundOperator | None, bool]] = []
        first_run_states_steps = [obs.clone()]
        step_count = 0
        rl_trigger_step = None  # Track when RL should start

        # Planning phase - track actions in replay bag
        while step_count < CFG.max_env_steps:
            step_count += 1
            action = res.action  # [B, *act_shape]

            # Apply frozen actions for colliding environments
            obs, _, _, _, infos = envs.step(action)
            first_run_states_steps.append(obs.clone())
            colliding = infos["is_colliding"].to(torch.bool)[0].item()

            # Store in replay bag
            replay_bag.append((action.clone(), res.op, colliding))

            # Check RL trigger condition
            # colliding, failed operator, failed objects match
            if (
                (rl_trigger_step is None)
                and colliding
                and (res.op is not None)
                and (res.op.parent.name == self.failed_op)
                and all(
                    obj_name == param_name
                    for obj_name, param_name in zip(
                        self.failed_objects,
                        [obj.name for obj in res.op.parameters],
                    )
                )
            ):
                rl_trigger_step = len(replay_bag) - 1
                break

            res = self.step(obs, 0.0, False, False, {})

            if infos["success"].sum():
                logging.info(
                    f"Episode succeeded early at step {step_count} without collision/new skill rl."
                )
                break

        # Phase 2: Data collection phase with replay (and maybe RL execution)
        step_count = 0
        # Reset environment to initial state for clean data collection
        obs, _ = envs.reset(options={"init_state": init_obs})

        # Data collection buffers
        states_steps = [obs.clone()]
        actions_steps = []
        ops_steps = []
        success = False
        if rl_trigger_step is not None:
            # Replay planning phase actions
            for i, (replay_action, replay_op, _) in enumerate(replay_bag):
                if i >= rl_trigger_step:
                    break
                step_count += 1
                actions_steps.append(replay_action.clone())
                ops_steps.append(replay_op)
                obs, _, _, _, infos = envs.step(replay_action)
                success |= infos["success"].to(torch.bool)
                states_steps.append(obs.clone())

            # Execute RL skill if allowed
            if not allow_rl:
                logging.info(
                    f"RL execution not allowed in this episode. " f"Directly fail."
                )
                return states_steps, actions_steps, ops_steps, False

            curr_op = replay_bag[rl_trigger_step - 1][1]  # Get the failed operator
            assert curr_op is not None, "Expected operator to be set"
            if curr_op.name != self.failed_op:
                logging.info(
                    f"Episode RL trigger operator {curr_op.name} "
                    f"does not match expected failed operator {self.failed_op}. "
                    f"Directly fail."
                )
                return states_steps, actions_steps, ops_steps, False

            failure_op = LiftedOperator(
                name=f"{curr_op.name}_{self.new_operator.name}",
                parameters=self.new_operator.parameters,  # NOTE: Assume same parameters
                preconditions=set(),
                add_effects=set(),
                delete_effects=set(),
            )
            scenario_info = self.learning_info[self.curr_learning_phase]
            train_objects = scenario_info.get("train_objects", "").split(",")
            max_rl_steps = scenario_info.get("max_rl_steps", 20)
            rl_static_steps = scenario_info.get("rl_static_steps", 3)
            operator_parameters = []
            for obj in train_objects:
                operator_parameters.append(self.perceiver.objects[obj])

            parameter_tuple = tuple(operator_parameters)
            ground_failed_op = failure_op.ground(parameter_tuple)

            for k in range(len(ops_steps)):
                if ops_steps[k] == curr_op:
                    # Replace the failed operator with the failure operator
                    ops_steps[k] = ground_failed_op

            # RL execution
            # RL static steps
            rl_static_actions = torch.zeros_like(actions_steps[-1])
            if not CFG.delta_finger_control:
                rl_static_actions[:, -1] = actions_steps[-1][:, -1]

            # Pre-RL stablize steps
            for _ in range(CFG.pre_rl_stabilize_steps):
                res, _ = self.step_policy_with_scenario(obs, policy, rl_static_actions)
                step_count += 1
                rl_action = res.action
                actions_steps.append(rl_action.clone())
                ops_steps.append(res.op)
                obs, _, _, _, infos = envs.step(rl_action)
                success |= infos["success"].to(torch.bool)
                states_steps.append(obs.clone())

            for rl_step in range(max_rl_steps):
                res, terminal = self.step_policy_with_scenario(obs, policy)
                if terminal.all():
                    logging.info(
                        f"Episode RL skill terminated early at step {rl_step}."
                    )
                    break
                step_count += 1
                rl_action = res.action
                actions_steps.append(rl_action.clone())
                ops_steps.append(res.op)
                obs, _, _, _, infos = envs.step(rl_action)
                success |= infos["success"].to(torch.bool)
                states_steps.append(obs.clone())

            for _ in range(rl_static_steps):
                res, _ = self.step_policy_with_scenario(obs, policy, rl_static_actions)
                step_count += 1
                rl_action = res.action
                actions_steps.append(rl_action.clone())
                ops_steps.append(res.op)
                obs, _, _, _, infos = envs.step(rl_action)
                success |= infos["success"].to(torch.bool)
                states_steps.append(obs.clone())

            # Continue with remaining planning steps
            try:
                res = self.reset(obs, infos)
            except TaskThenMotionPlanningFailure:
                logging.info("Initial planning failed, skipping episode.")
                return [init_obs], [], [], False
            # NOTE: Relabel RL static steps as the first operator action
            for k in range(len(ops_steps) - rl_static_steps, len(ops_steps)):
                ops_steps[k] = res.op
            curr_num_replay = len(actions_steps)
            for _ in range(CFG.max_env_steps - curr_num_replay):
                step_count += 1
                actions_steps.append(res.action)
                ops_steps.append(res.op)
                obs, _, _, _, infos = envs.step(res.action)
                success = infos["success"].to(torch.bool)[0].item()
                states_steps.append(obs.clone())
                res = self.step(obs, 0.0, False, False, {})
                if success:
                    logging.info(
                        f"Episode finished early at step {step_count} with RL executed."
                    )
                    break
        else:
            logging.info(f"Episode did not trigger new skill execution.")
            if allow_rl:
                logging.info(
                    f"Should have triggered RL skill but did not. Directly fail."
                )
                return states_steps, actions_steps, ops_steps, False
            if infos["success"].sum():
                # No need to replay, simply save first_run states
                # NOTE: Maniskills environment can't replay actions to get exact obs
                for i, (replay_action, replay_op, _) in enumerate(replay_bag):
                    step_count += 1
                    actions_steps.append(replay_action.clone())
                    ops_steps.append(replay_op)
                logging.info(
                    f"Episode finished early at step {step_count} with success."
                )
                return first_run_states_steps, actions_steps, ops_steps, True
            else:
                logging.info(
                    f"Episode finished early at step {step_count} with failure."
                )
                return states_steps, actions_steps, ops_steps, False

        return states_steps, actions_steps, ops_steps, success

    def _is_plan_valid_against_training_skeletons(
        self, candidate_plan: list[GroundOperator]
    ) -> bool:
        """Check if a candidate plan matches or is a subsequence of training plans.

        A plan is valid if its lifted operator sequence either:
        1. Exactly matches a training plan skeleton, or
        2. Is a subsequence of a training plan skeleton

        Args:
            candidate_plan: The ground operator plan to validate

        Returns:
            True if the plan is valid according to training skeletons, False otherwise
        """
        # If no training plans loaded, consider all plans valid
        if not self.training_lifted_plans:
            return True

        # Extract lifted operator sequence from candidate plan
        candidate_lifted = [op.parent for op in candidate_plan]

        # Check against each training skeleton
        for training_skeleton in self.training_lifted_plans:
            # Check for exact match
            if len(candidate_lifted) == len(training_skeleton):
                if all(
                    c_op.name == t_op.name
                    for c_op, t_op in zip(candidate_lifted, training_skeleton)
                ):
                    return True

            # Check if candidate is a subsequence of training skeleton
            if len(candidate_lifted) <= len(training_skeleton):
                # Try to find candidate as subsequence
                c_idx = 0
                for t_op in training_skeleton:
                    if c_idx < len(candidate_lifted):
                        if candidate_lifted[c_idx].name == t_op.name:
                            c_idx += 1
                if c_idx == len(candidate_lifted):
                    # Found all operators in sequence
                    return True

        return False

    def _create_task_plan(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> list[GroundOperator]:
        """Create task plan with local search to achieve goal.

        Biases towards plans that revisit previously seen abstract states earlier.
        """
        ground_operators, reachable_atoms = task_plan_grounding(
            init_atoms, objects, list(self.operators), allow_noops=True
        )
        heuristic = create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal,
            ground_operators,
            self.perceiver.predicates_container.as_set(),
            objects,
        )
        generator = task_plan(
            init_atoms,
            goal,
            ground_operators,
            reachable_atoms,
            heuristic,
            CFG.seed,
            CFG.task_planning_timeout,
            CFG.pred_search_max_skeletons_optimized,
        )

        # Step 1: Collect all shortest plans
        shortest_plans: list[tuple[list[GroundOperator], list[set[GroundAtom]]]] = []
        shortest_length: int | None = None

        while True:
            try:
                result = next(generator, None)
                if result is None:
                    break
                plan, atom_seq, _ = result

                plan_length = len(plan)

                # Initialize shortest length with first plan
                if shortest_length is None:
                    shortest_length = plan_length
                    shortest_plans.append((plan, atom_seq))
                elif plan_length == shortest_length:
                    # Same length as shortest, keep it
                    shortest_plans.append((plan, atom_seq))
                elif plan_length > shortest_length:
                    # Found a longer plan, stop collecting
                    break
            except AssertionError:
                # Generator exhausted or max skeletons reached
                break

        if not shortest_plans:
            return []

        # Step 2: If only one shortest plan, return it
        if len(shortest_plans) == 1:
            logging.info("Single shortest plan found")
            return shortest_plans[0][0]

        # Step 3: Multiple shortest plans - filter by training skeletons
        logging.info(f"Multiple shortest plans found: {len(shortest_plans)}")

        # Filter plans that match or are subsequences of training skeletons
        valid_plans = [
            (plan, atom_seq)
            for plan, atom_seq in shortest_plans
            if self._is_plan_valid_against_training_skeletons(plan)
        ]

        if valid_plans:
            logging.info(
                f"Found {len(valid_plans)} valid plans matching training skeletons"
            )
            # Return the first valid plan
            return valid_plans[0][0]
        else:
            logging.warning(
                "No plans match training skeletons, returning first shortest plan"
            )
            # Fall back to first shortest plan if none match training skeletons
            return shortest_plans[0][0]

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

    def _get_skill(self, operator: GroundOperator) -> LiftedOperatorSkill:
        """Get skill that can execute operator, but override RL operator to call
        self.policy."""
        # 1. find an example skill
        skills = [s for s in self.skills if s.can_execute(operator)]
        base_skill = skills[0]
        # fallback to the normal skill
        return base_skill
