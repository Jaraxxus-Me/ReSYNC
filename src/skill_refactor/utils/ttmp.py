"""Planning interface. Borrowed from Tom's task_then_motion_planning. The key difference
here is that the step function here accepts a batched observation and returns a batched
action, allowing for parallel execution of multiple environments.

Right now we assume that the plans for different environments are homogeneous at high-
level, i.e., the single task plan applies to all environments. And skill execution is
also homogeneous, i.e., the same skill is executed in all environments at all times.
"""

import abc
import logging
from typing import Any, List, Tuple, cast

import torch
from numpy.typing import NDArray
from relational_structs import (
    PDDLDomain,
    PDDLProblem,
    Type,
)
from torch import Tensor

from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action
from skill_refactor.utils.structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    Object,
    Perceiver,
    Predicate,
    Task,
)
from skill_refactor.utils.task_planning import (
    create_task_planning_heuristic,
    task_plan,
    task_plan_grounding,
)


class TaskThenMotionPlanningFailure(Exception):
    """Raised when task then motion planning fails."""


class TaskThenMotionPlanner(abc.ABC):
    """Run task then motion planning with greedy execution."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        perceiver: Perceiver,
        operators: set[LiftedOperator],
        skills: set[LiftedOperatorSkill],
        fallback_action: NDArray,
        normalize_action: bool,
        arm_action_low: Tensor,
        arm_action_high: Tensor,
        planner_id: str = "fd-sat",
        domain_name: str = "ttmp-domain",
    ) -> None:
        self._types = types
        self._predicates = predicates
        self._perceiver = perceiver
        self._operators = operators
        self._skills = skills
        self._planner_id = planner_id
        self._domain_name = domain_name
        self._domain = PDDLDomain(
            self._domain_name, self._operators, self._predicates, self._types
        )
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._last_operator: GroundOperator | None = None
        self._last_operator_reached_effects: Tensor = torch.zeros(1, dtype=torch.bool)
        self._current_operator: GroundOperator | None = None
        self._current_skill: LiftedOperatorSkill | None = None
        self._fallback_action = fallback_action
        self._normalize_action = normalize_action
        self._arm_action_low = arm_action_low
        self._arm_action_high = arm_action_high
        self._last_action: Tensor = torch.tensor(fallback_action)
        self._skill_reached_effects: Tensor = torch.zeros(1, dtype=torch.bool)
        self._skill_exhausted: Tensor = torch.zeros(1, dtype=torch.bool)

    def generate_task(self, obs: Tensor, info: dict[str, Any]) -> Task:
        """Generate a task from observation and the current perceiver."""
        assert obs.shape[0] == 1, "Task generation only supports single environment"
        objects, _, goal = self._perceiver.reset(obs, info)
        return Task(obs.cpu(), objects, goal, 0)

    def reset(self, obs: Tensor, info: dict[str, Any]) -> None:
        """Reset on a new task instance."""
        # We commit to the first successful symbolic plan across the batch.
        # This is a limitation of the current implementation.
        task_plans = []
        bs = obs.shape[0]
        if bs > 1:
            for b in range(bs - 1):
                objects, atoms, goal = self._perceiver.reset(obs[b : b + 1], info)
                batch_task_plan = self._create_task_plan(objects, atoms, goal)
                task_plans.append(batch_task_plan)
                if len(batch_task_plan) > 0:
                    break
        else:
            objects, atoms, goal = self._perceiver.reset(obs, info)
            batch_task_plan = self._create_task_plan(objects, atoms, goal)
            task_plans.append(batch_task_plan)
        self._last_action = (
            torch.tensor(self._fallback_action)
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(obs.device)
        )
        if len(task_plans) == 0:
            # will freeze at default fall back action
            self._current_task_plan = []
            self._current_operator = None
            self._current_skill = None
            return
        self._current_task_plan = task_plans[0]
        self._current_operator = None
        self._last_operator = None
        self._last_operator_reached_effects = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        self._current_skill = None
        # If some environments reach desired effects early, freeze them.
        self._skill_reached_effects = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        # If some environments exhaust the skill, freeze them.
        self._skill_exhausted = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )

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
            init_atoms, objects, list(self._operators), allow_noops=True
        )
        heuristic = create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal,
            ground_operators,
            self._perceiver.predicates_container.as_set(),
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

        if len(shortest_plans) > 1:
            logging.warning(
                f"Multiple shortest plans found of length {shortest_length}. "
                "Biasing towards plans that operate on 'obj1' first."
            )

            # Score each plan: lower is better
            # Plans where obj1 appears earlier get lower (better) scores
            def score_plan(plan: list[GroundOperator]) -> float:
                """Score a plan based on when obj1 is used.

                Lower is better.
                """
                for idx, op in enumerate(plan):
                    # Check if any object in this operator is named "obj1"
                    if any("obj1" in obj.name for obj in op.parameters):
                        # Return the position where obj1 first appears
                        # Earlier positions get lower scores
                        return float(idx)
                # If obj1 never appears, return worst score
                return float(len(plan))

            # Sort plans by score (ascending), so obj1-first plans come first
            shortest_plans.sort(key=lambda x: score_plan(x[0]))

        return shortest_plans[0][0]

    def step(self, obs: Tensor) -> Tuple[Tensor | None, GroundOperator | None]:
        """Get an action to execute."""
        # NOTE: We only step perceiver then the skill terminates.
        if self._current_skill is not None and self._current_skill.terminate(obs).any():
            terminate_mask = self._current_skill.terminate(obs)
            terminate_mask &= ~self._skill_reached_effects
            if terminate_mask.any():
                # Perceive the atoms at termination to evaluate effects/exhaustion.
                terminate_obs = obs[terminate_mask]
                terminate_atoms = self._perceiver.step(terminate_obs)

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
            # If there is no more task plan to execute, fail (freeze to last action).
            if not self._current_task_plan:
                actions = get_frozen_action(
                    self._last_action,
                    self._arm_action_low,
                    self._arm_action_high,
                    self._normalize_action,
                    CFG.control_mode,
                    obs=obs,
                )
                return actions, self._current_operator

            self._last_operator = (
                self._current_operator
                if self._current_operator
                else self._current_task_plan[0]
            )
            self._last_operator_reached_effects = self._skill_reached_effects.clone()
            # Pop the next operator from the task plan.
            self._current_operator = self._current_task_plan.pop(0)
            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill_for_operator(self._current_operator)
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
        return skill_action, self._current_operator

    def _get_skill_for_operator(self, operator: GroundOperator) -> LiftedOperatorSkill:
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]

    @property
    def exhausted(self) -> Tensor:
        """Get exhausted mask."""
        return self._skill_exhausted.clone()

    @property
    def last_operator_reached_effects(self) -> Tensor:
        """Get reached effects mask."""
        return self._last_operator_reached_effects.clone()

    @property
    def last_operator(self) -> GroundOperator | None:
        """Get the current operator being executed."""
        return self._last_operator
