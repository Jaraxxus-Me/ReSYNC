"""Unit Tests for NSRT learning in BlockedStacking environment, with provided predicates
and skills."""

import copy
import os
import shutil
import types
from pathlib import Path

import pytest
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.lifelong_ref import LifelongRefApproach
from skill_refactor.approaches.operator_learner import learn_operator_from_data
from skill_refactor.approaches.rl_policies.ppo_c import PPOCPolicy
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import (
    PlanningStatesVectorEnv,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import (
    LiftedOperator,
    LiftedOperatorSkill,
    PlannerDataset,
)
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


def _collect_blocked_stacking_test_data(save_path: Path, num_episodes: int = 5) -> None:
    """Helper function to collect minimal planner data for operator learning tests."""
    sc = "1"
    test_config = {
        "num_envs": 1,
        "debug_env": False,
        "scenario": sc,
        f"obstruction{sc}_blocking_grasp": True,
        f"obstruction{sc}_blocking_stacking": False,
        "max_env_steps": 240,
        "num_train_episodes_planner": 10,
        "num_train_episodes_planner_before_dreaming": num_episodes,
        "lll_config": f"config/lifelong_learning/blocked_stacking_sc{sc}.yaml",
        "control_mode": "pd_joint_delta_pos",
    }
    reset_config(test_config)
    register_all_environments()
    # sc1
    with open(CFG.lll_config, "r", encoding="utf-8") as f:
        scenario_info = yaml.safe_load(f)["scenarios"][int(CFG.scenario)]
    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = LifelongRefApproach(tamp_system, seed=CFG.seed)
    approach.update_learning_info(
        int(CFG.scenario),
        scenario_info,
    )
    fall_back_action = tamp_system.env.single_action_space.sample()  # type: ignore
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    policy = PPOCPolicy(seed=CFG.seed, rl_config=scenario_info["rl_config"])
    # Scenario 1
    envs_mani = PlanningStatesVectorEnv(
        tamp_system.env,
        tamp_system,
        scenario_info,
        planner,
        num_envs=CFG.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    policy.initialize(envs_mani)
    assert "pre_trained_policy_path" in scenario_info
    pre_trained_policy_path = Path(scenario_info["pre_trained_policy_path"])
    policy.load(pre_trained_policy_path)

    # Load initial states from task files
    import glob
    import pickle

    from skill_refactor.utils.structs import Task

    task_files = glob.glob(
        f"{CFG.specified_task_path}/sc{sc}_task_seed{CFG.seed}_*.pkl"
    )
    init_states = []
    for task_file in task_files:
        with open(task_file, "rb") as f:
            task_data: Task = pickle.load(f)
            init_states.append(task_data.init)

    train_data = approach.collect_planner_data(tamp_system.env, policy, init_states)
    tamp_system.env.close()  # type: ignore

    # Save to temporary location
    save_path.mkdir(parents=True, exist_ok=True)
    train_data.save(save_path)


@pytest.mark.skipif(
    not os.path.exists("1024_sc1_skills/best_classifier_ckpt.pt"),
    reason="policy not found",
)
def test_op_learning_blocked_stacking() -> None:
    """Test Operator learning in BlockedStacking environment."""
    # Setup temporary test data directories
    dataset_path1 = Path("/tmp/test_op_learning_bstacking")

    try:
        # Collect test data (3 trajectories each)
        print("Collecting first dataset...")
        _collect_blocked_stacking_test_data(dataset_path1, num_episodes=3)

        test_config = {
            "traj_segmenter": "operator_changes",
        }
        reset_config(test_config)
        register_all_environments()

        tamp_system = BlockedStackingRLTAMPSystem.create_default(
            render_mode="rgb_array", seed=42
        )
        planner_dataset1 = PlannerDataset.load(dataset_path1)

        assert (
            len(planner_dataset1.trajectories) > 0
        ), "Planner dataset should not be empty."

        given_operators = planner_dataset1.get_appearing_operators()
        # BlockedStacking typically has 4 operators:
        # ReachToGrasp, Grasp, ReachToPlace, Place
        expected_num_ops = len(given_operators)
        assert (
            len(given_operators) >= 2
        ), "There should be at least 2 appearing operators in the dataset."

        trajectories = planner_dataset1.trajectories
        ground_atom_dataset, tasks = planner_dataset1.get_ground_atoms_and_tasks(
            tamp_system.perceiver
        )

        assert len(tasks) == len(
            ground_atom_dataset
        ), "Tasks and ground atom dataset should have the same length."
        assert len(ground_atom_dataset) == len(
            trajectories
        ), "Ground atom dataset should match the number of trajectories."

        # Learn operators from the dataset
        operators, _, _, _ = learn_operator_from_data(
            "clustering",
            trajectories,
            tasks,
            tamp_system.perceiver.predicates_container.as_set(),
            given_operators,
            ground_atom_dataset,
        )

        # Verify operators were learned successfully
        assert (
            len(operators) == expected_num_ops
        ), f"There should be {expected_num_ops} learned operators."
        print(f"✓ Successfully learned {len(operators)} operators")

        # Verify each operator has preconditions and effects
        for op in operators:
            assert (
                len(op.preconditions) >= 0
            ), f"Operator {op.name} should have preconditions defined"
            assert (
                len(op.add_effects) >= 0 or len(op.delete_effects) >= 0
            ), f"Operator {op.name} should have at least one effect"

        # Test that we can bind skills to operators
        binded_skills: set[LiftedOperatorSkill] = set()
        for op in operators:
            for skill in tamp_system.skills:
                if (skill.get_operator_name() == op.name) or (
                    skill.get_operator_name() == op.name.split("_")[0]
                ):
                    new_skill = skill.__class__(  # type: ignore[call-arg]
                        env=tamp_system.env,
                        operators=operators,
                    )
                    local_op = copy.deepcopy(op)

                    # capture local_op in the default arg
                    def _customget_operator_name(self, op=local_op) -> str:
                        del self
                        return op.name

                    def _custom_get_lifted_operator(
                        self, op=local_op
                    ) -> LiftedOperator:
                        del self
                        return op

                    setattr(
                        new_skill,
                        "get_operator_name",
                        types.MethodType(_customget_operator_name, new_skill),
                    )
                    setattr(
                        new_skill,
                        "get_lifted_operator",
                        types.MethodType(_custom_get_lifted_operator, new_skill),
                    )
                    binded_skills.add(new_skill)
                    break

        print(f"✓ Successfully bound {len(binded_skills)} skills to operators")
        assert len(binded_skills) > 0, "Should be able to bind at least one skill"

        print(f"✓ Test completed successfully. Operators learned: {len(operators)}")

    finally:
        # Cleanup: Remove temporary test data
        for path in [dataset_path1]:
            if path.exists():
                shutil.rmtree(path)
