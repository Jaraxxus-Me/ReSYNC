"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg (args.py).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np


class GlobalSettings:
    """Unchanging settings."""

    seed: int = 0
    # global parameters
    render: bool = True
    debug_env: bool = False
    scenario: str = "1"  # 1, 2, or 3 in this work
    num_eval_episodes: int = 10  # num episodes per eval_env
    num_train_episodes_rl: int = 1
    num_train_episodes_planner: int = 1
    dreaming_noise_base_var: float = 1e-6
    num_train_episodes_planner_before_dreaming: int = 1
    num_failed_episodes_planner: int = 1
    max_env_steps: int = 300  # for the entire task
    record_training: bool = True
    training_record_interval: int = 5000
    num_envs: int = 16
    num_eval_envs: int = 16
    log_wandb: bool = False
    control_mode: str = "pd_joint_delta_pos"
    delta_finger_control: bool = True
    arm_joint_delta: float = 0.2  # for pd_joint_delta_pos control mode
    body_joint_delta: float = 0.1  # for pd_joint_delta_pos control mode
    normalize_action: bool = True
    # if False, use intrinsic reward for RL learning
    # NOTE: This is only used in debug mode for faster skill tuning.
    planner_eval: bool = True
    partial_planner_eval: bool = False
    intrinsic_reward_path: str = ""
    rl_next_skill: str = ""

    # Output settings
    results_dir: Path = Path("results")
    rl_policy_save_dir: Path = Path("trained_policies")
    rl_dataset_path: str = ""
    pred_net_save_dir: str = "trained_pred_nets"
    invented_pred_op_json: str = "invented_predicates_operators"
    tb_log_dir: Path = Path("tf_logs")
    # tb_log_dir = None
    exp_name: str = "debug_experiment"
    device: str = "cuda:0"  # "cpu" or "cuda"

    # lifelong learning parameters
    specified_task_path: str = ""
    lll_config: str = ""
    rl_config: str = ""
    pre_trained_policy_path: str = ""
    force_collect: bool = False
    pre_rl_stabilize_steps: int = 2

    # rl-training parameters
    rl_algo: str = "PPO"

    # planner-learning parameters
    real_scenario_name: str = ""
    traj_segmenter: str = "operator_changes"
    render_planner_collection_img: str = ""
    predicate_config: str = ""
    planner_dataset_path: str = ""
    planner_num_traj: int = -1
    max_consecutive_failures: int = 5
    known_invalid_sc: str = ""
    precondition_threshold: float = 0.8  # minimum fraction for precondition inclusion
    add_effect_threshold: float = 0.8  # minimum fraction for add effect inclusion
    delete_effect_threshold: float = 0.8  # minimum fraction for delete effect inclusion
    super_label = {
        "ignore": -1,  # wrong type match for the predicate
        "change_neg": 0,
        "change_pos": 1,
        "non_change_1": 2,  # the predicate is not in the effect, and the objects (pairs) are not operated
        "non_change_2": 3,  # the predicate is in the effect, but the objects are not operated
        "non_change_3": 4,  # the predicate is not in the effect, but the objects (pairs) are operated
    }
    num_middle_states: int = 10
    num_middle_states_close: int = 9
    middle_state_method: str = (
        "naive_binary"  # "naive_init", "soft_interpolation", "two_endpoint_weighted"
    )
    sesame_task_planning_heuristic: str = "lmcut"
    task_planning_timeout: float = 10.0  # seconds
    force_skip_pred_learning: bool = False
    quantify_basic: bool = True
    filter_preconditions: bool = False
    pred_search_precond_unsolvable_threshold: float = 0.3
    pred_search_num_trajectories: int = 40
    pred_search_expected_nodes_upper_bound = 1e5
    pred_search_max_skeletons_optimized = 5
    pred_search_expected_nodes_optimal_demo_prob = 1 - 1e-5
    pred_search_expected_nodes_backtracking_cost = 1e3
    pred_search_operator_learning_out_of_sample_penalty: float = 1e4
    pred_quantifiers = ["ForAll", ""]
    predicate_search_task_planning_timeout: float = 1.0  # seconds
    # dsg baseline toggle: learn only terminal predicates and reuse operators/skills
    dsg_mode: bool = False

    # environment parameters
    # ** Icy Transport 2D Environment **
    icy_infront_of_transport1: bool = True
    icy_infront_of_transport2: bool = False
    icy_infront_of_target: bool = False
    muddy_infront_of_transport1: bool = True
    muddy_infront_of_transport2: bool = False
    muddy_infront_of_target: bool = False
    # These can't be accessed directly from the observation,
    # but are used to set up the controller/planner.
    i_trans_world_size: tuple[float, float] = (5.0, 5.0)  # width, height
    i_trans_obj_pos_rnd_range: float = 0.1
    i_trans_door_width: float = 0.8
    i_trans_wall_thickness: float = 0.1
    robot_base_width: float = 0.3
    robot_base_length: float = 0.5
    robot_base_mass: float = 0.5
    robot_steering_length: float = 0.5 / 3
    i_transport_object_radius: float = 0.1
    i_transport_unique_region_shape: tuple[float, float] = (
        0.1,
        i_trans_door_width,
    )  # width, height
    i_transport_unique_region_offset: float = (
        2.1  # distance from center to region center along the wall normal
    )
    icy_inhand_fy_penalty: float = -1.5  # additional fy penalty when holding an object
    i_trans_icy_region_feature: tuple[float, float, float, float, float] = (
        0.0,
        0.0,
        0.8,
        0.0,
        0.0,
    )  # local_x, local_y, fx, fy, torque
    i_trans_muddy_region_feature: tuple[float, float, float, float, float] = (
        0.0,
        0.0,
        -0.8,
        0.0,
        0.0,
    )  # local_x, local_y, fx, fy, torque

    # Skill parameters
    kp_pos: float = 100.0
    kv_pos: float = 20.0
    kp_ang: float = 50.0
    kv_ang: float = 10.0
    kv_pos_rl: float = 20.0
    kv_ang_rl: float = 10.0
    sim_dt: float = 0.02  # simulation timestep
    i_trans_waypoint_threshold_trans: float = (
        0.02  # 2cm threshold for considering waypoint achieved
    )
    i_trans_waypoint_threshold_rot: float = 0.005  # ~5.7 degrees
    i_trans_max_waypoint_tries: int = 150  # Maximum attempts before skipping a waypoint
    i_trans_in_room_offset: float = (
        0.1  # Distance offset for considering robot in a room (meters)
    )
    icy_transport_reach_drift_trans: float = 0.0
    icy_transport_reach_drift_rot: float = 0.0

    # ** Blocked Stacking 2D Environment **
    obstruction1_blocking_grasp: bool = True
    obstruction1_blocking_stacking: bool = False
    obstruction2_blocking_grasp: bool = True
    obstruction2_blocking_stacking: bool = False
    obstruction3_blocking_grasp: bool = True
    obstruction3_blocking_stacking: bool = False
    # These can't be accessed directly from the observation,
    # but are used to set up the planner.
    blocked2d_robot_base_radius: float = 0.24
    blocked2d_robot_arm_length_max: float = 0.48
    blocked2d_gripper_base_width: float = 0.06
    blocked2d_gripper_base_height: float = 0.32
    blocked2d_gripper_finger_width: float = 0.2
    blocked2d_gripper_finger_height: float = 0.06
    blocked2d_reach_drift_trans: float = 0.0
    blocked2d_reach_drift_rot: float = 0.0

    # behavior Cloning
    bc_model: str = "gnn"  # "gnn" | "transformer"
    bc_num_epochs: int = 50
    bc_batch_size: int = 1
    bc_lr: float = 3e-4
    bc_weight_decay: float = 0.0
    bc_dataset_num_traj: int = -1
    bc_save_path: str = ""
    bc_load_path: str = ""
    bc_use_predicate_augmentation: bool = (
        True  # whether to augment BC-GNN with symbolic predicates
    )

    # GNN specific
    gnn_num_message_passing = 3
    gnn_num_layers = 2  # after the first linear, how many gnn_layer_size layers
    gnn_layer_size = 512
    gnn_learning_rate = 1e-3
    gnn_weight_decay = 0
    gnn_num_epochs = 800
    gnn_batch_size = 512
    gnn_do_normalization = False  # performs worse in Cover when True
    gnn_use_validation_set = True
    gnn_checkpoint_interval = 100  # Save checkpoint every N epochs

    # Weights & Biases settings
    use_wandb: bool = False  # whether to log to Weights & Biases
    wandb_project: str = "skill-refactor"  # W&B project name
    wandb_entity: Optional[str] = "neuro-symbolic-AI"  # W&B team/entity name

    # ** Cluttered Drawer Environment **
    drawer_blocking_grasp: bool = True
    drawer_blocking_stacking: bool = False
    block_blocking_grasp: bool = False
    block_blocking_stacking: bool = True
    wall_blocking_grasp: bool = False
    wall_blocking_stacking: bool = True
    cluttered_drawer_reach_drift_trans: float = 0.01
    cluttered_drawer_reach_drift_rot: float = 0.05
    c_drawer_waypoint_max_try_count: int = 3
    waypoint_threshold: float = 0.05  # 5cm threshold for considering waypoint achieved
    max_waypoint_tries: int = 100  # Maximum attempts before skipping a waypoint
    angular_threshold: float = (
        0.15  # radians, around 10 degrees, this is large as sometimes gripper collides with objects
    )
    # broader env that any pose should lie within
    c_drawer_world_min: tuple[float, float, float] = (-1.5, -1.5, 0.0)
    # c_drawer_world_max: tuple[float, float, float] = (1.4, 0.6, 1.3)
    c_drawer_world_max: tuple[float, float, float] = (1.5, 1.5, 1.0)
    # Robot and object initial placement parameters
    c_drawer_spot_body_x: float = -0.8
    c_drawer_spot_body_z: float = 0.0
    c_drawer_spot_body_init_dist: tuple[float, float] = (0.0, 0.1)
    c_drawer_spot_body_init_theta: tuple[float, float] = (0.0, 0.2)
    c_drawer_hammer_init_sampling_min: tuple[float, float, float] = (
        0.3,
        -0.8,
        -np.pi / 4,
    )
    c_drawer_hammer_init_sampling_max: tuple[float, float, float] = (
        0.4,
        0.8,
        np.pi / 4,
    )
    c_drawer_hammer_min_dist_y: float = 1.0
    c_drawer_hammer_height: float = 0.35
    # Drawer and object goal parameters
    c_drawer_hammer_drawer_delta_x: tuple[float, float] = (
        0.27,
        0.33,
    )  # when drawer_q is zero
    c_drawer_hammer_drawer_delta_y: tuple[float, float] = (-0.03, 0.03)
    c_drawer_drawer_height: float = 0.45
    c_drawer_goal_radius: float = 0.1
    c_drawer_goal_height: float = 0.05
    c_drawer_matched_dy: float = 0.2
    # Skill parameters
    c_drawer_place_rel_dx: float = 0.01
    c_drawer_place_rel_dy: float = 0.0
    c_drawer_place_rel_dz: float = -0.02
    c_drawer_place_height_tol: float = 0.06
    c_drawer_body_ready_grasp_dx: float = -0.85
    c_drawer_body_ready_grasp_hand_x_axis_degree: float = (
        0.12  # default hand is 0.05 rad
    )
    c_drawer_body_ready_grasp_hand_dz_thresh: float = 0.5  # default hand is 0.55 cm
    c_drawer_body_predicate_transition_thresh: float = 0.1  # 5 cm
    c_drawer_ready_grasp_xy: float = 0.05
    c_drawer_ready_grasp_z: float = 0.034
    c_drawer_reachtoplace_lift_hand_z: float = 0.25
    c_drawer_reachtograsp_lift_hand_z: float = 0.18
    # relative to drawer handle link
    # block relative parameters
    c_drawer_block_dx: tuple[float, float] = (
        0.08,
        0.085,
    )  # (0.085, 0.09) (0.33, 0.35)
    c_drawer_block_dy: tuple[float, float] = (
        -0.075,
        -0.07,
    )
    c_drawer_block_dz: float = 0.03
    c_drawer_wall_dx: tuple[float, float] = (
        -0.085,
        -0.08,
    )  # (0.085, 0.09) (0.33, 0.35)
    c_drawer_wall_dy: tuple[float, float] = (
        0.0,
        0.01,
    )
    c_drawer_spawn_center: tuple[float, float, float] = (1.2, 0.38, 0.7)
    c_drawer_spawn_scale: float = 1.4
    c_drawer_close_frac: float = 0.35
    nail_drawer_offset_x: float = 0.78
    c_drawer_sensor_cam_eye_pos: list[float] = [0.3, 0.0, 0.6]
    c_drawer_sensor_cam_target_pos: list[float] = [-0.1, 0.0, 0.1]
    c_drawer_human_cam_eye_pos: list[float] = [-1.5, 3.0, 2.0]
    c_drawer_human_cam_target_pos: list[float] = [0.5, 0.3, 0.9]
    # c_drawer_human_cam_eye_pos: list[float] = [0.8, 0.0, 5.0]
    # c_drawer_human_cam_target_pos: list[float] = [0.8, 0.0, 0.0]
    # c_drawer_human_cam_eye_pos: list[float] = [0.0, 2.0, 0.6]
    # c_drawer_human_cam_target_pos: list[float] = [0.0, 0.0, 0.6]

    # ** Cluttered Room Environment **
    chair_blocking_transport1: bool = True
    chair_blocking_transport2: bool = False
    chair_blocking_target1: bool = False
    chair_blocking_target2: bool = False
    can_blocking_target1: bool = True
    can_blocking_target2: bool = False

    maniskill_asset_dir: Path = Path("src/skill_refactor/assets/maniskill_assets")
    maniskill_rearrange_dir: Path = maniskill_asset_dir / Path(
        "data/scene_datasets/replica_cad_dataset/rearrange"
    )
    # Env reset and placement parameters

    can_blocking_rel_dx: tuple[float, float] = (-0.03, -0.01)
    can_blocking_rel_dy: tuple[float, float] = (-0.03, 0.03)
    can_blocking_theta_range: tuple[float, float] = (-np.pi, np.pi)

    c_room_force_held: bool = False
    c_room_waypoint_max_try_count: int = 6
    c_room_spot_body_x: float = -0.5
    c_room_spot_body_y: float = -0.5
    c_room_obj_goal_thresh: float = 0.1
    c_room_chair_scale: float = 0.6
    c_room_chair_z: float = 0.0  # Chair z position (floor level)
    c_room_can_z: float = -0.01  # Can z position (on table)
    c_room_world_min: tuple[float, float, float] = (
        0.0,
        -2.0,
        0.0,
    )  # Far away position for unused objects
    clamp_xy_to_radius: float = -1.0  # if >0, clamp relative xy to this radius
    # Region 1
    c_room_region1_x: tuple[float, float] = (-2.0, -1.9)
    c_room_region1_y: tuple[float, float] = (-2.3, -1.6)
    # Region 2
    c_room_region2_x: tuple[float, float] = (-2.0, -1.9)
    c_room_region2_y: tuple[float, float] = (-0.8, 0.1)
    c_room_obj1_z: float = 0.9
    c_room_obj2_z: float = 0.94
    c_room_goal_z: float = 0.9
    c_room_obj_min_dist: float = 0.5  # Minimum distance between objects in same region

    # Given skill and predicate parameters
    c_room_navigate_home_x: float = -0.8
    c_room_hand_empty_finger_opening_thresh: float = 0.04
    c_room_obj2_held_xy: float = 0.08
    c_room_obj2_held_z: float = 0.05
    c_room_obj2_held_gripper: float = 0.035
    # Bowl picking parameters
    c_room_navigate_bowl_rel_dx: float = 0.8
    c_room_pick_bowl_hand_dpitch: float = np.pi * 5 / 12
    c_room_place_hand_facing_goal_thresh_rad: float = np.pi / 12  # 15 degrees
    c_room_pick_bowl_hand_dx: float = 0.08
    c_room_pick_bowl_hand_dz: float = 0.05
    c_room_pick_bowl_hand_forward: float = 0.035

    # Box picking parameters
    c_room_navigate_box_rel_dx: float = 0.7
    c_room_pick_box_hand_dpitch: float = np.pi / 2
    c_room_pick_box_hand_dx1: float = 0.1
    c_room_pick_box_hand_dy1: float = 0.15
    c_room_pick_box_hand_dx2: float = 0.0
    c_room_pick_box_hand_dy2: float = 0.15
    c_room_pick_box_hand_forward: float = 0.14

    c_room_body_ready_pick_dist: float = 0.05
    c_room_finger_close_value: float = 0.01

    # Shared lifting parameters
    c_room_pick_obj_lifting_dz: float = 0.12  # Height to lift object after grasping
    c_room_pick_obj_lifting_dx: float = 0.2  # Forward offset when lifting object
    c_room_pre_goto_place_hand_body_dx: float = (
        0.54  # Forward offset before going to place
    )
    c_room_pre_goto_place_hand_body_dz: float = (
        1.12  # Height offset before going to place
    )
    c_room_pre_place_obj_lifting_dx: float = 0.07  # Forward offset before placing
    c_room_pre_place_obj_lifting_dz: float = 0.1  # Height above goal before placing
    c_room_place_obj_lifting_dz: float = 0.04  # Height above goal before placing
    c_room_post_place_obj_lifting_dz: float = 0.12  # Height to lift after placing

    # ******* Baseline Recovery Chaining Settings ********
    failured_det_operator_objects: Dict[str, list[int]] = {}
    failure_nn_config: str = ""  # Config for failure detection neural network
    failured_det_nn_path: str = ""
    enable_collision_terminate: bool = True

    @classmethod
    def get_arg_specific_settings(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the experiment-
        specific args."""
        del args  # unused
        return {}


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
