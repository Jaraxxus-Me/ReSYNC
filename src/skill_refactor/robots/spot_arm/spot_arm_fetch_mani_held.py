"""Spot with an arm robot for mobile manipulation with sticky YCB object attachment.

This robot extends SpotArmMobMani with a 6-DOF attachment mechanism for YCB objects. The
YCB object is attached via 6 joints (3 prismatic + 3 revolute) that can be switched
between free (zero stiffness/damping) and locked (high stiffness/damping) states.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link

from skill_refactor import PACKAGE_ASSET_DIR
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import PDBasePosControllerConfig

SPOT_LLEG_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links."""
SPOT_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base."""


@register_agent()
class SpotArmFetchManiHeld(BaseAgent):
    """Spot Arm robot with a fetch hand."""

    uid = "spot_arm_fetch_held"
    urdf_path = f"{PACKAGE_ASSET_DIR}/tidyspot_fetch_box/spot_fetch_mani.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            r_gripper_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            l_gripper_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = {
        "rest": Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([0.0, 0.0, 0.0, 0.0, -3.115, 3.0, 1.572, -0.019, -1.572, 0.015, 0.015]),  # fmt: skip
        )
    }

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the SpotArmStickyMobMani robot.

        Args:
            ycb_object_id: YCB object ID to attach (e.g., "024_bowl")
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.arm_joint_names = [
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "l_gripper_finger_joint",
            "r_gripper_finger_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "hand_frame"

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self) -> Dict:
        """Return the controller configurations for the robot."""
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-CFG.arm_joint_delta,
            upper=CFG.arm_joint_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            normalize_action=CFG.normalize_action,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.04,  # a trick to have force when the object is thin
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_delta_pos = PDBasePosControllerConfig(
            self.base_joint_names,
            lower=-CFG.body_joint_delta,
            upper=CFG.body_joint_delta,
            stiffness=1000,
            damping=100,
            force_limit=500,
            use_delta=True,
            normalize_action=CFG.normalize_action,
        )

        # NOTE: Base x y theta, arm-6 + gripper-1, same as qpos extraction order
        controller_configs = {
            "pd_joint_delta_pos": {
                "base": base_pd_joint_delta_pos,
                "arm": arm_pd_joint_delta_pos,
                "gripper": gripper_pd_joint_pos,
            },
            "pd_joint_pos": {
                "base": base_pd_joint_delta_pos,
                "arm": arm_pd_joint_pos,
                "gripper": gripper_pd_joint_pos,
            },
        }

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self) -> None:
        """After the robot is initialized, set up the links and collision groups."""
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_gripper_finger_link"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_gripper_finger_link"
        )
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "body"
        )

        self.held_obj_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "box_held_link"
        )

        self.collision_checker_links: List[Link] = [
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_wr1"),
            sapien_utils.get_obj_by_name(
                self.robot.get_links(), "l_gripper_finger_link"
            ),
            sapien_utils.get_obj_by_name(
                self.robot.get_links(), "r_gripper_finger_link"
            ),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "gripper_link"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_wr0"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_el1"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_el0"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_sh1"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_sh0"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "arm_link_hr0"),
            sapien_utils.get_obj_by_name(self.robot.get_links(), "box_held_link"),
        ]

        self.base_link.set_collision_group_bit(
            group=2, bit_idx=SPOT_BASE_COLLISION_BIT, bit=1
        )

        assert not CFG.delta_finger_control, "Delta finger control is not supported."

    def is_grasping(
        self,
        object: Actor,  # pylint: disable=redefined-builtin
        min_force: float = 10.0,
        max_angle: int = 60,
    ) -> torch.Tensor:
        """Check if the robot is grasping an object.

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is
                considered to be grasping the object in Newtons. Defaults to 30.0.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 60.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        """Check if the robot is static."""
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )

    def is_colliding(
        self,
        objects: List[Actor],  # pylint: disable=redefined-builtin
        min_force: float = 3.0,
    ) -> torch.Tensor:
        """Check if the robot is grasping an object.

        Args:
            entity (Actor): The object to check if the robot is colliding with.
            min_force (float, optional): Force threshold to consider a collision in Newtons. Defaults to 0.1.
        """
        contact_forces_list = []
        for link in self.collision_checker_links:
            for obj in objects:
                contact_forces_list.append(
                    self.scene.get_pairwise_contact_forces(link, obj)
                )
        contact_forces = torch.stack(contact_forces_list, dim=1)
        contact_forces = torch.linalg.norm(contact_forces, axis=2)
        contact_forces_all_links = torch.any(contact_forces >= min_force, dim=1)
        return contact_forces_all_links

    @property
    def tcp_pos(self) -> Pose:
        """Get the position of tool center point (TCP) as a Pose."""
        return self.tcp.pose.p

    @property
    def tcp_pose(self) -> Pose:
        """Get the pose of tool center point (TCP) as a Pose."""
        return self.tcp.pose

    @property
    def body_pose(self) -> Pose:
        """Get the pose of the robot body."""
        return self.base_link.pose
