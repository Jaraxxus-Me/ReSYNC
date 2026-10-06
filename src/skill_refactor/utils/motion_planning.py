"""GPU Batched, Tensor-based IK for Spot Robot Motion Skills."""

import logging
import math

# TYPE_CHECKING import to avoid circular dependency
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from skill_refactor.benchmarks.icy_transport.utils import (
    extract_robot_force,
    extract_robot_mass_moment,
    extract_robot_pose,
    extract_robot_vel,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import FINGER_ACTION_INDEX, get_frozen_action

if TYPE_CHECKING:
    from prbench.envs.dynamic2d.utils import CarRobot

SHOULDER_OFFSET = np.linalg.inv(
    np.array([[1, 0, 0, 0.162], [0, 1, 0, 0], [0, 0, 1, 0.42], [0, 0, 0, 1]])
)

SHOULDER_OFFSET_ROOM = np.linalg.inv(
    np.array([[1, 0, 0, 0.162], [0, 1, 0, 0], [0, 0, 1, 0.82], [0, 0, 0, 1]])
)

HAND2WRIST_POSE = Pose.create_from_pq(
    torch.tensor([-0.1955707, 0, 0]), torch.tensor([1.0, 0.0, 0.0, 0.0])
)

#############################
# Batched 2R IK for PyTorch
################################


def IK2R_batched(
    L1: float, L2: float, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 2R planar IK for link lengths L1, L2.

    Args:
        L1, L2: link lengths
        x, y: torch.Tensor of shape (B,), target end-effector coordinates in plane.
    Returns:
        sols: torch.Tensor of shape (B, 2, 2), each entry [i, j] = (q2, q3) solution j for sample i.
        valid: torch.BoolTensor of shape (B,) indicating if c2 in [-1,1].
    """
    # Law of cosines
    xy2 = x**2 + y**2
    c2 = (xy2 - L1**2 - L2**2) / (2 * L1 * L2)
    valid = c2.abs() <= 1.0
    c2_clamped = c2.clamp(-1.0, 1.0)

    # two elbow angles
    q3a = torch.acos(c2_clamped)
    q3b = -q3a

    # shoulder base angle
    theta = torch.atan2(y, x)
    alpha_a = torch.atan2(L2 * torch.sin(q3a), L1 + L2 * torch.cos(q3a))
    alpha_b = torch.atan2(L2 * torch.sin(q3b), L1 + L2 * torch.cos(q3b))

    q2a = theta - alpha_a
    q2b = theta - alpha_b

    # stack solutions: (B, 2, 2) dims: sample, solution-index, [q2, q3]
    sols = torch.stack(
        [torch.stack([q2a, q3a], dim=1), torch.stack([q2b, q3b], dim=1)], dim=1
    )
    return sols, valid


def analytic_spot_ik_6_torch(
    wrist_pose: torch.Tensor,  # (B,4,4)
    min_limits: torch.Tensor,  # (6,)
    max_limits: torch.Tensor,  # (6,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched analytic IK for Spot 6-DOF arm.

    Args:
        wrist_pose: (B,4,4) transform wrist->shoulder
        min_limits, max_limits: (6,) joint bounds
    Returns:
        solutions: torch.Tensor of shape (B, 8, 6)
        valid: torch.Tensor of shape (B, 8) mask of solutions within limits
    """
    B = wrist_pose.shape[0]
    # link lengths
    l2 = 0.3385
    l3 = (0.40330**2 + 0.0750**2) ** 0.5
    q3_off = torch.atan2(torch.tensor(0.0750), torch.tensor(0.40330))

    # extract wrist pos
    px = wrist_pose[:, 0, 3]
    py = wrist_pose[:, 1, 3]
    pz = wrist_pose[:, 2, 3]
    xl = torch.sqrt(px**2 + py**2)

    # first planar IK
    sols1, _ = IK2R_batched(l2, l3, xl, -pz)
    # second (rotated) planar IK
    sols2, _ = IK2R_batched(l2, l3, -xl, -pz)

    # build q1 for each sample
    q1_base = torch.atan2(py, px)  # (B,)
    q1_1 = q1_base.unsqueeze(1).expand(-1, 2)  # (B,2)
    q1_2 = (q1_base + math.pi).unsqueeze(1).expand(-1, 2)

    # assemble shoulder/elbow solutions: (B,4,3)
    sols1[:, :, 1] += q3_off  # adjust q3 by offset
    sols2[:, :, 1] += q3_off  # adjust q3 by offset
    q2q3_1 = sols1  # (B,2,2)
    q2q3_2 = sols2  # (B,2,2)
    sol123 = torch.cat(
        [
            torch.cat([q1_1.unsqueeze(2), q2q3_1], dim=2),
            torch.cat([q1_2.unsqueeze(2), q2q3_2], dim=2),
        ],
        dim=1,
    )  # (B,4,3)

    # expand for batch of solutions
    BS = B * 4
    sol123_bs = sol123.reshape(BS, 3)
    # prepare wrist poses per solution
    wrist_bs = wrist_pose.unsqueeze(1).expand(-1, 4, -1, -1).reshape(BS, 4, 4)

    # build T_r3 (BS,4,4)
    q1_bs = sol123_bs[:, 0]
    q23_sum = sol123_bs[:, 1] + sol123_bs[:, 2]

    # rotation matrices
    def rot_z(q: torch.Tensor) -> torch.Tensor:
        """Rotation matrix around Z-axis for angle q."""
        c = torch.cos(q)
        s = torch.sin(q)
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(q)
        R = torch.stack(
            [
                torch.stack([c, -s, zeros], dim=1),
                torch.stack([s, c, zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        return R

    def rot_y(q: torch.Tensor) -> torch.Tensor:
        """Rotation matrix around Y-axis for angle q."""
        c = torch.cos(q)
        s = torch.sin(q)
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(q)
        R = torch.stack(
            [
                torch.stack([c, zeros, s], dim=1),
                torch.stack([zeros, ones, zeros], dim=1),
                torch.stack([-s, zeros, c], dim=1),
            ],
            dim=1,
        )
        return R

    Rz = rot_z(q1_bs)  # (BS,3,3)
    Ry = rot_y(q23_sum)
    # make homogeneous
    T_r3 = torch.eye(4, device=wrist_bs.device).unsqueeze(0).repeat(BS, 1, 1)
    T_r3[:, :3, :3] = Rz @ Ry
    # invert
    T_r3_inv = torch.inverse(T_r3)

    # compute W = T_r3_inv @ wrist_bs
    W = T_r3_inv @ wrist_bs  # (BS,4,4)

    # extract needed
    W00 = W[:, 0, 0]
    W10 = W[:, 1, 0]
    W20 = W[:, 2, 0]
    W01 = W[:, 0, 1]
    W02 = W[:, 0, 2]

    # two wrist solutions
    q5a = torch.acos(W00)
    q5b = -q5a
    q5 = torch.stack([q5a, q5b], dim=1)  # (BS,2)
    s5 = torch.sin(q5)

    q4 = torch.atan2(W10.unsqueeze(1) / s5, -W20.unsqueeze(1) / s5)
    q6 = torch.atan2(W01.unsqueeze(1) / s5, W02.unsqueeze(1) / s5)

    # stack full solutions: (BS,2,6)
    sol123_expand = sol123_bs.unsqueeze(1).expand(-1, 2, -1)  # (BS,2,3)
    sol_ws = torch.stack([q4, q5, q6], dim=2)  # (BS,2,3)
    sol_full_bs = torch.cat([sol123_expand, sol_ws], dim=2)  # (BS,2,6)

    # reshape back: (B,4,2,6) -> (B,8,6)
    sol_full = sol_full_bs.view(B, 8, 6)

    # mask within limits
    lo = min_limits.view(1, 1, -1)
    hi = max_limits.view(1, 1, -1)
    valid = (sol_full >= lo) & (sol_full <= hi)
    valid = valid.all(dim=2)  # (B,8)

    return sol_full, valid


def angle_diff_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the angle difference between two tensors of angles."""
    two_pi = 2 * torch.pi
    z = (x - y) % two_pi
    return torch.where(z > torch.pi, z - two_pi, z)


def get_l1_distance_torch(sols1: torch.Tensor, sols2: torch.Tensor) -> torch.Tensor:
    """Compute the L1 distance between two sets of joint solutions."""
    diff = angle_diff_torch(sols1, sols2).abs()
    return diff.max(dim=-1).values


def rotate_vectors(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # v: (B, 2)
    # theta: (B,)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x = v[:, 0]
    y = v[:, 1]

    # Standard 2D rotation formula
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta

    return torch.stack((new_x, new_y), dim=1)


def select_solution_torch(
    solutions: torch.Tensor,  # (B, N, 6)
    curr_joint_positions: torch.Tensor,  # (B, 6)
    valid_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """Select the best IK solution per batch based on smallest max-angle difference to
    `curr_joint_positions`. Returns tensor of shape (B, 6).

    After selecting the tentative best, if any joint is ≈±π, we flip that joint (π->-π
    or -π->π) and keep the flip only if it reduces the max-angle error.
    """
    B = solutions.shape[0]

    # 1) Compute angle diffs and max-distance per solution
    diff = angle_diff_torch(
        solutions, curr_joint_positions.unsqueeze(1)
    ).abs()  # (B, N, 6)
    dist = diff.max(dim=2).values  # (B, N)
    if valid_mask is not None:
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(valid_mask, dist, inf)

    # 2) Pick the tentative best solution
    idx = dist.argmin(dim=1)  # (B,)
    batch_idx = torch.arange(B, device=solutions.device)
    tentative = solutions[batch_idx, idx]  # (B, 6)

    # 3) Detect joints near ±π
    # pi_thresh = 1e-3
    # pi = torch.tensor(math.pi, device=solutions.device)
    # near_pos_pi = (tentative - pi).abs() < pi_thresh    # (B, 6)
    # near_neg_pi = (tentative + pi).abs() < pi_thresh    # (B, 6)
    # flip_mask = near_pos_pi | near_neg_pi              # (B, 6)

    # # 4) If any joint needs flipping, build a flipped candidate
    # if flip_mask.any():
    #     flipped = tentative.clone()
    #     flipped[flip_mask] = -flipped[flip_mask]       # flip only the ±π entries

    #     # 5) Compute max-angle error for tentative vs. flipped
    #     orig_diff = (tentative - curr_joint_positions).abs()  # (B, 6)
    #     orig_dist = orig_diff.max(dim=1).values                             # (B,)

    #     flip_diff = (flipped - curr_joint_positions).abs()
    #     flip_dist = flip_diff.max(dim=1).values                             # (B,)

    #     # 6) For each batch, if flipped is strictly better, choose it
    #     use_flip = flip_dist < orig_dist                                    # (B,)
    #     if use_flip.any():
    #         tentative[use_flip] = flipped[use_flip]

    return tentative


def slerp_torch(
    q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two batches of unit quaternions.

    Args:
        q0: Tensor of shape (..., 4), start quaternions (must be normalized).
        q1: Tensor of shape (..., 4), end quaternions (must be normalized).
        t:  Tensor of shape (...) with interpolation factors in [0, 1].
        eps: small threshold to fall back to lerp when angles are very small.

    Returns:
        Tensor of shape (..., 4), interpolated unit quaternions.
    """
    # ensure same shape for elementwise ops
    t = t.unsqueeze(-1) if t.dim() + 1 == q0.dim() else t

    # Compute cosine between q0 and q1, shape (..., 1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Flip to take shortest path
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)

    # Decide between lerp and slerp
    DOT_THRESH = 1.0 - eps
    use_lerp = dot > DOT_THRESH  # boolean mask shape (..., 1)

    # LERP + normalize fallback
    lerp = q0 + t * (q1 - q0)
    lerp = lerp / lerp.norm(dim=-1, keepdim=True)

    # Standard SLERP
    theta_0 = torch.acos(dot)  # angle between
    sin_0 = torch.sin(theta_0)
    a = torch.sin((1.0 - t) * theta_0) / sin_0
    b = torch.sin(t * theta_0) / sin_0
    slerp = a * q0 + b * q1

    # Combine results
    return torch.where(use_lerp, lerp, slerp)


def concatenate_matrices(*mats: torch.Tensor) -> torch.Tensor:
    """Return the matrix product of a sequence of transformation matrices, batched. If
    no matrices are given, returns a 4×4 identity.

    Args:
        *mats: Tensors of shape (..., 4, 4).  Batch-shapes must all be broadcastable.

    Returns:
        Tensor of shape (..., 4, 4) = mats[0] @ mats[1] @ ... @ mats[-1].

    Examples:
        >>> M = torch.rand(4, 4) - 0.5
        >>> torch.allclose(M, concatenate_matrices_torch(M))
        True
        >>> torch.allclose(M @ M.T, concatenate_matrices_torch(M, M.T))
        True

        # batched example
        >>> A = torch.eye(4).unsqueeze(0).expand(5, 4, 4)
        >>> B = torch.rand(5, 4, 4)
        >>> C = torch.rand(5, 4, 4)
        >>> out = concatenate_matrices_torch(A, B, C)
        >>> torch.allclose(out, A @ B @ C)
        True
    """
    if len(mats) == 0:
        return torch.eye(4)

    # determine the common batch shape by broadcasting all batch dims
    # take the batch shape of the first matrix
    batch_shape = mats[0].shape[:-2]
    dtype = mats[0].dtype
    device = mats[0].device

    # start from batched identity
    identity = torch.eye(4, dtype=dtype, device=device)
    if batch_shape:
        identity = identity.view((1, 4, 4)).expand(*batch_shape, 4, 4)

    result = identity
    for M in mats:
        result = result.matmul(M)
    return result


class SpotMotion:
    """Toolkit for Spot IK based motion skills."""

    def __init__(
        self,
        device: torch.device,
        gripper_scale: float = 0.01,
    ):
        """Initialize SpotMotion with joint limits.

        Args:
            min_limits: Tensor of shape (7,) with minimum joint limits.
            max_limits: Tensor of shape (7,) with maximum joint limits.
        """
        self.min_limits = torch.as_tensor(
            [-2.6179938, -3.1415927, 0.0, -2.7925267, -1.8325957, -2.8797932],
            dtype=torch.float32,
            device=device,
        )
        self.max_limits = torch.as_tensor(
            [3.1415927, 0.5235988, 3.1415927, 2.7925267, 1.8325957, 2.8797932],
            dtype=torch.float32,
            device=device,
        )
        self.gripper_closed = 0.0  # for spot, larger finger angle is closing
        self.gripper_open = -1.57
        self.gripper_closing_delta = (
            self.gripper_closed - self.gripper_open
        ) * gripper_scale
        self.gripper_openning_delta = -self.gripper_closing_delta
        self.device = device
        self.hand2wrist_pose = HAND2WRIST_POSE.to(device)
        self.shoulder_offset = torch.tensor(
            SHOULDER_OFFSET, dtype=torch.float32, device=device
        )

    def solve_spot_ik(
        self,
        base_pose: Pose,
        end_effector_pose: Pose,
        curr_joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Solve Spot arm IK for a given wrist pose.

        Returns a joint configuration array of shape (8, 6). For now, use batch = 1, but
        this can be batched.
        """
        B = curr_joint_positions.shape[0]
        hand2wrist_pose = Pose.create_from_pq(
            self.hand2wrist_pose.p.repeat(B, 1), self.hand2wrist_pose.q.repeat(B, 1)
        )
        wrist_pose_worldF = end_effector_pose * hand2wrist_pose
        wrist_pose_robotF = base_pose.inv() * wrist_pose_worldF
        wrist_pose_robotF_mat = wrist_pose_robotF.to_transformation_matrix()
        shoulder_offset = self.shoulder_offset.unsqueeze(0).repeat(B, 1, 1)
        wrist_pose_shoulderF = concatenate_matrices(
            shoulder_offset, wrist_pose_robotF_mat
        )

        solutions, valid = analytic_spot_ik_6_torch(
            wrist_pose_shoulderF, self.min_limits, self.max_limits
        )
        selected_solution = select_solution_torch(
            solutions, curr_joint_positions, valid
        )

        return selected_solution

    def build_grasp_pose(
        self,
        approaching: torch.Tensor,
        closing: torch.Tensor,
        center: torch.Tensor,
        rel_dx: float = 0.01,
        rel_dy: float = 0.0,
        rel_dz: float = -0.04,
    ) -> Pose:
        """Build a grasp pose (spot_hand_frame)."""
        # assert (torch.abs(1 - torch.norm(approaching, dim=-1)) < 1e-3).all()
        # assert (torch.abs(1 - torch.norm(closing, dim=-1)) < 1e-3).all()
        # assert (
        #     torch.bmm(approaching.unsqueeze(1), closing.unsqueeze(-1)) <= 5e-3
        # ).all()
        B = approaching.shape[0]
        ortho = torch.cross(approaching, closing)  # x cross y = z
        T = torch.stack([approaching, closing, ortho], dim=2)
        q = matrix_to_quaternion(T)
        overlapping = Pose.create_from_pq(center, q)
        relative_pose = Pose.create_from_pq(
            torch.tensor([[rel_dx, rel_dy, rel_dz]] * B, dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B, dtype=torch.float32),
        )
        return overlapping * relative_pose

    def move_hand_from_to_pose(
        self,
        robot_worldF: Pose,
        curr_joint_positions: torch.Tensor,
        from_pose: Pose,
        to_pose: Pose,
        closing: torch.Tensor,
        interpolate_steps: int = 0,
    ) -> list[torch.Tensor]:
        """1) Seed robot at `from_joints` 2) Read out current end-effector pose
        ("from_pose") 3) Build a Cartesian trajectory of length (interpolate_steps+1)
        *excluding* the start, i.e. fractions = [1/(n+1), …, 1] 4) Solve IK at each
        fraction to get a joint waypoint 5) Return list of joint arrays."""
        # ——— 1) seed & get “from” pose ———
        from_pos = from_pose.p.clone()  # (x,y,z)
        from_q = from_pose.q.clone()  # (w,x,y,z)
        tgt_body_qpos = curr_joint_positions[:, :3]  # (B, body-dofs)

        # unpack goal
        to_pos = to_pose.p.clone()
        to_q = to_pose.q.clone()

        # ——— 2) if no interpolation, just solve final IK ———
        arm_joints = curr_joint_positions[:, 3:9]
        gripper_pos = curr_joint_positions[:, 9:].clone()
        closing_mask = closing.squeeze(-1)  # Squeeze to (B,) for proper indexing
        gripper_pos[closing_mask] += self.gripper_closing_delta
        gripper_pos[~closing_mask] = self.gripper_open
        if interpolate_steps <= 0:
            sol = self.solve_spot_ik(
                base_pose=robot_worldF,
                end_effector_pose=to_pose,
                curr_joint_positions=arm_joints,
            )
            if sol is None:
                raise RuntimeError(f"IK failed for pose {to_pos}, {to_q}")
            # append gripper state
            sol = torch.cat([tgt_body_qpos, sol, gripper_pos], dim=1)
            return [sol]

        # ——— 3) build fractions [1/(n+1), …, 1] ———
        n = interpolate_steps
        fractions = [i / (n + 1) for i in range(1, n + 2)]

        traj: list[torch.Tensor] = []
        prev_joints = arm_joints.clone()
        a_vec = torch.zeros_like(from_q[:, 0])  # (B,)
        da = 1 / (n + 1)  # step size for a
        for i, a in enumerate(fractions):
            # Cartesian interp
            p = (1 - a) * from_pos + a * to_pos
            a_vec += da
            q = slerp_torch(from_q, to_q, a_vec)

            # ——— 4) solve IK at fraction ———
            to_pose = Pose.create_from_pq(p, q)

            sol = self.solve_spot_ik(
                base_pose=robot_worldF,
                end_effector_pose=to_pose,
                curr_joint_positions=prev_joints,
            )

            if sol is None:
                raise RuntimeError(f"IK failed at a={a:.2f} → pos={p}, quat={q}")

            # append gripper state
            prev_joints = sol.clone()
            sol = torch.cat([tgt_body_qpos, sol, gripper_pos], dim=1)
            traj.append(sol)

        return traj

    def open_gripper(self, curr_qpos: torch.Tensor, t: int = 6) -> list[torch.Tensor]:
        """Open the gripper from its current position over `t` steps."""
        qpos = curr_qpos[:, :9]
        gripper_state = curr_qpos[:, 9:].clone()
        local_delta = (self.gripper_open - self.gripper_closed) / t
        actions = []
        for i in range(1, t + 1):
            curr_gripper_state = gripper_state + local_delta * i
            action = torch.cat([qpos, curr_gripper_state], dim=1)
            actions.append(action)
        return actions

    def close_gripper(self, curr_qpos: torch.Tensor, t: int = 6) -> list[torch.Tensor]:
        """Close the gripper from its current position over `t` steps."""
        qpos = curr_qpos[:, :9]
        gripper_state = curr_qpos[:, 9:].clone()
        local_delta = (self.gripper_closed - self.gripper_open) / t
        actions = []
        for i in range(1, t + 1):
            curr_gripper_state = gripper_state + local_delta * i
            action = torch.cat([qpos, curr_gripper_state], dim=1)
            actions.append(action)
        return actions

    def move_body_from_to_pose(
        self,
        robot_worldF_curr: Pose,
        robot_worldF_tgt: Pose,
        curr_joint_positions: torch.Tensor,
        closing: torch.Tensor,
        interpolate_steps: int = 50,
    ) -> list[torch.Tensor]:
        """Move the robot base from current pose to target pose with interpolation.

        Keeps arm joints and gripper state constant, only interpolates body motion.

        Args:
            robot_worldF_curr: Current robot base pose in world frame.
            robot_worldF_tgt: Target robot base pose in world frame.
            curr_joint_positions: Current joint positions (B, total_dofs).
            interpolate_steps: Number of interpolation steps.

        Returns:
            List of joint configurations with interpolated body motion.
        """

        # Extract arm and gripper states (keep these constant)
        arm_joints = curr_joint_positions[:, 3:9]
        gripper_pos = curr_joint_positions[:, 9:].clone()  # (B,1)
        closing_mask = closing.squeeze(-1)  # Squeeze to (B,) for proper indexing
        gripper_pos[closing_mask] += self.gripper_closing_delta
        gripper_pos[~closing_mask] = self.gripper_open

        # Extract current body pose from joint positions
        curr_body_qpos = curr_joint_positions[:, :3]  # (B, 3)

        # Calculate total delta from world frame poses
        curr_pos = robot_worldF_curr.p  # (B, 3)
        tgt_pos = robot_worldF_tgt.p  # (B, 3)

        dx_total = tgt_pos[:, 0] - curr_pos[:, 0]  # (B,)
        dy_total = tgt_pos[:, 1] - curr_pos[:, 1]  # (B,)

        # Extract yaw angle from rotation matrices
        curr_mat = robot_worldF_curr.to_transformation_matrix()  # (B, 4, 4)
        tgt_mat = robot_worldF_tgt.to_transformation_matrix()  # (B, 4, 4)

        # For 2D rotation around Z-axis: theta = atan2(R[1,0], R[0,0])
        curr_theta = torch.atan2(curr_mat[:, 1, 0], curr_mat[:, 0, 0])  # (B,)
        tgt_theta = torch.atan2(tgt_mat[:, 1, 0], tgt_mat[:, 0, 0])  # (B,)

        # Calculate angular difference (handle wrap-around)
        dtheta_total = angle_diff_torch(tgt_theta, curr_theta)  # (B,)

        # Build interpolated trajectory
        traj: list[torch.Tensor] = []
        for i in range(1, interpolate_steps + 1):
            alpha = i / interpolate_steps

            # Interpolated deltas
            dx = dx_total * alpha  # (B,)
            dy = dy_total * alpha
            dtheta = dtheta_total * alpha

            # Apply deltas to body qpos
            new_body_qpos = curr_body_qpos.clone()  # (B, 3)
            new_body_qpos[:, 0] = curr_body_qpos[:, 0] + dx
            new_body_qpos[:, 1] = curr_body_qpos[:, 1] + dy
            new_body_qpos[:, 2] = curr_body_qpos[:, 2] + dtheta

            # Concatenate: [body_qpos, arm_joints, gripper]
            full_qpos = torch.cat([new_body_qpos, arm_joints, gripper_pos], dim=1)
            traj.append(full_qpos)

        return traj


class WaypointTracker:
    """Tracks waypoints and computes delta actions for feedback control."""

    def __init__(
        self,
        plan: list[torch.Tensor],
        normalize_action: bool,
        arm_action_low: torch.Tensor,
        arm_action_high: torch.Tensor,
        angular_threshold: float,
        waypoint_threshold: float,
        device: torch.device,
    ):
        """Initialize waypoint tracker.

        Args:
            plan: List of target joint configurations.
            normalize_action: Whether to normalize actions.
            arm_action_low: Lower bound for action normalization.
            arm_action_high: Upper bound for action normalization.
            angular_threshold: Threshold for angular error (radians).
            waypoint_threshold: Threshold for position error (meters).
            device: Device for tensor operations.
        """
        self.plan = plan
        self.normalize_action = normalize_action
        self.arm_action_low = arm_action_low
        self.arm_action_high = arm_action_high
        self.angular_threshold = angular_threshold
        self.waypoint_threshold = waypoint_threshold
        self.device = device
        self.current_target: torch.Tensor | None = None
        self.curent_try_count = 0
        self.max_try_count = CFG.c_drawer_waypoint_max_try_count

    def subgoal_achieved(self, curr_q_pos: torch.Tensor) -> torch.Tensor:
        """Check if the current subgoal is achieved.

        Args:
            curr_q_pos: Current joint positions (B, 10).
        Returns:
            Boolean tensor (B,) indicating if subgoal is achieved.
        """
        if self.current_target is None:
            return torch.ones(curr_q_pos.shape[0], dtype=torch.bool, device=self.device)

        # Check if current waypoint is achieved (for all environments)
        # Only consider the arm joints (first 6 DOF), ignore gripper
        distance_angle = torch.norm(
            curr_q_pos[:, 2:FINGER_ACTION_INDEX]
            - self.current_target[:, 2:FINGER_ACTION_INDEX],
            dim=-1,
        )
        distance_pos = torch.norm(
            curr_q_pos[:, :2] - self.current_target[:, :2], dim=-1
        )
        achieved = (distance_angle < self.angular_threshold) & (
            distance_pos < self.waypoint_threshold
        )
        if achieved.all():
            self.curent_try_count = 0
        else:
            self.curent_try_count += 1
            if self.curent_try_count >= self.max_try_count:
                # logging.warning(
                #     f"Waypoint not achieved after {self.max_try_count} tries, skipping to next."
                # )
                achieved = torch.ones(
                    curr_q_pos.shape[0], dtype=torch.bool, device=self.device
                )
                self.curent_try_count = 0
        return achieved

    def compute_delta_actions(self, curr_q_pos: torch.Tensor) -> torch.Tensor:
        """Compute delta actions toward current waypoint with feedback control.

        Args:
            curr_q_pos: Current joint positions (B, 10).

        Returns:
            Delta actions (B, 10) to move toward current waypoint.
        """

        # If no current target, get the first one from the plan
        if self.current_target is None:
            if len(self.plan) == 0:
                raise RuntimeError("Plan completed, no more waypoints.")
            self.current_target = self.plan.pop(0)
        else:
            # Check if current waypoint is achieved (for all environments)
            # Only consider the arm joints (first 6 DOF), ignore gripper

            # If all environments are close enough, advance to next waypoint
            if self.subgoal_achieved(curr_q_pos=curr_q_pos).all():
                if len(self.plan) == 0:
                    # Plan is complete, return zero action
                    raise RuntimeError("Plan completed, no more waypoints.")
                self.current_target = self.plan.pop(0)

        # Compute delta action toward current target
        assert self.current_target is not None  # Type narrowing for mypy
        delta_qpos = self.current_target - curr_q_pos

        # Normalize if required
        if self.normalize_action:
            # Normalize the delta_qpos to be within [low, high]
            low = self.arm_action_low.unsqueeze(0).repeat(curr_q_pos.shape[0], 1)
            high = self.arm_action_high.unsqueeze(0).repeat(curr_q_pos.shape[0], 1)
            delta_arm_qpos = (
                delta_qpos[:, 3:FINGER_ACTION_INDEX] - 0.5 * (low[:, 3:] + high[:, 3:])
            ) / (0.5 * (high[:, 3:] - low[:, 3:]))
            delta_base_qpos = (delta_qpos[:, :3] - 0.5 * (low[:, :3] + high[:, :3])) / (
                0.5 * (high[:, :3] - low[:, :3])
            )
            delta_qpos_norm = torch.cat(
                [
                    delta_base_qpos,
                    delta_arm_qpos,
                    self.current_target[
                        :, FINGER_ACTION_INDEX : FINGER_ACTION_INDEX + 1
                    ],
                ],
                dim=-1,
            )
            return delta_qpos_norm
        else:
            # gripper position is not delta, use absolute position
            delta_arm_qpos = delta_qpos[:, 3:FINGER_ACTION_INDEX].clone()
            delta_base_qpos = delta_qpos[:, :3].clone()
            delta_qpos_norm = torch.cat(
                [
                    delta_base_qpos,
                    delta_arm_qpos,
                    self.current_target[
                        :, FINGER_ACTION_INDEX : FINGER_ACTION_INDEX + 1
                    ],
                ],
                dim=-1,
            )
            return delta_qpos_norm


class SpotMotionRoom(SpotMotion):
    """Toolkit for Spot IK based motion skills."""

    def __init__(
        self,
        device: torch.device,
        gripper_scale: float = 0.2,
    ):
        """Initialize SpotMotion with joint limits.

        Args:
            min_limits: Tensor of shape (7,) with minimum joint limits.
            max_limits: Tensor of shape (7,) with maximum joint limits.
        """
        super().__init__(device=device, gripper_scale=gripper_scale)
        self.gripper_closed = -0.04
        self.gripper_open = 0.05
        self.gripper_closing_delta = (
            self.gripper_closed - self.gripper_open
        ) * gripper_scale
        self.gripper_openning_delta = -self.gripper_closing_delta
        self.shoulder_offset = torch.tensor(
            SHOULDER_OFFSET_ROOM, dtype=torch.float32, device=device
        )

    def build_grasp_pose(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        target_pose: Pose,
        current_pose: Pose,
        object_name: str,
    ) -> List[Pose]:
        """Build a grasp pose (spot_hand_frame)."""
        assert object_name == "bowl", "Only bowl grasp is implemented."
        # 1. Moving Close Pose, x-axis pointing to object
        rel_pos = torch.tensor(
            [[0.2, 0.0, 0.1]], dtype=torch.float32, device=self.device
        ).repeat(target_pose.p.shape[0], 1)
        rel_q = euler2quat(0, -np.pi / 6, 0)
        relative_pose = Pose.create_from_pq(rel_pos, q=rel_q)
        close_pose_pointing_robot = target_pose * relative_pose
        rel_pose_pointing_object = Pose.create_from_pq(
            torch.tensor(
                [[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
            ).repeat(target_pose.p.shape[0], 1),
            euler2quat(0.0, 0.0, np.pi),
        )
        close_pose_pointing_object = (
            close_pose_pointing_robot * rel_pose_pointing_object
        )
        pose1 = Pose.create_from_pq(
            p=close_pose_pointing_object.p,
            q=current_pose.q,
        )

        # 3. Move closer pose
        rel_pos_closer = torch.tensor(
            [[0.05, 0.0, 0]], dtype=torch.float32, device=self.device
        ).repeat(target_pose.p.shape[0], 1)
        rel_pose_closer = Pose.create_from_pq(
            rel_pos_closer,
            euler2quat(0, 0, 0),
        )
        hand_ready_pose = close_pose_pointing_object * rel_pose_closer

        pose2 = Pose.create_from_pq(
            p=hand_ready_pose.p,
            q=current_pose.q,
        )

        return [pose1, pose2]

    def build_navigate_pose(
        self,
        target_pose: Pose,
        facing_axis: str = "x",
        distance: float = 0.5,
    ) -> Pose:
        """Build a navigate pose (spot_base_frame) given a target object pose."""
        rel_pos = torch.zeros_like(target_pose.p)
        if facing_axis == "x":
            rel_pos[:, 0] = distance
            relative_pose = Pose.create_from_pq(rel_pos, q=euler2quat(0.0, 0.0, 0.0))
            pose = target_pose * relative_pose
            xyz = pose.p.clone()
            xyz[:, 2] = 0.0  # keep ground level
            return Pose.create_from_pq(
                p=xyz,
                q=euler2quat(0.0, 0.0, np.pi),
            )

        raise NotImplementedError("Only 'x' facing axis is implemented.")


class CarPDController:
    """A batched PD controller for the CarRobot using tensor observations.

    Computes forward and steering forces based on position and heading errors. Works
    with batched observations and returns batched control actions.
    """

    def __init__(
        self,
        kp_pos: float = 10.0,
        kv_pos: float = 5.0,
        kp_ang: float = 20.0,
        kv_ang: float = 5.0,
        device: torch.device | None = None,
    ) -> None:
        """Initialize PD controller with gains.

        Args:
            kp_pos: Proportional gain for position control
            kv_pos: Derivative gain for velocity damping
            kp_ang: Proportional gain for angular control
            kv_ang: Derivative gain for angular velocity damping
            robot_mass: Robot mass for force computation
            robot_moment: Robot moment of inertia for torque computation
            device: Torch device for computations
        """
        self.kp_pos = kp_pos
        self.kv_pos = kv_pos
        self.kp_ang = kp_ang
        self.kv_ang = kv_ang
        self.device = device if device is not None else torch.device("cpu")

    def compute_control(
        self,
        obs: torch.Tensor,
        tgt_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control forces using PD control with batched observations.

        Computes errors:
        - Position error: distance to target
        - Angular error: heading toward target
        - Velocity error: current velocity (zero target)
        - Angular velocity error: current angular velocity (zero target)

        Then computes desired accelerations and converts to forces using:
        - forward_force = mass * desired_linear_accel
        - steering_force = moment * desired_angular_accel

        Args:
            obs: Batched observation tensor (B, obs_dim) containing:
                 obs[..., 0] = x, obs[..., 1] = y, obs[..., 2] = theta,
                 obs[..., 3] = vx_base, obs[..., 4] = vy_base, obs[..., 5] = omega_base
            tgt_positions: Target positions (B, 2) as [x, y]
            dt: Simulation timestep for force computation

        Returns:
            Control actions (B, 2) as [forward_force, steering_force]
        """
        # Extract current state from observation
        # PDControl for now
        curr_pose = extract_robot_pose(obs)
        # curr_force = extract_robot_force(obs)
        curr_vel = extract_robot_vel(obs)
        robot_mass_moment = extract_robot_mass_moment(obs)

        # Target positions
        tgt_pos = tgt_pose[:, :2]  # (B, 2)
        # Target angle convert to -pi to pi
        tgt_angle = tgt_pose[:, 2].clone()
        angle_large_mask = tgt_angle > torch.pi
        angle_small_mask = tgt_angle < -torch.pi
        tgt_angle[angle_large_mask] -= 2 * torch.pi
        tgt_angle[angle_small_mask] += 2 * torch.pi

        # Position error (world frame)
        pos_error = tgt_pos - curr_pose[:, :2]  # (B, 2)
        ang_error = tgt_angle - curr_pose[:, 2]
        ang_error = (ang_error + torch.pi) % (2 * torch.pi) - torch.pi

        vel_trans_error = -curr_vel[:, :2]  # (B, 2)
        vel_ang_error = -curr_vel[:, 2]  # (B,)

        # force_trans_error = -curr_force[:, :2]  # (B, 2)
        # torque_error = -curr_force[:, 2]  # (B,)

        # PD control: compute desired accelerations
        # Desired linear acceleration (to reduce position error and velocity)
        desired_linear_accel = self.kp_pos * pos_error + self.kv_pos * vel_trans_error

        # Desired angular acceleration (to reduce heading error and angular velocity)
        desired_angular_accel = self.kp_ang * ang_error + self.kv_ang * vel_ang_error

        # NOTE: We assume the robot is either rotating in place or moving forward,
        # so the norm of the desired force is the forward force.
        desired_force = robot_mass_moment[:, 0:1].repeat(1, 2) * desired_linear_accel
        desired_force_rotated = rotate_vectors(desired_force, -curr_pose[:, 2])
        # assert desired_force_rotated[:, 0].abs().max() < 1e-3, "Rotate first!"
        desired_force_norm = desired_force_rotated[:, 1:]  # (B, 1) forward force only
        # Desired torque is Bx1
        desired_torque = (robot_mass_moment[:, 1] * desired_angular_accel).unsqueeze(-1)
        # Steering force (norm) is torque divided by steering length
        # Note that steering force is x-positive, which means negative rotation
        steering_force_value = -desired_torque / CFG.robot_steering_length

        # NOTE: Given skills only outputs forward force y
        forward_force_x = torch.zeros_like(desired_force_norm)

        # Stack into (B, 3) control tensor
        control = torch.cat(
            [desired_force_norm, forward_force_x, steering_force_value], dim=-1
        )

        return control


class WaypointTrackerRoom:
    """Tracks waypoints and computes delta actions for feedback control."""

    def __init__(
        self,
        plan: list[torch.Tensor],
        normalize_action: bool,
        arm_action_low: torch.Tensor,
        arm_action_high: torch.Tensor,
        angular_threshold: float,
        waypoint_threshold: float,
        device: torch.device,
    ):
        """Initialize waypoint tracker.

        Args:
            plan: List of target joint configurations.
            normalize_action: Whether to normalize actions.
            arm_action_low: Lower bound for action normalization.
            arm_action_high: Upper bound for action normalization.
            angular_threshold: Threshold for angular error (radians).
            waypoint_threshold: Threshold for position error (meters).
            device: Device for tensor operations.
        """
        self.plan = plan
        self.normalize_action = normalize_action
        self.arm_action_low = arm_action_low
        self.arm_action_high = arm_action_high
        self.angular_threshold = angular_threshold
        self.waypoint_threshold = waypoint_threshold
        self.device = device
        self.current_target: torch.Tensor | None = None
        self.curent_try_count = 0
        self.max_try_count = CFG.c_room_waypoint_max_try_count

    def subgoal_achieved(
        self, curr_q_pos: torch.Tensor, curr_q_vel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Check if the current subgoal is achieved.

        Args:
            curr_q_pos: Current joint positions (B, 10).
        Returns:
            Boolean tensor (B,) indicating if subgoal is achieved.
        """
        if self.current_target is None:
            return torch.ones(curr_q_pos.shape[0], dtype=torch.bool, device=self.device)

        # Check if current waypoint is achieved (for all environments)
        # Only consider the arm joints (first 6 DOF), ignore gripper
        if curr_q_vel is not None:
            robot_joint_vel_trans = curr_q_vel[:, :2]
            robot_joint_vel_rot = curr_q_vel[:, 3:]
            static = (torch.norm(robot_joint_vel_trans, dim=-1) < 0.01) & (
                torch.norm(robot_joint_vel_rot, dim=-1) < 0.01
            )
        else:
            static = torch.ones(
                curr_q_pos.shape[0], dtype=torch.bool, device=self.device
            )

        distance_angle = torch.norm(
            curr_q_pos[:, 2:9] - self.current_target[:, 2:9],
            dim=-1,
        )
        distance_pos = torch.norm(
            curr_q_pos[:, :2] - self.current_target[:, :2], dim=-1
        )
        distance_finger = torch.norm(
            curr_q_pos[:, 9:] - self.current_target[:, 9:], dim=-1
        )
        achieved = (
            (distance_angle < self.angular_threshold)
            & (distance_pos < self.waypoint_threshold)
            & (distance_finger < 0.001)
        )
        achieved = achieved & static
        if achieved.all():
            self.curent_try_count = 0
        else:
            self.curent_try_count += 1
            if self.curent_try_count >= self.max_try_count:
                # logging.warning(
                #     f"Waypoint not achieved after {self.max_try_count} tries, skipping to next."
                # )
                achieved = torch.ones(
                    curr_q_pos.shape[0], dtype=torch.bool, device=self.device
                )
                self.curent_try_count = 0
        return achieved

    def compute_delta_actions(
        self, curr_q_pos: torch.Tensor, curr_q_vel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute delta actions toward current waypoint with feedback control.

        Args:
            curr_q_pos: Current joint positions (B, 11).

        Returns:
            Delta actions (B, 10) to move toward current waypoint.
        """

        # If no current target, get the first one from the plan
        if self.current_target is None:
            if len(self.plan) == 0:
                return get_frozen_action(
                    curr_q_pos[:, :-1],
                    self.arm_action_low,
                    self.arm_action_high,
                    self.normalize_action,
                    CFG.control_mode,
                )
            self.current_target = self.plan.pop(0)
        else:
            # Check if current waypoint is achieved (for all environments)
            # Only consider the arm joints (first 6 DOF), ignore gripper

            # If all environments are close enough, advance to next waypoint
            if self.subgoal_achieved(
                curr_q_pos=curr_q_pos, curr_q_vel=curr_q_vel
            ).all():
                if len(self.plan) == 0:
                    # Plan is complete, return zero action
                    return get_frozen_action(
                        curr_q_pos[:, :-1],
                        self.arm_action_low,
                        self.arm_action_high,
                        self.normalize_action,
                        CFG.control_mode,
                    )
                self.current_target = self.plan.pop(0)

        # Compute delta action toward current target
        assert self.current_target is not None  # Type narrowing for mypy
        delta_qpos = self.current_target - curr_q_pos

        # Normalize if required
        if self.normalize_action:
            # Normalize the delta_qpos to be within [low, high]
            low = self.arm_action_low.unsqueeze(0).repeat(curr_q_pos.shape[0], 1)
            high = self.arm_action_high.unsqueeze(0).repeat(curr_q_pos.shape[0], 1)
            delta_arm_qpos = (delta_qpos[:, 3:9] - 0.5 * (low[:, 3:] + high[:, 3:])) / (
                0.5 * (high[:, 3:] - low[:, 3:])
            )
            delta_base_qpos = (delta_qpos[:, :3] - 0.5 * (low[:, :3] + high[:, :3])) / (
                0.5 * (high[:, :3] - low[:, :3])
            )
            delta_qpos_norm = torch.cat(
                [
                    delta_base_qpos,
                    delta_arm_qpos,
                    self.current_target[:, 9:10],
                ],
                dim=-1,
            )
            return delta_qpos_norm
        else:
            # gripper position is not delta, use absolute position
            delta_arm_qpos = delta_qpos[:, 3:9].clone()
            delta_base_qpos = delta_qpos[:, :3].clone()
            delta_qpos_norm = torch.cat(
                [
                    delta_base_qpos,
                    delta_arm_qpos,
                    self.current_target[:, 9:10],
                ],
                dim=-1,
            )
            return delta_qpos_norm
