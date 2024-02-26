# =========================================================================================================
# code was copied and adapted from:
# https://github.com/ekorudiawan/CCD-Inverse-Kinematics-2D/blob/master/sources/CCD-Inverse-Kinematics-2D.py
# =========================================================================================================
import numpy as np
from numpy import ndarray
from ik_rl.robots.robot_arm import RobotArm

from ik_rl.solver.base_solver import IKSolver
from ik_rl.utils.geometry import angle_between_2D, angle_between_3D
from itertools import product


class CCD(IKSolver):
    def __init__(
        self, robot: RobotArm, max_iter: int = 10000, err_min: float = 0.1
    ) -> None:
        super().__init__(num_joints=robot.n_joints)
        self._robot = robot
        self._max_iter = max_iter
        self._err_min = err_min
        # link is a sequence of individual segment lengths
        # in our implementation the segment length is constant over all segments
        self._links = np.ones(self._robot.n_joints) * self._robot._links

        # value to zero-vector detection
        self._epsilon = 1e-10
        self._loop: int

    def solve(self, target: ndarray) -> ndarray:
        solved = False
        self._dist_error = np.inf
        angles = self._robot.rel_angles.copy()
        angle_func = (
            angle_between_3D
            if self._robot.positions.shape[1] == 3
            else angle_between_2D
        )
        len_arm = len(self._links)

        loop_iter = range(self._max_iter)
        joint_iter = range(len_arm - 1, -1, -1)
        for loop, i in product(loop_iter, joint_iter):
            # shape: (num_links + 1, 3)
            link_positions, _ = self._robot.forward(angles)
            self._dist_error = np.linalg.norm(target - link_positions[-1])
            if self._dist_error < self._err_min:
                solved = True
                break

            # Calculate distance between i-joint position to end effector position
            current_to_end = link_positions[-1] - link_positions[i]
            current_to_target = target - link_positions[i]

            # perform 0 vector detection
            if (
                np.linalg.norm(current_to_end) * np.linalg.norm(current_to_target)
                < 1e-10
            ):
                # if one of those two vectors is close to a zero vector set rotation to 0
                current_rotation = 0
            else:
                # angle between current joint to end and current joint to target
                current_rotation = angle_func(current_to_end, current_to_target)

            # Update current joint angle values by adding the delta angle to the current angle
            angles[i] += current_rotation
            angles[i] = angles[i] % (2 * np.pi)

        self._loop = loop
        self._solved = bool(solved)

        return angles

    @property
    def loop(self) -> int:
        return self._loop
