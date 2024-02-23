# =========================================================================================================
# code was copied and adapted from:
# https://github.com/ekorudiawan/CCD-Inverse-Kinematics-2D/blob/master/sources/CCD-Inverse-Kinematics-2D.py
# =========================================================================================================
import numpy as np
from numpy import ndarray

from ikrlenv.solver.base_solver import IKSolver
from ikrlenv.utils.geometry import angle_between


class CCD(IKSolver):
    def __init__(
        self, links: ndarray, max_iter: int = 10000, err_min: float = 0.1
    ) -> None:
        super().__init__(num_joints=len(links))

        self._max_iter = max_iter
        self._err_min = err_min
        self._links = links

        # value to zero-vector detection
        self._epsilon = 1e-10

        self._loop: int

    def solve(self, angles: ndarray, target: ndarray) -> ndarray:
        solved = False
        dist_error = np.inf

        len_arm = len(self._links)

        for loop in range(self._max_iter):
            for i in range(len_arm - 1, -1, -1):
                # shape: (num_links + 1, 3)
                link_positions = self._fk(angles)[:, :3, 3]
                dist_error = np.linalg.norm(target - link_positions[-1])
                if dist_error < self._err_min:
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
                    current_rotation = angle_between(current_to_end, current_to_target)

                # Update current joint angle values by adding the delta angle to the current angle
                angles[i] += current_rotation
                angles[i] = angles[i] % (2 * np.pi)

            if solved:
                break

        self._loop = loop
        self._solved = bool(solved)

        return angles

    @staticmethod
    def _rotate_z(thetas: float) -> ndarray:
        """create 2d rotation matrix over x and y:\\
        [[math.cos(theta), -math.sin(theta), 0, 0], \\
         [math.sin(theta), math.cos(theta), 0, 0], \\
         [0, 0, 1, 0], \\
         [0, 0, 0, 1],]

        Args:
            thetas (float): angles to rotate around z in radians. Shape (num_angles)

        Returns:
            ndarray: ration matrix over x and y. Shape (num_angles, 4, 4)
        """
        num_thetas = len(thetas)
        sin = np.sin(thetas)
        cos = np.cos(thetas)

        first_row = np.stack([cos, -sin])
        second_row = np.stack([sin, cos])

        rot_mat = np.stack([first_row, second_row])
        rot_mat = np.moveaxis(rot_mat, -1, 0)

        zeros = np.zeros((num_thetas, 2, 2))
        identity = np.eye(2)[None].repeat(num_thetas, axis=0)

        rz = np.block([[rot_mat, zeros], [zeros, identity]])
        return rz

    def _translate(self, vector: ndarray) -> ndarray:
        """create translation matrix (num_translation, 4, 4)
        Example:\\
        [[1, 0, 0, dx], \\
         [0, 1, 0, dy], \\
         [0, 0, 1, dz], \\
         [0, 0, 0,  1]] 

        Args:
            vector (ndarray): 1d translation vector. Shape (num_translation, translation_dim)

        Returns:
            ndarray: translation as augmented matrix. Shape: (num_translation, 4, 4)
        """
        num_samples, trans_dim = vector.shape
        trans = np.eye(4)[None].repeat(num_samples, axis=0)
        trans[:, :trans_dim, -1] = vector
        return trans

    def _fk(self, angles: ndarray) -> ndarray:
        """do a forward kinematics step on a 2D robot arm

        Args:
            angles (ndarray): sequence of angles for each joint in radians

        Returns:
            ndarray: _description_
        """
        P = []
        P = np.empty((self._num_joints + 1, 4, 4))
        P[0] = np.eye(4)

        rot_mats = self._rotate_z(angles)
        translations = np.zeros((self._num_joints, 3))
        translations[:, 0] = self._links
        trans_mats = self._translate(translations)

        for idx, (r, t) in enumerate(zip(rot_mats, trans_mats)):
            # print(r, t)
            P[idx + 1] = P[idx] @ r @ t
        return P

    @property
    def loop(self) -> int:
        return self._loop
