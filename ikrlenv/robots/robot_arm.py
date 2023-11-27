from typing import Tuple
import numpy as np
from numpy import ndarray

from ikrlenv.solver.ccd import ccd


class RobotArm:
    """implementation of a robt arm in 2D space. 
    """
    def __init__(self, n_joints: int = 1, segment_length: float = 1) -> None:
        self._n_joints = n_joints
        self._segment_length = segment_length

        self._arm_length = self._n_joints * segment_length
        # relative angles
        self._angles = np.zeros((self._n_joints))

        self._positions = np.zeros((self._n_joints + 1, 2))  # 2D, the plus one dim is the origin
        # init _positions
        self.set(self._angles)


    def reset(self):
        self._angles = np.zeros((self._n_joints))
        self.set(self._angles)

    def set(self, angles: ndarray):
        """applies given action to the arm 

        Args:
            angles (np.array): array with same length as number of joints
        """
        self._angles = angles % (2 * np.pi)

        for idx in range(self._n_joints):
            origin = self._positions[idx]
            
            # new position
            new_pos = np.array([np.cos(self._angles[idx]), np.sin(self._angles[idx])])
            # new_pos = np.array([np.cos(cumsum[idx]), np.sin(cumsum[idx])])
            new_pos *= self._segment_length

            # translate position
            new_pos += origin

            self._positions[idx + 1] = new_pos

    def inv_kin(self, target: ndarray, max_iter: int = 1000, error_min: float = 0.1) -> Tuple[float, bool, int]:
        """solve inverse kinematics for a given target

        Args:
            target (ndarray): _description_
            max_iter (int): maximum iteration for ccd solver. Defaults to 1000.
            error_min (float): error threshold for solving the inverse kinematics problem. Defaults to 0.1.

        Returns:
            Tuple[float, bool, int]. Error to target, solved, number of iterations needed
        """
        # for the angles you have to pass in the relative angles between the joints.
        # To get the absolute angels you can use the cum sum function 
        # To invert this you can use:
        # >>> z[1:] -= z[:-1].copy()
        rel_angles = self.angles.copy()
        # transform into degrees
        rel_angles = rel_angles / np.pi * 180
        rel_angles[1:] -= rel_angles[:-1].copy()

        # link is a sequence of individual segment lengths
        # in our implementation the segment length is constant over all segments
        link = np.ones(self.n_joints) * self._segment_length

        res_angles, error, solved, num_iter = ccd(target, rel_angles, link, max_iter=max_iter, err_min=error_min)

        # convert relative resulting angles (re_angles) into absolute angles
        abs_angles = np.deg2rad(np.array(res_angles).cumsum())
        self.set(abs_angles)

        return error, solved, num_iter

    @property
    def angles(self):
        return self._angles

    @property
    def n_joints(self):
        return self._n_joints
    
    @property
    def arm_length(self):
        return self._arm_length

    @property
    def positions(self):
        return self._positions
    
    @property
    def end_position(self):
        return self._positions[-1]

