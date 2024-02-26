from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Tuple
import numpy as np
from numpy import ndarray
from ik_rl.solver.base_solver import IKSolver


class RobotArm(ABC):
    def __init__(
        self, links: ndarray, solver_cls: type = None, solver_args: Dict[str, Any] = {}
    ) -> None:
        super().__init__()

        assert len(links) > 0
        self._num_joints = len(links)
        self._links = links

        self._arm_length = sum(links)
        # relative angles
        self._rel_angles: ndarray
        self._positions: ndarray

        if solver_cls is None:
            self._solver = None
        else:
            self._solver: IKSolver = solver_cls(self, **solver_args)

    def reset(self):
        self._rel_angles = np.zeros((self._num_joints))
        self.set_rel_angles(self._rel_angles)

    def forward(self, angles: ndarray) -> Tuple[ndarray, ndarray]:
        """do a forward kinematics step on a 2D robot arm

        Args:
            angles (ndarray): sequence of relative angles for each joint in radians.

        Returns:
            Tuple[ndarray, ndarray]: translations and rotations in homogeneous transformation for each joint
        """
        dim = self._positions.shape[1]
        positions = np.empty((self._num_joints + 1, dim + 1, dim + 1))
        positions[0] = np.eye(dim + 1)

        rot_mats = self._rotate_z(angles)
        translations = np.zeros((self._num_joints, dim))
        translations[:, 0] = self._links
        trans_mats = self._translate(translations)

        for idx, (r, t) in enumerate(zip(rot_mats, trans_mats)):
            positions[idx + 1] = positions[idx] @ r @ t

        translations = positions[:, :-1, -1]
        rotations = positions[:, :-1, :-1]
        return translations, rotations

    def backward(self, target: ndarray) -> Tuple[float, bool]:
        """solve inverse kinematics for a given target

        Args:
            target (ndarray): _description_

        Returns:
            Tuple[float, bool]. Error to target, solved
        """
        if self._solver is None:
            logging.error(
                "Inverse kinematics is not possible because there is no solver."
            )
            return

        # for the angles you have to pass in the relative angles between the joints.
        # To get the absolute angels you can use the cum sum function
        # To invert this you can use:
        # >>> z[1:] -= z[:-1].copy()

        res_angles = self._solver.solve(target=target)
        error = self._solver.dist_error
        solved = self._solver.solved

        # convert relative resulting angles (re_angles) into absolute angles
        # abs_angles = np.deg2rad(np.array(res_angles).cumsum())
        self.set_rel_angles(res_angles)

        return error, solved

    def set_rel_angles(self, angles: ndarray) -> None:
        """applies given action to the arm

        Args:
            angles (np.array): array with same length as number of joints
        """
        self._rel_angles = angles % (2 * np.pi)
        self._positions, _ = self.forward(angles)

    def set_abs_angles(self, angles: ndarray) -> None:
        """_summary_

        Args:
            angles (ndarray): _description_

        Returns:
            _type_: _description_
        """
        rel_angles = angles.copy()
        rel_angles[1:] -= rel_angles[:-1].copy()
        # abs_angles = np.cumsum(angles, axis=0)
        self.set_rel_angles(rel_angles)

    @staticmethod
    @abstractmethod
    def _rotate_z(thetas: ndarray) -> ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _translate(thetas: ndarray) -> ndarray:
        raise NotImplementedError

    @property
    def abs_angles(self) -> ndarray:
        abs_angles = np.cumsum(self._rel_angles)
        return abs_angles.copy()

    @property
    def rel_angles(self) -> ndarray:
        return self._rel_angles.copy()

    @property
    def n_joints(self) -> int:
        return self._num_joints

    @property
    def arm_length(self) -> float:
        return self._arm_length

    @property
    def positions(self) -> ndarray:
        return self._positions.copy()

    @property
    def end_position(self) -> ndarray:
        return self._positions[-1]


class RobotArm2D(RobotArm):
    """implementation of a robt arm in 2D space."""

    def __init__(
        self, links: ndarray, solver_cls: type = None, solver_args: Dict[str, Any] = {}
    ) -> None:
        super().__init__(links, solver_cls, solver_args)
        self._rel_angles: ndarray = np.zeros((self._num_joints))
        # 2D, the plus one dim is the origin
        self._positions = np.zeros((self._num_joints + 1, 2))
        # init _positions
        self.set_rel_angles(self._rel_angles)

    @staticmethod
    def _rotate_z(thetas: float) -> ndarray:
        """create 2d rotation matrix over x and y:\\
        [[math.cos(theta), -math.sin(theta), 0], \\
         [math.sin(theta), math.cos(theta), 0], \\
         [0, 0, 1]]

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

        zeros = np.zeros((num_thetas, 1, 2))
        identity = np.eye(1)[None].repeat(num_thetas, axis=0)

        rz = np.block([[rot_mat, zeros.swapaxes(-1, -2)], [zeros, identity]])
        return rz

    def _translate(self, vector: ndarray) -> ndarray:
        """create translation matrix (num_translation, 4, 4)
        Example:\\
        [[1, 0, dx], \\
         [0, 1, dy], \\
         [0, 0,  1]]

        Args:
            vector (ndarray): 1d translation vector. Shape (num_translation, translation_dim)

        Returns:
            ndarray: translation as augmented matrix. Shape: (num_translation, 4, 4)
        """
        num_samples, trans_dim = vector.shape
        trans = np.eye(3)[None].repeat(num_samples, axis=0)
        trans[:, :trans_dim, -1] = vector
        return trans


class RobotArm3D(RobotArm):
    def __init__(
        self, links: ndarray, solver_cls: type = None, solver_args: Dict[str, Any] = {}
    ) -> None:
        super().__init__(links, solver_cls, solver_args)
        self._rel_angles: ndarray = np.zeros(
            (self._num_joints, 2)
        )  # two angles to rotate on
        # 2D, the plus one dim is the origin
        self._positions = np.zeros((self._num_joints + 1, 3))
        # init _positions
        self.set_rel_angles(self._rel_angles)

        raise NotImplementedError

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
        # rot_mat = np.moveaxis(rot_mat, -1, 0)

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
