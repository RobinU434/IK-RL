from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np


class IKSolver(ABC):
    def __init__(self, num_joints: int) -> None:
        self._num_joints = num_joints

        self._solved: bool
        self._dist_error = np.inf

    @abstractmethod
    def solve(self, angles: ndarray, target: ndarray) -> ndarray:
        raise NotImplementedError

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def dist_error(self) -> float:
        return self._dist_error
