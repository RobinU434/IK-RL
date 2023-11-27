from abc import ABC, abstractmethod
from numpy import ndarray

class IKSolver(ABC):
    def __init__(self, num_joints: int) -> None:
        self._num_joints = num_joints

        self._solved: bool 

    @abstractmethod
    def solve(self, angles: ndarray, target: ndarray) -> ndarray:
        raise NotImplementedError
    

    @property
    def solved(self) -> bool:
        return self._solved