from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy import ndarray


class ObstacleGenerator(ABC):
    def __init__(self, amount_range: Tuple[int, int]) -> None:
        assert min(amount_range) >= 0
        self._amount_range = sorted(amount_range)

    def _get_number_obstacles(self) -> int:
        """how many obstacles would you like to have in the environment

        Returns:
            int: number of obstacles to generate
        """
        return np.random.randint(*self._amount_range)

    @abstractmethod
    def get_obstacles(self) -> ndarray:
        raise NotImplementedError


class CircleGenerator(ObstacleGenerator):
    def __init__(self, amount_range: Tuple[int], radius_range: Tuple[int, int]) -> None:
        super().__init__(amount_range)
        assert min(radius_range) >= 0
        self._radius_range = sorted(radius_range)

    def get_obstacles(self) -> ndarray:
        num_obstacles = self._get_number_obstacles()
        print(num_obstacles)


class PolygonGenerator(ObstacleGenerator):
    def __init__(self, amount_range: Tuple[int], num_edges: int) -> None:
        super().__init__(amount_range)
        self._num_edges = num_edges

    def get_obstacles(self) -> ndarray:
        num_obstacles = self._get_number_obstacles()
        print(num_obstacles)
