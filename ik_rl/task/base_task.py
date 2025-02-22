from abc import ABC, abstractmethod
from typing import Tuple

from ik_rl.task import NUM_TIME_STEPS


class BaseTask(ABC):
    """Base class for any Task you want ot submit to the plane robot environment"""

    def __init__(
        self, epsilon: float = 0.01, n_time_steps: int = NUM_TIME_STEPS, **kwargs
    ) -> None:
        self._epsilon = epsilon
        """float tolerance between target and action outcome"""
        self._n_time_steps = n_time_steps
        """int: maximum number of time steps for an episode"""
        self._step_counter = 0
        """int: counter how often self.reward was called"""

    def _update(self):
        self._step_counter += 1

    def reset(self):
        self._step_counter = 0

    def reward(self, *args, **kwargs) -> float:
        """returns reward of reward

        Returns:
            float: calls self._reward
        """
        self._update()
        return self._reward(*args, **kwargs)

    @abstractmethod
    def _reward(self, *args) -> float:
        """custom reward definition"""
        raise NotImplementedError

    def done(self, *args, **kwargs) -> Tuple[bool | bool]:
        """definition of done. Either Episode exceeds time limit or

        Returns:
            tuple(bool, book): truncated -> exceeds time limits or other bounds, done -> if the agent has completed the task
        """
        return self._exceeds_time_level(), self._done(*args, **kwargs)

    def _exceeds_time_level(self) -> bool:
        """returns true if update was called more often than

        Returns:
            bool: if time limit was exceeded
        """
        if self._step_counter >= self._n_time_steps:
            return True
        return False

    @abstractmethod
    def _done(self, *args, **kwargs) -> bool:
        """custom definition of done by task"""
        raise NotImplementedError

    @property
    def step_counter(self) -> int:
        return self._step_counter
