from abc import ABC, abstractmethod
import inspect

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

        # update docstring for each method
        _update_docstring = inspect.getdoc(self._reward)
        self.reward.__doc__ = _update_docstring
        _update_docstring = inspect.getdoc(self._done)
        self.done.__doc__ = _update_docstring

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

    def done(self, *args, **kwargs) -> bool:
        """definition of done. Either Episode exceeds time limit or

        Returns:
            bool: if episode is done or not
        """
        return self._exceeds_time_level() or self._done(*args, **kwargs)

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
