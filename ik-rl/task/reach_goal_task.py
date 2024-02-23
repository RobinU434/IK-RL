import numpy as np
from numpy import ndarray
from ikrlenv.task import NUM_TIME_STEPS
from ikrlenv.task.base_task import BaseTask


class ReachGoalTask(BaseTask):
    def __init__(
        self,
        arm_reach: float,
        epsilon: float = 0.1,
        n_time_steps: int = NUM_TIME_STEPS,
        bonus: int = 0,
        normalize: bool = True,
        **kwargs
    ) -> None:
        super().__init__(epsilon, n_time_steps, **kwargs)
        self._normalization_factor = self._get_normalize_factor(
            arm_reach=arm_reach, normalize=normalize
        )
        self._bonus = bonus
        """float: bonus if arm reached target position"""

    def _get_normalize_factor(self, arm_reach: float, normalize: bool = True) -> float:
        """to fit the mean distance value into the interval [0, 1]. Defaults to 1.0.

        Args:
            arm_reach (float): How far a arm can reach.
            normalize (bool, optional): _description_. Defaults to True.

        Returns:
            float: 1 / (arm_reach * 2) if normalize == True otherwise 1.0
        """
        if normalize:
            return 1 / (arm_reach * 2)
        return 1.0

    def _reward(
        self, arm_position: ndarray, target_position: ndarray, **kwargs
    ) -> float:
        """internal reward function

        Args:
            arm_position (ndarray): end effector position in 2D
            goal_position (ndarray): target position in 2D space around the origin with max radius = arm.length

        Returns:
            float: distance reward
        """
        # add bonus if the arm has reached its desired target
        bonus = self._bonus * self._is_near_target(arm_position, target_position)
        norm_target_distance = (
            np.linalg.norm(arm_position - target_position).item()
            * self._normalization_factor
        )
        reward = -norm_target_distance + bonus

        return reward

    def _done(self, arm_position: ndarray, target_position: ndarray, **kwargs) -> bool:
        """indicate if arm end effector position is near the target position

        Args:
            arm_position (ndarray): arm end effector position
            target_position (ndarray): target where the arm should go

        Returns:
            bool: _description_
        """
        return self._is_near_target(arm_position, target_position)

    def _is_near_target(self, arm_position: ndarray, target_position: ndarray) -> bool:
        """if arm position is near goal position -> return True, else False

        Args:
            arm_position (ndarray): position of arm. Shape: (num_target_dim)
            goal_position (ndarray): goal position. Shape: (num_target_dim)

        Returns:
            bool: true if arm position is new goal position
        """
        if np.linalg.norm(arm_position - target_position) <= self._epsilon:
            return True
        else:
            return False
