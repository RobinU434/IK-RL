import numpy as np
from numpy import ndarray
from envs.task.base_task import BaseTask


class ReachGoalTask(BaseTask):
    def __init__(self, epsilon: float, n_time_steps: int, n_joints: int, bonus: int = 0) -> None:
        super().__init__(epsilon, n_time_steps)
        self._normalization_factor = 1 / (n_joints * 2)
        """float: to fit the mean distance value into the interval [0, 1]. Defaults to 1.0."""
        self._bonus = bonus
        """float: bonus if arm reached target position"""

    def _reward(self, arm_position: ndarray, target_position: ndarray, **kwargs) -> float:
        """internal reward function

        Args:
            arm_position (ndarray): end effector position in 2D 
            goal_position (ndarray): target position in 2D space around the origin with max radius = arm.length

        Returns:
            float: distance reward
        """
        bonus = self._bonus * self._is_near_target(arm_position, target_position)
        norm_target_distance = np.linalg.norm(arm_position - target_position).item() * self._normalization_factor
       
        return -norm_target_distance + bonus

    def _done(self, arm_position: ndarray, target_position: ndarray, **kwargs) -> bool:
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
