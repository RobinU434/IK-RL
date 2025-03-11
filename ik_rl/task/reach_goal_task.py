import numpy as np
from numpy import ndarray
from ik_rl.task.config import NUM_TIME_STEPS
from ik_rl.task.base_task import _BaseTask


class ReachGoalTask(_BaseTask):
    def __init__(
        self,
        epsilon: float = 0.1,
        n_time_steps: int = NUM_TIME_STEPS,
        bonus: int = 0,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            arm_reach (float): how long is the robot arm
            epsilon (float, optional): when to have reached the goal successfully. Defaults to 0.1.
            n_time_steps (int, optional): . Defaults to NUM_TIME_STEPS.
            bonus (int, optional): _description_. Defaults to 0.
            normalize (bool, optional): _description_. Defaults to True.
        """
        super().__init__(epsilon, n_time_steps, **kwargs)
        self._bonus = bonus
        """float: bonus if arm reached target position"""

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
        norm_target_distance = np.linalg.norm(arm_position - target_position).item()
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
