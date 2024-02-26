import numpy as np
from numpy import ndarray
from ik_rl.task import NUM_TIME_STEPS
from ik_rl.task.base_task import BaseTask
from ik_rl.robots.robot_arm import RobotArm


class ImitationTask(BaseTask):
    def __init__(
        self,
        robot_arm: RobotArm,
        n_time_steps: int = NUM_TIME_STEPS,
        order: float = 2,
        epsilon: float = 0.01,
        **kwargs
    ) -> None:
        super().__init__(epsilon, n_time_steps, **kwargs)

        self._robot_arm = robot_arm
        self._target_pos = np.zeros(2)
        self._target_angles: ndarray

        self._order = order

    def _reward(
        self, target_position: ndarray, robot_arm_angles: ndarray, **kwargs
    ) -> float:
        """_summary_

        Args:
            target_position (ndarray): position which the robot arm should reach by learning to imitate the solver
            robot_arm_angles (ndarray): arm angles

        Returns:
            float: _description_
        """
        if (target_position != self._target_pos).any():
            self._update_target_angles(target_position)

        # MSE between target angles and current arm angles
        loss = -np.power(
            self.angle_diff(self._target_angles, robot_arm_angles), self._order
        ).mean()

        return loss

    def _update_target_angles(self, target_position: ndarray):
        # new target position
        self._target_pos = target_position
        self._robot_arm.reset()
        # apply inverse kinematics
        self._robot_arm.backward(target_position)

        self._target_angles = self._robot_arm.abs_angles
        # squash target angles because of the tanh function in PolicyNet.forward()
        # is a contradiction with the real_action unsqueeze function in PolicyNet.forward function
        # self.target_angles = (self.target_angles - np.pi) / np.pi

    @staticmethod
    def angle_diff(a: ndarray, b: ndarray):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + np.pi) % (2 * np.pi) - np.pi

    def _done(self, arm_position: ndarray, target_position: ndarray):
        return self._is_near_target(arm_position, target_position)

    def _is_near_target(self, arm_position: ndarray, target_position: ndarray) -> bool:
        return np.linalg.norm(arm_position - target_position).item() <= self._epsilon
