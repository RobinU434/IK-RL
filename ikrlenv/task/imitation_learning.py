import numpy as np
from envs.task.base_task import BaseTask
from envs.robots.robot_arm import RobotArm

class ImitationTask(BaseTask):
    def __init__(self, config) -> None:
        epsilon = config["target_epsilon"]
        n_time_steps = config["n_time_steps"]
        
        super().__init__(epsilon, n_time_steps)

        self.robot_arm = RobotArm(config["n_joints"])

        self.target_pos = np.zeros(2)
        self.target_angles = np.zeros(config["n_joints"])

        
    def _reward(self, target_position, robot_arm_angles):
        if (target_position != self.target_pos).any():
            # new target position
            self.target_pos = target_position
            self.robot_arm.reset()
            # bump up target position
            target_position = np.concatenate([target_position, np.zeros(1)])
            # apply inverse kinematics
            self.robot_arm.IK(target_position, error_min=self._epsilon)
            self.target_angles = self.robot_arm.angles
            # squash target angles because of the tanh function in PolicyNet.forward()
            # is a contradiction with the real_action unsquash function in PolicyNet.forward function
            # self.target_angles = (self.target_angles - np.pi) / np.pi
        
        # print("____")
        # if len(action.size()) > 1:
        #     print(target_angles)
        #     print(robot_arm_angles)
        # else:
        #     print(target_angles)
        #     print(robot_arm_angles)

        # print(self.target_angles)
        # print(robot_arm_angles)
        # TODO: make env easier
        # self.target_angles = np.zeros_like(self.target_angles)
        
        # TODO: make env easier
        # if np.linalg.norm(target_position) > 1:
        #     self.target_angles = np.zeros_like(self.target_angles)
        # else:
        #     self.target_angles = np.ones_like(self.target_angles) * 0.5

        # MSE between target angles and current arm angles
        loss = - np.power(self.angle_diff(self.target_angles, robot_arm_angles), 2).mean()
        
        return loss

    @staticmethod
    def angle_diff(a : np.array, b: np.array):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + np.pi) % (2 * np.pi) - np.pi 
    
    def done(self, arm_position, goal_position):
        time_limit_reached = super().done()
        is_near_target = self.is_near_target(arm_position, goal_position)

        if is_near_target or time_limit_reached:
            return True
        else:
            return False

    def is_near_target(self, arm_position, goal_position):
        if np.linalg.norm(arm_position - goal_position) <= self._epsilon:
            return True
        else:
            return False

        