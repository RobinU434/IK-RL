from gymnasium import RewardWrapper

from ik_rl.environment import _InvKinEnv


class NormalizeRewardWrapper(RewardWrapper):
    def __init__(self, env: _InvKinEnv):
        super().__init__(env)
        self.env: _InvKinEnv

    def reward(self, reward):
        factor = 1 / (self.env._robot_arm.arm_length)
        return reward * factor