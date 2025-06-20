from gymnasium import RewardWrapper, ActionWrapper
from gymnasium.spaces import Box
from ik_rl.environment import _InvKinEnv, InvKinEnvContinuous


class NormalizeRewardWrapper(RewardWrapper):
    def __init__(self, env: _InvKinEnv):
        super().__init__(env)
        self.env: _InvKinEnv

    def reward(self, reward):
        factor = 1 / (self.env._robot_arm.arm_length)
        return reward * factor
    

class ConstrainActionSpaceWrapper(ActionWrapper):
    def __init__(self, env, percentage: float = 1):
        assert isinstance(env, InvKinEnvContinuous), "Only applicable to Continuous environment."
        self.percentage = percentage
        action_space = Box(
            low=env.action_space.low * percentage,
            high=env.action_space.high * percentage,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
            seed=env.action_space.seed
        )
        env.action_space = action_space
        super().__init__(env)
        
    def action(self, action):
        # assert action in original space
        # down or upscale action 
        action = self.percentage * action
        return action
