from gymnasium import Env, RewardWrapper, ActionWrapper, Wrapper
from gymnasium.spaces import Box
from ik_rl.environment import _InvKinEnv, InvKinEnvContinuous


class NormalizeRewardWrapper(RewardWrapper):
    def __init__(self, env: _InvKinEnv, robot_arm_length: float):
        super().__init__(env)
        self.robot_arm_length = robot_arm_length

    def reward(self, reward):
        factor = 1 / (self.robot_arm_length)
        return reward * factor
    

class ActionScaleWrapper(ActionWrapper):
    def __init__(self, env: Env, percentage: float = 1):
        self.percentage = percentage
        action_space = Box(
            low=env.action_space.low * percentage,
            high=env.action_space.high * percentage,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
            seed=env.action_space._np_random
        )
        env.action_space = action_space
        super().__init__(env)
        
    def action(self, action):
        # assert action in original space
        # down or upscale action 
        action = self.percentage * action
        return action
    
    def check_for_continuous(self, env: Env):
        if issubclass(env, Wrapper):
            env: Wrapper
            self.check_for_continuous(env.env)
        else:
            msg = f"{type(self).__name__} is only applicable to {InvKinEnvContinuous.__name__}."
            assert isinstance(env, InvKinEnvContinuous), msg
            
