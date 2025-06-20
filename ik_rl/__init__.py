from gymnasium import register

register(
    id="IKEnv-2D-v0",
    entry_point="ik_rl.environment:InvKinDiscrete",
)

register(
    id="IKEnv-2D-v1",
    entry_point="ik_rl.environment:InvKinEnvContinuous",
)
