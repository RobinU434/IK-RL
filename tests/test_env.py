from ik_rl.environment import InvKinDiscrete, InvKinEnvContinuous

def test_init():
    segment_length = 1
    n_joints = 2
    InvKinEnvContinuous(n_joints=n_joints, segment_length=segment_length)
    InvKinDiscrete(n_joints=n_joints, segment_length=segment_length)


def test_reset():
    segment_length = 1
    n_joints = 2
    env = InvKinEnvContinuous(n_joints=n_joints, segment_length=segment_length)
    env.reset()
    env = InvKinDiscrete(n_joints=n_joints, segment_length=segment_length)
    env.reset()


def test_step():
    segment_length = 1
    n_joints = 2
    env = InvKinEnvContinuous(n_joints=n_joints, segment_length=segment_length)
    action = env.action_space.sample()
    env.step(action)

    env = InvKinDiscrete(n_joints=n_joints, segment_length=segment_length)
    action = env.action_space.sample()
    env.step(action)


def test_render():
    segment_length = 1
    n_joints = 2
    env = InvKinEnvContinuous(n_joints=n_joints, segment_length=segment_length)
    env.render()
    env = InvKinDiscrete(n_joints=n_joints, segment_length=segment_length)
    env.render()
