from ik_rl.environment import InvKinDiscrete, InvKinEnvContinous
from ik_rl.task import ReachGoalTask


def test_init():
    segment_length = 1
    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinEnvContinous(task=task, robot_config={"n_joints": n_joints})

    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinDiscrete(task=task, robot_config={"n_joints": n_joints})


def test_reset():
    segment_length = 1
    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinEnvContinous(task=task, robot_config={"n_joints": n_joints})
    env.reset()

    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinDiscrete(task=task, robot_config={"n_joints": n_joints})
    env.reset()


def test_step():
    segment_length = 1
    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinEnvContinous(task=task, robot_config={"n_joints": n_joints})
    action = env.action_space.sample()
    env.step(action)

    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinDiscrete(task=task, robot_config={"n_joints": n_joints})
    action = env.action_space.sample()
    env.step(action)


def test_render():
    segment_length = 1
    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinEnvContinous(task=task, robot_config={"n_joints": n_joints})
    env.render()

    n_joints = 2
    task = ReachGoalTask(arm_reach=n_joints * segment_length)
    env = InvKinDiscrete(task=task, robot_config={"n_joints": n_joints})
    env.render()
