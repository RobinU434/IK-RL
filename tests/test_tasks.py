import numpy as np
from ik_rl.robots.robot_arm import RobotArm2D
from ik_rl.solver.ccd import CCD
from ik_rl.task.imitation_task import ImitationTask
from ik_rl.task.reach_goal_task import ReachGoalTask


def test_task_init():
    for n_joints in [1, 2, 50]:
        robot_arm = RobotArm2D(links=np.ones(n_joints), solver_cls=CCD)
        ImitationTask(robot_arm=robot_arm)

    for n_joints in [1, 2, 50]:
        robot_arm = RobotArm2D(links=np.ones(n_joints), solver_cls=CCD)
        ReachGoalTask(arm_reach=robot_arm.arm_length)


def test_properties_imitation_task():
    num_joints = 2
    robot = RobotArm2D(links=np.ones(num_joints), solver_cls=CCD)
    task = ImitationTask(robot_arm=robot)

    task.reset()
    assert task.step_counter == 0

    reward = task.reward(target_position=np.array([2, 0]), robot_arm_angles=np.zeros(2))
    assert reward == 0
    reward = task.reward(target_position=np.array([0, 0]), robot_arm_angles=np.zeros(2))
    assert reward == np.pi**2 / -2

    assert task.step_counter == 2

    assert task.done(np.array([2, 0]), np.array([2, 0]))
    assert not task.done(np.array([2, 0]), np.array([0, 2]))


def test_properties_reach_goal():
    num_joints = 2
    robot = RobotArm2D(links=np.ones(num_joints), solver_cls=CCD)
    task = ImitationTask(robot_arm=robot)

    task.reset()
    assert task.step_counter == 0

    reward = task.reward(target_position=np.array([2, 0]), robot_arm_angles=np.zeros(2))
    assert reward == 0
    assert task.step_counter == 1

    assert task.done(np.array([2, 0]), np.array([2, 0]))
    assert not task.done(np.array([2, 0]), np.array([0, 2]))
