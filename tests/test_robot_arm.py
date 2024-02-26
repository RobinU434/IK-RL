import numpy as np
import pytest

from ik_rl.robots.robot_arm import RobotArm2D, RobotArm3D
from ik_rl.solver.ccd import CCD


def test_robot_arm_init():
    with pytest.raises(AssertionError):
        RobotArm2D(links=[])
    # with pytest.raises(AssertionError):
    #     RobotArm3D(links=[])

    for n_joints in [5, 2, 100]:
        print(n_joints)
        links = np.ones(n_joints)
        RobotArm2D(links=links)
        # RobotArm3D(links=links)


def test_angle_set():
    # abs_angles
    num_links = 4
    links = np.ones(4)
    robot = RobotArm2D(links=links)
    angles = np.zeros(num_links)
    robot.set_rel_angles(angles)
    np.testing.assert_allclose(robot.end_position, np.array([num_links, 0]), atol=1e-10)

    robot.set_rel_angles(angles)
    np.testing.assert_allclose(robot.end_position, np.array([num_links, 0]), atol=1e-10)

    angles = np.zeros(num_links)
    angles[0] = np.pi / 2
    robot.set_rel_angles(angles)
    np.testing.assert_allclose(robot.end_position, np.array([0, num_links]), atol=1e-10)

    angles = np.cumsum(angles)
    robot.set_abs_angles(angles)
    np.testing.assert_allclose(robot.end_position, np.array([0, num_links]), atol=1e-10)

    angles = np.zeros(num_links)
    angles[0] = np.pi
    robot.set_rel_angles(angles)
    np.testing.assert_allclose(
        robot.end_position, np.array([-num_links, 0]), atol=1e-10
    )

    angles = np.cumsum(angles)
    robot.set_abs_angles(angles)
    np.testing.assert_allclose(
        robot.end_position, np.array([-num_links, 0]), atol=1e-10
    )


def test_foreward_kinematics():
    num_links = 4
    links = np.ones(num_links)
    robot = RobotArm2D(links=links)
    positions, orientations = robot.forward(np.zeros(num_links))
    for idx, (pos, ori) in enumerate(zip(positions, orientations)):
        print("position: ", pos)
        print("orientation: ", ori)
        np.testing.assert_allclose(pos, np.array([idx, 0]), atol=1e-10)
        np.testing.assert_allclose(ori, np.eye(2), atol=1e-10)
    
    angles = np.zeros(num_links)
    angles[0] = np.pi / 2
    positions, orientations = robot.forward(angles)
    expected_orientation = np.array([[0, -1], [1, 0]])
    for idx, (pos, ori) in enumerate(zip(positions, orientations)):
        print("position: ", pos)
        print("orientation: ", ori)
        np.testing.assert_allclose(pos, np.array([0, idx]), atol=1e-10)
        if idx == 0:
            np.testing.assert_allclose(ori, np.eye(2), atol=1e-10)
        else:
            np.testing.assert_allclose(ori, expected_orientation, atol=1e-10)
            

def test_inverse_kinematics():
    num_links = 4
    links = np.ones(num_links)
    robot = RobotArm2D(links=links, solver_cls=CCD)
