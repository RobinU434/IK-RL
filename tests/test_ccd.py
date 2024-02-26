
import numpy as np
from ik_rl.robots.robot_arm import RobotArm2D
from ik_rl.solver.ccd import CCD
from itertools import product

def test_ccd():
    it = product([1e-5, 1e-10, 1e-15], [1, 5, 10])
    for tol, n_links in it:
        print(tol, n_links)
        robot = RobotArm2D(links=np.ones(n_links), solver_cls=CCD, solver_args={"err_min": tol})

        target = np.array([0, 1])
        robot.backward(target=target)
        np.testing.assert_allclose(robot.end_position, target, atol=tol)