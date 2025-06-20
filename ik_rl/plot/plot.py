from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy import ndarray

from ik_rl.robots.robot_arm import _RobotArm


def plot_base(ax: Axes, arm_reach: float) -> Axes:
    circle = Circle((0, 0), radius=arm_reach, color="grey", alpha=0.2)
    ax.add_patch(p=circle)
    ax.set_xlim(-arm_reach * 1.05, arm_reach * 1.05)
    ax.set_ylim(-arm_reach * 1.05, arm_reach * 1.05)

    return ax


def plot_target(ax: Axes, target_pos: ndarray, epsilon: float) -> Axes:
    circle = Circle(tuple(target_pos.tolist()), radius=epsilon, color="red", alpha=0.2)
    ax.add_patch(p=circle)
    ax.scatter(*target_pos, c="r", s=15)

    return ax


def plot_arm(ax: Axes, robot: _RobotArm) -> Axes:
    ax.plot(*robot.positions.T, ".-", color="orange")
    return ax


def plot_end_effector(ax, position: ndarray) -> Axes:
    ax.scatter(*position, c="green", s=10, marker="x", zorder=2)
    return ax
