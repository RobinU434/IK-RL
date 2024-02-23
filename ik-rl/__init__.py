import sys
import os

sys.path.append(os.getcwd() + "/envs/plane_robot_env")


from ikrlenv.task.imitation_task import ImitationTask
from ikrlenv.task.reach_goal_task import ReachGoalTask
