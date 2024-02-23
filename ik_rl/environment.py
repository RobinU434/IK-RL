import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from gymnasium import spaces
from ik_rl.robots.robot_arm import RobotArm
from ik_rl.task.base_task import BaseTask
from ik_rl.utils.sample_target import sample_target
from numpy import ndarray
from PIL import Image, ImageDraw


class PlaneRobotEnv(gym.Env):
    def __init__(
        self,
        task: BaseTask,
        n_joints: int = 1,
        segment_length: float = 1,
        discrete_mode: bool = False,
        action_mode: str = "strategic",
        **kwargs,
    ) -> None:
        self._task = task
        self._robot_arm: RobotArm = RobotArm(n_joints, segment_length)
        # init angles and other
        self.reset()

        # discrete mode = True is for discrete actions (+-1 / +-0 degrees)
        # discrete mode = False is for continuous actions
        self._discrete_mode = discrete_mode

        if action_mode == "strategic":
            self._strategic_mode = True
        elif action_mode == "one_shot":
            self._strategic_mode = False
        else:
            logging.error("action mode has to be either 'strategic' or 'one_shot'")

        self._target_position = self._get_target_position(self._robot_arm.arm_length)

        self._set_action_space()
        self._set_observation_space()

        self._step_counter = 0

    def _set_action_space(self) -> None:
        """
        an action is either +1 degree, -1 degree or 0 degrees of rotation per joint
        Therefor is one action a tensor with the length equal to the number of joints.
        """
        if self._discrete_mode:
            self.action_space = spaces.Box(-1, 1, (1, self._robot_arm.n_joints))
        else:
            self.action_space = spaces.Box(0, 2 * np.pi, (1, self._robot_arm.n_joints))

    def _set_observation_space(self) -> None:
        """
        observation space is a 4 dimensional tensor.
            - first two dimensions: the 2D position of the goal position
            - second two dimensions: the 2D position of the robot arm tip
        """
        self.observation_space = spaces.Box(
            -self._robot_arm.arm_length,
            self._robot_arm.arm_length,
            (2 + 2 + self._robot_arm.n_joints, 1),
        )

    @staticmethod
    def _get_target_position(radius: float):
        return sample_target(radius)

    def _apply_action(self, action: ndarray):
        """adds action to the robot arm angles

        Args:
            action (ndarray): continuous action shape: ()
        """
        if isinstance(action, torch.Tensor):
            # detach if the action tensor requires grad = True
            action = action.detach().numpy()

        # with discrete actions the action is -1 +1 or 0 which will be added on top of the current angle
        # with continuous actions the action itself is the delta angle which will be also added on top of the current angle
        action = np.squeeze(action)
        action = self._robot_arm.angles * self._strategic_mode + action  # type: ignore

        self._robot_arm.set(action)

    def _observe(self, normalize: bool = False) -> ndarray:
        if normalize:
            # normalize observations
            target_position = self._target_position / self._robot_arm.arm_length
            arm_end_position = self._robot_arm.end_position / self._robot_arm.arm_length
        else:
            target_position = self._target_position
            arm_end_position = self._robot_arm.end_position

        obs = np.concatenate(
            (target_position, arm_end_position, self._robot_arm.angles)
        )

        return obs

    def reset(self, target_position: ndarray = None) -> ndarray:  # type: ignore
        """resets the environment

        Args:
            target_position (ndarray): optional arguemt

        Returns:
            ndarray: _description_
        """
        self._robot_arm.reset()
        self._task.reset()

        self._step_counter = 0

        if target_position is None or len(target_position) != 2:
            msg = f"Sample target in env.reset(). Given target position is not sufficiant: {target_position=}"
            logging.info(msg)
            self._target_position = self._get_target_position(
                self._robot_arm.arm_length
            )
        else:
            self._target_position = target_position

        return self._observe()

    def step(self, action: ndarray) -> Tuple[ndarray, float, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (np.array):

        Returns:
            Tuple[np.array, float, bool, Dict[str, Any]]: _description_
        """
        self._step_counter += 1
        self._apply_action(action)

        # calculate reward
        kwargs = {
            "arm_position": self._robot_arm.end_position,
            "target_position": self._target_position,
            "robot_arm_angles": self._robot_arm.angles,
        }
        reward = self._task.reward(**kwargs)

        # get observation
        obs = self._observe()

        # determine if the env is done
        done = self._task.done(
            arm_position=self._robot_arm.end_position,
            target_position=self._target_position,
        )

        return obs, reward, done, {}

    def render(self, path: str = "test.png"):
        render_size = (
            int(self._robot_arm.arm_length * 1.1),
            int(self._robot_arm.arm_length * 1.1),
        )
        # origin is in the upper left corner
        img = Image.new("RGB", render_size, (256, 256, 256))
        draw = ImageDraw.Draw(img)

        # set origin to the center
        origin = (render_size[0] / 2, render_size[1] / 2)
        scale_factor = 20

        self._draw_goal(draw, origin + self._target_position * scale_factor)
        self._draw_segments(draw, origin, scale_factor)
        self._draw_joints(draw, origin, scale_factor)

        img.save(path)

    def _draw_goal(self, draw, origin: ndarray = np.zeros((2)), radius=4):
        x, y = origin
        # distances to origin
        # (left, upper, right, lower)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )

    def _draw_joints(self, draw, origin, scale_factor: float = 20):
        for position in self._robot_arm._positions[:-1].copy():
            # scale
            position *= scale_factor

            # move the segment
            position = position + origin

            self._draw_joint(draw, position)

    def _draw_segments(self, draw, origin, scale_factor: float = 20):
        for idx in range(self._robot_arm.n_joints):
            start = self._robot_arm._positions[idx].copy()
            end = self._robot_arm._positions[idx + 1].copy()

            # scale
            start *= scale_factor
            end *= scale_factor

            # move the segment
            start += origin
            end += origin

            self.draw_segment(draw, start, end)

    @staticmethod
    def _draw_joint(draw, origin: Tuple[float, float] = (0, 0), radius=4):
        """draws joint from robot arm as a grey circle

        Args:
            draw (_type_): _description_
            origin (tuple, optional): origin of joint. (x, y) Defaults to (0, 0).
            radius (int, optional): radius of joint. Defaults to 5.
        """
        x, y = origin
        # distances to origin
        # (left, upper, right, lower)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(100, 100, 100),
            outline=(0, 0, 0),
        )

    @staticmethod
    def draw_segment(
        draw,
        start: Tuple[float, float] = (0, 0),
        end: Tuple[float, float] = (1, 1),
        width: float = 3,
    ):
        """draw segment as yellow line

        Args:
            draw (_type_): draw object to draw in
            start (Tuple[float, float], optional): start position from line: (x, y). Defaults to (0, 0).
            end (Tuple[float, float], optional): end position from line: (x, y). Defaults to (1, 1).
            width (float, optional): width from line. Defaults to 2.
        """
        coord = [start[0], start[1], end[0], end[1]]
        draw.line(coord, fill=(255, 255, 0), width=width)

    @property
    def num_steps(self):
        return self._step_counter

    @property
    def target_position(self):
        return self._target_position

    @property
    def n_joints(self):
        return self._robot_arm.n_joints
