SEGMENT_LENGTH = 1

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, SupportsFloat, Tuple

import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy import ndarray

from ik_rl.plot.plot import plot_arm, plot_base, plot_end_effector, plot_target
from ik_rl.robots.robot_arm import RobotArm, RobotArm2D, RobotArm3D
from ik_rl.solver.ccd import CCD
from ik_rl.task.base_task import BaseTask
from ik_rl.utils.cast import tensor_to_ndarray
from ik_rl.utils.sample_target import sample_target


class Mode(Enum):
    STRATEGIC = 0
    ONE_SHOT = 1


class InvKinEnv(Env, ABC):
    def __init__(
        self,
        task: BaseTask,
        n_dims: Literal[2, 3] = 2,
        robot_config: Dict[str, Any] = {"n_joints": 1},
        relative_angles: bool = False,
        one_shot: bool = False,
    ) -> None:
        super().__init__()
        self._task = task
        self._n_dims = n_dims
        self._one_shot = one_shot
        self._robot_arm = self._build_robot(n_dims=n_dims, robot_config=robot_config)
        self._step_counter = 0
        self._relative_angles = relative_angles
        self._target_position = sample_target(self._robot_arm.arm_length)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def reset(
        self, *, seed: int | None = None, target_position: ndarray | None = None
    ) -> Tuple[Any | dict[str, Any]]:
        """reset the environment.
        Set the step counter to 0
        Set the arm in the initial position
        If there is a determined target_position. Set  the argument as the given target environment.

        Args:
            seed (int | None, optional): seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action. Defaults to None.
            target_position (ndarray | None, optional): Special target to set. Defaults to None.

        Returns:
            Tuple[ndarray | dict[str, Any]]: observation containing target position, end effector position, and angles of the arm. Second return type is additional information as a dictionary.
        """
        super().reset(seed=seed, options={})
        self._robot_arm.reset()
        self._task.reset()

        self._step_counter = 0

        if target_position is None or len(target_position) != 2:
            msg = f"Sample target in env.reset(). Given target position is not sufficiant: {target_position=}"
            logging.info(msg)
            self._target_position = sample_target(self._robot_arm.arm_length)
        else:
            self._target_position = target_position

        return self._observe(), {}

    def step(
        self, action: ndarray
    ) -> Tuple[ndarray | SupportsFloat | bool | bool | dict[str, Any]]:
        self._step_counter += 1
        assert self.action_space.contains(action)
        self._apply_action(action)

        # calculate reward
        kwargs = {
            "arm_position": self._robot_arm.end_position,
            "target_position": self._target_position,
            "robot_arm_angles": self._robot_arm.abs_angles,
        }
        reward = self._task.reward(**kwargs)

        # get observation
        obs = self._observe()

        # determine if the env is done
        truncated, done = self._task.done(
            arm_position=self._robot_arm.end_position,
            target_position=self._target_position,
        )

        return obs, reward, done, truncated, {}

    def render(self) -> ndarray:
        if self._n_dims == 2:
            fig, ax = plt.subplots()
            ax = plot_base(ax, arm_reach=self._robot_arm.arm_length)
            ax = plot_arm(ax, self._robot_arm)
            ax = plot_target(ax, target_pos=self._target_position)
            ax = plot_end_effector(ax, position=self._robot_arm.end_position)
        elif self._n_dims == 3:
            raise NotImplementedError
        else:
            raise ValueError(
                "plotting methods for arms in higher dimensional space than 3 are not possible"
            )
        fig.show()

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        logging.info("Close environment.")

    @staticmethod
    def _build_robot(n_dims: int, robot_config: Dict[str, Any]) -> RobotArm:
        """build a robot arm based on the given arguments

        Args:
            n_dims (int): in which space the robot arm should operate
            robot_config (Dict): arguments for the robot arm

        Raises:
            NotImplementedError: if n_dims > 3 or < 2

        Returns:
            RobotArm: build robot arm
        """
        cls: RobotArm
        match n_dims:
            case 2:
                cls = RobotArm2D
            case 3:
                raise NotImplementedError("3D robot arm is still in development")
                cls = RobotArm3D
            case _:
                raise NotImplementedError(
                    f"A robot arm for the requested dimension {n_dims} is not implemnted"
                )
        links = SEGMENT_LENGTH * np.ones(robot_config["n_joints"])
        return cls(links=links, solver_cls=CCD)

    @abstractmethod
    def _build_action_space(self):
        raise NotImplementedError

    def _build_observation_space(self) -> Box:
        """
        observation space is a 4 dimensional tensor.
            - first two dimensions: the 2D position of the goal position
            - second two dimensions: the 2D position of the robot arm tip
        """
        return Box(
            -self._robot_arm.arm_length,
            self._robot_arm.arm_length,
            (2 + 2 + self._robot_arm.n_joints, 1),
        )

    @abstractmethod
    def _transform_action(self, action: ndarray) -> ndarray:
        raise NotImplementedError

    def _apply_action(self, action: ndarray | torch.Tensor):
        """adds action to the robot arm angles

        Args:
            action (ndarray): continuous action shape: ()
        """
        action = self._transform_action(action)
        action = self._robot_arm.rel_angles * (1 - self._one_shot) + action  # type: ignore

        if self._relative_angles:
            self._robot_arm.set_rel_angles(action)
        else:  # alternative are absolute angles
            self._robot_arm.set_abs_angles(action)

    def _observe(self, normalize: bool = False) -> ndarray:
        """build an observation of the environment to the current time step.
        This observation contains [target_position, end_effector_position, robot_arm_angles]
        TODO: relative or absolute angles

        Args:
            normalize (bool, optional): Would you like to normalize the positions. So set the maximal radius to 1. Defaults to False.

        Returns:
            ndarray: array with num_joints + 2 * space_dimension. space_dimension = 2 or 3 dimensional
        """
        if normalize:
            # normalize observations
            target_position = self._target_position / self._robot_arm.arm_length
            arm_end_position = self._robot_arm.end_position / self._robot_arm.arm_length
        else:
            target_position = self._target_position
            arm_end_position = self._robot_arm.end_position

        obs = np.concatenate(
            (target_position, arm_end_position, self._robot_arm.abs_angles)
        )[:, None]

        return obs


class InvKinDiscrete(InvKinEnv):
    """Send discrete actions to the environment. You set the set of available actions while constructing the environment
    """
    def __init__(
        self,
        task: BaseTask,
        n_dims: Literal[2, 3] = 2,
        robot_config: Dict[str, Any] = {"n_joints": 1},
        available_actions: np.ndarray = np.array([-1, 0, 1]),
        relative_angles: bool = False,
    ) -> None:
        """init class

        Args:
            task (BaseTask): task to complete the in environment
            n_dims (Literal[2, 3], optional): robot arm in 2D or 3D space. Defaults to 2.
            robot_config (dict, optional): arguments for the robot. To look up which arguments are supported please refer to the robot class. Defaults to {"n_joints": 1}.
            available_actions (np.ndarray, optional): set of available actions. Defaults to np.array([-1, 0, 1]).
            relative_angles (bool, optional): Flag to determine if the actions will be seen as absolute actions against angle 0 or relative to the previous joint. Defaults to False.
        """
        self._available_actions = (
            available_actions
            if isinstance(available_actions, np.ndarray)
            else np.array(available_actions)
        )
        assert (
            len(self._available_actions.shape) == 1
        )  # expect only one dimensional array
        super().__init__(task, n_dims, robot_config, relative_angles, False)

    def _build_action_space(self):
        nvec = np.ones(self._robot_arm.n_joints) * len(self._available_actions)
        return spaces.MultiDiscrete(nvec=nvec)

    def _transform_action(self, action: ndarray) -> ndarray:
        """expect the action as a 2D array. first dimension is the robot arm length and second dimension is the distribution over actions
        Function takes the argmax over the second dimension and maps those indices to the available actions

        Args:
            action (ndarray): action array from neural network. Expected shape: num_joints. Each element identifies an action from the available actions.

        Returns:
            ndarray: one dimensional array with length == num_joints
        """
        action = tensor_to_ndarray(action)
        action = self._available_actions[action]

        # with continuous actions the action itself is the delta angle which will be also added on top of the current angle
        action = np.squeeze(action)
        return action


class InvKinEnvContinous(InvKinEnv):
    """send continuous actions to the robot arm. 
    """
    def __init__(
        self,
        task: BaseTask,
        n_dims: Literal[2, 3] = 2,
        robot_config: Dict[str, Any] = {"n_joints": 1},
        one_shot: bool = False,
        relative_angles: bool = False,
    ) -> None:
        """init class

        Args:
            task (BaseTask): task to complete the in environment
            n_dims (Literal[2, 3], optional): robot arm in 2D or 3D space. Defaults to 2.
            robot_config (dict, optional): arguments for the robot. To look up which arguments are supported please refer to the robot class. Defaults to {"n_joints": 1}.
            one_shot (bool, optional): would you like to have a sequential decision making process or predict the solution in one go. Defaults to False.
            relative_angles (bool, optional): Flag to determine if the actions will be seen as absolute actions against angle 0 or relative to the previous joint. Defaults to False.
        """
        super().__init__(task, n_dims, robot_config, relative_angles, one_shot)
        self._target_position = sample_target(self._robot_arm.arm_length)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_action_space(self) -> Box:
        """an action is either +1 degree, -1 degree or 0 degrees of rotation per joint
        Therefor is one action a tensor with the length equal to the number of joints.


        Returns:
            Box: build action space on either a discrete action or continuous action space
        """
        return spaces.Box(0, 2 * np.pi, (1, self._robot_arm.n_joints))

    def _transform_action(self, action: ndarray) -> ndarray:
        action = tensor_to_ndarray(action)
        action = np.squeeze(action)
        return action
