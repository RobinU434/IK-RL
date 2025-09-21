import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, Tuple, Type

import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.spaces import Box
from ik_rl.plot.plot import plot_arm, plot_base, plot_end_effector, plot_target
from ik_rl.robots.robot_arm import RobotArm2D, RobotArm3D, _RobotArm
from ik_rl.solver.ccd import CCD
from ik_rl.task.base_task import _BaseTask
from ik_rl.task.config import NUM_TIME_STEPS
from ik_rl.task.imitation_task import ImitationTask
from ik_rl.task.reach_goal_task import ReachGoalTask
from ik_rl.utils.sample_target import sample_target
from matplotlib import pyplot as plt
from numpy import ndarray
import pygame
from pygame import gfxdraw  # noqa: F401


class Mode(Enum):
    STRATEGIC = 0
    ONE_SHOT = 1


class _InvKinEnv(Env):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array", "human"], "render_fps": 2}

    def __init__(
        self,
        n_steps: int = NUM_TIME_STEPS,
        n_dims: Literal[2, 3] = 2,
        n_joints: int = 1,
        segment_length: int = None,
        robot_kwargs: Dict[str, Any] = None,
        task: Type[_BaseTask] = ReachGoalTask,
        task_kwargs: Dict[str, Any] = None,
        epsilon: float = 1e-2,
        relative_angles: bool = True,
        one_shot: bool = False,
        render_mode: str = None,
        render_size: int = 400,
        seed: int = None,
    ) -> None:
        super().__init__()
        self._n_steps = n_steps
        self._epsilon = epsilon
        self._n_dims = n_dims
        self._n_joints = n_joints
        self._segment_length = segment_length
        self._relative_angles = relative_angles
        self._one_shot = one_shot
        self.render_mode = render_mode
        self._render_size = render_size
        self.seed = seed

        self.robot_arm = self._build_robot(
            n_dims=self._n_dims,
            n_joints=self._n_joints,
            segment_length=self._segment_length,
            robot_kwargs=robot_kwargs,
        )
        self._task = self._build_task(task, task_kwargs)
        self._target_position = sample_target(self.robot_arm.arm_length)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        self._step_counter = 0

        # render with pygame
        self.screen = None
        self.clock = None

        super().reset(seed=self.seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
        target_position: ndarray | None = None,
        rand_arm_angles: bool = False,
    ) -> Tuple[ndarray, dict[str, Any]]:
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
        if seed is not None:
            self.seed = seed
        super().reset(seed=seed, options={})
        rel_angles = None
        if rand_arm_angles:
            rel_angles = np.random.rand(self.robot_arm.n_joints) * 2 * np.pi
        self.robot_arm.reset(rel_angles)
        self._task.reset()

        self._step_counter = 0

        if target_position is None or len(target_position) != 2:
            # msg = f"Sample target in env.reset(). Given target position is not sufficiant: {target_position=}"
            # logging.info(msg)
            self._target_position = sample_target(self.robot_arm.arm_length)
        else:
            self._target_position = target_position

        return self._observe(), {}

    def step(
        self, action: ndarray
    ) -> Tuple[ndarray, float, bool, bool, dict[str, Any]]:
        self._step_counter += 1
        assert self.action_space.contains(action)
        self._apply_action(action)

        # calculate reward
        kwargs = {
            "arm_position": self.robot_arm.end_position,
            "target_position": self._target_position,
            "robot_arm_angles": self.robot_arm.abs_angles,
        }
        reward = self._task.reward(**kwargs)

        # get observation
        obs = self._observe()

        # determine if the env is done
        truncated, done = self._task.is_done(
            arm_position=self.robot_arm.end_position,
            target_position=self._target_position,
        )

        return obs, reward, done, truncated, {}

    def _get_rgb_array(self) -> ndarray:
        """return rendered image as np.ndarray

        Returns:
            ndarray: (height, width, 3)
        """
        if self._n_dims == 2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax = plot_base(ax, arm_reach=self.robot_arm.arm_length)
            ax = plot_arm(ax, self.robot_arm)
            ax = plot_target(
                ax, target_pos=self._target_position, epsilon=self._epsilon
            )
            ax = plot_end_effector(ax, position=self.robot_arm.end_position)
            dist = np.linalg.norm(self._target_position - self.robot_arm.end_position)
            ax.set_title(f"{dist:.4f}, {self._step_counter}")
            fig.canvas.draw()
            plt.close(fig)
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            return data
        elif self._n_dims == 3:
            raise NotImplementedError
        else:
            raise ValueError(
                "plotting methods for arms in higher dimensional space than 3 are not possible"
            )

    def _human_render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self._render_size, self._render_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        rgb_array = self._get_rgb_array()
        rgb_array = np.swapaxes(rgb_array, axis1=0, axis2=1)
        surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(surface, (0, 0))
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def render(self) -> ndarray | None:
        if self.render_mode is None:
            return
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()
        elif self.render_mode == "human":
            self._human_render()
        else:
            raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

        return super().close()
        logging.info("Close environment.")

    @staticmethod
    def _build_robot(
        n_dims: int,
        n_joints: int = 1,
        segment_length: float = None,
        robot_kwargs: Dict[str, Any] = None,
    ) -> _RobotArm:
        """build a robot arm based on the given arguments

        Args:
            n_dims (int): in which space the robot arm should operate
            robot_config (Dict): arguments for the robot arm

        Raises:
            NotImplementedError: if n_dims > 3 or < 2

        Returns:
            RobotArm: build robot arm
        """
        cls: _RobotArm
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
        # if you want to have other segment length please replace the 1 with any other value or a RNG

        links = np.ones(n_joints)
        if segment_length is None:
            links = links / n_joints
        else:
            links = links * segment_length
        if robot_kwargs is None:
            robot_kwargs = {}
        return cls(links=links, solver_cls=CCD, **robot_kwargs)

    def _build_task(
        self, task_type: Type[_BaseTask], task_kwargs: Dict[str, Any] = None
    ) -> _BaseTask:
        args = {"epsilon": self._epsilon, "n_time_steps": self._n_steps}
        if task_kwargs is not None:
            args = {**args, **task_kwargs}

        if task_type == ImitationTask:
            args = {**args, "robot_arm": self.robot_arm}
        return task_type(**args)

    @abstractmethod
    def _build_action_space(self):
        raise NotImplementedError

    def _build_observation_space(self) -> Box:
        """
        observation space is a 4 dimensional tensor.
            - first two dimensions: the 2D position of the goal position
            - second two dimensions: the 2D position of the end effector position
            - rest relative angels of joints
        """
        return Box(
            -self.robot_arm.arm_length,
            self.robot_arm.arm_length,
            (2 + 2 + self.robot_arm.n_joints,),
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
        action = self.robot_arm.rel_angles * (1 - self._one_shot) + action  # type: ignore

        if self._relative_angles:
            self.robot_arm.set_rel_angles(action)
        else:  # alternative are absolute angles
            self.robot_arm.set_abs_angles(action)

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
            target_position = self._target_position / self.robot_arm.arm_length
            arm_end_position = self.robot_arm.end_position / self.robot_arm.arm_length
        else:
            target_position = self._target_position
            arm_end_position = self.robot_arm.end_position

        obs = np.concatenate(
            (target_position, arm_end_position, self.robot_arm.abs_angles)
        )
        obs = obs.astype(np.float32)

        return obs


class InvKinDiscrete(_InvKinEnv):
    """Send discrete actions to the environment. You set the set of available actions while constructing the environment"""

    def __init__(
        self,
        n_steps=NUM_TIME_STEPS,
        n_dims=2,
        n_joints=1,
        segment_length=None,
        robot_kwargs=None,
        task=ReachGoalTask,
        task_kwargs=None,
        epsilon=0.01,
        relative_angles=True,
        one_shot=False,
        render_mode=None,
        render_size=400,
        available_actions: np.ndarray = np.array([-1, 0, 1]),
        seed=None,
    ):
        super().__init__(
            n_steps,
            n_dims,
            n_joints,
            segment_length,
            robot_kwargs,
            task,
            task_kwargs,
            epsilon,
            relative_angles,
            one_shot,
            render_mode,
            render_size,
            seed,
        )
        """init class

        Args:
            task (BaseTask): task to complete the in environment
            n_dims (Literal[2, 3], optional): robot arm in 2D or 3D space. Defaults to 2.
            robot_config (dict, optional): arguments for the robot. To look up which arguments are supported please refer to the robot class. Defaults to {"n_joints": 1}.
            available_actions (np.ndarray, optional): set of available actions. Defaults to np.array([-1, 0, 1]).
            relative_angles (bool, optional): Flag to determine if the actions will be seen as absolute actions against angle 0 or relative to the previous joint. Defaults to False.
        """
        assert len(available_actions.shape) == 1  # expect only one dimensional array
        self._available_actions = available_actions[None].repeat(n_joints, axis=0)

        super().__init__(
            n_steps,
            n_dims,
            n_joints,
            segment_length,
            robot_kwargs,
            task,
            task_kwargs,
            epsilon,
            relative_angles,
            one_shot,
            render_mode,
            seed,
        )

    def _build_action_space(self):
        n_vec = np.ones(self.robot_arm.n_joints) * self._available_actions.shape[1]
        return spaces.MultiDiscrete(nvec=n_vec)

    def _transform_action(self, action: ndarray) -> ndarray:
        """expect the action as a 2D array. first dimension is the robot arm length and second dimension is the distribution over actions
        Function takes the argmax over the second dimension and maps those indices to the available actions

        Args:
            action (ndarray): action array from neural network. Expected shape: num_joints. Each element identifies an action from the available actions.

        Returns:
            ndarray: one dimensional array with length == num_joints
        """
        action = self._available_actions[np.arange(self._n_joints), action]
        # with continuous actions the action itself is the delta angle which will be also added on top of the current angle
        action = np.squeeze(action)
        return action


class   InvKinEnvContinuous(_InvKinEnv):
    """send continuous actions to the robot arm."""

    def __init__(
        self,
        n_steps=NUM_TIME_STEPS,
        n_dims=2,
        n_joints=1,
        segment_length=None,
        robot_kwargs=None,
        task=ReachGoalTask,
        task_kwargs=None,
        epsilon=0.01,
        relative_actions=False,
        one_shot=False,
        render_mode=None,
        render_size=400,
        seed=None,
    ):
        """init class

        Args:
            task (BaseTask): task to complete the in environment
            n_dims (Literal[2, 3], optional): robot arm in 2D or 3D space. Defaults to 2.
            robot_config (dict, optional): arguments for the robot. To look up which arguments are supported please refer to the robot class. Defaults to {"n_joints": 1}.
            one_shot (bool, optional): would you like to have a sequential decision making process or predict the solution in one go. Defaults to False.
            relative_angles (bool, optional): Flag to determine if the actions will be seen as absolute actions against angle 0 or relative to the previous joint. Defaults to False.
        """
        super().__init__(
            n_steps,
            n_dims,
            n_joints,
            segment_length,
            robot_kwargs,
            task,
            task_kwargs,
            epsilon,
            relative_actions,
            one_shot,
            render_mode,
            render_size,
            seed,
        )
        self._target_position = sample_target(self.robot_arm.arm_length)

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_action_space(self) -> Box:
        """an action is either +1 degree, -1 degree or 0 degrees of rotation per joint
        Therefor is one action a tensor with the length equal to the number of joints.


        Returns:
            Box: build action space on either a discrete action or continuous action space
        """
        return spaces.Box(-np.pi, np.pi, shape=(self.robot_arm.n_joints,))

    def _transform_action(self, action: ndarray) -> ndarray:
        return action
