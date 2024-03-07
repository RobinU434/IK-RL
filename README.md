# Inverse Kinematic Reinforcement Learning Environment with Scalable Action Space

This repository implements a 2D or 3D robotic arm environment for reinforcement learning, featuring a highly scalable action space. The goal is to control the arm to reach a target position in the environment.

The agent receives an observation of the environment, which includes the target position, the current end effector position, and the current joint angles of the robot arm. The agent then takes an action to change the joint angles, and receives a reward based on the distance between the end effector and the target position.

This environment is designed to be adaptable to various scenarios by offering a scalable action space. Whether you're dealing with a simple 2-joint arm or a complex multi-jointed manipulator, this environment can accommodate your needs.

## Installation

The required libraries can be seen [here](./pyproject.toml). You can always install the manually but we recommend to install them with poetry.
```bash
poetry install
```

## Usage
Import either the discrete or continuous version and a task you want to complete.

```python
from ik_rl.environment import InvKinDiscrete, InvKinEnvContinous
from ik_rl.task import ReachGoalTask
```

Create an environment and task instance:

```python
task = ReachGoalTask(arm_reach=n_joints * segment_length)
env = InvKinEnvContinous(task=task, robot_config={"n_joints": n_joints})
```

You are always free to replace `ReachGoalTask` with the specific task you want the robot arm to complete. This class should inherit from `ik_rl.task.base_task.BaseTask`.

Reset the environment:
```python
observation = env.reset()
```

Take an action and get the next observation, reward, and done flag:
Python
```python
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
```

Render the environment (optional):
```python
env.render()
```

Close the environment:
```python
env.close()
```

For an working example please refer to the [example notebook](./example.ipynb). For the example you are required to install [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

## Available environments:

- `InvKinEnv`: This is the base class for all inverse kinematic environments. It supports both 2D and 3D robot arms, and allows for choosing between discrete and continuous action spaces.  
- `InvKinDiscrete`: This class provides a discrete action space, where the agent can choose from a set of predefined joint angle changes.  
- `InvKinEnvContinous`: This class provides a continuous action space, where the agent can specify the desired joint angle changes directly.  


## Contributing

We welcome contributions to this project! Please see the CONTRIBUTING.md file for more information. Planned features are:

- expand to 3-D (Optional n-D)
- add obstacles to env
  - polygons, circles, ...
  - in 3D add table top (z has to be >= 0) constraint 
  - method to compute if is possible to solve for a given target position given the env with obstacles
- add parallel computing for as many arms as I want

## Acknowledgments

This project has benefited from the contributions of several individuals and resources. We would like to express our sincere gratitude to:

The developers of the following libraries:


- [OpenAI Gym](https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2)
- [Farama Gymnasium](https://gymnasium.farama.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch (optional)](https://pytorch.org/)

The contributors to various online communities and forums who shared their knowledge and insights on reinforcement learning and inverse kinematics.  
Jasper Hoffmann for their valuable feedback, guidance, and support during the development process.  
If you have contributed to this project and are not mentioned here, please feel free to reach out and we will be happy to include your acknowledgment.

We are grateful for the support of the community and look forward to further contributions to this project.



