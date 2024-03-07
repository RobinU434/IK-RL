# Contributing to the Inverse Kinematics Reinforcement Learning Environment

We welcome contributions to this project! This document outlines the guidelines and expectations for contributing to the codebase.

## How to Contribute

There are several ways you can contribute to this project:

- Reporting Bugs: If you encounter any bugs or unexpected behavior, please open an issue on the project's GitHub repository. Please provide a clear and concise description of the issue, including steps to reproduce it if possible.
- Suggesting Enhancements: If you have ideas for improving the environment or its documentation, feel free to open an issue or create a pull request with your proposed change.
- Submitting Pull Requests: If you've made changes to the codebase that you'd like to share, please submit a pull request. Before submitting your pull request, please ensure:
    - You have forked the repository and cloned it to your local machine.
    - You have created a new branch for your changes.
    - You have made your changes and committed them to your local branch.
    - You have pushed your changes to your remote branch on GitHub.
    - You have created a pull request from your remote branch to the main branch of the upstream repository.

- Code Style: We follow the PEP 8 style guide for Python code. Please ensure your code adheres to these guidelines before submitting a pull request. You can use tools like autopep8 to automatically format your code.

## Code Structure
The codebase is organized as follows:

- `ik_rl/`: This directory contains the main source code for the environment.
    - `envs/`: This directory contains the environment classes and related logic.
    - `tasks/`: This directory contains the task classes that define the specific objectives for the robot arm.
    - `utils/`: This directory contains utility functions used throughout the codebase.
- `tests/`: This directory contains unit and integration tests for the environment.
- `studiues/`: contains notebooks with a couple of experiments. Those notebooks are not important for the core mechanics of the environment.
- `README.md`: This file provides an overview of the project.
- `CONTRIBUTING.md`: This file (you are reading it now!) outlines the contribution guidelines.

## Licensing
This project is licensed under the Apache License 2.0. You can find the full license text in the LICENSE file.

## Community
We encourage open and respectful communication within the community. Please be considerate of others and avoid any discriminatory or offensive language.

We appreciate your contributions and hope you enjoy working on this project!