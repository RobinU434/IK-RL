# IK-RL
RL environment for solving inverse kinematics for a robot arm with n many joints in 2D


# Features

- new foreward pass based on matrix multiplication
- encase CCD in a general IK-Solver framework
- expand to 3-D (Optional n-D)
- add obstacles to env
  - polygons, circles, ...
  - in 3D add table top (z has to be >= 0) constraint 
  - method to compute if is possible to solve for a given target position given the env with obstacles
- add paralell computing for as many arms as I want 
