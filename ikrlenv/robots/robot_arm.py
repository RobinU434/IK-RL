import numpy as np

from envs.robots.ccd import IK


class RobotArm:
    def __init__(self, n_joints: int = 1, segment_lenght: float = 1) -> None:
        self._n_joints = n_joints
        self._segment_length = segment_lenght

        self._arm_length = self._n_joints * segment_lenght
        # relative angles
        self._angles = np.zeros((self._n_joints))

        self._positions = np.zeros((self._n_joints + 1, 2))  # 2D, the plus one dim is the origin
        # init _positions
        self.set(self._angles)


    def reset(self):
        self._angles = np.zeros((self._n_joints))
        self.set(self._angles)

    def set(self, angles):
        """applies given action to the arm 

        Args:
            angles (np.array): array with same length as number of joints
        """
        self._angles = angles % (2 * np.pi)

        for idx in range(self._n_joints):
            origin = self._positions[idx]
            
            # new position
            new_pos = np.array([np.cos(self._angles[idx]), np.sin(self._angles[idx])])
            # new_pos = np.array([np.cos(cumsum[idx]), np.sin(cumsum[idx])])
            new_pos *= self._segment_length

            # translate position
            new_pos += origin

            self._positions[idx + 1] = new_pos

    def IK(self, target, max_iter: int = 1000, error_min: float = 0.1):
        # for the angles you have to pass in the relative angles between the joints.
        # To get the absolute angels you can use the cum sum function 
        # To invert this you can use:
        # >>> z[1:] -= z[:-1].copy()
        rel_angles = self.angles.copy()
        # transform into degrees
        rel_angles = rel_angles / np.pi * 180
        rel_angles[1:] -= rel_angles[:-1].copy()

        # link is a sequence of individual segment lengths
        # in our implementation the segment length is constant over all segments
        link = np.ones(self.n_joints) * self._segment_length

        res_angles, error, _, _ = IK(target, rel_angles, link, max_iter=max_iter, err_min=error_min)

        # convert relative resulting angles (re_angles) into absolute angles
        abs_angles = np.array(res_angles).cumsum() / 180 * np.pi
        self.set(abs_angles)

    @property
    def angles(self):
        return self._angles

    @property
    def n_joints(self):
        return self._n_joints
    
    @property
    def arm_length(self):
        return self._arm_length

    @property
    def positions(self):
        return self._positions
    
    @property
    def end_position(self):
        return self._positions[-1]


if __name__ == "__main__":
    arm = RobotArm(20, 1)

    print(arm._positions)
    angles = np.zeros
    arm.set([0, np.pi / 2])
    print(arm._positions)
