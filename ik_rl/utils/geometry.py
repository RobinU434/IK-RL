import numpy as np
from numpy import ndarray


def unit_vector(vector: ndarray) -> ndarray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: ndarray, v2: ndarray) -> ndarray:
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    sign = np.sign(0.1 + np.cross(v1_u, v2_u)[-1])
    return sign * angle