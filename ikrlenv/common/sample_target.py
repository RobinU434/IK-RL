import numpy as np


def sample_target(radius: float, n_points: int = 1) -> np.ndarray:
    """
    sample goal position in a circular shape around the origin
    radius and angle is sampled uniformly

    Args:
        radius (float): maximum radius to sample fromm

    Returns:
        np.array: shape (n_points, 2) if  n_point == 1 -> shape: (2)
    """
    # angle to sample from
    theta = np.random.uniform(0, 2 * np.pi, size=n_points)
    radius_array = np.random.uniform(0, radius, size=n_points)
    vector = radius_array * np.array([np.cos(theta), np.sin(theta)])
    return vector.squeeze().T
