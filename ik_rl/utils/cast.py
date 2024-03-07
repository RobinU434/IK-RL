from numpy import ndarray
from torch import Tensor


def tensor_to_ndarray(tensor: Tensor | ndarray) -> ndarray:
    """assumes the input is either a ndarray or a tensor and returns a ndarray

    Args:
        tensor (Tensor | ndarray): input array

    Returns:
        ndarray: nd array
    """
    if isinstance(tensor, Tensor):
        # detach if the action tensor requires grad = True
        return tensor.detach().numpy()
    return tensor
