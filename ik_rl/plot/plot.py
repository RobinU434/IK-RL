"""from typing import Callable, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def env_plot(
    plot_func: Callable,
    fig: Figure = None,
    ax: Axes = None,
    *kwargs,
) -> Tuple[Figure, Axes]:

    def wrapper():
        if fig is None:
            fig, ax = plt.subplots()
        ax = plot_func(ax, *kwargs)

    return wrapper


def plot_base(ax: Axes):
    pass
"""
