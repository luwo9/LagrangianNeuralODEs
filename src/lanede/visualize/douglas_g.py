"""
This module contains logic for plotting the g matrix from the
Helmholtz conditions as formulated by Douglas.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from ._pyplot import subplots_with_custom_row_spacing


_formatter_g = ScalarFormatter()
_formatter_g.set_powerlimits((-2, 2))


def _plot_matrix_function_stacked(
    axs: Sequence[Axes],
    matrices: np.ndarray,
    components: np.ndarray,
    matrix_name: str,
    component_name: str,
    **kwargs,
) -> None:
    # Plots a n x n matrix function on n x n axes, plotting multiple
    # versions of the function on the same axes.
    # axs shape: (n_dim, n_dim)
    # matirces shape: (n_batch, n_steps, n_dim, n_dim)
    # components shape: (n_batch, n_steps)
    n_dim = axs.shape[0]
    matrices = matrices.transpose(2, 3, 0, 1)
    matrices = matrices.reshape(-1, *matrices.shape[2:])
    axs = axs.flatten()

    for (i, j), matrix_elements, ax in zip(np.ndindex(n_dim, n_dim), matrices, axs):
        ax.set_ylabel(rf"{matrix_name}$_{{{i+1}{j+1}}}$")
        ax.yaxis.set_major_formatter(_formatter_g)
        ax.tick_params(axis="x", which="both", labelsize="small")
        ax.tick_params(axis="y", which="both", labelsize="small")

        if i == n_dim - 1:
            ax.set_xlabel(component_name)
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if i + j > 0:
            ax.sharex(axs[0])

        for matrix_element, component in zip(matrix_elements, components):
            ax.plot(component, matrix_element, **kwargs)


def plot_g_matrix(
    g_matrix: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    xdot: np.ndarray,
    n_random: int = 10,
    t_component: bool = True,
    state_components: tuple[int] | None = None,
    derivative_components: tuple[int] | None = None,
    state_names: list[str] | None = None,
    derivative_names: list[str] | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot the t, x, and xdot dependence of the g matrix.

    Parameters
    ----------

    g_matrix : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
        The g matrix to plot at each time step.
    t : np.ndarray, shape (n_batch, n_steps)
        The time steps.
    x : np.ndarray, shape (n_batch, n_steps, n_dim)
        The state at the time steps.
    xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
        The derivative of the state at the time steps.
    n_random : int, optional
        The number of randomly selected time series to plot g for.
    t_component : bool, default=True
        Whether to plot the time dependence of g.
    state_components : tuple[int], optional
        The components of the state to plot. If None, all components
        are plotted.
    derivative_components : tuple[int], optional
        The components of the derivative to plot. If None, all
        components are plotted.
    component_names : list[str], optional
        The names of the components of the state selected by
        state_components. If None, the components are named
        "$x_i$", where $i$ is the index.
    derivative_names : list[str], optional
        The names of the components of the derivative selected by
        derivative_components. If None, the components are named
        "$\dot{x}_i$", where $i$ is the index.

    Returns
    -------

    Figure
        The matplotlib figure object.
    Sequence[Axes]
        The axes of the figure.
    """
    rng = np.random.default_rng()
    n_batch, _, n_dim, _ = g_matrix.shape

    state_components = state_components or tuple(range(n_dim))
    derivative_components = derivative_components or tuple(range(n_dim))
    state_components = np.array(state_components, dtype=int)
    derivative_components = np.array(derivative_components, dtype=int)
    state_names = state_names or [rf"$x_{i}$" for i in state_components]
    derivative_names = derivative_names or [rf"$\dot{{x}}_{i}$" for i in derivative_components]

    # Create figure and axes
    # TODO: Maybe use subplot_mosaic to get t central at top and
    # x, xdot next to each other below
    n_cols = n_dim
    n_rows = n_dim * (len(state_components) + len(derivative_components) + t_component)
    figsize = (n_cols * 3.6, n_rows * 2.2)
    min_space = 0.01
    gap_space = 0.3
    row_spacings = np.full(n_rows - 1, min_space)
    row_spacings[n_dim - 1 :: n_dim] = gap_space
    fig, axs = subplots_with_custom_row_spacing(
        n_rows,
        n_cols,
        row_spacings=list(row_spacings),
        layout="constrained",
        figsize=figsize,
    )

    common_plot_kwargs = {"alpha": 0.5}

    # Select subset of data to plot
    rng_indices = rng.choice(n_batch, n_random, replace=False)
    g_matrix = g_matrix[rng_indices]
    t = t[rng_indices]
    x = x[rng_indices][:, :, state_components].transpose(2, 0, 1)
    xdot = xdot[rng_indices][:, :, derivative_components].transpose(2, 0, 1)

    axs_by_component = np.split(axs, n_rows // n_dim)
    if t_component:
        _plot_matrix_function_stacked(
            axs_by_component[0],
            g_matrix,
            t,
            "$g$",
            "$t$",
            **common_plot_kwargs,
        )

    for components_name, components, axs_ in zip(state_names, x, axs_by_component[t_component:]):
        _plot_matrix_function_stacked(
            axs_,
            g_matrix,
            components,
            "$g$",
            components_name,
            **common_plot_kwargs,
        )

    start = t_component + len(state_components)
    for components_name, components, axs_ in zip(derivative_names, xdot, axs_by_component[start:]):
        _plot_matrix_function_stacked(
            axs_,
            g_matrix,
            components,
            "$g$",
            components_name,
            **common_plot_kwargs,
        )

    return fig, axs
