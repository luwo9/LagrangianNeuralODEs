"""
This module contains logic for plotting the g matrix from the
Helmholtz conditions as formulated by Douglas.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import cycler

from ._pyplot import subplots_with_custom_row_spacing


_formatter_g = ScalarFormatter()
_formatter_g.set_powerlimits((-2, 2))


def _plot_matrix_function_stacked(
    axs: Sequence[Axes],
    matrices: np.ndarray,
    components: np.ndarray,
    matrix_name: str,
    component_name: str,
    stack_plot_kwargs: cycler.Cycler | None = None,
) -> None:
    # Plots a n x n matrix function on n x n axes, plotting multiple
    # versions of the function on the same axes.
    # axs shape: (n_dim, n_dim)
    # matirces shape: (n_batch, n_steps, n_dim, n_dim)
    # components shape: (n_batch, n_steps)
    # stack_plot_kwargs is a cycler that is used when plotting the
    # stacked matrix elements
    n_dim = axs.shape[0]
    matrices = matrices.transpose(2, 3, 0, 1)
    matrices = matrices.reshape(-1, *matrices.shape[2:])
    axs = axs.flatten()

    for (i, j), matrix_element_stack, ax in zip(np.ndindex(n_dim, n_dim), matrices, axs):
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

        for matrix_element, component, kwargs in zip(
            matrix_element_stack, components, stack_plot_kwargs
        ):
            ax.plot(component, matrix_element, **kwargs)


def plot_g_matrix(
    g_matrix: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    xdot: np.ndarray,
    g_analytic: np.ndarray | None = None,
    n_random: int = 10,
    residuals: bool = False,
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
    g_analytic : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim), optional
        The analytic g matrix to plot at each time step. If None,
        no analytic g matrix is plotted.
    n_random : int, default=10
        The number of randomly selected time series to plot g for.
    residuals : bool, default=False
        Whether to plot the residuals of the g matrix, i.e., the
        difference between the g matrix and the analytic g matrix,
        relative to the analytic g matrix. The analytic g matrix
        must be provided for this.
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

    # Get the components to plot and their names
    state_components = tuple(range(n_dim)) if state_components is None else state_components
    derivative_components = tuple(range(n_dim)) if derivative_components is None else derivative_components
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

    # In the below, the analytic g matrix is stacked in the batch
    # dimension, so that only a single call to the plotting function
    # is needed. The plot properties can simply be adjusted by
    # constructing an appropriate cycler object.

    # If residuals are requested, plot them instead

    # Setup plot kwargs
    # TODO: allow e.g. sampling from colormap
    common = cycler.cycler(alpha=[0.5]) # length 1
    colors = plt.rcParams["axes.prop_cycle"]
    common_plot_kwargs = colors * common
    
    if g_analytic is not None and not residuals:
        # Make analytic trajectories dashed
        linestyle = cycler.cycler(linestyle=["-", "--"])
        common_plot_kwargs = linestyle * common_plot_kwargs # cycle color first

        # Label analytic and learned g matrix
        legend_elements = [
            Line2D([0], [0], color="black", linestyle="-", label="Learned $g$"),
            Line2D([0], [0], color="black", linestyle="--", label="Analytic $g$"),
        ]
        fig.legend(handles=legend_elements, loc="upper right")

    # Select subset of data to plot
    rng_indices = rng.choice(n_batch, n_random, replace=False)
    g_matrix = g_matrix[rng_indices]
    t = t[rng_indices]
    x = x[rng_indices][:, :, state_components]
    xdot = xdot[rng_indices][:, :, derivative_components]

    if g_analytic is not None:
        g_analytic = g_analytic[rng_indices]

        if residuals:
            # Use residuals instead of the actual g matrix.
            g_matrix = (g_matrix - g_analytic) / np.abs(g_analytic)
        else:
            # Stack the analytic g matrix in the batch dimension, see above
            g_matrix = np.concatenate((g_matrix, g_analytic), axis=0)
            t = np.concatenate((t, t), axis=0)
            x = np.concatenate((x, x), axis=0)
            xdot = np.concatenate((xdot, xdot), axis=0)
            

    # Permute to (n_dim, n_batch, n_steps) to loop over the components
    # first, as the plotting function is called for each component
    x = x.transpose(2, 0, 1)
    xdot = xdot.transpose(2, 0, 1)

    axs_by_component = np.split(axs, n_rows // n_dim)
    if t_component:
        _plot_matrix_function_stacked(
            axs_by_component[0],
            g_matrix,
            t,
            "$g$",
            "$t$",
            common_plot_kwargs,
        )

    for components_name, components, axs_ in zip(state_names, x, axs_by_component[t_component:]):
        _plot_matrix_function_stacked(
            axs_,
            g_matrix,
            components,
            "$g$",
            components_name,
            common_plot_kwargs,
        )

    start = t_component + len(state_components)
    for components_name, components, axs_ in zip(derivative_names, xdot, axs_by_component[start:]):
        _plot_matrix_function_stacked(
            axs_,
            g_matrix,
            components,
            "$g$",
            components_name,
            common_plot_kwargs,
        )

    return fig, axs
