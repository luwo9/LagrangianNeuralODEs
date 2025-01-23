"""
This module contains ploting functions for time series data, namely
a state and its derivatives over time and comparing it to the
corresponding prediction.
"""

from collections.abc import Sequence
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _process_data_inputs(
    predictions: tuple[np.ndarray | None, ...] | None = None,
    data: tuple[np.ndarray | None, ...] | None = None,
    true: tuple[np.ndarray | None, ...] | None = None,
) -> tuple[
    int,
    tuple[np.ndarray | None, ...],
    tuple[np.ndarray | None, ...],
    tuple[np.ndarray | None, ...],
]:
    # Check that the data is given in the correct format:
    # - Some data must be given
    # - If state or derivatives are given, time must be given

    # Return the number of quantities (state, derivatives), the number
    # of time series, the dimensionality of the state and the data
    # tuples with the data and None for missing data.
    supplied_data = [data_ for data_ in (predictions, data, true) if data_ is not None]
    if not supplied_data:
        raise ValueError("At least one of predictions, data or true must be given.")

    lengths = [len(data_) for data_ in supplied_data]
    n_quantities = max(lengths) - 1

    for data_ in supplied_data:
        is_none = [quantity is None for quantity in data_]
        if is_none[0] and not all(is_none):
            raise ValueError("Time must be given if state or derivatives are given.")

    out_data = []
    for data_ in (predictions, data, true):
        if data_ is None:
            out_data.append((None,) * (n_quantities + 1))
        else:
            out_data.append(data_ + (None,) * (n_quantities - len(data_) + 1))

    for data_ in out_data:
        for quantity in data_[1:]:
            if quantity is None:
                continue
            # Could check that n_batch and n_dim are the same for all
            # quantities.
            n_batch, _, n_dim = quantity.shape
            return n_quantities, n_batch, n_dim, out_data


def _subplots_with_custom_row_spacing(n_rows, n_cols, *, row_spacings=None, **kwargs):
    # Like plt.subplots, but with custom row spacings
    # May overwrite som kwargs
    kwargs["squeeze"] = False
    if row_spacings is None:
        return plt.subplots(n_rows, n_cols, **kwargs)

    n_gaps = n_rows - 1
    n_new_rows = n_rows + n_gaps
    height_ratios = []
    for spacing in row_spacings:
        height_ratios.extend([1, spacing])
    height_ratios.append(1)

    kwargs["height_ratios"] = height_ratios
    fig, axs = plt.subplots(n_new_rows, n_cols, **kwargs)

    # Turn off spacing axes and remove the gap axs from the array
    spacing_axes = axs[1::2]
    for ax in spacing_axes.flatten():
        ax.axis("off")
    axs = axs[::2]

    return fig, axs


# TODO: Add option to specify units
def plot_timeseries(
    predictions: tuple[np.ndarray | None, ...] | None = None,
    data: tuple[np.ndarray | None, ...] | None = None,
    true: tuple[np.ndarray | None, ...] | None = None,
    component_names: list[str] | None = None,
    n_random: int = 5,
) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot the time evolution of a state and its derivatives. Allows
    comparing a prediction (plotted as a line) to data (plotted as
    dots) and a ground truth (plotted as a dashed line).

    This function will plot the state with all derivatives of all
    components for several selected time series.

    Parameters
    ----------
    predictions : tuple[np.ndarray | None], optional
        The predicted state and its derivatives. See Notes for details.
    data : tuple[np.ndarray | None], optional
        The data state and its derivatives. See Notes for details.
    true : tuple[np.ndarray | None], optional
        The true state and its derivatives. See Notes for details.
    component_names : list[str], optional
        The names of the components of the state. If None, the
        components are named "Component 0", "Component 1", ...
    n_random : int, optional
        The number of randomly selected time series to plot. If it is
        0, all time series are plotted in order.

    Returns
    -------

    fig : Figure
        The figure containing the plots.
    axs : Sequence[Axes]
        The axes on which the data was plotted.

    Notes
    -----

    The input data is expected to be a tuple of numpy arrays of kind
    (time, state, derivative, second_derivative). The time has shape
    (n_steps), the state and its derivatives have shape
    (n_batch, n_steps, n_dim).

    State and derivatives are plotted to the respective axes in the
    order state, derivative, second_derivative.

    State or derivatives can be None, or only supplied upto a certain
    derivative. In this case, the missing data is not plotted.
    """
    # Process inputs and set defaults
    *dimensions, processed_inputs = _process_data_inputs(predictions, data, true)
    n_quantities, n_batch, n_dim = dimensions
    predictions, data, true = processed_inputs

    rng = np.random.default_rng()
    if n_random:
        time_series_indices_plot = rng.choice(n_batch, n_random, replace=False)
    else:
        # Keep the order
        time_series_indices_plot = np.arange(n_batch)
    n_time_series = len(time_series_indices_plot)

    if component_names is None:
        component_names = [f"Component {i}" for i in range(1, n_dim + 1)]
    # Maybe make arguments too:
    quantity_names = ["State", "Derivative", "Second Derivative"]
    plot_kwargs = [
        {"label": "Predicted", "color": "blue"},
        {"label": "Data", "color": "black", "marker": ".", "linestyle": "none"},
        {"label": "True", "color": "blue", "linestyle": "--"},
    ]

    # Set up the plot
    n_rows = n_quantities * n_time_series
    n_cols = n_dim
    figsize = (n_cols * 4, n_rows)
    # fig, axs = plt.subplots(n_rows, n_cols, sharex=True, layout="constrained", figsize=figsize)
    min_space = 0.01
    gap_space = 0.3
    row_spacings = np.full(n_rows - 1, min_space)
    row_spacings[n_quantities - 1 :: n_quantities] = gap_space
    fig, axs = _subplots_with_custom_row_spacing(
        n_rows,
        n_cols,
        sharex=True,
        layout="constrained",
        figsize=figsize,
        row_spacings=list(row_spacings),
    )

    # Label the plot as follows:
    # - The first column gets ylabels of the quantity names
    # - The first row gets titles of the component names
    # - The last row gets xlabels of time
    # - Top right corner gets a legend
    #
    # To make this more readable with the loop below, we supply this
    # information with every object in the numpy array.
    axs_with_notes = np.empty_like(axs, dtype=object)
    for (i, j), ax in np.ndenumerate(axs):
        new_ax = {
            "ax": ax,
            "title": i == 0,
            "ylabel": j == 0,
            "xlabel": i == n_rows - 1,
            "legend": i == 0 and j == n_cols - 1,
        }
        axs_with_notes[i, j] = new_ax

    # Go through the data by interpretation
    # (This involves sevaral loops, but is faster and more readable)
    for data_, plot_kwargs in zip((predictions, data, true), plot_kwargs):

        time = data_[0]
        if time is None:  # No data at all, skip this type completely
            continue

        axes_by_quantitiy = [axs_with_notes[i::n_quantities] for i in range(n_quantities)]

        for quantity, axs_quantity, quantity_name in zip(
            data_[1:], axes_by_quantitiy, quantity_names
        ):
            if quantity is None:  # No data for this quantity, skip it
                continue

            # Make the next two loops one loop by flattening the axes
            # and using itertools.product
            axs_by_series_and_component = axs_quantity.flatten()
            indices_series_and_component_with_names = itertools.product(
                time_series_indices_plot, enumerate(component_names)
            )

            for ax_with_notes, (i_series, (i_component, component_name)) in zip(
                axs_by_series_and_component, indices_series_and_component_with_names
            ):
                ax: Axes = ax_with_notes["ax"]
                ax.plot(time, quantity[i_series, :, i_component], **plot_kwargs)

                ax.tick_params(axis="y", which="both", labelsize="x-small")
                ax.tick_params(axis="x", which="both", labelsize="small")

                # Retrieve whether to add labels etc. and add the right
                # ones
                if ax_with_notes["title"]:
                    ax.set_title(component_name, fontsize="small")
                if ax_with_notes["ylabel"]:
                    ax.set_ylabel(quantity_name, fontsize="x-small")
                    ax.yaxis.set_label_coords(-0.12, 0.5)
                if ax_with_notes["xlabel"]:
                    ax.set_xlabel("Time", fontsize="small")
                if ax_with_notes["legend"]:
                    ax.legend(loc="upper right")

    # Add text to the left explaining the different series
    fig.supylabel(
        r"$\longleftarrow$Different time series$\longrightarrow$",
        x=-0.05,
        rotation="vertical",
        fontweight="bold",
    )

    return fig, axs
