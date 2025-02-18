"""Contains general helper functions for plotting with matplotlib."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def subplots_with_custom_row_spacing(
    n_rows: int, n_cols: int, *, row_spacings: list[float] | None = None, **kwargs
) -> tuple[Figure, Sequence[Axes]]:
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
