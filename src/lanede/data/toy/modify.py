"""
This module contains functions to modify the toy dataset, e.g., by
adding noise.
"""

import numpy as np


def add_noise(
    data: np.ndarray, noise_level: float = 0.05, component_wise: bool = False
) -> np.ndarray:
    """
    Add fake noise to the data.

    This is done by adding Gaussian noise with a standard deviation of
    mean(abs(data)) * noise_level.

    Parameters
    ----------

    data : np.ndarray
        The data to add noise to.
    noise_level : float
        The noise level as a fraction of the mean absolute value of the
        data.
    component_wise : bool
        If True, the noise is added component-wise, i.e., the standard
        deviation is calculated for each component along the last
        dimension. If False, the standard deviation is calculated for
        the entire data array.

    Returns
    -------

    np.ndarray
        The data with added noise.
    """
    n_dims_mean = data.ndim - component_wise
    mean_abs = np.mean(np.abs(data), axis=tuple(range(n_dims_mean)))

    noise_std = mean_abs * noise_level
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise