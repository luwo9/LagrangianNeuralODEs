"""
This module contains functions to modify the toy dataset, e.g., by
adding noise.
"""

import numpy as np


def add_noise(data: np.ndarray, noise_level: float = 0.05):
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

    Returns
    -------

    np.ndarray
        The data with added noise.
    """
    sigma = np.mean(np.abs(data)) * noise_level
    rng = np.random.default_rng()
    return data + rng.normal(0, sigma, size=data.shape)
