"""
This module defines functions to investigate system specific
quantities, like the energy of an oscillator.
"""

import numpy as np


def _oscillator_energy(x: np.ndarray, xdot: np.ndarray, omega: float) -> np.ndarray:
    r"""
    Compute the energy of a harmonic oscillator. Only valid if all
    dimensions have the same frequency.

    Parameters
    ----------

    x : np.ndarray, shape (*N, n_dim)
        The position time series of the oscillator.
    xdot : np.ndarray, shape (*N, n_dim)
        The velocity time series of the oscillator.
    omega : float
        The frequency of the oscillator.

    Returns
    -------

    energy : np.ndarray, shape (*N,)
        The energy time series of the oscillator.

    Notes
    -----

    The energy is computed as
    $$ E = \frac{1}{2} \left( \|\dot{x}\|^2 + \omega^2 \|x\|^2 \right).$$
    """
    x_norm = np.linalg.norm(x, axis=-1)
    xdot_norm = np.linalg.norm(xdot, axis=-1)
    energy = 0.5 * (xdot_norm**2 + omega**2 * x_norm**2)
    return energy


def oscillator_energy_mse(
    x_pred: np.ndarray,
    xdot_pred: np.ndarray,
    x_true: np.ndarray,
    xdot_true: np.ndarray,
    omega: float,
) -> float:
    r"""
    Compute the mean squared error of the energy of a harmonic
    oscillator. Only valid if all dimensions have the same frequency.

    Parameters
    ----------

    x_pred : np.ndarray, shape (*N, n_dim)
        The predicted position time series of the oscillator.
    xdot_pred : np.ndarray, shape (*N, n_dim)
        The predicted velocity time series of the oscillator.
    x_true : np.ndarray, shape (*N, n_dim)
        The true position time series of the oscillator.
    xdot_true : np.ndarray, shape (*N, n_dim)
        The true velocity time series of the oscillator.
    omega : float
        The frequency of the oscillator.

    Returns
    -------

    mse : float
        The mean squared error of the energy time series.
    """
    energy_pred = _oscillator_energy(x_pred, xdot_pred, omega)
    energy_true = _oscillator_energy(x_true, xdot_true, omega)
    mse = np.mean((energy_pred - energy_true) ** 2)
    return mse
