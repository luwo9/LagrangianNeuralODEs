"""
This module contains metrics to directly evaluate an ODE instead of its
solutions.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import qmc


def _strech_extend(min_val, max_val, factor):
    # Stretch the interval [min_val, max_val] by a given factor around
    # its center.
    center = 0.5 * (min_val + max_val)
    half_range = 0.5 * (max_val - min_val)
    new_half_range = factor * half_range
    return center - new_half_range, center + new_half_range


def _domain_evaluation(
    f: Callable,
    t_extend: tuple[float, float],
    x_extend: tuple[np.ndarray, np.ndarray],
    xdot_extend: tuple[np.ndarray, np.ndarray],
    n_points: int = 10**6,
    extrapolate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Evaluate a second order ODE $\ddot{x} = f(t, x, \dot{x})$ on the
    full domain using Sobol sampling.

    Parameters
    ----------

    f : Callable
        The ODE, with call signature f(t, x, xdot), where t has shape
        (n_batch, n_steps), x and xdot have shape
        (n_batch, n_steps, n_dim).
    t_extend : tuple[float, float]
        The time interval on which to evaluate the ODEs.
    x_extend : tuple[np.ndarray, np.ndarray]
        The minimum and maximum position for each dimension of the
        domain. Array shape should be (n_dim,).
    xdot_extend : tuple[np.ndarray, np.ndarray]
        The minimum and maximum derivative for each dimension of the
        domain. Array shape should be (n_dim,).
    n_points : int, default=10**6
        The total number of points to sample.
    extrapolate : float, default=1.0
        Factor by which to extend the domain in each direction,
        relative to the provided min and max values. The domain is then
        only the shell between the original and extended domain.
        Default (1.0) means no extrapolation, i.e., sampling inside the
        provided domain.

    Returns
    -------

    f_sample : np.ndarray, shape (n_points, 1, n_dim)
        The evaluated ODE at the sampled points.
    t_sample : np.ndarray, shape (n_points, 1)
        The sampled time points.
    x_sample : np.ndarray, shape (n_points, 1, n_dim)
        The sampled position points.
    xdot_sample : np.ndarray, shape (n_points, 1, n_dim)
        The sampled derivative points.
    """
    if extrapolate < 1.0:
        raise ValueError("extrapolate must be >= 1.0")
    rng = np.random.default_rng()
    n_dim = x_extend[0].shape[0]
    n_dim_total = 2 * n_dim + 1
    t_extend_extrap = _strech_extend(t_extend[0], t_extend[1], extrapolate)
    x_extend_extrap = _strech_extend(x_extend[0], x_extend[1], extrapolate)
    xdot_extend_extrap = _strech_extend(xdot_extend[0], xdot_extend[1], extrapolate)

    sampler = qmc.Sobol(d=n_dim_total)

    # If extrapolate = 1.0, this is just a simple sample from the
    # n_dim_total-dimensional unit cube, split into t, x, and xdot and
    # scaled to the respective intervals.
    # If extrapolate > 1.0, do a simple rejection sampling to only
    # keep points in the shell between the original and extended
    # domain.
    # Such a rejection sampling might violate some of the qmc
    # properties, but should be acceptable for this purpose.
    n_accepted = 0
    accepted = {"t": [], "x": [], "xdot": []}
    while n_accepted < n_points:
        log2_n_points = int(np.ceil(np.log2(n_points))) + 1  # Oversample a bit
        sample = sampler.random_base2(m=log2_n_points)

        t_sample = qmc.scale(sample[:, :1], t_extend_extrap[0], t_extend_extrap[1])
        x_sample = qmc.scale(sample[:, 1 : 1 + n_dim], x_extend_extrap[0], x_extend_extrap[1])
        xdot_sample = qmc.scale(
            sample[:, 1 + n_dim :], xdot_extend_extrap[0], xdot_extend_extrap[1]
        )

        is_shell = (
            (t_sample < t_extend[0]).any(-1)
            | (t_sample >= t_extend[1]).any(-1)
            | (x_sample < x_extend[0]).any(-1)
            | (x_sample >= x_extend[1]).any(-1)
            | (xdot_sample < xdot_extend[0]).any(-1)
            | (xdot_sample >= xdot_extend[1]).any(-1)
        )  # .any(-1) (faster here than above?)
        valid = is_shell if extrapolate > 1.0 else ~is_shell
        n_accepted += np.sum(valid)
        accepted["t"].append(t_sample[valid])
        accepted["x"].append(x_sample[valid])
        accepted["xdot"].append(xdot_sample[valid])

    # Randomly choose n_points from the accepted points.
    do_keep = rng.permutation(n_accepted)[:n_points]
    t_sample = np.concatenate(accepted["t"], axis=0)[do_keep]
    x_sample = np.concatenate(accepted["x"], axis=0)[do_keep]
    xdot_sample = np.concatenate(accepted["xdot"], axis=0)[do_keep]

    # Reshape with n_points as batch dimension and 1 as step/time
    # dimension.
    x_sample = x_sample.reshape(-1, 1, n_dim)
    xdot_sample = xdot_sample.reshape(-1, 1, n_dim)

    f_sample = f(t_sample, x_sample, xdot_sample)

    return f_sample, t_sample, x_sample, xdot_sample


def domain_mse(
    f_true: Callable,
    f_pred: Callable,
    t_extend: tuple[float, float],
    x_extend: tuple[np.ndarray, np.ndarray],
    xdot_extend: tuple[np.ndarray, np.ndarray],
    n_points: int = 10**6,
    extrapolate: float = 1.0,
) -> float:
    r"""
    Compute the mean squared error of two second order ODEs
    $\ddot{x} = f(t, x, \dot{x})$ on the full domain using
    Sobol sampling.

    Parameters
    ----------

    f_true : Callable
        The true ODE, with call signature f(t, x, xdot), where t has
        shape (n_batch, n_steps), x and xdot have shape
        (n_batch, n_steps, n_dim).
    f_pred : Callable
        The predicted ODE, with call signature f(t, x, xdot), where t
        has shape (n_batch, n_steps), x and xdot have shape
        (n_batch, n_steps, n_dim).
    t_extend : tuple[float, float]
        The time interval on which to evaluate the ODEs.
    x_extend : tuple[np.ndarray, np.ndarray]
        The minimum and maximum position for each dimension of the
        domain. Array shape should be (n_dim,).
    xdot_extend : tuple[np.ndarray, np.ndarray]
        The minimum and maximum derivative for each dimension of the
        domain. Array shape should be (n_dim,).
    n_points : int, default=10**6
        The total number of points to sample.
    extrapolate : float, default=1.0
        Factor by which to extend the domain in each direction,
        relative to the provided min and max values. The domain is then
        only the shell between the original and extended domain.
        Default (1.0) means no extrapolation, i.e., sampling inside the
        provided domain.

    Returns
    -------

    float
        The mean squared error of the two ODEs on the domain.
    """
    f_true_sample, *_ = _domain_evaluation(
        f_true, t_extend, x_extend, xdot_extend, n_points, extrapolate
    )
    f_pred_sample, *_ = _domain_evaluation(
        f_pred, t_extend, x_extend, xdot_extend, n_points, extrapolate
    )

    mse = np.mean((f_true_sample - f_pred_sample) ** 2)
    return mse
