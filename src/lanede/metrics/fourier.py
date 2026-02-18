"""
This module contains metrics to charecterize the fourier spectrum of
time series.
"""

import numpy as np


def _fft(time_series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the real FFT of a time series.

    Parameters
    ----------

    time_series : np.ndarray, shape (..., n_steps, n_dim)
        The time series to compute the FFT of.
    dt : float
        The time step of the time series.

    Returns
    -------

    np.ndarray, shape (n_freqs,)
        The frequencies corresponding to the FFT values.
    np.ndarray, shape (..., n_freqs, n_dim)
        The FFT of the time series.

    Notes
    -----
    Since the real FFT is used n_freqs = n_steps // 2 + 1.
    """
    n = time_series.shape[-2]
    frequency = np.fft.rfftfreq(n, d=dt)
    fft_values = np.fft.rfft(time_series, axis=-2)
    return frequency, fft_values


def fourier_metrics(
    time_series: np.ndarray, dt: float, average_dims: bool = False
) -> dict[str, np.ndarray]:
    """
    Compute several metrics that characterize the fourier spectrum of a
    time series.

    These metrics summarizing the fourier spectrum are averaged over a
    batch dimension and, optionally over the dimensions of the time
    series.

    Parameters
    ----------

    time_series : np.ndarray, shape (*N, n_batch, n_steps, n_dim)
        The time series to compute the metrics for.
    dt : float
        The time step of the time series.
    average_dims : bool, default=False
        If True, average over the dimensions (last axis) of the time
        series as well.

    Returns
    -------

    dict
        A dictionary where each value is an array of shape
        (*N, n_dim) if average_dims is False, or (*N)
        if average_dims is True.
        The keys are:
        - "mean": The power-weighted mean of the frequencies.
        - "bandwidth": The bandwidth of the frequencies.
        - "entropy": The entropy of the power spectrum.
    """
    if time_series.ndim < 3:
        raise ValueError("time_series must have at least 3 dimensions (n_batch, n_steps, n_dim)")

    frequency, fft_values = _fft(time_series, dt)
    frequency = frequency[..., None]  # Add a new axis for broadcasting

    power_spectrum = np.abs(fft_values) ** 2
    total_power = np.sum(power_spectrum, axis=-2)

    # Power-weighted mean frequency
    mean_frequency = np.sum(frequency * power_spectrum, axis=-2) / total_power

    # Bandwidth: standard deviation of the frequencies weighted by the power spectrum
    frequency_diff = frequency - mean_frequency[..., None, :]
    bandwidth = np.sum(frequency_diff**2 * power_spectrum, axis=-2) / total_power
    bandwidth = np.sqrt(bandwidth)

    # Entropy of the power spectrum
    norm_power_spectrum = power_spectrum / total_power[..., None, :]
    entropy = -np.sum(norm_power_spectrum * np.log(norm_power_spectrum + 1e-10), axis=-2)

    metrics = {
        "mean": mean_frequency,
        "bandwidth": bandwidth,
        "entropy": entropy,
    }
    mean_axis = (-2, -1) if average_dims else -2  # -2 for n_batch, -1 for n_dim
    for key, value in metrics.items():
        metrics[key] = np.mean(value, axis=mean_axis)

    return metrics


def fourier_mse(
    time_series_pred: np.ndarray,
    time_series_true: np.ndarray,
    dt: float,
    average_dims: bool = False,
) -> np.ndarray:
    """
    Compute the normalized mean squared error of the power spectrum
    of two time series in the fourier domain.

    Parameters
    ----------
    time_series_pred : np.ndarray, shape (*N, n_batch, n_steps, n_dim)
        The predicted time series.
    time_series_true : np.ndarray, shape (*N, n_batch, n_steps, n_dim)
        The true time series.
    dt : float
        The time step of the time series.
    average_dims : bool, default=False
        If True, average over the dimensions (last axis) of the time series as well.

    Returns
    -------
    np.ndarray, shape (*N,) or (*N, n_dim)
        The mean squared error of the power spectrum of the predicted
        and true time series. If average_dims is True, the result is
        shape (*N,). If average_dims is False, the result is shape
        (*N, n_dim).
    """
    stacked = np.stack((time_series_pred, time_series_true), axis=0)
    _, fft_values = _fft(stacked, dt)
    power_spectrum = np.abs(fft_values) ** 2
    predicted_power_spectrum, true_power_spectrum = power_spectrum[0], power_spectrum[1]
    # Always average over batch and steps
    # -3 for n_batch, -2 for n_steps, -1 for n_dim
    axis = (-3, -2, -1) if average_dims else (-3, -2)
    mse = np.mean((predicted_power_spectrum - true_power_spectrum) ** 2, axis=axis)
    norm = np.mean(true_power_spectrum**2, axis=axis)
    return np.mean(mse / (norm + 1e-10))
