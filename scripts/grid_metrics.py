"""
Compute additional metrics for a grid of oscillator models already
trained with grid_models.py.
"""

import pathlib
import json
import torch.multiprocessing as mp

import numpy as np

from lanede.api import LanedeAPI
from lanede.data.toy import add_noise, from_ode, DampedHarmonicOscillator
from lanede.metrics import fourier_mse, fourier_metrics, oscillator_energy_mse, domain_mse

# General settings
# As in grid_models.py
DIRECTORY = "grid_models1"
BASE_NAME = "oscill_grid"
CONFIG_NAME = "grid_config"

# File to save metrics to, if it already exists, it will be updated.
METRIC_NAME = "metrics/metrics"

# When exploring a time series, extrapolate to this time
EXTRAPOLATION_TIME = 3.0
# When exploring the full (t, x, xdot) domain:
# Factor by which the domain is extended beyond the data range for
# extrapolation
DOMAIN_EXTRAPOLATION = 3.0
# Number of points to sample in the domain
N_POINTS_DOMAIN = 10**7

# Technical settings
N_JOBS = 20

# General oscillator settings
NOISE_LEVEL_DATA = 0.05


def compute_metrics(config: dict, model: LanedeAPI) -> dict[str, float]:
    """
    Compute metrics given a configuration and a model.
    The configuration must be in line with grid_models.py.
    """
    # All metrics are computed in this one function for efficiency and simplicity,
    # as e.g. different parts of model and data may be used/reused.
    n_periods = config["n_periods"]
    damping = config["damping"]
    x_0_std = config["x_0_std"]

    all_metrics = {}

    omega = 2 * np.pi * n_periods
    # fmt: off
    spring_matrix = np.array([[omega**2, 0],
                              [0, omega**2]])
    damping_matrix = np.array([[omega, 0],
                               [0, omega]]) * damping
    # fmt: on

    rng = np.random.default_rng()
    oscillator = DampedHarmonicOscillator(spring_matrix, damping_matrix)

    # Get a high-resolution time series for the metric
    t_metric = np.linspace(0, EXTRAPOLATION_TIME, 2000)
    is_extrap = t_metric > 1
    x_0_metric = 1 + rng.normal(size=(6000, 2)) * x_0_std
    xdot_0_metric = np.sqrt(x_0_metric**2) / 10

    # Model prediction
    x_0_metric_noise = add_noise(x_0_metric, NOISE_LEVEL_DATA)
    x_pred, xdot_pred = model.predict(t_metric, x_0_metric_noise)
    t_plot_with_batches = np.tile(t_metric, (x_pred.shape[0], 1))
    xdotdot_pred = model.second_derivative(t_plot_with_batches, x_pred, xdot_pred)

    # True time series
    x_true, xdot_true, xdotdot_true = from_ode(oscillator, t_metric, x_0_metric, xdot_0_metric)

    # Sort out the extrapolation and interpolation parts
    t_interp = t_metric[~is_extrap]
    x_pred_interp = x_pred[:, ~is_extrap]
    xdot_pred_interp = xdot_pred[:, ~is_extrap]
    xdotdot_pred_interp = xdotdot_pred[:, ~is_extrap]
    x_true_interp = x_true[:, ~is_extrap]
    xdot_true_interp = xdot_true[:, ~is_extrap]
    xdotdot_true_interp = xdotdot_true[:, ~is_extrap]

    t_extrap = t_metric[is_extrap]
    x_pred_extrap = x_pred[:, is_extrap]
    xdot_pred_extrap = xdot_pred[:, is_extrap]
    xdotdot_pred_extrap = xdotdot_pred[:, is_extrap]
    x_true_extrap = x_true[:, is_extrap]
    xdot_true_extrap = xdot_true[:, is_extrap]
    xdotdot_true_extrap = xdotdot_true[:, is_extrap]

    # Metrics:
    names = ["x", "xdot", "xdotdot"]
    kinds = ["interp", "extrap"]
    # Metric 1: MSE of all kinematic quantities
    all_metrics["mse_x_interp"] = np.mean((x_pred_interp - x_true_interp) ** 2)
    all_metrics["mse_xdot_interp"] = np.mean((xdot_pred_interp - xdot_true_interp) ** 2)
    all_metrics["mse_xdotdot_interp"] = np.mean((xdotdot_pred_interp - xdotdot_true_interp) ** 2)

    all_metrics["mse_x_extrap"] = np.mean((x_pred_extrap - x_true_extrap) ** 2)
    all_metrics["mse_xdot_extrap"] = np.mean((xdot_pred_extrap - xdot_true_extrap) ** 2)
    all_metrics["mse_xdotdot_extrap"] = np.mean((xdotdot_pred_extrap - xdotdot_true_extrap) ** 2)

    # Metric 2: Fourier metrics
    dt = t_metric[1] - t_metric[0]
    # Stack the x, xdot and xdotdot, to compute fourier metrics in one
    # pass and saves space when computing n_metrics x 3 x 2 total
    # metrics
    stacked_pred_interp = np.stack((x_pred_interp, xdot_pred_interp, xdotdot_pred_interp), axis=0)
    stacked_pred_extrap = np.stack((x_pred_extrap, xdot_pred_extrap, xdotdot_pred_extrap), axis=0)
    stacked_fourier_metrics_both = [
        fourier_metrics(stacked_pred_interp, dt, average_dims=True),
        fourier_metrics(stacked_pred_extrap, dt, average_dims=True),
    ]
    # Loop over interpolation/extrapolation and kinematic quantities
    # and store metrics
    for stacked_fourier_metrics, kind in zip(stacked_fourier_metrics_both, kinds):
        for i, name in enumerate(names):
            mean_frequency = stacked_fourier_metrics["mean"][i] / n_periods
            all_metrics[f"fourier_mean_{name}_{kind}"] = mean_frequency

            bandwidth = stacked_fourier_metrics["bandwidth"][i] / n_periods
            all_metrics[f"fourier_bandwidth_{name}_{kind}"] = bandwidth

            all_metrics[f"fourier_entropy_{name}_{kind}"] = stacked_fourier_metrics["entropy"][i]

    # Fourier MSE
    all_metrics["fourier_mse_x_interp"] = fourier_mse(
        x_pred_interp, x_true_interp, dt, average_dims=True
    )
    all_metrics["fourier_mse_xdot_interp"] = fourier_mse(
        xdot_pred_interp, xdot_true_interp, dt, average_dims=True
    )
    all_metrics["fourier_mse_xdotdot_interp"] = fourier_mse(
        xdotdot_pred_interp, xdotdot_true_interp, dt, average_dims=True
    )

    all_metrics["fourier_mse_x_extrap"] = fourier_mse(
        x_pred_extrap, x_true_extrap, dt, average_dims=True
    )
    all_metrics["fourier_mse_xdot_extrap"] = fourier_mse(
        xdot_pred_extrap, xdot_true_extrap, dt, average_dims=True
    )
    all_metrics["fourier_mse_xdotdot_extrap"] = fourier_mse(
        xdotdot_pred_extrap, xdotdot_true_extrap, dt, average_dims=True
    )

    # Metric 3: Energy MSE
    all_metrics["energy_mse_interp"] = oscillator_energy_mse(
        x_pred_interp, xdot_pred_interp, x_true_interp, xdot_true_interp, omega
    )
    all_metrics["energy_mse_extrap"] = oscillator_energy_mse(
        x_pred_extrap, xdot_pred_extrap, x_true_extrap, xdot_true_extrap, omega
    )

    # Metric 4: Domain-based evaluation of ODE
    x_min = np.min(x_pred_interp, axis=(0, 1))
    x_max = np.max(x_pred_interp, axis=(0, 1))
    xdot_min = np.min(xdot_pred_interp, axis=(0, 1))
    xdot_max = np.max(xdot_pred_interp, axis=(0, 1))
    t_min = np.min(t_interp)
    t_max = np.max(t_interp)

    f_model = model.second_derivative
    f_true = oscillator
    all_metrics["ode_domain_mse_interp"] = domain_mse(
        f_true,
        f_model,
        (t_min, t_max),
        (x_min, x_max),
        (xdot_min, xdot_max),
        n_points=N_POINTS_DOMAIN,
    )
    all_metrics["ode_domain_mse_extrap"] = domain_mse(
        f_true,
        f_model,
        (t_min, t_max),
        (x_min, x_max),
        (xdot_min, xdot_max),
        n_points=N_POINTS_DOMAIN,
        extrapolate=DOMAIN_EXTRAPOLATION,
    )

    return all_metrics


def load_and_compute(model_path: pathlib.Path):
    """
    Worker function to load a model and compute metrics.
    """
    model = LanedeAPI.load(str(model_path))
    with open(f"{model_path}/{CONFIG_NAME}.json", "r") as f:
        config = json.load(f)

    metrics = compute_metrics(config, model)

    # Save metrics, update if already exists
    metric_path = f"{model_path}/{METRIC_NAME}.json"
    metric_path_ = pathlib.Path(metric_path)
    metric_path_.parent.mkdir(parents=True, exist_ok=True)

    # If the file already exists, load existing metrics and update them
    if metric_path_.exists():
        with open(metric_path, "r") as f:
            existing_metrics: dict = json.load(f)
        existing_metrics.update(metrics)
        metrics = existing_metrics

    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main():

    base_path = pathlib.Path(DIRECTORY)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {DIRECTORY} does not exist.")

    model_paths = list(base_path.glob(f"{BASE_NAME}*"))

    ctx = mp.get_context("spawn")
    with ctx.Pool(N_JOBS) as pool:
        pool.map(load_and_compute, model_paths)


if __name__ == "__main__":
    main()
