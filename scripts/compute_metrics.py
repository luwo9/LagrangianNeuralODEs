"""
Load models, compute performance metrics and optionally compare to
reference models.
"""

from collections import defaultdict
import json

import numpy as np

from lanede.api import LanedeAPI
from lanede.data.toy import from_ode, DampedHarmonicOscillator, add_noise
from lanede.metrics import fourier_mse, domain_mse


# Model names
# List of models to compute metrics for. They are averaged with a
# geometric mean.
MODEL_NAMES: list[str] = []
# If a list of reference models is given compute their metrics and
# yield the ratio of the metrics of the main models to the reference
# models.
REFERENCE_MODEL_NAMES: list[str] = []
# Set to None to only print and not save metrics:
METRIC_OUTPUT_FILE = "model_metrics.json"
# Extrapolation factor for extrapolation metrics.
EXTRAPOLATION_FACTOR = 2


# Define how to generate data to evaluate the model:
# Must return (x_data_test_0, t_true, x_true, xdot_true, xdotdot_true,
# ode)
# Where x_data_test_0 refers to the initial values for x of test set
# data points ans _true refers to a high resolution time series for
# computing the metrics. ode is the true second order ODE function.
# t_true must be evenly spaced, of sufficiently high resolution and
# cover the full time span.
def make_data_and_ode():
    # Oscillator settings
    n_periods = 1
    omega = 2 * np.pi * n_periods
    # fmt: off
    spring_matrix = np.array([[omega**2, 0],
                            [0, omega**2]])

    damping_matrix = np.array([[0.0, 0],
                            [0, 0.0]])
    # fmt: on
    n_time_steps = 7

    # Make test data
    rng = np.random.default_rng()
    ode = DampedHarmonicOscillator(spring_matrix, damping_matrix)
    t_test = np.linspace(0, 1, n_time_steps)
    x_0_test = 1 + rng.normal(size=(6000, 2)) / 10
    xdot_0_test = np.sqrt(x_0_test**2) / 10

    # Predict full test time series data points, eventhough it is not
    # needed directly. This is done as x_data_test_0 here uses the
    # noisy initial values, where the size of added noise depends on
    # the (mean abs value of) the full trajectory
    x_data_test, *_ = from_ode(ode, t_test, x_0_test, xdot_0_test)
    noise_level = 0.05
    x_data_test_0 = add_noise(x_data_test, noise_level)[:, 0, :]

    # Get a high resolution true time series for computing the metrics
    t_true = np.linspace(0, 1.0 * EXTRAPOLATION_FACTOR, 1000)
    x_true, xdot_true, xdotdot_true = from_ode(ode, t_true, x_0_test, xdot_0_test)
    return (
        x_data_test_0,
        t_true,
        x_true,
        xdot_true,
        xdotdot_true,
        ode,
    )


# Main logic of the script:
def compute_metrics_for_model(model_name: str):
    (
        x_data_test_0,
        t_true,
        x_true,
        xdot_true,
        xdotdot_true,
        ode,
    ) = make_data_and_ode()

    model = LanedeAPI.load(f"saves/{model_name}")

    # Predict the modeled system evolution
    x_pred, xdot_pred = model.predict(t_true, x_data_test_0)
    t_true_with_batches = np.tile(t_true, (x_pred.shape[0], 1))
    xdotdot_pred = model.second_derivative(t_true_with_batches, x_pred, xdot_pred)

    # Metrics
    t_interp_max = t_true[-1] / EXTRAPOLATION_FACTOR
    is_extrap = t_true > t_interp_max
    t_interp = t_true[~is_extrap]
    x_pred_interp = x_pred[:, ~is_extrap]
    xdot_pred_interp = xdot_pred[:, ~is_extrap]
    xdotdot_pred_interp = xdotdot_pred[:, ~is_extrap]
    x_true_interp = x_true[:, ~is_extrap]
    xdot_true_interp = xdot_true[:, ~is_extrap]
    xdotdot_true_interp = xdotdot_true[:, ~is_extrap]

    t_extrap = t_true[is_extrap]
    x_pred_extrap = x_pred[:, is_extrap]
    xdot_pred_extrap = xdot_pred[:, is_extrap]
    xdotdot_pred_extrap = xdotdot_pred[:, is_extrap]
    x_true_extrap = x_true[:, is_extrap]
    xdot_true_extrap = xdot_true[:, is_extrap]
    xdotdot_true_extrap = xdotdot_true[:, is_extrap]

    all_metrics = {}
    all_metrics["mse_x_interp"] = np.mean((x_pred_interp - x_true_interp) ** 2)
    all_metrics["mse_xdot_interp"] = np.mean((xdot_pred_interp - xdot_true_interp) ** 2)
    all_metrics["mse_xdotdot_interp"] = np.mean((xdotdot_pred_interp - xdotdot_true_interp) ** 2)

    all_metrics["mse_x_extrap"] = np.mean((x_pred_extrap - x_true_extrap) ** 2)
    all_metrics["mse_xdot_extrap"] = np.mean((xdot_pred_extrap - xdot_true_extrap) ** 2)
    all_metrics["mse_xdotdot_extrap"] = np.mean((xdotdot_pred_extrap - xdotdot_true_extrap) ** 2)

    dt = t_true[1] - t_true[0]
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

    x_min = np.min(x_pred_interp, axis=(0, 1))
    x_max = np.max(x_pred_interp, axis=(0, 1))
    xdot_min = np.min(xdot_pred_interp, axis=(0, 1))
    xdot_max = np.max(xdot_pred_interp, axis=(0, 1))
    t_min = np.min(t_interp)
    t_max = np.max(t_interp)

    f_model = model.second_derivative
    f_true = ode
    n_points_domain = 10**7
    all_metrics["ode_domain_mse_interp"] = domain_mse(
        f_true,
        f_model,
        (t_min, t_max),
        (x_min, x_max),
        (xdot_min, xdot_max),
        n_points=n_points_domain,
    )
    all_metrics["ode_domain_mse_extrap"] = domain_mse(
        f_true,
        f_model,
        (t_min, t_max),
        (x_min, x_max),
        (xdot_min, xdot_max),
        n_points=n_points_domain,
        extrapolate=EXTRAPOLATION_FACTOR,
    )
    return all_metrics


def geometric_mean(array):
    return np.exp(np.mean(np.log(array)))


def get_all_metrics():
    # Evaluate metrics for all models
    metrics_models = defaultdict(list)
    metrics_reference_models = defaultdict(list)
    models = [MODEL_NAMES, REFERENCE_MODEL_NAMES]
    metric_dicts = [metrics_models, metrics_reference_models]
    for model_group, metric_dict in zip(models, metric_dicts):
        for model_name in model_group:
            print(f"Processing model {model_name}...")
            metrics = compute_metrics_for_model(model_name)
            for k, v in metrics.items():
                metric_dict[k].append(v)

    if len(REFERENCE_MODEL_NAMES) == 0:
        return metrics_models

    # Compute metric ratios if reference models are given
    metric_ratios = {}
    for k in metrics_models.keys():
        model_values = np.array(metrics_models[k])
        reference_values = np.array(metrics_reference_models[k])
        # Average metrics first, to allow different lengths of model
        # and reference groups. Mathematically, the ratio of geometric
        # means is equal to the geometric mean of the ratios.
        model_means = geometric_mean(model_values)
        reference_means = geometric_mean(reference_values)
        ratios_mean = model_means / reference_means
        metric_ratios[k] = float(ratios_mean)

    metric_ratios_sorted = dict(sorted(metric_ratios.items(), key=lambda x: x[1]))
    return metric_ratios_sorted


def main():
    metrics = get_all_metrics()
    for k, v in metrics.items():
        print(f"{k}: {v:.2g}")
    print(f"Overall: {geometric_mean(list(metrics.values())):.2g}")
    if METRIC_OUTPUT_FILE is None:
        return

    with open(METRIC_OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
