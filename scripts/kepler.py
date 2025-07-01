"""
This script trains a Lagrangian Neural ODE on a toy dataset of the
Kepler problem.

A few general settings are defined at the beginning of the script.
"""

from pathlib import Path

import numpy as np

from lanede.api import LanedeAPI
from lanede.data.toy import KeplerProblem, from_ode, add_noise
from lanede.visualize import plot_timeseries

# Main settings
# General settings
NAME = "kepler"
N_TIME_STEPS = 30
NOISE_LEVEL_DATA = 0.01

# Data settings
N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 1000
P_MEAN = 1
P_STD = 0.1
ECCENTRICITY_MEAN = 0.05
ECCENTRICITY_STD = 0.02
ECCENTRICITY_MAX = 0.1

# Model settings
name = "simple_douglas"
cfg = {
    "dim": 2,
    "explicit_time_dependence_lagrangian": True,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.07,
        "sheduler_patience": 4000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200,
    },
    "ode": {
        "activation_fn": "Softplus",
        "hidden_layer_sizes": [16] * 1,
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "helmholtz": {
        "hidden_layer_sizes": [64] * 2,
        "activation_fn": "Softplus",
        "total_weight": 1.0,
        "condition_weights": [1.0, 1.0],
    },
    "initial_net": {
        "hidden_layer_sizes": [16] * 3,
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}


def main():
    rng = np.random.default_rng()
    # Data
    ode = KeplerProblem(P_MEAN)

    # Train
    t_data = np.linspace(0, 2, N_TIME_STEPS)
    eccentricities = rng.normal(ECCENTRICITY_MEAN, ECCENTRICITY_STD, size=N_SAMPLES_TRAIN)
    # Make sure semi-latus rectum and eccentricity are correlated.
    # This implies that the initial conditions for velocity depend on
    # those of the position.
    semi_lr = P_STD / ECCENTRICITY_STD * (eccentricities - ECCENTRICITY_MEAN) + P_MEAN
    eccentricities = np.clip(eccentricities, 0, ECCENTRICITY_MAX)
    semi_lr = np.abs(semi_lr)
    phi_0s = -np.clip(rng.normal(-1.5, 0.06, size=N_SAMPLES_TRAIN), -1.7, -1.4)

    x_0, xdot_0 = ode.get_initial_conditions(
        semi_latus_rectum=semi_lr, eccentricity=eccentricities, phi_0=phi_0s
    )

    x_data, *_ = from_ode(ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA, component_wise=True)

    # Test
    t_test = t_data
    eccentricities_test = rng.normal(ECCENTRICITY_MEAN, ECCENTRICITY_STD, size=N_SAMPLES_TEST)
    # Make sure semi-latus rectum and eccentricity are correlated:
    semi_lr_test = P_STD / ECCENTRICITY_STD * (eccentricities_test - ECCENTRICITY_MEAN) + P_MEAN
    eccentricities_test = np.clip(eccentricities_test, 0, ECCENTRICITY_MAX)
    semi_lr_test = np.abs(semi_lr_test)
    phi_0s_test = -np.clip(rng.normal(-1.5, 0.06, size=N_SAMPLES_TEST), -1.7, -1.4)

    x_0_test, xdot_0_test = ode.get_initial_conditions(
        semi_latus_rectum=semi_lr_test, eccentricity=eccentricities_test, phi_0=phi_0s_test
    )
    x_data_test, xdot_data_test, xdotdot_data_test = from_ode(ode, t_test, x_0_test, xdot_0_test)

    # Add noise to test data
    x_data_test = add_noise(x_data_test, NOISE_LEVEL_DATA, component_wise=True)
    xdot_data_test = add_noise(xdot_data_test, NOISE_LEVEL_DATA, component_wise=True)
    xdotdot_data_test = add_noise(xdotdot_data_test, NOISE_LEVEL_DATA, component_wise=True)

    # Train model
    model = LanedeAPI(name, cfg)
    model.train(t_data, x_data, n_epochs=500, batch_size=128, device="cpu")
    path = f"saves/{NAME}"
    model.save(path)

    # Evaluate and plot test data
    t_plot = np.linspace(0, 4, 1000)  # Higher resolution for curves

    # Get predicted time series with higher resolution
    x_pred, xdot_pred = model.predict(t_plot, x_data_test[:, 0, :])
    t_plot_with_batches = np.tile(t_plot, (x_pred.shape[0], 1))
    xdotdot_pred = model.second_derivative(t_plot_with_batches, x_pred, xdot_pred)

    # Get true time series with higher resolution
    x_true, xdot_true, xdotdot_true = from_ode(ode, t_plot, x_0_test, xdot_0_test)

    # Plot
    pred = (t_plot, x_pred, xdot_pred, xdotdot_pred)
    data = (t_data, x_data_test, xdot_data_test, xdotdot_data_test)
    true = (t_plot, x_true, xdot_true, xdotdot_true)

    fig, _ = plot_timeseries(predictions=pred, data=data, true=true, n_random=10)

    plot_path = Path(f"saves/{NAME}/plots")
    Path(plot_path).mkdir(exist_ok=True)
    fig.savefig(f"{plot_path}/timeseries.png", dpi=600)


if __name__ == "__main__":
    main()
