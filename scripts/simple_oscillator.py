"""
This script trains a Lagrangian Neural ODE on a toy dataset of a
harmonic oscillator.

A few general settings are defined at the beginning of the script.
"""

from pathlib import Path

import numpy as np

from lanede.api import LanedeAPI, EXAMPLES
from lanede.data.toy import from_ode, DampedHarmonicOscillator, add_noise
from lanede.visualize import plot_timeseries

# Main settings
# General settings
NAME = "oscillator_helmholtz"
N_TIME_STEPS = 150
NOISE_LEVEL_DATA = 0.05

# Oscillator settings
n_periods = 6
omega = 2 * np.pi * n_periods
# fmt: off
spring_matrix = np.array([[omega**2, 0],
                          [0, omega**2]])

damping_matrix = np.array([[0.0, 0],
                           [0, 0.0]])
# fmt: on

# Model settings
name = "simple_douglas"
cfg = {
    "dim": 2,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.05,
        "sheduler_patience": 2000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200,
    },
    "ode": {
        "activation_fn": "Softplus",
        "hidden_layer_sizes": [16] * 2,
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "helmholtz": {
        "hidden_layer_sizes": [64] * 2,
        "activation_fn": "Softplus",
        "total_weight": 1,
        "condition_weights": [1.0, 1.0, 1e-6],
    },
    "initial_net": {
        "hidden_layer_sizes": [16] * 3,
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}
# Other settings in the main function


def main():
    rng = np.random.default_rng()
    # Data
    oscillator = DampedHarmonicOscillator(spring_matrix, damping_matrix)

    # Train
    t_data = np.linspace(0, 1, N_TIME_STEPS)
    x_0 = 1 + rng.normal(size=(6000, 2)) / 10
    v_0 = np.sqrt(x_0**2) / 10
    x_data, *_ = from_ode(oscillator, t_data, x_0, v_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA)

    # Test (generate seperately instead of splitting)
    t_test = t_data
    x_0_test = 1 + rng.normal(size=(1000, 2)) / 10
    v_0_test = np.sqrt(x_0_test**2) / 10
    x_data_test, xdot_data_test, xdotdot_data_test = from_ode(
        oscillator, t_test, x_0_test, v_0_test
    )
    # Add noise to test data
    x_data_test = add_noise(x_data_test, NOISE_LEVEL_DATA)
    xdot_data_test = add_noise(xdot_data_test, NOISE_LEVEL_DATA)
    xdotdot_data_test = add_noise(xdotdot_data_test, NOISE_LEVEL_DATA)

    # Train and save
    model = LanedeAPI(name, cfg)
    model.train(t_data, x_data, n_epochs=600, batch_size=128, device="cpu")
    path = f"saves/{NAME}"
    model.save(path)

    # Evaluate and plot test data
    t_plot = np.linspace(0, 1, 1000)  # Higher resolution for curves

    # Get predicted time series with higher resolution
    x_pred, v_pred = model.predict(t_plot, x_data_test[:, 0, :])
    t_plot_with_batches = np.tile(t_plot, (x_pred.shape[0], 1))
    a_pred = model.second_derivative(t_plot_with_batches, x_pred, v_pred)

    # Get true time series with higher resolution
    x_true, xdot_true, xdotdot_true = from_ode(oscillator, t_plot, x_0_test, v_0_test)

    # Plot
    pred = (t_plot, x_pred, v_pred, a_pred)
    data = (t_data, x_data_test, xdot_data_test, xdotdot_data_test)
    true = (t_plot, x_true, xdot_true, xdotdot_true)

    fig, _ = plot_timeseries(predictions=pred, data=data, true=true, n_random=10)

    plot_path = f"saves/{NAME}/plots"
    # Stay with strings but use pathlib to create directory
    Path(plot_path).mkdir(exist_ok=True)
    fig.savefig(f"{plot_path}/timeseries.png", dpi=600)


if __name__ == "__main__":
    main()
