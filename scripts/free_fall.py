"""
This script trains a Lagrangian Neural ODE on a toy dataset of a
1D free fall.

A few general settings are defined at the beginning of the script.
"""

from pathlib import Path

import numpy as np

from lanede.api import LanedeAPI, EXAMPLES
from lanede.data.toy import from_ode, add_noise, FreeFallWithDrag
from lanede.visualize import plot_timeseries

# Main settings
# General settings
NAME = "ff"
N_TIME_STEPS = 20
NOISE_LEVEL_DATA = 1e-3  # Data has small std/mean ratio

# Free fall settings
g = 9.81
TERMINAL_VELOCITY = 50.0

TIME_SPAN = 2 * TERMINAL_VELOCITY / g

HEIGHT_MEAN = 2000.0
HEIGHT_STD = 200.0
VELOCITY_MEAN = 0.0
VELOCITY_STD = 10.0

# Model settings
name = "simple_douglas"
cfg = {
    "dim": 1,
    "explicit_time_dependence_lagrangian": True,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.07,
        "sheduler_patience": 2000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200 // 2,
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
        "hidden_layer_sizes": [16] * 2,
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
    ode = FreeFallWithDrag(TERMINAL_VELOCITY, g)

    # Train
    t_data = np.linspace(0, TIME_SPAN, N_TIME_STEPS)
    # t_data = np.concatenate((np.linspace(0, 0.25, 5), np.linspace(0.75, 1, 5)))
    # t_data = np.array([0., 0.12, 0.154, 0.22, 0.31, 0.35, 0.68, 0.73, 0.8, 1.0])
    x_0 = rng.normal(HEIGHT_MEAN, HEIGHT_STD, size=(6000, 1))
    xdot_0 = VELOCITY_STD / HEIGHT_STD * (x_0 - HEIGHT_MEAN) + VELOCITY_MEAN
    x_data, *_ = from_ode(ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA)

    # Test (generate seperately instead of splitting)
    t_test = t_data
    x_0_test = rng.normal(HEIGHT_MEAN, HEIGHT_STD, size=(1000, 1))
    xdot_0_test = VELOCITY_STD / HEIGHT_STD * (x_0_test - HEIGHT_MEAN) + VELOCITY_MEAN
    x_data_test, xdot_data_test, xdotdot_data_test = from_ode(ode, t_test, x_0_test, xdot_0_test)
    # Add noise to test data
    x_data_test = add_noise(x_data_test, NOISE_LEVEL_DATA)
    xdot_data_test = add_noise(xdot_data_test, NOISE_LEVEL_DATA)
    xdotdot_data_test = add_noise(xdotdot_data_test, NOISE_LEVEL_DATA)

    # Train and save
    model = LanedeAPI(name, cfg)
    model.train(t_data, x_data, n_epochs=200, batch_size=128, device="cpu")
    path = f"saves/{NAME}"
    model.save(path)

    # Evaluate and plot test data
    t_plot = np.linspace(0, 2 * TIME_SPAN, 1000)  # Higher resolution for curves

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

    plot_path = f"saves/{NAME}/plots"
    # Stay with strings but use pathlib to create directory
    Path(plot_path).mkdir(exist_ok=True)
    fig.savefig(f"{plot_path}/timeseries.png", dpi=600)


if __name__ == "__main__":
    main()
