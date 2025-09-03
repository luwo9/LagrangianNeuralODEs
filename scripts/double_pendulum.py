"""
This script trains a Lagrangian Neural ODE on a toy dataset of the
double pendulum.
"""

from pathlib import Path

import numpy as np

from lanede.api import LanedeAPI
from lanede.data.toy import from_ode, add_noise, DoublePendulum
from lanede.visualize import plot_timeseries

# Main settings
# General settings
NAME = "double_pendulum"
N_TIME_STEPS = 30
NOISE_LEVEL_DATA = 0.05

# Pendulum settings
length_1 = 1.0
length_2 = 1.0
mass_1 = 1.0
mass_2 = 1.0
g = 9.81

N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 1000

T_SPAN = 6
ANGLE_MEANS = np.array([np.pi / 6, np.pi / 6])
ANGLE_STDS = np.array([0.2, 0.2])
VELOCITY_MEANS = np.array([0.0, 0.0])
VELOCITY_STDS = np.array([0.2, 0.2])

# Model settings
name = "simple_douglas"
cfg = {
    "dim": 2,
    "explicit_time_dependence_lagrangian": True,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.07,
        "sheduler_patience": 2000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200 * 3,
    },
    "ode": {
        "activation_fn": "Softplus",
        "hidden_layer_sizes": [32] * 1,
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "helmholtz": {
        "hidden_layer_sizes": [100] * 2,
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


# Other settings in the main function


def main():
    rng = np.random.default_rng()
    # Data
    ode = DoublePendulum(
        length_1=length_1,
        length_2=length_2,
        mass_1=mass_1,
        mass_2=mass_2,
        g=g,
    )

    # Train
    t_data = np.linspace(0, T_SPAN, N_TIME_STEPS)
    # t_data = np.concatenate(
    #     (
    #         np.linspace(0, 0.25 * T_SPAN, N_TIME_STEPS - N_TIME_STEPS // 2),
    #         np.linspace(0.75 * T_SPAN, T_SPAN, N_TIME_STEPS // 2),
    #     )
    # )
    x_0 = rng.normal(loc=ANGLE_MEANS, scale=ANGLE_STDS, size=(N_SAMPLES_TRAIN, 2))
    # Make sure they are correlated:
    v_0 = VELOCITY_MEANS + VELOCITY_STDS / ANGLE_STDS * (x_0 - ANGLE_MEANS)
    x_data, *_ = from_ode(ode, t_data, x_0, v_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA)

    # Test (generate seperately instead of splitting)
    t_test = t_data
    x_0_test = rng.normal(loc=ANGLE_MEANS, scale=ANGLE_STDS, size=(N_SAMPLES_TEST, 2))
    xdot_0_test = VELOCITY_MEANS + VELOCITY_STDS / ANGLE_STDS * (x_0_test - ANGLE_MEANS)
    x_data_test, xdot_data_test, xdotdot_data_test = from_ode(ode, t_test, x_0_test, xdot_0_test)
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
    t_plot = np.linspace(0, 4 * T_SPAN, 1000)  # Higher resolution for curves

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
