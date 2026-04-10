"""
Load a model and plot predicted and true time series.
"""

import numpy as np
import matplotlib.pyplot as plt

from lanede.api import LanedeAPI
from lanede.data.toy import from_ode, DampedHarmonicOscillator, add_noise
from lanede.visualize import plot_timeseries


# Model name
# Names of models to plot (should be in saves/)
MODEL_NAMES: list[str] = []
# Plots are saved as f"{PLOT_PREFIX}{model_name}.{SAVE_AS}":
# E.g. "pdf" or "png"; None for saving instead of showing:
SAVE_AS = "pdf"
PLOT_PREFIX = "timeseries_"

EXTRAPOLATION_FACTOR = 2
N_TRAJECTORIES_PLOT = 2
# If True, plot test set values also for derivatives:
PLOT_DATA_DERIVATIVES = False


# Define how to generate data to evaluate the model:
# Must return (t_test, x_data_test, xdot_data_test, xdotdot_data_test,
# t_true, x_true, xdot_true, xdotdot_true). Here _test refers to the
# test data points, while _true refers to a high resolution time series
# for plotting the trajectory as a line.
def make_data():
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
    # Make sure there are enough trajectories such that add_noise gets
    # an accurate estimate of mean(abs(x_data_test))
    n_trajectories = max(1000, N_TRAJECTORIES_PLOT)
    rng = np.random.default_rng()
    ode = DampedHarmonicOscillator(spring_matrix, damping_matrix)
    t_test = np.linspace(0, 1, n_time_steps)
    x_0_test = 1 + rng.normal(size=(n_trajectories, 2)) / 10
    xdot_0_test = np.sqrt(x_0_test**2) / 10
    x_data_test, xdot_data_test, xdotdot_data_test = from_ode(ode, t_test, x_0_test, xdot_0_test)
    # Add noise to test data
    noise_level = 0.05
    x_data_test = add_noise(x_data_test, noise_level)
    xdot_data_test = add_noise(xdot_data_test, noise_level)
    xdotdot_data_test = add_noise(xdotdot_data_test, noise_level)

    # Get a high resolution true time series for plotting and metrics
    t_true = np.linspace(0, 1.0 * EXTRAPOLATION_FACTOR, 1000)
    x_true, xdot_true, xdotdot_true = from_ode(ode, t_true, x_0_test, xdot_0_test)
    return (
        t_test,
        x_data_test,
        xdot_data_test,
        xdotdot_data_test,
        t_true,
        x_true,
        xdot_true,
        xdotdot_true,
    )


# Main logic of the script:
def plot_model(model_name: str):
    (
        t_test,
        x_data_test,
        xdot_data_test,
        xdotdot_data_test,
        t_true,
        x_true,
        xdot_true,
        xdotdot_true,
    ) = make_data()

    model = LanedeAPI.load(f"saves/{model_name}")

    # Predict the modeled system evolution
    x_pred, xdot_pred = model.predict(t_true, x_data_test[:, 0, :])
    t_true_with_batches = np.tile(t_true, (x_pred.shape[0], 1))
    xdotdot_pred = model.second_derivative(t_true_with_batches, x_pred, xdot_pred)

    # Plot
    if not PLOT_DATA_DERIVATIVES:
        xdot_data_test = None
        xdotdot_data_test = None

    pred = (t_true, x_pred, xdot_pred, xdotdot_pred)
    data = (t_test, x_data_test, xdot_data_test, xdotdot_data_test)
    true = (t_true, x_true, xdot_true, xdotdot_true)

    fig, _ = plot_timeseries(predictions=pred, data=data, true=true, n_random=N_TRAJECTORIES_PLOT)

    if SAVE_AS is None:
        plt.show()
        return

    fig.savefig(f"{PLOT_PREFIX}{model_name}.{SAVE_AS}")


def main():
    for model_name in MODEL_NAMES:
        plot_model(model_name)


if __name__ == "__main__":
    main()
