"""
This script investigates the behaviour of `lanede.core.TryLearnDouglas`
when measuring the Helmholtz conditions on analytic, fixed ODEs of
the harmonic oscillator.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from lanede.api import standard_douglas_on_fixed
from lanede.core import Identity, TryLearnDouglas
from lanede.data.toy import (
    DampedHarmonicOscillator,
    from_ode,
    add_noise,
    HarmonicOscillatorODE,
)
from lanede.visualize import plot_g_matrix

# Main settings
N_DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Much faster on GPU
G_HIDDEN_LAYER_SIZES = [64] * 2
BATCH_SIZE = 128
INIT_LR = 1e-3
N_EPOCHS = 1000
N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 12000
LR_SCHEDULER_OPTIONS = {"factor": 0.5, "patience": 1500, "threshold": 1e-2}
# If saving files below, this is used as an identifier
SAVE_NAME = "3p_05d"


# Oscillator settings
N_PEAKS = 3
T_SPAN = 4  # Keep acceleration values low without normalizer
OMEGA = 2 * np.pi / (T_SPAN / N_PEAKS)
# fmt: off
SPRING_MATRIX = np.array([[(OMEGA*0.8)**2, 0],
                          [0, OMEGA**2]])

DAMPING_MATRIX = np.array([[OMEGA*0.8, 0],
                           [0, OMEGA]])*0.5
# fmt: on
ode = HarmonicOscillatorODE(
    torch.tensor(SPRING_MATRIX, dtype=torch.float32),
    torch.tensor(DAMPING_MATRIX, dtype=torch.float32),
)

# Nomralizer must be identity, otherwise the analytic ODE will not be
# correctly represented.
normalizer = Identity()
normalizer.fit(1, 2)  # Some dummy values


# Create the dataset
# This must fit the ODE used above.


def make_data():
    """
    Generate the data.

    Returns
    -------

    np.ndarray
        The time data.
    np.ndarray
        The x training data.
    np.ndarray
        The xdot training data.
    np.ndarray
        The x test data.
    np.ndarray
        The xdot test data.
    """
    rng = np.random.default_rng()

    data_ode = DampedHarmonicOscillator(SPRING_MATRIX, DAMPING_MATRIX)

    n_steps = 150
    t_data = np.linspace(0, T_SPAN, n_steps)
    x_0 = 1 + rng.normal(size=(N_SAMPLES_TRAIN, N_DIM))
    xdot_0 = rng.normal(size=(N_SAMPLES_TRAIN, N_DIM))
    x_data, xdot_data, _ = from_ode(data_ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data)
    xdot_data = add_noise(xdot_data)

    # Test set
    x_0_test = rng.normal(size=(N_SAMPLES_TEST, N_DIM))
    xdot_0_test = 1 + rng.normal(size=(N_SAMPLES_TEST, N_DIM))
    x_data_test, xdot_data_test, _ = from_ode(data_ode, t_data, x_0_test, xdot_0_test)
    x_data_test = add_noise(x_data_test)
    xdot_data_test = add_noise(xdot_data_test)

    return (
        t_data,
        x_data,
        xdot_data,
        x_data_test,
        xdot_data_test,
    )


# Main logic


def main():
    # Generate the data
    t_data, x_data, xdot_data, x_data_test, xdot_data_test = make_data()

    # Create the fixed ODE model
    model = standard_douglas_on_fixed(
        ode,
        normalizer,
        N_DIM,
        G_HIDDEN_LAYER_SIZES,
        init_lr=INIT_LR,
        scheduler_options=LR_SCHEDULER_OPTIONS,
        ode_options={"rtol": 1e-6, "atol": 1e-6, "use_adjoint": False},  # Not really needed here
    )

    # Evaluate the model on the test set before training
    t_test_with_batches = np.tile(t_data, (x_data_test.shape[0], 1))
    helmholtz_metric, individual_helmholtz = model.helmholtzmetric(
        t_test_with_batches, x_data_test, xdot_data_test, individual_metrics=True
    )

    # Train the model
    info = model.train(
        t_data,
        x_data,
        xdot_data,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device=DEVICE,
    )
    # np.save(f"hloss_{SAVE_NAME}.npy", info["helmholtz"])
    # for key, value in info.items():
    #     if key != "helmholtz" and key != "error":
    #         np.save(f"{key}loss_{SAVE_NAME}.npy", value)

    print("Before training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")
    print("Begin training")

    # Evaluate the model on the test set, after training
    t_test_with_batches = np.tile(t_data, (x_data_test.shape[0], 1))
    helmholtz_metric, individual_helmholtz = model.helmholtzmetric(
        t_test_with_batches, x_data_test, xdot_data_test, individual_metrics=True
    )
    print("After training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")

    # Evaluate g matrix vs. analytic g matrix
    metric: TryLearnDouglas = model._model._helmholtz_metric  # TODO: Expose this in the API

    # Predict g matrix
    g = metric.evaluate_g(
        torch.tensor(t_test_with_batches, dtype=torch.float32),
        torch.tensor(x_data_test, dtype=torch.float32),
        torch.tensor(xdot_data_test, dtype=torch.float32),
    )
    g = g.detach().cpu().numpy()

    # Compute analytical g matrix
    # NOTE: This only works for (2D) diagonal damping and spring matrices:
    damp_1 = DAMPING_MATRIX[0, 0]
    damp_2 = DAMPING_MATRIX[1, 1]
    # fmt: off
    exponent_matrix = np.array([[damp_1, 0.5 * (damp_1 + damp_2)],
                                [0.5 * (damp_1 + damp_2), damp_2]])
    # fmt: on
    exponent_analytic = np.einsum("ij,ab->abij", exponent_matrix, t_test_with_batches)

    # Use learned initial condition for analytic g matrix
    # NOTE: If there just happens to be a bigger error on it, it may
    # not be a good representation.
    g_0_learned = g[:, 0].copy()
    # See if the damped frequencies are equal, if so, set off-diagonal
    # elements g_s to zero (from first Helmholtz condition). If they
    # are not equal, g_s could still be zero, but don't enforce it.
    omega_sq_1 = SPRING_MATRIX[0, 0]
    omega_sq_2 = SPRING_MATRIX[1, 1]
    damped_omega_sq_1 = omega_sq_1 - damp_1**2 * 0.25
    damped_omega_sq_2 = omega_sq_2 - damp_2**2 * 0.25
    damped_omega_equal = np.isclose(damped_omega_sq_1, damped_omega_sq_2)  # Not always safe!
    if not damped_omega_equal:
        print("Detecting different damped frequencies, enforcing 0 on off-diagonal elements.")
    g_0_learned[:, 1, 0] *= damped_omega_equal
    g_0_learned[:, 0, 1] *= damped_omega_equal

    g_analytic = np.einsum("aij,abij->abij", g_0_learned, np.exp(exponent_analytic))

    # Compute errors and plot/print
    normalized_errors = np.abs(g_analytic - g) / np.mean(np.abs(g_analytic))
    # np.save(f"pred_analyt_err_{SAVE_NAME}.npy", np.stack((g, g_analytic, normalized_errors), axis=0))

    print(
        f"Error results: Mean: {normalized_errors.mean():.2e}, "
        f"Error results: Median: {np.median(normalized_errors):.2e}, "
        f"90th percentile: {np.percentile(normalized_errors, 90):.2e}, "
        f"95th percentile: {np.percentile(normalized_errors, 95):.2e}, "
        f"80th percentile: {np.percentile(normalized_errors, 80):.2e}, "
        f"20th percentile: {np.percentile(normalized_errors, 20):.2e}, "
    )
    bins = np.logspace(
        np.log10(np.clip(np.min(normalized_errors), 1e-8, None)),
        np.log10(np.max(normalized_errors)),
        50,
    )
    plt.hist(normalized_errors.flatten(), log=True, bins=bins)
    plt.xscale("log")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    # plt.savefig(f"error_histogram_{SAVE_NAME}.pdf")
    plt.show()

    # Plot g matrix with analytic result
    fig, _ = plot_g_matrix(
        g,
        t_test_with_batches,
        x_data_test,
        xdot_data_test,
        g_analytic=g_analytic,
        n_random=10,
        # residuals=True,
        state_components=(),
        derivative_components=(),
    )
    # fig.savefig(f"g_matrix_{SAVE_NAME}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
