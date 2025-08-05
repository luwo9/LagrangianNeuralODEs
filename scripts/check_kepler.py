"""
This script investigates the behaviour of `lanede.core.TryLearnDouglas`
when measuring the Helmholtz conditions on analytic, fixed ODEs of
the Kepler problem.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from lanede.api import standard_douglas_on_fixed
from lanede.core import Identity, TryLearnDouglas
from lanede.data.toy import (
    KeplerProblem,
    from_ode,
    add_noise,
    KeplerODE,
)
from lanede.visualize import plot_g_matrix

# Main settings
N_DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Much faster on GPU
G_HIDDEN_LAYER_SIZES = [64] * 2
BATCH_SIZE = 128
INIT_LR = 1e-2
N_EPOCHS = 1200
N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 12000
LR_SCHEDULER_OPTIONS = {"factor": 0.5, "patience": 1500, "threshold": 1e-2}
# Whether to compute the analytic g matrix and compare
COMPUTE_ANALYTIC_G = True
# If saving files below, this is used as an identifier
SAVE_NAME = "1p_05e_2o"

# Chose analytic ODE
SEMI_MAJOR_AXIS_INIT = 1.0
N_ORBITS = 2
T_SPAN = 8.0
ode = KeplerODE(semi_major_axis=SEMI_MAJOR_AXIS_INIT, orbital_period=T_SPAN / N_ORBITS)

P_MEAN = 1
P_STD = 0.1
ECCENTRICITY_MEAN = 0.5 * 1
ECCENTRICITY_STD = 0.2 * 1
ECCENTRICITY_MAX = 0.8 * 1

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

    data_ode = KeplerProblem(
        semi_major_axis=SEMI_MAJOR_AXIS_INIT, orbital_period=T_SPAN / N_ORBITS
    )

    n_steps = 500
    t_data = np.linspace(0, T_SPAN, n_steps)
    eccentricities = rng.normal(ECCENTRICITY_MEAN, ECCENTRICITY_STD, size=N_SAMPLES_TRAIN)
    semi_lr = rng.normal(P_MEAN, P_STD, size=N_SAMPLES_TRAIN)
    eccentricities = np.clip(eccentricities, 0, ECCENTRICITY_MAX)
    semi_lr = np.abs(semi_lr)
    phi_0s = np.clip(rng.normal(-1.5, 0.06, size=N_SAMPLES_TRAIN), -1.7, -1.4)
    x_0, xdot_0 = data_ode.get_initial_conditions(
        semi_latus_rectum=semi_lr, eccentricity=eccentricities, phi_0=phi_0s
    )
    x_data, xdot_data, _ = from_ode(data_ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data, component_wise=True)
    xdot_data = add_noise(xdot_data, component_wise=True)

    # Test set
    eccentricities_test = rng.normal(ECCENTRICITY_MEAN, ECCENTRICITY_STD, size=N_SAMPLES_TEST)
    semi_lr_test = rng.normal(P_MEAN, P_STD, size=N_SAMPLES_TEST)
    eccentricities_test = np.clip(eccentricities_test, 0, ECCENTRICITY_MAX)
    semi_lr_test = np.abs(semi_lr_test)
    phi_0s_test = np.clip(rng.normal(-1.5, 0.06, size=N_SAMPLES_TEST), -1.7, -1.4)
    x_0_test, xdot_0_test = data_ode.get_initial_conditions(
        semi_latus_rectum=semi_lr_test,
        eccentricity=eccentricities_test,
        phi_0=phi_0s_test,
    )

    x_data_test, xdot_data_test, _ = from_ode(data_ode, t_data, x_0_test, xdot_0_test)
    x_data_test = add_noise(x_data_test, component_wise=True)
    xdot_data_test = add_noise(xdot_data_test, component_wise=True)

    return (
        t_data,
        x_data,
        xdot_data,
        x_data_test,
        xdot_data_test,
    )


def error_matrix(g_predicted, g_analytic):
    diff = g_predicted - g_analytic
    g_analytic_diag = np.diagonal(g_analytic, axis1=-2, axis2=-1)
    diag_inds = np.diag_indices(2, 2)
    diff[..., diag_inds[0], diag_inds[1]] /= g_analytic_diag
    return np.abs(diff)


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

    print("Before training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")

    # Evaluate the model on the test set, after training
    t_test_with_batches = np.tile(t_data, (x_data_test.shape[0], 1))
    helmholtz_metric, individual_helmholtz = model.helmholtzmetric(
        t_test_with_batches, x_data_test, xdot_data_test, individual_metrics=True
    )
    print("After training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")

    if not COMPUTE_ANALYTIC_G:
        return

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
    # Take g_11_0 (=reduced mass)
    reduced_mass_learned = g[:, 0, 0, 0].reshape(-1, 1)
    r = x_data_test[:, :, 0]
    g_analytic = np.zeros_like(g)
    g_analytic[:, :, 0, 0] = reduced_mass_learned
    g_analytic[:, :, 1, 1] = reduced_mass_learned * r**2

    # Compute errors and plot/print
    error_matrix_ = error_matrix(g, g_analytic)
    # np.save(f"pred_analyt_err_{SAVE_NAME}.npy", np.stack((g, g_analytic, error_matrix_), axis=0))

    print(
        f"Mean error matrix: {error_matrix_.mean():.2e}, "
        f"Median: {np.median(error_matrix_):.2e}, "
        f"90th percentile: {np.percentile(error_matrix_, 90):.2e}, "
        f"95th percentile: {np.percentile(error_matrix_, 95):.2e}, "
        f"80th percentile: {np.percentile(error_matrix_, 80):.2e}, "
        f"20th percentile: {np.percentile(error_matrix_, 20):.2e}, "
    )
    bins = np.logspace(
        np.log10(np.clip(np.min(error_matrix_), 1e-8, None)),
        np.log10(np.max(error_matrix_)),
        50,
    )
    n_points = len(error_matrix_.flatten())
    plt.hist(error_matrix_.flatten(), log=False, bins=bins, weights=np.ones(n_points) / n_points)
    plt.xscale("log")
    plt.xlabel(r"Error matrix $\Delta_g$")
    plt.ylabel("Relative count")
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
        t_component=False,
        # residuals=True,
        state_components=(0,),
        derivative_components=(),
    )
    # fig.savefig(f"g_matrix_{SAVE_NAME}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
