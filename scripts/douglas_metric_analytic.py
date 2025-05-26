"""
This script investigates the behaviour of `lanede.core.TryLearnDouglas`
when measuring the Helmholtz conditions on analytic, fixed ODEs.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from lanede.api import standard_douglas_on_fixed
from lanede.core import Identity
from lanede.data.toy import (
    NonExtremalCaseIIIb,
    HarmonicOscillatorODE,
    from_ode,
    add_noise,
    CaseIIIbODE,
    DampedHarmonicOscillator,
)

# Main settings
N_DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Much faster on GPU
G_HIDDEN_LAYER_SIZES = [64] * 3
BATCH_SIZE = 128
INIT_LR = 1e-3
N_EPOCHS = 1000
N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 1000
LR_SCHEDULER_OPTIONS = {"factor": 0.5, "patience": 1500, "threshold": 1e-2}


# Chose analytic ODE
ode = CaseIIIbODE()

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

    data_ode = NonExtremalCaseIIIb()

    n_steps = 150
    t_data = np.linspace(0, 1, n_steps)
    x_0 = rng.normal(size=(N_SAMPLES_TRAIN, N_DIM))
    xdot_0 = rng.normal(size=(N_SAMPLES_TRAIN, N_DIM))
    x_data, xdot_data, _ = from_ode(data_ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data)
    xdot_data = add_noise(xdot_data)

    # Test set
    x_0_test = rng.normal(size=(N_SAMPLES_TEST, N_DIM))
    xdot_0_test = rng.normal(size=(N_SAMPLES_TEST, N_DIM))
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
    print("Before training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")
    print("Begin training")

    # Train the model
    info = model.train(
        t_data,
        x_data,
        xdot_data,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device=DEVICE,
    )

    # Evaluate the model on the test set, after training
    t_test_with_batches = np.tile(t_data, (x_data_test.shape[0], 1))
    helmholtz_metric, individual_helmholtz = model.helmholtzmetric(
        t_test_with_batches, x_data_test, xdot_data_test, individual_metrics=True
    )
    print("After training:")
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    print(f"Individual Helmholtz: {individual_helmholtz}")


if __name__ == "__main__":
    main()
