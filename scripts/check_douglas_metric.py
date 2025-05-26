"""
This script investigates the behaviour of `lanede.core.TryLearnDouglas`
when measuring the Helmholtz conditions of harmonic oscillator data.
"""

# DEPRECATED SCRIPT. Use douglas_metric_analytic.py instead.
print(
    "This script is deprecated. Use scripts/douglas_metric_analytic.py instead."
)

# NOTE: This script uses the core package directly and is thus somewhat
#       more low-level. It could be rewritten to use the API by
#       defining a suitable (mock) LagrangianNeuralODEModel, but this
#       is unnecessary here.

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, RAdam
import numpy as np
import matplotlib.pyplot as plt

from lanede.core import TryLearnDouglas, NeuralNetwork, SecondOrderNeuralODE
from lanede.core import make_dataloader, TrainingInfo
from lanede.data.toy import DampedHarmonicOscillator, from_ode, add_noise
from lanede.visualize import plot_g_matrix

# Main settings
# General settings
device_train = "cuda" if torch.cuda.is_available() else "cpu"  # Much faster on GPU
N_TIME_STEPS = 150
noise_sigma = 0.05

# Oscillator settings
n_peaks = 6
t_span = 4
omega = 2 * np.pi / (t_span / n_peaks)
# fmt: off
spring_matrix = np.array([[omega**2, 0],
                          [0, omega**2]])

damping_matrix = np.array([[omega, 0],
                           [0, omega]])
# fmt: on

# Model settings
hidden_layer_sizes = [64] * 3


def _format_training_info(info: dict[str, float]) -> str:
    loss_info = "  |  ".join(f"{k}: {v:.2e}" for k, v in info.items())
    return loss_info


def train_metric(
    metric: TryLearnDouglas,
    f: SecondOrderNeuralODE,
    data: DataLoader,
    n_epochs: int,
    optimizer: Optimizer,
):
    """
    Train the metric on the given data.

    Parameters
    ----------
    metric : TryLearnDouglas
        The metric to train.
    f : SecondOrderNeuralODE
        The ODE to use the metric on.
        Must not have trainable parameters!
    data : DataLoader
        The training data. Must return a tuple of torch.Tensor's of the
        form (t, x, xdot). Shapes: t: (n_steps,),
        x: (n_batch, n_steps, n_dim), xdot: (n_batch, n_steps, n_dim).
    n_epochs : int
        The number of epochs to train the metric.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training. Must be initialized with the
        metric's parameters.
    """
    if len(list(f.parameters())) != 0:
        raise ValueError("The ODE must not have trainable parameters.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=30,
        threshold=1e-2,
    )
    device = metric.device
    f_old_device = f.device
    f = f.to(device)

    f_old = f
    f = f.second_order_function

    info_log = TrainingInfo()
    n_digits = len(str(n_epochs))

    for epoch in range(n_epochs):

        for t, x, xdot in data:
            t = t.to(device)
            x = x.to(device)
            xdot = xdot.to(device)

            t_with_batches = t.repeat(x.shape[0], 1)

            optimizer.zero_grad()

            helmholtz_loss, individual_helmholtz = metric.forward(
                f, t_with_batches, x, xdot, individual_metrics=True
            )
            helmholtz_loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 10)
            optimizer.step()

            info_log.update_step(helmholtz_metric=helmholtz_loss, **individual_helmholtz)

        info_log.end_epoch()

        scheduler.step(info_log.current_info()["helmholtz_metric"])
        loss_info = _format_training_info(info_log.current_info())
        print(f"Epoch {epoch:>{n_digits}}:   {loss_info}")

    f_old.to(f_old_device)
    return info_log.numpy_dict()


class HarmonicOscillatorODE(SecondOrderNeuralODE):
    """
    Like `DampedHarmonicOscillator`, but in pytorch.
    """

    def __init__(self, K: torch.Tensor, C: torch.Tensor):
        """
        Set the parameters of the ODE.

        Parameters
        ----------

        K : torch.Tensor, shape (n_dim, n_dim)
            The spring constant matrix.
        C : torch.Tensor, shape (n_dim, n_dim)
            The damping matrix.
        """
        super().__init__()
        self.register_buffer("_K", K)
        self.register_buffer("_C", C)

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        return -self._matmul(self._K, x) - self._matmul(self._C, xdot)

    @property
    def device(self):
        return self._K.device

    @staticmethod
    def _matmul(matrix, vector):
        # (double) Batched vector times non-batched matrix
        return torch.einsum("ij,abj->abi", matrix, vector)


def main():
    rng = np.random.default_rng()
    n_dim = len(spring_matrix)

    # Data
    oscillator = DampedHarmonicOscillator(spring_matrix, damping_matrix)

    # Train
    t_data = np.linspace(0, t_span, N_TIME_STEPS)
    x_0 = rng.normal(size=(6000, n_dim))
    v_0 = rng.normal(size=(6000, n_dim))
    x_data, xdot_data, _ = from_ode(oscillator, t_data, x_0, v_0)
    x_data = add_noise(x_data, noise_sigma)
    xdot_data = add_noise(xdot_data, noise_sigma)

    # Test (generate seperately instead of splitting)
    t_test = t_data
    x_0_test = rng.normal(size=(1000, n_dim))
    v_0_test = rng.normal(size=(1000, n_dim))
    x_data_test, xdot_data_test, _ = from_ode(oscillator, t_test, x_0_test, v_0_test)
    # Add noise to test data
    x_data_test = add_noise(x_data_test, noise_sigma)
    xdot_data_test = add_noise(xdot_data_test, noise_sigma)

    t_test_with_batches = np.tile(t_test, (x_data_test.shape[0], 1))
    t_test_with_batches = torch.tensor(t_test_with_batches, dtype=torch.float32)
    x_data_test = torch.tensor(x_data_test, dtype=torch.float32)
    xdot_data_test = torch.tensor(xdot_data_test, dtype=torch.float32)

    # Build the model
    neural_net = NeuralNetwork(
        2 * n_dim + 1, hidden_layer_sizes, (n_dim * (n_dim + 1)) // 2, torch.nn.Softplus
    )
    metric = TryLearnDouglas(neural_net, 1.0, 1.0)
    optimizer = RAdam(metric.parameters(), lr=0.003)

    # The ODE function to measure the Helmholtz conditions of
    ode_check = HarmonicOscillatorODE(
        torch.tensor(spring_matrix, dtype=torch.float32),
        torch.tensor(damping_matrix, dtype=torch.float32),
    )

    # Before Training: evaluate the metric on the test data
    helmholtz_metric, individual_helmholtz = metric(
        ode_check.second_order_function,
        t_test_with_batches,
        x_data_test,
        xdot_data_test,
        individual_metrics=True,
    )
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    formatted = _format_training_info(individual_helmholtz)
    print(f"Individual Helmholtz: {formatted}")

    # Train the metric
    data = make_dataloader(t_data, x_data, xdot_data, batch_size=128)
    metric.to(device_train)
    train_metric(metric, ode_check, data, 1000, optimizer)
    metric.to("cpu")

    # After Training: evaluate the metric on the test data
    helmholtz_metric, individual_helmholtz = metric(
        ode_check.second_order_function,
        t_test_with_batches,
        x_data_test,
        xdot_data_test,
        individual_metrics=True,
    )
    print(f"Test Helmholtz loss: {helmholtz_metric:.2e}")
    formatted = _format_training_info(individual_helmholtz)
    print(f"Individual Helmholtz: {formatted}")

    # Plot g components vs. t, x, xdot
    g = metric.evaluate_g(t_test_with_batches, x_data_test, xdot_data_test)
    g = g.detach().cpu().numpy()

    fig, _ = plot_g_matrix(
        g,
        t_test_with_batches.numpy(),
        x_data_test.numpy(),
        xdot_data_test.numpy(),
        n_random=10,
        state_components=(0,),
        derivative_components=(0,),
    )
    # fig.savefig("g_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
