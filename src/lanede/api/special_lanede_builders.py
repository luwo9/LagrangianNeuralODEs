"""
This module contains logic for creating predefined
`LagrangianNeuralODE`'s that cannot be created via the API from simple
configuration dictionaries.

This may be the case, e.g., for non-JSON-compatible configurations.
"""

from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lanede.core import (
    NeuralNetwork,
    SecondOrderNeuralODE,
    SolvedSecondOrderNeuralODE,
    TryLearnDouglas,
    DouglasOnFixedODE,
    Normalizer,
    LagrangianNeuralODE,
)


# This can not be created via the API, as, e.g., the ODE might be
# a (pre-trained) neural ODE, which is not JSON-compatible as a
# configuration.
def standard_douglas_on_fixed(
    ode: SecondOrderNeuralODE,
    normalizer: Normalizer,
    n_dim: int,
    hidden_layer_sizes: list[int],
    activation_fn: type[nn.Module] = torch.nn.Softplus,
    init_lr: float = 1e-3,
    scheduler_options: dict[str, Any] = None,
    metric_options: dict[str, Any] = None,
    ode_options: dict[str, Any] = None,
) -> LagrangianNeuralODE:
    """
    Create a `LagrangianNeuralODE` that, using a `DouglasOnFixedODE`,
    learns the `TryLearnDouglas` metric on a fixed
    `SecondOrderNeuralODE`.

    Parameters
    ----------
    ode : SecondOrderNeuralODE
        The ODE to use. It is kept fixed, see `DouglasOnFixedODE`.
    normalizer : Normalizer
        The normalizer to use. Must be compatible with the
        `SecondOrderNeuralODE` (see notes).
    n_dim : int
        The number of dimensions of the ODE/data.
    hidden_layer_sizes : list[int]
        The sizes of the hidden layers of the metric neural network.
    activation_fn : str, optional
        The activation function to use in the metric neural network.
        Defaults to "Softplus".
    init_lr : float, default=1e-3
        The initial learning rate for the RAdam optimizer. Defaults to 1e-3.
    scheduler_options : dict[str, Any], optional
        Options for the learning rate scheduler. If not provided, a
        `ReduceLROnPlateau` scheduler with default parameters is used.
        (You likely wnat to change the parameters, see
        `DouglasOnFixedODE`.)
    metric_options : dict[str, Any], optional
        Options for the `TryLearnDouglas` metric.
    ode_options : dict[str, Any], optional
        Options for the `SolvedSecondOrderNeuralODE`.

    Returns
    -------
    LagrangianNeuralODE
        The constructed Lagrangian Neural ODE.

    Notes
    -----

    The `normalizer` must be compatible with the
    `SecondOrderNeuralODE`. That is, for an analytic ODE, (probably)
    the Identity and for a neural ODE, the normalizer used when
    training it. Otherwise the ODE is simply not feasible to describe
    the data.
    """
    if scheduler_options is None:
        scheduler_options = {}
    if metric_options is None:
        metric_options = {}
    if ode_options is None:
        ode_options = {}

    neural_ode = SolvedSecondOrderNeuralODE(ode, **ode_options)
    neural_net = NeuralNetwork(
        2 * n_dim + 1, hidden_layer_sizes, (n_dim * (n_dim + 1)) // 2, activation_fn=activation_fn
    )
    metric = TryLearnDouglas(neural_net, **metric_options)
    optimizer = torch.optim.RAdam(metric.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_options)
    fixed_ode_model = DouglasOnFixedODE(neural_ode, metric, optimizer, scheduler)
    model = LagrangianNeuralODE(fixed_ode_model, normalizer, normalizer_prefitted=True)

    return model
