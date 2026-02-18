"""
This module contains logic for creating predefined
`LagrangianNeuralODE`'s from simple configuration dictionaries.
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lanede.core import (
    NeuralNetwork,
    LNNNeuralNetwork,
    FreeSecondOrderNeuralODE,
    EulerLagrangeNeuralODE,
    SolvedSecondOrderNeuralODE,
    TryLearnDouglas,
    DummyHelmholtzMetric,
    SimultaneousLearnedMetricOnlyX,
    LagrangianNeuralODE,
    SigmoidTemporalScheduler,
    MeanStd,
)

# NOTE: Every function listed here should have signature
# `def func_name(config: JSONDict) -> LagrangianNeuralODE:`
# For every function, there should be a corresponding example
# configuration dictionary named `example_<func_name>`.


JSONPrimitive = (
    dict[str, "JSONPrimitive"] | list["JSONPrimitive"] | str | int | float | bool | None
)
JSONDict = dict[str, JSONPrimitive]


# Maps for converting configuration strings to classes
_activation_fn_map = {
    "ReLU": torch.nn.ReLU,
    "Sigmoid": torch.nn.Sigmoid,
    "Tanh": torch.nn.Tanh,
    "Softplus": torch.nn.Softplus,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
}

_optimizer_map = {
    "Adam": torch.optim.Adam,
    "RAdam": torch.optim.RAdam,
    "SGD": torch.optim.SGD,
    "Adagrad": torch.optim.Adagrad,
    "AdamW": torch.optim.AdamW,
    "Adamax": torch.optim.Adamax,
    "RMSprop": torch.optim.RMSprop,
    "LBFGS": torch.optim.LBFGS,
}

_normalizer_map = {
    "MeanStd": MeanStd,
}


example_simple_douglas_only_x: JSONDict = {
    "dim": 3,
    "explicit_time_dependence_lagrangian": True,
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
        "hidden_layer_sizes": [16, 16],
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "helmholtz": {
        "hidden_layer_sizes": [64, 64],
        "activation_fn": "Softplus",
        "total_weight": 1.0,
        "condition_weights": [1.0, 1.0],
    },
    "initial_net": {
        "hidden_layer_sizes": [16, 16],
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}


def simple_douglas_only_x(config: JSONDict) -> LagrangianNeuralODE:
    """
    Create a `LagrangianNeuralODE` with a `SimultaneousLearnedMetricOnlyX`
    model from a simple configuration dictionary.
    """
    supress_time_dependence = not config["explicit_time_dependence_lagrangian"]
    dim = config["dim"]
    symmetric_dim = dim * (dim + 1) // 2
    full_dim = 2 * dim + 1

    normalizer = _normalizer_map[config["normalizer"]["type"]]()

    # Assemble neural ode
    ode_config = config["ode"]
    hidden_layer_sizes = ode_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[ode_config["activation_fn"]]
    ode_net = NeuralNetwork(full_dim, hidden_layer_sizes, dim, activation_fn)
    ode = FreeSecondOrderNeuralODE(ode_net, supress_time_dependence=supress_time_dependence)

    rtol = ode_config["rtol"]
    atol = ode_config["atol"]
    use_adjoint = ode_config["use_adjoint"]
    ode = SolvedSecondOrderNeuralODE(ode, rtol=rtol, atol=atol, use_adjoint=use_adjoint)

    # Assemble helmholtz metric
    helmholtz_config = config["helmholtz"]
    hidden_layer_sizes = helmholtz_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[helmholtz_config["activation_fn"]]
    metric_net = NeuralNetwork(full_dim, hidden_layer_sizes, symmetric_dim, activation_fn)
    weights = helmholtz_config["condition_weights"]
    metric = TryLearnDouglas(metric_net, *weights, supress_time_dependence=supress_time_dependence)

    # Assemble model
    # Assemble initial condition network
    initial_net_config = config["initial_net"]
    hidden_layer_sizes = initial_net_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[initial_net_config["activation_fn"]]
    initial_net = NeuralNetwork(dim, hidden_layer_sizes, dim, activation_fn)

    # Assemble optimizer
    optimizer = _optimizer_map[config["learning"]["optimizer"]]
    lr = config["learning"]["lr"]
    all_params = (
        list(metric.parameters()) + list(ode.parameters()) + list(initial_net.parameters())
    )
    optimizer = optimizer(all_params, lr=lr)

    # Assemble learning rate scheduler
    patience = config["learning"]["sheduler_patience"]
    factor = config["learning"]["sheduler_factor"]
    threshold = config["learning"]["sheduler_threshold"]
    lr_scheduler = ReduceLROnPlateau(
        optimizer, patience=patience, factor=factor, threshold=threshold
    )

    # Assemble temporal scheduler
    half_at = config["learning"]["half_time_series_steps"]
    temporal_scheduler = SigmoidTemporalScheduler(half_at)

    total_weight = helmholtz_config["total_weight"]
    model = SimultaneousLearnedMetricOnlyX(
        ode,
        metric,
        initial_net,
        optimizer,
        total_weight,
        lr_scheduler=lr_scheduler,
        temporal_scheduler=temporal_scheduler,
    )

    return LagrangianNeuralODE(model, normalizer)


example_simple_LNN_only_x: JSONDict = {
    "dim": 2,
    "explicit_time_dependence_lagrangian": False,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.05,
        "sheduler_patience": 2000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200,
    },
    "ode": {
        "Lagrangian_activation_fn": "Softplus",
        "hidden_layer_sizes": [16, 16],
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "LNN_initialization": True,
    "initial_net": {
        "hidden_layer_sizes": [16, 16],
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}


# Even though LNN is not necessarily the most precise way to refer to
# this model within the context of this framework, it corresponds
# almsot exactly to this widely known architecture and is likely
# primarily used to assess LNN performance, which is why the name is
# chosen like this.
def simple_LNN_only_x(config: JSONDict) -> LagrangianNeuralODE:
    """
    Create a `LagrangianNeuralODE` that is by construction an
    Euler-Lagrange equation from a simple configuration dictionary.
    Correspons to a Lagrangian Neural Network (LNN) [1]_.

    Note that the LNN initialization scheme is optimized for 2D systems
    with no explicit time dependence.

    References
    ----------

    .. [1] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). Lagrangian neural networks. arXiv preprint arXiv:2003.04630.
    """
    time_dependence = config["explicit_time_dependence_lagrangian"]
    dim = config["dim"]

    normalizer = _normalizer_map[config["normalizer"]["type"]]()

    # Assemble neural ode
    ode_config = config["ode"]
    hidden_layer_sizes = ode_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[ode_config["Lagrangian_activation_fn"]]
    use_LNN_initialization = config["LNN_initialization"]
    network_type = LNNNeuralNetwork if use_LNN_initialization else NeuralNetwork
    # Add time dimension to neural network input dimension only if
    # necessary, as desired by EulerLagrangeNeuralODE.
    full_dim = 2 * dim + time_dependence
    ode_net = network_type(full_dim, hidden_layer_sizes, 1, activation_fn)
    ode = EulerLagrangeNeuralODE(ode_net, supress_time_dependence=not time_dependence)

    rtol = ode_config["rtol"]
    atol = ode_config["atol"]
    use_adjoint = ode_config["use_adjoint"]
    ode = SolvedSecondOrderNeuralODE(ode, rtol=rtol, atol=atol, use_adjoint=use_adjoint)

    # Assemble helmholtz metric (dummy metric, is expected to be zero,
    # since the ODE is by construction an Euler-Lagrange equation)
    metric = DummyHelmholtzMetric()

    # Assemble model
    # Assemble initial condition network
    initial_net_config = config["initial_net"]
    hidden_layer_sizes = initial_net_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[initial_net_config["activation_fn"]]
    initial_net = NeuralNetwork(dim, hidden_layer_sizes, dim, activation_fn)

    # Assemble optimizer
    optimizer = _optimizer_map[config["learning"]["optimizer"]]
    lr = config["learning"]["lr"]
    # Add metric parameters, even though they are empty
    all_params = (
        list(metric.parameters()) + list(ode.parameters()) + list(initial_net.parameters())
    )
    optimizer = optimizer(all_params, lr=lr)

    # Assemble learning rate scheduler
    patience = config["learning"]["sheduler_patience"]
    factor = config["learning"]["sheduler_factor"]
    threshold = config["learning"]["sheduler_threshold"]
    lr_scheduler = ReduceLROnPlateau(
        optimizer, patience=patience, factor=factor, threshold=threshold
    )

    # Assemble temporal scheduler
    half_at = config["learning"]["half_time_series_steps"]
    temporal_scheduler = SigmoidTemporalScheduler(half_at)

    model = SimultaneousLearnedMetricOnlyX(
        ode,
        metric,
        initial_net,
        optimizer,
        lr_scheduler=lr_scheduler,
        temporal_scheduler=temporal_scheduler,
    )

    return LagrangianNeuralODE(model, normalizer)
