"""
This module contains logic for creating predefined
`LagrangianNeuralODE`'s from simple configuration dictionaries.
"""
import torch
from lanede.core import (
    NeuralNetwork,
    FreeSecondOrderNeuralODE,
    SolvedSecondOrderNeuralODE,
    TryLearnDouglas,
    SimultaneousLearnedDouglasOnlyX,
    LagrangianNeuralODE,
    MeanStd,
)


# NOTE: Every function listed here should have signature
# `def func_name(config: JSONDict) -> LagrangianNeuralODE:`
# For every function, there should be a corresponding example
# configuration dictionary named `example_<func_name>`.


JSONPrimitive = dict[str, "JSONPrimitive"] | list["JSONPrimitive"] | str | int | float | bool | None
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
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.01,
    },
    "ode": {
        "activation_fn": "Softplus",
        "hidden_layer_sizes": [16, 16],
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": True,
    },
    "helmholtz": {
        "hidden_layer_sizes": [16, 16],
        "activation_fn": "ReLU",
        "total_weight": 100.0,
        "condition_weights": [1.0, 1.0, 0.1],
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
    Create a `LagrangianNeuralODE` with a `SimultaneousLearnedDouglasOnlyX`
    model from a simple configuration dictionary.
    """
    dim = config["dim"]
    symmetric_dim = dim * (dim + 1) // 2
    full_dim = 2 * dim + 1

    normalizer = _normalizer_map[config["normalizer"]["type"]]()

    # Assemble neural ode
    ode_config = config["ode"]
    hidden_layer_sizes = ode_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[ode_config["activation_fn"]]
    ode_net = NeuralNetwork(full_dim, hidden_layer_sizes, dim, activation_fn)
    ode = FreeSecondOrderNeuralODE(ode_net)
    
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
    metric = TryLearnDouglas(metric_net, *weights)

    # Assemble model
    # Assemble initial condition network
    initial_net_config = config["initial_net"]
    hidden_layer_sizes = initial_net_config["hidden_layer_sizes"]
    activation_fn = _activation_fn_map[initial_net_config["activation_fn"]]
    initial_net = NeuralNetwork(dim, hidden_layer_sizes, dim, activation_fn)

    optimizer = _optimizer_map[config["learning"]["optimizer"]]
    lr = config["learning"]["lr"]
    all_params = (list(metric.parameters())
                  + list(ode.parameters())
                  + list(initial_net.parameters()))
    optimizer = optimizer(all_params, lr=lr)

    total_weight = helmholtz_config["total_weight"]
    model = SimultaneousLearnedDouglasOnlyX(ode, metric, initial_net, optimizer, total_weight)

    return LagrangianNeuralODE(model, normalizer)