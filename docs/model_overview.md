# API presets overview

As explained in the [API Quickstart](docs/api_quickstart.md), models created using the API are based on predefined model types. These presets are identified by their name and the creation of a model only requires a corresponding configuration dictionary. For every preset, an example configuration dictionary is provided in the `EXAMPLES` dictionary:

```python
from lanede.api import LanedeAPI, EXAMPLES
name = "simple_douglas"
config = EXAMPLES[name]
model = LanedeAPI(name, config)
```

Here, all available presets are listed and explained, together with the corresponding configuration dictionary.

Note that, since the configuration dictionaries are JSON-serializable, they contain strings like `"Softplus"`, `"RAdam"` or `MeanStd` to specify, e.g., the activation function, the optimizer or the Normalizer, which are then converted to the corresponding classes in the background.

For an overview and an explanation of underlying internally used components such as Normalizers, or Helmholtz metrics, see [the core framework documentation](docs/core_framework.md).

## simple_douglas

The `simple_douglas` model directly predicts the acceleration using a neural network and requires only positional data for training. Euler-Lagrange dynamics are enforced by a regularizing with a `TryLearnDouglas` Helmholtz metric. This metric predicts $g$, the Hessian of the Lagrangian, with a neural network. The initial velocity is predicted from the initial position using a third neural network. The model uses `SimultaneousLearnedMetricOnlyX` as internal model, i.e., all three networks are trained simultaneously with a linear combination of the MSE loss and the Helmholtz metric loss. The gradients of the Helmholtz metric loss onto the neural ODE are clipped to find Euler-Lagrange equations that are data-consistent. The MSE loss is only computed from the state (as the derivative is not supplied). Time points are rolled out, i.e., at the beginning of training, only two time points are used for training, and more time points are added during training until the full trajectories are used. This is done in a sigmoid fashion. The learning rate is intended to be relatively high, such that the model stays close to the data, but it is decreased on plateaus of the Helmholtz metric loss (or the MSE, if regularization is disabled).

The configuration dictionary for this model is given by:
```python
{
    "dim": 3,
    "explicit_time_dependence_lagrangian": True,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.07,
        "sheduler_patience": 2000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 1200,
    },
    "ode": {
        "activation_fn": "Softplus",
        "hidden_layer_sizes": [16],
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
        "hidden_layer_sizes": [16, 16, 16],
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}
```

There,
- `dim` specifies the dimension of the system.
- `explicit_time_dependence_lagrangian` specifies whether the Lagrangian is explicitly time dependent, which allows to learn non-conservative systems.
- `learning` contains the settings for the learning process:
    - `optimizer` specifies the optimizer to use.
    - `lr` specifies the initial learning rate.
    - `sheduler_patience`, `sheduler_factor` and `sheduler_threshold` specify the settings for the learning rate scheduler, see pytorch's `ReduceLROnPlateau`.
    - `half_time_series_steps` specifies after how many training steps the number of time points used for training is increased to half of the total time points.
- `ode` contains the settings for the neural ODE:
    - `activation_fn` specifies the activation function to use in the neural ODE (should be smoothly differentiable).
    - `hidden_layer_sizes` specifies the hidden layer sizes of the neural ODE.
    - `rtol` and `atol` specify the relative and absolute error tolerance for the adaptive solver.
    - `use_adjoint` specifies whether to use the adjoint method for neural ODE training.
- `helmholtz` contains the settings for the Helmholtz metric:
    - `hidden_layer_sizes` specifies the hidden layer sizes of the neural network that predicts $g$, the Hessian of the Lagrangian.
    - `activation_fn` specifies the activation function to use in this neural network (should be smoothly differentiable).
    - `total_weight` specifies the weight of the Helmholtz metric loss in the total loss, mostly used binary, i.e. 1 or 0, to enable or disable the regularization.
    - `condition_weights` specifies the weights of the different conditions in the Helmholtz metric, which are explained in the documentation of the `TryLearnDouglas` metric.
- `initial_net` contains the settings for the neural network that predicts the initial velocity from the initial position:
    - `hidden_layer_sizes` specifies the hidden layer sizes of this neural network.
    - `activation_fn` specifies the activation function to use in this neural network.
- `normalizer` contains the settings for the normalizer that is applied to the data:
    - `type` specifies the type of the normalizer.

The number of training steps is given by the number of batches the model has been supplied and therefore updated on. If the data contains $N$ trajectories and a batch size of $B$ is used, then the number of training steps per eopch will usually be $\lceil N/B \rceil$.

## simple_LNN

The `simple_LNN` preset trains a second order Neural ODE, which by construction is an Euler-Lagrange equation. Only positional data is required for training. Here, the Lagrangian itself is predicted by a neural network and the explicit Euler-Lagrange equation is computed to obtain the second order ODE. Thus, regularization with a Helmholtz metric is not required and a dummy metric is used, that always returns zero. This is basically the implementation of a Lagrangian Neural Network (LNN, [Cranmer et al., 2020](https://arxiv.org/abs/2003.04630)) within the Lagrangian neural ODE framework. The prediction MSE loss is computed only from the state, as the derivative is not supplied. Time points are rolled out during training in a sigmoid fashion, as described for the `simple_douglas` preset. The learning rate is decreased on plateaus of the MSE loss.

To stabilise the LNN training, the neural network prediction of the Lagrangian is added a quadratic kinetic energy term for better initialization and gradients are clipped, as discussed in [this paper](https://arxiv.org/abs/2601.12519) and [this paper](https://arxiv.org/abs/2106.00026).

The configuration dictionary for this model is given by:
```python
{
    "dim": 2,
    "explicit_time_dependence_lagrangian": False,
    "learning": {
        "optimizer": "RAdam",
        "lr": 0.07,
        "sheduler_patience": 1000,
        "sheduler_factor": 0.5,
        "sheduler_threshold": 1e-2,
        "half_time_series_steps": 400,
    },
    "ode": {
        "Lagrangian_activation_fn": "Softplus",
        "hidden_layer_sizes": [32],
        "rtol": 1e-6,
        "atol": 1e-6,
        "use_adjoint": False,
    },
    "LNN_initialization": True,
    "initial_net": {
        "hidden_layer_sizes": [16, 16, 16],
        "activation_fn": "ReLU",
    },
    "normalizer": {
        "type": "MeanStd",
    },
}
```

There,
- `dim` specifies the dimension of the system.
- `explicit_time_dependence_lagrangian` specifies whether the Lagrangian is explicitly time dependent, which allows to learn non-conservative systems.
- `learning` contains the settings for the learning process:
    - `optimizer` specifies the optimizer to use.
    - `lr` specifies the initial learning rate.
    - `sheduler_patience`, `sheduler_factor` and `sheduler_threshold` specify the settings for the learning rate scheduler, see pytorch's `ReduceLROnPlateau`.
    - `half_time_series_steps` specifies after how many training steps the number of time points used for training is increased to half of the total time points.
- `ode` contains the settings for the neural ODE:
    - `Lagrangian_activation_fn` specifies the activation function to use in the neural network that predicts the Lagrangian (should be smoothly differentiable).
    - `hidden_layer_sizes` specifies the hidden layer sizes of this neural network.
    - `rtol` and `atol` specify the relative and absolute error tolerance for the adaptive solver.
    - `use_adjoint` specifies whether to use the adjoint method for neural ODE training.
- `LNN_initialization` specifies whether to use the custom LNN initialization presented in [the original LNN paper](https://arxiv.org/abs/2003.04630). It is only optimized for time-independent Lagrangians, i.e., when `explicit_time_dependence_lagrangian` is `False`.
- `initial_net` contains the settings for the neural network that predicts the initial velocity from the initial position:
    - `hidden_layer_sizes` specifies the hidden layer sizes of this neural network.
    - `activation_fn` specifies the activation function to use in this neural network.
- `normalizer` contains the settings for the normalizer that is applied to the data:
    - `type` specifies the type of the normalizer.

The number of training steps is given by the number of batches the model has been supplied and therefore updated on. If the data contains $N$ trajectories and a batch size of $B$ is used, then the number of training steps per eopch will usually be $\lceil N/B \rceil$.