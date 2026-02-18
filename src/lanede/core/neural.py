"""
Provides basic neural networks and logic.
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with linear layers
    and an activation function.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation_fn: type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Initialize the neural network.

        Parameters
        ----------

        input_size : int
            The size/dimension of the input.
        hidden_sizes : list[int]
            The sizes of the hidden layers.
        output_size : int
            The size/dimension of the output.
        activation_fn : type[nn.Module], default=nn.ReLU
            The activation function to use between the linear layers.
        """
        super().__init__()

        self._layers = []
        for hidden_size in hidden_sizes:
            self._layers.append(nn.Linear(input_size, hidden_size))
            self._layers.append(activation_fn())
            input_size = hidden_size
        self._layers.append(nn.Linear(input_size, output_size))

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------

        x : torch.Tensor
            The input to the neural network.

        Returns
        -------

        torch.Tensor
            The output of the neural network.
        """
        return self._layers(x)

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self.parameters()).device


class LNNNeuralNetwork(nn.Module):
    """
    A neural network that is tailored to predict the Lagrangian of a
    system.

    More precisely, it utilizes the custom initialization scheme
    presented in the Lagrangian Neural Network paper [1]_.

    Note, that the initialization scheme was optimized for 2D systems
    with no explicit time dependence.

    References
    ----------

    .. [1] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). Lagrangian neural networks. arXiv preprint arXiv:2003.04630.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation_fn: type[nn.Module] = nn.Softplus,
    ) -> None:
        """
        Initialize the neural network.

        Parameters
        ----------

        input_size : int
            The size/dimension of the input.
        hidden_sizes : list[int]
            The sizes of the hidden layers. All layers must have equal
            size.
        output_size : int
            The size/dimension of the output.
        activation_fn : type[nn.Module], default=nn.Softplus
            The activation function to use between the linear layers.
        """
        super().__init__()

        if len(set(hidden_sizes)) != 1:
            raise ValueError("Hidden layers are required and must have equal size.")

        n_layers = len(hidden_sizes) + 1
        n_hidden = hidden_sizes[0]

        self._layers = []
        for index, hidden_size in enumerate(hidden_sizes):
            layer = nn.Linear(input_size, hidden_size)
            self._initialize_lnn_layer(layer, n_hidden, n_layers, index)
            self._layers.append(layer)
            self._layers.append(activation_fn())
            input_size = hidden_size
        layer = nn.Linear(input_size, output_size)
        self._initialize_lnn_layer(layer, n_hidden, n_layers, n_layers - 1)
        self._layers.append(layer)

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------

        x : torch.Tensor
            The input to the neural network.

        Returns
        -------

        torch.Tensor
            The output of the neural network.
        """
        return self._layers(x)

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self.parameters()).device

    @staticmethod
    def _initialize_lnn_layer(
        layer: nn.Linear, n_hidden: int, n_layers: int, layer_index: int
    ) -> None:
        """
        Initializes a linear layer as presented in the LNN paper [1]_.

        Parameters
        ----------

        layer : nn.Linear
            The linear layer to initialize.
        n_hidden : int
            The size of the hidden layers.
        n_layers : int
            The total number of layers (i.e., nn.Linear instances) in
            the network.
        layer_index : int
            The index of the layer to initialize. The first nn.Linear
            layer has index 0 and the last nn.Linear layer has index
            n_layers-1.

        References
        ----------

        .. [1] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). Lagrangian neural networks. arXiv preprint arXiv:2003.04630.
        """
        # Biases are initialized to zero
        nn.init.zeros_(layer.bias)

        # Weights are sampled from a normal distribution with mean 0
        # and standard deviation sqrt(1/n_hidden) mutiplied by
        # 2.2 in the first layer
        # 0.58*i in the hidden layers, where i is the layer index
        # n_hidden in the last layer
        sigma = torch.sqrt(torch.tensor(1 / n_hidden)).item()
        if layer_index == 0:
            sigma *= 2.2
        elif layer_index < n_layers - 1:
            sigma *= 0.58 * layer_index
        else:
            sigma *= n_hidden

        nn.init.normal_(layer.weight, mean=0.0, std=sigma)
