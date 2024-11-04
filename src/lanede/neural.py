"""
Provides basic neural networks and logic.
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with ReLU activation functions.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, activation_fn: nn.Module = nn.ReLU) -> None:
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
        activation_fn : nn.Module, default=nn.ReLU
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
        return next(self.parameters()).device