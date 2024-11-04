"""
Provides basic neural networks and logic.
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with ReLU activation functions.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, activation_fn: nn.Module = nn.ReLU):
        """
        Initialize the neural network.

        :param input_size: int
        :param hidden_sizes: list of int
        :param output_size: int
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

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self._layers(x)
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device