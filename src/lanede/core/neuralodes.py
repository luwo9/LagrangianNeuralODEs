"""
Provides models for for second order neural ODEs.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SecondOrderNeuralODE(nn.Module, ABC):
    """
    Abstract class for second order neural ODEs. I.e. ODEs of the form $\ddot{x} = f^\\ast(t, x, \dot{x})$.

    Methods
    -------

    forward(t, x_and_xdot)
        Usual PyTorch forward pass of a neural ODE reduced to a first order system.

    second_order_function(t, x, xdot)
        Computes the second order deriative, i.e. $f^\\ast(t, x, \dot{x})$.

    Attributes
    ----------

    device
        The device on which the model is stored.

    Notes
    -----

    This is a nn.Module, and thus has all usual properties of a PyTorch
    module.
    """

    @abstractmethod
    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second order deriative. More precisely, for the underlying ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$ this function is the $f^\\ast$.

        Parameters
        ----------

        t : torch.Tensor, shape (n_batch, n_steps)
            The time at which the state is evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The state at time t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time t.

        Returns
        -------

        torch.Tensor, shape (n_batch, n_steps, n_dim)
            The second order derivative of the state.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        pass

    def forward(self, t: torch.Tensor, x_and_xdot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Computes the neural ODE reduced to a first order system.

        Namely, given a state s, always assumed in the form [x, xdot], computes the $f$ in the ODE $\dot{s} = f(t, s)$.

        Parameters
        ----------

        t : torch.Tensor, scalar
            The time at which the state is evaluated.
        x_and_xdot : torch.Tensor, shape (n_batch, 2*n_dim)
            The state at time t. Always assumed in the form [x, xdot].

        Returns
        -------

        torch.Tensor, shape (n_batch, 2*n_dim)
            The derivative of the state at time t.
        """
        dim = x_and_xdot.shape[1] // 2

        # Prepare shape for the second order function
        n_batch = x_and_xdot.shape[0]
        t = t.repeat(n_batch, 1)
        x_and_xdot = x_and_xdot.unsqueeze(1)

        x, xdot = torch.split(x_and_xdot, dim, dim=2)  # dim=2 as unsqueezed
        xdotdot = self.second_order_function(t, x, xdot)
        x_and_xdot_dot = torch.cat([xdot, xdotdot], dim=2).squeeze(1)  # = sdot = d/dt x_and_xdot

        return x_and_xdot_dot


class FreeSecondOrderNeuralODE(SecondOrderNeuralODE):
    """
    A second order neural ODE of the form $\ddot{x} = f^\\ast(t, x, \dot{x})$, where the function $f^\\ast(t, x, \dot{x})$ is given by a neural network.

    Methods
    -------

    second_order_function(t, x, xdot)
        Computes the second order deriative $f^\\ast(t, x, \dot{x})$ using a neural network.
    """

    def __init__(self, neural_network: nn.Module) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        neural_network : nn.Module
            The neural network that computes the function $f^\\ast(t, x, \dot{x})$. Must map 2n_dim+1 to n_dim.
        """
        super().__init__()
        self._neural_network = neural_network

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self._neural_network.parameters()).device

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second order deriative. More precisely, for the underlying ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$ this function is the $f^\\ast$.

        Parameters
        ----------

        t : torch.Tensor, shape (n_batch, n_steps)
            The time at which the state is evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The state at time t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time t.

        Returns
        -------

        torch.Tensor, shape (n_batch, n_steps, n_dim)
            The second order derivative of the state.
        """
        t = t.unsqueeze(2)
        input = torch.cat([t, x, xdot], dim=2)

        xdotdot = self._neural_network(input)
        return xdotdot
