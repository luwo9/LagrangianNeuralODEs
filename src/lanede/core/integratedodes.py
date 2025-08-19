"""
Provides integrated or solved versions of second order neural ODEs.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

from .neuralodes import SecondOrderNeuralODE


class SolvedSecondOrderNeuralODE(nn.Module):
    """
    Represents an integrated/solved second order neural ODE.

    It solves the initial value problem, such that the state x and its derivative xdot are known
    at any time t given the initial state.

    Methods
    -------

    forward(t, x_0, xdot_0)
        Computes the state x and derivative xdot at time t.

    second_order_function(t, x, xdot)
        Computes the second order derivative of the state.

    Notes
    -----

    This is a nn.Module, and thus has all usual properties of a PyTorch
    module.
    """

    def __init__(
        self, neural_ode: SecondOrderNeuralODE, use_adjoint: bool = True, **odeint_kwargs
    ) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        neural_ode : SecondOrderNeuralODE
            The second order neural ODE that is integrated.
        use_adjoint : bool, default=True
            Whether to use the adjoint method for integrating the ODE. (See torchdiffeq documentation.)
        odeint_kwargs
            Additional keyword arguments for the odeint solver. See torchdiffeq documentation.
        """
        super().__init__()
        self._neural_ode = neural_ode
        self._odeint = self._get_odeint_solver(use_adjoint)
        self._odeint_kwargs = odeint_kwargs

    def forward(
        self, t: torch.Tensor, x_0: torch.Tensor, xdot_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the state x at time t given the initial state x_0 and its derivative xdot_0,
        by integrating the underlying ODE. The derivative xdot is computed aswell.

        Parameters
        ----------

        t : torch.Tensor, shape (n_steps,)
            The time steps at which the batch of states should be evaluated.
        x_0 : torch.Tensor, shape (n_batch, n_dim)
            The initial state.
        xdot_0 : torch.Tensor, shape (n_batch, n_dim)
            The initial value of the derivative of the state.

        Returns
        -------

        torch.Tensor, shape (n_batch, n_steps, n_dim)
            The state at time (steps) t.
        torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time (steps) t.
        """
        dim = x_0.shape[1]
        x_and_xdot_0 = torch.cat([x_0, xdot_0], dim=1)

        x_and_xdot = self._odeint(self._neural_ode, x_and_xdot_0, t, **self._odeint_kwargs)
        x_and_xdot = x_and_xdot.permute(1, 0, 2)  # timesteps at second dimension

        x, xdot = torch.split(x_and_xdot, dim, dim=2)
        return x, xdot

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second order derivative. More precisely, for the underlying ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$ this function is the $f^\\ast.

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
        return self._neural_ode.second_order_function(t, x, xdot)

    @staticmethod
    def _get_odeint_solver(use_adjoint: bool) -> Callable:
        return odeint_adjoint if use_adjoint else odeint

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return self._neural_ode.device
