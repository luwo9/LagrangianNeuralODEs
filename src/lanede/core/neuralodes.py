"""
Provides models for for second order neural ODEs.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ._autograd import restore_dims_from_vmap


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

    def __init__(self, neural_network: nn.Module, supress_time_dependence: bool = False) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        neural_network : nn.Module
            The neural network that computes the function $f^\\ast(t, x, \dot{x})$. Must map 2n_dim+1 to n_dim.
        supress_time_dependence : bool, default=False
            If True, the time dependence is supressed, i.e.
            $f^\\ast(x, \dot{x})$ is computed instead of
            $f^\\ast(t, x, \dot{x})$. This is done by setting t=0 in
            the neural network input.
        """
        super().__init__()
        self._neural_network = neural_network
        # Get as a factor to multiply the time input with
        self._time_factor = float(not supress_time_dependence)

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
        # Suppress time dependence if configured
        t = t * self._time_factor
        input = torch.cat([t, x, xdot], dim=2)

        xdotdot = self._neural_network(input)
        return xdotdot


class EulerLagrangeNeuralODE(SecondOrderNeuralODE):
    """
    A second order neural ODE that by construction is an Euler-Lagrange
    equation, where the Lagrangian is learned by a neural network.

    This is basically the implementation of a Lagrangian Neural Network
    (LNN) [1]_ within the Lagrangian neural ODE framework.

    References
    ----------

    .. [1] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). Lagrangian neural networks. arXiv preprint arXiv:2003.04630.
    """

    def __init__(
        self,
        neural_network: nn.Module,
        supress_time_dependence: bool = False,
    ) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        neural_network : nn.Module
            The neural network that computes the Lagrangian. If
            supress_time_dependence is False, it must map 2n_dim+1 to
            1, else it must map 2n_dim to 1. See Notes.
        supress_time_dependence : bool, default=False
            If True, the time dependence is supressed, i.e.
            $L(x, \dot{x})$ is computed instead of
            $L(t, x, \dot{x})$.

        Notes
        -----

        Internally torch.func is used, so the neural network should be
        side-effect free. See [1]_ for more information.

        References
        ----------

        .. [1] PyTorch documentation: UX Limitations https://pytorch.org/docs/stable/func.ux_limitations.html
        """
        super().__init__()
        self._neural_network = neural_network
        # The neural network may not be implemented to allow time input
        # (with the custom LNN implementation), so the time is not set
        # to zero but not used at all.
        # To avoid if statements in the forward pass, define a
        # concatenation here
        if supress_time_dependence:
            self._make_input = lambda t, x, xdot: torch.cat([x, xdot], dim=2)
        else:
            self._make_input = lambda t, x, xdot: torch.cat([t, x, xdot], dim=2)

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self._neural_network.parameters()).device

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the second order deriative. More precisely, for the
        underlying ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$ this
        function is the $f^\\ast$.

        It is computed from the explicit Euler-Lagrange equation

        $$
        \ddot{x} = g^{-1} \left( \frac{\partial L}{\partial x}
        - \frac{\partial^2 L}{\partial x \partial \dot{x}} \dot{x}
        - \frac{\partial^2 L}{\partial t \partial \dot{x}} \right),
        $$

        with $g = \frac{\partial^2 L}{\partial \dot{x}^2}$.

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
        Lagrangian = restore_dims_from_vmap(self.Lagrangian, (0, 0))

        # Returns tuple of (dL/dx, dL/dxdot):
        jacobian_fn = torch.func.jacrev(Lagrangian, argnums=(1, 2))

        def jacobian_fn_with_aux(t, x, xdot):
            jacobian_x, jacobian_xdot = jacobian_fn(t, x, xdot)
            # Prepare the outout for jacrev of type
            # (need_derivative_of, need_value_of)
            # Here, higher order derivatives are only needed from
            # dL/dxdot and the value is only needed from dL/dx:
            return jacobian_xdot, jacobian_x

        # Returns tuple of ((d^2L/dtdxdot, d^2L/dxdxdot, d^2L/dxdot^2), dL/dx):
        all_jacobians_fn = torch.func.jacrev(jacobian_fn_with_aux, argnums=(0, 1, 2), has_aux=True)

        # Vmap over the batch and time dimensions, otherwise the
        # jacobians would be computed w.r.t. those aswell.
        all_jacobians_fn = torch.func.vmap(all_jacobians_fn, in_dims=0, out_dims=0)
        all_jacobians_fn = torch.func.vmap(all_jacobians_fn, in_dims=0, out_dims=0)

        # Now compute the values of the derivatives of the Lagrangian
        (jacobian_tv, jacobian_xv, jacobian_vv), jacobian_x = all_jacobians_fn(t, x, xdot)

        # Compute the second order derivative using the explicit
        # Euler-Lagrange equation

        # Use pseudo inverse as in the LNN paper
        g_inverse = torch.pinverse(jacobian_vv)
        xdot_term = torch.einsum("abij,abj->abi", jacobian_xv, xdot)
        generalized_force = jacobian_x - xdot_term - jacobian_tv
        xdotdot = torch.einsum("abij,abj->abi", g_inverse, generalized_force)
        return xdotdot

    def Lagrangian(self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor) -> torch.Tensor:
        """
        Computes the Lagrangian at the given time, state and state derivative.

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

        torch.Tensor, shape (n_batch, n_steps)
            The Lagrangian at the given time, state and state derivative.
        """
        t = t.unsqueeze(2)
        input_ = self._make_input(t, x, xdot)
        # Output scalar, as compatible with torch.func.jacrev
        return self._neural_network(input_).squeeze(2)
