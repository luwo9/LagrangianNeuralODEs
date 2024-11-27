"""
Provides functionality to measure how well the Helmholtz conditions are satisfied.
The Helmholtz conditions are a set of (neccessary and sufficient) conditions for a second order ODE
to originate from Euler-Lagrange equations of a Lagrangian. See e.g. [1]_ for more information.

References
----------

.. [1] Wikipedia: "Inverse problem for Lagrangian mechanics" https://en.wikipedia.org/wiki/Inverse_problem_for_Lagrangian_mechanics
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn

from ._autograd import restore_dims_from_vmap


class HelmholtzMetric(nn.Module, ABC):
    """
    Abstract class for a metric to measure the satisfaction of the Helmholtz conditions for a second order ODE.

    Methods
    -------

    forward(f, t, x, xdot)
        Computes the metric for the Helmholtz conditions.

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
    def forward(self, f: Callable, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, scalar: bool = True, combined: bool = True) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for a given second order ODE at given points in time and state.
        Depending on its arguments, it returns either a single metric
        for all conditions combined or individual metrics for each
        condition. These may either be for each point supplied or
        averaged over all points.

        Parameters
        ----------

        f : Callable
            The function $f^\\ast(t, x, \dot{x})$ of the second order ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$.
        t : torch.Tensor, shape (n_batch, n_steps)
            The time points at which the Helmholtz conditions should be evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time steps t.
        scalar : bool, default=True
            Whether to return a single scalar metric or a pointwise
            result.
        combined : bool, default=True
            Whether to return a single metric for all conditions
            combined or individual metrics for each condition.


        Returns
        -------

        torch.Tensor or tuple[torch.Tensor, ...], shape (n_batch, n_steps) or scalar
            The metric for the Helmholtz conditions.
        """
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        pass


class TryLearnDouglas(HelmholtzMetric):
    """
    Uses the Helmholtz conditions after Douglas (see [1]_).
    The matrix g is represented by a neural network.
    The metric is constructed by a sum of the absolute differences of
    both hand sides of the Helmholtz conditions. This metric can act as
    a loss function for training the neural network for g.

    References
    ----------

    Douglas J. (1939). Solution of the Inverse Problem of the Calculus of Variations. Proceedings of the National Academy of Sciences of the United States of America, 25(12), 631-637. https://doi.org/10.1073/pnas.25.12.631
    """

    def __init__(self, neural_network: nn.Module, h_2_weight: float = 1.0, h_3_weight: float = 1.0, non_singular_weight: float = 0.1) -> None:
        """
        Initializes the metric.

        Parameters
        ----------

        neural_network : nn.Module
            The neural network to learn the g matrix. Must map 2n_dim+1
            to n_dim(n_dim+1)/2, where n_dim is the dimension of
            the state x.
        h_2_weight : float, default=1.0
            The relative weight for the second Helmholtz condition in the loss (see notes).
        h_3_weight : float, default=1.0
            The relative weight for the third Helmholtz condition in the loss (see notes).
        non_singular_weight : float, default=0.1
            The relative weight to ensure that the matrix g is non-singular.

        Notes
        -----

        The order and convention of the Helmholtz conditions
        is taken to be the same as in Wikipedia [1]_.

        References
        ----------

        .. [1] Wikipedia: "Inverse problem for Lagrangian mechanics" https://en.wikipedia.org/wiki/Inverse_problem_for_Lagrangian_mechanics
        """
        super().__init__()
        self._neural_network = neural_network
        self._h_2_weight = h_2_weight
        self._h_3_weight = h_3_weight
        self._non_singular_weight = non_singular_weight

    def forward(self, f: Callable, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, scalar: bool = True, combined: bool = True) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for a given second order ODE at given points in time and state.
        Depending on its arguments, it returns either a single metric
        for all conditions combined or individual metrics for each
        condition. These may either be for each point supplied or
        averaged over all points.
        Also returns a measure for singularity of the g matrix.

        Parameters
        ----------

        f : Callable
            The function $f^\\ast(t, x, \dot{x})$ of the second order ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$.
        t : torch.Tensor, shape (n_batch, n_steps)
            The time points at which the Helmholtz conditions should be evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time steps t.
        scalar : bool, default=True
            Whether to return a single scalar metric or a pointwise
            result.
        combined : bool, default=True
            Whether to return a single metric for all conditions
            combined or individual metrics for each condition.

        Returns
        -------

        torch.Tensor or tuple[torch.Tensor, ...], shape (n_batch, n_steps) or scalar
            The metric for the Helmholtz conditions. If not combined,
            returns 3 tensors for the Helmoltz conditions and one for
            the non-singularity of the g matrix. See Notes.

        Notes
        -----

        The order of the Helmholtz conditions is as usual
        (see `__init__`).
        
        Internaly torch.func is used, so f should be side-effect free.
        See [1]_ for more information.
        The singularity measure is the absolute value of the
        inverse of the determinant of the g matrix.

        References
        ----------

        .. [1] PyTorch documentation: UX Limitations https://pytorch.org/docs/stable/func.ux_limitations.html
        """
        helmholtz_1, helmholtz_2, helmholtz_3, loss_non_singular = self._evaluate_helmholtz_conditions(f, t, x, xdot, scalar)

        if not combined:
            return helmholtz_1, helmholtz_2, helmholtz_3, loss_non_singular

        metric = (helmholtz_1 + self._h_2_weight * helmholtz_2 + self._h_3_weight * helmholtz_3
                   + self._non_singular_weight * loss_non_singular)
        
        return metric
    
    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self._neural_network.parameters()).device
    
    def _evaluate_helmholtz_conditions(self, f: Callable, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, scalar: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the Helmholtz conditions for a given second order ODE
        at given points in time and state. Returns the absolute of the
        difference of left and right hand side of the Helmholtz
        conditions.
        Also returns a measure for singularity of the g matrix.
        Averaged according to `scalar`.

        Parameters
        ----------

        f : Callable
            The function $f^\\ast(t, x, \dot{x})$ of the second order ODE,
            $\ddot{x} = f^\\ast(t, x, \dot{x})$,
            which should be evaluated. See Notes for important requirements.
        t : torch.Tensor, shape (n_batch, n_steps)
            The time points at which the Helmholtz conditions should be evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time steps t.
        scalar : bool, default=True
            Whether to take the mean over all dimensions or just the
            non-batch dimensions.

        Returns
        -------

        torch.Tensor, shape (n_batch, n_steps) or scalar
            The difference of the left and right hand side of the
            first Helmholtz condition. (Absolute value)
        torch.Tensor, shape (n_batch, n_steps) or scalar
            The difference of the left and right hand side of the
            second Helmholtz condition. (Absolute value)
        torch.Tensor, shape (n_batch, n_steps) or scalar
            The difference of the left and right hand side of the
            third Helmholtz condition. (Absolute value)
        torch.Tensor, shape (n_batch, n_steps) or scalar
            The measure for singularity of the g matrix.
        
        Notes
        -----

        Internaly torch.func is used, so f should be side-effect free.
        See [1]_ for more information.
        The singularity measure is the absolute value of the
        inverse of the determinant of the g matrix.

        References
        ----------

        .. [1] PyTorch documentation: UX Limitations https://pytorch.org/docs/stable/func.ux_limitations.html
        """
        # Computing jacobians etc. with torch.func does not work in inference mode.
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Computing Helmholtz conditions in"
                                "inference mode is not supported.")

        f_values, jacobian_x, jacobian_v, total_derivative_of_jacobian_v = self._compute_f_terms(f, t, x, xdot)
        # shapes: f: (n_batch, n_step, n_dim), rest: (n_batch, n_step, n_dim, n_dim)
        g, total_derivative_of_g, g_jacobian_v, loss_non_singular = self._compute_g_terms(t, x, xdot, f_values)
        # shapes: g and dg/dt: (n_batch, n_steps, n_dim, n_dim),
        # jacobian: (n_batch, n_steps, n_dim, n_dim, n_dim),
        # loss: (n_batch, n_steps)

        # H1: gPhi - (gPhi)^T
        jacobian_v_squared = torch.einsum("abik,abkj->abij", jacobian_v, jacobian_v)
        Phi = 0.5 * total_derivative_of_jacobian_v - jacobian_x - 0.25 * jacobian_v_squared

        gPhi = torch.einsum("abij,abjk->abik", g, Phi)
        helmholtz_1 = gPhi - gPhi.transpose(-2, -1)

        # H2: dg/dt + 1/2df/dv^T g + 1/2 df/dv g
        jacobian_v_times_g = 0.5 * torch.einsum("abki,abkj->abij", jacobian_v, g)
        helmholtz_2 = total_derivative_of_g + jacobian_v_times_g + jacobian_v_times_g.transpose(-2, -1)

        # H3: dg/dv - dg/dv^T
        helmholtz_3 = g_jacobian_v - g_jacobian_v.transpose(-2, -1)

        # Take mean according to "scalar"
        if scalar:
            dim_h1h2 = (0, 1, 2, 3)
            dim_h3 = (0, 1, 2, 3, 4)

            loss_non_singular = torch.mean(loss_non_singular)
        else:
            dim_h1h2 = (2, 3)
            dim_h3 = (2, 3, 4)
        
        helmholtz_1 = torch.mean(helmholtz_1.abs(), dim=dim_h1h2)
        helmholtz_2 = torch.mean(helmholtz_2.abs(), dim=dim_h1h2)
        helmholtz_3 = torch.mean(helmholtz_3.abs(), dim=dim_h3)

        return helmholtz_1, helmholtz_2, helmholtz_3, loss_non_singular

    def _compute_f_terms(self, f: Callable, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Computes f and its derivatives for the Helmholtz conditions,
        # namely f, df/dx, df/dv, d/dt df/dv (v=xdot).
        # Uses only a single autograd pass.

        # For the total time derivative d/dt df/dv, we compute the "double"
        # jacobians of df/dv, i.e. d^2f/dxdv, d^2f/dv^2, d^2d/dtdv (partial).
        # And then multiply with the v and vdot (xdotdot), respectively.
        # This way of coumputing the jvp seems to be more efficient
        # than torch.func.jvp here.

        def double_result(t, x, v):
            # Return double result for keeping jacobian and value.
            result = f(t, x, v)
            return result, result
        
        fn = restore_dims_from_vmap(double_result, (0, 0))
        # Returns (df/dx, df/dv), f
        fn = torch.func.jacrev(fn, argnums=(1, 2), has_aux=True)

        def double_result2(t, x, v):
            # Again Double result for getting double jacobians and single ones and result.
            jacobians, result = fn(t, x, v)
            jacobian_v = jacobians[1]
            return jacobian_v, (jacobians, result)
        
        # Returns (d^2d/dtdv (partial), d^2f/dxdv, d^2f/dv^2), ((df/dx, df/dv), f)
        fn2 = torch.func.jacrev(double_result2, argnums=(0,1,2), has_aux=True)
        # vmap over first two dims as they are batch dims
        # otherwise jacrev would compute the jacobian also w.r.t. the batch dims
        fn2 = torch.func.vmap(fn2, in_dims=0, out_dims=0)
        fn2 = torch.func.vmap(fn2, in_dims=0, out_dims=0)

        double_jacobians, (jacobians, f_values) = fn2(t, x, xdot)
        jacobian_tv, jacobian_xv, jacobian_vv = double_jacobians
        jacobian_x, jacobian_v = jacobians

        total_derivative_of_jacobian_v = self._compute_total_derivative(jacobian_tv, jacobian_xv, jacobian_vv, xdot, f_values)

        return f_values, jacobian_x, jacobian_v, total_derivative_of_jacobian_v
    
    def _compute_g_terms(self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, xdotdot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the g matrix and its derivatives.
        # Uses only a single autograd pass. Returns a loss term to enforce
        # non-singularity of g for all points.

        # NOTE: G must be symmetric and non-singular.
        # While parametrizing a symmetric matrix is trivial (see below)
        # and ensuring non-singularity is possible by learning in a suitable
        # PLU decomposition, I have not found a way to enforce both
        # constraints simultaneously.
        # 
        # An idea would be to learn g=Q^TDQ, where Q is orthogonal
        # (e.g. via matrix exponential of skew-symmetric matrix) and D is
        # the diagonal matrix of eigenvalues. However, for an in general
        # indefinite matrix g, the eigenvalues can be any number but zero,
        # where it is hard to find a proper way to enforce this by
        # construction.
        #
        # Therefore symmetry is by construction and non-singularity by adding
        # a loss term.
        #
        # Thus, the neural network outputs the number of independent parameters
        # of a symmetric matrix and it is then assembled for computation.

        # For a slightly more efficient computation, all derivatives
        # are computed only of the independent parameters, not the full
        # matrix.

        def double_result(t, x, v):
            # Return double result for keeping jacobian and value.
            t = t.unsqueeze(2)
            result = self._neural_network(torch.cat([t,x,v], dim=2))
            return result, result
        
        fn = restore_dims_from_vmap(double_result, (0, 0))
        # Returns (dg/dt (partial), dg/dx, dg/dv), g
        fn = torch.func.jacrev(fn, argnums=(0,1,2), has_aux=True)
        fn = torch.func.vmap(fn, in_dims=0, out_dims=0)
        fn = torch.func.vmap(fn, in_dims=0, out_dims=0)

        jacobians_vector, g_vector = fn(t, x, xdot)

        # Assemble matrices from free parameters (vector).
        n_dim = x.shape[-1]
        g = self._assemble_symmetric(g_vector, n_dim)

        jacobian_t_vector = jacobians_vector[0]
        jacobians = [self._assemble_symmetric(jacobian_t_vector, n_dim)]

        # _asemble_symmetric expects shape (..., n_dim*(n_dim+1)/2), but
        # jacobians are (...,n_dim*n_dim(n_dim+1)/2, n_dim) so permute
        # before and after call.
        for jacobian in jacobians_vector[1:]:
            jacobian = jacobian.permute(0, 1, 3, 2)
            jacobian = self._assemble_symmetric(jacobian, n_dim)
            jacobian = jacobian.permute(0, 1, 3, 4, 2)

            jacobians.append(jacobian)

        jacobian_t, jacobian_x, jacobian_v = jacobians

        total_derivative_of_g = self._compute_total_derivative(jacobian_t, jacobian_x, jacobian_v, xdot, xdotdot)

        # Loss term to enforce non-singularity of g.
        # Penalize 0 determinant directly, this is more efficient than
        # doing it e.g. in Q^TDQ decomposition
        det_g = torch.det(g)
        loss_non_singular = torch.abs(1 / det_g)
        
        return g, total_derivative_of_g, jacobian_v, loss_non_singular

    @staticmethod
    def _assemble_symmetric(vector: torch.Tensor, n_dim: int) -> torch.Tensor:
        # Assemble a symmetric matrix from a vector.

        shape = vector.shape[:-1]

        symmetric_matrix = vector.new_zeros(*shape, n_dim, n_dim)
        triu_indices = torch.triu_indices(n_dim, n_dim)
        symmetric_matrix[..., triu_indices[0], triu_indices[1]] = vector
        symmetric_matrix += symmetric_matrix.transpose(-2, -1)

        return symmetric_matrix

    @staticmethod
    def _compute_total_derivative(jacobian_t, jacobian_x, jacobian_xdot, xdot, xdotdot) -> torch.Tensor:
        # Compute total time derivative given the jacobians and the derivatives.
        # Works with f (double jacobians) and g (single jacobians),
        # as they have the same shape. Contracts over last dimension.

        jvp = torch.einsum("abcij,abj->abci", jacobian_x, xdot)
        jvp += torch.einsum("abcij,abj->abci", jacobian_xdot, xdotdot)
        jvp += jacobian_t

        return jvp