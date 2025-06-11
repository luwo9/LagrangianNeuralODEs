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
    def forward(
        self,
        f: Callable,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for a given second order ODE at given points in time and state.
        Depending on its arguments, it returns a single metric for all
        conditions combined and optionally individual metrics for each
        condition. These metrics may either be for each point supplied
        or averaged over all points.

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
        individual_metrics : bool, default=False
            Whether to additionally return individual metrics for each
            condition.


        Returns
        -------

        torch.Tensor, shape scalar or (n_batch, n_steps)
            The combined metric for the Helmholtz conditions.
        dict[str, torch.Tensor], optional, shapes scalar or
        (n_batch, n_steps)
            Individual metrics for the Helmholtz conditions. Returned
            only if `individual_metrics` is True.
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
    The metric is constructed by a sum of the squared differences of
    both hand sides of the Helmholtz conditions, relative to the L(-2)
    operator norm of g. This metric can act as a loss function for
    training the neural network for g. The L(-2) normalization ensures
    appropriate non-singularity of g and a certain scale invariance.

    Additional Methods
    ------------------

    evaluate_g(t, x, xdot)
        Directly evaluate the g matrix of the Helmholtz conditions.

    References
    ----------

    Douglas J. (1939). Solution of the Inverse Problem of the Calculus of Variations. Proceedings of the National Academy of Sciences of the United States of America, 25(12), 631-637. https://doi.org/10.1073/pnas.25.12.631
    """

    def __init__(
        self,
        neural_network: nn.Module,
        h_2_weight: float = 1.0,
        h_3_weight: float = 1.0,
        supress_time_dependence: bool = False,
        metric_clip_func: Callable = None,
    ) -> None:
        """
        Initializes the metric.

        Parameters
        ----------

        neural_network : nn.Module
            The neural network to learn the g matrix. Must map 2n_dim+1
            to n_dim(n_dim+1)/2, where n_dim is the dimension of
            the state x. The output is scaled by a hyperbolic sine.
        h_2_weight : float, default=1.0
            The relative weight for the second Helmholtz condition in the loss (see notes).
        h_3_weight : float, default=1.0
            The relative weight for the third Helmholtz condition in the loss (see notes).
        supress_time_dependence : bool, default=False
            If True, the time dependence is supressed, i.e. g does not
            explicitly depend on t. This is done by setting t=0 in the
            neural network input.
        metric_clip_func : Callable, optional
            A function to clip each metric with. None (default) means 10*tanh(x/10).
            Useful to prevent exploding values.


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
        self._time_factor = float(not supress_time_dependence)
        self._clip_func = metric_clip_func
        if self._clip_func is None:
            self._clip_func = lambda x: 10 * torch.tanh(x / 10)

    def forward(
        self,
        f: Callable,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for a given second order ODE at given points in time and state.
        Depending on its arguments, it returns a single metric for all
        conditions combined and optionally, individual metrics for each
        condition. These metrics may either be for each point supplied
        or averaged over all points.

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
        individual_metrics : bool, default=False
            Whether to additionally return individual metrics for each
            condition.

        Returns
        -------

        torch.Tensor, shape scalar or (n_batch, n_steps)
            The combined metric for the Helmholtz conditions.
        dict[str, torch.Tensor], optional, shapes scalar or
        (n_batch, n_steps)
            Individual metrics for the Helmholtz conditions. Returned
            only if `individual_metrics` is True. Contains the keys
            "H1", "H2" and "H3".

        Notes
        -----

        The order of the Helmholtz conditions is as usual
        (see `__init__`).

        Internaly torch.func is used, so f should be side-effect free.
        See [1]_ for more information.

        References
        ----------

        .. [1] PyTorch documentation: UX Limitations https://pytorch.org/docs/stable/func.ux_limitations.html
        """
        helmholtz_1, helmholtz_2, helmholtz_3 = self._evaluate_helmholtz_conditions(
            f, t, x, xdot, scalar
        )

        metric = helmholtz_1 + self._h_2_weight * helmholtz_2 + self._h_3_weight * helmholtz_3

        if not individual_metrics:
            return metric

        # Maybe return weighted individual metrics?
        individual_metrics = {
            "H1": helmholtz_1,
            "H2": helmholtz_2,
            "H3": helmholtz_3,
        }

        return metric, individual_metrics

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return next(self._neural_network.parameters()).device

    def evaluate_g(self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor) -> torch.Tensor:
        """
        Directly evaluate the g matrix of the Helmholtz conditions.
        This is not needed for the using the metric, but can provide
        additional insights.
        The result is detached from the computation graph.

        Parameters
        ----------

        t : torch.Tensor, shape (n_batch, n_steps)
            The time points at which the Helmholtz conditions should be evaluated.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time steps t.

        Returns
        -------

        torch.Tensor, shape (n_batch, n_steps, n_dim, n_dim)
            The g matrix evaluated at the given points.
        """
        # This could also be done in the forward method, by allowing
        # an option that also outputs the g (as non-scalar).
        # That would be more efficient, if the condition is needed
        # anyways. But the overhead is minimal here.
        *_, n_dim = x.shape
        with torch.no_grad():
            g_vector = self._evaluate_network(t, x, xdot)
            g = self._assemble_symmetric(g_vector, n_dim)

        return g

    def _evaluate_helmholtz_conditions(
        self,
        f: Callable,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the Helmholtz conditions for a given second order ODE
        at given points in time and state. Returns the square of the
        difference of left and right hand side of the Helmholtz
        conditions, relative to the L(-2) operator norm of the g
        matrix. Averaged according to `scalar`.

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
            The squared difference of the left and right hand side of
            the first Helmholtz condition.
        torch.Tensor, shape (n_batch, n_steps) or scalar
            The squared difference of the left and right hand side of
            the second Helmholtz condition.
        torch.Tensor, shape (n_batch, n_steps) or scalar
            The squared difference of the left and right hand side of
            the third Helmholtz condition.

        Notes
        -----

        Internaly torch.func is used, so f should be side-effect free.
        See [1]_ for more information.

        References
        ----------

        .. [1] PyTorch documentation: UX Limitations https://pytorch.org/docs/stable/func.ux_limitations.html
        """
        # Computing jacobians etc. with torch.func does not work in inference mode.
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Computing Helmholtz conditions in" "inference mode is not supported."
            )

        f_values, jacobian_x, jacobian_v, total_derivative_of_jacobian_v = self._compute_f_terms(
            f, t, x, xdot
        )
        # shapes: f: (n_batch, n_step, n_dim), rest: (n_batch, n_step, n_dim, n_dim)
        g, total_derivative_of_g, g_jacobian_v = self._compute_g_terms(t, x, xdot, f_values)
        # shapes: g and dg/dt: (n_batch, n_steps, n_dim, n_dim),
        # jacobian: (n_batch, n_steps, n_dim, n_dim, n_dim),

        # H1: gPhi - (gPhi)^T
        jacobian_v_squared = torch.einsum("abik,abkj->abij", jacobian_v, jacobian_v)
        Phi = 0.5 * total_derivative_of_jacobian_v - jacobian_x - 0.25 * jacobian_v_squared

        gPhi = torch.einsum("abij,abjk->abik", g, Phi)
        helmholtz_1 = gPhi - gPhi.transpose(-2, -1)

        # H2: dg/dt + 1/2df/dv^T g + 1/2 df/dv g
        jacobian_v_times_g = 0.5 * torch.einsum("abki,abkj->abij", jacobian_v, g)
        helmholtz_2 = (
            total_derivative_of_g + jacobian_v_times_g + jacobian_v_times_g.transpose(-2, -1)
        )

        # H3: dg/dv - dg/dv^T
        helmholtz_3 = g_jacobian_v - g_jacobian_v.transpose(-2, -1)

        # The metrics for the Helmholtz conditions are defined relative
        # to the -2 (operator) norm of the g matrix.
        
        # TODO: Use torch.linalg.eigvalsh and an analytical formula (2D)
        # to get the smallest absolute eigenvalue of g more efficiently.
        epsilon = 1e-6
        inverse_norm_g = 1 / (torch.linalg.matrix_norm(g, ord=-2) + epsilon)

        helmholtz_1 = torch.einsum("abij,ab->abij", helmholtz_1, inverse_norm_g)
        helmholtz_2 = torch.einsum("abij,ab->abij", helmholtz_2, inverse_norm_g)
        helmholtz_3 = torch.einsum("abijk,ab->abijk", helmholtz_3, inverse_norm_g)
        # Clip to prevent exploding values
        helmholtz_1 = self._clip_func(helmholtz_1)
        helmholtz_2 = self._clip_func(helmholtz_2)
        helmholtz_3 = self._clip_func(helmholtz_3)

        # Take mean according to "scalar"
        if scalar:
            dim_h1h2 = (0, 1, 2, 3)
            dim_h3 = (0, 1, 2, 3, 4)
        else:
            dim_h1h2 = (2, 3)
            dim_h3 = (2, 3, 4)

        helmholtz_1 = torch.mean(helmholtz_1**2, dim=dim_h1h2)
        helmholtz_2 = torch.mean(helmholtz_2**2, dim=dim_h1h2)
        helmholtz_3 = torch.mean(helmholtz_3**2, dim=dim_h3)

        return helmholtz_1, helmholtz_2, helmholtz_3

    def _compute_f_terms(
        self, f: Callable, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Returns (df/dx, df/dv), f:
        fn = torch.func.jacrev(fn, argnums=(1, 2), has_aux=True)

        def double_result2(t, x, v):
            # Again Double result for getting double jacobians and single ones and result.
            jacobians, result = fn(t, x, v)
            jacobian_v = jacobians[1]
            return jacobian_v, (jacobians, result)

        # Returns (d^2d/dtdv (partial), d^2f/dxdv, d^2f/dv^2), ((df/dx, df/dv), f):
        fn2 = torch.func.jacrev(double_result2, argnums=(0, 1, 2), has_aux=True)
        # vmap over first two dims as they are batch dims
        # otherwise jacrev would compute the jacobian also w.r.t. the batch dims
        fn2 = torch.func.vmap(fn2, in_dims=0, out_dims=0)
        fn2 = torch.func.vmap(fn2, in_dims=0, out_dims=0)

        double_jacobians, (jacobians, f_values) = fn2(t, x, xdot)
        jacobian_tv, jacobian_xv, jacobian_vv = double_jacobians
        jacobian_x, jacobian_v = jacobians

        total_derivative_of_jacobian_v = self._compute_total_derivative(
            jacobian_tv, jacobian_xv, jacobian_vv, xdot, f_values
        )

        return f_values, jacobian_x, jacobian_v, total_derivative_of_jacobian_v

    def _compute_g_terms(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor, xdotdot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the g matrix and its derivatives.
        # Uses only a single autograd pass.

        # NOTE: G must be symmetric and non-singular.
        # Doing so by construction is possible (e.g., spectral decomposition
        # combined with matrix exponentials and canonical form of O(n)
        # matrices), but usually requires a mapping R -> {-1,1}, which is not
        # really compatible with gradient descent. Also one may get singular/
        # ill-conditioned matrices.
        #
        # Therefore, symmetry is by construction and non-singularity via
        # appropriate scaling of the loss.
        #
        # Thus, the neural network outputs the number of independent parameters
        # of a symmetric matrix and it is then assembled for computation.
        #
        # For a slightly more efficient computation, all derivatives
        # are computed only of the independent parameters, not the full
        # matrix.

        def double_result(t, x, v):
            # Return double result for keeping jacobian and value.
            result = self._evaluate_network(t, x, v)
            return result, result

        fn = restore_dims_from_vmap(double_result, (0, 0))
        # Returns (dg/dt (partial), dg/dx, dg/dv), g:
        fn = torch.func.jacrev(fn, argnums=(0, 1, 2), has_aux=True)
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

        total_derivative_of_g = self._compute_total_derivative(
            jacobian_t, jacobian_x, jacobian_v, xdot, xdotdot
        )

        return g, total_derivative_of_g, jacobian_v

    def _evaluate_network(self, t, x, v) -> torch.Tensor:
        # NOTE: The output of the neural network is scaled by a hyperbolic sine.
        # This allows for a better learning of functions like the exponential
        # function, that may result from the Helmholtz conditions
        # (see e.g. H2 with df/dv = const.).
        t = t.unsqueeze(2)
        # Suppress time dependence if configured.
        t = t * self._time_factor
        result = self._neural_network(torch.cat([t, x, v], dim=2))
        result = torch.sinh(result)
        return result

    @staticmethod
    def _assemble_symmetric(vector: torch.Tensor, n_dim: int) -> torch.Tensor:
        # Assemble a symmetric matrix from a vector.

        shape = vector.shape[:-1]

        symmetric_matrix = vector.new_zeros(*shape, n_dim, n_dim)
        triu_indices = torch.triu_indices(n_dim, n_dim)
        symmetric_matrix[..., triu_indices[0], triu_indices[1]] = vector
        symmetric_matrix = symmetric_matrix + symmetric_matrix.transpose(-2, -1)

        return symmetric_matrix

    @staticmethod
    def _compute_total_derivative(
        jacobian_t, jacobian_x, jacobian_xdot, xdot, xdotdot
    ) -> torch.Tensor:
        # Compute total time derivative given the jacobians and the derivatives.
        # Works with f (double jacobians) and g (single jacobians),
        # as they have the same shape. Contracts over last dimension.

        jvp = torch.einsum("abcij,abj->abci", jacobian_x, xdot)
        jvp += torch.einsum("abcij,abj->abci", jacobian_xdot, xdotdot)
        jvp += jacobian_t

        return jvp
