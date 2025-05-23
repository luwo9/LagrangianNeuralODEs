"""
Contains models for lagrangian neural ODEs. These models combine
an integrated second order neural ODE with a Helmholtz metric, to
allow predictions and training of the model.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .helmholtzmetrics import TryLearnDouglas
from .integratedodes import SolvedSecondOrderNeuralODE
from .neural import NeuralNetwork
from .temporal_schedulers import TemporalScheduler


# NOTE: Maybe n_dim is not strictly required below, if e.g. z is
# inferred from x and y. (Where x/xdot is optional)
class LagrangianNeuralODEModel(nn.Module, ABC):
    """
    Abstract class for a Lagrangian neural ODE model. It bundels all
    necessary components and logic to make predictions, update the
    model and evaluate it.

    By calling it (which is calling the `update` method), the model can
    learn a second order ODE, that satisfies the Helmholtz conditions,
    from data. A measure/metric for the fullfillment of the Helmholtz
    conditions is provided. See Notes for more information.

    The model can also evaluate the (integrated) second order ODE.

    Methods
    -------

    update(t, x, xdot, individual_metrics)
        Update the model based on data.

    second_order_function(t, x, xdot)
        Compute the second order derivative.

    predict(t, x_0, xdot_0)
        Predict the state and its derivative at time t.

    error(t, x_pred, xdot_pred, x_true, xdot_true)
        Compute the prediction error/loss.

    helmholtzmetric(t, x, xdot, scalar, individual_metrics)
        Evaluate the metric for the Helmholtz conditions.

    forward(t, x, xdot, individual_metrics)
        See `update`.

    Attributes
    ----------

    device
        The device on which the model is stored.

    Notes
    -----

    This is a nn.Module, and thus has all usual properties of a PyTorch
    module. However, all gradient based optimization is performed
    within the methods itself. All returned errors/losses are thus
    detached from the computation graph.
    """

    @abstractmethod
    def update(
        self,
        t: torch.Tensor,
        x: torch.Tensor | None = None,
        xdot: torch.Tensor | None = None,
        individual_metrics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Update the model based on data of the state and its derivative
        at times t.

        Parameters
        ----------

        t : torch.Tensor, shape (n_steps,)
            The time points.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The (data) states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The (data) derivative of the state at time steps t.
        individual_metrics : bool, default=False
            Whether to additionally return individual metrics for the
            Helmholtz conditions or just the combined metric.

        Returns
        -------

        torch.Tensor, shape scalar
            The combined metric of the Helmholtz conditions.
        torch.Tensor, shape scalar
            The prediction error/loss.
        dict[str, torch.Tensor], optional
            Individual metrics for the Helmholtz conditions. Returned
            only if `individual_metrics` is True.

        Notes
        -----

        Note that to update only a part of the state or its derivative
        may be required. Thus some of the arguments are optional.

        The return values equal the results the methods
        `helmholtzmetric` and `error` would give, where
        `individual_metrics` is passed on to `helmholtzmetric`.

        They are detached from the computation graph.
        """
        pass

    @abstractmethod
    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second order derivative. More precisely, for the
        underlying ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$,
        this function is the $f^\\ast.

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

    def predict(
        self, t: torch.Tensor, x_0: torch.Tensor | None = None, xdot_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the state and its derivative at time t.

        Parameters
        ----------

        t : torch.Tensor, shape (n_steps,)
            The time steps at which the batch of states should be evaluated.
        x_0 : torch.Tensor, shape (n_batch, n_dim), optional
            The initial state.
        xdot_0 : torch.Tensor, shape (n_batch, n_dim), optional
            The initial derivative of the state.

        Returns
        -------

        tuple[torch.Tensor, torch.Tensor], shapes (n_batch, n_steps, n_dim)
            The state and its derivative at time t.

        Notes
        -----

        Some implementations may infer parts of the initial state
        or its derivative.
        """
        with torch.inference_mode():
            result = self._predict(t, x_0, xdot_0)
        return result

    def error(
        self,
        t: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        xdot_pred: torch.Tensor | None = None,
        x_true: torch.Tensor | None = None,
        xdot_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the pediction error/loss of the neural ode model.

        Parameters
        ----------
        t : torch.Tensor, shape (n_steps,)
            The time steps at which the batch of states should be evaluated.
        x_pred : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The predicted states.
        xdot_pred : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The predicted derivatives of the states.
        x_true : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The true states.
        xdot_true : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The true derivatives of the states.

        Returns
        -------

        torch.Tensor, scalar
            The prediction error/loss.

        Notes
        -----

        The error may only be defined on part of the state
        or its derivative, thus some of the arguments are optional.
        """
        with torch.inference_mode():
            result = self._error(t, x_pred, xdot_pred, x_true, xdot_true)
        return result

    def helmholtzmetric(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for the second order ODE at given points in time and state.
        Depending on its arguments, it returns a single metric for all
        conditions combined and optionally individual metrics for each
        condition. These metrics may either be for each point supplied
        or averaged over all points.

        Parameters
        ----------

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
        # Inference mode is likely not supported for gradient based
        # Helmholtz conditions, so use torch.no_grad. Otherwise silent
        # 0-gradients may occur.
        with torch.no_grad():
            result = self._helmholtzmetric(t, x, xdot, scalar, individual_metrics)
        return result

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor | None = None,
        xdot: torch.Tensor | None = None,
        individual_metrics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Updates the model and returns detached loss values. See `update`.
        """
        return self.update(t, x, xdot, individual_metrics)

    @abstractmethod
    def _predict(
        self, t: torch.Tensor, x_0: torch.Tensor | None = None, xdot_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # See predict
        pass

    @abstractmethod
    def _error(
        self,
        t: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        xdot_pred: torch.Tensor | None = None,
        x_true: torch.Tensor | None = None,
        xdot_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # See error
        pass

    @abstractmethod
    def _helmholtzmetric(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # See helmholtzmetric
        pass


class SimultaneousLearnedDouglasOnlyX(LagrangianNeuralODEModel):
    """
    Uses Douglas based Helmholtz conditions, where the g matrix is
    learned. The neural ODE and g matrix are learned simultaneously, by
    minimizing a total loss w.r.t. the parameters of both. The total
    loss is given by the weighted sum of the prediction error and the
    metric of the Helmholtz conditions.
    The derivative of the state is not expected to be supplied.
    Thus the initial value is inferred from the initial state
    and the prediction error is computed only on the state
    as well.
    """

    def __init__(
        self,
        neural_ode: SolvedSecondOrderNeuralODE,
        helmholtz_metric: TryLearnDouglas,
        xdot_0_network: NeuralNetwork,
        common_optimizer: Optimizer,
        helmholtz_weight: float = 1.0,
        x_loss_function: Callable | None = None,
        lr_scheduler: LRScheduler | None = None,
        temporal_scheduler: TemporalScheduler | None = None,
    ) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        neural_ode : SolvedSecondOrderNeuralODE
            The integrated second order neural ODE to represent the
            dynamics. See notes.
        helmholtz_metric : TryLearnDouglas
            The metric to use for the Helmholtz conditions.
        xdot_network : NeuralNetwork
            The neural network that predicts the initial condition of
            the derivative of the state from the initial state. Must
            map n_dim to n_dim.
        common_optimizer : Optimizer
            The optimizer to be used to update the neural ODE, the
            Helmholtz metric and the initial condition network.
            Must be initialized with the parameters of all three.
        helmholtz_weight : float, default=1e2
            The weight of the Helmholtz metric relative to the
            prediction error in the total loss.
        x_loss_function : Callable, optional
            The loss function to use for the prediction error.
            If None, then torch.nn.MSELoss is used.
        lr_scheduler : LRScheduler, optional
            A scheduler to use for the learning rate. Must be initialized
            with the optimizer. It is called each update step with the
            value of the Helmholtz metric as the loss metric. Thus, its
            `.step()` method should expect such a value.
        temporal_scheduler : TemporalScheduler, optional
            A temporal scheduler that determines the fraction of time
            steps to use for updating the neural ODE in each training
            step. If None, then all time steps are used in all steps.

        Notes
        -----

        Since the second order method of the neural ODE is used when
        computing the Helmholtz metric, its requirements must be met.
        See `TryLearnDouglas` documentation for more information.

        All models are moved to be on the device of the neural ODE.
        """
        super().__init__()
        self._n_train_steps = 0

        self._neural_ode = neural_ode
        self._helmholtz_metric = helmholtz_metric
        self._xdot_network = xdot_0_network
        self._optimizer = common_optimizer
        self._helmholtz_weight = helmholtz_weight
        self._x_loss_function = torch.nn.MSELoss() if x_loss_function is None else x_loss_function
        # TODO: Add option what is the metric for the scheduler?
        self._lr_scheduler = lr_scheduler
        self._temporal_scheduler = temporal_scheduler

        # Enforce single device for all components
        device = self._neural_ode.device
        self._helmholtz_metric.to(device)
        self._xdot_network.to(device)

    def update(
        self,
        t: torch.Tensor,
        x: torch.Tensor | None = None,
        xdot: torch.Tensor | None = None,
        individual_metrics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Update the model based on data of the state.

        Parameters
        ----------

        t : torch.Tensor, shape (n_steps,)
            The time points.
        x : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The (data) states at time steps t.
        xdot : torch.Tensor, shape (n_batch, n_steps, n_dim), optional
            The (data) derivative of the state at time steps t.
            Ignored, see notes.
        individual_metrics : bool, default=False
            Whether to additionally return individual metrics for the
            Helmholtz conditions or just the combined metric.

        Returns
        -------

        torch.Tensor, shape scalar
            The combined metric of the Helmholtz conditions.
        torch.Tensor, shape scalar
            The prediction error/loss.
        dict[str, torch.Tensor], optional
            Individual metrics for the Helmholtz conditions. Returned
            only if `individual_metrics` is True.

        Notes
        -----

        The derivative of the state will be ignored by
        `error` and `predict` and thus also here.

        The return values equal the results the methods
        `helmholtzmetric` and `error` would give, where
        `individual_metrics` is passed on to `helmholtzmetric`.

        They are detached from the computation graph.
        """
        t, x, xdot = self._get_time_series_fraction(t, x, xdot)

        x_0 = x[:, 0, :] if x is not None else None
        xdot_0_pred = xdot[:, 0, :] if xdot is not None else None
        x_pred, xdot_pred = self._predict(t, x_0, xdot_0_pred)

        regression_loss = self._error(t, x_pred, xdot_pred, x, xdot)

        n_batch = x.shape[0]

        # Detach, as gradients should only affect f via its explicit
        # appearance in the metric, not implicitly via the prediction
        # of the trajectory.
        helmholtz_metrics = self._helmholtzmetric(
            t.repeat(n_batch, 1),
            x_pred.detach(),
            xdot_pred.detach(),
            individual_metrics=individual_metrics,
        )
        if individual_metrics:
            helmholtz_loss, individual_helmholtz_metrics = helmholtz_metrics
        else:
            helmholtz_loss = helmholtz_metrics

        self._optimizer.zero_grad()
        # Currently helmholtz_loss and regression_loss are
        # backpropagated separately, to allow clipping only the
        # gradients of the Helmholtz metric. Could be changed later
        # for an efficiency gain.
        (self._helmholtz_weight * helmholtz_loss).backward()
        # Clip gradients of the Helmholtz metric for stability
        torch.nn.utils.clip_grad_norm_(self._neural_ode.parameters(), 0.05)
        torch.nn.utils.clip_grad_norm_(self._helmholtz_metric.parameters(), 5)
        regression_loss.backward()
        self._optimizer.step()
        self._n_train_steps += 1

        # Clone should not be necessary
        helmholtz_loss = helmholtz_loss.detach()
        regression_loss = regression_loss.detach()

        # Make a LR scheduler step.
        scheduler_metric = regression_loss if self._helmholtz_weight == 0 else helmholtz_loss
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(scheduler_metric)

        if not individual_metrics:
            return helmholtz_loss, regression_loss

        individual_helmholtz_metrics = {
            k: v.detach() for k, v in individual_helmholtz_metrics.items()
        }
        return helmholtz_loss, regression_loss, individual_helmholtz_metrics

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second order derivative as given by the neural ode.
        More precisely, for the underlying ODE
        $\ddot{x} = f^\\ast(t, x, \dot{x})$, this function is the
        $f^\\ast.

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
        with torch.inference_mode():
            result = self._neural_ode.second_order_function(t, x, xdot)
        return result

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is stored.
        """
        return self._neural_ode.device

    def _predict(
        self, t: torch.Tensor, x_0: torch.Tensor | None = None, xdot_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # See predict
        # Predict initial condition of xdot then neural ode prediction
        xdot_0_pred = self._xdot_network(x_0)
        return self._neural_ode(t, x_0, xdot_0_pred)

    def _error(
        self,
        t: torch.Tensor,
        x_pred: torch.Tensor | None = None,
        xdot_pred: torch.Tensor | None = None,
        x_true: torch.Tensor | None = None,
        xdot_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # See error
        x_loss = self._x_loss_function(x_pred, x_true)
        return x_loss

    def _helmholtzmetric(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        xdot: torch.Tensor,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # See helmholtzmetric
        f = self._neural_ode.second_order_function
        return self._helmholtz_metric.forward(f, t, x, xdot, scalar, individual_metrics)

    def _get_time_series_fraction(
        self, t: torch.Tensor, x: torch.Tensor | None = None, xdot: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._temporal_scheduler is None:
            return t, x, xdot
        # Cut time series after number of time steps given by the
        # temporal scheduler
        ratio_train = self._temporal_scheduler.get_ratio(self._n_train_steps)
        n_train = int(ratio_train * t.shape[0])
        n_train = max(n_train, 1)
        t = t[:n_train]
        x = x[:, :n_train, :] if x is not None else None
        xdot = xdot[:, :n_train, :] if xdot is not None else None
        return t, x, xdot
