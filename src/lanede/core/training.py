"""
This module contains the training logic for Lagrangian Neural ODE models.
"""

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .lanedemodels import LagrangianNeuralODEModel


_VALIDATION_IDENTIFIER = "validation_"


# Simple class to clean up the training loop.
# Not very general, but serves its purpose.
class TrainingInfo:
    """
    A simple class to accumulate training information in the training loop.
    """

    def __init__(self) -> None:
        self._stepwise_info = defaultdict(lambda: [0])
        self._epochwise_info = defaultdict(list)
        self._n_step_epoch = 0

    def update_epoch(self, **kwargs: torch.Tensor) -> None:
        """
        Update with epochwise information.
        For quantities that are not averaged over the epoch.
        Should always be called with the same kwarg names.
        """
        for key, value in kwargs.items():
            self._epochwise_info[key].append(value.item())

    def update_step(self, **kwargs: torch.Tensor) -> None:
        """
        Update with training information of a training step.
        For quantities that are averaged over the epoch.
        Should always be called with the same kwarg names.
        """
        # Could check kwargs for consistency and just append 0
        # to inputs, and initialize self._stepwise_info as
        # defaultdict(list)
        if self._n_step_epoch == 0:
            for value in self._stepwise_info.values():
                value.append(0)

        for key, value in kwargs.items():
            self._stepwise_info[key][-1] += value.item()

        self._n_step_epoch += 1

    def end_epoch(self) -> None:
        """
        End the current epoch. Must be called immediately after the
        last training step/`update_step` call in the epoch.
        """
        # Compute the mean over the epoch now
        for value in self._stepwise_info.values():
            value[-1] /= self._n_step_epoch

        self._n_step_epoch = 0

    def current_info(self) -> dict[str, float]:
        """
        Returns the training information of the current epoch.
        """
        stepwise = {k: v[-1] for k, v in self._stepwise_info.items()}
        epochwise = {k: v[-1] for k, v in self._epochwise_info.items()}
        return stepwise | epochwise

    def numpy_dict(self) -> dict[str, np.ndarray]:
        """
        Convert the training information to numpy arrays.
        """
        out_dict = self._stepwise_info | self._epochwise_info
        return {k: np.array(v) for k, v in out_dict.items()}


def _format_training_info(info: dict[str, float], validation: bool = False) -> str:
    """
    Format the training information for printing.
    """
    loss_info = "  |  ".join(
        f"{k}: {v:.2e}"
        for k, v in info.items()
        if k.startswith(_VALIDATION_IDENTIFIER) == validation
    )
    return loss_info


def train_lagrangian_neural_ode(
    model: LagrangianNeuralODEModel,
    train_data: DataLoader,
    n_epochs: int,
    t_validation: torch.Tensor | None = None,
    x_validation: torch.Tensor | None = None,
    xdot_validation: torch.Tensor | None = None,
    print_every: int = 1,
    validation_every: int = 1,
    out_file: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Trains a Lagrangian Neural ODE model.
    The device is set to the device of the model.

    Parameters
    ----------

    model : LagrangianNeuralODEModel
        The model to train.
    train_data : DataLoader
        The training data. Must return a tuple of the form (t, x, xdot).
        Shapes: t: (n_steps,), x: (n_batch, n_steps, n_dim),
        xdot: (n_batch, n_steps, n_dim). x or xdot can be None,
        as determined by the model.
    n_epochs : int
        The number of epochs to train the model.
    t_validation : torch.Tensor, optional
        The validation time steps. For shape see train_data.
    x_validation : torch.Tensor, optional
        The validation states. For shape see train_data.
    xdot_validation : torch.Tensor, optional
        The validation derivatives. For shape see train_data.
    print_every : int, default=1
        Print training information every `print_every` epochs.
    validation_every : int, default=1
        Validate the model every `validation_every` epochs. Prints
        validation information.
    out_file : str, optional
        If not None, writes the training information to this file
        instead of printing it.

    Returns
    -------
    A dictionary containing the training information.
    It always has the keys:

        helmholtz : np.ndarray
            The training Helmholtz loss.
        error : np.ndarray
            The training prediction error.

    Aswell as the keys of the individual helmholtz metrics (as output
    by `model.update` with `individual_metrics=True`).

    If validation data is provided, the dictionary also has the same
    keys with 'validation_' prepended to them.
    """
    # TODO: Make out_file work
    validate = t_validation is not None

    device = model.device

    info_log = TrainingInfo()

    if validate:
        t_validation_repeated = t_validation.repeat(x_validation.shape[0], 1)

    # For nicer printing
    n_digits = len(str(n_epochs))

    for epoch in range(n_epochs):

        for t, x, xdot in train_data:
            t = t.to(device)
            x = x.to(device) if x is not None else None
            xdot = xdot.to(device) if xdot is not None else None

            helmholtz, error, individual_helmholtz = model.update(
                t, x, xdot, individual_metrics=True
            )

            info_log.update_step(helmholtz=helmholtz, error=error, **individual_helmholtz)

        info_log.end_epoch()

        # Print training information
        if epoch % print_every == 0:
            loss_info = info_log.current_info()
            loss_info = _format_training_info(loss_info)
            print(f"Epoch {epoch:>{n_digits}}:   {loss_info}", file=out_file)

        if validate and epoch % validation_every == 0:
            x_val_0 = x_validation[:, 0, :] if x_validation is not None else None
            xdot_val_0 = xdot_validation[:, 0, :] if xdot_validation is not None else None

            x_val_pred, xdot_val_pred = model.predict(t_validation, x_val_0, xdot_val_0)
            helmholtz_val, individual_helmholtz_val = model.helmholtzmetric(
                t_validation_repeated, x_val_pred, xdot_val_pred, individual_metrics=True
            )
            error_val = model.error(
                t_validation, x_val_pred, xdot_val_pred, x_validation, xdot_validation
            )

            individual_helmholtz_val = {
                (f"{_VALIDATION_IDENTIFIER}{k}"): v for k, v in individual_helmholtz_val.items()
            }
            info_log.update_epoch(
                validation_helmholtz=helmholtz_val,
                validation_error=error_val,
                **individual_helmholtz_val,
            )
            # Print validation information
            loss_info = info_log.current_info()
            loss_info = _format_training_info(loss_info, validation=True)
            print(f"Validation Epoch {epoch:>{n_digits}}:  {loss_info}", file=out_file)

    return info_log.numpy_dict()
