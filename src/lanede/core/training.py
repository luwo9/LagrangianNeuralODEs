"""
This module contains the training logic for Lagrangian Neural ODE models.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .lanedemodels import LagrangianNeuralODEModel


def train_lagrangian_neural_ode(
    model: LagrangianNeuralODEModel,
    train_data: DataLoader,
    n_epochs: int,
    t_validation: torch.Tensor | None = None,
    x_validation: torch.Tensor | None = None,
    xdot_validation: torch.Tensor | None = None,
    print_every: int = 1,
    validation_every: int = 1,
    out_file: str | None = None) -> dict[str, np.ndarray]:
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
        xdot: (n_batch, n_steps, n_dim).
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
    It may have the keys:

        helmholtz : np.ndarray
            The training Helmholtz loss.
        error : np.ndarray
            The training prediction error.
        validation_helmholtz : np.ndarray
            The validation Helmholtz loss.
            (Key only present if validation data is given)
        validation_error : np.ndarray
            The validation prediction error.
            (Key only present if validation data is given)
    """
    # TODO: Make out_file work
    validate = t_validation is not None
    
    device = model.device

    out_dict = {
        "helmholtz": [],
        "error": [],
    }

    if validate:
        out_dict.update({
        "validation_helmholtz": [],
        "validation_error": [],
        })

        t_repeated = t_validation.repeat(x_validation.shape[0], 1)

    for epoch in range(n_epochs):

        n_step = 0

        # Initialize loss as 0 and add to it, to later compute the mean
        for key in out_dict:
            out_dict[key].append(0)

        for t, x, xdot in train_data:
            t = t.to(device)
            x = x.to(device) if x is not None else None
            xdot = xdot.to(device) if xdot is not None else None

            helmholtz, error = model.update(t, x, xdot)

            out_dict["helmholtz"][-1] += helmholtz.item()
            out_dict["error"][-1] += error.item()

            n_step += 1

        # Compute the mean over the epoch now
        out_dict["helmholtz"][-1] /= n_step
        out_dict["error"][-1] /= n_step

        if epoch % print_every == 0:
            print(f"Epoch {epoch} - "
                  f"Helmholtz: {out_dict['helmholtz'][-1]:.4f}, "
                  f"Error: {out_dict['error'][-1]:.4f}", file=out_file)
            
        if validate and epoch % validation_every == 0:
            x_val_0 = x_validation[:, 0, :] if x_validation is not None else None
            xdot_val_0 = xdot_validation[:, 0, :] if xdot_validation is not None else None

            x_val_pred, xdot_val_pred = model.predict(t_validation, x_val_0, xdot_val_0)
            helmholtz_val = model.helmholtzmetric(t_repeated, x_val_pred, xdot_val_pred)
            error_val = model.error(t_validation, x_val_pred, xdot_val_pred,
                                    x_validation, xdot_validation)

            # Since validation is anyways only done per epoch, don't
            # average here
            out_dict["validation_helmholtz"][-1] = helmholtz_val.item()
            out_dict["validation_error"][-1] = error_val.item()

            print(f"Validation Epoch {epoch} - "
                  f"Helmholtz: {out_dict['validation_helmholtz'][-1]:.4f}, "
                  f"Error: {out_dict['validation_error'][-1]:.4f}", file=out_file)

    for key, val in out_dict.items():
        out_dict[key] = np.array(val)

    return out_dict