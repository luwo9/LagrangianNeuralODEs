"""
Provides a single class for using Lagrangian Neural ODEs. This is the
top-level interface, where they can be trained, evaluated, saved and
loaded directly using dedicated methods.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate_fn_map, collate

from .lanedemodels import LagrangianNeuralODEModel
from .normalize import Normalizer
from .training import train_lagrangian_neural_ode


# NOTE: The logic for the data handling is in an intermediate state
# of being public and private. This is as it is not really expected
# that the user will use the internals of LagrangianNeuralODE manually.
# However, if so, the user should be able to use the data handling
# logic.
class OptionalStatesSingleTime(Dataset):
    """
    A Dataset that respects that only a single series of time steps
    is present in the data and that state or its derivative might not
    be supplied. Otherwise like TensorDataset.
    """

    def __init__(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------

        t : np.ndarray, shape (n_steps,)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        """
        self._t = t
        self._x = x
        self._xdot = xdot
        self._length = len(self._x) if self._x is not None else len(self._xdot)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Make a copy to be sure
        t = torch.tensor(self._t, dtype=torch.float32)
        x = torch.tensor(self._x[idx], dtype=torch.float32) if self._x is not None else None
        xdot = torch.tensor(self._xdot[idx], dtype=torch.float32) if self._xdot is not None else None
        return t, x, xdot


def _collate_single_time_and_None(batch):
    # Allows to collate batches where Nones are present like:
    # list[tuple[[torch.Tensor, torch.Tensor, None]]]
    # -> tuple[torch.Tensor, torch.Tensor, None]
    # Simply expands pytorchs collate with a function for None
    # And it does not stack the time steps, as they are the same
    collate_fn_map = default_collate_fn_map.copy()
    collate_fn_map[type(None)] = lambda *args, **kwargs: None
    t, x, xdot = collate(batch, collate_fn_map=collate_fn_map)
    return t[0], x, xdot


def make_dataloader(t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None, *, batch_size: int, **kwargs) -> DataLoader:
    """
    Creates a DataLoader from the given data.
    State or its derivative might not be supplied.

    Parameters
    ----------

    t : np.ndarray, shape (n_steps,)
        Time Steps.
    x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
        State at times t.
    xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
        Derivative of the state at times t.
    batch_size : int
        The batch size to use.
    kwargs
        Additional keyword arguments for the DataLoader.

    Returns
    -------

    DataLoader
        The DataLoader for the data.
    """
    dataset = OptionalStatesSingleTime(t, x, xdot)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_single_time_and_None, **kwargs)


class LagrangianNeuralODE:
    """
    A class for training and evaluating Lagrangian Neural ODEs.
    Lagraingian Neural ODEs learn a second order neural ODE, satisfying
    the Helmholtz conditions, so that they originate from a Lagrangian.
    A metric for their satisfaction can be evaluated.

    Methods
    -------

    train(t, x=None, xdot=None, n_epochs, batch_size=32, device="cpu",
    **kwargs)
        Trains the model on the given data.
    predict(t, x_0=None, xdot_0=None)
        Predicts the state and its derivative at time t form the given
        initial values using the ODE.
    second_derivative(t, x, xdot)
        Computes the second order derivative of the state at time t.
    helmholtzmetric(t, x, xdot, scalar=True, combined=True)
        Computes the metric of fullfilment of the Helmholtz conditions
        for the second order ODE at given points in time and state.
    error(t, x_pred=None, xdot_pred=None, x_true=None, xdot_true=None)
        Computes the pediction error/loss of the neural ode model.
    state_dict()
        Returns a dictionary containing the state of the
        Lagrangian Neural ODE.
    load_state_dict(state_dict)
        Loads the state of the Lagrangian Neural ODE from the given
        state dictionary.
    """

    def __init__(self, model: LagrangianNeuralODEModel, normalizer: Normalizer) -> None:
        """
        Initializes the model.

        Parameters
        ----------

        model : LagrangianNeuralODEModel
            The underlying Lagrangian Neural ODE model to use.
        normalizer : Normalizer
            A normalizer for the data.
        """
        self._model = model
        self._normalizer = normalizer
        self._normalizer_was_fitted = False

    def train(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None, *, n_epochs: int, batch_size: int = 32, device: torch.device | str = "cpu", **kwargs) -> dict[str, np.ndarray]:
        """
        Trains the model on the given data. Depending on the underlying
        model, this may or may not need both states and their
        derivatives. This will fit the normalizer correspondingly.

        Parameters
        ----------

        t : np.ndarray, shape (n_steps,)
            Time Steps to train on.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        n_epochs : int
            The number of epochs to train the model.
        device : torch.device or str, default="cpu"
            The device on which to train the model.
        kwargs
            Additional keyword arguments for the training function.
            See Notes.

        Returns
        -------

        A dictionary containing the training information. See Notes.

        Notes
        -----

        The `kwargs` are passed to the training function, refer to the
        documentation of for full information. Note that here numpy
        arrays are used, which are converted to torch tensors
        internally. Allowed keyword arguments are:

        t_validation : np.ndarray, optional
            The validation time steps. For shape see train data.
        x_validation : np.ndarray, optional
            The validation states. For shape see train data.
        xdot_validation : np.ndarray, optional
            The validation derivatives. For shape see train data.
        print_every : int, default=1
            Print training information every `print_every` epochs.
        validation_every : int, default=1
            Validate the model every `validation_every` epochs. Prints
            validation information.
        out_file : str, optional
            If not None, writes the training information to this file
            instead of printing it.

        The returned dictionary is passed on from the training function.
        It contains the keys:

        helmholtz : np.ndarray
            The training Helmholtz loss.
        error : np.ndarray
            The training prediction error.
        validation_helmholtz : np.ndarray
            The validation Helmholtz loss.
        validation_error : np.ndarray
            The validation prediction error.
        """
        # Bring t in shape (n_batch, n_steps) for normalizer
        n_batches = x.shape[0] if x is not None else xdot.shape[0]
        t_with_batches = np.tile(t, (n_batches, 1))
        if not self._normalizer_was_fitted:
            self._normalizer.fit(t_with_batches, x, xdot)
            self._normalizer_was_fitted = True
        
        t_with_batches, x, xdot = self._normalizer.transform(t_with_batches, x, xdot)
        # Bring back to (n_steps,)
        t = t_with_batches[0]
        train_data = make_dataloader(t, x, xdot, batch_size=batch_size)

        # Process validation data accordingly, if present
        validation_keys = ["t_validation", "x_validation", "xdot_validation"]
        t_val, x_val, xdot_val = [kwargs.get(k, None) for k in validation_keys]
        if t_val is not None:
            n_batches = (x_val.shape[0] if x_val is not None else xdot_val.shape[0])
            t_val_with_batches = np.tile(t_val, (n_batches, 1))
            t_val, x_val, xdot_val = self._normalizer.transform(t_val_with_batches, x_val, xdot_val)
            t_val = t_val[0]
            for k, v in zip(validation_keys, [t_val, x_val, xdot_val]):
                if v is None:
                    continue
                kwargs[k] = torch.tensor(v, dtype=torch.float32)

        old_device = self._model.device
        model = self._model.to(device)
        training_info = train_lagrangian_neural_ode(model, train_data, n_epochs, **kwargs)
        self._model = model.to(old_device)

        return training_info
    
    def predict(self, t: np.ndarray, x_0: np.ndarray | None = None, xdot_0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the state and its derivative at time t form the given
        initial values using the ODE.

        Parameters
        ----------

        t : np.ndarray, shape (n_steps,)
            The time steps at which the batch of states should be evaluated.
        x_0 : np.ndarray, shape (n_batch, n_dim), optional
            The initial state.
        xdot_0 : np.ndarray, shape (n_batch, n_dim), optional
            The initial derivative of the state.

        Returns
        -------

        tuple[np.ndarray, np.ndarray], shapes (n_batch, n_steps, n_dim)
            The state and its derivative at time t.

        Notes
        -----

        Depending on the underlying model parts of the initial state
        or its derivative may be iferred and thus not needed.
        Only may be called after calling `train`.
        """
        # To tranform the inital conditions, supply the right time
        # steps for the normalizer
        n_batches = x_0.shape[0] if x_0 is not None else xdot_0.shape[0]
        t_with_batches = np.tile(t, (n_batches, 1))
        # Only the first time step, but keep the shape
        t_0_with_batches = t_with_batches[:, :1]
        x_0 = np.expand_dims(x_0, axis=1) if x_0 is not None else None
        xdot_0 = np.expand_dims(xdot_0, axis=1) if xdot_0 is not None else None

        _, x_0, xdot_0 = self._normalizer.transform(t_0_with_batches, x_0, xdot_0)
        # Tranform time independently
        t_with_batches = self._normalizer.transform(t_with_batches)[0]

        x_0 = torch.tensor(x_0, dtype=torch.float32).squeeze(1) if x_0 is not None else None
        xdot_0 = (torch.tensor(xdot_0, dtype=torch.float32).squeeze(1)
                  if xdot_0 is not None else None)
        t_with_batches = torch.tensor(t_with_batches, dtype=torch.float32)

        x, xdot = self._model.predict(t_with_batches[0], x_0, xdot_0)

        x = x.detach().numpy()
        xdot = xdot.detach().numpy()
        t_with_batches = t_with_batches.detach().numpy()

        _, x, xdot = self._normalizer.inverse_transform(t_with_batches, x, xdot)

        return x, xdot
    
    def second_derivative(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        """
        Computes the second order derivative of the state at time t.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The time at which the state is evaluated.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            The state at time t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time t.

        Returns
        -------

        np.ndarray, shape (n_batch, n_steps, n_dim)
            The second derivative of the state at time t.

        Notes
        -----

        Only may be called after calling `train`.
        """
        t, x, xdot = self._normalizer.transform(t, x, xdot)
        t = torch.tensor(t, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        xdot = torch.tensor(xdot, dtype=torch.float32)

        xdotdot = self._model.second_order_function(t, x, xdot)
        xdotdot = xdotdot.detach().numpy()

        xdotdot = self._normalizer.inverse_transform(t, x, xdot, xdotdot)[3]

        return xdotdot
    
    def helmholtzmetric(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray, scalar: bool = True, combined: bool = True) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for the second order ODE at given points in time and state.
        Depending on its arguments, it returns either a single metric
        for all conditions combined or individual metrics for each
        condition. These may either be for each point supplied or
        averaged over all points.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The time points at which the Helmholtz conditions should be evaluated.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            The states at time steps t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time steps t.
        scalar : bool, default=True
            Whether to return a single scalar metric or a pointwise
            result.
        combined : bool, default=True
            Whether to return a single metric for all conditions
            combined or individual metrics for each condition.


        Returns
        -------

        np.ndarray or tuple[np.ndarray, ...], shape (n_batch, n_steps) or scalar
            The metric for the Helmholtz conditions.

        Notes
        -----

        Only may be called after calling `train`.

        This method essentially wraps the `helmholtzmetric` method of
        the underlying model. However, note that since data is
        normalized, before training, it is also normalized here.
        This will induce a normalization of the metric as well.
        However, this normalization is not undone, sucht that the
        metric values might not be directly physically interpretable,
        but only provide a metric.
        """
        t, x, xdot = self._normalizer.transform(t, x, xdot)
        t = torch.tensor(t, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        xdot = torch.tensor(xdot, dtype=torch.float32)

        metric = self._model.helmholtzmetric(t, x, xdot, scalar=scalar, combined=combined)

        if scalar:
            return metric.detach().numpy()
        
        return tuple(metric_.detach().numpy() for metric_ in metric)

    def error(self, t: np.ndarray, x_pred: np.ndarray | None = None, xdot_pred: np.ndarray | None = None, x_true: np.ndarray | None = None, xdot_true: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the pediction error/loss of the neural ode model.

        Parameters
        ----------
        t : np.ndarray, shape (n_steps,)
            The time steps at which the batch of states should be evaluated.
        x_pred : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            The predicted states.
        xdot_pred : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            The predicted derivatives of the states.
        x_true : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            The true states.
        xdot_true : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            The true derivatives of the states.

        Returns
        -------

        np.ndarray, scalar
            The prediction error/loss.

        Notes
        -----

        Which of the states and derivatives are needed depends on the
        underlying model. Only may be called after calling `train`.
        """
        # Transform the data and convert to torch tensors
        n_batches = x_true.shape[0] if x_true is not None else xdot_true.shape[0]
        t_with_batches = np.tile(t, (n_batches, 1))
        _, x_pred, xdot_pred = self._normalizer.transform(t_with_batches, x_pred, xdot_pred)
        transformed = self._normalizer.transform(t_with_batches, x_true, xdot_true)
        t_with_batches, x_true, xdot_true = transformed
        t = t_with_batches[0]

        t = torch.tensor(t, dtype=torch.float32)
        x_pred = torch.tensor(x_pred, dtype=torch.float32) if x_pred is not None else None
        xdot_pred = (torch.tensor(xdot_pred, dtype=torch.float32)
                     if xdot_pred is not None else None)
        x_true = torch.tensor(x_true, dtype=torch.float32) if x_true is not None else None
        xdot_true = (torch.tensor(xdot_true, dtype=torch.float32)
                     if xdot_true is not None else None)

        error = self._model.error(t, x_pred, xdot_pred, x_true, xdot_true)
        error = error.detach().numpy()

        return error
    
    def state_dict(self) -> dict:
        """
        Returns a dictionary containing the state of the
        Lagrangian Neural ODE. Works like pytorchs `state_dict`.

        Returns
        -------

        dict
            The state dict of the model.
        """
        state_dict = {
            "model": self._model.state_dict(),
            "normalizer": self._normalizer.state_dict(),
            "normalizer_was_fitted": self._normalizer_was_fitted
        }

        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the state of the Lagrangian Neural ODE from the given
        state dictionary. Works like pytorchs `load_state_dict`.

        Parameters
        ----------

        state_dict : dict
            The state dictionary to load.
        """
        self._model.load_state_dict(state_dict["model"])
        self._normalizer.load_state_dict(state_dict["normalizer"])
        self._normalizer_was_fitted = state_dict["normalizer_was_fitted"]