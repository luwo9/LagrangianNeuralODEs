"""
Provides a top-level API class for Lagrangian Neural ODEs.
Allows easily defining, training, evaluating, loading and saving
models.

Simmilar to `lanede.core.LagrangianNeuralODE`, but allows easily creating models
from configuration dictionaries as well as saving and loading them.
"""

from __future__ import annotations
from collections.abc import Callable
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from lanede.core import LagrangianNeuralODE
from ._lanede_builders import (
    JSONDict,
    example_simple_douglas_only_x,
    simple_douglas_only_x,
    example_simple_LNN_only_x,
    simple_LNN_only_x,
)

# NOTE: Move this to a separate module if it gets too big:

# Map names to the corresponding builder functions and
# example configurations
_BUILDERS: dict[str, Callable[[JSONDict], LagrangianNeuralODE]] = {
    "simple_douglas": simple_douglas_only_x,
    "simple_LNN": simple_LNN_only_x,
}

EXAMPLES: dict[str, JSONDict] = {
    "simple_douglas": example_simple_douglas_only_x,
    "simple_LNN": example_simple_LNN_only_x,
}


class LanedeAPI:
    """
    API for creating, training, evaluating, saving and loading
    Lagrangian Neural ODE models.

    To use a model, either create it by initializing this class with
    a configuration dictionary, or use the `load` classmethod to load a
    saved model.

    Attributes
    ----------

    losses : list[dict[str, np.ndarray]]
        The losses of the model during training.

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
    helmholtzmetric(t, x, xdot, scalar=True, individual_metrics=False)
        Computes the metric of fullfilment of the Helmholtz conditions
        for the second order ODE at given points in time and state.
    error(t, x_pred=None, xdot_pred=None, x_true=None, xdot_true=None)
        Computes the pediction error/loss of the neural ode model.
    save(path, losses=True)
        Saves the model to the given path.
    load(path)
        Loads a model from the given path.
    """

    def __init__(self, preset_name: str, preset_config: JSONDict):
        """
        Initialize the API with a preset model and its configuration
        dictionary.

        Parameters
        ----------

        preset_name : str
            Name of the preset model to use. See below for available
            presets.
        preset_config : JSONDict
            Configuration dictionary for the preset model. See below
            for details.

        Presets
        -------

        Available presets are: `simple_douglas` and `simple_LNN`.

        Fore more information on the pre defined models and their
        configuration dictionaries, see their documentation. For every
        preset, there is a corresponding example configuration
        dictionary: `lanede.api.EXAMPLES["<preset_name>"]`.
        """
        self._preset_name = preset_name
        self._config = preset_config
        self._model = _BUILDERS[preset_name](preset_config)
        self._losses = []

    def train(
        self,
        t: np.ndarray,
        x: np.ndarray | None = None,
        xdot: np.ndarray | None = None,
        *,
        n_epochs: int,
        batch_size: int = 32,
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> dict[str, np.ndarray]:
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
        It always has the keys:

        helmholtz : np.ndarray
            The training Helmholtz loss.
        error : np.ndarray
            The training prediction error.

        Aswell as the keys of the individual helmholtz metrics (see the
        `helmholtzmetric` method with `individual_metrics=True`).

        If validation data is provided, the dictionary also has the same
        keys with 'validation_' prepended to them.
        """
        metrics = self._model.train(
            t, x, xdot, n_epochs=n_epochs, batch_size=batch_size, device=device, **kwargs
        )
        self._losses.append(metrics)
        return metrics

    def predict(
        self, t: np.ndarray, x_0: np.ndarray | None = None, xdot_0: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return self._model.predict(t, x_0, xdot_0)

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
        return self._model.second_derivative(t, x, xdot)

    def helmholtzmetric(
        self,
        t: np.ndarray,
        x: np.ndarray,
        xdot: np.ndarray,
        scalar: bool = True,
        individual_metrics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Computes the metric of fullfilment of the Helmholtz conditions
        for the second order ODE at given points in time and state.
        Depending on its arguments, it returns a single metric for all
        conditions combined and optionally individual metrics for each
        condition. These metrics may either be for each point supplied
        or averaged over all points.

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
        individual_metrics : bool, default=False
            Whether to additionally return individual metrics for each
            condition.


        Returns
        -------

        np.ndarray, shape scalar or (n_batch, n_steps)
            The combined metric for the Helmholtz conditions.
        dict[str, np.ndarray], optional, shapes scalar or
        (n_batch, n_steps)
            Individual metrics for the Helmholtz conditions. Returned
            only if `individual_metrics` is True.

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
        return self._model.helmholtzmetric(
            t, x, xdot, scalar=scalar, individual_metrics=individual_metrics
        )

    def error(
        self,
        t: np.ndarray,
        x_pred: np.ndarray | None = None,
        xdot_pred: np.ndarray | None = None,
        x_true: np.ndarray | None = None,
        xdot_true: np.ndarray | None = None,
    ) -> np.ndarray:
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
        return self._model.error(t, x_pred, xdot_pred, x_true, xdot_true)

    @property
    def losses(self) -> list[dict[str, np.ndarray]]:
        """
        The losses of the model during training. Each entry in the list
        is a dictionary containing the losses of one training run.
        See `save` for an important remark.

        Returns
        -------

        list[dict[str, np.ndarray]]
            The losses of the model during training.
        """
        return deepcopy(self._losses)

    def save(self, path: str, losses: bool = True):
        """
        Saves the model to the given path. Depending on the specified
        path, it is either saved as a single file or as a directory.
        See Notes for details.

        Parameters
        ----------

        path : str
            The path to which the model should be saved. If it ends
            with `.pt`, it is saved as a single file. Otherwise, it is
            saved as a directory.
        losses : bool, default=True
            Whether to save the losses as well. See Notes.

        Notes
        -----

        If saving as a directory, the following structure is used:

        path/
            config.json
            model.pt
            losses/
                loss_<i>_<name>.txt

        Where `config.json` contains the configuration dictionary and
        the preset name, `model.pt` contains the model itself and
        `losses/` contains the losses. The losses are saved as text
        for readability. The name refers to the loss name
        (see `train`). The index `i` refers to the number of training
        runs logged. Note that when saving with `losses=False`, the
        losses are not saved and a later training and saving will
        only show the new losses starting from 0 again.
        """
        if path.endswith(".pt"):
            self._single_save(path, losses=losses)
        else:
            self._dir_save(path, losses=losses)

    def _single_save(self, path: str, losses: bool = True):
        # Convert to dict, that can be saved and loaded with pytorch
        # even when using weights_only=True in torch.load

        save_dict = {
            "preset_name": self._preset_name,
            "config": self._config,
            "model": self._model.state_dict(),
        }

        if losses:
            # Convert numpy arrays to lists
            list_losses = [{k: v.tolist() for k, v in loss.items()} for loss in self._losses]
            save_dict["losses"] = list_losses

        torch.save(save_dict, path)

    def _dir_save(self, path: str, losses: bool = True):
        # Save the model as a directory
        # Structure:
        # path/
        #   config.json
        #   model.pt
        #   losses/
        #     loss_<i>_<name>.txt

        # Make the directory. If it exists, don't clear it but proceed with
        # saving the model.
        path: Path = Path(path)
        path.mkdir(exist_ok=True)

        # Save all the components
        config_with_name = {
            "preset_name": self._preset_name,
            "config": self._config,
        }
        config_path = path / "config.json"
        with config_path.open("w") as f:
            json.dump(config_with_name, f, indent=4)

        torch.save(self._model.state_dict(), path / "model.pt")

        if losses:
            losses_dir = path / "losses"
            losses_dir.mkdir(exist_ok=True)
            for i, loss in enumerate(self._losses):
                for name, values in loss.items():
                    np.savetxt(losses_dir / f"loss_{i}_{name}.txt", values)

    @classmethod
    def load(cls, path: str) -> LanedeAPI:
        """
        Loads a model from the given path. Depending on the specified
        path, it is either loaded from a single file or from a directory.
        See `save` for details.

        Parameters
        ----------

        path : str
            The path from which the model should be loaded. If it ends
            with `.pt`, it is loaded as a single file. Otherwise, it is
            loaded as a directory.

        Returns
        -------

        LanedeAPI
            The loaded model.
        """
        if path.endswith(".pt"):
            return cls._single_load(path)

        return cls._dir_load(path)

    @classmethod
    def _single_load(cls, path: str) -> LanedeAPI:
        # Inverse of _single_save
        save_dict = torch.load(path, weights_only=True, map_location="cpu")

        preset_name = save_dict["preset_name"]
        config = save_dict["config"]

        model = _BUILDERS[preset_name](config)
        model.load_state_dict(save_dict["model"])

        api = cls(preset_name, config)
        api._model = model

        if "losses" in save_dict:
            api._losses = [
                {k: np.array(v) for k, v in loss.items()} for loss in save_dict["losses"]
            ]

        return api

    @classmethod
    def _dir_load(cls, path: str) -> LanedeAPI:
        # Inverse of _dir_save
        path: Path = Path(path)

        config_path = path / "config.json"
        with config_path.open() as f:
            config_with_name = json.load(f)

        preset_name = config_with_name["preset_name"]
        config = config_with_name["config"]

        model = _BUILDERS[preset_name](config)
        model_dict = torch.load(path / "model.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(model_dict)

        api = cls(preset_name, config)
        api._model = model

        losses_dir = path / "losses"
        if losses_dir.exists():
            losses = cls._load_losses(losses_dir)
            api._losses = losses

        return api

    @staticmethod
    def _load_losses(losses_dir: Path) -> list[dict[str, np.ndarray]]:
        # Get all loss files and sort them by "i" in the filename
        loss_files = losses_dir.glob("loss_*.txt")
        loss_files = sorted(loss_files, key=lambda p: int(p.stem.split("_")[1]))
        losses = []
        for loss_file in loss_files:
            i, name = loss_file.stem.split("_", maxsplit=2)[1:]
            i = int(i)
            if i >= len(losses):
                losses.append({})
            losses[i][name] = np.loadtxt(loss_file)

        return losses
