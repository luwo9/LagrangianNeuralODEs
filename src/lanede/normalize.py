"""
This module contains functions for transforming data to be
usable for neural networks.
"""

# Could use typing.Self, but make sure it works with python 3.10
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Normalizer(ABC):
    """
    A normalizer for data. API is chosen to be similar to scikit-learn.
    Note that all except the last dimension of the input data are
    considered to be batch dimensions. The last dimension is considered
    to be the feature dimension.

    Methods
    -------

    fit(t, x=None, xdot=None)
        Fit the normalizer to the data.
    
    transform(t, x=None, xdot=None)
        Transform the data.
    
    inverse_transform(t, x=None, xdot=None)
        Inverse transform the data.

    state_dict()
        Returns the state of the normalizer as a dictionary.

    load_state_dict(state_dict)
        Loads the normalizer state from the dictionary.
    """ 
    
    @abstractmethod
    def fit(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> Normalizer:
        """
        Fit the normalizer to the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.

        Returns
        -------

        self : Normalizer
            The fitted normalizer.
        """
        pass

    @abstractmethod
    def transform(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """
        Transform the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        
        Returns
        -------

        Tuple with transformed data. Returns only transformed versions
        of non-None inputs, depending on the input:

        t : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.
        """
        pass

    @abstractmethod
    def inverse_transform(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """
        Inverse transform the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        
        Returns
        -------

        Tuple with transformed data. Returns only transformed versions
        of non-None inputs, depending on the input:

        t : np.ndarray, shape (n_batch, n_steps)
            Inverse transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed derivative of the state at times t.
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Returns the state of the normalizer as a dictionary.
        
        Returns
        -------
        dict
            The state of the normalizer.
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the normalizer state from the dictionary.
        
        Parameters
        ----------
        state_dict : dict
            The state of the normalizer.
        """
        pass


class MeanStdNormalizer(Normalizer):
    """
    A normalizer that normalizes the data by subtracting the mean
    and dividing by the standard deviation. The mean and standard
    deviation are computed from the data.

    Attributes
    ----------
    mean : np.ndarray
        The mean of the data.
    std : np.ndarray
        The standard deviation of the data.
    """
    
    def __init__(self) -> None:
        self._names = ["t", "x", "xdot"]
        self._reset()

    def fit(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> MeanStdNormalizer:
        """
        Fit the normalizer to the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.

        Returns
        -------

        self : MeanStdNormalizer
            The fitted normalizer.
        """
        self._reset()
        data_types = [t, x, xdot]
        for name, data in zip(self._names, data_types):
            if data is None:
                continue
            self._means[name] = np.mean(data, axis=(0, 1))
            self._stds[name] = np.std(data, axis=(0, 1))

        return self

    def transform(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        
        Returns
        -------
        
        Tuple with transformed data. Returns only transformed versions
        of non-None inputs, depending on the input:

        t : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.
        """
        transformed = []
        for name, data in zip(self._names, [t, x, xdot]):
            if data is None:
                continue
            self._assert_transformable(name)

            transformed_data = (data - self._means[name]) / self._stds[name]
            transformed.append(transformed_data)

        return tuple(transformed)
    
    def inverse_transform(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse transform the data.
        
        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        
        Returns
        -------
        
        Tuple with transformed data. Returns only transformed versions
        of non-None inputs, depending on the input:

        t : np.ndarray, shape (n_batch, n_steps)
            Inverse transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed derivative of the state at times t.
        """
        transformed = []
        for name, data in zip(self._names, [t, x, xdot]):
            if data is None:
                continue
            self._assert_transformable(name)

            transformed_data = data * self._stds[name] + self._means[name]
            transformed.append(transformed_data)

        return tuple(transformed)
    
    def state_dict(self) -> dict:
        """
        Returns the state of the normalizer as a dictionary.
        
        Returns
        -------
        dict
            The state of the normalizer.
        """
        return {
            "means": self._means.tolist(),
            "stds": self._stds.tolist()
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the normalizer state from the dictionary.
        
        Parameters
        ----------
        state_dict : dict
            The state of the normalizer.
        """
        self._means = np.array(state_dict["means"])
        self._stds = np.array(state_dict["stds"])

    def _reset(self) -> None:
        self._means = {name: None for name in self._names}
        self._stds = {name: None for name in self._names}

    def _assert_transformable(self, name):
        if self._means[name] is None or self._stds[name] is None:
            raise ValueError(f"Normalizer not fitted yet for {name}.")