"""
This module contains functions for transforming data to be
usable for neural networks.
"""

# Could use typing.Self, but make sure it works with python 3.10
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np


class _OperationMode(Enum):
    """
    Enum for the operation mode of the normalizer, given by what is
    supplied to it. See its documentation for more information.
    """
    NOT_SET = auto()
    ONLY_STATE = auto()
    ONLY_DERIVATIVE = auto()
    BOTH = auto()


class Normalizer(ABC):
    """
    A normalizer for data. API is chosen to be similar to scikit-learn.
    Note that n_batch and n_steps are considered batch dimensions. Only
    the other dimensions are considered as feature dimensions. See
    Notes for important information on needing to supply state and/or
    derivative in the methods.
    The `fit` method needs to be called before the `transform` or
    `inverse_transform` methods can be called.

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

    Notes
    -----

    The time must always be supplied. Depending on the use case, the
    normalizer may or may not transform the state and/or the derivative.
    The `fit` method needs to be called with all data that is to be
    transformed later (`transform` method).
    However, the 'inverse_transform' method must be able to transform
    time, state and derivative in every case.

    To achive this several methods must be implemented including the
    derivatives and inverse transformations. This allows that this
    class automatically inferes the missing transformations.

    If both state and derivative are supplied, the direct inverse is
    applied and no interaction is assumed between the state and its
    derivative.

    If only the state is supplied, the derivatives of the
    transformation is used to infer the transformation of the
    derivative via the chain rule.

    If only the derivative of the state is supplied, its transformation
    must be linear or affine in it.
    """ 

    def __init__(self) -> None:
        self._mode = _OperationMode.NOT_SET
        self._transform_functions = {
            _OperationMode.BOTH: self._handle_both,
            _OperationMode.ONLY_STATE: self._handle_state,
            _OperationMode.ONLY_DERIVATIVE: self._handle_derivative
        }
        self._inverse_transform_functions = {
            _OperationMode.BOTH: self._handle_inverse_both,
            _OperationMode.ONLY_STATE: self._handle_inverse_state,
            _OperationMode.ONLY_DERIVATIVE: self._handle_inverse_derivative
        }

    def fit(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> Normalizer:
        """
        Fit the normalizer to the data.
        All data that is to be transformed later must be supplied here.
        
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
        if self._mode != _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is already fitted.")
        if x is None and xdot is None:
            raise ValueError("At least one of x or xdot must be supplied.")
        if x is not None and xdot is not None:
            self._mode = _OperationMode.BOTH
        elif x is not None:
            self._mode = _OperationMode.ONLY_STATE
        else:
            self._mode = _OperationMode.ONLY_DERIVATIVE

        return self._fit(t, x, xdot)

    def transform(self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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

        Tuple with transformed data. Returns None for the respective
        data that was not supplied.

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Transformed derivative of the state at times t.
        """
        if self._mode == _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is not fitted.")
        if x is None and xdot is None:
            raise ValueError("At least one of x or xdot must be supplied.")
        
        return self._transform_functions[self._mode](t, x, xdot)

    def inverse_transform(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse transform the data.
        
        Parameters
        ----------
        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Derivative of the state at times t.
        
        Returns
        -------

        Tuple with transformed data.

        t : np.ndarray, shape (n_batch, n_steps)
            Inverse transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed derivative of the state at times t.
        """
        if self._mode == _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is not fitted.")
        
        return (self._inverse_transform_functions[self._mode]
                (t_transformed, x_transformed, xdot_transformed))
    
    @abstractmethod
    def _fit(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> Normalizer:
        # See fit.
        pass

    @abstractmethod
    def _transform_both(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the state and its derivative.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Derivative of the state at times t.

        Returns
        -------

        Tuple with transformed data.

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.
        """
        pass

    @abstractmethod
    def _inverse_transform_both(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse of `_transform_both`.
        """
        pass

    @abstractmethod
    def _transform_state(self, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform the state.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.

        Returns
        -------

        Tuple with transformed data.

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        """
        pass

    @abstractmethod
    def _inverse_transform_state(self, t_transformed: np.ndarray, x_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse of `_transform_state`.
        """
        pass

    @abstractmethod
    def _jacobian_inverse_transform_state(self, t_transformed: np.ndarray, x_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Jacobian of `_inverse_transform_state`.

        Parameters
        ----------

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.

        Returns
        -------

        Jacobian of the inverse transformation of the state
        $(t, x)=(f_t, f_x)(t_trf, x_trf)$, flattened.

        df_t_dt_trf : np.ndarray, shape (n_batch, n_steps)
            Partial time derivative of the inverse transformation of
            the time.
        df_t_dx_trf : np.ndarray, shape (n_batch, n_steps, n_dim)
            Partial state derivative of the inverse transformation of
            the time.
        df_x_dt_trf : np.ndarray, shape (n_batch, n_steps, n_dim)
            Partial time derivative of the inverse transformation of
            the state.
        df_x_dx_trf : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            Partial state derivative of the inverse transformation of
            the state.
        """
        pass

    def _handle_both(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # See transform and _transform_both.
        if x is None or xdot is None:
            raise ValueError("Both state and derivative must be supplied.")
        return self._transform_both(t, x, xdot)
    
    def _handle_inverse_both(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # See inverse_transform and _inverse_transform_both.
        return self._inverse_transform_both(t_transformed, x_transformed, xdot_transformed)
    
    def _handle_state(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # See transform and _transform_state.
        if x is None:
            raise ValueError("State must be supplied.")
        if xdot is not None:
            raise ValueError("Only state must be supplied.")
        return *self._transform_state(t, x), None
    
    def _handle_inverse_state(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transform the state back and infer the derivative transformation.
        # See inverse_transform and _inverse_transform_state.
        t, x = self._inverse_transform_state(t_transformed, x_transformed)

        # The transformation of the derivative is inferred.
        # With (t, x) = (f_t, f_x)(t', x'), one has
        # dx/dt = (df_x/dt' + df_x/dx' * dx'/dt')/(df_t/dt' + df_t/dx' * dx'/dt')
        # where the derivatives are evaluated at the transformed data.

        jacobian = self._jacobian_inverse_transform_state(t_transformed, x_transformed)
        df_t_dt_trf, df_t_dx_trf, df_x_dt_trf, df_x_dx_trf = jacobian

        jvp = np.einsum("abij,abj->abi", df_x_dx_trf, xdot_transformed)
        scalar_product = np.einsum("abj,abj->ab", df_t_dx_trf, xdot_transformed)
        scalar_terms = (df_t_dt_trf + scalar_product)

        xdot = (df_x_dt_trf + jvp)/np.expand_dims(scalar_terms, axis=-1)

        return t, x, xdot

    def _handle_derivative(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # See transform and _transform_derivative_time_transform,
        # _affine_derivative_coefficients
        if xdot is None:
            raise ValueError("Derivative must be supplied.")
        if x is not None:
            raise ValueError("Only derivative must be supplied.")
        
        raise NotImplementedError("Support for supplying only the "
                                    "derivative is not implemented yet.")
    
    def _handle_inverse_derivative(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transform derivative back and infer the state transformation.
        # See inverse_transform, _affine_derivative_coefficients and
        # _transform_derivative_time_inverse_transform.

        raise NotImplementedError("Support for supplying only the "
                                    "derivative is not implemented yet.")

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


class MeanStd(Normalizer):
    """
    A normalizer that normalizes the data by subtracting the mean and
    dividing by the standard deviation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._names = ["t", "x", "xdot"]
        self._means = {name: None for name in self._names}
        self._stds = {name: None for name in self._names}

    def _fit(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> MeanStd:
        # If supplied, compute the mean and standard deviation.
        # Everything else is handled by the parent class.
        for name, data in zip(self._names, [t, x, xdot]):
            if data is None:
                continue
            self._means[name] = np.mean(data, axis=(0, 1))
            self._stds[name] = np.std(data, axis=(0, 1))
        return self
    
    def _transform(self, names: list[str], values: list[np.ndarray], inverse: bool = False) -> tuple[np.ndarray, ...]:
        # Simply apply the transformation and its inverse.
        transformed = []
        if inverse:
            transform_fn = lambda x, mean, std: x*std + mean
        else:
            transform_fn = lambda x, mean, std: (x - mean)/std

        for name, value in zip(names, values):
            transformed.append(transform_fn(value, self._means[name], self._stds[name]))

        return tuple(transformed)
    
    def _transform_both(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._transform(self._names, [t, x, xdot])
    
    def _inverse_transform_both(self, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._transform(self._names, [t_transformed, x_transformed, xdot_transformed],
                                inverse=True)
    
    def _transform_state(self, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._transform(self._names[:2], [t, x])
    
    def _inverse_transform_state(self, t_transformed: np.ndarray, x_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._transform(self._names[:2], [t_transformed, x_transformed],
                                inverse=True)
    
    def _jacobian_inverse_transform_state(self, t_transformed: np.ndarray, x_transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # The Jacobian is constant and diagonal
        n_batch, n_steps, n_dim = x_transformed.shape

        df_t_dt_trf = np.ones((n_batch, n_steps))*self._stds["t"]
        df_t_dx_trf = np.zeros((n_batch, n_steps, n_dim))
        df_x_dt_trf = np.zeros((n_batch, n_steps, n_dim))
        # np.zeros followed by a write to np.diag_indices could be faster:
        df_x_dx_trf = np.tile(np.diag(self._stds["x"]), (n_batch, n_steps, 1, 1))

        return df_t_dt_trf, df_t_dx_trf, df_x_dt_trf, df_x_dx_trf
    
    def state_dict(self) -> dict:
        # Convert the numpy arrays to lists (then e.g. pytorch can
        # handle it better).
        mean_dict = {}
        std_dict = {}
        for name in self._names:
            mean = self._means[name]
            std = self._stds[name]
            mean_dict[name] = None if mean is None else mean.tolist()
            std_dict[name] = None if std is None else std.tolist()
        return {"means": mean_dict, "stds": std_dict}
    
    def load_state_dict(self, state_dict: dict) -> None:
        mean_dict = state_dict["means"]
        std_dict = state_dict["stds"]
        for name in self._names:
            mean = mean_dict[name]
            std = std_dict[name]
            self._means[name] = None if mean is None else np.array(mean)
            self._stds[name] = None if std is None else np.array(std)