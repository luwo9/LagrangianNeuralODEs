"""
This module contains functions for transforming data to be
usable for neural networks. See the documentation for more
information.
"""

# NOTE: Many explanations of the normalization involved in this module
# can be found in the documentation / transforms.md.


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
    its derivative in the methods.
    The `fit` method needs to be called before the `transform` or
    `inverse_transform` methods can be called.

    Methods
    -------

    fit(t, x=None, xdot=None)
        Fit the normalizer to the data.

    transform(t, x=None, xdot=None, xdotdot=None)
        Transform time and/or state and/or derivatives.

    inverse_transform(t, x=None, xdot=None, xdotdot=None)
        Inverse transform time and/or state and/or derivatives.

    state_dict()
        Returns the state of the normalizer as a dictionary.

    load_state_dict(state_dict)
        Loads the normalizer state from the dictionary.

    Notes
    -----

    For the training of second order ODEs, it is not always neccessary
    to supply both the state and its derivative. Thus, when normalizing
    the train data, i.e., when calling `fit`(which should always be
    called on the train data only) there may not be a transformation
    fitted for the state or the derivative.

    However, the transformation of one (and of time) will always induce
    a transformation of the other, which might be needed for inference.
    This normalizer is intended to infer the missing transformation,
    aswell as the transformation of the second derivative, if needed.
    To be able to do this, e.g., derivatives of the transformations
    need to be implemented. For more details see documentation.

    When calling the `fit` method, it is decided, what data is supplied
    and what transformations to infer and how. The `transform` method
    can then tranform most combinations of supplied data.
    """

    def __init__(self) -> None:
        self._mode = _OperationMode.NOT_SET
        self._transform_functions = {
            _OperationMode.BOTH: self._handle_both,
            _OperationMode.ONLY_STATE: self._handle_state,
            _OperationMode.ONLY_DERIVATIVE: self._handle_derivative,
        }
        self._inverse_transform_functions = {
            _OperationMode.BOTH: self._handle_inverse_both,
            _OperationMode.ONLY_STATE: self._handle_inverse_state,
            _OperationMode.ONLY_DERIVATIVE: self._handle_inverse_derivative,
        }

    def fit(
        self, t: np.ndarray, x: np.ndarray | None = None, xdot: np.ndarray | None = None
    ) -> Normalizer:
        """
        Fit the normalizer to the data.

        Can only be called once. The data supplied is assumed to be
        transformed and used for training. See Notes.

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

        Notes
        -----

        If state or derivative is not supplied, the respective
        transformation will be inferred later.

        Only supplying the derivative is not supported yet.
        """
        if self._mode != _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is already fitted.")
        if x is None and xdot is None:
            # Add only time support?
            raise ValueError("At least one of x or xdot must be supplied.")
        if x is not None and xdot is not None:
            self._mode = _OperationMode.BOTH
            # TODO: Correct implementation for fitting on both state
            # and derivative. For second order neural odes, the
            # inferred and manual transformation must coincide.
            raise NotImplementedError(
                "Currently, fitting on both state and derivative is not completely implemented."
            )
        elif x is not None:
            self._mode = _OperationMode.ONLY_STATE
        else:
            self._mode = _OperationMode.ONLY_DERIVATIVE

        return self._fit(t, x, xdot)

    def transform(
        self,
        t: np.ndarray,
        x: np.ndarray | None = None,
        xdot: np.ndarray | None = None,
        xdotdot: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        """
        Transform the data.

        See Notes for information on what data needs to be supplied.

        Parameters
        ----------
        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Derivative of the state at times t.
        xdotdot : np.ndarray, shape (n_batch, n_steps, n_dim), optional
            Second derivative of the state at times t.

        Returns
        -------

        Tuple with transformed data. Returns None for the respective
        data that was not supplied. Second derivative is only returned
        if it was supplied.

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Transformed derivative of the state at times t.
        xdotdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed second derivative of the state at times t. Only
            returned if supplied.

        Notes
        -----

        Only time may be supplied, as it transforms independently.

        If only the state was supplied during fitting, the
        transformation of the derivative will be inferred. To transform
        the state it needs to be supplied. To transform the derivative
        or the second derivative, additionally the derivative needs to
        be supplied.

        If both state and derivative were supplied during fitting, they
        are both needed to be supplied in any case (except for the time
        only).

        Only supplying the derivative during fitting is not supported
        yet.
        """
        if self._mode == _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is not fitted.")

        return self._transform_functions[self._mode](t, x, xdot, xdotdot)

    def inverse_transform(
        self,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray | None = None,
        xdot_transformed: np.ndarray | None = None,
        xdotdot_transformed: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        """
        Inverse transform the data.

        This is the inverse of `transform`. See Notes for information
        on what data needs to be supplied.

        Parameters
        ----------
        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Time Steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Derivative of the state at times t.
        xdotdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Second derivative of the state at times t.

        Returns
        -------

        Tuple with transformed data. Returns None for the respective
        data that was not supplied. Second derivative is only returned
        if it was supplied.

        t : np.ndarray, shape (n_batch, n_steps)
            Inverse transformed time steps.
        x : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Inverse transformed state at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim) or None
            Inverse transformed derivative of the state at times t.
        xdotdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Inverse transformed second derivative of the state at
            times t. Only returned if supplied.

        Notes
        -----

        Only time may be supplied, as it transforms independently.

        If only the state was supplied during fitting, the
        transformation of the derivative will be inferred. To transform
        the state it needs to be supplied. To transform the derivative
        or the second derivative, additionally the derivative needs to
        be supplied.

        If both state and derivative were supplied during fitting, they
        are both needed to be supplied in any case (except for the time
        only).

        Only supplying the derivative during fitting is not supported
        yet.
        """
        if self._mode == _OperationMode.NOT_SET:
            raise RuntimeError("Normalizer is not fitted.")

        return self._inverse_transform_functions[self._mode](
            t_transformed, x_transformed, xdot_transformed, xdotdot_transformed
        )

    @abstractmethod
    def _fit(self, t: np.ndarray, x: np.ndarray | None, xdot: np.ndarray | None) -> Normalizer:
        # See fit.
        # This is where the actual fitting happens.
        pass

    @abstractmethod
    def _transform_time(self, t: np.ndarray) -> np.ndarray:
        """
        Transform the time.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            Time Steps.

        Returns
        -------

        t_transformed : np.ndarray, shape (n_batch, n_steps)
            Transformed time steps.
        """
        # This could theoretically be implemented on a per case basis,
        # but does not seem to be necessary for now.
        pass

    @abstractmethod
    def _inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse of `transform_time`.
        """
        pass

    @abstractmethod
    def _derivative_inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        """
        Derivative of `_inverse_transform_time`.
        """
        pass

    @abstractmethod
    def _double_derivative_inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        """
        Double derivative of `_inverse_transform_time`.
        """
        pass

    @abstractmethod
    def _transform_both(
        self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

        Tuple with transformed state and derivative.

        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.
        """
        pass

    @abstractmethod
    def _inverse_transform_both(
        self, t: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inverse of `_transform_both`.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The (untransformed) time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.

        Returns
        -------

        Tuple with untransformed state and derivative.

        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            Derivative of the state at times t.
        """
        pass

    @abstractmethod
    def _jacobian_inverse_transform_both(
        self, t: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Jacobian of the second function component of
        `_inverse_transform_both`.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The (untransformed) time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        xdot_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed derivative of the state at times t.

        Returns
        -------

        Tuple with the Jacobian components.

        time_derivative : np.ndarray, shape (n_batch, n_steps, n_dim)
            The (partial) derivative with respect to (untransformed)
            time.
        state_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            The (partial) derivative with respect to the (transformed)
            state.
        derivative_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            The (partial) derivative with respect to the (transformed)
            derivative of the state.
        """
        pass

    @abstractmethod
    def _transform_state(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
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

        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.
        """
        pass

    @abstractmethod
    def _inverse_transform_state(self, t: np.ndarray, x_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse of `_transform_state`.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The (untransformed) time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.

        Returns
        -------

        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            State at times t.

        """
        pass

    @abstractmethod
    def _jacobian_inverse_transform_state(
        self, t: np.ndarray, x_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Jacobian of `_inverse_transform_state`.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The (untransformed) time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.

        Returns
        -------

        Tuple with the Jacobian components.

        time_derivative : np.ndarray, shape (n_batch, n_steps, n_dim)
            The (partial) derivative with respect to (untransformed)
            time.
        state_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            The (partial) derivative with respect to the (transformed)
            state.
        """
        pass

    @abstractmethod
    def _double_jacobian_inverse_transform_state(
        self, t: np.ndarray, x_transformed: np.ndarray
    ) -> np.ndarray:
        """
        Double Jacobian of `_inverse_transform_state`.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The (untransformed) time steps.
        x_transformed : np.ndarray, shape (n_batch, n_steps, n_dim)
            Transformed state at times t.

        Returns
        -------

        Tuple with the Double Jacobian components.

        time_time_derivative : np.ndarray, shape (n_batch, n_steps, n_dim)
            The double (partial) derivative with respect to
            (untransformed) time.
        time_state_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            The (partial) derivative with respect to (transformed)
            state and (untransformed) time. (First state then time)
        state_time_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim)
            The (partial) derivative with respect to (untransformed)
            time and (transformed) state. (First time then state)
        state_state_derivative : np.ndarray, shape (n_batch, n_steps, n_dim, n_dim, n_dim)
            The double (partial) derivative with respect to the
            (transformed) state.
        """
        pass

    def _handle_both(
        self,
        t: np.ndarray,
        x: np.ndarray | None,
        xdot: np.ndarray | None,
        xdotdot: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See transform and _transform_both.
        # Infers the transformation of the second derivative only.
        t_transformed = self._transform_time(t)

        no_x_xdot = [x is None, xdot is None]
        if all(no_x_xdot) and xdotdot is None:
            return t_transformed, None, None

        if any(no_x_xdot):
            raise ValueError("Both state and derivative must be supplied.")

        x_transformed, xdot_transformed = self._transform_both(t, x, xdot)
        if xdotdot is None:
            return t_transformed, x_transformed, xdot_transformed

        xdotdot_transformed = self._both_infer_second_derivative(
            t, t_transformed, x_transformed, xdot_transformed, xdotdot
        )

        return t_transformed, x_transformed, xdot_transformed, xdotdot_transformed

    def _handle_inverse_both(
        self,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray | None = None,
        xdot_transformed: np.ndarray | None = None,
        xdotdot_transformed: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See inverse_transform, _inverse_transform_both
        # and _handle_both.
        t = self._inverse_transform_time(t_transformed)

        no_x_xdot = [x_transformed is None, xdot_transformed is None]
        if all(no_x_xdot) and xdotdot_transformed is None:
            return t, None, None

        if any(no_x_xdot):
            raise ValueError("Both state and derivative must be supplied.")

        x, xdot = self._inverse_transform_both(t, x_transformed, xdot_transformed)
        if xdotdot_transformed is None:
            return t, x, xdot

        xdotdot = self._both_infer_inverse_second_derivative(
            t, t_transformed, x_transformed, xdot_transformed, xdotdot_transformed
        )

        return t, x, xdot, xdotdot

    def _handle_state(
        self,
        t: np.ndarray,
        x: np.ndarray | None,
        xdot: np.ndarray | None,
        xdotdot: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See transform and _transform_state.
        # Infers the transformation of the derivative and the second
        # derivative.
        t_transformed = self._transform_time(t)

        if x is None and xdot is None and xdotdot is None:
            return t_transformed, None, None

        if x is None:
            raise ValueError("State must be supplied.")

        x_transformed = self._transform_state(t, x)
        xdot_transformed = (
            None
            if xdot is None
            else self._state_infer_derivative(t, t_transformed, x_transformed, xdot)
        )

        if xdotdot is None:
            return t_transformed, x_transformed, xdot_transformed

        if xdot_transformed is None:
            raise ValueError("Derivative must be supplied.")

        xdotdot_transformed = self._state_infer_second_derivative(
            t, t_transformed, x_transformed, xdot_transformed, xdotdot
        )

        return t_transformed, x_transformed, xdot_transformed, xdotdot_transformed

    def _handle_inverse_state(
        self,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray | None = None,
        xdot_transformed: np.ndarray | None = None,
        xdotdot_transformed: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See inverse_transform, _inverse_transform_state
        # and _handle_state.
        # Infers the transformation of the derivative and the second
        # derivative.
        t = self._inverse_transform_time(t_transformed)

        if x_transformed is None and xdot_transformed is None and xdotdot_transformed is None:
            return t, None, None

        if x_transformed is None:
            raise ValueError("State must be supplied.")

        x = self._inverse_transform_state(t, x_transformed)
        xdot = (
            None
            if xdot_transformed is None
            else self._state_infer_inverse_derivative(
                t, t_transformed, x_transformed, xdot_transformed
            )
        )

        if xdotdot_transformed is None:
            return t, x, xdot

        if xdot is None:
            raise ValueError("Derivative must be supplied.")

        xdotdot = self._state_infer_inverse_second_derivative(
            t, t_transformed, x_transformed, xdot_transformed, xdotdot_transformed
        )

        return t, x, xdot, xdotdot

    def _handle_derivative(
        self,
        t: np.ndarray,
        x: np.ndarray | None,
        xdot: np.ndarray | None,
        xdotdot: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See transform.
        # Infers the transformation of the state and the second
        # derivative.
        raise NotImplementedError(
            "Support for supplying only the " "derivative is not implemented yet."
        )

    def _handle_inverse_derivative(
        self,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray | None = None,
        xdot_transformed: np.ndarray | None = None,
        xdotdot_transformed: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, ...]:
        # See inverse_transform.
        # Infers the transformation of the state and the second
        # derivative.
        raise NotImplementedError(
            "Support for supplying only the " "derivative is not implemented yet."
        )

    def _both_infer_inverse_second_derivative(
        self,
        t: np.ndarray,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray,
        xdot_transformed: np.ndarray,
        xdotdot_transformed: np.ndarray,
    ) -> np.ndarray:
        # Infer the second derivative inverse transformation, using
        # the Jacobian of the inverse transformation.
        jacobian = self._jacobian_inverse_transform_both(t, x_transformed, xdot_transformed)
        dk_dt, dk_dx_trf, dk_dxdot_trf = jacobian

        dg_dt_trf = self._derivative_inverse_transform_time(t_transformed)
        dt_trf_dt = 1 / dg_dt_trf

        dx_trf_dt = np.einsum("abi,ab->abi", xdot_transformed, dt_trf_dt)
        dxdot_trf_dt = np.einsum("abi,ab->abi", xdotdot_transformed, dt_trf_dt)

        xdotdot = dk_dt
        xdotdot += np.einsum("abij,abj->abi", dk_dx_trf, dx_trf_dt)
        xdotdot += np.einsum("abij,abj->abi", dk_dxdot_trf, dxdot_trf_dt)
        return xdotdot

    def _both_infer_second_derivative(
        self,
        t: np.ndarray,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray,
        xdot_transformed: np.ndarray,
        xdotdot: np.ndarray,
    ) -> np.ndarray:
        # Infer the second derivative transformation, by simply
        # algebraicly inverting `_both_infer_inverse_second_derivative`.
        raise NotImplementedError(
            "Normalizing the second derivative "
            "inferred from a state and derivative "
            "fit is not implemented yet."
        )

    def _state_infer_inverse_derivative(
        self,
        t: np.ndarray,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray,
        xdot_transformed: np.ndarray,
    ) -> np.ndarray:
        # Infer the first derivative inverse transformation, using the
        # Jacobian of the inverse transformation.
        df_dt, df_dx_trf = self._jacobian_inverse_transform_state(t, x_transformed)

        dg_dt_trf = self._derivative_inverse_transform_time(t_transformed)
        dt_trf_dt = 1 / dg_dt_trf
        dx_trf_dt = np.einsum("abi,ab->abi", xdot_transformed, dt_trf_dt)
        xdot = df_dt + np.einsum("abij,abj->abi", df_dx_trf, dx_trf_dt)
        return xdot

    def _state_infer_derivative(
        self, t: np.ndarray, t_transformed: np.ndarray, x_transformed: np.ndarray, xdot: np.ndarray
    ) -> np.ndarray:
        # Infer the first derivative transformation, by simply
        # algebraicly inverting `_state_infer_inverse_derivative`.

        df_dt, df_dx_trf = self._jacobian_inverse_transform_state(t, x_transformed)
        dg_dt_trf = self._derivative_inverse_transform_time(t_transformed)

        xdot_transformed = xdot - df_dt
        x_transformed = np.einsum("abi,ab->abi", xdot_transformed, dg_dt_trf)
        # Much faster and stable than np.linalf.inv:
        xdot_transformed = np.linalg.solve(df_dx_trf, x_transformed)
        return xdot_transformed

    def _state_infer_inverse_second_derivative(
        self,
        t: np.ndarray,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray,
        xdot_transformed: np.ndarray,
        xdotdot_transformed: np.ndarray,
    ) -> np.ndarray:
        # Infer the second derivative inverse transformation, using the
        # hessian of the inverse transformation.

        # Again, see transforms.md.

        double_jacobian = self._double_jacobian_inverse_transform_state(t, x_transformed)
        df_dt_dt, df_dt_dx_trf, df_dx_trf_dt, df_dx_trf_dx_trf = double_jacobian

        jacobian = self._jacobian_inverse_transform_state(t, x_transformed)
        _, df_dx_trf = jacobian

        dg_dt_trf_dt_trf = self._double_derivative_inverse_transform_time(t_transformed)
        dg_dt_trf = self._derivative_inverse_transform_time(t_transformed)
        dt_trf_dt = 1 / dg_dt_trf

        # Inverse function theorem for the second derivative
        dt_trf_dt_dt = -dg_dt_trf_dt_trf * dt_trf_dt**3

        dx_trf_dt = np.einsum("abi,ab->abi", xdot_transformed, dt_trf_dt)

        xdotdot = df_dt_dt + np.einsum("abij,abj->abi", df_dx_trf_dt, dx_trf_dt)
        xdotdot += np.einsum("abij,abj->abi", df_dt_dx_trf, dx_trf_dt)
        # Contract rank 3 double jacobian tensor twice with a vector
        xdotdot += np.einsum(
            "abij,abj->abi", np.einsum("abijk,abk->abij", df_dx_trf_dx_trf, dx_trf_dt), dx_trf_dt
        )

        dx_trf_dt_dt = np.einsum("abi,ab->abi", xdotdot_transformed, dt_trf_dt**2)
        dx_trf_dt_dt += np.einsum("abi,ab->abi", xdot_transformed, dt_trf_dt_dt)

        xdotdot += np.einsum("abij,abj->abi", df_dx_trf, dx_trf_dt_dt)

        return xdotdot

    def _state_infer_second_derivative(
        self,
        t: np.ndarray,
        t_transformed: np.ndarray,
        x_transformed: np.ndarray,
        xdot_transformed: np.ndarray,
        xdotdot: np.ndarray,
    ) -> np.ndarray:
        # Infer the second derivative transformation, by simply
        # algebraicly inverting `_state_infer_inverse_second_derivative`.
        raise NotImplementedError(
            "Normalizing the second derivative "
            "inferred from a state-only fit is "
            "not implemented yet."
        )

    def state_dict(self) -> dict:
        """
        Returns the state of the normalizer as a dictionary.

        Returns
        -------
        dict
            The state of the normalizer.
        """
        return {"operation_mode": self._mode.name}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the normalizer state from the dictionary.

        Parameters
        ----------
        state_dict : dict
            The state of the normalizer.
        """
        self._mode = _OperationMode[state_dict["operation_mode"]]


# NOTE: Could add subclass for independent transformation(s), for cases
# where, e.g., xdot' = xdot'(xdot). This would then allow only
# supplying the derivative, when transforming it. When using the base
# class time (and possibly) state would always need to be supplied,
# eventhough the underlying transformation does not require it.


class MeanStd(Normalizer):
    """
    A normalizer that normalizes the data by subtracting the mean and
    dividing by the standard deviation.
    """

    def __init__(self) -> None:
        super().__init__()
        # Could do enum for names, but this is simpler.
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

    def _transform(
        self, names: list[str], values: list[np.ndarray], inverse: bool = False
    ) -> tuple[np.ndarray, ...]:
        # Simply apply the transformation and its inverse.
        assert set(names).issubset(self._names), "Invalid names."
        transformed = []
        if inverse:
            transform_fn = lambda x, mean, std: x * std + mean
        else:
            transform_fn = lambda x, mean, std: (x - mean) / std

        for name, value in zip(names, values):
            transformed.append(transform_fn(value, self._means[name], self._stds[name]))

        return tuple(transformed)

    def _default_inverse_jacobian(
        self, names: list[str], batch_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, ...]:
        # Simply return diagonal jacobians.
        assert set(names).issubset(self._names), "Invalid names."
        jacobians = []
        for name in names:
            dim = self._means[name].shape
            if len(dim) == 0:
                jacobian = np.full(batch_shape, self._stds[name])
                jacobians.append(jacobian)
                continue
            dim = dim[0]
            jacobian = np.zeros(batch_shape + (dim, dim))
            diag_indices = np.diag_indices(dim)
            jacobian[..., diag_indices[0], diag_indices[1]] = self._stds[name]
            jacobians.append(jacobian)
        return tuple(jacobians)

    def _transform_time(self, t: np.ndarray) -> np.ndarray:
        return self._transform(["t"], [t])[0]

    def _inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        return self._transform(["t"], [t_transformed], inverse=True)[0]

    def _derivative_inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        return self._default_inverse_jacobian(["t"], t_transformed.shape)[0]

    def _double_derivative_inverse_transform_time(self, t_transformed: np.ndarray) -> np.ndarray:
        # The transformation is linear, so the second derivative is
        # zero.
        return np.zeros_like(t_transformed)

    def _transform_both(
        self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._transform(["x", "xdot"], [x, xdot])

    def _inverse_transform_both(
        self, t: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._transform(["x", "xdot"], [x_transformed, xdot_transformed], inverse=True)

    def _jacobian_inverse_transform_both(
        self, t: np.ndarray, x_transformed: np.ndarray, xdot_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_batch, n_steps, n_dim = x_transformed.shape

        # t, x, xdot transform independently.
        time_derivative = np.zeros((n_batch, n_steps, n_dim))
        state_derivative = np.zeros((n_batch, n_steps, n_dim, n_dim))
        derivative_derivative = self._default_inverse_jacobian(["xdot"], (n_batch, n_steps))[0]
        return time_derivative, state_derivative, derivative_derivative

    def _transform_state(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self._transform(["x"], [x])[0]

    def _inverse_transform_state(self, t: np.ndarray, x_transformed: np.ndarray) -> np.ndarray:
        return self._transform(["x"], [x_transformed], inverse=True)[0]

    def _jacobian_inverse_transform_state(
        self, t: np.ndarray, x_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_batch, n_steps, n_dim = x_transformed.shape

        # t, x transform independently.
        time_derivative = np.zeros((n_batch, n_steps, n_dim))
        state_derivative = self._default_inverse_jacobian(["x"], (n_batch, n_steps))[0]
        return time_derivative, state_derivative

    def _double_jacobian_inverse_transform_state(
        self, t: np.ndarray, x_transformed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_batch, n_steps, n_dim = x_transformed.shape

        # All transformations are linear, so all second derivatives are
        # zero.
        time_time_derivative = np.zeros((n_batch, n_steps, n_dim))
        time_state_derivative = np.zeros((n_batch, n_steps, n_dim, n_dim))
        state_time_derivative = np.zeros((n_batch, n_steps, n_dim, n_dim))
        state_state_derivative = np.zeros((n_batch, n_steps, n_dim, n_dim, n_dim))
        return (
            time_time_derivative,
            time_state_derivative,
            state_time_derivative,
            state_state_derivative,
        )

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        # Convert the numpy arrays to lists (then e.g. pytorch can
        # handle it better).
        mean_dict = {}
        std_dict = {}
        for name in self._names:
            mean = self._means[name]
            std = self._stds[name]
            mean_dict[name] = None if mean is None else mean.tolist()
            std_dict[name] = None if std is None else std.tolist()
        state_dict.update({"means": mean_dict, "stds": std_dict})
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        mean_dict = state_dict.pop("means")
        std_dict = state_dict.pop("stds")
        for name in self._names:
            mean = mean_dict[name]
            std = std_dict[name]
            self._means[name] = None if mean is None else np.array(mean)
            self._stds[name] = None if std is None else np.array(std)

        super().load_state_dict(state_dict)
