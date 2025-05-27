"""
This module contains ODEs for toy problems. They can be integrated to
generate data.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


# TODO: Might want to add wrapping methods that e.g. directly generate
# harmonic oscillator data from a single function call. Without the
# need to specify all of: t, x_0, xdot_0, oscillator parameters, etc.


# NOTE: The following overlaps with `lanede.core`'s
# SecondOrderNeuralODE and SolvedSecondOrderNeuralODE, however this
# code should be able to use numpy. The overlap currently is minimal
# and removing it would make things more complicated. If in the future
# the overlap becomes more significant, the ODEs below could be made to
# inherit `lanede.core.SecondOrderNeuralODE` and integrated using
# `lanede.core.SolvedSecondOrderNeuralODE`, while translating between
# numpy and torch as needed.


def from_ode(
    ode: ODE, t: np.ndarray, x_0: np.ndarray, xdot_0: np.ndarray, **solve_ivp_kwargs
) -> np.ndarray:
    """
    Generate data from an ODE. To do this the ode is integrated using
    `scipy.integrate.solve_ivp`.

    Parameters
    ----------

    ode : ODE
        The ODE to integrate.
    t : np.ndarray, shape (n_steps,)
        The time steps at which to evaluate the ODE.
    x_0 : np.ndarray, shape (n_batch, n_dim)
        The initial state.
    xdot_0 : np.ndarray, shape (n_batch, n_dim)
        The initial derivative of the state.
    **solve_ivp_kwargs
        Additional keyword arguments to pass to `scipy.integrate.solve_ivp`.

    Returns
    -------

    np.ndarray, shape (n_steps, n_dim)
        The state at each time step.
    np.ndarray, shape (n_steps, n_dim)
        The derivative of the state at each time step.
    np.ndarray, shape (n_steps, n_dim)
        The second derivative of the state at each time step.
    """
    kwargs_with_defaults = {
        "method": "DOP853",
        "rtol": 1e-9,
        "atol": 1e-9,
    }
    kwargs_with_defaults.update(solve_ivp_kwargs)

    n_batch, n_dim = x_0.shape

    # Integrate the ODE to get the state and its derivative
    def scipy_ode_func(t, y):
        # t is scalar, y is 1D array
        # Convert to fit ODE signature
        y = y.reshape((n_batch, 1, 2 * n_dim))
        x, xdot = np.split(y, 2, axis=2)
        t = np.full((n_batch, 1), t)

        xdotdot = ode(t, x, xdot)

        # Return as 1D array
        return np.concatenate([xdot, xdotdot], axis=2).flatten()

    y_0 = np.concatenate([x_0, xdot_0], axis=1).flatten()

    sol = solve_ivp(scipy_ode_func, (t[0], t[-1]), y_0, t_eval=t, **kwargs_with_defaults)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    y = sol.y.reshape((n_batch, 2 * n_dim, len(t)))
    y = y.transpose((0, 2, 1))

    x, xdot = np.split(y, 2, axis=2)

    # Now simply compute the second derivative
    t_with_batches = np.tile(t, (n_batch, 1))
    xdotdot = ode(t_with_batches, x, xdot)

    return x, xdot, xdotdot


class ODE(ABC):
    """
    Abstract base class for ODEs.
    """

    @abstractmethod
    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        """
        The function of the explicit second order ODE.
        More precisely, in the ODE $\ddot{x} = f^\\ast(t, x, \dot{x})$,
        this function is the $f^\\ast$.

        Parameters
        ----------

        t : np.ndarray, shape (n_batch, n_steps)
            The time steps at which the function should be evaluated.
        x : np.ndarray, shape (n_batch, n_steps, n_dim)
            The state at time t.
        xdot : np.ndarray, shape (n_batch, n_steps, n_dim)
            The derivative of the state at time t.

        Returns
        -------

        np.ndarray, shape (n_batch, n_steps, n_dim)
            The second order derivative of the state at time t.
        """
        pass


class DampedHarmonicOscillator(ODE):
    """
    A damped harmonic oscillator in n dimensions.

    The ODE is given by

    $$
    \ddot{x} = -K x - C \dot{x},
    $$

    where $K, C$ are n x n matrices.
    """

    def __init__(self, K: np.ndarray, C: np.ndarray):
        """
        Set the parameters of the ODE.

        Parameters
        ----------

        K : np.ndarray, shape (n_dim, n_dim)
            The spring constant matrix.
        C : np.ndarray, shape (n_dim, n_dim)
            The damping matrix.
        """
        self._K = K
        self._C = C

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        return -self._matmul(self._K, x) - self._matmul(self._C, xdot)

    @staticmethod
    def _matmul(matrix, vector):
        # (double) Batched vector times non-batched matrix
        return np.einsum("ij,abj->abi", matrix, vector)


class NonExtremalCaseIIIb(ODE):
    r"""
    An example of an ODE that does not originate from a Lagrangian.

    The ODE is given by:

    $$
    \begin{pmatrix}
        \ddot{x}_1 \\
        \ddot{x}_2
        \end{pmatrix}
    =
    \begin{pmatrix}
        x_1^2 + x_2^2 \\
        0
    \end{pmatrix}
    $$

    This example was given by Douglas himself for the case IIIb in the paper [1]_.

    References
    ----------

    .. [1] Douglas J. (1941). Solution of the Inverse Problem of the Calculus of Variations. Transactions of the American Mathematical Society, 50(1), 71-128.
    https://doi.org/10.1090/S0002-9947-1941-0004740-5.
    """

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        f_1 = np.einsum("abi,abi->ab", x, x)
        f_2 = np.zeros_like(f_1)
        result = np.stack([f_1, f_2], axis=-1)
        return result


class NonExtremalCaseIV(ODE):
    r"""
    An example of an ODE that does not originate from a Lagrangian.

    The ODE is given by:

    $$
    \begin{pmatrix}
        \ddot{x}_1 \\
        \ddot{x}_2
        \end{pmatrix}
    =
    \begin{pmatrix}
        x_1^2 + x_2^2 \\
        x_1
    \end{pmatrix}
    $$

    This example was given by Douglas himself for the case IV in the paper [1]_.

    References
    ----------

    .. [1] Douglas J. (1941). Solution of the Inverse Problem of the Calculus of Variations. Transactions of the American Mathematical Society, 50(1), 71-128.
    https://doi.org/10.1090/S0002-9947-1941-0004740-5.
    """

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        f_1 = np.einsum("abi,abi->ab", x, x)
        f_2 = x[:, :, 0]
        result = np.stack([f_1, f_2], axis=-1)
        return result
