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


class KeplerProblem(ODE):
    r"""
    The Kepler problem, which describes the motion of two bodies
    (masses $m_1$ and $m_2$) under the influence of their mutual
    gravitational attraction.

    Here it is formulated in relative, polar coordinates
    $(r, \varphi)$.

    Defining $M = m_1 + m_2$ and $\mu = m_1 m_2 / M$, the Lagrangian is given by:
    $$
    L(r, \dot{r}, \varphi, \dot{\varphi}) = \frac{1}{2} \mu (\dot{r}^2 + r^2 \dot{\varphi}^2) + \frac{G m_1 m_2}{r}
    $$

    Thus the equations of motion are given by:
    $$
    \begin{pmatrix}
        \ddot{r} \\
        \ddot{\varphi}
    \end{pmatrix}
    =
    \begin{pmatrix}
        r\dot{\varphi}^2 - \frac{G M}{r^2} \\
        -\frac{2 \dot{r} \dot{\varphi}}{r}
    \end{pmatrix}
    $$

    Additional Methods
    ------------------

    get_initial_conditions(semi_latus_rectum, eccentricity, phi_0)
        Get initial conditions that yield an orbit with the
        specified characteristics.
    """

    def __init__(self, semi_major_axis: float, orbital_period: float = 1.0):
        r"""
        Set up the Kepler problem.

        The gravitational parameter $GM$ is computed to fit the
        supplied characteristics of the orbit, using Kepler's third
        law:
        $$ T^2 = \frac{4 \pi^2 a^3}{GM} $$

        Parameters
        ----------

        semi_major_axis : float
            The semi-major axis of an orbit that takes `orbital_period`
            time to complete.
        orbital_period : float, default=1.0
            The time it takes to complete an orbit with the given
            semi-major axis.

        Notes
        -----

        This does not mean all orbits have the specified
        semi-major axis or orbital period. They are solely used to
        choose the gravitational parameter $GM$ such that an orbit
        with the given semi-major axis has the specified orbital
        period.

        Depending on the initial conditions, the actual orbit may
        differ in both semi-major axis and orbital period.
        """
        self._GM = 4 * np.pi**2 * semi_major_axis**3 / orbital_period**2
        if self._GM < 0:
            raise ValueError(
                "Detected negative gravitational parameter GM, perhaps the semi-major"
                " axis has the wrong sign?"
            )

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        r = x[:, :, 0]
        rdot = xdot[:, :, 0]
        phidot = xdot[:, :, 1]

        rdotdot = r * phidot**2 - self._GM / r**2
        phiddot = -2 * rdot * phidot / r
        return np.stack([rdotdot, phiddot], axis=-1)

    def get_initial_conditions(
        self, semi_latus_rectum: np.ndarray, eccentricity: np.ndarray, phi_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Get initial conditions for the Kepler problem from orbit
        characteristics.

        Parameters
        ----------

        semi_latus_rectum : np.ndarray, shape (n_batch,)
            The semi-latus rectum of the orbit, see notes below.
        eccentricity : np.ndarray, shape (n_batch,)
            The (numerical) eccentricity of the orbit.
        phi_0 : np.ndarray, shape (n_batch,)
            The initial angle in radians.

        Returns
        -------

        np.array, shape (n_batch, 2)
            The initial state in polar coordinates $(r_0, \varphi_0)$.
        np.array, shape (n_batch, 2)
            The initial velocities in polar coordinates
            $(\dot{r}_0, \dot{\varphi}_0)$.

        Notes
        -----

        The semi-latus rectum is half the width of the conic section,
        perpendicular to the major axis and related to the semi-major
        axis $a$ and eccentricity $e$ by $p = a(1 - e^2)$.

        For eccentricies $1 \leq e \lessim 20$ angles
        $-1.6 \lesssim e \lesssim 1.6$ should provide a nice
        description of the orbit.
        """
        # (Specific) angular momentum (from solution of Kepler problem)
        angular_momentum = np.sqrt(self._GM * semi_latus_rectum)

        # Define r as smallest distance at phi = 0
        # In this way for parabulas r is finite around phi = 0
        r_0 = semi_latus_rectum / (1 + eccentricity * np.cos(phi_0))
        # From d/dt r = dr/dphi h/r^2 (h_0 = h = angular momentum):
        rdot_0 = self._GM / angular_momentum * eccentricity * np.sin(phi_0)
        phidot_0 = angular_momentum / r_0**2

        x_0 = np.stack([r_0, phi_0], axis=-1)
        xdot_0 = np.stack([rdot_0, phidot_0], axis=-1)

        return x_0, xdot_0


class FreeFallWithDrag(ODE):
    r"""
    A free fall problem with drag force.

    The ODE is given by:

    $$
    \ddot{x} = -g - \frac{g}{v_\infty^2} \dot{x}|\dot{x}|,
    $$

    where $g$ is the gravitational acceleration and $v_\infty$ is the
    terminal velocity.
    """

    def __init__(self, terminal_velocity: float, g: float):
        """
        Set the parameters of the ODE.

        Parameters
        ----------

        terminal_velocity : float
            The terminal velocity of the object.
        g : float
            The gravitational acceleration.
        """
        self._g = g
        self._drag_coeff = g / terminal_velocity**2

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        return -self._g - self._drag_coeff * xdot * np.abs(xdot)


class DoublePendulum(ODE):
    r"""
    A double pendulum with masses $m_1$ and $m_2$ and lengths $l_1$ and
    $l_2$. The gravitational acceleration is given by $g$.

    The Lagrangian is given by:

    $$
    g l_{1} \left(m_{1} + m_{2}\right) \cos{\left(x_{1} \right)} + g
    l_{2} m_{2} \cos{\left(x_{2} \right)} + \frac{l_{1}^{2}
    \dot{x}_{1}^{2} \left(m_{1} + m_{2}\right)}{2} + l_{1} l_{2} m_{2}
    \dot{x}_{1} \dot{x}_{2} \cos{\left(x_{1} - x_{2} \right)} +
    \frac{l_{2}^{2} m_{2} \dot{x}_{2}^{2}}{2}
    $$

    The explicit second order ODE is given by:

    $$
    \begin{pmatrix}
        \ddot{x}_1 \\
        \ddot{x}_2
    \end{pmatrix}
    =
    \begin{pmatrix}\frac{- g \left(m_{1} + m_{2}\right) \sin{\left(x_{1}
    \right)} - l_{2} m_{2} \dot{x}_{2}^{2} \sin{\left(x_{1} - x_{2}
    \right)} + m_{2} \left(g \sin{\left(x_{2} \right)} - l_{1}
    \dot{x}_{1}^{2} \sin{\left(x_{1} - x_{2} \right)}\right)
    \cos{\left(x_{1} - x_{2} \right)}}{l_{1} \left(m_{1} + m_{2}
    \sin^{2}{\left(x_{1} - x_{2} \right)}\right)}\\ \frac{- \left(m_{1}
    + m_{2}\right) \left(g \sin{\left(x_{2} \right)} - l_{1}
    \dot{x}_{1}^{2} \sin{\left(x_{1} - x_{2} \right)}\right) + \left(g
    \left(m_{1} + m_{2}\right) \sin{\left(x_{1} \right)} + l_{2} m_{2}
    \dot{x}_{2}^{2} \sin{\left(x_{1} - x_{2} \right)}\right)
    \cos{\left(x_{1} - x_{2} \right)}}{l_{2} \left(m_{1} + m_{2}
    \sin^{2}{\left(x_{1} - x_{2} \right)}\right)}\end{pmatrix}.
    $$
    """

    def __init__(
        self, mass_1: float, mass_2: float, length_1: float, length_2: float, g: float = 9.81
    ):
        """
        Set the parameters of the ODE.

        Parameters
        ----------

        mass_1 : float
            The mass of the first pendulum.
        mass_2 : float
            The mass of the second pendulum.
        length_1 : float
            The length of the first pendulum.
        length_2 : float
            The length of the second pendulum.
        g : float, default=9.81
            The gravitational acceleration.
        """
        self._mass_1 = mass_1
        self._mass_2 = mass_2
        self._length_1 = length_1
        self._length_2 = length_2
        self._g = g

    def __call__(self, t: np.ndarray, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        x_1 = x[:, :, 0]
        x_2 = x[:, :, 1]
        xdot_1 = xdot[:, :, 0]
        xdot_2 = xdot[:, :, 1]

        xdotdot_1 = (
            -self._g * (self._mass_1 + self._mass_2) * np.sin(x_1)
            - self._length_2 * self._mass_2 * xdot_2**2 * np.sin(x_1 - x_2)
            + self._mass_2
            * (self._g * np.sin(x_2) - self._length_1 * xdot_1**2 * np.sin(x_1 - x_2))
            * np.cos(x_1 - x_2)
        ) / (self._length_1 * (self._mass_1 + self._mass_2 * np.sin(x_1 - x_2) ** 2))

        xdotdot_2 = (
            -(self._mass_1 + self._mass_2)
            * (self._g * np.sin(x_2) - self._length_1 * xdot_1**2 * np.sin(x_1 - x_2))
            + (
                self._g * (self._mass_1 + self._mass_2) * np.sin(x_1)
                + self._length_2 * self._mass_2 * xdot_2**2 * np.sin(x_1 - x_2)
            )
            * np.cos(x_1 - x_2)
        ) / (self._length_2 * (self._mass_1 + self._mass_2 * np.sin(x_1 - x_2) ** 2))

        return np.stack([xdotdot_1, xdotdot_2], axis=-1)
