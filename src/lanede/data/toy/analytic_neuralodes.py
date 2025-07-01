"""
This module contains analytic "neural" ODEs for toy problems, for use
within the core framework. They may, for example, be used, when one
wants to measure the Helmholtz metric of a known ODE.
"""

import torch
import numpy as np

from lanede.core import SecondOrderNeuralODE


# Some of the implementations may double with the ones in odes.py,
# but they are implemented in pytorch. One could try to remove this
# redundancy, but its not straightforward and only a minimal
# overlap.


class HarmonicOscillatorODE(SecondOrderNeuralODE):
    """
    Like `DampedHarmonicOscillator`, but in pytorch.
    """

    def __init__(self, K: torch.Tensor, C: torch.Tensor):
        """
        Set the parameters of the ODE.

        Parameters
        ----------

        K : torch.Tensor, shape (n_dim, n_dim)
            The spring constant matrix.
        C : torch.Tensor, shape (n_dim, n_dim)
            The damping matrix.
        """
        super().__init__()
        self.register_buffer("_K", K)
        self.register_buffer("_C", C)

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        return -self._matmul(self._K, x) - self._matmul(self._C, xdot)

    @property
    def device(self):
        return self._K.device

    @staticmethod
    def _matmul(matrix, vector):
        # (double) Batched vector times non-batched matrix
        return torch.einsum("ij,abj->abi", matrix, vector)


class CaseIIIbODE(SecondOrderNeuralODE):
    """
    Like `NonExtremalCaseIIIb`, but in pytorch.
    """

    def __init__(self):
        super().__init__()
        # Use a dummy tensor to track the device (not the best way, but
        # works for now)
        self.register_buffer("_track_device", torch.tensor(0.0))

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        f_1 = torch.einsum("abi,abi->ab", x, x)
        f_2 = torch.zeros_like(f_1)
        result = torch.stack([f_1, f_2], axis=-1)
        return result

    @property
    def device(self):
        return self._track_device.device


class CaseIVODE(SecondOrderNeuralODE):
    """
    Like `NonExtremalCaseIV`, but in pytorch.
    """

    def __init__(self):
        super().__init__()
        # See `CaseIIIbODE` for explanation
        self.register_buffer("_track_device", torch.tensor(0.0))

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        f_1 = torch.einsum("abi,abi->ab", x, x)
        f_2 = x[:, :, 0]
        result = torch.stack([f_1, f_2], axis=-1)
        return result

    @property
    def device(self):
        return self._track_device.device


class KeplerODE(SecondOrderNeuralODE):
    """
    Like `KeplerProblem`, but in pytorch.
    """

    def __init__(self, semi_major_axis: float, orbital_period: float = 1.0):
        """
        Initialize the Kepler ODE. Both values are only used to compute
        the gravitational constant GM of the ODE.

        Parameters
        ----------
        semi_major_axis : float
            The semi-major axis of the orbit.
        orbital_period : float
            The orbital period of the orbit.
        """
        super().__init__()
        GM = 4 * np.pi**2 * semi_major_axis**3 / orbital_period**2
        self.register_buffer("_GM", torch.tensor(GM, dtype=torch.float32))

    def second_order_function(
        self, t: torch.Tensor, x: torch.Tensor, xdot: torch.Tensor
    ) -> torch.Tensor:
        r = x[:, :, 0]
        rdot = xdot[:, :, 0]
        phidot = xdot[:, :, 1]

        rdotdot = r * phidot**2 - self._GM / r**2
        phiddot = -2 * rdot * phidot / r
        return torch.stack([rdotdot, phiddot], axis=-1)

    @property
    def device(self):
        return self._GM.device
