"""
This module contains temporal schedulers for training neural ODEs.
Given the number of training steps so far, they give the fraction of
time steps to use for updating the neural ODE.
"""

from abc import ABC, abstractmethod
from math import log, exp


class TemporalScheduler(ABC):
    """
    Abstract class for temporal schedulers. Given the number of training
    steps so far, they give the fraction of time steps to use for updating
    the neural ODE.

    Methods
    -------

    get_ratio(n_steps_so_far: int) -> float
    """

    @abstractmethod
    def get_ratio(self, n_steps_so_far: int) -> float:
        """
        Get the ratio of time steps to use for updating the neural ODE.

        Parameters
        ----------

        n_steps_so_far : int
            The number of training steps so far.

        Returns
        -------

        float
            The ratio of time steps to use for updating the neural ODE.
        """
        pass


class SigmoidTemporalScheduler(TemporalScheduler):
    """
    Sigmoid temporal scheduler. Starts with a small fraction of time
    steps and increases it to 1 in a sigmoid fashion. Rounds to 1
    after a specified number of steps.
    """

    def __init__(self, half_at: int, max_steps: int | None = None, width: int | None = None):
        """
        Initialize the sigmoid temporal scheduler.

        Parameters
        ----------

        half_at : int
            The number of training steps at which the fraction of time
            steps is 0.5.
        max_steps : int, optional
            The number of training steps after which the fraction of time
            steps is 1. If None, it is determined such that the
            the fraction is rounded to 1 if greater than 0.999.
        width : int, optional
            The width of the sigmoid function. If None, it is set to
            half_at / 2.
        """
        if width is None:
            width = half_at / 2  # float is fine
        if max_steps is None:
            max_steps = -log(1.0 / 0.999 - 1) * width / log(3) + half_at
        self._half_at = half_at
        self._max_steps = max_steps
        self._width = width

    def get_ratio(self, n_steps_so_far: int) -> float:
        """
        Get the ratio of time steps to use for updating the neural ODE.

        Parameters
        ----------

        n_steps_so_far : int
            The number of training steps so far.

        Returns
        -------

        float
            The ratio of time steps to use for updating the neural ODE.
        """
        if n_steps_so_far >= self._max_steps:
            return 1.0
        return self._sigmoid(n_steps_so_far, 1, self._half_at, self._width)

    @staticmethod
    def _sigmoid(x: float, value_at_inf: float, half_at: float, half_width: float) -> float:
        """
        Sigmoid function.

        Parameters
        ----------

        x : float
            The x-value at which to evaluate the sigmoid function.
        value_at_inf : float
            The upper asymptote m,so that f(x)->m as x->inf.
        half_at : float
            The x-value a at which f(a)=0.5*m.
        half_width : float
            The half-width b, so that f(a-b)=0.25*m and f(a+b)=0.75*m.
        """
        return value_at_inf / (1 + exp(-log(3) / half_width * (x - half_at)))
