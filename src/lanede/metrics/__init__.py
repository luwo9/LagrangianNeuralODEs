"""
lanede.metrics
==============

This subpackage provides metrics for neural ODEs, and (predicted) time
series.

Functions and Classes defined here:

General:
--------

fourier_metrics
fourier_mse
domain_mse

System specific:
---------------------------

oscillator_energy_mse
"""

from .fourier import fourier_metrics, fourier_mse
from .system_quantities import oscillator_energy_mse
from .odes import domain_mse

__all__ = ["fourier_metrics", "fourier_mse", "oscillator_energy_mse", "domain_mse"]
