"""
lanede.data.toy
===============

This subpackage provides toy toy models to generate data for testing
Lagrangian Neural ODEs.

Data can be directly generated from the ODEs using the `from_ode`
function. There are predefined ODEs available, and custom ODEs can
be defined by subclassing the `ODE` class.

Functions and Classes defined here:

General
-------

add_noise

ODEs
----

ODE (abstract)
DampedHarmonicOscillator
"""

from .odes import from_ode, ODE, DampedHarmonicOscillator
from .modify import add_noise
