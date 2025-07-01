"""
lanede.data.toy
===============

This subpackage provides toy toy models to generate data for testing
Lagrangian Neural ODEs.

Data can be directly generated from a given ODE using the `from_ode`
function. There are predefined ODEs available, and custom ODEs can
be defined by subclassing the `ODE` class.

Also, analytic "neural" ODEs are provided, which can be used in the
core framework. These subclass the `lanede.core.SecondOrderNeuralODE`
class.

Functions and Classes defined here:

General
-------

from_ode
add_noise

ODEs
----

ODE (abstract)
DampedHarmonicOscillator
NonExtremalCaseIIIb
NonExtremalCaseIV
KeplerProblem

Analytic Neural ODEs
--------------------

HarmonicOscillatorODE
CaseIIIbODE
CaseIVODE
KeplerODE
"""

from .odes import (
    from_ode,
    ODE,
    DampedHarmonicOscillator,
    NonExtremalCaseIIIb,
    NonExtremalCaseIV,
    KeplerProblem,
)
from .modify import add_noise
from .analytic_neuralodes import HarmonicOscillatorODE, CaseIIIbODE, CaseIVODE, KeplerODE
