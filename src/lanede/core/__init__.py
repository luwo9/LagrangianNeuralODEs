"""
`lanede.core`
=================

This subpackage contains the main logic for Lagrangian Neural ODE's.

It provides all necessary classes to create a `LagrangianNeuralODE`
model, that provides an interface to train and evaluate a Lagrangian
Neural ODE with dedicated methods.

Included in the logic are neural ODE's, metrics for the Helmholtz
conditions and prediction and training logic.

Functions and Classes defined here:

Neural Networks:
----------------

NeuralNetwork

Neural ODE's:
------------

SecondOrderNeuralODE (abstract)
SolvedSecondOrderNeuralODE
FreeSecondOrderNeuralODE

Helmholtz Metrics:
------------------

HelmholtzMetric (abstract)
TryLearnDouglas

Lagrangian Neural ODE Models:
-----------------------------

LagrangianNeuralODEModel (abstract)
SimultaneousLearnedDouglasOnlyX

Lagrangian Neural ODE:
----------------------

LagrangianNeuralODE

Training:
---------

train_lagrangian_neural_ode

Data Normalization:
-------------------

Normalizer (abstract)
MeanStd

Semi-public functions and classes:

make_dataloader
TrainingInfo
"""

from .neural import NeuralNetwork
from .neuralodes import SecondOrderNeuralODE, FreeSecondOrderNeuralODE
from .integratedodes import SolvedSecondOrderNeuralODE
from .helmholtzmetrics import HelmholtzMetric, TryLearnDouglas
from .lanedemodels import LagrangianNeuralODEModel, SimultaneousLearnedDouglasOnlyX
from .lanede import LagrangianNeuralODE, make_dataloader
from .training import train_lagrangian_neural_ode, TrainingInfo
from .normalize import Normalizer, MeanStd

__all__ = [
    "NeuralNetwork",
    "SecondOrderNeuralODE",
    "FreeSecondOrderNeuralODE",
    "SolvedSecondOrderNeuralODE",
    "HelmholtzMetric",
    "TryLearnDouglas",
    "LagrangianNeuralODEModel",
    "SimultaneousLearnedDouglasOnlyX",
    "LagrangianNeuralODE",
    "train_lagrangian_neural_ode",
    "Normalizer",
    "MeanStd",
]
