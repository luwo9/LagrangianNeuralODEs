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
LNNNeuralNetwork

Neural ODE's:
------------

SecondOrderNeuralODE (abstract)
SolvedSecondOrderNeuralODE
FreeSecondOrderNeuralODE
EulerLagrangeNeuralODE

Helmholtz Metrics:
------------------

HelmholtzMetric (abstract)
TryLearnDouglas
DummyHelmholtzMetric

Lagrangian Neural ODE Models:
-----------------------------

LagrangianNeuralODEModel (abstract)
SimultaneousLearnedMetricOnlyX
DouglasOnFixedODE

Lagrangian Neural ODE:
----------------------

LagrangianNeuralODE

Training:
---------

train_lagrangian_neural_ode
TemporalScheduler (abstract)
SigmoidTemporalScheduler

Data Normalization:
-------------------

Normalizer (abstract)
MeanStd
Identity

Semi-public functions and classes:

make_dataloader
TrainingInfo
"""

from .neural import NeuralNetwork, LNNNeuralNetwork
from .neuralodes import SecondOrderNeuralODE, FreeSecondOrderNeuralODE, EulerLagrangeNeuralODE
from .integratedodes import SolvedSecondOrderNeuralODE
from .helmholtzmetrics import HelmholtzMetric, TryLearnDouglas, DummyHelmholtzMetric
from .lanedemodels import (
    LagrangianNeuralODEModel,
    SimultaneousLearnedMetricOnlyX,
    DouglasOnFixedODE,
)
from .lanede import LagrangianNeuralODE, make_dataloader
from .training import train_lagrangian_neural_ode, TrainingInfo
from .temporal_schedulers import TemporalScheduler, SigmoidTemporalScheduler
from .normalize import Normalizer, MeanStd, Identity

__all__ = [
    "NeuralNetwork",
    "LNNNeuralNetwork",
    "SecondOrderNeuralODE",
    "FreeSecondOrderNeuralODE",
    "EulerLagrangeNeuralODE",
    "SolvedSecondOrderNeuralODE",
    "HelmholtzMetric",
    "TryLearnDouglas",
    "DummyHelmholtzMetric",
    "LagrangianNeuralODEModel",
    "SimultaneousLearnedMetricOnlyX",
    "DouglasOnFixedODE",
    "LagrangianNeuralODE",
    "train_lagrangian_neural_ode",
    "TemporalScheduler",
    "SigmoidTemporalScheduler",
    "Normalizer",
    "MeanStd",
    "Identity",
]
