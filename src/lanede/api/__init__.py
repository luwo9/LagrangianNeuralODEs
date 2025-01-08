"""
`lanede.api`
============

This submodule provides an easy-to-use interface for defining and using
Lagrangian Neural ODE's.

The main component is the `LanedeAPI` class, which provides methods to
train, evaluate and predict with predefined models. The interface for
usage is simmilar to `lanede.core.LagrangianNeuralODE`, but models can
be saved and loaded directly and easily configured from a dictionary.

The names for the predefined configurablle models, together with
corresponding example configuration dictionaries are defined in
the `EXAMPLES` dictionary.
"""
from .api import LanedeAPI, EXAMPLES

__all__ = ["LanedeAPI", "EXAMPLES"]