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

Additionally, special builder functions are provided for models that do
not fit the API. These return a `LagrangianNeuralODE`, that is, in
particular, they cannot be saved and loaded.

Functions and Classes defined here:

API
---

LanedeAPI
    The main API class for Lagrangian Neural ODE's.
    Provides methods to train, evaluate and predict with predefined
    models.
EXAMPLES
    A dictionary containing example configuration dictionaries for
    predefined models. The keys are the names of the models, the values
    are the example configuration dictionaries.

Special Builders
----------------

standard_douglas_on_fixed
"""

from .api import LanedeAPI, EXAMPLES
from .special_lanede_builders import standard_douglas_on_fixed

__all__ = ["LanedeAPI", "EXAMPLES", "standard_douglas_on_fixed"]
