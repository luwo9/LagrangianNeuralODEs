"""
lanede
======

A package for Lagrangian Neural ODEs.

Lagrangian Neural ODEs are second order neural odes, that are
regularized to originate from a Lagrangian, by satisfying the
Helmholtz conditions. This package provides a framework to train,
evaluate and predict with these models, aswell as measures for
fulfillment of the Helmholtz conditions.

Subpackages
-----------

lanede.api
    Made to easily create, use, save and load predefined types of
    models.
lanede.visualize
    Submodule for visualizing the results of the models.
lanede.data
    Aimed at obtaining data for the models.
lanede.core
    Core logic for the model and its components, like neural ODEs,
    metrics and training logic. Intended for more advanced use cases.
"""
# TODO: Add appropriate imports here.