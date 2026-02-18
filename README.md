# LagrangianNeuralODEs
AI for physics: Neural ODEs that come from a Lagrangian.

Trains a usual second order ODE, but regularizes it to be an Euler-Lagrange equation using Helmholtz metrics.

### Install

1. Clone the repository
2. Install the preferred version of pytorch.
3. Install the dependencies using

```shell
pip install -r requirements.txt
```

4. Install the local package using

```shell
pip install -e .
```

5. (optional) Install the dev requirements using

```shell
pip install -r requirements_dev.txt
```

### Getting started

Some more detailed, documentation should be available soon. For now, the scripts directory contains examples for usage. The code and documentation for the api is found in src/lanede/api, the core code is found in src/lanede/core.
