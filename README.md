# LagrangianNeuralODEs
Lagrangian Neural ODEs refer to neural ordinary differential equations (see [the Neural ODE paper](https://arxiv.org/abs/1806.07366)) that are of second order and originate from a Lagrangian. They allow to learn an Euler-Lagrange equation from positional data and thus are able to predict the full dynamics of the system, including the velocity and acceleration at any given time. While the Euler-Lagrange characteristic may be enforced by construction, plain Neural ODEs can also be regularized to originate from a Lagrangian, where ***Helmholtz metrics*** are used to quantify how well the learned ODE resembles an Euler-Lagrange equation. The latter approach is the key innovation of our work and this package.

The `lanede` (**La**grangian **Ne**ural O**DE**) python package provides a framework for these models, including:
- An API to construct, train, evaluate, load and save Lagrangian Neural ODEs with minimal code.
- Implementations of Helmholtz metrics.
- A modular codebase to easily extend the package with new features, such as new metrics or model architectures.
- Tools to generate toy data, evaulate model performance and visualize results.

## Install

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

## API Quickstart

The API allows to create a predefined model type using only a model name and a configuration dictionary:

```python
from lanede.api import LanedeAPI, EXAMPLES

name = "simple_douglas"
config = EXAMPLES[name]
model = LanedeAPI(name, config)
```
The `EXAMPLES` dictionary contains predefined configurations for every model type. For all available model types and their configurations, see [Model Overview](docs/model_overview.md).

The model can then be trained using the `train` method, requiring the time points (shape (n_points,)), the positional data (shape (n_trajectories, n_points, n_dim)) and some training settings:

```python
model.train(t_data, x_data, n_epochs=200, batch_size=128, device="cpu")
```

The model can then be saved and loaded using the `save` and `load` methods:

```python
model.save("path/to/save")
model = LanedeAPI.load("path/to/save")
```

Inference can be done using the `predict` method, which requires the time points (shape (n_points,)) and the initial conditions (shape (n_trajectories, n_dim)):

```python
x_pred, xdot_pred = model.predict(t_data, x_data[:, 0, :])
```

For a more detailed tutorial on how to use the API, see [API Quickstart](docs/api_quickstart.md) or the [example scripts](scripts/).

## Documentation

A full documentation is work in progress. For now, you can find some documentation in [docs/](docs/) and docstrings in the code.

## Citation and Papers

If you use this code or the ideas arround the Helmholtz metric in your work, cite the following paper:

```
@misc{wolf2025lagrangianneuralodesmeasuring,
      title={Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics}, 
      author={Luca Wolf and Tobias Buck and Bjoern Malte Schaefer},
      year={2025},
      eprint={2510.06367},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.06367}, 
}
```

In [this paper](https://arxiv.org/abs/2510.06367), we presented the Helmholtz metrics and first results of Lagrangian Neural ODEs regularized with these metrics. We are currently working on a follow-up paper, which will discuss the latter in more detail and compare it to other existing approaches.