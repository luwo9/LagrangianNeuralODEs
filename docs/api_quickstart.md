# API Quickstart

An easy-to-use API is provided in `lanede.api` to construct, train, evaluate, load and save Lagrangian Neural ODEs with minimal code. The main component of the API is the `LanedeAPI` class, which provides methods for all these functionalities. To this end, several predefined model types are implemented, which can be easily configured using only a configuration dictionary.

## Defining a model
The API model object is created using the preset model name and the corresponding configuration dictionary, that allows to specify, e.g., network size and regularization strength:

```python
from lanede.api import LanedeAPI, EXAMPLES
name = "simple_douglas"
config = EXAMPLES[name]
model = LanedeAPI(name, config)
```

All available presets and their configuration dictionaries are explained in the [Model Overview](docs/model_overview.md). For every preset model type, an example configuration dictionary is provided in the `EXAMPLES` dictionary.

## Data
The model is now configured and ready to be trained, given the positional data (shape (n_trajectories, n_points, n_dim)) and the time points the positional data is observed at (shape (n_points,)).

For testing purposes, toy systems are available in `lanede.data.toy`. E.g., one can create a harmonic oscillator ODE:

```python
from lanede.data.toy import DampedHarmonicOscillator
n_dim = 2
spring_matrix = np.eye(n_dim)*(2*np.pi)**2
damping_matrix = np.zeros((n_dim, n_dim))
ode = DampedHarmonicOscillator(spring_matrix, damping_matrix)
```

Given initial conditions and time points to obtain the trajectories at, the data can be generated via integration and noise can be added:

```python
from lanede.data.toy import from_ode, add_noise
rng = np.random.default_rng()

n_points = 7
n_trajectories = 6000
t_data = np.linspace(0, 1, n_points)
x_0 = 1 + rng.normal(size=(6000, 2)) / 10
xdot_0 = np.sqrt(x_0**2) / 10

x_data, *_ = from_ode(oscillator, t_data, x_0, xdot_0)
x_data = add_noise(x_data, 0.05)
```

If using no velocity data, in particular no initial velocity data, the models will usually attempt to predict the initial velocity from the initial position, so they are chosen to be correlated in this example.

## Training, saving and loading
The model can be trained using the `train` method, which requires the time points, the positional data and some training settings:

```python
losses = model.train(t_data, x_data, n_epochs=200, batch_size=128, device="cpu")
```

Training on cpu is usually sufficient for the toy systems. In the background, calling `train` will do two things automatically: 1) It fits a normalizer to the data and applies its transformations to the data. This normalisation is automatically applied for all inputs and outputs of prediciton methods. However, quantities such as the training error, the helmholtz metric and the adaptive solver error tolerance are all given in the normalised space, without applying the inverse transformation. 2) It saves the loss values internally and also returns them as a dictionary at the end of training.

A trained model can be saved using the `save` method, given a path:
```python
model.save("path/to/save")
```
If the path ends with `.pt`, the model will be saved as a single pytorch file (for portability), otherwise it is saved as a directory (recommended). The latter allows to create new subdirectories for, e.g., plots and saves losses and configuration in txt and json files, which makes them accessible without loading the model.

A saved model can be loaded using the `load` classmethod, given only the path it was saved to, with no knowledge of the model type or configuration required. The losses of the loaded model are still accessible via the `losses` attribute:

```python
model = LanedeAPI.load("path/to/save")
print(model.losses)
```

## Inference
Once the `train` method has been called (and the normalizer has been fitted), the model can predict the trajectory given the time points (shape (n_points,)) and the initial conditions (shape (n_trajectories, n_dim)) of the position:

```python
x_pred, xdot_pred = model.predict(t_data, x_data[:, 0, :])
```

The acceleration can also be predicted using the `second_derivative` method, requiring the time points, positions and velocities. Here, the time points need to be supplied in the shape (n_trajectories, n_points):
```python
t_w_batch = np.tile(t_data, (n_trajectories, 1))
xddot_pred = model.second_derivative(t_w_batch, x_pred, xdot_pred)
```

Further, the prediction error of the model and the helholtz metric can be evaluated using the `error` and `helmholtz_metric` methods, respectively.

## Special lanede builders
There are some special predefined models that are not compatible with the general API. These models are created with special builders, that are functions with fixed keyword arguments and are given by an instance of `lanede.core.LagrangianNeuralODE`. These objects have the same train and evaluation methods, but miss additional features, in particular saving, loading and configuration dictionaries.

## Final remarks
Note that there may be models (presets) that do not use position only for training and prediction, but also velocity data. In this case, the `train` and `predict` methods will require the velocity data as well. For more details on the available models and their requirements, see the [Model Overview](docs/model_overview.md).