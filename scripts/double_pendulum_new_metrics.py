"""
This script trains a set of a Lagrangian Neural ODEs on a double
pendulum for different time resolutions. It then fits fresh Helmholtz
metrics on the learnt ODE and saves their values for comparison.
"""

import pathlib
import json
import itertools

import numpy as np
import torch.multiprocessing as mp

from lanede.data.toy import from_ode, add_noise, DoublePendulum
from lanede.api import LanedeAPI, standard_douglas_on_fixed

# Main settings
# General settings
BASE_NAME = "double_pendulum"
DIRECTORY = "double_pendulum_helmholtzmetrics1"
CONFIG_NAME = "metric_config"
METRICS_NAME = "helmholtz"

# Settings for the models to train
N_TIME_STEPS = np.geomspace(15, 200, 8, dtype=int).tolist()
N_MODELS_PER_N_TIME_STEPS = 5
DO_UNREGULARIZED = True
EXTRAPOLATION_FACTOR = 2.0

# Technical settings
N_JOBS = 20
N_RESTARTS_MAX = 5

# Training settings
N_EPOCHS_ODE = 600
N_EPOCHS_METRIC = 1000
GET_HIGH_RESOLUTION_N_TIME_STEPS = lambda n_time_steps: int(np.clip(n_time_steps * 3, 250, 500))

# Settings for the data
T_SPAN = 6
NOISE_LEVEL_DATA = 0.05
N_SAMPLES = 6000

length_1 = 1.0
length_2 = 1.0
mass_1 = 1.0
mass_2 = 1.0
g = 9.81

ANGLE_MEANS = np.array([np.pi / 6, np.pi / 6])
ANGLE_STDS = np.array([0.2, 0.2])
VELOCITY_MEANS = np.array([0.0, 0.0])
VELOCITY_STDS = np.array([0.2, 0.2])


def sample_initial_conditions() -> tuple[np.ndarray, np.ndarray]:
    """
    Sample initial conditions for the double pendulum.
    """
    rng = np.random.default_rng()
    x_0 = rng.normal(loc=ANGLE_MEANS, scale=ANGLE_STDS, size=(N_SAMPLES, 2))
    # Make sure they are correlated:
    xdot_0 = VELOCITY_MEANS + VELOCITY_STDS / ANGLE_STDS * (x_0 - ANGLE_MEANS)
    return x_0, xdot_0


def make_lanede_data(n_time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the data for training a Lagrangian Neural ODE.
    """

    data_ode = DoublePendulum(
        length_1=length_1,
        length_2=length_2,
        mass_1=mass_1,
        mass_2=mass_2,
        g=g,
    )

    t_data = np.linspace(0, T_SPAN, n_time_steps)

    x_0, xdot_0 = sample_initial_conditions()
    x_data, *_ = from_ode(data_ode, t_data, x_0, xdot_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA)
    return t_data, x_data


def make_helmholtz_data(
    model: LanedeAPI, n_time_steps: int, extrapolate: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the data for training a Helmholtz metric.

    Parameters
    ----------

    model : LanedeAPI
        The trained Lagrangian Neural ODE model, to generate
        trajectories from.
    n_time_steps : int
        The number of time steps to use.
    extrapolate : bool, default=False
        Whether to return data for extrapolation (beyond T_SPAN),
        by a factor of EXTRAPOLATION_FACTOR.

    Returns
    -------

    np.ndarray
        The time data.
    np.ndarray
        The x data.
    np.ndarray
        The xdot data.
    """
    # For extrapolation also get [0, T_SPAN] such that integration
    # starts at 0.
    t_end = T_SPAN * EXTRAPOLATION_FACTOR if extrapolate else T_SPAN
    n_time_steps = int(n_time_steps * EXTRAPOLATION_FACTOR) if extrapolate else n_time_steps
    t_data = np.linspace(0, t_end, n_time_steps)

    x_0, _ = sample_initial_conditions()
    x_0_noise = add_noise(x_0, NOISE_LEVEL_DATA)

    x_data, xdot_data = model.predict(t_data, x_0_noise)
    # No noise is added as the model prediction is available without
    # noise

    if not extrapolate:
        return t_data, x_data, xdot_data

    # For extrapolation only keep the part beyond T_SPAN
    do_use = t_data > T_SPAN
    t_data = t_data[do_use]
    x_data = x_data[:, do_use]
    xdot_data = xdot_data[:, do_use]
    return t_data, x_data, xdot_data


def train_ode_model(helmholtz_weight: float, n_time_steps: int) -> LanedeAPI:
    """
    Train a model on the double pendulum.
    """

    t_data, x_data = make_lanede_data(n_time_steps)

    # Model settings
    name = "simple_douglas"
    cfg = {
        "dim": 2,
        "explicit_time_dependence_lagrangian": True,
        "learning": {
            "optimizer": "RAdam",
            "lr": 0.07,
            "sheduler_patience": 2000,
            "sheduler_factor": 0.5,
            "sheduler_threshold": 1e-2,
            "half_time_series_steps": 1200 * 3,
        },
        "ode": {
            "activation_fn": "Softplus",
            "hidden_layer_sizes": [32] * 1,
            "rtol": 1e-6,
            "atol": 1e-6,
            "use_adjoint": False,
        },
        "helmholtz": {
            "hidden_layer_sizes": [100] * 2,
            "activation_fn": "Softplus",
            "total_weight": helmholtz_weight,
            "condition_weights": [1.0, 1.0],
        },
        "initial_net": {
            "hidden_layer_sizes": [16] * 3,
            "activation_fn": "ReLU",
        },
        "normalizer": {
            "type": "MeanStd",
        },
    }
    # Allow restarts if training fails.
    n_restarts = 0
    while n_restarts <= N_RESTARTS_MAX:
        try:
            model = LanedeAPI(name, cfg)
            model.train(t_data, x_data, n_epochs=N_EPOCHS_ODE, batch_size=128, device="cpu")
            return model
        except Exception as e:
            n_restarts += 1
            print(f"Restarting training {n_restarts}/5 due to error: {e}", flush=True)
    return None


def train_helmholtz_metric(
    model: LanedeAPI, n_time_steps: int, extrapolate: bool = False
) -> np.ndarray:
    """
    Train a Helmholtz metric on the learnt ODE of a model. If
    extrapolate is True only trains on data beyond T_SPAN.
    """

    # TODO: Expose this in the API
    ode = model._model._model._neural_ode._neural_ode
    normalizer = model._model._normalizer

    t_data, x_data, xdot_data = make_helmholtz_data(model, n_time_steps, extrapolate)

    # Create the fixed ODE model
    metric_model = standard_douglas_on_fixed(
        ode,
        normalizer,
        2,
        [100] * 2,
        init_lr=3e-2,
        scheduler_options={"patience": 2000, "factor": 0.5, "threshold": 1e-2},
        ode_options={"rtol": 1e-6, "atol": 1e-6, "use_adjoint": False},  # Not really needed here
        metric_options={"supress_time_dependence": False},
    )
    try:
        info = metric_model.train(
            t_data,
            x_data,
            xdot_data,
            batch_size=128,
            n_epochs=N_EPOCHS_METRIC,
            device="cpu",
        )

        t_test, x_data_test, xdot_data_test = make_helmholtz_data(model, n_time_steps, extrapolate)
        t_test_with_batches = np.tile(t_test, (x_data_test.shape[0], 1))
        helmholtz_metric = metric_model.helmholtzmetric(
            t_test_with_batches, x_data_test, xdot_data_test
        )
    except Exception as e:
        # If training fails, return -1.0 as the metric value to indicate failure.
        return -1.0, -1.0
    # Return train and test metric
    return info["helmholtz"][-1], helmholtz_metric.item()


def train_worker(helmholtz_weight: float, n_time_steps: int, index: int):
    """
    Worker function to train a model, evaluate the Helmholtz metric and
    save the results.
    """
    model_path = f"{DIRECTORY}/{BASE_NAME}_{n_time_steps}po_{helmholtz_weight}h_ens{index}"
    n_time_steps_highres = GET_HIGH_RESOLUTION_N_TIME_STEPS(n_time_steps)
    # Train Lagrangian Neural ODE
    model = train_ode_model(helmholtz_weight, n_time_steps)
    print(f"Trained model {helmholtz_weight} {n_time_steps} {index}", flush=True)
    if model is not None:
        model.save(model_path)

        # Fit Helmholtz metrics on the learnt ODE
        # np.linspace is called with the same arguments and thus gives the
        # same time points as in training (if not extrapolating)
        train_points = train_helmholtz_metric(model, n_time_steps)
        print(
            f"Fitted normal resolution Helmholtz metric {helmholtz_weight} {n_time_steps} {index}",
            flush=True,
        )
        high_resolution = train_helmholtz_metric(model, n_time_steps_highres)
        print(
            f"Fitted high resolution Helmholtz metric {helmholtz_weight} {n_time_steps} {index}",
            flush=True,
        )
        extrapolated = train_helmholtz_metric(model, n_time_steps_highres, extrapolate=True)
        print(
            f"Fitted extrapolated Helmholtz metric {helmholtz_weight} {n_time_steps} {index}",
            flush=True,
        )

        # Save the results
        helmholtz_train, helmholtz_test = train_points
        highres_helmholtz_train, highres_helmholtz_test = high_resolution
        extrap_helmholtz_train, extrap_helmholtz_test = extrapolated
    else:
        print(f"Failed model {helmholtz_weight} {n_time_steps} {index}", flush=True)
        # Still make the model directory, to save the artificial failed
        # model results
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        helmholtz_train, helmholtz_test = -1.0, -1.0
        highres_helmholtz_train, highres_helmholtz_test = -1.0, -1.0
        extrap_helmholtz_train, extrap_helmholtz_test = -1.0, -1.0
    metrics = {
        "helmholtz_train": helmholtz_train,
        "helmholtz_test": helmholtz_test,
        "highres_helmholtz_train": highres_helmholtz_train,
        "highres_helmholtz_test": highres_helmholtz_test,
        "extrap_helmholtz_train": extrap_helmholtz_train,
        "extrap_helmholtz_test": extrap_helmholtz_test,
    }
    config = {
        "n_time_steps": n_time_steps,
        "helmholtz_weight": helmholtz_weight,
    }
    with open(f"{model_path}/{METRICS_NAME}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(f"{model_path}/{CONFIG_NAME}.json", "w") as f:
        json.dump(config, f, indent=4)

    return config, metrics


def process_results(results: list) -> dict:
    # Parse a list of train_worker results into a dictionary of lists
    n_different_time_steps = len(N_TIME_STEPS)
    time_steps_index_map = {n: i for i, n in enumerate(N_TIME_STEPS)}
    out_dict = {"regularized": {}, "unregularized": {}}
    for config, metric in results:
        n_time_steps = config["n_time_steps"]
        i_time_steps = time_steps_index_map[n_time_steps]
        helmholtz_weight = config["helmholtz_weight"]
        regularization = "regularized" if helmholtz_weight > 0.0 else "unregularized"
        for metric_name, value in metric.items():
            metric_dict = out_dict[regularization].setdefault(
                metric_name, [[] for _ in range(n_different_time_steps)]
            )
            metric_dict[i_time_steps].append(value)
    return out_dict


def main():
    # Create the directory if it does not exist
    directory = pathlib.Path(DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)

    # Create model settings
    helmholtz_weights = [1.0, 0.0] if DO_UNREGULARIZED else [1.0]
    run_numbers = range(N_MODELS_PER_N_TIME_STEPS)

    settings = list(itertools.product(helmholtz_weights, N_TIME_STEPS, run_numbers))
    # Run different trials last, to get information about the results
    # early on.
    settings = sorted(settings, key=lambda x: (x[2], x[1], x[0]))

    ctx = mp.get_context("spawn")
    with ctx.Pool(N_JOBS) as pool:
        results = pool.starmap(train_worker, settings)

    # Directly save a convenient version of the results:
    metrics = process_results(results)

    if not DO_UNREGULARIZED:
        metrics = metrics["regularized"]

    global_config = {
        "n_time_steps": N_TIME_STEPS,
        "do_unregularized": DO_UNREGULARIZED,
    }
    with open(f"{DIRECTORY}/{METRICS_NAME}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(f"{DIRECTORY}/{CONFIG_NAME}.json", "w") as f:
        json.dump(global_config, f, indent=4)


if __name__ == "__main__":
    main()
