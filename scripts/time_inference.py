"""
This script measures the inference time of LanedeAPI models.
"""

from timeit import timeit
from functools import partial

import numpy as np

from lanede.api import LanedeAPI

# Model to time
MODEL_NAME = "oscillator_model_1"

# Mode: "integate" or "acceleration"
MODE = "integrate"

# Timing settings
N_TRAJECTORIES = 512
N_RUNS = 20

# Inference settings
T_START = 0
T_END = 1
N_POINTS = 200

# Initial conditions to use for prediction.
rng = np.random.default_rng()
# Example for an oscillator:
x_0 = 1 + rng.normal(size=(N_TRAJECTORIES, 2)) / 10


# Logic
path = f"saves/{MODEL_NAME}"


def integrate(
    model: LanedeAPI, t_eval: np.ndarray, x_0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return model.predict(t_eval, x_0)


def acceleration(
    model: LanedeAPI, t_eval: np.ndarray, x: np.ndarray, xdot: np.ndarray
) -> np.ndarray:
    t_eval_with_batches = np.tile(t_eval, (x.shape[0], 1))
    return model.second_derivative(t_eval_with_batches, x, xdot)


def main():
    model = LanedeAPI.load(path)
    t_eval = np.linspace(T_START, T_END, N_POINTS)
    get_trajectories = partial(integrate, model, t_eval, x_0)
    if MODE == "integrate":
        func = get_trajectories
    elif MODE == "acceleration":
        x_pred, xdot_pred = get_trajectories()
        func = partial(acceleration, model, t_eval, x_pred, xdot_pred)
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    time = timeit(func, number=N_RUNS) / N_RUNS
    print(f"Model: {MODEL_NAME}")
    print(f"Average time over {N_RUNS} runs: {time:.4f} seconds.")


if __name__ == "__main__":
    main()
