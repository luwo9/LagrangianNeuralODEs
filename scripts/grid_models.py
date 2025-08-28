"""
This script trains models on a grid of hyperparameters.
"""

import pathlib
import json
import itertools
from multiprocessing.sharedctypes import Synchronized
from queue import Empty
import time

import numpy as np
import torch.multiprocessing as mp

from lanede.api import LanedeAPI
from lanede.data.toy import from_ode, DampedHarmonicOscillator, add_noise

# General settings
BASE_NAME = "oscill_grid"
DIRECTORY = "grid_models1"
CONFIG_NAME = "grid_config"

IS_RERUN = False  # Set to True if resuming a previous run

# Training settings
DEVICES = ["cpu"] * 45  # E.g. ["cuda:0", "cuda:0", "cuda:1", "cpu"]
N_RESTARTS = 12
N_EPOCHS = 600
BATCH_SIZE = 128

DO_PLOT = False

# General oscillator settings
NOISE_LEVEL_DATA = 0.05

# Grid settings, must be JSON serializable
GRID = {
    # The number of time steps is computed based on the number of
    # periods. The values below are just the bin numbers for the grid.
    "time_points_bin_nb": list(range(7)),
    "n_periods": [1, 2, 3, 4, 5],
    "x_0_std": [0.1, 0.25, 0.6],
    "damping": [0.0, 0.25],
    "time_gap": [True, False],
    "helmholtz_weight": [1, 0],
    "run_number": list(range(1, 4)),
}


def get_n_time_steps(n_periods):
    """
    Get the number of time steps based on the number of periods.
    """
    n_bins = len(GRID["time_points_bin_nb"])
    # The sampling rate must be greater than two times the frequency to
    # avoid aliasing (Nyquist-Shannon sampling theorem).
    # See e.g. https://en.wikipedia.org/wiki/Nyquist-Shannon_sampling_theorem
    shannon_nyquist = 2 * n_periods
    n_points_min = max(0.8 * shannon_nyquist, 5)
    n_points_max = 15 + 8 * (n_periods - 1)
    n_time_points = np.logspace(np.log10(n_points_min), np.log10(n_points_max), n_bins)
    n_time_points = np.round(n_time_points).astype(int)
    n_time_points = np.unique(n_time_points)  # Remove duplicates
    if len(n_time_points) < n_bins:
        raise ValueError(
            f"Not enough points for n_periods={n_periods}, got {len(n_time_points)} points."
        )
    return n_time_points.tolist()


# This will also assure that the error is raised already here if the
# time point bins are not unique:
N_TIME_STEP_MAP = {n_periods: get_n_time_steps(n_periods) for n_periods in GRID["n_periods"]}


def get_time_steps(time_points_bin_nb, n_periods, time_gap):
    """
    Get the time steps from the grid settings.
    """
    n_time_points = N_TIME_STEP_MAP[n_periods][time_points_bin_nb]
    if not time_gap:
        return np.linspace(0, 1, n_time_points)

    # Gap size is half the number of periods, but at most 2 periods.
    n_periods_gap = np.clip(n_periods / 2, None, 2)
    fraction_data = 1 - n_periods_gap / n_periods
    n_points_after_gap = n_time_points // 2
    n_points_before_gap = n_time_points - n_points_after_gap
    t_before_gap = np.linspace(0, fraction_data / 2, n_points_before_gap)
    t_after_gap = np.linspace(1 - fraction_data / 2, 1, n_points_after_gap)
    t_data = np.concatenate((t_before_gap, t_after_gap))
    return t_data


# Define model training function
def train_model(
    time_points_bin_nb: int,
    n_periods: int,
    x_0_std: float,
    damping: float,
    time_gap: bool,
    helmholtz_weight: int,
    run_number: int,
    device: str,
) -> float:
    # Args like in grid
    save_id = (
        f"{time_points_bin_nb}t_{n_periods}p_{x_0_std}x_{damping}d_gap{time_gap}_h"
        f"{helmholtz_weight}_ens{run_number}"
    )
    save_path = f"{DIRECTORY}/{BASE_NAME}_{save_id}"
    if pathlib.Path(save_path).exists():
        return 0  # Skip if already exists

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
            "half_time_series_steps": 1200,
        },
        "ode": {
            "activation_fn": "Softplus",
            "hidden_layer_sizes": [16] * 1,
            "rtol": 1e-6,
            "atol": 1e-6,
            "use_adjoint": False,
        },
        "helmholtz": {
            "hidden_layer_sizes": [64] * 2,
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

    omega = 2 * np.pi * n_periods
    # fmt: off
    spring_matrix = np.array([[omega**2, 0],
                              [0, omega**2]])

    damping_matrix = np.array([[omega, 0],
                              [0, omega]])*damping
    # fmt: on

    # Data
    rng = np.random.default_rng()
    oscillator = DampedHarmonicOscillator(spring_matrix, damping_matrix)
    t_data = get_time_steps(time_points_bin_nb, n_periods, time_gap)

    x_0 = 1 + rng.normal(size=(6000, 2)) * x_0_std
    v_0 = np.sqrt(x_0**2) / 10
    x_data, *_ = from_ode(oscillator, t_data, x_0, v_0)
    x_data = add_noise(x_data, NOISE_LEVEL_DATA)

    model = LanedeAPI(name, cfg)
    start_time = time.perf_counter()
    model.train(t_data, x_data, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=device)
    end_time = time.perf_counter()
    model.save(save_path)

    config_dict = {
        "time_points_bin_nb": time_points_bin_nb,
        "n_periods": n_periods,
        "x_0_std": x_0_std,
        "damping": damping,
        "time_gap": time_gap,
        "helmholtz_weight": helmholtz_weight,
        "run_number": run_number,
        "time_points": t_data.tolist(),
    }
    with open(f"{save_path}/{CONFIG_NAME}.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    return end_time - start_time


# Logic for multi-process training:
def train_worker(
    queue: mp.Queue,
    device: str,
    n_restarts_total: Synchronized,
    n_finished_total: Synchronized,
    n_runs_total: int,
):
    """
    Woker to train models obtained from a queue on a fixed device.
    Restarts the training if it fails.
    """
    print_at = [int(n_runs_total * i / 10) for i in range(1, 11)]
    while True:
        try:
            grid_combination = queue.get(timeout=1)
        except Empty:
            print(f"Device {device} finished all tasks.")
            return

        for n_restart in range(N_RESTARTS + 1):  # +1 to include the first try
            try:
                time_span = train_model(**grid_combination, device=device)
                print(
                    f"Successfully trained model {grid_combination} after {n_restart} restarts on "
                    f"device {device}. Time taken: {time_span/3600:.1f} hours."
                )
                break  # Exit the restart loop if training was successful
            except Exception as e:
                if n_restart == N_RESTARTS:
                    print(
                        f"Model {grid_combination} failed after {N_RESTARTS}restarts. Skipping "
                        "this model."
                    )
                    break

        # += is not atomic, use the lock explicitly
        with n_restarts_total.get_lock():
            n_restarts_total.value += n_restart

        with n_finished_total.get_lock():
            n_finished_total.value += 1
            n_tot = n_finished_total.value
            if n_tot in print_at:
                print(
                    f"Finished {n_tot}/{n_runs_total} models "
                    f"({int((n_tot / n_runs_total) * 100)}%)."
                )


def main():
    if DO_PLOT:
        # Keep script shorter:
        raise NotImplementedError("Plotting is not implemented yet.")

    base_path = pathlib.Path(DIRECTORY)
    base_path.mkdir(parents=True, exist_ok=IS_RERUN)
    # Save grid configuration
    with open(f"{DIRECTORY}/{CONFIG_NAME}.json", "w") as f:
        json.dump(GRID, f, indent=4)

    # Make grid of all combinations
    grid_combinations = itertools.product(*GRID.values())
    grid_combinations = [{k: v for k, v in zip(GRID.keys(), comb)} for comb in grid_combinations]
    n_runs = len(grid_combinations)

    print(f"Running {n_runs} models on {len(DEVICES)} devices.")
    print(f"Using {N_RESTARTS} restarts per model.")

    # Set up multiprocessing
    # To be more device safe:
    ctx = mp.get_context("spawn")

    task_queue = ctx.Queue()
    n_restarts_total = ctx.Value("i", 0)
    n_finished_total = ctx.Value("i", 0)

    for grid_combination in grid_combinations:
        task_queue.put(grid_combination)

    start_time = time.perf_counter()
    print("Starting training...")

    processes = []
    for device in DEVICES:
        p = ctx.Process(
            target=train_worker,
            args=(task_queue, device, n_restarts_total, n_finished_total, n_runs),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.perf_counter()
    print("All models finished training.")
    print(f"Successfully finished models: {n_finished_total.value}/{n_runs}")
    print(f"Skipped models: {n_runs - n_finished_total.value}/{n_runs}")
    print(f"Total restarts: {n_restarts_total.value}")
    print(f"Total time taken: {(end_time - start_time) / (3600* 24):.1f} days.")


if __name__ == "__main__":
    main()
