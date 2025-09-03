"""
Takes metrics computed with `grid_metrics.py` of models trained with
`grid_models.py` and collects them into a single array for comparison.

Creates three files:

- A numpy array with all the metrics for all models, saved to
`OUTPUT_NAME`.
- A json file mapping the model properties to the axes and indices of the array,
saved to `AXES_OUTPUT_NAME`.
- A json file mapping the metrics to their indices in the last axes of
the array, saved to `METRICS_OUTPUT_NAME`.
"""

import pathlib
import json

import numpy as np

# General settings
# As in grid_metrics.py
DIRECTORY = "grid_models1"
BASE_NAME = "oscill_grid"
CONFIG_NAME = "grid_config"

# File to retrieve metrics from
METRIC_NAME = "metrics/metrics"

# TODO: Use pandas multiindex or xarray for this?
# Where to save the collected metrics
OUTPUT_NAME = "grid_models1/metric_array.npy"
AXES_OUTPUT_NAME = "grid_models1/axes_map.json"
METRICS_OUTPUT_NAME = "grid_models1/metrics_map.json"


def main():
    directory = pathlib.Path(DIRECTORY)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {DIRECTORY} does not exist.")

    model_paths = list(directory.glob(f"{BASE_NAME}*"))

    # First load the grid configuration
    config_path = directory / f"{CONFIG_NAME}.json"
    with open(config_path, "r") as f:
        grid_config: dict = json.load(f)

    # Create the corresponding maps for destination in the array
    config_axis_map = {}  # Map config key to axis and index map for that axis
    for i, (key, values) in enumerate(grid_config.items()):
        value_to_index_map = {v: j for j, v in enumerate(values)}
        config_axis_map[key] = {"axis": i, "value_to_index_map": value_to_index_map}

    # Get a valid model to extract metric names
    for model_path in model_paths:
        metric_path = model_path / f"{METRIC_NAME}.json"
        if not metric_path.exists():
            continue
        with open(metric_path, "r") as f:
            metrics: dict = json.load(f)
        metric_names = list(metrics.keys())
        break

    # Now create the map for metrics aswell
    metric_index_map = {name: i for i, name in enumerate(metric_names)}

    # Prepare the array to hold all metrics
    shape = [1.0] * (len(config_axis_map) + 1)
    for key, value in config_axis_map.items():
        axis = value["axis"]
        n_values = len(value["value_to_index_map"])
        shape[axis] = n_values
    shape[-1] = len(metric_index_map)
    all_metrics = np.full(tuple(shape), np.nan)

    # Now loop over all models and fill the array
    for model_path in model_paths:
        metric_path = model_path / f"{METRIC_NAME}.json"
        config_path = model_path / f"{CONFIG_NAME}.json"
        if not metric_path.exists():
            continue
        with open(metric_path, "r") as f:
            metrics: dict = json.load(f)
        with open(config_path, "r") as f:
            config: dict = json.load(f)

        # Access array position by names for config and for each metric
        indices_config = [1.0] * len(config_axis_map)
        for key, value in config.items():
            if key not in config_axis_map:
                continue
            axis = config_axis_map[key]["axis"]
            index = config_axis_map[key]["value_to_index_map"][value]
            indices_config[axis] = index
        for metric_name, metric_value in metrics.items():
            index_metric = metric_index_map[metric_name]
            all_metrics[(*indices_config, index_metric)] = metric_value

    np.save(OUTPUT_NAME, all_metrics)
    with open(AXES_OUTPUT_NAME, "w") as f:
        json.dump(config_axis_map, f, indent=4)
    with open(METRICS_OUTPUT_NAME, "w") as f:
        json.dump(metric_index_map, f, indent=4)


if __name__ == "__main__":
    main()
