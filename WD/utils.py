import torch
import subprocess
from pathlib import Path
import xarray as xr
import numpy as np
import uuid
# import h5py

from torch import tensor


def check_devices():
    if torch.cuda.is_available() or 1:
        print(f"Number of Devices {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Cuda is not available")


def get_git_revision_hash() -> str:
    grv_dm_zoo = (
        subprocess.check_output(["git", "rev-parse", "HEAD:dm_zoo"])
        .decode("ascii")
        .strip()
    )
    grv_weather_diff = (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )

    return [grv_dm_zoo, grv_weather_diff]


def generate_uid():
    return uuid.uuid4().hex.upper()[0:6]


def transformation_function(x, eps=0.001):
    return np.log1p(x / eps)


def inverse_transformation_function(y, eps=0.001):
    return eps * np.expm1(y)


def transform_precipitation(pr: xr.Dataset) -> xr.Dataset:
    """Apply a transformation to the precipitation values to make training easier.
    For now, follow Rasp & Thuerey

    Args:
        pr (xr.Dataset): The precipitation array on the original scale

    Returns:
        xr.Dataset: A rescaled version of the precipitation array
    """  # noqa: E501
    return xr.apply_ufunc(
        transformation_function,
        pr["tp"],
        dask="parallelized",
    )


def inverse_transform_precipitation(
    pr_transform: xr.Dataset,
) -> xr.Dataset:
    """Undo the precipitation transformation.

    Args:
        pr_transform (xr.Dataset): The rescaled precipitation array

    Returns:
        xr.Dataset: The precipitation, rescaled to the original resolution
    """
    return xr.apply_ufunc(
        inverse_transformation_function,
        pr_transform["tp"],
        dask="parallelized",
    )


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def n_generated_channels(config):
    n_level = 0
    for k, v in config.data_specs.output_vars.items():
        n_level = (
            n_level + len(v.levels) if v.levels is not None else n_level + 1
        )

    return n_level


def n_condition_channels(config):
    n_level = 0
    for k, v in config.data_specs.conditioning_vars.items():
        n_level = (
            n_level + len(v.levels) if v.levels is not None else n_level + 1
        )
    n_level = n_level * len(config.data_specs.conditioning_time_step)
    n_level = (
        n_level + len(config.data_specs.constant_vars)
        if config.data_specs.constant_vars is not None
        else n_level
    )
    return n_level

"""
def norm_area(deg=5.625):
    with h5py.File("/data/compoundx/WeatherBench/gridarea.h5", "r") as f:
        return f[f"norm_gridarea_{int(360/deg)}x{int(180/deg)}"][:]
"""
"""
class AreaWeightedMSELoss:
    def __init__(self, spatial_res):
        deg = float(spatial_res[:-3])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = torch.tensor(norm_area(deg=deg), device=device)

    def loss_fn(self, input: tensor, target: tensor):
        return torch.sum(self.weight * (input - target) ** 2)
"""

class WeightedMSELoss:
    def __init__(self, weights):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights = torch.tensor(weights, device=device)   

    def loss_fn(self, input: tensor, target: tensor):
        return (self.weights * (input - target) ** 2).mean()

class AreaWeightedMSELoss(WeightedMSELoss):
    def __init__(self, lat, lon):
        super().__init__(weights=comp_area_weights_simple(lat, lon))
    


def comp_area_weights_simple(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """An easier way to calculate the (already normalized) area weights.

    Args:
        lat (np.ndarray): Array of latitudes of grid center points
        lon (np.ndarray): Array of lontigutes of grid center points

    Returns:
        np.ndarray: 2d array of relative area sizes.
    """
    area_weights = np.cos(lat * (2 * np.pi) / 360)
    area_weights = area_weights.reshape(-1, 1).repeat(lon.shape[0],axis=-1)
    area_weights = (lat.shape[0]*lon.shape[0] / np.sum(area_weights)) * area_weights
    return area_weights