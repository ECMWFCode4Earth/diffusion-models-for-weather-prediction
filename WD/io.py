from typing import List

from omegaconf import DictConfig

import xarray as xr
import numpy as np
import pandas as pd

import torch
from datetime import datetime, timedelta

from WD.utils import (
    get_git_revision_hash,
    inverse_transform_precipitation,
)
import os
import yaml
# from munch import Munch


def create_xr(data: np.array, var: List, data_description: str):
    start_dt = datetime(2000, 1, 1)
    lon_deg = 360 / data.shape[3]
    lat_deg = 180 / data.shape[2]

    assert lon_deg == lat_deg
    deg = lon_deg
    lon = np.arange(0, 360, deg) + deg / 2
    lat = np.arange(-90, 90, deg) + deg / 2
    time_series = [start_dt + timedelta(days=i) for i in range(data.shape[0])]
    ds = xr.Dataset(
        data_vars={
            var[i]: (
                ["time", "lat", "lon"],
                data[:, i, :, :],
            )
            for i in range(data.shape[1])
        },
        coords=dict(
            lon=(["lon"], lon),
            lat=(["lat"], lat),
            time=time_series,
            reference_time=start_dt,
        ),
        attrs=dict(description=data_description),
    )

    return ds


def write_config(config):
    if type(config) == dict:
        config = Munch(config)

    dm_zoo, weather_diff = get_git_revision_hash()

    config.git_rev_parse = (
        Munch()
    )  # need to initialize this subthing first, otherwise get errors below.
    config.git_rev_parse.dm_zoo = dm_zoo
    config.git_rev_parse.WeatherDiff = weather_diff

    fname = config.ds_id
    if "model_id" in config:
        fname = fname + "_" + config.model_id
        print("Writing model configuration file.")
    else:
        print("Writing dataset configuration file.")

    path = f"/data/compoundx/WeatherDiff/config_file/{fname}.yml"

    config_dict = config.toDict()

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    os.chmod(path, 0o444)

    print(f"File {fname}.yml written (locked in read-only mode).")


def load_config(config):
    if type(config) == str:
        with open(config) as f:
            config = yaml.safe_load(f)
    config = Munch.fromDict(config)
    if "data_specs" in config:
        config.n_generated_channels = n_generated_channels(config)
        config.n_condition_channels = n_condition_channels(config)

    return config


def n_generated_channels(config):
    ov = config.data_specs.output_vars
    n_level = 0
    for k, v in ov.items():
        if v is None:
            n_level += 1
        else:
            n_level = (
                n_level + len(v["level"])
                if v["level"] is not None
                else n_level + 1
            )
    return n_level


def n_condition_channels(config):
    n_level = 0
    for (
        k,
        v,
    ) in config.data_specs.conditioning_vars.items():
        if v is None:
            n_level += 1
        else:
            n_level = (
                n_level + len(v["level"])
                if v["level"] is not None
                else n_level + 1
            )
    n_level = n_level * len(config.data_specs.conditioning_time_step)
    n_level = (
        n_level + len(config.data_specs.constants)
        if config.data_specs.constants is not None
        else n_level
    )
    return n_level


def undo_scaling(
    dataset: xr.Dataset, dataset_min_max: xr.Dataset
) -> xr.Dataset:
    res = xr.Dataset()
    for varname in dataset.var():
        dmin = dataset_min_max[varname + "_min"]
        dmax = dataset_min_max[varname + "_max"]

        res[varname] = dataset[varname] * (dmax - dmin) + dmin

        if varname == "tp":
            res[varname] = inverse_transform_precipitation(
                dataset[
                    [
                        varname,
                    ]
                ]
            )

    return res


def create_xr_output_variables(
    data: torch.tensor,
    zarr_path: str,
    config: DictConfig,
    min_max_file_path: str,
) -> None:
    """Create an xarray dataset with dimensions [ensemble_member, init_time, lead_time, lat, lon] from a data tensor with shape (n_ensemble_members, n_init_times, n_variables, n_lat, n_lon)

    Args:
        data (torch.tensor): Data to be rescaled and read into an xarray dataset.
        dates (str): Path to the zarr file where the dataset is saved. Is required because we need to load the time axis from there.
        config_file_path (str): Path to the used configuration file
        min_max_file_path (str): Path to the netcdf4 file in which training set maxima and minima are stored.
    """  # noqa: E501
    # loading config information:

    spatial_resolution = config.template.data_specs.spatial_resolution
    root_dir = config.paths.dir_WeatherBench
    lead_time = config.template.data_specs.lead_time
    max_conditioning_time_steps = max(abs(np.array(config.template.data_specs.conditioning_time_step)))
    # load time:
    dates = xr.open_zarr(zarr_path).time.rename({"time":"init_time"}).isel({"init_time": slice(max_conditioning_time_steps, -lead_time)})

    # create dataset and set up coordinates:
    ds = xr.Dataset()

    assert os.path.isfile(
        os.path.join(
            root_dir,
            "constants/constants_{}.nc".format(spatial_resolution),
        )
    ), (
        "The file {} is required to extract the coordinates, but doesn't"
        " exist.".format(
            os.path.join(
                root_dir,
                "constants/constants_{}.nc".format(spatial_resolution),
            )
        )
    )
    coords = xr.open_dataset(
        os.path.join(
            root_dir,
            "constants/constants_{}.nc".format(spatial_resolution),
        )
    ).coords
    ds.coords.update(coords)

    if data.ndim == 5:  # (ensemble_member, bs, channels, lat, lon)
        ds = ds.expand_dims({"lead_time": 1}).assign_coords(
            {"lead_time": [lead_time]}
        )
    elif data.ndim == 6:  # (ensemble_member, bs, len_traj, channels, lat, lon)
        ds = ds.expand_dims({"lead_time": 1}).assign_coords(
            {"lead_time": [(i+1)*lead_time for i in range(data.shape[2])]}
        )
    else:
        raise ValueError("Invalid number of dimensions of input data.")     
       
    ds = ds.expand_dims({"ensemble_member": data.shape[0]}).assign_coords(
        {"ensemble_member": np.arange(data.shape[0])}
    )
    ds = ds.expand_dims(init_time=dates)

    # get list of variables:
    assert os.path.isfile(min_max_file_path), (
        "The file {} is required to extract minima and"
        " maxima, but doesn't exist.".format(min_max_file_path)
    )
    ds_min_max = xr.open_dataset(min_max_file_path)
    # get list of variables, hopefully in the same order as the channels:
    var_names = [
        name.replace("_max", "")
        for name in list(ds_min_max.var())
        if "_max" in name
    ]

    if data.ndim == 5:  # (ensemble_member, bs, channels, lat, lon)
        for i in range(data.shape[-3]):
            ds[var_names[i]] = xr.DataArray(
                data[..., i : i + 1, :, :],
                dims=("ensemble_member", "init_time", "lead_time", "lat", "lon"),
                coords={
                    "ensemble_member": ds.ensemble_member,
                    "lat": ds.lat,
                    "lon": ds.lon,
                    "lead_time": ds.lead_time,
                    "init_time": ds.init_time,
                },
            )
    elif data.ndim == 6:  # (ensemble_member, bs, len_traj, channels, lat, lon)
        for i in range(data.shape[-3]):
            ds[var_names[i]] = xr.DataArray(
                data[..., i, :, :],
                dims=("ensemble_member", "init_time", "lead_time", "lat", "lon"),
                coords={
                    "ensemble_member": ds.ensemble_member,
                    "lat": ds.lat,
                    "lon": ds.lon,
                    "lead_time": ds.lead_time,
                    "init_time": ds.init_time,
                },
            )        
    else:
        raise ValueError("Invalid number of dimensions of input data.")    
    return undo_scaling(ds, ds_min_max)
