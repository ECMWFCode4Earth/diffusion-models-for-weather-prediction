from typing import Dict, List, Tuple, Union
import xarray as xr
import numpy as np

import os
import torch
from torch.utils.data import Dataset

from WD.io import write_config, load_config

from WD.utils import (
    transform_precipitation,
    inverse_transform_precipitation,
    generate_uid,
)
from munch import Munch

from datetime import datetime


def open_datasets(
    root_dir: str, variables: Dict, spatial_resolution: str
) -> xr.Dataset:
    res_datasets = []

    for foldername, var_config in variables.items():
        path = os.path.join(root_dir, foldername, "*_{}.nc".format(spatial_resolution))
        ds = xr.open_mfdataset(path)

        assert len(ds.keys()) == 1
        varname = list(ds.keys())[0]

        if varname == "tp":
            ds = ds.rolling(time=6).sum()  # take 6 hour average
            ds["tp"] = transform_precipitation(ds)

        # extract desired pressure levels:
        if var_config["level"] is not None:
            datasets = create_variables_from_pressure_levels(
                ds=ds, var_config=var_config
            )
            res_datasets.extend(datasets)
        else:
            if "level" in ds.dims:
                assert (
                    ds.level.size == 1
                ), "The given dataset is defined at more than one pressure level, but no pressure levels were selected in the configuration file."
            if "level" in ds.var():
                ds = ds.drop_vars("level")
            res_datasets.append(ds)
    return xr.merge(res_datasets)


def create_variables_from_pressure_levels(
    ds: xr.Dataset, var_config: Munch
) -> List[xr.Dataset]:
    """Given a dataset ds with multiple pressure levels and a var_config,
    that tells us which of these variables we want to use, return a list of datasets,
    each of which contains a single variable, which is the variable of ds
    at a single pressure level, renamed as <varname>_<pressurelevel>.

    Args:
        ds (xr.Dataset): An input dataset, containing a single variable at
        multiple pressure levels.
        var_config (Munch): A configuration object which contains an attribute "levels"
        that specifies a list of pressure levels we want to use.

    Returns:
        List[xr.Dataset]: A list of xarray datasets,
        each of which contains a single variable at a single pressure level.
    """

    ds = ds.sel({"level": var_config["level"]})
    grouped = ds.groupby("level")
    group_indices = grouped.groups
    datasets = []
    for group_name, group_index in group_indices.items():
        group_data = ds.isel(level=group_index)
        renamed_vars = {}
        for var_name, var_data in group_data.data_vars.items():
            new_var_name = f"{var_name}_{group_name}"
            renamed_vars[new_var_name] = var_data
        group_ds = xr.Dataset(renamed_vars).drop_vars("level")
        datasets.append(group_ds)

    return datasets


def open_constant_datasets(
    root_dir: str, spatial_resolution: str, constant_vars: Union[List[str], None]
) -> xr.Dataset:
    """Open the constant fields, i.e. variables that stay constant over the entire time domain.

    Args:
        root_dir (str): Directory WeatherBench data is stored in.
        spatial_resolution (str): The spatial resolution we want to work with.
        constant_vars (Union[List[str], None]): A list containing the variable names of all fields we want to use, e.g. "orography".

    Returns:
        xr.Dataset: A dataset containing all selected constant variables.
    """
    constant_datasets = []
    ds_constants = xr.open_dataset(
        os.path.join(
            root_dir, "constants", "constants_{}.nc".format(spatial_resolution)
        )
    )

    if constant_vars is not None:
        for cv in constant_vars:
            ds = ds_constants[
                [
                    cv,
                ]
            ]
            constant_datasets.append(ds)
        return xr.merge(constant_datasets)
    else:
        return (
            xr.Dataset()
        )  # if no constant vars were selected, return an empty xarray.


def write_conditional_datasets(config_path: str) -> None:
    """Save a preprocessed version of the WeatherBench dataset into a single file.

    Args:
        root_dir (str): Directory in which the WeatherBench Dataset is stored.
        train_limits (Tuple[datetime, datetime]): Start and end date of the training set.
        test_limits (Tuple[datetime, datetime]): Start and end date of the test set.
        conditioning_variables (Dict[str, str]): Variables we want to use for conditioning. Dict containing the filename as keys and the variables as values.
        output_variables (Dict[str, str]): Variables we want the model to forecast. Dict containing the filename as keys and the variables as values.
        conditioning_timesteps (List[int]): List of timesteps that should be used as conditioning information. If None, no conditioning will be applied.
        lead_time (Union[int,None]): The lead time at which we want to produce the prediction. In units of delta_t.
        validation_limits (Union[Tuple[datetime, datetime], None], optional): Start and end date of the training set. Can be None, if no validation set is used.. Defaults to None.
        spatial_resolution (str, optional): The spatial resolution of the dataset we want to load. Defaults to "5.625deg".
        delta_t (int, optional): Interval between consecutive timesteps in hours. Defaults to 6.
        out_dir (Union[None, str], optional): Directory to save the datasets in, if None use the same as the input directory. Defaults to None.
        out_filename (Union[None, str], optional): Name to save the dataset as, if not provided use a default name. Defaults to None.
    """

    print("Load config file.")
    config = load_config(config_path)
    from_train = config.exp_data.train.start
    to_train = config.exp_data.train.end

    from_val = config.exp_data.val.start
    to_val = config.exp_data.val.end

    from_test = config.exp_data.test.start
    to_test = config.exp_data.test.end

    root_dir = config.file_structure.dir_WeatherBench
    out_dir = config.file_structure.dir_model_input
    train_limits = (from_train, to_train)
    validation_limits = (from_val, to_val)
    test_limits = (from_test, to_test)
    conditioning_variables = config.data_specs.conditioning_vars.toDict()
    output_variables = config.data_specs.output_vars.toDict()
    spatial_resolution = config.data_specs.spatial_resolution
    delta_t = config.data_specs.delta_t
    constant_vars = config.data_specs.constants

    if out_dir is None:
        out_dir = root_dir

    if out_dir is None:
        out_dir = root_dir

    config.ds_id = generate_uid()
    out_filename = f"{config.ds_id}"

    print("Open datasets.")
    # load output variables
    output_dataset = open_datasets(
        root_dir=root_dir,
        variables=output_variables,
        spatial_resolution=spatial_resolution,
    )
    # load conditioning variables
    conditioning_dataset = open_datasets(
        root_dir=root_dir,
        variables=conditioning_variables,
        spatial_resolution=spatial_resolution,
    )
    # load constant variables
    constant_dataset = open_constant_datasets(
        root_dir=root_dir,
        spatial_resolution=spatial_resolution,
        constant_vars=constant_vars,
    )

    print("Number of conditioning variables:", len(list(conditioning_dataset.keys())))
    print("Number of constant variables:", len(list(constant_dataset.keys())))
    print("Number of output variables:", len(list(output_dataset.keys())))

    # filter to temporal resolution delta_t
    output_dataset = output_dataset.resample(time="{}H".format(delta_t)).nearest()
    conditioning_dataset = conditioning_dataset.resample(
        time="{}H".format(delta_t)
    ).nearest()
    print("Normalize datasets.")

    (
        conditioning_dataset,
        conditioning_train_min,
        conditioning_train_max,
    ) = normalize_dataset(conditioning_dataset, train_limits=train_limits)

    output_dataset, output_train_min, output_train_max = normalize_dataset(
        output_dataset, train_limits=train_limits
    )

    constant_dataset, constant_train_min, constant_train_max = normalize_dataset(
        constant_dataset, train_limits=None
    )

    # save min and max to separate file:
    xr.merge([output_train_min, output_train_max]).to_netcdf(
        os.path.join(out_dir, "{}_output_min_max.nc".format(out_filename))
    )
    xr.merge([conditioning_train_min, conditioning_train_max]).to_netcdf(
        os.path.join(out_dir, "{}_conditioning_min_max.nc".format(out_filename))
    )
    xr.merge([constant_train_min, constant_train_max]).to_netcdf(
        os.path.join(out_dir, "{}_constant_min_max.nc".format(out_filename))
    )

    print("Slice into train, test and validation set and write to .pt files.")
    test_inputs = conditioning_dataset.sel({"time": slice(*test_limits)})
    assert contains_no_nans(
        test_inputs
    ), "test_inputs data set contains missing values, possibly because of the precipitation computation."
    train_inputs = conditioning_dataset.sel({"time": slice(*train_limits)})
    assert contains_no_nans(
        train_inputs
    ), "train_inputs data set contains missing values, possibly because of the precipitation computation."
    test_targets = output_dataset.sel({"time": slice(*test_limits)})
    assert contains_no_nans(
        test_targets
    ), "test_targets data set contains missing values, possibly because of the precipitation computation."
    train_targets = output_dataset.sel({"time": slice(*train_limits)})
    assert contains_no_nans(
        train_targets
    ), "train_targets data set contains missing values, possibly because of the precipitation computation."
    write_to_pytorch(
        "train",
        train_inputs,
        train_targets,
        constants=constant_dataset,
        out_dir=out_dir,
        out_filename=out_filename,
    )
    write_to_pytorch(
        "test",
        test_inputs,
        test_targets,
        constants=constant_dataset,
        out_dir=out_dir,
        out_filename=out_filename,
    )
    if validation_limits is not None:
        val_inputs = conditioning_dataset.sel({"time": slice(*validation_limits)})
        assert contains_no_nans(
            val_inputs
        ), "val_inputs data set contains missing values, possibly because of the precipitation computation."
        val_targets = output_dataset.sel({"time": slice(*validation_limits)})
        assert contains_no_nans(
            val_targets
        ), "val_targets data set contains missing values, possibly because of the precipitation computation."

        write_to_pytorch(
            "val",
            val_inputs,
            val_targets,
            constants=constant_dataset,
            out_dir=out_dir,
            out_filename=out_filename,
        )
    # store the config file.
    write_config(config)


def normalize_dataset(
    dataset: xr.Dataset, train_limits: Union[Tuple[datetime, datetime], None]
) -> xr.Dataset:
    """Normalize datasets, such that range in training set is [0,1].

    Args:
        dataset (xr.Dataset): Dataset to be normalized (entire time domain).
        train_limits (Union[Tuple[datetime, datetime], None]): Limits of the training set in the time domain.
        If None, don't use any slicing.

    Returns:
        xr.Dataset: Rescaled dataset.
    """
    if train_limits is not None:
        train_max = (
            dataset.sel({"time": slice(*train_limits)}).max(keep_attrs=True).compute()
        )
        train_min = (
            dataset.sel({"time": slice(*train_limits)}).min(keep_attrs=True).compute()
        )
    else:
        train_max = dataset.max(keep_attrs=True).compute()
        train_min = dataset.min(keep_attrs=True).compute()
    dataset = (dataset - train_min) / (train_max - train_min)

    for var_name in train_min.data_vars:
        train_min = train_min.rename({var_name: var_name + "_min"})
    for var_name in train_max.data_vars:
        train_max = train_max.rename({var_name: var_name + "_max"})

    return dataset, train_min, train_max


def write_to_pytorch(
    ds_type: str,
    inputs: xr.Dataset,
    targets: xr.Dataset,
    constants: xr.Dataset,
    out_dir: str,
    out_filename: str,
) -> None:
    if len(constants.var()) > 0:
        torch.save(
            {
                "time": torch.tensor(inputs.time.values.astype(np.int64)),
                "inputs": torch.tensor(
                    xr.Dataset.to_array(inputs).values, dtype=torch.float
                ).transpose(1, 0),
                "targets": torch.tensor(
                    xr.Dataset.to_array(targets).values, dtype=torch.float
                ).transpose(1, 0),
                "constants": torch.tensor(
                    xr.Dataset.to_array(constants).values, dtype=torch.float
                ),
            },
            os.path.join(out_dir, "{}_{}.pt".format(out_filename, ds_type)),
        )
    else:
        torch.save(
            {
                "time": torch.tensor(inputs.time.values.astype(np.int64)),
                "inputs": torch.tensor(
                    xr.Dataset.to_array(inputs).values, dtype=torch.float
                ).transpose(1, 0),
                "targets": torch.tensor(
                    xr.Dataset.to_array(targets).values, dtype=torch.float
                ).transpose(1, 0),
                "constants": torch.tensor([], dtype=torch.float),
            },
            os.path.join(out_dir, "{}_{}.pt".format(out_filename, ds_type)),
        )


def contains_no_nans(ds: xr.Dataset):
    return bool(ds.to_array().notnull().all().any())


def expand_time_dimension(ds: xr.Dataset) -> xr.Dataset:
    """For a given dataset containing variables both with and without time dimension,
    add time dimension to all variables.

    Args:
        ds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset with time dimension for all variables.
    """
    vars_constant = [k for k in ds.keys() if "time" not in ds[k].coords]
    vars_nonconstant = [k for k in ds.keys() if "time" in ds[k].coords]

    assert len(vars_nonconstant) > 0

    ds_nconst = ds[vars_nonconstant]
    ds_const = ds[vars_constant]

    ds_const = ds_const.expand_dims(time=ds.time)

    return xr.merge([ds_const, ds_nconst])


class Conditional_Dataset(Dataset):
    """Dataset when using past steps as conditioning information and predicting into the future."""

    def __init__(self, pt_file_path, config_file_path):
        self.path = pt_file_path
        data = torch.load(self.path)

        # we need to load the lead time and the conditioning time steps
        config = load_config(config_file_path)
        self.lead_time = config.data_specs.lead_time
        self.conditioning_timesteps = torch.tensor(
            config.data_specs.conditioning_time_step, dtype=torch.int
        )
        self.max_abs_c_t = max(abs(self.conditioning_timesteps))

        assert self.lead_time > 0

        self.time = data["time"]
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        self.constants = data["constants"]
        self.indices = torch.arange(
            len(self.inputs) - self.lead_time - self.max_abs_c_t, dtype=torch.int
        )

        if len(data["constants"]) == 0:
            self.constants = torch.empty(
                1, 0, *self.targets.shape[2:]
            ).float()  # TODO remove for newer datasets
        else:
            self.constants = data["constants"].float()  # TODO remove for newer datasets

        self.indices = torch.arange(
            len(self.inputs) - self.lead_time - self.max_abs_c_t, dtype=torch.int
        )

    def __len__(self):
        return len(self.inputs) - self.lead_time - self.max_abs_c_t

    def __getitem__(self, idx):
        # this is not optimal - but the only way I could come up with to also be able to use slices here.
        indices = self.indices[idx]
        input = self.inputs[
            indices.view(-1, 1)
            + self.conditioning_timesteps.view(1, -1)
            + self.max_abs_c_t,
            :,
            :,
            :,
        ]
        input = input.view(
            input.shape[0], input.shape[1] * input.shape[2], *input.shape[3:]
        )
        input = torch.concatenate(
            (input, self.constants.repeat(input.shape[0], 1, 1, 1)), dim=1
        ).squeeze(dim=0)
        target = (
            self.targets[indices + self.max_abs_c_t + self.lead_time].view(
                -1, *self.targets.shape[1:]
            )
        ).squeeze(
            dim=0
        )  # torch.squeeze is necessary because we want a trivial first dimension if it has only one element.

        # return the init_time of the forecast:
        time = self.time[indices + self.max_abs_c_t]

        return input, target, time


# to be used in dataloader for ensemble evaluation:
def custom_collate(batch, num_copies):
    inputs = torch.stack([sample[0] for sample in batch])
    targets = torch.stack([sample[1] for sample in batch])
    dates = torch.stack([sample[2] for sample in batch])
    return [inputs.repeat_interleave(num_copies, dim=0), targets.repeat_interleave(num_copies, dim=0), dates.repeat_interleave(num_copies, dim=0)]