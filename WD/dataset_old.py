import xarray as xr
from datetime import datetime
import os
from typing import Union, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from WD.utils import (
    transform_precipitation,
    inverse_transform_precipitation,
    generate_uid,
)

from WD.io import write_config, load_config
import yaml  # Write Config


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

    # root_dir: str,
    # train_limits: Tuple[datetime, datetime],
    # test_limits: Tuple[datetime, datetime],
    # conditioning_variables: List[int],
    # output_variables: List[int],
    # conditioning_timesteps: List[int],
    # lead_time: Union[int, None],
    # validation_limits: Union[Tuple[datetime, datetime], None] = None,
    # spatial_resolution: str = "5.625deg",
    # delta_t: int = 6,
    # out_dir: Union[None, str] = None,
    # out_filename: Union[None, str] = None,

    # Read the config file

    config = load_config(config_path)

    from_train = config.exp_data.train.time_start
    to_train = config.exp_data.train.time_end

    from_val = config.exp_data.val.time_start
    to_val = config.exp_data.val.time_end

    from_test = config.exp_data.test.time_start
    to_test = config.exp_data.test.time_end

    root_dir = config.file_structure.dir_WeatherBench
    out_dir = config.file_structure.dir_pytorch_data
    train_limits = (from_train, to_train)
    validation_limits = (from_val, to_val)
    test_limits = (from_test, to_test)
    conditioning_variables = config.data_specs.conditioning_vars
    output_variables = config.data_specs.output_vars
    conditioning_timesteps = config.data_specs.conditioning_time_step
    lead_time = config.data_specs.lead_t
    spatial_resolution = config.data_specs.spatial_resolution
    delta_t = config.data_specs.delta_t
    constant_vars = config.data_specs.constant_vars

    config.ds_id = generate_uid()
    out_filename = f"{config.ds_id}"

    # load all files:
    output_datasets = []
    conditioning_datasets = []

    print("Open files.")
    # output files:
    for foldername, var_config in output_variables.items():
        path = os.path.join(root_dir, foldername, "*_{}.nc".format(spatial_resolution))
        ds = xr.open_mfdataset(path)

        assert len(ds.keys()) == 1
        varname = list(ds.keys())[0]

        if varname == "tp":
            ds = ds.rolling(time=6).sum()  # take 6 hour average
            ds["tp"] = inverse_transform_precipitation(ds)

        # extract desired pressure levels:
        if var_config.levels is not None:
            ds = ds.sel({"level": var_config.levels})

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
            output_datasets.extend(datasets)
        else:
            if "level" in ds.var():
                ds = ds.drop_vars("level")
            output_datasets.append(ds)
    output_dataset = xr.merge(output_datasets)

    # conditioning files:
    for foldername, var_config in conditioning_variables.items():
        path = os.path.join(root_dir, foldername, "*_{}.nc".format(spatial_resolution))
        print(foldername, path)
        ds = xr.open_mfdataset(path)

        assert len(ds.keys()) == 1
        varname = list(ds.keys())[0]

        if varname == "tp":
            ds = ds.rolling(time=6).sum()  # take 6 hour average
            ds["tp"] = inverse_transform_precipitation(ds)

        # extract desired pressure levels:
        if var_config.levels is not None:
            ds = ds.sel({"level": var_config.levels})
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
            conditioning_datasets.extend(datasets)
        else:
            conditioning_datasets.append(ds)

    # append constant fields:
    ds_constants = xr.open_dataset(
        os.path.join(
            root_dir, "constants", "constants_{}.nc".format(spatial_resolution)
        )
    )  # "/data/compoundx/WeatherBench/constants/constants_5.625deg.nc"

    if constant_vars is not None:
        for cv in constant_vars:
            ds = ds_constants[
                [
                    cv,
                ]
            ]
            conditioning_datasets.append(ds)
        conditioning_dataset = xr.merge(conditioning_datasets)

    print("Number of conditioning variables:", len(list(conditioning_dataset.keys())))

    # pre-processing:

    # filter to temporal resolution delta_t
    output_dataset = output_dataset.resample(time="{}H".format(delta_t)).nearest()
    conditioning_dataset = conditioning_dataset.resample(
        time="{}H".format(delta_t)
    ).nearest()

    # calculate training set maxima and minima - will need these to rescale the data to [0,1] range.
    print("Compute train set minima and maxima.")

    # use these to rescale the datasets.
    print("Rescale datasets")
    conditioning_dataset = rescale_dataset(conditioning_dataset, train_limits)
    output_dataset = rescale_dataset(output_dataset, train_limits)

    print("Split into train, test, validation sets.")

    # create datasets for train, test and validation
    train_targets, _ = prepare_datasets(
        output_dataset.sel({"time": slice(*train_limits)}),
        lead_time=lead_time,
        conditioning_timesteps=conditioning_timesteps,
    )
    _, train_inputs = prepare_datasets(
        conditioning_dataset.sel({"time": slice(*train_limits)}),
        lead_time=lead_time,
        conditioning_timesteps=conditioning_timesteps,
    )

    test_targets, _ = prepare_datasets(
        output_dataset.sel({"time": slice(*test_limits)}),
        lead_time=lead_time,
        conditioning_timesteps=conditioning_timesteps,
    )
    _, test_inputs = prepare_datasets(
        conditioning_dataset.sel({"time": slice(*test_limits)}),
        lead_time=lead_time,
        conditioning_timesteps=conditioning_timesteps,
    )

    assert bool(
        train_targets.to_array().notnull().all().any()
    ), "train_targets data set contains missing values, possibly because of the precipitation computation."
    assert bool(
        train_inputs.to_array().notnull().all().any()
    ), "train_inputs data set contains missing values, possibly because of the precipitation computation."
    assert bool(
        test_targets.to_array().notnull().all().any()
    ), "test_targets data set contains missing values, possibly because of the precipitation computation."
    assert bool(
        test_inputs.to_array().notnull().all().any()
    ), "test_inputs data set contains missing values, possibly because of the precipitation computation."

    if validation_limits is not None:
        validation_targets, _ = prepare_datasets(
            output_dataset.sel({"time": slice(*validation_limits)}),
            lead_time=lead_time,
            conditioning_timesteps=conditioning_timesteps,
        )
        _, validation_inputs = prepare_datasets(
            conditioning_dataset.sel({"time": slice(*validation_limits)}),
            lead_time=lead_time,
            conditioning_timesteps=conditioning_timesteps,
        )
        assert bool(
            validation_targets.to_array().notnull().all().any()
        ), "validation_targets data set contains missing values, possibly because of the precipitation computation."
        assert bool(
            validation_inputs.to_array().notnull().all().any()
        ), "validation_inputs data set contains missing values, possibly because of the precipitation computation."

    # write the files:
    if out_filename is None:
        out_filename = "ds"

    print("write output")
    torch.save(
        {
            "inputs": torch.tensor(xr.Dataset.to_array(train_inputs).values).transpose(
                1, 0
            ),
            "targets": torch.tensor(
                xr.Dataset.to_array(train_targets).values
            ).transpose(1, 0),
        },
        os.path.join(out_dir, "{}_train.pt".format(out_filename)),
    )
    torch.save(
        {
            "inputs": torch.tensor(xr.Dataset.to_array(test_inputs).values).transpose(
                1, 0
            ),
            "targets": torch.tensor(xr.Dataset.to_array(test_targets).values).transpose(
                1, 0
            ),
        },
        os.path.join(out_dir, "{}_test.pt".format(out_filename)),
    )
    # if we want a validation set, create one:
    if validation_limits is not None:
        torch.save(
            {
                "inputs": torch.tensor(
                    xr.Dataset.to_array(validation_inputs).values
                ).transpose(1, 0),
                "targets": torch.tensor(
                    xr.Dataset.to_array(validation_targets).values
                ).transpose(1, 0),
            },
            os.path.join(out_dir, "{}_val.pt".format(out_filename)),
        )

    write_config(config)


def prepare_datasets(
    ds: xr.DataArray, lead_time: int, conditioning_timesteps: List[int]
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Given a dataset, a lead time and conditioning timesteps, which we want to use for conditioning the prediction,
    return the one dataset that contains all valid target data and
    one dataset that contains the combined conditioning information for the target data.

    Args:
        ds (xr.DataArray): A dataset we want to work with - ideally already restricted to train / test / validation set.
        lead_time (int): Lead time at which we want to make predictions, in units of delta_t.
        conditioning_timesteps (List[int]): Timesteps we want to use in the conditioning, in units of delta_t, e.g. 0 for current time step.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: Target and Conditioning datasets.
    """

    ds_target = ds.isel(
        {"time": slice(-(min(conditioning_timesteps)) + lead_time, None)}
    )
    ds_target["time"] = ds.isel(
        {"time": slice(-(min(conditioning_timesteps)), -lead_time)}
    )["time"]

    dsts_condition = []

    conditioning_vars_constant = [k for k in ds.keys() if "time" not in ds[k].coords]
    conditioning_vars_nonconstant = [k for k in ds.keys() if "time" in ds[k].coords]

    ds_conditional_nconst = ds[conditioning_vars_nonconstant]
    ds_conditional_const = ds[conditioning_vars_constant]

    for t_c in conditioning_timesteps:
        ds_c = ds_conditional_nconst.isel(
            {"time": slice(t_c - (min(conditioning_timesteps)), -lead_time + t_c)}
        )
        ds_c["time"] = ds_conditional_nconst.isel(
            {"time": slice(-(min(conditioning_timesteps)), -lead_time)}
        )["time"]
        keys = ds_c.keys()
        values = ["{}_{}".format(k, t_c) for k in keys]
        renaming_dict = dict(zip(keys, values))
        ds_c = xr.Dataset.rename_vars(ds_c, name_dict=renaming_dict)
        dsts_condition.append(ds_c)

    # add time dimension to constant fields:
    ds_conditional_const = ds_conditional_const.expand_dims(
        time=ds.isel({"time": slice(-(min(conditioning_timesteps)), -lead_time)})[
            "time"
        ]
    )

    ds_condition = xr.merge(dsts_condition)
    ds_condition = xr.merge([ds_condition, ds_conditional_const])

    return ds_target, ds_condition


def write_datasets(
    root_dir: str,
    train_limits: Tuple[datetime, datetime],
    test_limits: Tuple[datetime, datetime],
    output_variables: Dict[str, str],
    validation_limits: Union[Tuple[datetime, datetime], None] = None,
    spatial_resolution: str = "5.625deg",
    delta_t: int = 6,
    out_dir: Union[None, str] = None,
    out_filename: Union[None, str] = None,
) -> None:
    """Save a preprocessed version of the WeatherBench dataset into a single file.

    Args:
        root_dir (str): Directory in which the WeatherBench Dataset is stored.
        train_limits (Tuple[datetime, datetime]): Start and end date of the training set.
        test_limits (Tuple[datetime, datetime]): Start and end date of the test set.
        output_variables (Dict[str, str]): Variables we want the model to forecast. Dict containing the filename as keys and the variables as values.
        validation_limits (Union[Tuple[datetime, datetime], None], optional): Start and end date of the training set. Can be None, if no validation set is used.. Defaults to None.
        spatial_resolution (str, optional): The spatial resolution of the dataset we want to load. Defaults to "5.625deg".
        delta_t (int, optional): Interval between consecutive timesteps in hours. Defaults to 6.
        out_dir (Union[None, str], optional): Directory to save the datasets in, if None use the same as the input directory. Defaults to None.
        out_filename (Union[None, str], optional): Name to save the dataset as, if not provided use a default name. Defaults to None.
    """

    # load all files:
    output_datasets = []

    # output files:
    for foldername, varname in output_variables.items():
        path = os.path.join(root_dir, foldername, "*_{}.nc".format(spatial_resolution))
        ds = xr.open_mfdataset(path)
        if varname == "tp":
            ds = ds.rolling(time=6).sum()  # take 6 hour average
        output_datasets.append(ds)

    output_dataset = xr.merge(output_datasets)

    # pre-processing:

    # filter to temporal resolution delta_t
    output_dataset = output_dataset.resample(time="{}H".format(delta_t)).nearest()

    # calculate training set maxima and minima - will need these to rescale the data to [0,1] range.
    print("Compute train set minima and maxima.")
    train_output_set_max = (
        output_dataset.sel({"time": slice(*train_limits)}).max().compute()
    )
    train_output_set_min = (
        output_dataset.sel({"time": slice(*train_limits)}).min().compute()
    )

    # use these to rescale the datasets.
    print("rescale datasets")
    output_dataset = (output_dataset - train_output_set_min) / (
        train_output_set_max - train_output_set_min
    )

    # create datasets for train, test and validation
    train_targets = output_dataset.sel({"time": slice(*train_limits)})
    assert bool(
        train_targets.to_array().notnull().all().any()
    ), "Training data set contains missing values, possibly because of the precipitation computation."  # assert that there are no missing values in the training set:

    test_targets = output_dataset.sel({"time": slice(*test_limits)})
    assert bool(
        test_targets.to_array().notnull().all().any()
    ), "Training data set contains missing values, possibly because of the precipitation computation."  # assert that there are no missing values in the test set:

    if validation_limits is not None:
        validation_targets = output_dataset.sel({"time": slice(*validation_limits)})
        assert bool(
            validation_targets.to_array().notnull().all().any()
        ), "Validation data set contains missing values, possibly because of the precipitation computation."  # assert that there are no missing values in the val set.

    # write the files:
    if out_filename is None:
        out_filename = "ds"

    print("write output")
    torch.save(
        torch.tensor(xr.Dataset.to_array(train_targets).values).transpose(1, 0),
        os.path.join(out_dir, "{}_train.pt".format(out_filename)),
    )
    torch.save(
        torch.tensor(xr.Dataset.to_array(test_targets).values).transpose(1, 0),
        os.path.join(out_dir, "{}_test.pt".format(out_filename)),
    )
    # if we want a validation set, create one:
    if validation_limits is not None:
        torch.save(
            torch.tensor(xr.Dataset.to_array(validation_targets).values).transpose(
                1, 0
            ),
            os.path.join(out_dir, "{}_val.pt".format(out_filename)),
        )


class Conditional_Dataset(Dataset):
    """Dataset when using past steps as conditioning information and predicting into the future."""

    def __init__(self, path):
        self.path = path
        data = torch.load(self.path)

        self.inputs = data["inputs"]
        self.targets = data["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx]
        target = self.targets[idx]

        return input, target


class Unconditional_Dataset(Dataset):
    """Dataset for unconditional image generation."""

    def __init__(self, path):
        self.path = path
        self.data = torch.load(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        return sample


def rescale_dataset(dataset, limits):
    print("Compute minima and maxima.")
    set_max = dataset.sel({"time": slice(*limits)}).max().compute()
    set_min = dataset.sel({"time": slice(*limits)}).min().compute()

    dataset = (dataset - set_min) / (set_max - set_min)

    return dataset
