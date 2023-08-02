import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dm_zoo.dff.UNetRegression import (
    UNetRegression,
)
from WD.datasets import Conditional_Dataset_Zarr_Iterable, custom_collate
from WD.utils import create_dir
from WD.io import load_config, create_xr_output_variables
from WD.io import write_config  # noqa F401
import pytorch_lightning as pl

parser = argparse.ArgumentParser(
    prog="Evalulate Model",
    description="Evaluate Model based on dataset id and model id",
    epilog="Arg parser for vanilla conditional diffusion model",
)

parser.add_argument(
    "-did",
    "--dataset_id",
    type=str,
    help="path under which the selected config file is stored.",
)


parser.add_argument(
    "-mid",
    "--model_id",
    type=str,
    help="path under which the selected config file is stored.",
)

args = parser.parse_args()

ds_id = args.dataset_id
run_id = args.model_id

B = 128

model_config_path = "/data/compoundx/WeatherDiff/config_file/{}_{}.yml".format(
    ds_id, run_id
)
model_output_dir = Path("/data/compoundx/WeatherDiff/model_output/")

print(model_config_path)
model_config = load_config(model_config_path)

model_load_dir = (
    Path(model_config.file_structure.dir_saved_model)
    / "lightning_logs/version_0/checkpoints/"
)

ds = Conditional_Dataset_Zarr_Iterable(
    "/data/compoundx/WeatherDiff/model_input/{}_test.zarr".format(ds_id),
    "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
    shuffle_chunks=False,
    shuffle_in_chunks=False
)

model_ckpt = [x for x in model_load_dir.iterdir()][0]

restored_model = UNetRegression.load_from_checkpoint(
    model_ckpt,
    generated_channels=model_config.model_hparam["generated_channels"],
    condition_channels=model_config.model_hparam["condition_channels"],
    cylindrical_padding=True
)



"""
dl = DataLoader(
    ds,
    batch_size=B,
    shuffle=False,
    collate_fn=lambda x: custom_collate(x, num_copies=num_copies),
)
"""

# todo: rewrite data loader

dl = DataLoader(
    ds,
    batch_size=B # important to keep this at 1! otherwise can we guarantee that the data will arrive in the right order?
)


trainer = pl.Trainer()

nens = 1

out = []
for i in range(nens):
    out.extend(trainer.predict(restored_model, dl))

out = torch.cat(out, dim=0)
out = out.view(nens, -1, *out.shape[1:])

model_output_dir = model_output_dir / model_config.ds_id
create_dir(model_output_dir)

# need the view to create axis for
# different ensemble members (although only 1 here).

targets = torch.tensor(ds.data.targets.data[ds.start+ds.lead_time:ds.stop+ds.lead_time], dtype=torch.float).unsqueeze(dim=0)

gen_xr = create_xr_output_variables(
    out,
    zarr_path="/data/compoundx/WeatherDiff/model_input/{}_test.zarr/targets".format(ds_id),
    config_file_path="/data/compoundx/WeatherDiff/config_file/{}.yml".format(
        ds_id
    ),
    min_max_file_path=(
        "/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(
            ds_id
        )
    ),
)

target_xr = create_xr_output_variables(
    targets,
    zarr_path="/data/compoundx/WeatherDiff/model_input/{}_test.zarr/targets".format(ds_id),
    config_file_path="/data/compoundx/WeatherDiff/config_file/{}.yml".format(
        ds_id
    ),
    min_max_file_path=(
        "/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(
            ds_id
        )
    ),
)

gen_dir = model_output_dir / f"{model_config.model_id}_gen.nc"
gen_xr.to_netcdf(gen_dir)
print(f"Generated data written at: {gen_dir}")

target_dir = model_output_dir / f"{model_config.model_id}_target.nc"
target_xr.to_netcdf(target_dir)
print(f"Target data written at: {target_dir}")

model_config.file_structure.dir_model_output = str(model_output_dir)

# print(model_config)

# write_config(model_config) #! Check this
# Write config is possible deletes and rewrites
