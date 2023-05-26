from dm_zoo.dff.PixelDiffusion import PixelDiffusionConditional
from pathlib import Path
from WD.datasets import Conditional_Dataset
from WD.utils import create_dir
from WD.io import load_config, create_xr_output_variables

import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch

ds_id = "2C730B"
run_id = "F7BBAE"

model_config_path = "/data/compoundx/WeatherDiff/config_file/{}_{}.yml".format(
    ds_id, run_id
)
model_output_dir = Path("/data/compoundx/WeatherDiff/model_output/")

model_config = load_config(model_config_path)

model_load_dir = (
    Path(model_config.file_structure.dir_saved_model)
    / "lightning_logs/version_0/checkpoints/"
)

ds = Conditional_Dataset(
    "/data/compoundx/WeatherDiff/model_input/{}_test.pt".format(ds_id),
    "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
)

model_ckpt = [x for x in model_load_dir.iterdir()][0]

restored_model = PixelDiffusionConditional.load_from_checkpoint(
    model_ckpt,
    generated_channels=model_config.model_hparam["generated_channels"],
    condition_channels=model_config.model_hparam["condition_channels"],
).to(
    "cuda"
)  # to cuda makes it so that the GPU gets used for constructing the images!#


B = 256

dl = DataLoader(ds, batch_size=B, shuffle=False)

trainer = pl.Trainer()
out = trainer.predict(restored_model, dl)


out = torch.cat(out, dim=0)
print(type(out), out.get_device())

model_output_dir = model_output_dir / model_config.ds_id
create_dir(model_output_dir)

gen = out
targets = ds[:][1]
dates = ds[:][2]

gen_xr = create_xr_output_variables(
    gen,
    dates=dates,
    config_file_path="/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
    min_max_file_path="/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(
        ds_id
    ),
)
target_xr = create_xr_output_variables(
    targets,
    dates=dates,
    config_file_path="/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
    min_max_file_path="/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(
        ds_id
    ),
)

gen_dir = model_output_dir / f"{model_config.model_id}_gen.nc"
gen_xr.to_netcdf(gen_dir)
print(f"Generated data written at: {gen_dir}")

target_dir = model_output_dir / f"{model_config.model_id}_target.nc"
target_xr.to_netcdf(target_dir)
print(f"Target data written at: {target_dir}")

model_config.file_structure.dir_model_output = str(model_output_dir)

# write_config(model_config) - I think modifying the config here is not possible because it is write locked? maybe?
