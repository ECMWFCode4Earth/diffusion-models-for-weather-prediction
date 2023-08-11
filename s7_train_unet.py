import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from dm_zoo.dff.EMA import EMA
from dm_zoo.dff.UNetRegression import (
    UNetRegression,
)
from WD.datasets import Conditional_Dataset_Zarr_Iterable, Conditional_Dataset
import torch
from WD.utils import check_devices, create_dir, generate_uid, AreaWeightedMSELoss
from WD.io import write_config, load_config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

parser = argparse.ArgumentParser(
    prog="WeatherDiffUNetBaseline",
    description="UNet baseline",
    epilog="Arg parser for UNet baseline model",
)

parser.add_argument(
    "-did",
    "--dataset_id",
    type=str,
    help="unique id for the dataset.",
)

parser.add_argument(
    "-lrs",
    "--lr_schedule",
    type=str,
    help="LR-Schedule to be used. Must be compatible with what is written in the UNetRegression class.",
)

print(f"The torch version being used is {torch.__version__}")
print(f"The torch version being used is {torch.cuda.is_available()}")

check_devices()

args = parser.parse_args()

ds_id = args.dataset_id
lrs = args.lr_schedule

print(f"Loading dataset from {ds_id}.yaml")

ds_config_path = f"/data/compoundx/WeatherDiff/config_file/{ds_id}.yml"
ds_config = load_config(ds_config_path)


# datasets:

train_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_train.zarr"
train_ds = Conditional_Dataset_Zarr_Iterable(train_ds_path, ds_config_path, shuffle_chunks=True, shuffle_in_chunks=True)

val_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_val.zarr"
val_ds = Conditional_Dataset_Zarr_Iterable(val_ds_path, ds_config_path, shuffle_chunks=True, shuffle_in_chunks=True)

"""
train_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_train.pt"
train_ds = Conditional_Dataset(train_ds_path, ds_config_path)

val_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_val.pt"
val_ds = Conditional_Dataset(val_ds_path, ds_config_path)
"""

# loss function: 
# lat_grid = train_ds.data.targets.lat[:]
# lon_grid =  train_ds.data.targets.lon[:]

loss_fn = torch.nn.functional.mse_loss  # AreaWeightedMSELoss(lat_grid, lon_grid).loss_fn  # torch.nn.functional.mse_loss

# possible schedulers: None, "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "CosineAnnealingWarmupRestarts"

model = UNetRegression(
    train_dataset=train_ds,
    valid_dataset=val_ds,
    generated_channels=ds_config.n_generated_channels,
    condition_channels=ds_config.n_condition_channels,
    batch_size=64,
    cylindrical_padding=True,
    lr=1e-4,
    num_workers=4,
    loss_fn=loss_fn,
    lr_scheduler_name=lrs
)

model_config = model.config()

model_id = generate_uid()
model_dir = f"/data/compoundx/WeatherDiff/saved_model/{ds_id}/{model_id}/"
create_dir(model_dir)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

# pytorch lightening hyperparams

pl_hparam = {
    "max_steps": 5e7,
    "ema_decay": 0.9999,
    "limit_val_batches": 10,
    "limit_test_batches": 1.0,
    "accelerator": "cuda",
    "devices": -1,
}

assert (
    pl_hparam["limit_val_batches"] > 0
)  # So that validation step is carried out

lr_monitor = LearningRateMonitor(logging_interval="step")
early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", patience=50, min_delta=0
)
pl_args = {}
for key, val in pl_hparam.items():
    if key == "ema_decay":
        pl_args["callbacks"] = [EMA(val), lr_monitor, early_stopping]
    else:
        pl_args[key] = val

trainer = pl.Trainer(
    logger=tb_logger,
    **pl_args,
)

model_config["pl_hparam"] = pl_hparam
model_config["ds_id"] = ds_id
model_config["model_id"] = model_id
model_config["file_structure"] = {
    "dir_model_input": ds_config.file_structure.dir_model_input,
    "dir_saved_model": model_dir,
    "dir_config_file": ds_config_path,
}

write_config(model_config)
trainer.fit(model)