import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from dm_zoo.dff.EMA import EMA
from dm_zoo.dff.PixelDiffusion import (
    PixelDiffusionConditional,
)
from dm_zoo.dff.DenoisingDiffusionProcess.DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
)

from WD.datasets import Conditional_Dataset_Zarr_Iterable, Conditional_Dataset
import torch

import os

from argparse import ArgumentParser

from WD.utils import check_devices, create_dir, generate_uid, AreaWeightedMSELoss
from WD.io import write_config, load_config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

parser = ArgumentParser(
    prog="WeatherDiffCondVanilla",
    description="Vanilla Conditional Diffusion Model",
    epilog="Arg parser for vanilla conditional diffusion model",
)


print(f"The torch version being used is {torch.__version__}")
check_devices()


data_parser = parser.add_argument_group("DataSetup")
data_parser.add_argument("--dataset_id", type=str, required=True)
data_parser.add_argument("--ds_config_base_path", type=str, default="/data/compoundx/WeatherDiff/config_file/")
data_parser.add_argument("--train_shuffle_chunks", type=bool, default=True)
data_parser.add_argument("--train_shuffle_in_chunks", type=bool, default=True)
data_parser.add_argument("--val_shuffle_chunks", type=bool, default=True)
data_parser.add_argument("--val_shuffle_in_chunks", type=bool, default=True)

highlevel_parser = parser.add_argument_group("HighLevelModelDetails")
highlevel_parser.add_argument("--loss_fn_name", type=str, default="MSE_Loss", choices=["MSE_Loss", "AreaWeighted_MSE_Loss"]) 
highlevel_parser.add_argument("--sampler_name", type=str, default="DDPM", choices=["DDPM",]) 

training_parser = parser.add_argument_group("TrainingDetails")
training_parser.add_argument("--max_steps", type=int, default=5e6)
training_parser.add_argument("--ema_decay", type=float, default=0.9999) 
training_parser.add_argument("--limit_val_batches", type=int, default=10) 
training_parser.add_argument("--accelerator", type=str, default="cuda")
training_parser.add_argument("--devices", type=int, default=-1)
training_parser.add_argument("--patience", type=int, default=50)

PixelDiffusionConditional.add_model_specific_args(parser)
DenoisingDiffusionConditionalProcess.add_model_specific_args(parser)

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

# load config

print(f"Loading dataset from {args.dataset_id}.yml")
ds_config_path = os.path.join(args.ds_config_base_path, f"{args.dataset_id}.yml")
ds_config = load_config(ds_config_path)


# set up datasets:

train_ds_path = ds_config.file_structure.dir_model_input + f"{args.dataset_id}_train.zarr"
train_ds = Conditional_Dataset_Zarr_Iterable(train_ds_path, ds_config, shuffle_chunks=args.train_shuffle_chunks, 
                                             shuffle_in_chunks=args.train_shuffle_in_chunks)

val_ds_path = ds_config.file_structure.dir_model_input + f"{args.dataset_id}_val.zarr"
val_ds = Conditional_Dataset_Zarr_Iterable(val_ds_path, ds_config, shuffle_chunks=args.val_shuffle_chunks, shuffle_in_chunks=args.val_shuffle_in_chunks)

# select loss_fn:

if args.loss_fn_name == "MSE_Loss":
    loss_fn = torch.nn.functional.mse_loss
elif args.loss_fn_name == "AreaWeighted_MSE_Loss":
    lat_grid = train_ds.data.targets.lat[:]
    lon_grid =  train_ds.data.targets.lon[:]
    AreaWeightedMSELoss(lat_grid, lon_grid).loss_fn
else:
    raise NotImplementedError("Invalid loss function.")

if args.sampler_name == "DDPM":  # this is the default case
    sampler = None
else:
    raise NotImplementedError("This sampler has not been implemented.")    

# create unique model id and create directory to save model in:

model_id = generate_uid()
model_dir = f"/data/compoundx/WeatherDiff/saved_model/{args.dataset_id}/{model_id}/"
create_dir(model_dir)

# set up logger:

tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

# set up diffusion model:


print("generated channels: {} conditioning channels: {}".format(ds_config.n_generated_channels, ds_config.n_condition_channels))
model = PixelDiffusionConditional(
    args,
    generated_channels=ds_config.n_generated_channels,
    conditioning_channels=ds_config.n_condition_channels,
    loss_fn=loss_fn,
    sampler=sampler,
    train_dataset=train_ds,
    valid_dataset=val_ds
)

# model_config = model.config()

# pytorch lightening hyperparams

"""
pl_hparam = {
    "max_steps": 5e7,
    "ema_decay": 0.9999,
    "limit_val_batches": 10,
    "accelerator": "cuda",
    "devices": -1,
}
"""

lr_monitor = LearningRateMonitor(logging_interval="step")

early_stopping = EarlyStopping(
    monitor="val_loss_new", mode="min", patience=args.patience
)

trainer = pl.Trainer(
    max_steps=args.max_steps,
    limit_val_batches=args.limit_val_batches,
    accelerator=args.accelerator,
    devices=args.devices,
    callbacks=[EMA(args.ema_decay), lr_monitor, early_stopping],
    logger=tb_logger
)

"""
model_config["pl_hparam"] = pl_hparam
model_config["ds_id"] = ds_id
model_config["model_id"] = model_id
model_config["file_structure"] = {
    "dir_model_input": ds_config.file_structure.dir_model_input,
    "dir_saved_model": model_dir,
    "dir_config_file": ds_config_path,
}

write_config(model_config)
"""

trainer.fit(model)
