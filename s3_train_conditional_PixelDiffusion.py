import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from dm_zoo.dff.EMA import EMA
from dm_zoo.dff.PixelDiffusion import (
    PixelDiffusionConditional,
)
from WD.datasets import Conditional_Dataset
import torch
from WD.utils import check_devices, create_dir, generate_uid
from WD.io import write_config, load_config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

parser = argparse.ArgumentParser(
    prog="WeatherDiffCondVanilla",
    description="Vanilla Conditional Diffusion Model",
    epilog="Arg parser for vanilla conditional diffusion model",
)

parser.add_argument(
    "-did",
    "--dataset_id",
    type=str,
    default="738F8B",
    help="unique id for the dataset, else defaults to 738F8B",
)

print(f"The torch version being used is {torch.__version__}")
check_devices()


args = parser.parse_args()

ds_id = args.dataset_id

if ds_id == "9A9F63":
    print(f"Loading DEFAULT dataset from {ds_id}.yaml")
else:
    print(f"Loading dataset from {ds_id}.yaml")

ds_config_path = f"/data/compoundx/WeatherDiff/config_file/{ds_id}.yml"
ds_config = load_config(ds_config_path)


train_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_train.pt"
train_ds = Conditional_Dataset(train_ds_path, ds_config_path)

val_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_val.pt"
val_ds = Conditional_Dataset(val_ds_path, ds_config_path)

# pytorch lightening hyperparams

# model = PixelDiffusionConditional(train_dataset=train_ds, **model_hparam)

model = PixelDiffusionConditional(
    train_dataset=train_ds,
    valid_dataset=val_ds,
    generated_channels=ds_config.n_generated_channels,
    condition_channels=ds_config.n_condition_channels,
    lr=1e-4,
    batch_size=64,
    cylindrical_padding=True
)

model_config = model.config()

model_id = generate_uid()
model_dir = f"/data/compoundx/WeatherDiff/saved_model/{ds_id}/{model_id}/"
create_dir(model_dir)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)


pl_hparam = {
    "max_steps": 5e5,
    "ema_decay": 0.9999,
    "limit_val_batches": 1.0,
    "accelerator": "cuda",
    "devices": -1,
}

assert (
    pl_hparam["limit_val_batches"] > 0
)  # So that validation step is carried out

lr_monitor = LearningRateMonitor(logging_interval="step")
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    min_delta=0
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
