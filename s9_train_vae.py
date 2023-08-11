import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dm_zoo.dff.EMA import EMA
from WD.io import load_config
from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.utils import create_dir, generate_uid
from latent.vae.train import VAE
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)



### Required args

ds_id = "291F23" # To load the dataset

#### VAE hyperparameter
inp_shape = (11, 32, 64) # 11 n_generated channel 32x64 from spatial resolution 
dim = None # if in_channel needs to be changed (from 11 to 32, 64, etc)
beta = 1.0 #betaVAE
batch_size = 128 # batch_size
lr = 1e-3 # lr
lr_scheduler_name =  "ReduceLROnPlateau" # 
num_workers = 1
channel_mult = [1,2]

## pytorch hyperparameters

pl_hparam = {
    "max_steps": 5e7,
    "ema_decay": 0.9999,
    "limit_val_batches": 10,
    "limit_test_batches": 1.0,
    "accelerator": "cpu",
    "devices": 1,
} 

ds_config_path = f"/data/compoundx/WeatherDiff/config_file/{ds_id}.yml"
ds_config = load_config(ds_config_path)

print(f"Loading dataset from {ds_id}.yaml")

# datasets:
train_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_train.zarr"
train_ds = Conditional_Dataset_Zarr_Iterable(train_ds_path, ds_config_path, shuffle_chunks=True, shuffle_in_chunks=True)

val_ds_path = ds_config.file_structure.dir_model_input + f"{ds_id}_val.zarr"
val_ds = Conditional_Dataset_Zarr_Iterable(val_ds_path, ds_config_path, shuffle_chunks=True, shuffle_in_chunks=True)

model = VAE(inp_shape = inp_shape, train_dataset=train_ds, valid_dataset=val_ds, dim=dim, channel_mult = channel_mult, 
        batch_size = batch_size,
        lr = lr,
        lr_scheduler_name=lr_scheduler_name,
        num_workers = num_workers,
        beta = beta)

model_id = generate_uid()
model_dir = f"/data/compoundx/WeatherDiff/saved_model/vae/{ds_id}/{model_id}/"
create_dir(model_dir)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)



assert (
    pl_hparam["limit_val_batches"] > 0
)  # So that validation step is carried out

lr_monitor = LearningRateMonitor(logging_interval="step")
early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", patience=10, min_delta=0
)
pl_args = {}
for key, val in pl_hparam.items():
    if key == "ema_decay":
        pl_args["callbacks"] = [EMA(val), lr_monitor]  # , early_stopping]
    else:
        pl_args[key] = val

trainer = pl.Trainer(
    logger=tb_logger,
    **pl_args,
)

trainer.fit(model)