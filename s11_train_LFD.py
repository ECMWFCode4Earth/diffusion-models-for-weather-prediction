import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)
import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dm_zoo.dff.EMA import EMA
from dm_zoo.diffusion.LFD_lightning import LatentForecastDiffusion

from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.utils import check_devices, create_dir, AreaWeightedMSELoss

@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/training", config_name="config")

def train_LFD(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part
    exp_name = hydra_cfg['runtime']['choices']['experiment']

    print(f"The torch version being used is {torch.__version__}")
    check_devices()

    # load config
    print(f"Loading dataset {config.experiment.data.template}")
    # ds_config_path = os.path.join(conf.base_path, f"{conf.template}.yml")
    # ds_config = load_config(ds_config_path)
    ds_config = OmegaConf.load(f"{config.paths.hydra_config_dir}/{config.experiment.data.template}/.hydra/config.yaml")

    train_ds_path = config.paths.data_dir + f"{config.experiment.data.template}_train.zarr"
    train_ds = Conditional_Dataset_Zarr_Iterable(train_ds_path, ds_config.template, shuffle_chunks=config.experiment.data.train_shuffle_chunks, 
                                                shuffle_in_chunks=config.experiment.data.train_shuffle_in_chunks)

    val_ds_path = config.paths.data_dir + f"{config.experiment.data.template}_val.zarr"
    val_ds = Conditional_Dataset_Zarr_Iterable(val_ds_path, ds_config.template, shuffle_chunks=config.experiment.data.val_shuffle_chunks, shuffle_in_chunks=config.experiment.data.val_shuffle_in_chunks)

    # select loss_fn:
    if config.experiment.model.loss_fn_name == "MSE_Loss":
        loss_fn = torch.nn.functional.mse_loss
    elif config.experiment.model.loss_fn_name == "AreaWeighted_MSE_Loss":
        lat_grid = train_ds.data.targets.lat[:]
        lon_grid =  train_ds.data.targets.lon[:]
        loss_fn = AreaWeightedMSELoss(lat_grid, lon_grid).loss_fn
    else:
        raise NotImplementedError("Invalid loss function.")

    if config.experiment.model.diffusion.sampler_name == "DDPM":  # this is the default case
        sampler = None
    else:
        raise NotImplementedError("This sampler has not been implemented.")  

    model_dir = f"{config.paths.save_model_dir}/{config.experiment.data.template}/{exp_name}/{dir_name}/"
    create_dir(model_dir)

    # set up logger:
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

    # set up diffusion model:

    conditioning_channels = train_ds.array_inputs.shape[1] * len(train_ds.conditioning_timesteps) + train_ds.array_constants.shape[0]
    generated_channels = train_ds.array_targets.shape[1]
    print("generated channels: {} conditioning channels: {}".format(generated_channels, conditioning_channels))

    image_size = train_ds.array_inputs.shape[-2:]
    
    with open_dict(config):
        config.experiment.model.image_size = image_size
        config.experiment.model.generated_channels = generated_channels
        config.experiment.model.conditioning_channels = conditioning_channels
    
    
    model= LatentForecastDiffusion(config.experiment.model,
                                   train_dataset=train_ds,
                                   valid_dataset=val_ds, 
                                   loss_fn = loss_fn,
                                   sampler = sampler)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(
        monitor="val_reconstruction_loss", mode="min", patience=config.experiment.training.patience
    )

    trainer = pl.Trainer(
        max_steps=config.experiment.training.max_steps,
        limit_val_batches=config.experiment.training.limit_val_batches,
        accelerator=config.experiment.training.accelerator,
        devices=config.experiment.training.devices,
        callbacks=[EMA(config.experiment.training.ema_decay), lr_monitor, early_stopping],
        logger=tb_logger,
        gradient_clip_val=0.5 # Gradient clip value for exploding gradient
    )

    trainer.fit(model)
    
if __name__ == '__main__':
    train_LFD()