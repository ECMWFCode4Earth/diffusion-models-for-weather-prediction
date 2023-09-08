import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dm_zoo.dff.EMA import EMA

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from WD.io import load_config
from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.utils import create_dir, generate_uid, check_devices
from dm_zoo.latent.vae.vae_lightning_module import VAE
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import (
    EarlyStopping,
)

@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/training", config_name="config")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    exp_name = hydra_cfg['runtime']['choices']['experiment']
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part

    ds_config = OmegaConf.load(f"{config.paths.dir_HydraConfigs}/data/{config.experiment.data.template}/.hydra/config.yaml")

    print(f"The torch version being used is {torch.__version__}")
    check_devices()

    # load config
    print(f"Loading dataset {config.experiment.data.template}")

    train_ds_path = config.paths.dir_PreprocessedDatasets + f"{config.experiment.data.template}_train.zarr"
    train_ds = Conditional_Dataset_Zarr_Iterable(train_ds_path, ds_config.template, shuffle_chunks=config.experiment.data.train_shuffle_chunks, 
                                                shuffle_in_chunks=config.experiment.data.train_shuffle_in_chunks)

    val_ds_path = config.paths.dir_PreprocessedDatasets + f"{config.experiment.data.template}_val.zarr"
    val_ds = Conditional_Dataset_Zarr_Iterable(val_ds_path, ds_config.template, shuffle_chunks=config.experiment.data.val_shuffle_chunks, shuffle_in_chunks=config.experiment.data.val_shuffle_in_chunks)

    if config.experiment.vae.type == "input":
        n_channels = train_ds.array_inputs.shape[1] * len(train_ds.conditioning_timesteps) + train_ds.array_constants.shape[0]
    else:
        n_channels = train_ds.array_targets.shape[1]
    in_shape = (n_channels, *train_ds.array_targets.shape[:-2])

    model = VAE(inp_shape = in_shape, train_dataset=train_ds, valid_dataset=val_ds, 
                dim=config.experiment.vae.dim, 
                channel_mult = config.experiment.vae.channel_mult, 
                batch_size = config.experiment.vae.batch_size, 
                lr = config.experiment.vae.lr, 
                lr_scheduler_name=config.experiment.vae.lr_scheduler_name, 
                num_workers = config.experiment.vae.num_workers, 
                beta = config.experiment.vae.beta,
                data_type = config.experiment.vae.type)

    model_dir = f"{config.paths.dir_SavedModels}/{config.experiment.data.template}/{exp_name}/{dir_name}/"
    create_dir(model_dir)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=model_dir)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=config.experiment.training.max_steps,
        limit_val_batches=config.experiment.training.limit_val_batches,
        accelerator=config.experiment.training.accelerator,
        devices=config.experiment.training.devices,
        callbacks=[EMA(config.experiment.training.ema_decay), lr_monitor], #, early_stopping],
        logger=tb_logger
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()