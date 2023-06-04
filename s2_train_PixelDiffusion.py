import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dm_zoo.dff.EMA import EMA
from dm_zoo.dff.PixelDiffusion import PixelDiffusion
from WD.datasets_old import SingleDataset
import torch
from WD.utils import check_devices
import sys

print(f"The torch version being used is {torch.__version__}")
check_devices()

train_ds = SingleDataset(split="train")

model = PixelDiffusion(
    generated_channels=1,
    train_dataset=train_ds,
    lr=1e-4,
    batch_size=16,
)

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="/data/compoundx/ml_models/diffusion_models/"
)

print(sys.getsizeof(train_ds))

trainer = pl.Trainer(
    max_steps=2e5,
    callbacks=[EMA(0.9999)],
    limit_val_batches=0,
    logger=tb_logger,
    accelerator="cuda",
    devices=-1,
)

trainer.fit(model)
