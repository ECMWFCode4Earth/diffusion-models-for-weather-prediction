from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from dm_zoo.diffusion import Diffusion
import os

from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.utils import create_dir
from WD.io import create_xr_output_variables
# from WD.io import load_config, write_config  # noqa F401
import pytorch_lightning as pl

@hydra.main(version_base = None, config_path="/data/compoundx/WeatherDiff/config/training", config_name="config")
def train_diffusion(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part
    exp_name = hydra_cfg['runtime']['choices']['experiment']











if __name__ == '__main__':
    train_diffusion()
