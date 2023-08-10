import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from dm_zoo.dff.PixelDiffusion import (
    PixelDiffusionConditional,
)
from WD.datasets import Conditional_Dataset_Zarr_Iterable, custom_collate
from WD.utils import create_dir
from WD.io import create_xr_output_variables
# from WD.io import load_config, write_config  # noqa F401
import pytorch_lightning as pl



@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/training", config_name="config")
def main(config: DictConfig) -> None:

    ds_id = config.data.template
    model_name = config.model_name  # we have to pass this to the bash file every time! (should contain a string).
    nens = config.n_ensemble_members  # we have to pass this to the bash file every time!

    ds_config = OmegaConf.load(f"/data/compoundx/WeatherDiff/hydra_configs/{config.data.template}/.hydra/config.yaml")
    ml_config = OmegaConf.load(f"/data/compoundx/WeatherDiff/hydra_configs/train/{config.model_name}/.hydra/config.yaml")

    model_output_dir = config.inference_dir

    model_load_dir = Path(f"/data/compoundx/saved_model/{config.model_name}/lightning_logs/version_0/checkpoints/")

    test_ds_path = ds_config.template.file_structure.dir_model_input + f"{config.data.template}_test.zarr"
    ds = Conditional_Dataset_Zarr_Iterable(test_ds_path, ds_config.template, shuffle_chunks=config.shuffle_chunks, 
                                                shuffle_in_chunks=config.shuffle_in_chunks)
    
    model_ckpt = [x for x in model_load_dir.iterdir()][0]

    conditioning_channels = ds.array_inputs.shape[1] * len(ds.conditioning_timesteps) + ds.array_constants.shape[0]
    generated_channels = ds.array_targets.shape[1]

    restored_model = PixelDiffusionConditional.load_from_checkpoint(
        model_ckpt,
        config=ml_config,
        generated_channels=conditioning_channels,
        condition_channels=generated_channels,
        loss_fn=config.loss_fn,
        sampler=config.sampler,
    )

    dl = DataLoader(ds, batch_size=config.batchsize)
    trainer = pl.Trainer()

    out = []
    for i in range(nens):
        out.extend(trainer.predict(restored_model, dl))

    out = torch.cat(out, dim=0)
    out = out.view(nens, -1, *out.shape[1:])

    model_output_dir = Path(model_output_dir / config.data.template / model_name)
    create_dir(model_output_dir)

    # need the view to create axis for
    # different ensemble members (although only 1 here).

    targets = torch.tensor(ds.data.targets.data[ds.start+ds.lead_time:ds.stop+ds.lead_time], dtype=torch.float).unsqueeze(dim=0)

    gen_xr = create_xr_output_variables(
        out,
        zarr_path="/data/compoundx/WeatherDiff/model_input/{}_test.zarr/targets".format(config.data.template),
        config_file_path=f"/data/compoundx/WeatherDiff/hydra_configs/{config.data.template}/.hydra/config.yaml",
        min_max_file_path="/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(config.data.template)
    )

    target_xr = create_xr_output_variables(
        targets,
        zarr_path="/data/compoundx/WeatherDiff/model_input/{}_test.zarr/targets".format(config.data.template),
        config_file_path=f"/data/compoundx/WeatherDiff/hydra_configs/{config.data.template}/.hydra/config.yaml",
        min_max_file_path="/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc".format(config.data.template)
    )

    gen_dir = Path(model_output_dir / "gen.nc")
    gen_xr.to_netcdf(gen_dir)
    print(f"Generated data written at: {gen_dir}")

    target_dir = Path(model_output_dir / "target.nc")
    target_xr.to_netcdf(target_dir)
    print(f"Target data written at: {target_dir}")

if __name__ == '__main__':
    main()