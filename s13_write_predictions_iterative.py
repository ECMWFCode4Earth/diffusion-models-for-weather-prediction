import argparse
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from dm_zoo.dff.PixelDiffusion import (
    PixelDiffusionConditional,
)
from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.utils import create_dir
from WD.io import create_xr_output_variables
# from WD.io import load_config, write_config  # noqa F401
import pytorch_lightning as pl



@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/inference", config_name="config")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part

    model_name = config.model_name  # we have to pass this to the bash file every time! (should contain a string of the date the run was started).
    nens = config.n_ensemble_members  # we have to pass this to the bash file every time!

    ds_config = OmegaConf.load(f"{config.paths.hydra_config_dir}/{config.data.template}/.hydra/config.yaml")
    ml_config = OmegaConf.load(f"{config.paths.hydra_config_dir}/training/{config.data.template}/{config.experiment}/{config.model_name}/.hydra/config.yaml")

    model_output_dir = config.paths.inference_dir

    model_load_dir = Path(f"{config.paths.save_model_dir}/{config.data.template}/{config.experiment}/{config.model_name}/lightning_logs/version_0/checkpoints/")

    test_ds_path = f"{config.paths.data_dir}{config.data.template}_test.zarr"


    assert config.shuffle_in_chunks is False, "no shuffling allowed for iterative predictions"
    assert config.shuffle_chunks is False, "no shuffling allowed for iterative predictions"
    
    ds = Conditional_Dataset_Zarr_Iterable(test_ds_path, ds_config.template, shuffle_chunks=config.shuffle_chunks, 
                                                shuffle_in_chunks=config.shuffle_in_chunks)
    
    model_ckpt = [x for x in model_load_dir.iterdir()][0]

    conditioning_channels = ds.array_inputs.shape[1] * len(ds.conditioning_timesteps) + ds.array_constants.shape[0]
    generated_channels = ds.array_targets.shape[1]

    restored_model = PixelDiffusionConditional.load_from_checkpoint(
        model_ckpt,
        config=ml_config.experiment.pixel_diffusion,
        conditioning_channels=conditioning_channels,
        generated_channels=generated_channels,
        loss_fn=config.loss_fn,
        sampler=config.sampler,
    )

    dl = DataLoader(ds, batch_size=config.batchsize)
    
    n_steps =  config.n_steps

    constants = torch.tensor(ds.array_constants[:], dtype=torch.float).to(restored_model.device)

    out = []
    for i in range(nens):  # loop over ensemble members
        ts = []
        for b in dl:  # loop over batches in test set
            input = b
            trajectories = torch.zeros(size=(b[1].shape[0], n_steps, *b[1].shape[1:]))
            for step in range(n_steps):
                restored_model.eval()
                with torch.no_grad():  
                    res = restored_model.forward(input)  # is this a list of tensors or a tensor?
                    trajectories[:,step,...] = res
                    input = [torch.concatenate([res, constants.unsqueeze(0).expand(res.size(0), *constants.size())], dim=1), None]  # we don't need the true target here
            ts.append(trajectories)
        out.append(torch.cat(ts, dim=0))
    out = torch.stack(out, dim=0)
                    

    # get the targets:
    targets = torch.tensor(ds.data.targets.data[ds.start+ds.lead_time:ds.stop+ds.lead_time])
    l = len(targets)

    indices = torch.tensor([torch.arange(i, i+n_steps) for i in range(l-n_steps+1)])
    targets = targets[indices,:,:,:].unsqueeze(dim=0)
    # fill targets with infty values until we reach the shape we need
    torch.cat([targets, torch.ones((targets.shape[0], n_steps-1, *targets.shape[2:]))*torch.inf], dim=1)

    print(res.shape, targets.shape)

    model_output_dir = os.path.join(model_output_dir, config.data.template, config.experiment, model_name, dir_name)
    create_dir(model_output_dir)

    gen_xr = create_xr_output_variables(
        out,
        zarr_path=f"{config.paths.data_dir}/{config.data.template}_test.zarr/targets",
        config=ds_config,
        min_max_file_path=f"{config.paths.data_dir}/{config.data.template}_output_min_max.nc"
    )

    target_xr = create_xr_output_variables(
        targets,
        zarr_path=f"{config.paths.data_dir}/{config.data.template}_test.zarr/targets",
        config=ds_config,
        min_max_file_path=f"{config.paths.data_dir}/{config.data.template}_output_min_max.nc"
    )

    gen_dir = os.path.join(model_output_dir, "gen.nc")
    gen_xr.to_netcdf(gen_dir)
    print(f"Generated data written at: {gen_dir}")

    target_dir = os.path.join(model_output_dir, "target.nc")
    target_xr.to_netcdf(target_dir)
    print(f"Target data written at: {target_dir}")

if __name__ == '__main__':
    main()

