import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from WD.utils import create_dir

from dm_zoo.diffusion.LFD_lightning import LatentForecastDiffusion

from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.io import create_xr_output_variables

from WD.utils import AreaWeightedMSELoss


@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/inference", config_name="config")
def LFD_inference(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part

    model_name = config.model_name  # we have to pass this to the bash file every time! (should contain a string).
    nens = config.n_ensemble_members
    ds_config = OmegaConf.load(f"{config.paths.hydra_config_dir}/{config.data.template}/.hydra/config.yaml")
    ml_config = OmegaConf.load(f"{config.paths.hydra_config_dir}/training/{config.data.template}/{config.experiment}/{config.model_name}/.hydra/config.yaml")

    model_output_dir = config.paths.inference_dir

    model_load_dir = Path(f"{config.paths.save_model_dir}/{config.data.template}/{config.experiment}/{config.model_name}/lightning_logs/version_0/checkpoints/")

    test_ds_path = f"{config.paths.data_dir}{config.data.template}_test.zarr"

    ds = Conditional_Dataset_Zarr_Iterable(test_ds_path, ds_config.template, shuffle_chunks=config.shuffle_chunks, 
                                                shuffle_in_chunks=config.shuffle_in_chunks)
    
    conditioning_channels = ds.array_inputs.shape[1] * len(ds.conditioning_timesteps) + ds.array_constants.shape[0]
    generated_channels = ds.array_targets.shape[1]
    img_size = ds.array_targets.shape[-2:]

    print(ml_config)
    
    model_ckpt = [x for x in model_load_dir.iterdir()][0]
    if ml_config.experiment.model.loss_fn_name == "MSE_Loss":
        loss_fn = torch.nn.functional.mse_loss
    elif ml_config.experiment.model.loss_fn_name == "AreaWeighted_MSE_Loss":
        lat_grid = ds.data.targets.lat[:]
        lon_grid =  ds.data.targets.lon[:]
        loss_fn = AreaWeightedMSELoss(lat_grid, lon_grid).loss_fn
    else:
        raise NotImplementedError("Invalid loss function.")
    
    if ml_config.experiment.model.diffusion.sampler_name == "DDPM":  # this is the default case
        sampler = None
    else:
        raise NotImplementedError("This sampler has not been implemented.")  
    
    with open_dict(ml_config):
        ml_config.experiment.model.image_size = img_size
        ml_config.experiment.model.generated_channels = generated_channels
        ml_config.experiment.model.conditioning_channels = conditioning_channels

    restored_model = LatentForecastDiffusion.load_from_checkpoint(model_ckpt, map_location="cpu", 
                            model_config = ml_config.experiment.model,
                            loss_fn = loss_fn,
                            sampler = sampler)
    
    dl = DataLoader(ds, batch_size=ml_config.experiment.model.batch_size)
    trainer = pl.Trainer()

    out = []
    for i in range(nens):
        pred = trainer.predict(restored_model, dl)
        pred = torch.cat(pred, dim=0).unsqueeze(dim=0)
        out.append(pred)

    out = torch.cat(out, dim=0) # to keep compatible with the version that uses ensemble members

    print(out.shape)
    model_output_dir = os.path.join(model_output_dir, config.data.template, config.experiment, model_name, dir_name)
    create_dir(model_output_dir)

    # need the view to create axis for
    # different ensemble members (although only 1 here).

    targets = torch.tensor(ds.data.targets.data[ds.start+ds.lead_time:ds.stop+ds.lead_time], dtype=torch.float).unsqueeze(dim=0)

    
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
    LFD_inference()

