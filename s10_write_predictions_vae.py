import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as L

import hydra
from omegaconf import DictConfig, OmegaConf
from WD.utils import create_dir

from dm_zoo.latent.vae.vae_lightning_module import VAE
from WD.datasets import Conditional_Dataset_Zarr_Iterable
from WD.io import create_xr_output_variables

import numpy as np

@hydra.main(version_base=None, config_path="./config", config_name="inference")
def vae_inference(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    dir_name = hydra_cfg['runtime']['output_dir']  # the directory the hydra log is written to.
    dir_name = os.path.basename(os.path.normpath(dir_name))  # we only need the last part

    model_name = config.model_name  # we have to pass this to the bash file every time! (should contain a string).
    experiment_name = hydra_cfg['runtime']['choices']['experiment']

    ds_config = OmegaConf.load(f"{config.paths.dir_HydraConfigs}/data/{config.data.template}/.hydra/config.yaml")
    ml_config = OmegaConf.load(f"{config.paths.dir_HydraConfigs}/training/{config.data.template}/{experiment_name}/{config.model_name}/.hydra/config.yaml")

    model_output_dir = config.paths.dir_ModelOutput

    model_load_dir = Path(f"{config.paths.dir_SavedModels}/{config.data.template}/{experiment_name}/{config.model_name}/lightning_logs/version_0/checkpoints/")

    test_ds_path = f"{config.paths.dir_PreprocessedData}{config.data.template}_test.zarr"

    ds = Conditional_Dataset_Zarr_Iterable(test_ds_path, ds_config.template, shuffle_chunks=config.shuffle_chunks, 
                                                shuffle_in_chunks=config.shuffle_in_chunks)
    
    conditioning_channels = ds.array_inputs.shape[1] * len(ds.conditioning_timesteps) + ds.array_constants.shape[0]
    generated_channels = ds.array_targets.shape[1]
    img_size = ds.array_targets.shape[-2:]

    print(ml_config)

    if ml_config.experiment.vae.type == "input":
        n_channel = conditioning_channels
    elif ml_config.experiment.vae.type == "output":
        n_channel = generated_channels
    else:
        raise AssertionError
    
    in_shape = (n_channel, img_size[0], img_size[1])

    model_ckpt = [x for x in model_load_dir.iterdir()][0]

    restored_model = VAE.load_from_checkpoint(model_ckpt, map_location="cpu", 
                inp_shape = in_shape, 
                dim=ml_config.experiment.vae.dim, 
                channel_mult = ml_config.experiment.vae.channel_mult, 
                batch_size = ml_config.experiment.vae.batch_size, 
                lr = ml_config.experiment.vae.lr, 
                lr_scheduler_name=ml_config.experiment.vae.lr_scheduler_name, 
                num_workers = ml_config.experiment.vae.num_workers, 
                beta = ml_config.experiment.vae.beta,
                data_type = ml_config.experiment.vae.type)
    
    dl = DataLoader(ds, batch_size=ml_config.experiment.vae.batch_size)

    out = []
    for i, data in enumerate(dl):
        
        r, _, x, _ = restored_model(data)
        
        if i==0:
            print(r.shape, x.shape)
            print(f"Input reduction factor: {np.round(np.prod(r.shape[1:])/np.prod(x.shape[1:]), decimals=2)}")
        
        out.append(r)
        
    out = torch.cat(out, dim=0).unsqueeze(dim=0) # to keep compatible with the version that uses ensemble members

    model_output_dir = os.path.join(model_output_dir, config.data.template, experiment_name, model_name, dir_name)
    create_dir(model_output_dir)

    # need the view to create axis for
    # different ensemble members (although only 1 here).

    targets = torch.tensor(ds.data.targets.data[ds.start+ds.lead_time:ds.stop+ds.lead_time], dtype=torch.float).unsqueeze(dim=0)

    
    gen_xr = create_xr_output_variables(
        out,
        zarr_path=f"{config.paths.dir_PreprocessedDatasets}/{config.data.template}_test.zarr/targets",
        config=ds_config,
        min_max_file_path=f"{config.paths.dir_PreprocessedDatasets}/{config.data.template}_output_min_max.nc"
    )


    target_xr = create_xr_output_variables(
        targets,
        zarr_path=f"{config.paths.dir_PreprocessedDatasets}/{config.data.template}_test.zarr/targets",
        config=ds_config,
        min_max_file_path=f"{config.paths.dir_PreprocessedDatasets}/{config.data.template}_output_min_max.nc"
    )

    gen_dir = os.path.join(model_output_dir, "gen.nc")
    gen_xr.to_netcdf(gen_dir)
    print(f"Generated data written at: {gen_dir}")

    target_dir = os.path.join(model_output_dir, "target.nc")
    target_xr.to_netcdf(target_dir)
    print(f"Target data written at: {target_dir}")

if __name__ == '__main__':
    vae_inference()

