# inbuilt packages
import argparse
from pathlib import Path
from time import time

# Standard packages
import numpy as np
import torch
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Custom packages
from dm_zoo.dff.PixelDiffusion import (
    PixelDiffusionConditional,
)
from WD.datasets import Conditional_Dataset, custom_collate
from WD.utils import create_dir
from WD.io import load_config, create_xr_output_variables

parser = argparse.ArgumentParser(
    prog="Evalulate Model",
    description="Evaluate Model based on dataset id and model id",
    epilog="Arg parser for vanilla conditional diffusion model",
)

parser.add_argument(
    "-did",
    "--dataset_id",
    type=str,
    help="path under which the selected config file is stored.",
)


parser.add_argument(
    "-mid",
    "--model_id",
    type=str,
    help="path under which the selected config file is stored.",
)

parser.add_argument(
    "-nens",
    "--n_ensemble_members",
    type=int,
    help="the number of ensemble members to be produced.",
)

args = parser.parse_args()

ds_id = args.dataset_id
run_id = args.model_id
nens = args.n_ensemble_members


B = 1024
num_copies = nens

start_time = time()


def write_dataset(
    restored_model, ds, B, num_copies, model_config, epoch, model_output_dir
):
    dl = DataLoader(
        ds,
        batch_size=B,
        shuffle=False,
        collate_fn=lambda x: custom_collate(x, num_copies=num_copies),
    )
    trainer = pl.Trainer()
    out = trainer.predict(restored_model, dl)

    out = torch.cat(out, dim=0)
    out = out.view(-1, num_copies, *out.shape[1:]).transpose(0, 1)

    model_output_dir = model_output_dir / model_config.ds_id / str(epoch)
    create_dir(model_output_dir)

    targets = ds[:][1].view(1, *ds[:][1].shape)
    # need the view to create axis for different
    # ensemble members (although only 1 here).
    dates = ds[:][2]

    gen_xr = create_xr_output_variables(
        out,
        dates=dates,
        config_file_path=(
            "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id)
        ),
        min_max_file_path=(
            "/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc"
            .format(ds_id)
        ),
    )

    target_xr = create_xr_output_variables(
        targets,
        dates=dates,
        config_file_path=(
            "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id)
        ),
        min_max_file_path=(
            "/data/compoundx/WeatherDiff/model_input/{}_output_min_max.nc"
            .format(ds_id)
        ),
    )

    return F.mse_loss(
        torch.tensor(gen_xr["z_500"].values),
        torch.tensor(target_xr["z_500"].values),
    )


model_config_path = "/data/compoundx/WeatherDiff/config_file/{}_{}.yml".format(
    ds_id, run_id
)
model_output_dir = Path("/data/compoundx/WeatherDiff/model_output/")

print(model_config_path)
model_config = load_config(model_config_path)

model_load_dir = (
    Path(model_config.file_structure.dir_saved_model)
    / "lightning_logs/version_0/checkpoints/"
)

mse_loss_test = []
mse_loss_val = []
epoch_list = []
for epoch in range(0, 120, 5):
    model_ckpt = [
        x for x in model_load_dir.iterdir() if f"epoch={epoch}-" in str(x)
    ][0]

    restored_model = PixelDiffusionConditional.load_from_checkpoint(
        model_ckpt,
        generated_channels=model_config.model_hparam["generated_channels"],
        condition_channels=model_config.model_hparam["condition_channels"],
        cylindrical_padding=True,
    )
    print(epoch)
    ds_test = Conditional_Dataset(
        "/data/compoundx/WeatherDiff/model_input/{}_test.pt".format(ds_id),
        "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
    )

    loss = write_dataset(
        restored_model,
        ds_test,
        B,
        num_copies,
        model_config,
        epoch,
        model_output_dir,
    )
    mse_loss_test.append(loss)
    print(loss)
    ds_val = Conditional_Dataset(
        "/data/compoundx/WeatherDiff/model_input/{}_val.pt".format(ds_id),
        "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
    )

    loss = write_dataset(
        restored_model,
        ds_val,
        B,
        num_copies,
        model_config,
        epoch,
        model_output_dir,
    )
    mse_loss_val.append(loss)
    epoch_list.append(epoch)
    print(loss)

mse_loss = np.vstack([epoch_list, mse_loss_test, mse_loss_val])
np.savetxt("test_val_loss.txt", np.array(mse_loss))

print(f"Total time taken is {np.round(time()-start_time, 2)} seconds")

# ds_train = Conditional_Dataset(
#     "/data/compoundx/WeatherDiff/model_input/{}_train.pt".format(ds_id),
#     "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
# )


# ds_val = Conditional_Dataset(
#     "/data/compoundx/WeatherDiff/model_input/{}_val.pt".format(ds_id),
#     "/data/compoundx/WeatherDiff/config_file/{}.yml".format(ds_id),
# )


# model_config.file_structure.dir_model_output = str(model_output_dir)


# write_config(model_config)
# Write config is possible deletes and rewrites
