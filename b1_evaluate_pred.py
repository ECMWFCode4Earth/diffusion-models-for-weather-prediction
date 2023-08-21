import xarray as xr
import numpy as np

import cartopy.crs as ccrs

import copy
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

from WD.plotting import plot_map, add_label_to_axes
from benchmark.bm.score import compute_weighted_rmse, compute_weighted_mae, compute_weighted_acc
from WD.utils import create_dir

# usually this would go on top of the notebook:
plt.rcParams.update({'font.size': 8})  # sets font size for all cells
plt.rcParams['figure.dpi'] = 300  # sets dpi for all cells

alphabet_letters = np.array(list(map(chr, range(65, 91))))  # used for labelling subplots
textwidth = 170  # 144  # in mm
mm_to_inch = 0.0393701
textwidth = textwidth * mm_to_inch # textwidth in inches

# ds_id = "96FE8A" # "278771"
# model_id = "290AD8" # "011A3B" "C7A2A3"

# targets = xr.load_dataset(f"/data/compoundx/WeatherDiff/model_output/{ds_id}/{model_id}_target.nc")
# predictions = xr.load_dataset(f"/data/compoundx/WeatherDiff/model_output/{ds_id}/{model_id}_gen.nc")


# data_template_name = "geopotential_500_highres"
# experiment_name = "fourcastnet_highres"
# run_date = "2023-08-18_13-10-40"
# eval_date = "2023-08-19_11-01-41"


data_template_name = "geopotential_500"
experiment_name = "fourcastnet_small"
run_date = "2023-08-18_16-46-00"
eval_date = "2023-08-18_19-01-32"


# data_template_name = "rasp_thuerey_geopotential"
# experiment_name = "fourcastnet_rasp_thuerey"
# run_date = "2023-08-18_14-18-40"
# eval_date = "2023-08-18_17-47-29"


# data_template_name = "geopotential_500_highres"
# experiment_name = "fourcastnet_highres"
# run_date = "2023-08-17_20-18-22"
# eval_date = "2023-08-18_12-11-04"


# data_template_name = "geopotential_500"
# experiment_name = "fourcastnet"
# run_date = "2023-08-18_13-10-44"
# eval_date = "2023-08-18_16-07-24"

targets = xr.load_dataset(f"/data/compoundx/WeatherDiff/model_output/{data_template_name}/{experiment_name}/{run_date}/{eval_date}/target.nc")
predictions = xr.load_dataset(f"/data/compoundx/WeatherDiff/model_output/{data_template_name}/{experiment_name}/{run_date}/{eval_date}/gen.nc")

diff = targets - predictions

n_images = 8

timesteps = np.random.choice(np.arange(len(predictions["init_time"])), size=(n_images,))

# do configurations for plotting - these can also be shared or "inherited" for plots that are similar!

config = {
    "CMAP": "viridis",
    "NORM": matplotlib.colors.Normalize(vmin=49000, vmax=59000),
    "RASTERIZED": True, # don't plot map pixels as individual points to make created files smaller
    "SHOW_COLORBAR": True,
    "CBAR_ORIENTATION": "horizontal",
    "CBAR_EXTEND": "both",
    "SHOW_COLORBAR_LABEL": False,
    "CBAR_LABEL": r"Geopotential [$m^2/s^2$]",
    "TITLE": "",
    "TITLE_FONTSIZE": 8,
    "PROJECTION": ccrs.Robinson(), # this is not called by plot_map, but by the function we create the entire plot with.
    "ASPECT_RATIO": 6/5  # can be used to calculate a figsize that looks nice for a given type of plot
}


config_diff = copy.deepcopy(config)

config_diff["CMAP"] = "RdBu"
config_diff["NORM"] = matplotlib.colors.Normalize(vmin=-3000, vmax=3000)

n_rows = n_images
n_cols = 3

figure_width = textwidth
# calculate height from number of rows, cols and aspect ratio (+ do some fine tuning)
figure_height = textwidth * (n_rows / n_cols) / config["ASPECT_RATIO"]

fig = plt.figure(figsize = [figure_width, figure_height])
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, width_ratios=[1,1,1])

for i, i_t in enumerate(timesteps):
    ax = fig.add_subplot(gs[i, 0], projection=config["PROJECTION"])
    # plot the map:
    plot_map(ax, data=targets.isel({"init_time":i_t, "lead_time":0, "ensemble_member": 0})[list(targets.keys())], plotting_config=config, title="Target")
    # add a lael to the panel of the plot:
    add_label_to_axes(ax, "({}1)".format(alphabet_letters[i]))

    ax = fig.add_subplot(gs[i, 1], projection=config["PROJECTION"])
    # plot the map:
    plot_map(ax, data=predictions.isel({"init_time":i_t, "lead_time":0, "ensemble_member": 0})[list(predictions.keys())], plotting_config=config, title="Prediction")
    # add a lael to the panel of the plot:
    add_label_to_axes(ax, "({}2)".format(alphabet_letters[i]))

    ax = fig.add_subplot(gs[i, 2], projection=config["PROJECTION"])
    # plot the map:
    plot_map(ax, data=diff.isel({"init_time":i_t, "lead_time":0, "ensemble_member": 0})[list(diff.keys())], plotting_config=config_diff, title="Difference:")
    # add a lael to the panel of the plot:
    add_label_to_axes(ax, "({}3)".format(alphabet_letters[i]))


fig.canvas.draw()
fig.tight_layout()

save_folder = f"/data/compoundx/WeatherDiff/benchmarks/{data_template_name}/{experiment_name}/{run_date}/{eval_date}/"
create_dir(save_folder)
plt.savefig(save_folder+"predictions.png")

rmse = compute_weighted_rmse(predictions.isel({"ensemble_member": 0}), targets.isel({"ensemble_member": 0}))
mae = compute_weighted_mae(predictions.isel({"ensemble_member": 0}), targets.isel({"ensemble_member": 0}))
acc = compute_weighted_acc(predictions.isel({"ensemble_member": 0}), targets.isel({"ensemble_member": 0}))

bm_dict = {
    "experiment": experiment_name, 
    "run_date": run_date,
    "eval_date": eval_date,
    "n_ensemble": predictions["ensemble_member"].shape[0],
    "RMSE": np.round(rmse.z_500.values, decimals=2 ),
    "MAE": np.round(mae.z_500.values, decimals=2 ),
    "ACC": np.round(acc.z_500.values, decimals=2 ),

}
def write_csv(csv_path, bm_dict):
    header = ["experiment", "run_date", "eval_date", "n_ensemble", "RMSE", "MAE", "ACC"]    
    with open(csv_path, "a", newline="") as csvfile:
        writer=csv.DictWriter(csvfile, fieldnames=header)
        writer.writerow(bm_dict)

write_csv(f"/data/compoundx/WeatherDiff/benchmarks/benchmark.csv", bm_dict=bm_dict)



