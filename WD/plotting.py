import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from cartopy.util import add_cyclic_point

import xarray as xr
from typing import Dict


def plot_map(
    ax: plt.Axes, data: xr.Dataset, plotting_config: Dict, title: str = ""
) -> None:
    """Plot an xarray dataset as a map. The dataset is assumed to have dimensions lat and lon and trivial other dimensions and only contain a single variable. The function works by modifying an existing Axes object.

    Args:
        ax (plt.Axes): A matplotlib Axes to plot in.
        data (xr.Dataset): Data to plot, should have non-trivial dimensions lat and lon  only
        plotting_config (Dict): Some configurations regarding plotting, infer details from code below.
        title (str, optional): Title for this specific panel of the plot. Defaults to "".
    """

    p_data = data[list(data.keys())[0]]

    lat = data.lat.values
    lon = data.lon.values

    ax.set_global()
    # remove white line
    field, lon_plot = add_cyclic_point(p_data, coord=lon)
    lo, la = np.meshgrid(lon_plot, lat)
    ax.pcolormesh(
        lo,
        la,
        field,
        transform=ccrs.PlateCarree(),
        cmap=plotting_config["CMAP"],
        norm=plotting_config["NORM"],
        rasterized=plotting_config["RASTERIZED"],
    )

    if plotting_config["SHOW_COLORBAR"]:
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap=plotting_config["CMAP"], norm=plotting_config["NORM"]
            ),
            spacing="proportional",
            orientation=plotting_config["CBAR_ORIENTATION"],
            extend=plotting_config["CBAR_EXTEND"],
            ax=ax,
        )
        if plotting_config["SHOW_COLORBAR_LABEL"]:
            cbar.set_label(plotting_config["CBAR_LABEL"])

    ax.coastlines()
    if title != "":
        ax.set_title(title, fontsize=plotting_config["TITLE_FONTSIZE"])


def add_label_to_axes(ax, label, fontsize=None):
    if fontsize is None:
        ax.text(0.01, 0.99, label, ha="left", va="top", transform=ax.transAxes)
    else:
        ax.text(
            0.01,
            0.99,
            label,
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=fontsize,
        )
