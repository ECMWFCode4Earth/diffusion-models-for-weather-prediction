import xarray as xr
from WD.utils import comp_area_lat_lon
import h5py

deg = 5.625
data_path = (
    "/data/compoundx/WeatherBench/2m_temperature/"
    + f"2m_temperature_1979_{deg}deg.nc"
)

data = xr.load_dataset(data_path)
area = comp_area_lat_lon(data.lat.values, data.lon.values)
area = area / area.sum()

save_path = "/data/compoundx/WeatherBench/"

with h5py.File(save_path + "gridarea.h5", "a") as f:
    # ! We append the grid area folder and not rewrite them
    f[f"norm_gridarea_{int(360/deg)}x{int(180/deg)}"] = area
