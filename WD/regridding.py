# adapted from https://github.com/pangeo-data/WeatherBench/blob/master/src/regrid.py

import xarray as xr
import xesmf as xe

def regrid(
        ds_in,
        ds_res,
        method='bilinear',
        reuse_weights=True
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param ds_res: Output xarray dataset used to extract the output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})

    if 'latitude' in ds_res.coords:
        ds_res = ds_res.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create regridder
    regridder = xe.Regridder(
        ds_in, ds_res, method, periodic=True, reuse_weights=reuse_weights,
    )

    """    
    # Hack to speed up regridding of large files
    ds_list = []
    chunk_size = 500
    n_chunks = len(ds_in.time) // chunk_size + 1
    for i in range(n_chunks):
        ds_small = ds_in.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
        ds_list.append(regridder(ds_small).astype('float32'))
    ds_out = xr.concat(ds_list, dim='time')
    """

    ds_out = regridder(ds_in).astype('float32')
    
    # Set attributes since they get lost during regridding
    for var in ds_out:
        ds_out[var].attrs =  ds_in[var].attrs
    ds_out.attrs.update(ds_in.attrs)

    # # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def regrid_to_res(ds_in:xr.Dataset, out_res: str, weatherbench_path:str="/data/compoundx/WeatherBench", *args, **kwargs)-> xr.Dataset:
    """Regrid to a given resolution contained in the WeatherBench dataset.

    Args:
        ds_in (xr.Dataset): Dataset to be interpolated.
        out_res (str): Target resolution. Must be contained in WeatherBench, and corresponding files must be downloaded.
        weatherbench_path (str): Path under which the WeatherBench directory is stored.
    Returns:
        xr.Dataset: Interpolated dataset.
    """

    assert "deg" in out_res, "Resolution specification must be of the type 1.2345deg"
    
    ds_res = xr.open_dataset("{}/constants/constants_{}.nc".format(weatherbench_path, out_res))

    return regrid(ds_in, ds_res, *args, **kwargs)
