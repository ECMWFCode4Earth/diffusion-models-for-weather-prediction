import xarray as xr
import numpy as np
import pytest

from benchmark.bm.ensemble_score import get_ranks, get_ranks_distributions, get_entropy_of_distributions

def test_get_ranks():
    da = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
    ds_wrong_dims = xr.Dataset({"testvar": da})
    with pytest.raises(AssertionError):
        get_ranks(ds_wrong_dims, ds_wrong_dims)


def test_get_entropy_of_distributions():
    test_dict = {"varname": np.array([1,1,1])}
    assert get_entropy_of_distributions(test_dict) == {"varname": np.log(3)} 