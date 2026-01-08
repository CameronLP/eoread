from numpy.random import randint
from eoread.utils import filter_metadata, spatial_resample
import xarray as xr
import pytest

@pytest.fixture
def metadata(): return {'a':0, 'b': {'c':1, 'd':2}}

@pytest.fixture(params=[(5,5),(20,30),(500,500)])
def shape(request): return request.param

@pytest.fixture
def array(shape): return xr.DataArray(randint(0,10,shape))


def test_filter_metadata(metadata):
    dico = filter_metadata(metadata, ['a',['b','d']])
    assert dico['b'].get('c') is None

@pytest.mark.parametrize('method', ['linear','repeat'])
def test_spatial_resample(array, method):
    dims = dict(dim_0=50, dim_1=50)
    arr = spatial_resample(array, dims, 10, method)
    assert arr.shape == (50,50)