from pathlib import Path
import xarray as xr
import pytest

from eoread.modis import Level1_MODIS, get_sample
from . import generic


@pytest.fixture(scope="module")
def level1_modis() -> Path: return get_sample(1)

@pytest.fixture(params=[200, (200, 300)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_modis(level1_modis):
    return Level1_MODIS(level1_modis)


def test_l1c_instantiation(chunks, level1_modis):
    Level1_MODIS(level1_modis, chunks=chunks)
    
def test_l1c_main(product_modis):
    generic.test_main(product_modis, angle_data=False)
    
def test_l1c_time(chunks, level1_modis): 
    params = {'filepath': level1_modis, 'chunks': chunks}
    generic.test_execution_time(Level1_MODIS, params)

def test_l1c_subset(product_modis):
    generic.test_subset(product_modis)

def test_l1c_v1_compat(level1_modis):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_MODIS(level1_modis, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_modis.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(product_modis):
    generic.test_lazy_load(product_modis)