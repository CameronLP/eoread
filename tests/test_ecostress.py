import pytest
import xarray as xr

from pathlib import Path
from eoread.ecostress import Level1_ECOSTRESS, get_sample
from . import generic


@pytest.fixture(scope="session")
def level1C_ecostress(): return get_sample(1)

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_ecostress(level1C_ecostress, chunks):
    return Level1_ECOSTRESS(level1C_ecostress, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################
    
def test_l1c_instantiation(level1C_ecostress, chunks):
    Level1_ECOSTRESS(level1C_ecostress, chunks=chunks)
    
def test_l1c_main(product_ecostress):
    generic.test_main(product_ecostress, angle_data=False)
    
def test_l1c_time(level1C_ecostress, chunks): 
    params = {'filepath': level1C_ecostress, 'chunks': chunks}
    generic.test_execution_time(Level1_ECOSTRESS, params)

def test_l1c_subset(product_ecostress):
    generic.test_subset(product_ecostress)
    
def test_l1c_v1_compat(level1C_ecostress):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_ECOSTRESS(level1C_ecostress, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1C_ecostress.stem+f'_res'))
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(product_ecostress):
    generic.test_lazy_load(product_ecostress)


################################################################################
# Tests for Level-2
################################################################################

@pytest.fixture
def level2_msi(): pass

@pytest.mark.skip('test should be updated')
def test_level2(level2_msi: Path):
    assert level2_msi.exists()