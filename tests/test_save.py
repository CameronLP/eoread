import pytest 

from core.save import to_netcdf
from eoread.landsat9_oli import Level1_L9_OLI

from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture()
def level1_example():
    filepath = '/archive2/proj/QTIS_TRISHNA/L8L9/USA/LC09_L1TP_014034_20220618_20230411_02_T1'
    return Level1_L9_OLI(filepath)


@pytest.mark.skip('to_netcdf does not support xr.DataArray')
def test_to_netcdf_dataarray(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.nc'
        to_netcdf(level1_example['Rtoa'], filename=outpath)
        
def test_to_netcdf_dataset(level1_example):
    with TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)/'test.nc'
        to_netcdf(level1_example, filename=outpath)
