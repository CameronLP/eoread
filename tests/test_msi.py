#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import xarray as xr

from . import generic
from pathlib import Path
from eoread.msi import get_sample, Level1_MSI
from eoread import eo

resolutions = ['10', '20', '60']


@pytest.fixture(scope="session")
def level1_msi() -> Path: return get_sample(1)

@pytest.fixture(params=resolutions)
def resolution(request):
    return request.param

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, resolution, chunks):
    return Level1_MSI(level1_msi, resolution, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_msi, resolution, chunks):
    Level1_MSI(level1_msi, resolution, chunks=chunks)

def test_main(S2_product):
    generic.test_main(S2_product, angle_data=True)
    
def test_time(level1_msi, resolution, chunks): 
    params = {'dirname': level1_msi, 'resolution': resolution, 'chunks': chunks}
    generic.test_execution_time(Level1_MSI, params)

def test_v1_compat(level1_msi):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_MSI(level1_msi, '60', v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_msi.stem+'_60'))
    generic.compare_version(l1, old)
    
def test_lazy_load(S2_product):
    generic.test_lazy_load(S2_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.test_read(S2_product, param, indices, scheduler)

def test_subset(level1_msi, resolution, chunks): 
    l1 = Level1_MSI(level1_msi, resolution, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, S2_product):
    generic.test_plot(request, S2_product, 4)

################################################################################
# Tests for Level-2
################################################################################

@pytest.fixture
def level2_msi(): pass

@pytest.mark.skip('test should be updated')
def test_level2(level2_msi: Path):
    assert level2_msi.exists()
