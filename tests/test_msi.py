#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import xarray as xr

from . import generic
from pathlib import Path
from eoread.msi import get_sample, Level1_MSI
from eoread import eo
from core.env import getdir

resolutions = ['10', '20', '60']


@pytest.fixture(scope="session")
def level1_msi(): 
    return get_sample(1)

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture(params=[True, False])
def concat(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, chunks, concat):
    return Level1_MSI(level1_msi, chunks=chunks, concat=concat)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_msi, chunks):
    Level1_MSI(level1_msi, chunks=chunks)

def test_main(level1_msi, chunks):
    l1 = Level1_MSI(level1_msi, chunks=chunks, concat=True)
    generic.test_main(l1, angle_data=True)
    
def test_time(level1_msi, chunks): 
    params = {'dirname': level1_msi, 'chunks': chunks}
    generic.test_execution_time(Level1_MSI, params)

def test_v1_compat(level1_msi):
    v1_data = getdir("DIR_V1_COMPAT_DATA")
    l1 = Level1_MSI(level1_msi, concat=True, v1_compat=True)
    old = xr.open_dataset(list(v1_data.glob('S2*_60'))[0])
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

def test_subset(level1_msi, chunks): 
    l1 = Level1_MSI(level1_msi, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, level1_msi, chunks):
    l1 = Level1_MSI(level1_msi, chunks=chunks, concat=True)
    generic.test_plot(request, l1, 4)
