#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import xarray as xr

from eoread.sgli import get_sample, Level1_SGLI
from . import generic


sgli_filename = get_sample()

@pytest.fixture(scope="module")
def level1_sgli() -> Path: return get_sample()

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def sgli_product(level1_sgli, chunks):
    return Level1_SGLI(level1_sgli, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_sgli, chunks):
    Level1_SGLI(level1_sgli, chunks=chunks)

def test_main(sgli_product):
    generic.test_main(sgli_product, angle_data=True)
    
def test_time(level1_sgli, chunks): 
    params = {'filepath': level1_sgli, 'chunks': chunks}
    generic.test_execution_time(Level1_SGLI, params)

def test_v1_compat(level1_sgli):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_SGLI(level1_sgli, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_sgli.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_lazy_load(sgli_product):
    generic.test_lazy_load(sgli_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(sgli_product, param, indices, scheduler):
    generic.test_read(sgli_product, param, indices, scheduler)

def test_subset(level1_sgli, chunks): 
    l1 = Level1_SGLI(level1_sgli, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, sgli_product):
    generic.test_plot(request, sgli_product, 4)