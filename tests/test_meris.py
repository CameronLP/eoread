#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
from eoread.meris import Level1_MERIS, get_sample
import xarray as xr
from core.env import getdir
from . import generic


product = '/archive2/data/EOREAD_TESTDATA/MERIS/MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1'

@pytest.fixture(scope="module")
def level1_meris(): return get_sample(1)

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def product_meris(level1_meris, chunks):
    return Level1_MERIS(level1_meris, chunks=chunks)


def test_instantiation(level1_meris, chunks):
    Level1_MERIS(level1_meris, chunks=chunks)

def test_main(product_meris):
    generic.test_main(product_meris, angle_data=True)
    
def test_time(level1_meris, chunks): 
    params = {'filepath': level1_meris, 'chunks': chunks}
    generic.test_execution_time(Level1_MERIS, params)

def test_v1_compat(level1_meris):
    v1_data = getdir("DIR_V1_COMPAT_DATA")
    l1 = Level1_MERIS(level1_meris, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_meris.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_lazy_load(product_meris):
    generic.test_lazy_load(product_meris)
    
def test_subset(level1_meris, chunks): 
    l1 = Level1_MERIS(level1_meris, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, product_meris):
    generic.test_plot(request, product_meris, 4)

