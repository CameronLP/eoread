#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from eoread.venus import *
from . import generic

import pytest
import xarray as xr


# product_l1 = pytest.fixture(lambda: get_sample(1), scope='module')
# product_l2 = pytest.fixture(lambda: get_sample(2), scope='module')
product_l1 = Path('/mnt/ceph/data/VENUS/VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1/')
product_l2 = Path('/mnt/ceph/data/VENUS/VENUS-XS_20230116-112657-000_L2A_VILAINE_C_V3-1/')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def VENUS_product(chunks):
    return Level1_VENUS(product_l1, chunks=chunks)

    
def test_l1c_instantiation(chunks):
    Level1_VENUS(product_l1, chunks=chunks)
    
def test_l1c_main(VENUS_product):
    generic.test_main(VENUS_product, angle_data=False)
    
def test_l1c_time(chunks): 
    params = {'dirname': product_l1, 'chunks': chunks}
    generic.test_execution_time(Level1_VENUS, params)

def test_l1c_subset(VENUS_product):
    generic.test_subset(VENUS_product)
    
def test_l1c_v1_compat():
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_VENUS(product_l1, v1_compat=True)
    old = xr.open_dataset(v1_data/(product_l1.stem+f'_res'))
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(VENUS_product):
    generic.test_lazy_load(VENUS_product)

def test_level2(chunks):
    Level2_VENUS(product_l2, chunks=chunks)