#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import xarray as xr

from eoread.landsat_oli import get_sample, Level1_OLI
from . import generic


product_l1 = pytest.fixture(lambda: get_sample(1), scope='module')
product_l2 = pytest.fixture(lambda: get_sample(2), scope='module')


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_l8_oli(product_l1, chunks):
    return Level1_OLI(product_l1, chunks=chunks)


def test_l1c_instantiate(product_l1):
    l1 = Level1_OLI(product_l1)

def test_l1c_main(product_l8_oli):
    generic.test_main(product_l8_oli, angle_data=False)
    
def test_l1c_time(product_l1, chunks): 
    params = {'dirname': product_l1, 'chunks': chunks}
    generic.test_execution_time(Level1_OLI, params)

def test_l1c_subset(product_l8_oli):
    generic.test_subset(product_l8_oli)

@pytest.mark.skip("No version 1 output file")
def test_l1c_v1_compat(product_l1):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_OLI(product_l1, v1_compat=True)
    old = xr.open_dataset(v1_data/(product_l1.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(product_l8_oli):
    generic.test_lazy_load(product_l8_oli)