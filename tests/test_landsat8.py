#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from eoread.landsat_oli import *
from . import generic


# product_l1 = pytest.fixture(lambda: get_sample(1), scope='module')
# product_l2 = pytest.fixture(lambda: get_sample(2), scope='module')
product_l1 = 'data/sample_products/LC08_L1TP_180054_20250104_20250111_02_T1'
product_l2 = '/mnt/ceph/data/LAN/'


@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture()
def product_l8_oli(chunks):
    return Level1_OLI(product_l1, chunks=chunks)


def test_l1c_instantiate():
    l1 = Level1_OLI(product_l1)

def test_l1c_main(product_l8_oli):
    generic.test_main(product_l8_oli, angle_data=False)
    
def test_l1c_time(chunks): 
    params = {'dirname': product_l1, 'chunks': chunks}
    generic.test_execution_time(Level1_OLI, params)

def test_l1c_subset(product_l8_oli):
    generic.test_subset(product_l8_oli)

@pytest.mark.skip('No output from version 1')
def test_l1c_v1_compat():
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_OLI(product_l1, v1_compat=True)
    old = xr.open_dataset(v1_data/(product_l1.stem+f'_res'))
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(product_l8_oli):
    generic.test_lazy_load(product_l8_oli)