#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import xarray as xr 
from eoread.olci import *
from eoread.olci import get_valid_l2_pixels
from eoread import eo
from . import generic


olci_level1 = pytest.fixture(lambda: get_sample(1), scope='module')
olci_level2 = pytest.fixture(lambda: get_sample(2), scope='module')

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def OLCI_product(chunks, olci_level1):
    return Level1_OLCI(olci_level1, chunks=chunks)

    
def test_l1c_instantiation(chunks, olci_level1):
    Level1_OLCI(olci_level1, chunks=chunks)
    
def test_l1c_main(OLCI_product):
    generic.test_main(OLCI_product, angle_data=False)
    
def test_l1c_time(chunks, olci_level1): 
    params = {'dirname': olci_level1, 'chunks': chunks}
    generic.test_execution_time(Level1_OLCI, params)

def test_l1c_subset(OLCI_product):
    generic.test_subset(OLCI_product)
    
def test_l1c_v1_compat(olci_level1):
    v1_data = Path('/mnt/ceph/data/eoread')
    l1 = Level1_OLCI(olci_level1, v1_compat=True)
    old = xr.open_dataset(v1_data/(olci_level1.stem))
    old = old.reset_coords(['altitude','latitude','longitude'])
    generic.compare_version(l1, old)
    
def test_l1c_lazy_load(OLCI_product):
    generic.test_lazy_load(OLCI_product)

@pytest.mark.skip('No level 2')
def test_level2(chunks, olci_level2):
    Level2_OLCI(olci_level2, chunks=chunks)

@pytest.mark.skip('No level 2')
def test_sub_pt(olci_level1):
    ds = Level1_OLCI(olci_level1)
    lat0 = ds.latitude[500, 500]
    lon0 = ds.longitude[500, 500]
    eo.sub_pt(ds, lat0, lon0, 3)

@pytest.mark.skip('No level 2')
def test_olci_level2_flags(olci_level2):
    l2 = Level2_OLCI(olci_level2)

    eo.getflags(l2.wqsf)
    get_valid_l2_pixels(l2.wqsf)
