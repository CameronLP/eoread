#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from eoread import eo
from eoread.landsat9_oli import *

from . import generic
from .generic import indices, param, scheduler


# product_l1 = pytest.fixture(lambda: get_sample(1), scope='module')
# product_l2 = pytest.fixture(lambda: get_sample(2), scope='module')
product_l1 = '/mnt/ceph/data/LANDSAT9/USA/LC09_L1TP_014034_20220618_20230411_02_T1/'
product_l2 = '/mnt/ceph/data/LAN/'

@pytest.fixture(scope='module')
def sample_landsat9_oli(): return get_sample()


@pytest.mark.parametrize('split', [True, False])
def test_instantiate(sample_landsat9_oli, split):
    l1 = Level1_L9_OLI(sample_landsat9_oli, split=split)

    if split:
        assert 'Rtoa' not in l1
        assert 'Rtoa_440' in l1
    else:
        assert 'Rtoa' in l1
        assert 'Rtoa_440' not in l1


def test_main():
    l1 = Level1_L9_OLI(product_l1)
    generic.test_main(l1)

def test_subset(sample_landsat9_oli):
    l1 = Level1_L9_OLI(sample_landsat9_oli)
    generic.test_subset(l1)

@pytest.mark.parametrize('radio, angle', [
    ('radiance', False),
    ('reflectance', True)])
def test_radiometry(sample_landsat9_oli, radio, angle):
    l1 = Level1_L9_OLI(sample_landsat9_oli, radiometry=radio)
    generic.test_main(l1, angle)