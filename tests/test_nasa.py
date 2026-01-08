#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import pytest

# from eoread.nasa import Level1_NASA
# from eoread.make_L1C import makeL1C
# from . import generic


# nasa_products = [
#     '/archive2/data/EOREAD_TESTDATA/SeaWiFS/S2004115125135.L1A_GAC.Z',
#     '/archive2/data/EOREAD_TESTDATA/VIIRS/V2019086125400.L1A_SNPP.nc'
# ]

# @pytest.fixture(params=nasa_products)
# def filename(request): return request.param


# def test_L1C(filename):
#     assert makeL1C(filename)

# def test_instantiate(filename):
#     Level1_NASA(makeL1C(filename))

# def test_main(filename):
#     l1 = Level1_NASA(makeL1C(filename))
#     generic.test_main(l1)

# def test_read(filename, param, indices):
#     l1 = Level1_NASA(makeL1C(filename))
#     generic.test_read(l1, param, indices, scheduler='sync')

# def test_subset(filename):
#     l1 = Level1_NASA(makeL1C(filename))
#     generic.test_subset(l1)
