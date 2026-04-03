#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from core.env import getvar

from eoread.make_L1C import makeL1C
from eoread.nasa import Level1_NASA
from tests import generic

nasa_products_L1A = {
    "HAWKEYE": {
        # SEAHAWK1_HAWKEYE.20230701T160442.L1A.nc
        "var": "LEVEL1_HAWKEYE",
        "band_nir": 867,
        "poi": {"x": 100, "y": 5000},
    },
    "SEAWIFS": {
        # SEASTAR_SEAWIFS_GAC.20000312T030717.L1A.nc
        "var": "LEVEL1_SEAWIFS",
        "band_nir": 865,
        "poi": {"x": 50, "y": 750},
    },
    "MODIS": {
        # A2008106124500.L1A_LAC
        "var": "LEVEL1_MODISA",
        "band_nir": 859,
        "poi": {"x": 500, "y": 500},
    },
    "VIIRS": {
        # V2019086125400.L1A_SNPP.nc
        "var": "LEVEL1_VIIRS_SNPP",
        "band_nir": 862,
        "poi": {"x": 1750, "y": 2250},
    },

}


@pytest.fixture(params=nasa_products_L1A.values(), ids=list(nasa_products_L1A.keys()))
def product_L1A(request):
    prod = request.param
    assert Path(getvar(prod["var"])).exists()
    return prod


@pytest.mark.parametrize(
    "method",
    [
        "docker",
        "shell",
    ],
)
def test_L1C(method, product_L1A):
    with TemporaryDirectory() as tmpdir:
        p = Path(getvar(product_L1A["var"]))
        out = makeL1C(p, Path(tmpdir), method=method, eline=100)
        assert out.exists()
        print(f'Generated {out}')


def test_instantiate(product_L1A: dict):
    p = Path(getvar(product_L1A["var"]))
    product_L1C = makeL1C(p)
    Level1_NASA(product_L1C)


def test_plot(request, product_L1A: dict):
    p = Path(getvar(product_L1A["var"]))
    product_L1C = makeL1C(p)
    l1 = Level1_NASA(product_L1C)
    generic.plot(request, l1, product_L1A["band_nir"], product_L1A.get("poi", None))
