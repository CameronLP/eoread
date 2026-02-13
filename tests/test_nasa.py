#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from core.env import getvar

from eoread.make_L1C import makeL1C
from eoread.nasa import Level1_NASA
from tests import generic

nasa_products_L1A = [
    {"path": Path(getvar("LEVEL1A_SAMPLE_HAWKEYE")), "band_nir": 867},
    {"path": Path(getvar("LEVEL1A_SAMPLE_SEAWIFS")), "band_nir": 865},
    # TODO: MODIS and VIIRS
]


@pytest.fixture(
    params=nasa_products_L1A, ids=[x["path"].name for x in nasa_products_L1A]
)
def product_L1A(request):
    prod = request.param
    assert prod["path"].exists()
    return prod


@pytest.mark.parametrize(
    "method",
    [
        # 'docker',   # TODO
        "shell",
    ],
)
def test_L1C(method, product_L1A: dict):
    with TemporaryDirectory() as tmpdir:
        makeL1C(product_L1A["path"], Path(tmpdir), method=method, eline=100)


def test_instantiate(product_L1A: dict):
    product_L1C = makeL1C(product_L1A["path"])
    Level1_NASA(product_L1C)


def test_plot(request, product_L1A: dict):
    product_L1C = makeL1C(product_L1A["path"])
    l1 = Level1_NASA(product_L1C)
    generic.plot(request, l1, product_L1A["band_nir"])
