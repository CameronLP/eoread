import pytest

from eoread.ecostress import Level1_ECOSTRESS, get_sample
from . import generic


# @pytest.fixture(scope="module")
# def level1_ecostress(): return get_sample(1)

@pytest.fixture()
def level1_ecostress(): return "/home/nathan/proj/eoread/data/sample_products/ECOv002_L1CG_RAD_30110_005_20231028T094350_0711_01.h5"

@pytest.fixture()
def product_ecostress(level1_ecostress):
    return Level1_ECOSTRESS(level1_ecostress)

def test_main(product_ecostress):
    generic.test_main(product_ecostress)

def test_subset(product_ecostress):
    generic.test_subset(product_ecostress)