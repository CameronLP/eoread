from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from core import env
from core.tools import xrcrop

from eoread.dem import GTOPO30, SRTM
from eoread.landsat_oli import Level1_OLI


@pytest.fixture
def l1_path() -> Path:
    return env.getdir('DIR_SAMPLE_LANDSAT9')

def test_srtm(l1_path: Path):
    with TemporaryDirectory() as tmpdir:
        l1 = Level1_OLI(l1_path)
        srtm = SRTM(directory=tmpdir, missing=0)
        sub = xrcrop(srtm, latitude=l1.latitude, longitude=l1.longitude)
        sub.compute(scheduler='sync')

def test_gtopo(l1_path: Path):
    with TemporaryDirectory() as tmpdir:
        l1 = Level1_OLI(l1_path)
        gtopo = GTOPO30(directory=tmpdir, missing=0)
        sub = xrcrop(gtopo, latitude=l1.latitude, longitude=l1.longitude)
        sub.compute()