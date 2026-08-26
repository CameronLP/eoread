from pathlib import Path
from eoread.hypso import Level1_HYPSO, _hypso_format, get_sample
import xarray as xr
from core.env import getdir

from . import generic

import pytest


# Confirmed real on-disk examples of both HYPSO L1C layouts (see
# eoread.hypso module docstring): hypso-processing-pipeline is still writing
# the original "grouped" layout as of 2026-08-25 (not yet migrated onto
# hypso-package's refactored writer), and hypso-package's own reference
# capture (used by its test suite) produces the current "flat" layout.
# Skipped automatically wherever these paths aren't present (e.g. CI, a
# fresh checkout without HYPSO_DATA_AOC mounted).
_GROUPED_SAMPLE = Path("/home/camerop/HYPSO_DATA_AOC/aeronetgalata_2025-01-02T08-52-34Z/"
                       "aeronetgalata_2025-01-02T08-52-34Z-moved-l1c.nc")


@pytest.mark.skipif(not _GROUPED_SAMPLE.is_file(), reason="grouped-layout sample not present")
def test_hypso_format_detects_grouped():
    assert _hypso_format(_GROUPED_SAMPLE) == "grouped"


@pytest.mark.skipif(not _GROUPED_SAMPLE.is_file(), reason="grouped-layout sample not present")
def test_level1_hypso_reads_grouped_layout():
    ds = Level1_HYPSO(_GROUPED_SAMPLE, chunks=500, verbose=False)
    assert ds.Ltoa.ndim == 3
    assert ds.latitude.shape == ds.Ltoa.isel(bands=0).shape
    assert "_moved" in ds.attrs["sensor"] or "_original" in ds.attrs["sensor"] or "_adjusted" in ds.attrs["sensor"]


@pytest.fixture(scope="module")
def level1_hypso() -> Path: return get_sample()

@pytest.fixture(params=[500, (400, 600)])
def chunks(request): return request.param

@pytest.fixture
def hypso_product(level1_hypso, chunks):
    return Level1_HYPSO(level1_hypso, chunks=chunks)


################################################################################
# Tests for Level-1
################################################################################

def test_instantiation(level1_hypso, chunks):
    Level1_HYPSO(level1_hypso, chunks=chunks)

def test_main(hypso_product):
    generic.test_main(hypso_product, angle_data=True)
    
def test_time(level1_hypso, chunks): 
    params = {'filepath': level1_hypso, 'chunks': chunks}
    generic.test_execution_time(Level1_HYPSO, params)

def test_v1_compat(level1_hypso):
    v1_data = getdir("DIR_V1_COMPAT_DATA")
    l1 = Level1_HYPSO(level1_hypso, v1_compat=True)
    old = xr.open_dataset(v1_data/(level1_hypso.stem+'_res'))
    generic.compare_version(l1, old)
    
def test_lazy_load(hypso_product):
    generic.test_lazy_load(hypso_product)

@pytest.mark.skip()
@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(hypso_product, param, indices, scheduler):
    generic.test_read(hypso_product, param, indices, scheduler)

def test_subset(level1_hypso, chunks): 
    l1 = Level1_HYPSO(level1_hypso, chunks=chunks, metadata_template=[])
    generic.test_subset(l1)

def test_plot(request, hypso_product):
    generic.test_plot(request, hypso_product, 4)