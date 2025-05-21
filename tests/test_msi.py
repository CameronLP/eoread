#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
import xarray as xr
from eoread.msi import Level1_MSI, get_sample, get_SRF
from . import generic
from eoread import eo
from . import conftest
from matplotlib import pyplot as plt
from .generic import param, indices  # noqa

resolutions = ['10', '20', '60']


@pytest.fixture(scope="module")
def level1_msi() -> Path: return get_sample(1)

@pytest.fixture(params=resolutions)
def resolution(request):
    return request.param

@pytest.fixture(params=[500, (400, 600)])
def chunks(request):
    return request.param

@pytest.fixture
def S2_product(level1_msi, resolution, chunks):
    return Level1_MSI(level1_msi, resolution, chunks=chunks)


def test_instantiation(level1_msi, resolution, chunks):
    Level1_MSI(level1_msi, resolution, chunks=chunks)


@pytest.mark.parametrize('param', ['sza', 'vza', 'saa', 'vaa', 'latitude', 'longitude'])
def test_msi_merged(S2_product, param):
    l1 = S2_product
    print(l1)
    assert 'Rtoa_443' not in l1
    assert 'Rtoa' in l1

    # check parameter consistency through windowing
    xr.testing.assert_allclose(
        l1[param][1000, 500],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][0, 0])

    xr.testing.assert_allclose(
        l1[param][1000:1010, 500:510],
        l1.isel(y=slice(1000, None),
                x=slice(500, None))[param][:10, :10])

    if resolution == '60':
        full = l1[param].compute(scheduler='single-threaded')
        xr.testing.assert_allclose(
            full[1000:1010, 500:510],
            l1.isel(y=slice(1000, None),
                    x=slice(500, None))[param][:10, :10])


def test_main(S2_product):
    generic.test_main(S2_product)


@pytest.mark.parametrize('scheduler', [
    'single-threaded',
    'threads',
])
def test_read(S2_product, param, indices, scheduler):
    eo.init_geometry(S2_product)
    generic.test_read(S2_product, param, indices, scheduler)


def test_subset(S2_product):
    generic.test_subset(S2_product)


def test_plot(request, level1_msi):
    l1 = Level1_MSI(level1_msi)
    plt.imshow(
        l1.Rtoa.sel(bands=865),
        vmin=0, vmax=0.5)
    plt.colorbar()

    conftest.savefig(request)


@pytest.fixture(params=[
    {'product': 'S2B_MSIL2A_20190901T105619_N0213_R094_T30TWT_20190901T141237',
     'source': 'google'},
    {'product': 'S2A_MSIL2A_20230418T105621_N0509_R094_T31UCR_20230418T170158',
     'source': 'scihub'},
])
def level2_msi(request):
    dir_samples = Path(__file__).parent.parent/'SAMPLE_DATA'
    source = request.param['source']
    product = request.param['product']
    if source == 'google':
        return download_S2_google(product, dir_samples)
    elif source == 'scihub':
        target = dir_samples/product
        download_sentinelapi(target)
        return target
    else:
        raise ValueError

@pytest.mark.skip('test should be updated')
def test_level2(request, level2_msi: Path):
    assert level2_msi.exists()


@pytest.mark.parametrize('sensor', ["S2A", "S2B"])
def test_srf(sensor, request):
    srf = get_SRF(sensor)
    