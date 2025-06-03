#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

import os
import tempfile

import dask
import numpy as np
import pytest

from core.tools import datetime
from core.files import to_netcdf
from core.geo import n


@pytest.fixture(params=[
    'single-threaded',
    'threads',
    # 'processes',
])
def scheduler(request):
    return request.param

@pytest.fixture(params=[
    n.rtoa.name,
    n.lat.name,
    n.lon.name,
    n.vza.name,
    n.sza.name,
    n.raa.name,
    n.flags.name,
])
def param(request):
    return request.param

@pytest.fixture(params=[
    (20, 20),                                   # two ints
    (slice(120, 130), slice(122, 135)),         # two slices
    (20, slice(120, 130)),                      # int and slice
    (slice(510, 530, 3), slice(-25, -20, 3)),   # two slices with steps
])
def indices(request):
    return request.param


def test_main(ds, radiometry='reflectance', angle_data=False):
    assert radiometry in ['radiance','reflectance'], \
    f"radiometry arg should be set to radiance or reflectance, not {radiometry}"

    # test chunks consistency
    ds.chunks

    # check dimensions
    assert n.rows.name in ds.dims
    assert n.columns.name in ds.dims
    # if radiometry: 
    #     if n.rtoa.name in ds: assert ds[n.rtoa.name].dims == n.dim3 
    #     if n.bt.name in ds: assert ds[n.bt.name].dims == n.dim3_tir
    #     if n.rtoa.name not in ds and n.bt.name not in ds: 
    #         raise ValueError(f'{n.rtoa.name} or {n.bt.name} is missing')
    # else: 
    #     if n.ltoa_ir.name in ds: assert ds[n.ltoa_ir.name].dims == n.dim3_tir
    #     elif n.ltoa.name in ds  : assert ds[n.ltoa.name].dims == n.dim3
    #     if n.ltoa_ir.name not in ds and n.ltoa.name not in ds:
    #         raise ValueError(f'{n.ltoa.name} or {n.ltoa_ir.name} is missing')

    # # spectral data
    # # either just provide wav (per-band central wavelength)
    # # or per-pixel wavelength + central wavelength
    # assert n.wav.name in ds
    # if ds.wav.ndim == 3:
    #     assert n.cwav.name in ds
    #     assert ds.cwav.ndim == 1
    # else:
    #     assert (ds.wav.ndim == 1)


    # check that attributes exist and are not empty
    assert ds.datetime
    datetime(ds)
    assert ds.platform
    assert ds.sensor
    assert ds.product_name
    # assert ds.resolution
    # assert ds.input_directory

    # test datasets
    assert n.flags.name in ds
    assert ds[n.flags.name].dtype == n.flags_dtype

    # TODO: test footprint
    assert n.lat.name in ds
    assert n.lon.name in ds
    
    # test angle data
    if angle_data:
        assert n.vaa.name in ds
        assert n.vza.name in ds
        assert n.saa.name in ds
        assert n.sza.name in ds


def test_read(ds, param, indices, scheduler):
    idx1, idx2 = indices
    assert param in ds

    with dask.config.set(scheduler=scheduler):
        # # v = da.compute()
        # expected_dtype = np.dtype(n.expected_dtypes[param])

        # res = ds[param].sel({n.rows:idx1, n.columns:idx2}).compute()
        # assert ds[param].dtype == expected_dtype,\
        #     f'Dtype error: expected {expected_dtype}, found {ds[param].dtype}'
        # assert res.dtype == expected_dtype,\
        #     f'Dtype error: expected {expected_dtype}, found {res.dtype} (after compute)'
        
        # # for the "stepped" indices, check that result is consistent with "non-stepped"
        # # (also with an offset)
        # if (isinstance(idx1, slice) and isinstance(idx2, slice) and idx1.step and idx2.step):
        #     A = ds[param].sel({n.rows:idx1, n.columns:idx2}).compute()
        #     B = ds[param].sel({
        #             n.rows:slice(idx1.start-1, idx1.stop),
        #             n.columns:slice(idx2.start-1, idx2.stop),
        #         }).compute()[..., 1::idx1.step, 1::idx2.step]
        #     np.testing.assert_allclose(A, B)
        pass


def test_subset(ds):
    sub = ds.isel({
        n.rows.name:slice(300, 400),
        n.columns.name:slice(500, 570)})

    with tempfile.TemporaryDirectory() as tmpdir,\
            dask.config.set(scheduler='single-threaded'):
        target = os.path.join(tmpdir, 'test.nc')
        to_netcdf(sub, target)
