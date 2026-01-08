#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

from tempfile import TemporaryDirectory
from .conftest import savefig
from datetime import datetime
from pathlib import Path

import numpy as np
import dask
import pytest

from core import log
from core.files import to_netcdf
from core.monitor import Chrono
from core.geo import n


@pytest.fixture(params=[
    'single-threaded',
    'threads',
    # 'processes',
])
def scheduler(request):
    return request.param

@pytest.fixture(params=[
    (20, 20),                                   # two ints
    (slice(120, 130), slice(122, 135)),         # two slices
    (20, slice(120, 130)),                      # int and slice
    (slice(510, 530, 3), slice(-25, -20, 3)),   # two slices with steps
])
def indices(request):
    return request.param


def test_main(ds, angle_data):

    # test chunks consistency
    ds.chunks

    # check spatial dimensions
    log.check(n.rows.name in ds.dims, f'Strange dimensions, got {list(ds.dims)}')
    log.check(n.columns.name in ds.dims, f'Strange dimensions, got {list(ds.dims)}')
    
    # Check variables
    check_vars(ds)
    
    # Check band dimensions have correct coordinates
    for bands in [n.bands.name, n.bands_nvis.name, n.bands_ir.name]:
        if bands not in ds.dims: continue
        log.check(ds[bands].dtype == int, f'Coordinates of {bands} should be integers')
        if bands == n.bands.name: continue
        log.check(all(b.data in ds[n.bands.name] for b in ds[bands]))
    
    # check spectral dimensions
    if n.ltoa.name in ds:
        log.check(n.bands.name in ds[n.ltoa.name].dims, 
        f'{n.ltoa.name} variable should have a dimension called {n.bands.name}')
        
    if n.rtoa.name in ds:
        log.check(n.bands_nvis.name in ds[n.rtoa.name].dims, 
        f'{n.rtoa.name} variable should have a dimension called {n.bands_nvis.name}')
        
    if n.bt.name in ds:
        log.check(n.bands_ir.name in ds[n.bt.name].dims, 
        f'{n.bt.name} variable should have a dimension called {n.bands_ir.name}')
    
    # Check that following variables exist
    for name, longname in [
            (n.cwav.name, n.cwav.desc),
            (n.bnames.name, n.bnames.desc),
            (n.lat.name, n.lat.desc),
            (n.lon.name, n.lon.desc),
        ]:
        log.check(name in ds, 
        f'Dataset should contain a variable for {longname} named {name}')

    # check that attributes exist and its type
    for name, longname, types in [
            (n.datetime.name, n.datetime.desc, str),
            (n.platform.name, n.platform.desc, str),
            (n.sensor.name, n.sensor.desc, str),            
            (n.resolution.name, n.resolution.desc, int),
            (n.product_name.name, n.product_name.desc, str), 
            (n.input_directory.name, n.input_directory.desc, str),
        ]:
        log.check(name in ds.attrs, f'Dataset should contain a attribute named {name}')
        log.check(isinstance(ds.attrs[name], types), f'Wrong type for attribute {name}')
    
    # Check that datetime is in isoformat
    try: datetime.fromisoformat(ds.attrs[n.datetime.name])
    except: log.error('Issue to read the format of datetime. '
                      f'Should be isoformat, got {ds.attrs[n.datetime.name]}')

    # Check that footprints are 2-dimensional 
    dims2 = (n.rows.name, n.columns.name)
    log.check(ds[n.lat.name].dims == dims2, 'Latitude should be 2-dimensional')
    log.check(ds[n.lon.name].dims == dims2, 'Longitude should be 2-dimensional')
    
    # test angle data
    if angle_data:
        
        for name, longname in [
                (n.vaa.name, n.vaa.desc),
                (n.vza.name, n.vza.desc),
                (n.saa.name, n.saa.desc),
                (n.sza.name, n.sza.desc),
            ]:
            log.check(name in ds, 
            f'Dataset should contain a variable for {longname} named {name}')

def test_plot(request, ds, index_band):
    if n.ltoa.name in ds: bands, var = n.bands.name, n.ltoa.name
    else: bands, var = n.bands_nvis.name, n.rtoa.name
    ds[var].isel({bands:index_band}).plot.imshow()
    savefig(request)

def test_execution_time(reader_fn, params: dict):
    with Chrono('Reading operation', unit='s'): reader_fn(**params)

def test_subset(ds):
    sub = ds.isel({
        n.rows.name:slice(300, 400),
        n.columns.name:slice(500, 570)})

    with TemporaryDirectory() as tmpdir,\
            dask.config.set(scheduler='single-threaded'):
        target = Path(tmpdir)/'test.nc'
        to_netcdf(sub, target, clean_attrs=True)

def compare_version(v2, v1):
    
    # Check dimensions
    for d in v1.dims:
        assert d in v2.dims, f'[Different dimensions] {d} not in v2 : {v2.dims}'
    
    # Check coordinates
    for c, val in v1.coords.items():
        assert val.data in v2.coords[c].data, \
        f'[Different Coordinate] {c} has different values in v2 : {v2.coords[c].data} | v1 : {val.data}'
        
    # Check variables
    for v, val in v1.variables.items():
        assert v in v2.variables, f'[Different variables] {v} not in v2 : {list(v2.variables)}'
        assert val.sizes == v2.variables[v].sizes
    
    # Check attributes
    for key, val in v1.attrs.items():
        if key == 'git_commit': continue
        assert key in v2.attrs
        if isinstance(val, np.ndarray): 
            if any(val != v2.attrs[key]):
                log.warning(f'[Different values] For {key}, should be {val} but '
                            f'got {v2.attrs[key]}')            
        else:
            if val != v2.attrs[key]:
                log.warning(f'[Different values] For {key}, should be {val} but '
                            f'got {v2.attrs[key]}')

def test_lazy_load(ds):
    
    specifics = [n.bnames.name, n.cwav.name]
    
    # Check that variables are lazy-loaded except tie_points and some specific variables
    for key, variable in ds.variables.items():
        if 'tie' in key: continue
        if key not in ds.data_vars or key in specifics: continue
        assert variable.chunks is not None, f'{key} is not lazy-loaded'


def check_vars(ds):
    
    # Check that at least one spectral variable is present
    log.check((n.ltoa.name in ds) or (n.rtoa.name in ds) or (n.bt.name in ds), 
    'No acquisitions stored in Dataset. Output Dataset should contain at least '
    f'{n.ltoa.name} or {n.rtoa.name}  or {n.bt.name}')
    
    # Check that each variable has a unit attribute
    for name in [n.ltoa.name, n.rtoa.name, n.bt.name]:
        if name not in ds: continue
        log.check(hasattr(ds[name],'unit'), f'{name} does not have a unit field')
    
    # Check spectral variables values