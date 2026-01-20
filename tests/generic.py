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
    log.check(str(n.rows) in ds.dims, f'Strange dimensions, got {list(ds.dims)}')
    log.check(str(n.columns) in ds.dims, f'Strange dimensions, got {list(ds.dims)}')
    
    # Check variables
    check_vars(ds)
    
    # Check band dimensions have correct coordinates
    for bands in [str(n.bands), str(n.bands_nvis), str(n.bands_ir)]:
        if bands not in ds.dims: continue
        log.check(ds[bands].dtype == int, f'Coordinates of {bands} should be integers')
        if bands == str(n.bands): continue
        log.check(all(b.data in ds[str(n.bands)] for b in ds[bands]))
    
    # check spectral dimensions
    if str(n.ltoa) in ds:
        log.check(str(n.bands) in ds[str(n.ltoa)].dims, 
        f'{str(n.ltoa)} variable should have a dimension called {str(n.bands)}')
        
    if str(n.rtoa) in ds:
        log.check(str(n.bands_nvis) in ds[str(n.rtoa)].dims, 
        f'{str(n.rtoa)} variable should have a dimension called {str(n.bands_nvis)}')
        
    if str(n.bt) in ds:
        log.check(str(n.bands_ir) in ds[str(n.bt)].dims, 
        f'{str(n.bt)} variable should have a dimension called {str(n.bands_ir)}')
    
    # Check that following variables exist
    for name, longname in [
            (str(n.cwav), n.cwav.desc),
            (str(n.bnames), n.bnames.desc),
            (str(n.lat), n.lat.desc),
            (str(n.lon), n.lon.desc),
        ]:
        log.check(name in ds, 
        f'Dataset should contain a variable for {longname} named {name}')

    # check that attributes exist and its type
    for name, longname, types in [
            (str(n.datetime), n.datetime.desc, str),
            (str(n.platform), n.platform.desc, str),
            (str(n.sensor), n.sensor.desc, str),            
            (str(n.resolution), n.resolution.desc, int),
            (str(n.product_name), n.product_name.desc, str), 
            (str(n.input_directory), n.input_directory.desc, str),
        ]:
        log.check(name in ds.attrs, f'Dataset should contain a attribute named {name}')
        log.check(isinstance(ds.attrs[name], types), f'Wrong type for attribute {name}')
    
    # Check that datetime is in isoformat
    try: datetime.fromisoformat(ds.attrs[str(n.datetime)])
    except: log.error('Issue to read the format of datetime. '
                      f'Should be isoformat, got {ds.attrs[str(n.datetime)]}')

    # Check that footprints are 2-dimensional 
    dims2 = (str(n.rows), str(n.columns))
    log.check(ds[str(n.lat)].dims == dims2, 'Latitude should be 2-dimensional')
    log.check(ds[str(n.lon)].dims == dims2, 'Longitude should be 2-dimensional')
    
    # test angle data
    if angle_data:
        
        for name, longname in [
                (str(n.vaa), n.vaa.desc),
                (str(n.vza), n.vza.desc),
                (str(n.saa), n.saa.desc),
                (str(n.sza), n.sza.desc),
            ]:
            log.check(name in ds, 
            f'Dataset should contain a variable for {longname} named {name}')

def test_plot(request, ds, index_band):
    if str(n.ltoa) in ds: bands, var = str(n.bands), str(n.ltoa)
    else: bands, var = str(n.bands_nvis), str(n.rtoa)
    ds[var].isel({bands:index_band}).plot.imshow()
    savefig(request)

def test_execution_time(reader_fn, params: dict):
    with Chrono('Reading operation', unit='s'): reader_fn(**params)

def test_subset(ds):
    sub = ds.isel({
        str(n.rows):slice(300, 400),
        str(n.columns):slice(500, 570)})

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
    
    specifics = [str(n.bnames), str(n.cwav)]
    
    # Check that variables are lazy-loaded except tie_points and some specific variables
    for key, variable in ds.variables.items():
        if 'tie' in key: continue
        if key not in ds.data_vars or key in specifics: continue
        assert variable.chunks is not None, f'{key} is not lazy-loaded'


def check_vars(ds):
    
    # Check that at least one spectral variable is present
    log.check((str(n.ltoa) in ds) or (str(n.rtoa) in ds) or (str(n.bt) in ds), 
    'No acquisitions stored in Dataset. Output Dataset should contain at least '
    f'{str(n.ltoa)} or {str(n.rtoa)}  or {str(n.bt)}')
    
    # Check that each variable has a unit attribute
    for name in [str(n.ltoa), str(n.rtoa), str(n.bt)]:
        if name not in ds: continue
        log.check(hasattr(ds[name],'unit'), f'{name} does not have a unit field')
    
    # Check spectral variables values