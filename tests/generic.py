#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Generic tests implementation
"""

from tempfile import TemporaryDirectory

from matplotlib import pyplot as plt
from .conftest import savefig
from eoread.utils import xrimshow, downsample
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core import log
from core.tools import drop_unused_dims
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
    if str(n.rows) not in ds.dims:
        raise ValueError(f'Strange dimensions, got {list(ds.dims)}')
    if str(n.columns) not in ds.dims:
        raise ValueError(f'Strange dimensions, got {list(ds.dims)}')
    
    # Check variables
    check_vars(ds)
    
    # Check band dimensions have correct coordinates
    str_types = [str] + [f'<U{i}' for i in range(1,33)]
    if ds[str(n.bands)].dtype not in str_types:
        raise ValueError(f'Coordinates of {n.bands} should be bands names')
    if str(n.bgroup) not in ds: 
        raise ValueError(f'Coordinate {n.bgroup} is missing')
    if ds[str(n.bgroup)].dtype not in str_types:
        raise ValueError(f'Coordinates of {n.bands} should be band groups')
    if len(ds[str(n.bgroup)].dims) != 1 or ds[str(n.bgroup)].dims[0] != str(n.bands):
        raise ValueError(f'{n.bgroup} should be indexed by {n.bands}')
    
    # check spectral dimensions
    if str(n.ltoa) in ds and str(n.bands) not in ds[str(n.ltoa)].dims:
        raise ValueError(f'{str(n.ltoa)} variable should have a dimension called {str(n.bands)}')
        
    if str(n.rtoa) in ds and str(n.bands) not in ds[str(n.rtoa)].dims:
        raise ValueError(f'{str(n.rtoa)} variable should have a dimension called {str(n.bands)}')
        
    if str(n.bt) in ds and str(n.bands) not in ds[str(n.bt)].dims:
        raise ValueError(f'{str(n.bt)} variable should have a dimension called {str(n.bands)}')
    
    # Check that following variables exist
    for name, longname in [
            (str(n.cwav), n.cwav.desc),
            (str(n.lat), n.lat.desc),
            (str(n.lon), n.lon.desc),
        ]:
        if name not in ds: 
            raise ValueError(f'Dataset should contain a variable for {longname} named {name}')

    # check that attributes exist and its type
    for name, longname, types in [
            (str(n.datetime), n.datetime.desc, str),
            (str(n.platform), n.platform.desc, str),
            (str(n.sensor), n.sensor.desc, str),            
            (str(n.resolution), n.resolution.desc, int|None),
            (str(n.product_name), n.product_name.desc, str), 
            (str(n.input_directory), n.input_directory.desc, str),
        ]:
        if name not in ds.attrs:
            raise ValueError(f'Dataset should contain a attribute named {name}')
        if not isinstance(ds.attrs[name], types):
            raise ValueError(f'Wrong type for attribute {name}')
    
    # Check that datetime is in isoformat
    try: 
        datetime.fromisoformat(ds.attrs[str(n.datetime)])
    except: 
        log.error('Issue to read the format of datetime. '
                  f'Should be isoformat, got {ds.attrs[str(n.datetime)]}')

    # Check that footprints are 2-dimensional 
    dims2 = (str(n.rows), str(n.columns))
    if ds[str(n.lat)].dims != dims2:
        raise ValueError('Latitude should be 2-dimensional')
    if ds[str(n.lon)].dims != dims2:
        raise ValueError('Longitude should be 2-dimensional')
    
    # test angle data
    if angle_data:
        
        for name, longname in [
                (str(n.vaa), n.vaa.desc),
                (str(n.vza), n.vza.desc),
                (str(n.saa), n.saa.desc),
                (str(n.sza), n.sza.desc),
            ]:
            if name not in ds: 
                raise ValueError(f'Dataset should contain a variable for {longname} named {name}')

def test_plot(request, ds, index_band):
    if str(n.ltoa) in ds: var = str(n.ltoa)
    elif str(n.rtoa) in ds: var = str(n.rtoa)
    elif str(n.bt) in ds: var = str(n.bt)
    else: raise ValueError
    
    ds[var].isel({str(n.bands): index_band}).plot.imshow()
    savefig(request)


def plot(
    request, l1: xr.Dataset, band_nir, poi: dict | None = None, yincrease: bool = True
):
    """
    Plot typical level 1 parameters to give an overview of the product
    """
    for da in [
        l1[n.sza],
        l1[n.vza],
        l1[n.lat],
        l1[n.lon],
        l1[n.rtoa].sel(bands=band_nir),
    ]:
        data = da.compute()
        print(data.name, data.attrs)
        
        _, ax, _ = xrimshow(downsample(data), yincrease=yincrease)

        # Show point of interest
        if poi is not None:
            ax.plot(
                poi[data.dims[1]],
                poi[data.dims[0]],
                "r+",
                markersize=7,
                markeredgewidth=1,
            )
            print(data.name, ':', data.isel(poi).values.item())

        savefig(request)
    
    # Plot spectrum over the point of interest
    if poi is not None:
        l1c = l1.sel(poi).compute()
        for dim in ['wav', 'cwav']:
            if (dim in l1c) and (l1c[dim].ndim == 1):
                l1c = l1c.assign_coords(bands=l1c[dim])
            
        l1c[n.rtoa].plot(figsize=(5, 3), marker='+')
        plt.grid(True)
        plt.title('rho_toa')
        plt.tight_layout()
        plt.axis(ymin=0.)
        savefig(request)

        with xr.set_options(display_max_rows=999):
            print(l1c)
        

def test_execution_time(reader_fn, params: dict):
    with Chrono('Reading operation', unit='s'): reader_fn(**params)

def test_subset(ds):
    sub = ds.isel({
        str(n.rows): slice(300, 400),
        str(n.columns): slice(500, 570)}
    )

    with TemporaryDirectory() as tmpdir,\
            dask.config.set(scheduler='single-threaded'):
        target = Path(tmpdir)/'test.nc'
        to_netcdf(sub, target, clean_attrs=True)

def compare_version(v2, v1):
    
    v1 = drop_unused_dims(v1)
    
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
    if str(n.ltoa) not in ds and str(n.rtoa) not in ds and str(n.bt) not in ds: 
        raise ValueError(
            'No acquisitions stored in Dataset. Output Dataset should '
            f'contain at least {str(n.ltoa)} or {str(n.rtoa)} or {str(n.bt)}'
        )
    
    # Check that each variable has a unit attribute
    for name in [str(n.ltoa), str(n.rtoa), str(n.bt)]:
        if name not in ds: 
            continue
        if not hasattr(ds[name],'unit'):
            raise ValueError(f'{name} does not have a unit field')
    
    # Check spectral variables values