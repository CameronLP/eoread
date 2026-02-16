#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read NASA Level1 files from MODIS, VIIRS, SeaWiFS

Use the L1C approach: L1C files are generated with SeaDAS (l2gen) to
get all radiometric correction

How to install SeaDAS OCSSW (see https://seadas.gsfc.nasa.gov/downloads/)

    ./install_ocssw --install_dir $HOME/ocssw --tag V2022.0 --seadas --modisa --seawifs --viirsn
"""

from pathlib import Path
import xarray as xr
import numpy as np
from datetime import datetime
import subprocess

from core.files.uncompress import uncompress_decorator
from core.network.download import download_url
from core.geo.naming import names as n
from .common import DataArray_from_array
from . import eo


def check_nasa_download(filename):
    '''
    sanity check of file downloaded on NASA earthdata
    check that downloaded file is not HTML
    raise an error if it is not the case (authentication error)
    '''
    errormsg = 'Error authenticating to NASA EarthData for downloading ancillary data. ' \
    'Please provide authentication through .netrc. See more information on ' \
    'https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/how-to-access-urs-gated-data-with-curl-and-wget'
    with open(filename, 'rb') as fp:
        filehead = fp.read(100)
        if filehead.startswith((
            b'<!DOCTYPE html>',
            # may be the case after Oct 2023 when NASA changed the APIs
            b'404 Error',
            b'403 Error')):
            raise RuntimeError(errormsg)


def nasa_download(product, dirname, tmpdir=None, verbose=True, wget_extra=""):
    '''
    Download a product on oceandata.sci.gsfc.nasa.gov

    Example:
        nasa_download('A2005005002500.L1A_LAC.bz2', '/data/')
    
    Note: a full URL can be provided instead of just the product name
    '''
    if product.startswith('https://'):
        url = product
    elif product.startswith('S3'):
        url= f'https://oceandata.sci.gsfc.nasa.gov/sentinel/getfile/{product}.zip'
    else:
        url = f'https://oceandata.sci.gsfc.nasa.gov/getfile/{product}'

    return download_url(
        url,
        dirname,
        verbose=verbose,
        tmpdir=tmpdir,
        wget_opts='-nv --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies ' \
                  '--keep-session-cookies --auth-no-challenge '+wget_extra,
        check_function=check_nasa_download,
        lock_timeout=3600,
        if_exists='skip',
        )


def nasa_download_uncompress(product, dirname) -> Path:
    """
    Download a product on oceandata.sci.gsfc.nasa.gov with
    `nasa_download` and uncompress the result
    """
    return uncompress_decorator()(nasa_download)(product, dirname)


def nasa_search(**kwargs):
    """
    Search for files on oceancolor server

    Args are passed directly to the query

    Example:

    nasa_search(sensor='seawifs',
                sdate='2000-04-17',
                edate='2000-04-17',
                dtype='L1',
                search='*L1A_GAC'):

    See https://oceancolor.gsfc.nasa.gov/data/download_methods/#api
    """
    query = [f'{k}={v}' for k, v in kwargs.items()]
    query += ['addurl=0', 'results_as_file=1']

    query_str = '&'.join(query)
    cmd = f'wget -q --post-data="{query_str}" -O - https://oceandata.sci.gsfc.nasa.gov/api/file_search'
    return subprocess.check_output(cmd, shell=True).decode().split()


def Level1_NASA(filename, chunks=500):
    ds = xr.open_dataset(filename, chunks=chunks)
    # TODO: use xr.open_datatree instead of several xr.open_dataset

    dstart = datetime.strptime(ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S.%fZ")
    dstop = datetime.strptime(ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S.%fZ")
    d = dstart + (dstop - dstart)//2
    ds.attrs[str(n.datetime)] = d.isoformat()
    ds.attrs[str(n.sensor)] = ds.attrs['instrument']
    ds.attrs[str(n.input_directory)] = str(Path(filename).parent)

    sensor_band = xr.open_dataset(filename, group='/sensor_band_parameters', chunks=chunks)
    bands = sensor_band['wavelength'].values[sensor_band.number_of_reflective_bands.values].astype('int32')

    navi = xr.open_dataset(filename, group='navigation_data', chunks=chunks)
    navi = navi.rename_dims({'number_of_lines':str(n.rows), 'pixels_per_line':str(n.columns)})
    ds[str(n.lat)] = navi.latitude
    ds[str(n.lon)] = navi.longitude
    
    geo_data = xr.open_dataset(filename, group='/geophysical_data', chunks=chunks)
    geo_data = geo_data.rename_dims({'number_of_lines':str(n.rows), 'pixels_per_line':str(n.columns)})
    for _n,r,p in [(str(n.rtoa)+f'_{b}', f'rhot_{b}', f'polcor_{b}') for b in bands]:
        try:
            ds[_n] = geo_data[r]/geo_data[p]
        except:
            pass

    for (name, param) in [(str(n.sza), 'solz'),
                          (str(n.vza), 'senz'),
                          (str(n.saa), 'sola'),
                          (str(n.vaa), 'sena'),
                          ]:
        ds[name] = geo_data[param]

    eo.init_geometry(ds)

    ds[str(n.flags)] = xr.zeros_like(ds[str(n.lat)], dtype=n.flags.dtype)
    for flag, flag_list, flag_value in [
        ("LAND", ["LAND"], 1),
        ("L1_INVALID", ["ATMFAIL", "PRODFAIL"], 4),
    ]:
        flag_mask = 0
        for f in flag_list:
            flag_mask += geo_data.l2_flags.flag_masks[geo_data.l2_flags.flag_meanings.split().index(f)]

        eo.raiseflag(
            ds[n.flags],
            flag,
            flag_value,
            DataArray_from_array(
                (geo_data.l2_flags & flag_mask != 0), ds.sza.dims, chunks=chunks
            ),
        )

    ds = eo.merge(ds, dim=str(n.bands))
    return ds
