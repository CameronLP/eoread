#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
ERA5 Ancillary data provider
'''

import argparse
import warnings

from core import env
from core.tools import wrap
from core.files.fileutils import filegen, mdir
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

import xarray as xr
import cdsapi

from .common import floor_dt, ceil_dt

warnings.warn("This module will be deprecated. Please use HARP instead.")

def _expver_to_int(v):
    # expver comes back as a plain int (1/5) from the legacy CDS client,
    # or as a zero-padded 4-char experiment-version string ("0001"/"0005")
    # from the newer ecmwf-datastores client - normalize both to int.
    return int(v)


def resolve_expver(ds, allow_preliminary=True):
    '''
    Resolve/collapse ERA5's `expver` field, however it's represented.

    CDS's unified ERA5 dataset serves preliminary "ERA5T" data for the most
    recent ~5 days-3 months before final ERA5 is published (expver=1/"0001":
    final ERA5, expver=5/"0005": preliminary ERA5T). A request spanning both
    regimes comes back with `expver` as a genuine dimension with both
    values; a request that's cleanly on one side of the boundary still
    carries `expver` as a plain scalar coordinate holding whichever single
    value applies - it is NOT safe to assume that scalar case means
    'final' (confirmed 2026-09-15: a ~3-week-old date came back as scalar
    expver='0005', i.e. still preliminary - ERA5's actual publication lag
    can exceed the nominal ~5 days).

    Returns (ds_resolved, source) where source is 'final', 'preliminary',
    or 'unknown(expver=<value>)' for an unrecognized code. Raises if only
    preliminary data is available and allow_preliminary=False, or if
    neither expver value has data.
    '''
    if 'expver' not in ds.variables:
        return ds, 'final'

    def _all_nan(d):
        return all(bool(np.all(np.isnan(d[v].values))) for v in d.data_vars)

    if 'expver' not in ds.dims:
        # Single value already selected by CDS - just report it.
        expver_value = _expver_to_int(ds.expver.values)
        if expver_value == 5 and not allow_preliminary:
            raise Exception(
                'Only preliminary ERA5T data is available for this date, '
                'and allow_preliminary=False')
        source = {1: 'final', 5: 'preliminary'}.get(
            expver_value, f'unknown(expver={expver_value})')
        return ds, source

    expver_values = [_expver_to_int(v) for v in ds.expver.values]

    if 1 in expver_values:
        ds_final = ds.isel(expver=expver_values.index(1))
        if not _all_nan(ds_final):
            return ds_final, 'final'

    if 5 in expver_values:
        if not allow_preliminary:
            raise Exception(
                'Only preliminary ERA5T data is available for this date, '
                'and allow_preliminary=False')
        ds_prelim = ds.isel(expver=expver_values.index(5))
        if not _all_nan(ds_prelim):
            return ds_prelim, 'preliminary'

    raise Exception('No valid ERA5/ERA5T data found (expver={})'.format(
        list(expver_values)))


def open_ERA5(filename, allow_preliminary=True):
    '''
    Open an ERA5 file and format it for consistency
    with the other ancillary data sources
    '''
    ds = xr.open_dataset(filename, chunks={})
    # The newer ecmwf-datastores CDS client names the time dimension
    # "valid_time" instead of the legacy "time" - ERA5.get()'s
    # xr.concat(dim='time')/.interp(time=dt) below need "time" to exist as
    # an actual (datetime-indexed) dimension, not just have one silently
    # fabricated as a plain integer range because nothing matched.
    if 'time' not in ds.dims and 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    ds, source = resolve_expver(ds, allow_preliminary=allow_preliminary)
    ds["horizontal_wind"] = np.sqrt(ds.u10**2 + ds.v10**2)
    # "sp" (surface_pressure) and "msl" (mean_sea_level_pressure) are
    # different physical quantities - sp is pressure at actual terrain
    # elevation, msl is reduced to sea level. This used to rename "sp" to
    # "sea_level_pressure" (marked "FIXME: SP/SLP" in this same line,
    # unfixed since at least before this module was deprecated) - close
    # enough over open ocean that most captures didn't notice, but wrong
    # for any capture with land/elevated terrain in the scene.
    #
    # Deliberately NOT exposing "surface_pressure" here (sp is no longer
    # requested at all - see ERA5.__init__'s variables list): eotools.
    # rayleigh.calc_odr prefers "surface_pressure" over "sea_level_pressure"
    # whenever both are present in the ancillary dataset, feeding it into
    # column_number_density's pressure="surface" branch with no altitude
    # reduction. NASA's ancillary provider (eoread.ancillary_nasa) never
    # supplies "surface_pressure" at all, so that branch has never
    # actually been exercised/validated against the Rayleigh LUT's valid
    # range before - confirmed 2026-09-15: supplying it from ERA5 pushed
    # a real HYPSO capture's Rayleigh optical depth out of the LUT's
    # 0.0-0.4 bounds where NASA ancillary never has. Only exposing
    # "sea_level_pressure" (now correctly sourced from msl, not sp) keeps
    # ERA5 on the same, already-validated code path NASA ancillary uses.
    ds = ds.rename({
        "msl": "sea_level_pressure",
        "tco3": "total_column_ozone",
    }).squeeze()
    ds.attrs['ancillary_source'] = source
    return wrap(ds, 'longitude', -180, 180)


class ERA5:
    """ Ancillary data provider using ERA5
    https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5

    Parameters
    ----------
    directory : _type_, optional
        base directory for storing the ERA5 files, by default None
    pattern : str, optional
        pattern for storing the ERA5 files in NetCDF format, by default '%Y/%m/%d/era5_%Y%m%d_%H%M%S.nc'
    time_resolution : timedelta, optional
        time resolution, by default timedelta(hours=1)
    offline : bool, optional
        Offline mode (reluy only on existing files, avoid downloading), by default False
    variables : list, optional
        List of required variables, by default [ '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_column_ozone', 'total_column_water_vapour', ]
    verbose : bool, optional
        Verbose mode, by default False

    """
    def __init__(self,
                 directory=None,
                 pattern='%Y/%m/%d/era5_%Y%m%d_%H%M%S.nc',
                 time_resolution=timedelta(hours=1),
                 offline=False,
                 variables=[
                     '10m_u_component_of_wind',
                     '10m_v_component_of_wind',
                     'mean_sea_level_pressure',
                     'total_column_ozone',
                     'total_column_water_vapour',
                 ],
                 verbose=False,
                 allow_preliminary=True,
                 ):
        if directory is None:
            self.directory = mdir(env.getdir('DIR_ANCILLARY')/'ERA5')
        else:
            self.directory = Path(directory)

        self.pattern = pattern
        self.time_resolution = time_resolution
        self.client = None
        self.offline = offline
        self.verbose = verbose
        # Whether to accept CDS's preliminary "ERA5T" data for a date whose
        # final ERA5 isn't published yet (~5 days-3 months lag) - see
        # resolve_expver(). Mirrors polymer.ancillary_era5.Ancillary_ERA5's
        # own allow_preliminary, for the interface main_v5's pipeline
        # actually uses (ApplyAncillary's ancillary_provider.get(dt)).
        self.allow_preliminary = allow_preliminary

        self.variables = list(variables)

        if not self.directory.exists():
            raise Exception(
                f'Directory "{self.directory}" does not exist. '
                'Please create it for hosting ERA5 files.')


    def get(self, dt):
        """
        Download and initialize ERA5 (interpolated) product for a given date

        dt: datetime
        """
        delta = self.time_resolution

        # search the bracketing dates
        (d0, d1) = (floor_dt(dt, delta), ceil_dt(dt, delta))

        dates = [d0 + i*delta for i in range((d1-d0)//delta + 1)]

        concatenated = xr.concat([self.download(d) for d in dates], dim='time')

        interpolated = concatenated.interp(time=dt)

        return interpolated


    # if_exists='skip' (not filegen's own default 'error'): this decorator
    # IS the on-disk cache-hit check (see skip() in core.files.fileutils) -
    # download() calls this unconditionally on every get(), relying on it
    # to no-op when `target` is already cached. With the default 'error',
    # any second call for a date whose file was already downloaded (e.g. a
    # re-run after an unrelated failure, or another capture landing in the
    # same bracketing hour) raised FileExistsError instead of reusing the
    # cache (confirmed 2026-09-15).
    @filegen(1, if_exists='skip')
    def download_file(self, target, dt):
        if self.client is None:
            self.client = cdsapi.Client()

        print(f'Downloading {target}...')
        self.client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': self.variables,
                'year':[f'{dt.year}'],
                'month':[f'{dt.month:02}'],
                'day':[f'{dt.day:02}'],
                'time': f'{dt.hour:02}:00',
                'format':'netcdf'
            },
            target)

    def download(self, dt):
        """
        Download ERA5 at a given time `dt` and returns the corresponding dataset

        Args:
        -----
        dt: datetime
        """
        assert dt.minute == 0
        assert dt.second == 0

        target = self.directory/dt.strftime(self.pattern)
        self.download_file(target, dt=dt)

        return open_ERA5(target, allow_preliminary=self.allow_preliminary)


def parse_date(dstring):
    return datetime.strptime(dstring, '%Y-%m-%d')


if __name__ == "__main__":
    # command line mode: download all ERA5 files
    # for a given time range d0 to d1
    parser = argparse.ArgumentParser(
        description='Download all ERA5 files for a given time range: `python -m eoread.era5...`')
    parser.add_argument('d0', type=parse_date,
                        help='start date (YYYY-MM-DD)')
    parser.add_argument('d1', type=parse_date,
                        help='stop date (YYYY-MM-DD)')
    parser.add_argument('--time_resolution', type=int,
                        default=1, help='time resolution in hours')
    args = parser.parse_args()

    res = ERA5().get((args.d0, args.d1))
    print(res)