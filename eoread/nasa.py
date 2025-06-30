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

from core.geo import n
from .common import DataArray_from_array
from . import eo

def Level1_NASA(filename, chunks=500):
    ds = xr.open_dataset(filename, chunks=chunks)

    dstart = datetime.strptime(ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S.%fZ")
    dstop = datetime.strptime(ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S.%fZ")
    d = dstart + (dstop - dstart)//2
    ds.attrs[n.datetime.name] = d.isoformat()
    ds.attrs[n.sensor.name] = ds.attrs['instrument']
    ds.attrs[n.input_directory.name] = str(Path(filename).parent)

    sensor_band = xr.open_dataset(filename, group='/sensor_band_parameters', chunks=chunks)
    bands = sensor_band['wavelength'].values[sensor_band.number_of_reflective_bands.values].astype('int32')
    ds[n.wav.name] = np.array(bands, dtype='float32')

    navi = xr.open_dataset(filename, group='navigation_data', chunks=chunks)
    navi = navi.rename_dims({'number_of_lines':n.rows.name, 'pixel_control_points':n.columns.name})
    ds[n.lat.name] = DataArray_from_array(navi.latitude.values.astype('float32'), naming.dim2, chunks=chunks)
    ds[n.lon.name] = DataArray_from_array(navi.longitude.values.astype('float32'), naming.dim2, chunks=chunks)
    
    geo_data = xr.open_dataset(filename, group='/geophysical_data', chunks=chunks)
    geo_data = geo_data.rename_dims({'number_of_lines':n.rows.name, 'pixels_per_line':n.columns.name})
    for n,r,p in [(n.rtoa.name+f'_{b}', f'rhot_{b}', f'polcor_{b}') for b in bands]:
        try:
            ds[n] = geo_data[r]/geo_data[p]
        except:
            pass

    for (name, param) in [(n.sza.name, 'solz'),
                          (n.vza.name, 'senz'),
                          (n.saa.name, 'sola'),
                          (n.vaa.name, 'sena'),
                          ]:
        ds[name] = geo_data[param]

    eo.init_geometry(ds)

    ds[naming.flags] = xr.zeros_like(ds[naming.lat], dtype=naming.flags_dtype)
    for (flag, flag_list) in [('LAND',['LAND']), ('L1_INVALID',['ATMFAIL','PRODFAIL'])]:
        flag_value = 0
        for f in flag_list:
            flag_value += geo_data.l2_flags.flag_masks[geo_data.l2_flags.flag_meanings.split().index(f)]

        eo.raiseflag(ds[naming.flags],flag, flags[flag], DataArray_from_array((geo_data.l2_flags&flag_value!=0), naming.dim2, chunks=chunks))

    ds = eo.merge(ds, dim=n.bands.name)
    return ds
