#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
MERIS level1 reader

l1 = Level1_MERIS('MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
'''


from datetime import datetime
from pathlib import Path
from threading import Lock

import epr
import numpy as np
import pandas as pd
import xarray as xr

from core.geo import n
from core import env, log
from core.tools import merge, drop_unused_dims
from eoread.common import AtIndex, DataArray_from_array
from eoread.utils import filter_metadata
from core.monitor import Chrono


user_guide = 'https://archive.org/details/manualzilla-id-5933919/page/n13/mode/2up'

def Level1_MERIS(filepath: str|Path,
                 dir_smile: str|Path = None,
                 read_auxdata: bool = False, 
                 chunks: int|tuple = 500,
                 metadata_template: list = None,
                 v1_compat: bool = False):
    '''
    Read an MERIS Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances,
    the angles on the full grid, etc.

    Arguments:
        filepath: Path of the MERIS file path (ex: 'MER_FRS_1PNPDE20060822_092058_000001972050_00308_23408_0077.N1')
        dir_smile: Relative path to MERIS per-detector characterization (default: '../auxdata/meris/')
        read_auxdata: Option read auxilary data contained in dir_smile
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    
    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    bname = filepath.name

    # epr api is not thread safe: we have to use a lock for safe file access
    lock = Lock()

    prod = epr.Product(str(filepath))
    ds.attrs['totalwidth'] = prod.get_scene_width()
    ds.attrs['totalheight'] = prod.get_scene_height()
    
    # Read metadata
    log.debug('read metadata')
    metadata = _read_metadata(ds, prod, metadata_template)
    bands_names = [f'b{i+1}' for i in range(metadata['NUM_BANDS'])]
    ds = ds.assign({n.bnames.name: ((n.bands.name), bands_names),
                    n.cwav.name: ((n.bands.name), metadata['BAND_WAVELEN']/1e3)})

    # read all rasters
    log.debug('load raster')
    for name in prod.get_band_names():
        band = prod.get_band(name)
        ds[name] = DataArray_from_array(
            _READ_MERIS(band, lock),
            (n.rows.name, n.columns.name),
            chunks=chunks,
        )
        ds[name].attrs['unit'] = band.unit
        ds[name].attrs['description'] = band.description
        
    # Rename several variables and compile radiance rasters
    log.debug('concatenate ltoa rasters')
    ds = _rename_meris(ds)
    ds = merge(ds, dim=n.bands.name)
    ds = ds.chunk({n.bands.name:1})
    
    log.debug('read auxilary data')
    if dir_smile is None: dir_smile = Path(__file__).parent/'auxdata'/'meris'
    else: dir_smile = Path(dir_smile)
    assert dir_smile.exists(), f'{dir_smile} does not exists'
    
    if bname.startswith('MER_RR'): res = 'rr'
    elif bname.startswith('MER_FR'): res = 'fr'
    else: raise Exception(f'Error, could not identify whether MERIS file is RR or FR ({bname})')

    file_sun_spectral_flux = dir_smile/f'sun_spectral_flux_{res}.txt'
    file_detector_wavelength = dir_smile/f'central_wavelen_{res}.txt'
    F0 = pd.read_csv(file_sun_spectral_flux, dtype='float32', delimiter='\t').to_xarray()
    detector_wavelength = pd.read_csv(file_detector_wavelength, delimiter='\t').to_xarray()

    assert len(F0) == len(ds[n.bands.name]) + 1
    assert len(detector_wavelength) == len(ds[n.bands.name]) + 1
    
    if read_auxdata:
        
        # Compute solar Flux
        valid_mask = (ds.detector_index >= 0).load()
        F0 = F0.sel(index=ds.detector_index.where(valid_mask, 0))
        F0 = merge(F0, dim=n.bands.name, pattern=r'(.+)_band(\d+)')
        ds[n.F0.name] = F0['E0'].where(valid_mask, np.nan)
        
        # Compute wavelengths
        wav = detector_wavelength.sel(index=ds.detector_index.where(valid_mask, 0))
        wav = merge(wav, dim=n.bands.name, pattern=r'(.+)_band(\d+)')
        ds[n.wav.name] = wav['lam'].where(valid_mask, np.nan)

    # Read attributes
    ds.attrs[n.platform.name] = 'ENVISAT'
    ds.attrs[n.sensor.name] = 'MERIS'
    ds.attrs[n.resolution.name] = 300
    ds.attrs[n.product_name.name] = metadata['PRODUCT'].decode()
    ds.attrs[n.input_directory.name] = str(filepath.parent)
    ds.attrs['user_guide'] = user_guide

    # Read date
    dstart = _read_date(metadata['SENSING_START'])
    dstop = _read_date(metadata['SENSING_STOP'])
    d = dstart + (dstop - dstart)//2
    ds.attrs[n.datetime.name] = d.isoformat()
    
    if v1_compat: return _v1_compat(ds, prod, lock, chunks)
    return drop_unused_dims(ds).unify_chunks()


def _rename_meris(ds):
    ds = ds.rename({v: v.replace('radiance', n.ltoa.name) 
                    for v in ds.variables if 'radiance' in v})
    return ds.rename({
        'latitude'    : n.lat.name,
        'longitude'   : n.lon.name,
        'view_zenith' : n.vza.name, 
        'view_azimuth': n.vaa.name,
        'sun_zenith'  : n.sza.name,
        'sun_azimuth' : n.saa.name, 
    })

def _read_metadata(ds, product, template):
    metadata = {}
    for ph in [product.get_mph(), product.get_sph()]:
        metadata.update({
            f.get_name(): f.get_elem(0) if f.get_num_elems() == 1 else f.get_elems()
            for f in ph.fields()})
    
    filter_fn = (lambda x,y: x) if template is None else filter_metadata 
    ds.attrs['metadata'] = filter_fn(metadata, template)   
    
    return metadata

def _read_date(dat):
    dat = dat.decode('utf-8')
    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    for i,m in enumerate(months): dat = dat.replace(f'-{m}-', f'-{i+1:02d}-')
    return datetime.strptime(dat, '%d-%m-%Y %H:%M:%S.%f')


# FIXME : Following classes should be revized
class _READ_MERIS:
    '''
    An array-like to read data from a given MERIS band
    '''
    def __init__(self, band, lock):
        self.width = band.product.get_scene_width()
        self.height = band.product.get_scene_height()
        self.band = band
        self.lock = lock
        self.shape = (self.height, self.width)
        self.dtype = {
            'float': np.float32,
            'short': np.int16,
            'uchar': np.uint8,
        }[epr.data_type_id_to_str(band.data_type)]
        self.ndim = len(self.shape)

    def __getitem__(self, keys):
        assert len(keys) == self.ndim
        start = []
        steps = []
        sizes = []
        sel = []
        for k in keys:
            if isinstance(k, slice):
                st = k.start or 0
                start.append(st)
                steps.append(k.step or 1)
                sizes.append(k.stop - st)
                sel.append(slice(None))
            else:  # Indexing with int
                start.append(0)
                steps.append(1)
                sizes.append(1)
                sel.append(0)

        with self.lock:
            if (sizes[0] > steps[0]) and (sizes[1] > steps[1]):
                r = self.band.read_as_array(
                    yoffset=start[0],
                    xoffset=start[1],
                    height=sizes[0],
                    width=sizes[1],
                    ystep=steps[0],
                    xstep=steps[1],
                )
            else:
                r = self.band.read_as_array(
                    yoffset=start[0],
                    xoffset=start[1],
                    height=sizes[0],
                    width=sizes[1],
                )[::steps[0], ::steps[1]]
        assert r.dtype == self.dtype
        return r[sel[0], sel[1]]

def get_sample(level: int=1, use_cache:bool=True) -> Path:
    """
    Bring a MERIS file path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    sample = Path('/archive2/data/EOREAD_TESTDATA/MERIS/MER_RR__1PRACR20080701_014028_000026402070_00003_33123_0000.N1')
    assert sample.exists()
    return sample

def _v1_compat(ds, prod, lock, chunks):
    
    from core.tools import raiseflag
    from .common import len_slice
    
    class READ_BITMASK:
        '''
        An array-like to read MERIS bitmask
        '''
        def __init__(self, prod, bmexpr, lock):
            self.width = prod.get_scene_width()
            self.height = prod.get_scene_height()
            self.prod = prod
            self.lock = lock
            self.bmexpr = bmexpr
            self.shape = (self.height, self.width)
            self.ndim = len(self.shape)
            self.dtype = np.bool

        def __getitem__(self, keys):
            width = len_slice(keys[1], self.width)
            height = len_slice(keys[0], self.height)
            raster = epr.create_bitmask_raster(
                width, height,
                xstep=keys[1].step or 1,
                ystep=keys[0].step or 1,
                )
            with self.lock:
                self.prod.read_bitmask_raster(
                    self.bmexpr,
                    xoffset=keys[1].start or 0,
                    yoffset=keys[0].start or 0,
                    raster=raster)

            return raster.data
    
    BANDS_MERIS = [412, 443, 490, 510, 560,
                620, 665, 681, 709, 754,
                760, 779, 865, 885, 900]
    
    # Define central wavelength as coordinates for band dimension
    ds = ds.assign_coords(bands=BANDS_MERIS)
    
    # Add other computed variables
    ds['horizontal_wind'] = np.sqrt(ds['zonal_wind']**2 + ds['merid_wind']**2)
    ds['total_column_ozone'] = ds['ozone']
    ds['sea_level_pressure'] = ds['atm_press']

    # Flags
    ds['flags'] = xr.zeros_like(ds[n.lat.name], dtype='uint8')
    for (flag, val, bmexpr) in [
            ('LAND', 1, 'l1_flags.LAND_OCEAN'),
            ('L1_INVALID', 4, '(l1_flags.INVALID) OR (l1_flags.SUSPECT) OR (l1_flags.COSMETIC)'),
        ]:
        raiseflag(
            ds['flags'],
            flag,
            val,
            DataArray_from_array(
                READ_BITMASK(prod, bmexpr, lock),
                ('y','x'),
                chunks=chunks,
            ),
        )
    
    # Level up metadata in attribute dictionary 
    ds.attrs.update(ds.attrs['metadata'])
    
    return ds