#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import xarray as xr
import dask.array as da

from os import system
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from eoread.utils import filter_metadata, open_raster

from core import log, env
from core.interpolate import interp, Linear
from core.tools import merge, drop_unused_dims
from core.table import read_xml
from core.geo import n, convert_latlon_2D


# Central wavelengths aren't described in metadata. Thus, they are hard-coded
cwvl = [442.96,482.04,561.41,654.59,864.67,1608.86,2200.73,1373.43,10895,12050,]

user_guide = 'https://greenpolicy360.net/images/Landsat8DataUsersHandbook.pdf'

def Level1_OLI(
        dirname: Union[str, Path],
        l9_angles: Union[str, Path, None] = None,
        chunks: Union[int, tuple] = 500,
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Landsat-8 or Landsat-9 OLI Level1 product as an xarray.Dataset.
    
    OLI (Operational Land Imager) provides 9 spectral bands from coastal aerosol
    to SWIR with 30m resolution, plus a 15m panchromatic band.

    Args:
        dirname: Path to the Landsat OLI directory
                (Example: 'LC09_L1TP_014034_20220618_20230411_02_T1/')
        l9_angles: Path to l9_angles executable for generating angle files when missing.
                  The program generates sensor and solar angles with:
                  `l9_angles LC0*_ANG.txt BOTH 1 -b 1`
                  
                  Available at: https://www.usgs.gov/land-resources/nli/landsat/
                  solar-illumination-and-sensor-viewing-angle-coefficient-files
                  
                  Can be compiled with:
                  ```
                  wget https://landsat.usgs.gov/sites/default/files/documents/L9_ANGLES_2_7_0.tgz
                  tar xzf L9_ANGLES_2_7_0.tgz && cd l9_angles && make
                  ```
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        v1_compat: If True, formats output to match version 1 structure
        
    Example:
        >>> ds = Level1_OLI('LC09_L1TP_014034_20220618_20230411_02_T1/')
    """
    
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Directory does not exists'

    # Read metadata
    if verbose: log.debug('read metadata')
    metadata = _read_metadata(ds, dirname, metadata_template)
    if isinstance(chunks, int): chunks = [chunks]*2

    # get datetime
    d = metadata['IMAGE_ATTRIBUTES']['DATE_ACQUIRED']
    t = metadata['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME']
    ds.attrs[str(n.datetime)] = d+'T'+t
    
    # Reading different rasters
    if verbose: log.debug('read geometric angles')
    _read_geometry(ds, dirname, l9_angles, chunks)
    if verbose: log.debug('read TOA rasters')
    ds = _read_radiometry(ds, dirname, chunks)
    if verbose: log.debug('read masks')
    _read_masks(ds, dirname, chunks)
    _read_coordinates(ds, chunks)

    # other attributes
    if verbose: log.debug('add important attributes')
    ds.attrs[str(n.platform)] = metadata['IMAGE_ATTRIBUTES']['SPACECRAFT_ID']
    ds.attrs[str(n.sensor)] = metadata['IMAGE_ATTRIBUTES']['SENSOR_ID']
    ds.attrs[str(n.product_name)] = metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']
    ds.attrs[str(n.input_directory)] = str(dirname.parent)
    ds.attrs[str(n.resolution)] = 30
    ds.attrs['user_guide'] = user_guide
    
    # Manage dimensions
    ds = ds.assign({str(n.cwav):((str(n.bands)), cwvl)})    
    ds = ds.rename({'y': str(n.rows), 'x': str(n.columns)})   
    ds = drop_unused_dims(ds).unify_chunks()
    ds = ds.set_coords(n.bgroup)
    
    if v1_compat: return _v1_compat(ds)
    else: return ds


def _read_metadata(ds: xr.Dataset, dirname: Path, template: Union[list, None]) -> dict:
    """Read and parse MTL XML metadata file."""
    filter_fn = (lambda x,y: x) if template is None else filter_metadata
    files_mtl = list(dirname.glob('LC*_MTL.xml'))
    assert len(files_mtl) == 1, 'XML file not found'
    data_mtl = read_xml(files_mtl[0])
    ds.attrs['metadata'] = filter_fn(data_mtl, template)
    return data_mtl


def _read_coordinates(ds: xr.Dataset, chunks: list) -> None:
    """Compute latitude and longitude arrays from corner coordinates."""
    # Compute tie points
    points = ds.metadata['PROJECTION_ATTRIBUTES']
    lat = xr.DataArray([
        [points['CORNER_UL_LAT_PRODUCT'],points['CORNER_UR_LAT_PRODUCT']],
        [points['CORNER_LL_LAT_PRODUCT'],points['CORNER_LR_LAT_PRODUCT']],
    ])
    lon = xr.DataArray([
        [points['CORNER_UL_LON_PRODUCT'],points['CORNER_UR_LON_PRODUCT']],
        [points['CORNER_LL_LON_PRODUCT'],points['CORNER_LR_LON_PRODUCT']],
    ])
    
    # Compute latlon arrays
    y, x = convert_latlon_2D(da.linspace(0,1,len(ds[str(n.rows)])),
                          da.linspace(0,1,len(ds[str(n.columns)])))
    x = xr.DataArray(x, dims=(str(n.rows), str(n.columns))).chunk(chunks)
    y = xr.DataArray(y, dims=(str(n.rows), str(n.columns))).chunk(chunks)
    ds[str(n.lat)] = interp(lat, dim_0=Linear(y), dim_1=Linear(x))
    ds[str(n.lon)] = interp(lon, dim_0=Linear(y), dim_1=Linear(x))

    ds.attrs['totalheight'] = ds.y.size
    ds.attrs['totalwidth'] = ds.x.size


def _gen_l9_angles(dirname: Path, l9_angles: Union[str, Path, None] = None) -> None:
    """Generate angle files using the l9_angles executable."""
    log.debug(f'Geometry file is missing in {dirname}, generating it with {l9_angles}...')
    angles_txt_file = list(dirname.glob('LC*_ANG.txt'))
    assert len(angles_txt_file) == 1, 'angle file is missing'
    assert l9_angles is not None and Path(l9_angles).exists(), \
    'Please provide a valid executable file to compute angles'
    path_exe = Path(l9_angles).absolute()
    path_angles = Path(angles_txt_file[0]).absolute()
    with TemporaryDirectory() as tmpdir:
        system(f"cd {tmpdir} && {path_exe} {path_angles} BOTH 1 -b 1")
        system(f"cp -v {tmpdir/'*'} {dirname}")


def _read_geometry(ds: xr.Dataset, dirname: Path, l9_angles: Union[str, Path, None], chunks: list) -> None:
    """Read or generate sensor and solar angle rasters."""
    # read sensor and solar angles
    for name, search in [(str(n.saa), 'LC*_SAA.TIF'),
                         (str(n.sza), 'LC*_SZA.TIF'),
                         (str(n.vaa), 'LC*_VAA.TIF'),
                         (str(n.vza), 'LC*_VZA.TIF')]:
        data = open_raster(dirname, search, engine='rasterio').chunk(chunks)
        ds[name] = (data/100).astype('float32')
    
    if (str(n.saa) not in ds) and (l9_angles is not None):
        _gen_l9_angles(dirname, l9_angles)


def _read_radiometry(ds: xr.Dataset, dirname: Path, chunks: list) -> xr.Dataset:
    """Read and calibrate radiance, reflectance, and brightness temperature."""
    rescale = ds.metadata['LEVEL1_RADIOMETRIC_RESCALING']
    thermal = ds.metadata['LEVEL1_THERMAL_CONSTANTS']
    
    # Read Panchromatic band
    dims = (str(n.columns)+'_pan', str(n.rows)+'_pan')
    files = list(dirname.glob(f'LC*_B8.TIF'))
    assert len(files) == 1, 'None or several files have been found for panchromatic band'
    a, m = rescale[f'RADIANCE_ADD_BAND_8'], rescale[f'RADIANCE_MULT_BAND_8']
    data = xr.open_dataarray(files[0], engine='rasterio').chunk([1]+list(chunks))
    ds['Panchromatic'] = (dims,(m*data.squeeze()+a).data.astype('float32'))
    
    for f in dirname.glob(f'LC*_B*.TIF'):
        
        # Retrieve band name
        search = re.search(r'_B[0-9]*', f.name)
        b = f.name[search.start():search.end()]
        
        # Drop Panchromatic band
        if 'B8' in b: continue
        
        # read radiances
        a = rescale[f'RADIANCE_ADD_BAND_{b[2:]}']
        m = rescale[f'RADIANCE_MULT_BAND_{b[2:]}']
        data = xr.open_dataarray(f, engine='rasterio').chunk([1]+list(chunks))
        ds[str(n.ltoa)+b] = (m*data.squeeze()+a).astype('float32')
    
        # read reflectances
        if f'REFLECTANCE_ADD_BAND_{b[2:]}' not in rescale:
            ds[str(n.rtoa)+b] = xr.full_like(ds[str(n.ltoa)+b], np.nan, dtype='float32')
        else:        
            a = rescale[f'REFLECTANCE_ADD_BAND_{b[2:]}']
            m = rescale[f'REFLECTANCE_MULT_BAND_{b[2:]}']
            ds[str(n.rtoa)+b] = (m*data.squeeze()+a).astype('float32')
            ds[n.bgroup+b] = 'bands_vnir'      
        
        # read brightness temperatures
        if f'K1_CONSTANT_BAND_{b[2:]}' not in thermal:
            ds[str(n.bt)+b] = xr.full_like(ds[str(n.ltoa)+b], np.nan, dtype='float32')
        else:        
            k1 = thermal[f'K1_CONSTANT_BAND_{b[2:]}']
            k2 = thermal[f'K2_CONSTANT_BAND_{b[2:]}']
            rad = ds[str(n.ltoa)+b]
            ds[str(n.bt)+b] = (k2/np.log(k1/rad + 1)).astype('float32')
            ds[n.bgroup+b] = 'bands_ir'    
        
    ds = merge(ds, dim=str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)
    ds[str(n.ltoa)].attrs['unit'] = 'W/sr/m^2'
    ds[str(n.rtoa)].attrs['unit'] = None
    ds[str(n.bt)].attrs['unit'] = 'Kelvin'

    return ds

def _read_masks(ds: xr.Dataset, dirname: Path, chunks: list) -> None:
    """Read quality assurance (QA) mask files."""
    for t in dirname.glob('*_QA_*'):
        search = re.search(r'QA_[A-Z]*', t.name)
        name = t.name[search.start():search.end()]
        ds[name] = xr.open_dataarray(t, engine='rasterio').chunk([1]+list(chunks)).squeeze()


def _v1_compat(ds: xr.Dataset) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    return ds


def get_sample(level: int, mission: int = 8, use_cache: bool = True) -> Path:
    """
    Retrieve a sample Landsat OLI product directory for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Args:
        level: Processing level (1 for Level1, 2 for Level2)
        mission: Landsat mission number (8 for Landsat-8, 9 for Landsat-9)
        use_cache: Legacy parameter, not currently used

    Returns:
        Path to the Landsat OLI product directory
        
    Raises:
        ValueError: If level is not 1 or 2
        
    Example:
        >>> oli_dir = get_sample(level=1, mission=9)
        >>> ds = Level1_OLI(oli_dir)
    """
    # return Path('data/sample_products/LC08_L1TP_180054_20250104_20250111_02_T1')
    if level == 1:
        return env.getdir(f'DIR_L{mission}_L1C')
    elif level == 2:
        return env.getdir(f'DIR_L{mission}_L2A')
    else:
        raise ValueError(level)
    
    # try: 
    #     from sand.usgs import DownloadUSGS
    #     from sand.sample_product import products
    # except ImportError:
    #     raise ImportError('To use get_sample function, you need to install SAND module')
    
    # sensor = f'LANDSAT-{mission}-OLI'
    # dl = DownloadUSGS()
    # prod_id = products[sensor][f'l{level}_product']
    # return dl.download_file(prod_id, env.getdir('DIR_SAMPLES'))