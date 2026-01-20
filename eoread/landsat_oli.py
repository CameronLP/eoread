#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Landsat-9 OLI reader

Example:
    l1 = Level1_L9_OLI('LC09_L1TP_014034_20220618_20230411_02_T1/')

Data access:
    * https://earthexplorer.usgs.gov/
    * https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1
'''

import re
import numpy as np
import xarray as xr
import dask.array as da

from os import system
from pathlib import Path
from tempfile import TemporaryDirectory
from eoread.utils import filter_metadata, open_raster

from core import log, env
from core.interpolate import interp, Linear
from core.tools import merge, drop_unused_dims
from core.table import read_xml
from core.geo import n, convert_latlon_2D


# Central wavelengths aren't described in metadata. Thus, they are hard-coded
cwvl = [442.96,482.04,561.41,654.59,864.67,1608.86,2200.73,1373.43,10895,12050,]


def Level1_OLI(dirname: str|Path,
               l9_angles = None,
               chunks: int|tuple = 500,
               metadata_template: list|None = None,
               v1_compat: bool = False):
    '''
    Read an Landsat-8 or Landsat-9 OLI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, 
    brightness temperatures, the angles on the full grid, etc.

    Arguments:
        dirname: Path of the OLI directory path (Example: 'LC09_L1TP_014034_20220618_20230411_02_T1/')
        l9_angles: executable name of l9_angles program (ex: 'l9_angles/l9_angles'), used to generate the angles
                files automatically when missing, with the following command:
            l9_angles LC08_..._ANG.txt BOTH 1 -b 1
            l9_angles is available at:
            https://www.usgs.gov/land-resources/nli/landsat/solar-illumination-and-sensor-viewing-angle-coefficient-files

            It can be compiled with the following commands:
                wget https://landsat.usgs.gov/sites/default/files/documents/L9_ANGLES_2_7_0.tgz
                tar xzf L9_ANGLES_2_7_0.tgz
                rm -fv L9_ANGLES_2_7_0.tgz
                cd l9_angles
                make
                cd ..
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Directory does not exists'

    # Read metadata
    log.debug('read metadata')
    metadata = _read_metadata(ds, dirname, metadata_template)
    if isinstance(chunks, int): chunks = [chunks]*2

    # get datetime
    d = metadata['IMAGE_ATTRIBUTES']['DATE_ACQUIRED']
    t = metadata['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME']
    ds.attrs[str(n.datetime)] = d+'T'+t
    
    # Reading different rasters
    log.debug('read geometric angles')
    _read_geometry(ds, dirname, l9_angles, chunks)
    log.debug('read TOA rasters')
    ds = _read_radiometry(ds, dirname, chunks)
    log.debug('read masks')
    _read_masks(ds, dirname, chunks)
    _read_coordinates(ds, chunks)

    # other attributes
    log.debug('add important attributes')
    ds.attrs[str(n.platform)] = metadata['IMAGE_ATTRIBUTES']['SPACECRAFT_ID']
    ds.attrs[str(n.sensor)] = metadata['IMAGE_ATTRIBUTES']['SENSOR_ID']
    ds.attrs[str(n.product_name)] = metadata['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']
    ds.attrs[str(n.input_directory)] = str(dirname.parent)
    ds.attrs[str(n.resolution)] = 30
    
    
    # Sort band dimension
    ds = ds.assign({str(n.bnames): ((str(n.bands)), ds[str(n.bands)].data)})
    ds = ds.assign_coords({d: c.data.astype(int) 
                           for d,c in ds.coords.items() if str(n.bands) in d})
    ds = ds.sortby([str(n.bands), str(n.bands_ir), 'bands_nvis'])
    
    # define bands
    ds = ds.assign({str(n.cwav):((str(n.bands)), cwvl)})
    
    ds = ds.rename({'y': str(n.rows), 'x': str(n.columns)})   
    ds = drop_unused_dims(ds).unify_chunks()
    
    if v1_compat: return _v1_compat(ds)
    else: return ds


def _read_metadata(ds, dirname, template):
    filter_fn = (lambda x,y: x) if template is None else filter_metadata
    files_mtl = list(dirname.glob('LC*_MTL.xml'))
    assert len(files_mtl) == 1, 'XML file not found'
    data_mtl = read_xml(files_mtl[0])
    ds.attrs['metadata'] = filter_fn(data_mtl, template)
    return data_mtl


def _read_coordinates(ds, chunks):
    '''
    read lat/lon
    '''
    
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


def _gen_l9_angles(dirname, l9_angles=None):
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


def _read_geometry(ds, dirname, l9_angles, chunks):
    
    # read sensor and solar angles
    for name, search in [(str(n.saa), 'LC*_SAA.TIF'),
                         (str(n.sza), 'LC*_SZA.TIF'),
                         (str(n.vaa), 'LC*_VAA.TIF'),
                         (str(n.vza), 'LC*_VZA.TIF')]:
        data = open_raster(dirname, search, engine='rasterio').chunk(chunks)
        ds[name] = (data/100).astype('float32')
    
    if (str(n.saa) not in ds) and (l9_angles is not None):
        _gen_l9_angles(dirname, l9_angles)


def _read_radiometry(ds, dirname, chunks):
    
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
    
    ds = merge(ds, dim=str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)
    ds[str(n.ltoa)].attrs['unit'] = 'W/sr/m^2'
    
    for f in dirname.glob(f'LC*_B*.TIF'):
        
        # Retrieve band name
        search = re.search(r'_B[0-9]*', f.name)
        b = f.name[search.start():search.end()]
        if f'REFLECTANCE_ADD_BAND_{b[2:]}' not in rescale: continue
        
        # Drop Panchromatic band
        if 'B8' in b: continue
        
        # read reflectances
        a = rescale[f'REFLECTANCE_ADD_BAND_{b[2:]}']
        m = rescale[f'REFLECTANCE_MULT_BAND_{b[2:]}']
        filenames = list(dirname.glob(f'LC*{b}.TIF'))
        data = xr.open_dataarray(filenames[0], engine='rasterio').chunk([1]+list(chunks))
        ds[str(n.rtoa)+b] = (m*data.squeeze()+a).astype('float32')
    
    ds = merge(ds, dim='bands_nvis', pattern=r'(.+)_B(.+)', dtype=str)
    ds[str(n.rtoa)].attrs['unit'] = None
    
    for f in dirname.glob(f'LC*_B*.TIF'):
        
        # Retrieve band name
        search = re.search(r'_B[0-9]*', f.name)
        b = f.name[search.start():search.end()]
        if f'K1_CONSTANT_BAND_{b[2:]}' not in thermal: continue
        
        # read brightness temperatures
        k1 = thermal[f'K1_CONSTANT_BAND_{b[2:]}']
        k2 = thermal[f'K2_CONSTANT_BAND_{b[2:]}']
        rad = ds[str(n.ltoa)].sel({str(n.bands):b[2:]})
        ds[str(n.bt)+b] = (k2/np.log(k1/rad + 1)).astype('float32')

    ds = merge(ds, dim=str(n.bands_ir), pattern=r'(.+)_B(.+)', dtype=str)
    ds[str(n.bt)].attrs['unit'] = 'Kelvin'

    return ds

def _read_masks(ds, dirname, chunks):
    for t in dirname.glob('*_QA_*'):
        search = re.search(r'QA_[A-Z]*', t.name)
        name = t.name[search.start():search.end()]
        ds[name] = xr.open_dataarray(t, engine='rasterio').chunk([1]+list(chunks)).squeeze()


def _v1_compat(ds):
    return ds


def get_sample(level:int, use_cache:bool=True) -> Path:
    """
    Bring a Landsat-8 OLI directory path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    # # FIXME
    # return Path('data/sample_products/LC08_L1TP_180054_20250104_20250111_02_T1')
    try: 
        from sand.usgs import DownloadUSGS
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'LANDSAT-8-OLI'
    dl = DownloadUSGS()
    prod_id = products[sensor][f'l{level}_product']
    return dl.download_file(prod_id, env.getdir('DIR_SAMPLES'))