#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
List of MSI bands:
-----------------

Band Use         Wavelength Resolution
B1   Aerosols    443nm      60m
B2   Blue        490nm      10m
B3   Green       560nm      10m
B4   Red         665nm      10m
B5   Red Edge 1  705nm      20m
B6   Red Edge 2  740nm      20m
B7   Red Edge 3  783nm      20m
B8   NIR         842nm      10m
B8a  Red Edge 4  865nm      20m
B9   Water vapor 940nm      60m
B10  Cirrus      1375nm     60m
B11  SWIR 1      1610nm     20m
B12  SWIR 2      2190nm     20m
'''


# Update processing baseline 4.00
# https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming

from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from core.tools import merge, drop_unused_dims
from core.table import read_xml
from core import env, log
from core.geo import n
from core.interpolate import interp, Linear

from eoread.utils import filter_metadata, spatial_resample
from eoread.common import DataArray_from_array


user_guide = 'https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook.pdf/8869acdf-fd84-43ec-ae8c-3e80a436a16c?t=1438278087000'

def Level1_MSI(dirname : str|Path,
               resolution: Literal['10','20','60'] = '60',
               chunks: int|tuple = 500,
               metadata_template: list = None, 
               v1_compat: bool = False):
    '''
    Read an MSI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances,
    the angles on the full grid, etc.

    Arguments:
        dirname: Path of the MSI folder
        resolution: '60', '20' or '10' (in m)
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    ds = xr.Dataset()
    dirname = Path(dirname).resolve()
    assert isinstance(resolution, str)
    if isinstance(chunks, int): chunks = [chunks]*2

    if list(dirname.glob('GRANULE')):
        granules = list((dirname/'GRANULE').glob('*'))
        assert len(granules) == 1
        granule_dir = granules[0]
    else: granule_dir = dirname

    # load xml files
    xmlgranule = granule_dir/'MTD_TL.xml'
    xmlroot = dirname/'MTD_MSIL1C.xml'
    assert xmlgranule.exists()
    assert xmlroot.exists()
    log.debug('Reading metadata files')
    xmlgranule = read_xml(xmlgranule)
    xmlroot = read_xml(xmlroot)

    # load main xml file
    product_image = xmlroot['General_Info']['Product_Image_Characteristics']
    quantif = product_image['QUANTIFICATION_VALUE']['values']
    processing_baseline = xmlroot['General_Info']['Product_Info']['PROCESSING_BASELINE']
    
    # Extract bands wavelength
    cwvl, wvl_name = [],[]
    log.debug('Extract central wavelength')
    for spec in product_image['Spectral_Information_List']['Spectral_Information']:
        cwvl.append(spec['Wavelength']['CENTRAL']['values'])
        wvl_name.append(spec['attributes']['physicalBand'])

    # get platform
    tile_id = xmlgranule['General_Info']['TILE_ID']['values']
    platform = tile_id[:3]
    assert platform in ['S2A', 'S2B', 'S2C']

    # read image size for current resolution
    geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
    for e in geocoding.get('Size'):
        if e['attributes']['resolution'] == str(resolution):
            ds.attrs['totalheight'] = e.get('NROWS')
            ds.attrs['totalwidth'] = e.get('NCOLS')
            break

    # attributes
    log.debug('Add important attributes')
    sensing_time = xmlgranule['General_Info']['SENSING_TIME']['values']
    ds.attrs[n.datetime.name] = sensing_time
    ds.attrs[n.platform.name] = platform
    ds.attrs[n.resolution.name] = int(resolution)
    ds.attrs[n.sensor.name] = 'MSI'
    ds.attrs[n.product_name.name] = dirname.name
    ds.attrs[n.input_directory.name] = str(dirname.parent)
    ds.attrs['user_guide'] = user_guide

    # lat-lon
    log.debug('Extract central wavelength')
    _msi_read_latlon(ds, chunks, xmlgranule)

    # msi_read_geometry
    log.debug('Read and compute geometric angles')
    tileangles = xmlgranule['Geometric_Info']['Tile_Angles']
    ds = _msi_read_geometry(ds, tileangles, chunks)
    
    # msi read quality mask
    log.debug('WARNING: SKIPPING >> Read quality masks')
    # ds = _msi_read_qi(ds, granule_dir, chunks)

    # msi_read_toa and quality masks
    log.debug('Read top of atmosphere data')
    ds = _msi_read_toa(ds, granule_dir, quantif, processing_baseline, product_image, chunks, wvl_name)
    ds = ds.assign({n.cwav.name: ((n.bands.name), cwvl), 
                    n.bnames.name: ((n.bands.name), wvl_name)})
    
    # msi assign new bands coordinates
    ds = ds.assign_coords({
        n.bands_nvis.name: da.arange(1, len(ds[n.bands_nvis.name])+1),
        n.bands.name: da.arange(1, len(ds[n.bands_nvis.name])+1)
    })
    
    # Filter metadata
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
    ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)

    ds = drop_unused_dims(ds)
    if v1_compat: return _v1_compat(ds)
    return ds.unify_chunks()
    

def _msi_read_latlon(ds, chunks, xmlgranule):

    dims = (n.rows.name,n.columns.name)
    geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
    
    ds[n.lat.name] = DataArray_from_array(
        _LATLON(geocoding, 'lat', ds), dims,
        chunks=chunks,
    )

    ds[n.lon.name] = DataArray_from_array(
        _LATLON(geocoding, 'lon', ds), dims,
        chunks=chunks,
    )

def _msi_read_qi(ds, granule_dir, chunks):
    for filename in (granule_dir/'QI_DATA').glob(f'*.jp2'):
        
        if '_PVI' in filename.stem: continue
        arr = xr.open_dataarray(filename, engine='rasterio')
        arr = arr.chunk([1]+list(chunks))
        arr = arr.rename(x='x_red', y='y_red').astype('float32')
        ds[filename.stem] = arr.rename({'band':n.detector.name})
    
    ds = ds.rename_vars({'MSK_CLASSI_B00':'MSK_CLASSI'})
    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)

    return ds

def _msi_read_toa(ds, granule_dir, quantif, processing_baseline, product_image, chunks, bnames):
    
    # Retrieve radiometric offset
    if float(processing_baseline) >= 4:
        radio_offset = [
            int(x['values'])
            for x in product_image['Radiometric_Offset_List']['RADIO_ADD_OFFSET']]
    else: radio_offset = [0]*len(bnames)
    
    # Open deserved bands
    indexes = []
    for filename in (granule_dir/'IMG_DATA').glob(f'*.jp2'):
        
        # Add band to dataset
        band = filename.stem.split('_')[-1]
        if 'TCI' == band: continue
        iband = list(bnames).index(band.replace('B0','B'))
        indexes.append(iband)
        
        arr = xr.open_dataarray(filename, engine='rasterio').chunk([1]+list(chunks))
        arr = ((arr+radio_offset[iband])/quantif).astype('float32')
        
        # Resample the array
        ratio = {n.columns.name: ds.totalheight, n.rows.name: ds.totalwidth} 
        arr_resampled = spatial_resample(arr.squeeze(), ratio, chunks)
        ds[n.rtoa.name+f'_{band}'] = arr_resampled

    ds = merge(ds, dim=n.bands_nvis.name, pattern=r'(.+)_B(.+)', dtype=str)
    ds[n.rtoa.name].attrs.update(unit=None)
    
    # Reorder bands
    order = [indexes.index(i) for i in range(len(indexes))]    
    return ds.isel({n.bands_nvis.name: order})


def _msi_read_geometry(ds, tileangles, chunks):
    """
    Reads and processes geometric data from MSI tiles.

    Parameters:
        ds (xarray Dataset): Input dataset containing MSI data
        tileangles (dict): Dictionary containing XML blocks for solar and view angles
        chunks (tuple): Chunk sizes for dask arrays

    Returns:
        xarray Dataset: Output dataset with updated coordinates and geometry variables
    """
    
    # read solar angles at tiepoints
    dims = ('tie_rows', 'tie_columns')
    sza = _read_xml_block(tileangles['Sun_Angles_Grid']['Zenith'], dims)
    saa = _read_xml_block(tileangles['Sun_Angles_Grid']['Azimuth'], dims)

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    tie_shape = None
    vza, vaa = {}, {}
    for e in tileangles.get('Viewing_Incidence_Angles_Grids'):

        # Reading zenith angles
        data: np.ndarray = _read_xml_block(e['Zenith'], dims)
        bandid = int(e['attributes']['bandId'])
        
        if tie_shape is None: tie_shape = data.shape # in case the size is not constant
        data = data.values.flatten()
        
        if bandid not in vza: vza[bandid] = data
        valid = ~np.isnan(data) # indexes where the data is not null
        vza[bandid][valid] = data[valid]

        # Reading azimuth angles
        data = _read_xml_block(e['Azimuth'], dims)
        bandid = int(e['attributes']['bandId'])
        
        data: np.ndarray = data.values.flatten()
        
        if bandid not in vaa: vaa[bandid] = data
        valid = ~np.isnan(data) # indexes where the data is not null
        vaa[bandid][valid] = data[valid]

    # reshape to original 
    for b in vza:
        vza[b] = vza[b].reshape(tie_shape)
    for b in vaa:
        vaa[b] = vaa[b].reshape(tie_shape)
    

    # TODO: check for 
    # use the first band as vza and vaa
    vza = vza[0]
    vaa = vaa[0]

    ntie_rows, ntie_columns = sza.shape
    tie_rows    = np.int32(da.linspace(0, shp[0]-1, ntie_rows))              # tie resolution, with target values
    tie_columns = np.int32(da.linspace(0, shp[1]-1, ntie_columns))           # tie resolution, with target values
    ds = ds.assign_coords(tie_rows = tie_rows, tie_columns = tie_columns)

    # initialize the dask arrays
    x = xr.DataArray(da.arange(len(ds.x), chunks=chunks[0]), dims=('x'))
    y = xr.DataArray(da.arange(len(ds.y), chunks=chunks[1]), dims=('y'))
    for name, tie in [(n.sza, sza),(n.saa, saa),(n.vza, vza),(n.vaa, vaa)]:
        ds[name.name+'_tie'] = xr.DataArray(tie, dims=dims)
        interp_tie = interp(ds[name.name+'_tie'], tie_rows=Linear(x), tie_columns=Linear(y))
        ds[name.name] = interp_tie
        ds[name.name] = ds[name.name].chunk(chunks)
    
    return ds


def _read_xml_block(item, dims):
    '''
    read a block of xml data and returns it as a xarray float32 DataArray
    '''
    return xr.DataArray([i.split() for i in item['Values_List']['VALUES']], 
                        dims=dims).astype('float32')


class _LATLON:
    '''
    An array-like to calculate the MSI lat-lon
    '''
    def __init__(self, geocoding, kind, ds):
        self.kind = kind

        code = geocoding.get('HORIZONTAL_CS_CODE')
        self.proj = pyproj.Proj(code)

        # lookup position in the UTM grid
        for e in geocoding.get('Geoposition'):
            if e['attributes']['resolution'] == str(ds.resolution):
                ULX = e.get('ULX')
                ULY = e.get('ULY')
                XDIM = e.get('XDIM')
                YDIM = e.get('YDIM')

        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        self.x = ULX + XDIM//2 + XDIM*da.arange(ds.totalheight)
        self.y = ULY + YDIM//2 + YDIM*da.arange(ds.totalwidth)

        self.shape = (ds.totalheight, ds.totalwidth)
        self.ndim = 2
        self.dtype = 'float32'

    def __getitem__(self, key):
        
        X, Y = self.x[key[1]], self.y[key[0]]
        if isinstance(key[0], slice) and isinstance(key[1], slice):
            # keys are both slices
            X, Y = da.meshgrid(X, Y)
        else:
            X, Y = da.broadcast_arrays(X, Y)

        lon, lat = self.proj(X, Y, inverse=True)

        if self.kind == 'lat': return lat.astype(self.dtype)
        else: return lon.astype(self.dtype)


def Level2_MSI(dirname):
    """
    Read an MSI level2 product as xarray.Dataset
    """
    raise NotImplementedError


def get_sample(level:int=1, use_cache:bool=True) -> Path:
    """
    Bring a MSI directory path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    try: 
        from core.files import cache_dataframe
        from sand.copernicus_dataspace import DownloadCDSE
        from sand.sample_product import products
    except ImportError:
        log.error('To use get_sample function, you need to install SAND module',
                  e=ImportError)
    
    cachefile = env.getdir('DIR_STATIC')/'query_s2.pickle'
    if use_cache: cache_deco = cache_dataframe(cachefile)
    else: cache_deco = lambda x: x
    
    sensor = 'SENTINEL-2-MSI'
    params = products[sensor][f'level{level}']
    dl = DownloadCDSE(sensor, level)
    ls = cache_deco(dl.query)(**params)
    return dl.download(ls.iloc[0], env.getdir('DIR_SAMPLES'))

def _v1_compat(ds):
    
    # Remove metadata
    ds.attrs['metadata_granule'] = filter_metadata(ds.attrs['metadata_granule'], [])
    ds.attrs['metadata'] = filter_metadata(ds.attrs['metadata'], [])
    
    # rename bands variable
    ds = ds.assign({n.rtoa.name: ((n.bands.name, n.rows.name, n.columns.name), ds[n.rtoa.name].data)})
    ds = ds.drop_dims(n.bands_nvis.name)
    
    # Apply previous rounded central wavelengths
    msi_band = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
    ds = ds.assign_coords(bands=msi_band)
    
    # rename wavelength variable
    ds = ds.rename({n.cwav.name:'wav'})
    
    # add flags
    from core.tools import raiseflag
    ds[n.flags.name] = xr.zeros_like(
        ds.vza,
        dtype=n.flags.dtype)
    raiseflag(
        ds[n.flags.name],
        'L1_INVALID', 4,
        np.isnan(ds.vza)
        )
    
    return drop_unused_dims(ds)