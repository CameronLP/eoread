#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
List of VENµs bands:
-----------------

Band Use         Wavelength Bandwidth Resolution
B1   Atmo Correc 420nm      40nm      5m
B2   Aerosol     443nm      40nm      5m
B3   Water       490nm      40nm      5m
B4   Land        555nm      40nm      5m
B5   Vege Index  620nm      40nm      5m
B6   Image quali 620nm      40nm      5m
B7   Red Edge 1  667nm      30nm      5m
B8   Red Edge 2  702nm      24nm      5m
B9   Red Edge 3  742nm      16nm      5m
B10  Red Edge 4  782nm      16nm      5m
B11  Vege Index  865nm      40nm      5m
B12  Water vapor 910nm      20nm      5m
'''

# https://www.eoportal.org/satellite-missions/venus#vssc-ven%C2%B5s-superspectral-camera

from pathlib import Path
import dask.array as da
import pandas as pd
import xarray as xr
import pyproj

from core.geo import n
from core.table import read_xml
from core.download import download_url
from core.files import mdir
from core.tools import merge, drop_unused_dims
from core import env, log

from eoread.utils import *
from eoread.common import DataArray_from_array


def Level1_VENUS(dirname, 
                 chunks: int|tuple = 500,
                 read_masks: bool = False, 
                 metadata_template: list = None,
                 v1_compat: bool = False):
    '''
    Read an Venµs Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA reflectances,
    the angles on the full grid, etc.

    Arguments:
        dirname: Path of the VENµS directory
        chunks: Size of chunks for spatial axis
        read_masks: Option to read compressed masks
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exists'
    if isinstance(chunks, int): chunks = [chunks]*2    
    
    # read metadata
    log.debug('Reading metadata')
    ds, metadata_granule = _venus_read_metadata(ds, dirname, metadata_template)

    # read geaometry
    log.debug('Read and compute geometric angles')
    ds = _venus_read_geometry(ds, dirname, chunks)

    # read TOA
    log.debug('Read top of atmosphere data')
    radio_info = metadata_granule['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = _venus_read_toa(ds, dirname, quantif, chunks)

    # lat-lon
    log.debug('Compute LatLon raster')
    geocoding = metadata_granule['Geoposition_Informations']
    _venus_read_latlon(ds, geocoding, chunks)
    
    # read cloud altitude
    log.debug('Open masks')
    ratio = {n.columns.name: ds.totalwidth, n.rows.name: ds.totalheight} 
    cld = open_raster(dirname/'DATA', '*CLA_ALL.tif')
    cld = cld.rename(x=n.columns.name, y=n.rows.name)
    ds['CLA_ALL'] = spatial_resample(cld, ratio, chunks, 'repeat')
    
    if read_masks:
        
        # read cloud mask
        cld = open_raster(dirname/'MASKS','*CLD_XS.zip','.zip').chunk(chunks)
        ds['CLD_XS'] = cld.rename(x=n.columns.name, y=n.rows.name)
        
        # read cloud mask
        usi = open_raster(dirname/'MASKS','*USI_XS.zip','.zip').chunk(chunks)
        ds['USI_XS'] = usi.rename(x=n.columns.name, y=n.rows.name)
    
        # Read quality masks
        for bn in ds[n.bnames.name]:
            
            pix = open_raster(dirname/'MASKS',f'*PIX_{bn.values}.zip','.zip').chunk(chunks)
            ds[f'PIX_{bn.values}'] = pix.rename(x=n.columns.name, y=n.rows.name)
            
            sat = open_raster(dirname/'MASKS',f'*SAT_{bn.values}.zip','.zip').chunk(chunks) 
            ds[f'SAT_{bn.values}'] = sat.rename(x=n.columns.name, y=n.rows.name)
    
    else: 
        log.debug('Masks are not red due to uncompression time consuming. '
                  'Active option read_masks to read them')
        
    ds = merge(ds, n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)    
    ds = ds.assign_coords({n.bands.name: ds[n.bands.name].data.astype(int),
                           n.bands_nvis.name: ds[n.bands.name].data.astype(int)})    
    return drop_unused_dims(ds).unify_chunks()


def Level2_VENUS(dirname, 
                 chunks: int|tuple = 500,
                 metadata_template: list = None):
    '''
    Read an Venµs Level2 product as an xarray.Dataset

    Arguments:
        chunk: size of a single chunk
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exists'
    if isinstance(chunks, int): chunks = [chunks]*2
    
    # read metadata
    log.debug('Reading metadata')
    ds, metadata_granule = _venus_read_metadata(ds, dirname, metadata_template)
    
    # lat-lon
    log.debug('Compute LatLon raster')
    geocoding = metadata_granule['Geoposition_Informations']
    _venus_read_latlon(ds, geocoding, chunks)

    # read geaometry
    log.debug('Read and compute geometric angles')
    ds = _venus_read_geometry(ds, dirname, chunks)

    # read reflectances
    radio_info = metadata_granule['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = _venus_read_rho(ds, dirname, quantif, chunks)
    
    # read cloud mask
    log.debug('Open masks')
    cld = open_raster(dirname/'MASKS','*CLM_XS.tif').chunk(chunks)
    ds['CLM_XS'] = cld.rename(x=n.columns.name, y=n.rows.name)
    
    # read other masks
    usi = open_raster(dirname/'MASKS','*USI_XS.tif').chunk(chunks)
    ds['USI_XS'] = usi.rename(x=n.columns.name, y=n.rows.name)
    
    cld = open_raster(dirname/'MASKS','*SAT_XS.tif').chunk(chunks)
    ds['SAT_XS'] = cld.rename(x=n.columns.name, y=n.rows.name)
    
    usi = open_raster(dirname/'MASKS','*PIX_XS.tif').chunk([1]+list(chunks))
    ds['PIX_XS'] = usi.rename(x=n.columns.name, y=n.rows.name, band=n.bands.name)
    
    cld = open_raster(dirname/'MASKS','*IAB_XS.tif').chunk(chunks)
    ds['IAB_XS'] = cld.rename(x=n.columns.name, y=n.rows.name)
    
    usi = open_raster(dirname/'MASKS','*EDG_XS.tif').chunk(chunks)
    ds['EDG_XS'] = usi.rename(x=n.columns.name, y=n.rows.name)
    
    return drop_unused_dims(ds).unify_chunks()


def _venus_read_metadata(ds, dirname, metadata_template):

    # load xml file
    xmlfiles = list((dirname/'DATA').glob('*UII_ALL.xml'))
    assert len(xmlfiles) == 1
    xmlroot = read_xml(xmlfiles[0])

    # load main xml file
    xmlfiles = list(dirname.glob('*MTD_ALL.xml'))
    assert len(xmlfiles) == 1
    xmlgranule = read_xml(xmlfiles[0])
    
    # Extract resolution, band names and wavelength
    resolution = None
    bandnames, cwvl = [], []
    log.debug('Extract central wavelength')
    radio_info = xmlgranule['Radiometric_Informations']['Spectral_Band_Informations_List']
    for band in radio_info['Spectral_Band_Informations']:
        r = band['SPATIAL_RESOLUTION']['values']
        if resolution: assert resolution == r
        else: resolution = r
        cwvl.append(band['Wavelength']['CENTRAL']['values'])
        bandnames.append(band['attributes']['band_id'])
    ds = ds.assign({n.cwav.name: ((n.bands.name),cwvl),
                    n.bnames.name: ((n.bands.name),bandnames)})
    
    # read date
    date = xmlgranule['Product_Characteristics']['ACQUISITION_DATE']

    # get platform
    platform = xmlgranule['Product_Characteristics']['PLATFORM']
    assert platform == 'VENUS'

    # read image size for current resolution
    shape_info = xmlgranule['Geoposition_Informations']['Geopositioning']['Group_Geopositioning_List']
    ds.attrs['totalheight'] = shape_info['Group_Geopositioning']['NROWS']
    ds.attrs['totalwidth'] = shape_info['Group_Geopositioning']['NCOLS']
    
    # attributes
    log.debug('Add important attributes')
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs[n.datetime.name] = date
    ds.attrs[n.platform.name] = platform
    ds.attrs[n.resolution.name] = resolution
    ds.attrs[n.sensor.name] = 'VENUS'
    ds.attrs[n.product_name.name] = xmlgranule['Product_Characteristics']['PRODUCT_ID']
    ds.attrs[n.input_directory.name] = str(dirname.parent)
    ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
    ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)
    
    return ds, xmlgranule


def _venus_read_latlon(ds, geocoding, chunks):
    
    ds[n.lat.name] = DataArray_from_array(
        _LATLON(geocoding, 'lat', ds),
        (n.rows.name, n.columns.name),
        chunks=chunks,
    )

    ds[n.lon.name] = DataArray_from_array(
        _LATLON(geocoding, 'lon', ds),
        (n.rows.name, n.columns.name),
        chunks=chunks,
    )

def _venus_read_toa(ds, granule_dir, quantif, chunks):
    
    for name in ds[n.bnames.name]:
        
        arr = open_raster(granule_dir, f'*REF_{name.values}.tif').chunk(chunks)
        arr = (arr/quantif).astype('float32')
        
        ratio = {n.rows.name: ds.totalheight, n.columns.name: ds.totalwidth}        
        arr_resampled = spatial_resample(arr, ratio, chunks)
        ds[n.rtoa.name+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=n.bands_nvis.name, pattern=r'(.+)_B(.+)', dtype=str)
    ds[n.rtoa.name].attrs.update(unit=None)
    return ds


def _venus_read_rho(ds, granule_dir, quantif, chunks):

    for rho, var in zip(['SRE','FRE'],['rho_surface','rho_flat']):
        for name in ds[n.bnames.name]:
            
            arr = open_raster(granule_dir, f'*{rho}_{name.values}.tif').chunk(chunks)
            arr = (arr/quantif).astype('float32')

            ratio = {'y': ds.totalheight, 'x': ds.totalwidth}  
            arr_resampled = spatial_resample(arr, ratio, chunks)
            ds[var+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)
    
    # read Aerosol_Optical_Thickness of waper vapor content
    atb = open_raster(granule_dir, '*ATB_XS.tif').chunk([1]+list(chunks))
    ds['water_vapor'] = atb.sel(band=1)
    ds['aod'] = atb.sel(band=2)

    return ds

def _venus_read_geometry(ds, dirname, chunks):

    # read solar angles
    sa = open_raster(dirname/'DATA','*SOL_ALL.tif').chunk([1]+list(chunks))
    ds['SOL_ALL'] = sa.rename(x=n.columns.name+'_tie', y=n.rows.name+'_tie')
    
    # read view angles
    va = open_raster(dirname/'DATA','*VIE_ALL.tif').chunk([1]+list(chunks))
    ds['VIE_ALL'] = va.rename(x=n.columns.name+'_tie', y=n.rows.name+'_tie')
    
    return ds.rename(band=n.bands.name+'_angle')

class _LATLON:
    '''
    An array-like to calculate the VENUS lat-lon
    '''
    def __init__(self, geocoding, kind, ds):
        self.kind = kind

        code = geocoding['Coordinate_Reference_System']['Horizontal_Coordinate_System']['HORIZONTAL_CS_CODE']

        self.proj = pyproj.Proj('EPSG:{}'.format(code))

        # lookup position in the UTM grid
        geopos = geocoding['Geopositioning']['Group_Geopositioning_List']
        geopos = geopos['Group_Geopositioning']
        ULX = int(geopos['ULX'])
        ULY = int(geopos['ULY'])
        XDIM = int(geopos['XDIM'])
        YDIM = int(geopos['YDIM'])

        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        self.x = ULX + XDIM//2 + XDIM*da.arange(ds.totalwidth)
        self.y = ULY + YDIM//2 + YDIM*da.arange(ds.totalheight)

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

        if self.kind == 'lat':
            if hasattr(lat, 'astype'):
                return lat.astype(self.dtype)
            else:
                return da.array(lat, dtype=self.dtype)
        else:
            if hasattr(lon, 'astype'):
                return lon.astype(self.dtype)
            else:
                return da.array(lon, dtype=self.dtype)


def get_SRF(
    ds_in: xr.Dataset = None, dir_data: Path = None
) -> xr.Dataset:
    """
    Load Venµs spectral response functions (SRF)

    If ds_in is provided, the output bands are references by ds_in.bands
    """
    if dir_data is None:
        dir_data = mdir(env.getdir('DIR_STATIC')/'venus')

    url = 'https://labo.obs-mip.fr/wp-content-labo/uploads/sites/19/2018/09/rep6S.txt'
    srf_file = download_url(url, dir_data)
    nbands = 12
    ibands = range(1, nbands+1)
    df = pd.read_csv(
        srf_file,
        sep=None,
        names=['wav_um', *ibands])

    ds = xr.Dataset()
    ds.attrs["desc"] = 'Spectral response functions for VENµS'

    if ds_in is None:
        bids = ibands
    else:
        assert len(ds_in.bands) == nbands
        bids = ds_in.bands.values
    for i in range(nbands):
        ds[bids[i]] = xr.DataArray(
            df[ibands[i]].values,
            dims=["wav"],
            attrs={"band_info": f"VENUS band {bids[i]}"},
        )

    ds = ds.assign_coords(wav=df['wav_um'].values*1000)
    ds[n.wav].attrs["units"] = "nm"

    return ds

def get_sample(level:int=1, use_cache:bool=True) -> Path:
    """
    Bring a VENµS directory path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    return Path('/mnt/ceph/data/VENUS/VENUS-XS_20230116-112657-000_L1C_VILAINE_C_V3-1/')
    try: 
        from core.files import cache_dataframe
        from sand.geodes import DownloadCNES
        from sand.sample_product import products
    except ImportError:
        log.error('To use get_sample function, you need to install SAND module',
                  e=ImportError)
        
    cachefile = env.getdir('DIR_STATIC')/'query_venus.pickle'
    if use_cache: cache_deco = cache_dataframe(cachefile)
    else: cache_deco = lambda x: x
    
    sensor = 'VENUS'
    params = products[sensor][f'level{level}']
    dl = DownloadCNES(sensor, level)
    ls = cache_deco(dl.query)(**params)
    return dl.download(ls.iloc[0], env.getdir('DIR_SAMPLES'))