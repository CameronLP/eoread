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

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from core.tools import merge, drop_unused_dims
from core.table import read_xml
from core import env, log
from core.geo import n

from .common import DataArray_from_array, Interpolator, Repeat


def Level1_MSI(dirname,
               resolution='60',
               chunks=500):
    '''
    Read an MSI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances,
    the angles on the full grid, etc.

    Arguments:
        resolution: '60', '20' or '10' (in m)
    '''
    ds = xr.Dataset()
    dirname = Path(dirname).resolve()
    assert isinstance(resolution, str)
    if isinstance(chunks, int): chunks = [chunks]*2

    if list(dirname.glob('GRANULE')):
        granules = list((dirname/'GRANULE').glob('*'))
        assert len(granules) == 1
        granule_dir = granules[0]
    else:
        granule_dir = dirname

    # load xml files
    xmlgranule = granule_dir/'MTD_TL.xml'
    xmlroot = dirname/'MTD_MSIL1C.xml'
    assert xmlgranule.exists()
    assert xmlroot.exists()
    log.debug('Reading metadata files')
    ds.attrs['metadata_granule'] = read_xml(xmlgranule)
    ds.attrs['metadata'] = read_xml(xmlroot)

    # load main xml file
    product_image = ds.attrs['metadata']['General_Info']['Product_Image_Characteristics']
    quantif = product_image['QUANTIFICATION_VALUE']['values']
    processing_baseline = ds.attrs['metadata']['General_Info']['Product_Info']['PROCESSING_BASELINE']
    
    # Extract bands wavelength
    cwvl, wvl_name = [],[]
    log.debug('Extract central wavelength')
    for spec in product_image['Spectral_Information_List']['Spectral_Information']:
        cwvl.append(spec['Wavelength']['CENTRAL']['values'])
        wvl_name.append(spec['attributes']['physicalBand'])
    ds = ds.assign({n.cwav.name: ((n.bands.name),cwvl), 
                    n.bnames.name: ((n.bands.name),wvl_name)})
    
    if float(processing_baseline) >= 4:
        radio_offset_list = [
            int(x['values'])
            for x in product_image['Radiometric_Offset_List']['RADIO_ADD_OFFSET']]
    else:
        radio_offset_list = [0]*len(cwvl)

    # get platform
    tile_id = ds.attrs['metadata_granule']['General_Info']['TILE_ID']['values']
    platform = tile_id[:3]
    assert platform in ['S2A', 'S2B']

    # read image size for current resolution
    geocoding = ds.attrs['metadata_granule']['Geometric_Info']['Tile_Geocoding']
    for e in geocoding.get('Size'):
        if e['attributes']['resolution'] == str(resolution):
            ds.attrs['totalheight'] = e.get('NROWS')
            ds.attrs['totalwidth'] = e.get('NCOLS')
            break

    # attributes
    log.debug('Add important attributes')
    ds.attrs[n.datetime.name] = ds.attrs['metadata_granule']['General_Info']['SENSING_TIME']['values']
    ds.attrs[n.platform.name] = platform
    ds.attrs['resolution'] = resolution
    ds.attrs['sensor'] = 'MSI'
    ds.attrs['product_name'] = dirname.name
    ds.attrs['input_directory'] = str(dirname.parent)

    # lat-lon
    log.debug('Extract central wavelength')
    msi_read_latlon(ds, chunks)

    # msi_read_geometry
    log.debug('Read and compute geometric angles')
    tileangles = ds.attrs['metadata_granule']['Geometric_Info']['Tile_Angles']
    ds = msi_read_geometry(ds, tileangles, chunks)

    # msi_read_toa and quality masks
    log.debug('Read top of atmosphere data')
    ds = msi_read_toa(ds, granule_dir, quantif, radio_offset_list, chunks)
    log.debug('Read quality masks')
    ds = msi_read_qi(ds, granule_dir, chunks)

    # # flags
    # ds[n.flags] = xr.zeros_like(
    #     ds.vza,
    #     dtype=n.flags_dtype)
    # raiseflag(
    #     ds[n.flags],
    #     'L1_INVALID',
    #     flags['L1_INVALID'],
    #     np.isnan(ds.vza)
    #     )

    ds = drop_unused_dims(ds)
    return ds
    

def msi_read_latlon(ds, chunks):

    dims = (n.rows.name,n.columns.name)
    geocoding = ds.attrs['metadata_granule']['Geometric_Info']['Tile_Geocoding']
    
    ds[n.lat.name] = DataArray_from_array(
        LATLON(geocoding, 'lat', ds), dims,
        chunks=chunks,
    )

    ds[n.lon.name] = DataArray_from_array(
        LATLON(geocoding, 'lon', ds), dims,
        chunks=chunks,
    )

def msi_read_qi(ds, granule_dir, chunks):
    for filename in (granule_dir/'QI_DATA').glob(f'*.jp2'):
        
        if '_PVI' in filename.stem: continue
        arr = xr.open_dataarray(filename).chunk([1]+chunks).astype('float32')
        arr = arr.rename(x='x_red', y='y_red')
        ds[filename.stem] = arr.rename({'band':n.detector.name})
    
    ds = ds.rename_vars({'MSK_CLASSI_B00':'MSK_CLASSI'})
    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)

    return ds

def msi_read_toa(ds, granule_dir, quantif, radio_offset, chunks):

    for filename in (granule_dir/'IMG_DATA').glob(f'*.jp2'):
        
        # Add band to dataset
        band = filename.stem.split('_')[-1]
        if 'TCI' == band: continue
        iband = list(ds[n.bnames.name]).index(band.replace('B0','B'))
                
        arr = xr.open_dataarray(filename) + radio_offset[iband]
        arr = (arr.chunk([1]+chunks)/quantif).astype('float32')
        
        # Resample the array
        xrat = len(arr.x)/ds.totalwidth
        yrat = len(arr.y)/ds.totalheight
        
        arr_resampled = msi_resample(arr, (xrat,yrat), chunks)
        ds[n.rtoa.name+f'_{band}'] = arr_resampled

    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)
    return ds

def msi_resample(arr, ratio, chunks):
    
    arr = arr.squeeze()
    x,y = int(ratio[0]), int(ratio[1])
    if x >= 1.:
        # downsample
        arr_resampled = 0.
        for i in range(x):
            for j in range(y):
                arr_resampled += arr.isel(x=slice(i,None,x), y=slice(j,None,y))
        arr_resampled /= x*y
    else:
        # over-sample
        arr_resampled = DataArray_from_array(
            Repeat(arr, (int(1/ratio[1]), int(1/ratio[0]))),
            (n.rows.name,n.columns.name),
            chunks=chunks,
        )

    arr_resampled = arr_resampled.rename({
        'x': n.columns.name,
        'y': n.rows.name})
    
    return arr_resampled

# def msi_read_spectral(ds):
#     # read srf
#     # TODO: deprecate in favour of get_SRF
#     dir_aux_msi = mdir(env.getdir('DIR_STATIC')/'msi')
#     platform = ds.platform
#     get_SRF(platform)
#     srf_file = dir_aux_msi/f'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_{platform}.csv'

#     assert srf_file.exists(), srf_file

#     srf_data = pd.read_csv(srf_file)
#     wav = srf_data.SR_WL

#     wav_data = []

#     for bn in ds[n.bnames.name]:
#         col = platform + '_SR_AV_' + str(bn.values)
#         srf = srf_data[col]
#         wav_eq = np.trapz(wav*srf)/np.trapz(srf)
#         wav_data.append(wav_eq)

#     ds[n.wav.name] = xr.DataArray(
#         da.from_array(wav_data),
#         dims=(n.bands.name),
#     ).chunk({n.bands.name: 1})


def msi_read_geometry(ds, tileangles, chunks):

    dims = ('tie_rows', 'tie_columns')
    # read solar angles at tiepoints
    sza = read_xml_block(tileangles['Sun_Angles_Grid']['Zenith'], dims)
    saa = read_xml_block(tileangles['Sun_Angles_Grid']['Azimuth'], dims)

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    vza, vaa = {}, {}
    for e in tileangles.get('Viewing_Incidence_Angles_Grids'):

        # read zenith angles
        data = read_xml_block(e['Zenith'], dims)
        bandid = int(e['attributes']['bandId'])
        if bandid not in vza: vza[bandid] = data
        else: vza[bandid] = vza[bandid].where(~data.isnull(), data)

        # read azimuth angles
        data = read_xml_block(e['Azimuth'], dims)
        bandid = int(e['attributes']['bandId'])
        if bandid not in vaa: vaa[bandid] = data
        else: vaa[bandid] = vaa[bandid].where(~data.isnull(), data)

    # use the first band as vza and vaa
    k = sorted(vza.keys())[0]
    assert k in vaa

    # initialize the dask arrays
    for name, tie in [(n.sza, sza),(n.saa, saa),(n.vza, vza[k]),(n.vaa, vaa[k])]:
        ds[name.name+'_tie'] = xr.DataArray(tie, dims=dims)
        ds[name.name] = DataArray_from_array(
            Interpolator(shp, ds[name.name+'_tie']),
            (n.rows.name,n.columns.name),
            chunks,
        )
    
    return ds.assign_coords(tie_rows = da.linspace(0, shp[0]-1, sza.shape[0]),
                         tie_columns = da.linspace(0, shp[1]-1, sza.shape[1]))


def read_xml_block(item, dims):
    '''
    read a block of xml data and returns it as a xarray float32 DataArray
    '''
    return xr.DataArray([i.split() for i in item['Values_List']['VALUES']], 
                        dims=dims).astype('float32')


class LATLON:
    '''
    An array-like to calculate the MSI lat-lon
    '''
    def __init__(self, geocoding, kind, ds):
        self.kind = kind

        code = geocoding.get('HORIZONTAL_CS_CODE')
        self.proj = pyproj.Proj(code)

        # lookup position in the UTM grid
        for e in geocoding.get('Geoposition'):
            if e['attributes']['resolution'] == ds.resolution:
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