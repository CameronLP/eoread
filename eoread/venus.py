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

from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pyproj

from core.geo import n
from core.table import read_xml
from core.download import download_url
from core.files import mdir, uncompress
from core import env, log

from .common import DataArray_from_array, Interpolator, Repeat
from core.tools import raiseflag, merge


def Level1_VENUS(dirname, chunks:int = 500):
    '''
    Read an Venµs Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA reflectances,
    the angles on the full grid, etc.

    Arguments:
        chunk: size of a single chunk
    '''
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists()
    
    # read metadata
    ds = venus_read_metadata(ds, dirname)
    if isinstance(chunks, int): chunks = [chunks]*2

    # lat-lon
    geocoding = ds.attrs['Geoposition_Informations']
    venus_read_latlon(ds, geocoding, chunks)

    # read geaometry
    venus_read_geometry(ds, dirname, chunks)

    # read TOA
    radio_info = ds.attrs['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = venus_read_toa(ds, dirname, quantif, chunks)

    with TemporaryDirectory() as tmpdir:
        
        # read cloud altitude
        cloudfiles = list((dirname/'DATA').glob('*CLA_ALL.tif'))
        assert len(cloudfiles) == 1
        cld = xr.open_dataarray(cloudfiles[0]).chunk([1]+chunks).squeeze()
        ds['cloud_altitude'] = DataArray_from_array(
            Interpolator(ds.latitude.shape, cld),
            ('y','x'),
            chunks,
        )
        
        # read cloud mask
        filenames = list((dirname/'MASKS').glob('*CLD_XS.zip'))
        assert len(filenames) == 1
        cldpath = uncompress(filenames[0], tmpdir)
        ds[cldpath.stem] = xr.open_dataarray(cldpath).chunk([1]+chunks).squeeze()
        
        # read cloud mask
        filenames = list((dirname/'MASKS').glob('*USI_XS.zip'))
        assert len(filenames) == 1
        usipath = uncompress(filenames[0], tmpdir)
        ds[usipath.stem] = xr.open_dataarray(usipath).chunk([1]+chunks).squeeze()
        
        # Read quality masks
        for bn in ds[n.bnames.name]:
            
            filenames = list((dirname/'MASKS').glob(f'*PIX_{bn.values}.zip'))
            assert len(filenames) == 1
            pixpath = uncompress(filenames[0], tmpdir)
            ds[pixpath.stem] = xr.open_dataarray(pixpath).chunk([1]+chunks).squeeze()
            
            filenames = list((dirname/'MASKS').glob(f'*SAT_{bn.values}.zip'))
            assert len(filenames) == 1
            satpath = uncompress(filenames[0], tmpdir)
            ds[satpath.stem] = xr.open_dataarray(satpath).chunk([1]+chunks).squeeze()     
        
    ds = merge(ds, n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)        
    return ds


def Level2_VENUS(dirname, chunks=500):
    '''
    Read an Venµs Level2 product as an xarray.Dataset

    Arguments:
        chunk: size of a single chunk
        split: whether the wavelength dependent variables should be split in multiple 2D variables
    '''
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists()
    
    ds = venus_read_metadata(ds, dirname)
    if isinstance(chunks, int): chunks = [chunks]*2
    
    # lat-lon
    geocoding = ds.attrs['Geoposition_Informations']
    venus_read_latlon(ds, geocoding, chunks)

    # read geaometry
    venus_read_geometry(ds, dirname, chunks)

    # read reflectances
    radio_info = ds.attrs['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = venus_read_rho(ds, dirname, quantif, chunks)

    return ds


def venus_read_metadata(ds, dirname):

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
    ds.attrs[n.product_name.name] = xmlroot['Scene_Useful_Image_Informations']['SCENE_ID']
    platform = xmlgranule['Product_Characteristics']['PLATFORM']
    assert platform == 'VENUS'

    # read image size for current resolution
    shape_info = xmlgranule['Geoposition_Informations']['Geopositioning']['Group_Geopositioning_List']
    ds.attrs['totalheight'] = shape_info['Group_Geopositioning']['NROWS']
    ds.attrs['totalwidth'] = shape_info['Group_Geopositioning']['NCOLS']
    
    # attributes
    ds.attrs['datetime'] = date
    ds.attrs[n.platform] = platform
    ds.attrs[n.resolution] = resolution
    ds.attrs[n.sensor] = 'VENUS'
    ds.attrs[n.product_name] = xmlgranule['Product_Characteristics']['PRODUCT_ID']
    ds.attrs.update(xmlgranule)
    ds.attrs.update(xmlroot)
    
    return ds


def venus_read_latlon(ds, geocoding, chunks):
    ds[n.lat.name] = DataArray_from_array(
        LATLON(geocoding, 'lat', ds),
        ('y','x'),
        chunks=chunks,
    )

    ds[n.lon.name] = DataArray_from_array(
        LATLON(geocoding, 'lon', ds),
        ('y','x'),
        chunks=chunks,
    )

def venus_resample(arr, ratio, chunks):
    
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

def venus_read_toa(ds, granule_dir, quantif, chunks):

    for name in ds[n.bnames.name]:
        
        filenames = list(granule_dir.glob(f'*REF_{name.values}.tif'))
        assert len(filenames) == 1
        filename = filenames[0]
        
        arr = xr.open_dataarray(filename).chunk([1]+chunks)
        arr = (arr/quantif).astype('float32')

        xrat = len(arr.x)/ds.totalwidth
        yrat = len(arr.y)/ds.totalheight
        
        arr_resampled = venus_resample(arr, (xrat,yrat), chunks)
        ds[n.rtoa.name+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)
    return ds


def venus_read_rho(ds, granule_dir, quantif, chunks):

    for rho, var in zip(['SRE','FRE'],['rho_surface','rho_flat']):
        for name in ds[n.bnames.name]:
            filenames = list(granule_dir.glob(f'*{rho}_{name.values}.tif'))
            assert len(filenames) == 1
            filename = filenames[0]
            
            arr = xr.open_dataarray(filename).chunk([1]+chunks)
            arr = (arr/quantif).astype('float32')

            xrat = len(arr.x)/ds.totalwidth
            yrat = len(arr.y)/ds.totalheight

            arr_resampled = venus_resample(arr, (xrat,yrat), chunks)
            ds[var+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)
    
    # read Aerosol_Optical_Thickness of waper vapor content
    filenames = list(granule_dir.glob('*ATB_XS.tif'))
    assert len(filenames) == 1
    atb = xr.open_dataarray(filenames[0]).chunk([1]+chunks)
    ds['water_vapor'] = atb.sel(band=1)
    ds['aod'] = atb.sel(band=2)

    return ds

def venus_read_geometry(ds, dirname, chunks):

    # read solar angles
    solarfiles = list((dirname/'DATA').glob('*SOL_ALL.tif'))
    assert len(solarfiles) == 1
    sa = xr.open_dataarray(solarfiles[0]).chunk([1]+chunks)
    
    # read view angles
    viewfiles = list((dirname/'DATA').glob('*VIE_ALL.tif'))
    assert len(viewfiles) == 1
    va = xr.open_dataarray(viewfiles[0]).chunk([1]+chunks)

class LATLON:
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
        self.x = ULX + XDIM//2 + XDIM*np.arange(ds.totalwidth)
        self.y = ULY + YDIM//2 + YDIM*np.arange(ds.totalheight)

        self.shape = (ds.totalheight, ds.totalwidth)
        self.ndim = 2
        self.dtype = 'float32'

    def __getitem__(self, key):
        X, Y = self.x[key[1]], self.y[key[0]]
        if isinstance(key[0], slice) and isinstance(key[1], slice):
            # keys are both slices
            X, Y = np.meshgrid(X, Y)
        else:
            X, Y = np.broadcast_arrays(X, Y)

        lon, lat = self.proj(X, Y, inverse=True)

        if self.kind == 'lat':
            if hasattr(lat, 'astype'):
                return lat.astype(self.dtype)
            else:
                return np.array(lat, dtype=self.dtype)
        else:
            if hasattr(lon, 'astype'):
                return lon.astype(self.dtype)
            else:
                return np.array(lon, dtype=self.dtype)


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