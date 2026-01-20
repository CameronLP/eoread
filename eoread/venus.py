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
from core.network.download import download_url
from core.files import mdir
from core.tools import merge, drop_unused_dims
from core import env, log

from eoread.utils import open_raster, spatial_resample
from eoread.common import DataArray_from_array


def Level1_VENUS(dirname, 
                 chunks: int|tuple = 500,
                 read_masks: bool = False, 
                 metadata_template: list|None = None,
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
    ratio = {str(n.columns): ds.totalwidth, str(n.rows): ds.totalheight} 
    cld = open_raster(dirname/'DATA', '*CLA_ALL.tif', engine='rasterio')
    cld = cld.rename(x=str(n.columns), y=str(n.rows))
    ds['CLA_ALL'] = spatial_resample(cld, ratio, chunks, 'repeat')
    
    if read_masks:
        
        # read cloud mask
        cld = open_raster(dirname/'MASKS','*CLD_XS.zip','.zip').chunk(chunks)
        ds['CLD_XS'] = cld.rename(x=str(n.columns), y=str(n.rows))
        
        # read cloud mask
        usi = open_raster(dirname/'MASKS','*USI_XS.zip','.zip').chunk(chunks)
        ds['USI_XS'] = usi.rename(x=str(n.columns), y=str(n.rows))
    
        # Read quality masks
        for bn in ds[str(n.bnames)]:
            
            pix = open_raster(dirname/'MASKS',f'*PIX_{bn.values}.zip','.zip').chunk(chunks)
            ds[f'PIX_{bn.values}'] = pix.rename(x=str(n.columns), y=str(n.rows))
            
            sat = open_raster(dirname/'MASKS',f'*SAT_{bn.values}.zip','.zip').chunk(chunks) 
            ds[f'SAT_{bn.values}'] = sat.rename(x=str(n.columns), y=str(n.rows))
    
    else: 
        log.debug('Masks are not red due to uncompression time consuming. '
                  'Active option read_masks to read them')
        
    ds = merge(ds, str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)    
    ds = ds.assign_coords({str(n.bands): ds[str(n.bands)].data.astype(int),
                           str(n.bands_nvis): ds[str(n.bands)].data.astype(int)})  
    
    if v1_compat: return _v1_compat(ds, chunks)  
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
    cld = open_raster(dirname/'MASKS','*CLM_XS.tif', engine='rasterio').chunk(chunks)
    ds['CLM_XS'] = cld.rename(x=str(n.columns), y=str(n.rows))
    
    # read other masks
    usi = open_raster(dirname/'MASKS','*USI_XS.tif', engine='rasterio').chunk(chunks)
    ds['USI_XS'] = usi.rename(x=str(n.columns), y=str(n.rows))
    
    cld = open_raster(dirname/'MASKS','*SAT_XS.tif', engine='rasterio').chunk(chunks)
    ds['SAT_XS'] = cld.rename(x=str(n.columns), y=str(n.rows))
    
    usi = open_raster(dirname/'MASKS','*PIX_XS.tif', engine='rasterio').chunk([1]+list(chunks))
    ds['PIX_XS'] = usi.rename(x=str(n.columns), y=str(n.rows), band=str(n.bands))
    
    cld = open_raster(dirname/'MASKS','*IAB_XS.tif', engine='rasterio').chunk(chunks)
    ds['IAB_XS'] = cld.rename(x=str(n.columns), y=str(n.rows))
    
    usi = open_raster(dirname/'MASKS','*EDG_XS.tif', engine='rasterio').chunk(chunks)
    ds['EDG_XS'] = usi.rename(x=str(n.columns), y=str(n.rows))
    
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
    ds = ds.assign({str(n.cwav): ((str(n.bands)),cwvl),
                    str(n.bnames): ((str(n.bands)),bandnames)})
    
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
    ds.attrs[str(n.datetime)] = date
    ds.attrs[str(n.platform)] = platform
    ds.attrs[str(n.resolution)] = resolution
    ds.attrs[str(n.sensor)] = 'VENUS'
    ds.attrs[str(n.product_name)] = xmlgranule['Product_Characteristics']['PRODUCT_ID']
    ds.attrs[str(n.input_directory)] = str(dirname.parent)
    ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
    ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)
    
    return ds, xmlgranule


def _venus_read_latlon(ds, geocoding, chunks):
    
    ds[str(n.lat)] = DataArray_from_array(
        _LATLON(geocoding, 'lat', ds),
        (str(n.rows), str(n.columns)),
        chunks=chunks,
    )

    ds[str(n.lon)] = DataArray_from_array(
        _LATLON(geocoding, 'lon', ds),
        (str(n.rows), str(n.columns)),
        chunks=chunks,
    )

def _venus_read_toa(ds, granule_dir, quantif, chunks):
    
    for name in ds[str(n.bnames)]:
        
        arr = open_raster(granule_dir, f'*REF_{name.values}.tif', engine='rasterio').chunk(chunks)
        arr = (arr/quantif).astype('float32')
        
        ratio = {str(n.rows): ds.totalheight, str(n.columns): ds.totalwidth}        
        arr_resampled = spatial_resample(arr, ratio, chunks)
        ds[str(n.rtoa)+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=str(n.bands_nvis), pattern=r'(.+)_B(.+)', dtype=str)
    ds[str(n.rtoa)].attrs.update(unit=None)
    return ds


def _venus_read_rho(ds, granule_dir, quantif, chunks):

    for rho, var in zip(['SRE','FRE'],['rho_surface','rho_flat']):
        for name in ds[str(n.bnames)]:
            
            arr = open_raster(granule_dir, f'*{rho}_{name.values}.tif', engine='rasterio').chunk(chunks)
            arr = (arr/quantif).astype('float32')

            ratio = {'y': ds.totalheight, 'x': ds.totalwidth}  
            arr_resampled = spatial_resample(arr, ratio, chunks)
            ds[var+f'_{name.values}'] = arr_resampled

    ds = merge(ds, dim=str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)
    
    # read Aerosol_Optical_Thickness of waper vapor content
    atb = open_raster(granule_dir, '*ATB_XS.tif', engine='rasterio').chunk([1]+list(chunks))
    ds['water_vapor'] = atb.sel(band=1)
    ds['aod'] = atb.sel(band=2)

    return ds

def _venus_read_geometry(ds, dirname, chunks):

    # read solar angles
    sa = open_raster(dirname/'DATA','*SOL_ALL.tif', engine='rasterio').chunk([1]+list(chunks))
    ds['SOL_ALL'] = sa.rename(x=str(n.columns)+'_tie', y=str(n.rows)+'_tie')
    
    # read view angles
    va = open_raster(dirname/'DATA','*VIE_ALL.tif', engine='rasterio').chunk([1]+list(chunks))
    ds['VIE_ALL'] = va.rename(x=str(n.columns)+'_tie', y=str(n.rows)+'_tie')
    
    return ds.rename(band=str(n.bands)+'_angle')

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

def get_sample(level:int=1) -> Path:
    """
    Bring a VENµS directory path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
    """
    try: 
        from sand.cnes import DownloadCNES
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'VENUS'
    params = products[sensor]['constraint']
    dl = DownloadCNES()
    query = dl.query(collection_sand=sensor, level=level, **params)
    return dl.download(query[0], env.getdir('DIR_SAMPLES'))

def _v1_compat(ds, chunks):
    
    import numpy as np
    
    def read_xml_block(item):
        '''
        read a block of xml data and returns it as a numpy float32 array
        '''
        d = [i.split() for i in item]
        return np.array(d, dtype='float32')
    
    # Redefine geometric angles based on grnaule metadata
    angles = ds.attrs['metadata_granule']['Geometric_Informations']['Angles_Grids_List']
    sza = read_xml_block(angles['Sun_Angles_Grids']['Zenith']['Values_List']['VALUES'])
    saa = read_xml_block(angles['Sun_Angles_Grids']['Azimuth']['Values_List']['VALUES'])

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    vza = {}
    vaa = {}
    via_list = angles['Viewing_Incidence_Angles_Grids_List']['Band_Viewing_Incidence_Angles_Grids_List']
    for e in via_list['Viewing_Incidence_Angles_Grids']:

        # read zenith angles
        data = read_xml_block(e['Zenith']['Values_List']['VALUES'])
        bandid = int(e['attributes']['detector_id'])
        if bandid not in vza:
            vza[bandid] = data
        else:
            ok = ~np.isnan(data)
            vza[bandid][ok] = data[ok]

        # read azimuth angles
        data = read_xml_block(e['Azimuth']['Values_List']['VALUES'])
        bandid = int(e['attributes']['detector_id'])
        if bandid not in vaa:
            vaa[bandid] = data
        else:
            ok = ~np.isnan(data)
            vaa[bandid][ok] = data[ok]

    # use the first band as vza and vaa
    k = sorted(vza.keys())[0]
    assert k in vaa

    # initialize the dask arrays
    dims = ('tie_rows', 'tie_columns')
    out = dict(zip(dims, ds[str(n.lat)].shape))
    for name, tie in [(str(n.sza), sza),
                      (str(n.saa), saa),
                      (str(n.vza), vza[k]),
                      (str(n.vaa), vaa[k]),
                      ]:
        da_tie = xr.DataArray(
            tie,
            dims=dims,
            coords={'tie_rows': np.linspace(0, shp[0]-1, sza.shape[0]),
                    'tie_columns': np.linspace(0, shp[1]-1, sza.shape[1])})
        ds[name+'_tie'] = da_tie
        ds[name] = spatial_resample(da_tie, out, chunks, 'linear')
    
    # Assign central wavelengths as band coordinates
    venus_band_names = [420,443,490,555,620,622,667,702,742,782,865,910]
    ds = ds.assign_coords(bands=venus_band_names)
    
    # Drop NVIS bands dimension
    ds = ds.assign(Rtoa=(('bands','y','x'),ds[str(n.rtoa)].data))
    
    # Flags 
    ds['flags'] = xr.zeros_like(ds.vza, dtype='uint8')
    
    # Add CRS 
    crs = ds.attrs['metadata_granule']['Geoposition_Informations']['Coordinate_Reference_System']['Horizontal_Coordinate_System']['HORIZONTAL_CS_CODE']
    ds.attrs[str(n.crs)] = 'epsg:'+str(crs)
    
    return ds