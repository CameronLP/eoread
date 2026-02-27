#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://www.eoportal.org/satellite-missions/venus#vssc-ven%C2%B5s-superspectral-camera

from pathlib import Path
from typing import Union, Literal

import dask.array as da
import pandas as pd
import xarray as xr
import pyproj

from core.files import mdir
from core.table import read_xml
from core.network.download import download_url
from core.files import mdir
from core.tools import merge, drop_unused_dims
from core import env, log

from eoread.tools import (
    open_raster, 
    spatial_resample, 
    filter_metadata, 
    format_chunks
)


user_guide = 'https://www.cesbio.cnrs.fr/multitemp/ven%c2%b5s-product-format/'

def Level1_VENUS(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        read_masks: bool = False, 
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False, 
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a VENµS Level1C product as an xarray.Dataset.
    
    Formats the Dataset to contain TOA reflectances, viewing/solar angles
    on the full grid, and geolocation information.
    
    VENµS (Vegetation and Environment monitoring on a New Micro-Satellite) provides
    12 superspectral bands from 420nm to 910nm with 5m spatial resolution.

    Args:
        dirname: Path to the VENµS product directory
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
                If tuple, should be (rows_chunk, columns_chunk)
        read_masks: If True, reads compressed quality masks (PIX, SAT, CLD, USI).
                   Warning: Uncompressing masks is time-consuming.
        metadata_template: List of metadata keys to include. If None, includes all metadata.
        v1_compat: If True, formats output to match version 1 structure
    
    Raises:
        AssertionError: If the directory does not exist
        
    Example:
        >>> ds = Level1_VENUS('VENUS-XS_*_L1C_*', chunks=1000)
        >>> print(ds.Rtoa.sel(bands='B8'))  # Red edge band
    """
    
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists(), 'Folder does not exists'
    if isinstance(chunks, int): chunks = [chunks]*2    
    
    # read metadata
    if verbose: log.debug('Reading metadata')
    ds, metadata_granule = _venus_read_metadata(ds, dirname, metadata_template)

    # read geaometry
    if verbose: log.debug('Read and compute geometric angles')
    ds = _venus_read_geometry(ds, dirname, chunks)

    # read TOA
    if verbose: log.debug('Read top of atmosphere data')
    radio_info = metadata_granule['Radiometric_Informations']
    quantif = float(radio_info['REFLECTANCE_QUANTIFICATION_VALUE'])
    ds = _venus_read_toa(ds, dirname, quantif, chunks)

    # lat-lon
    if verbose: log.debug('Compute LatLon raster')
    geocoding = metadata_granule['Geoposition_Informations']
    _venus_read_latlon(ds, geocoding, chunks)
    
    # read cloud altitude
    if verbose: log.debug('Open masks')
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
    
    elif verbose:
        log.debug('Masks are not red due to uncompression time consuming. '
                  'Active option read_masks to read them')
        
    ds = drop_unused_dims(ds)
    groups = ['bands_vnir']*len(ds[str(n.bands)])
    ds = merge(ds, str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)    
    ds = ds.assign_coords({str(n.bgroup): (str(n.bands), groups)})
    
    if v1_compat: return _v1_compat(ds, chunks)  
    return ds.unify_chunks()


def Level2_VENUS(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        metadata_template: Union[list, None] = None
    ) -> xr.Dataset:
    """
    Read a VENµS Level2A product as an xarray.Dataset.
    
    Processes Level2A surface reflectance products with atmospheric correction.

    Args:
        dirname: Path to the VENµS Level2A product directory
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
        metadata_template: List of metadata keys to include. If None, includes all metadata.
    """
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
    
    ds = drop_unused_dims(ds)
    groups = ['bands_vnir']*len(ds[str(n.bands)])
    ds = ds.assign_coords({str(n.bgroup): (str(n.bands), groups)})
    
    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample VENµS product directory for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Args:
        level: Processing level (1 for Level1C, 2 for Level2A)

    Returns:
        Path to the VENµS product directory
        
    Raises:
        ValueError: If level is not 1 or 2
    """
    
    # Check if user has provided a path
    variable = env.getvar(f'LEVEL{level}_VENUS', default='')
    
    # If not provided, try to download a sample with SAND
    if variable == '':
        
        # Check SAND importation
        try: 
            from sand.sample_product import products
            from sand.cnes import DownloadCNES
        except ImportError:
            raise ImportError('To use get_sample function, you need to install SAND module')
        
        # Retrieve name of example product
        sand_collection = 'VENUS'
        params = products[sand_collection]['constraint']
        
        # Download product with SAND
        dl = DownloadCNES()
        directory = env.getdir('DIR_SAMPLES')/sand_collection
        query = dl.query(collection_sand=sand_collection, level=level, **params)
        target = dl.download(query[0], directory)
        
        assert target.exists()
        return target
        
    else:
        return Path(variable)
    
    
def get_SRF(
    ds_in: Union[xr.Dataset, None] = None, 
    dir_data: Union[Path, None] = None
) -> xr.Dataset:
    """
    Load VENµS spectral response functions (SRF) for radiometric calculations.
    
    Downloads SRF data from the official repository if not already cached.

    Args:
        ds_in: Optional dataset with band names. If provided, output bands
               are referenced by ds_in.bands. Otherwise uses band IDs 1-12.
        dir_data: Directory to cache SRF data. If None, uses default static directory.

    Returns:
        xarray.Dataset containing:
            - SRF curves for each VENµS band
            - wav: Wavelength coordinate in nanometers
            - Band variables named by band ID or from ds_in.bands
    
    Example:
        >>> srf = get_SRF()
        >>> print(srf.sel(wav=550, method='nearest'))  # SRF at 550nm
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

def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample VENµS product directory for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Args:
        level: Processing level (1 for Level1C, 2 for Level2A)

    Returns:
        Path to the VENµS product directory
        
    Raises:
        ValueError: If level is not 1 or 2
        
    Example:
        >>> venus_dir = get_sample(level=1)
        >>> ds = Level1_VENUS(venus_dir)
    """
    if level == 1:
        return env.getdir('DIR_VENUS_L1C')
    elif level == 2:
        return env.getdir('DIR_VENUS_L2A')
    else:
        raise ValueError(level)
    # try: 
    #     from sand.cnes import DownloadCNES
    #     from sand.sample_product import products
    # except ImportError:
    #     raise ImportError('To use get_sample function, you need to install SAND module')
    
    # sensor = 'VENUS'
    # params = products[sensor]['constraint']
    # dl = DownloadCNES()
    # query = dl.query(collection_sand=sensor, level=level, **params)
    # return dl.download(query[0], env.getdir('DIR_SAMPLES'))

def _v1_compat(ds: xr.Dataset, chunks: list) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
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